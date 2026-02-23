"""
Polymarket Executor — Paper and live bet placement for Polymarket markets.

Reads edge opportunities from the polymarket_scanner SQLite table,
applies risk limits and Kelly sizing, places paper bets, tracks positions,
and checks resolution.

Runs in-process as part of the Renaissance bot's 5-minute cycle.
No file bridges, no external processes.

Architecture:
  polymarket_scanner.py -> SQLite -> polymarket_executor.py -> SQLite
                                                             -> dashboard API
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"

# Time-remaining thresholds (seconds) by market type
TIME_FILTERS: Dict[str, Dict[str, float]] = {
    'DIRECTION': {
        'min_seconds': 60,           # Don't bet with <60s left
        'max_seconds': 300,          # 5m markets: don't bet if >5min
        'sweet_spot_min': 60,        # Best zone: 1-4 minutes remaining
        'sweet_spot_max': 240,
    },
    'HIT_PRICE': {
        'min_seconds': 7 * 86400,    # Don't bet with <7 days left
        'max_seconds': float('inf'),
    },
    'THRESHOLD': {
        'min_seconds': 3 * 86400,    # Don't bet with <3 days left
        'max_seconds': float('inf'),
    },
    'RANGE': {
        'min_seconds': 86400,        # Don't bet with <1 day left
        'max_seconds': float('inf'),
    },
}


class PolymarketExecutor:
    """Paper-trade Polymarket prediction markets using scanner edges."""

    def __init__(
        self,
        config: dict,
        db_path: str = "data/renaissance_bot.db",
        logger: Optional[logging.Logger] = None,
    ):
        self.db_path = db_path
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        pm_cfg = config.get("polymarket", {})
        self.paper_mode = pm_cfg.get("paper_mode", True)
        self.enabled = pm_cfg.get("executor_enabled", True)
        self.bankroll = pm_cfg.get("initial_bankroll", 500.0)

        # Risk limits
        self.max_positions = pm_cfg.get("max_positions", 10)
        self.max_per_market_pct = 0.20      # 20% of bankroll per market
        self.max_per_asset_pct = 0.40       # 40% of bankroll per asset
        self.max_total_exposure_pct = 0.60  # 60% of bankroll total
        self.min_edge_direction = 0.03      # 3% min edge for DIRECTION
        self.min_edge_hit_price = 0.05      # 5% min edge for HIT_PRICE
        self.max_bets_per_cycle = 3         # Max new bets per 5-min cycle

        # Resolution check rate limiting — every 12 cycles (~1 hour)
        self._resolution_check_counter = 0

        self._ensure_tables()
        self._load_bankroll()

        # One-time repair of any prematurely resolved positions
        self.repair_premature_resolutions()

    # ------------------------------------------------------------------
    # DB setup
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create position tracking tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS polymarket_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id TEXT UNIQUE NOT NULL,
                condition_id TEXT NOT NULL,
                slug TEXT,
                question TEXT,
                market_type TEXT,
                asset TEXT,
                direction TEXT,
                entry_price REAL,
                shares REAL,
                bet_amount REAL,
                edge_at_entry REAL,
                our_prob_at_entry REAL,
                crowd_prob_at_entry REAL,
                target_price REAL,
                deadline TEXT,
                status TEXT DEFAULT 'open',
                exit_price REAL,
                pnl REAL,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS polymarket_bankroll_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                bankroll REAL NOT NULL,
                event TEXT,
                position_id TEXT,
                amount REAL
            );

            CREATE INDEX IF NOT EXISTS idx_pm_positions_status
                ON polymarket_positions(status);
            CREATE INDEX IF NOT EXISTS idx_pm_positions_asset
                ON polymarket_positions(asset);
        """)
        conn.commit()
        conn.close()

    def _load_bankroll(self) -> None:
        """Load current bankroll from log (last entry) or use initial."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT bankroll FROM polymarket_bankroll_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            self.bankroll = row[0]

    def _log_bankroll(self, event: str, position_id: Optional[str] = None,
                      amount: float = 0.0) -> None:
        """Log a bankroll change."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO polymarket_bankroll_log (timestamp, bankroll, event, position_id, amount) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), self.bankroll, event, position_id, amount),
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Repair prematurely resolved positions
    # ------------------------------------------------------------------

    def repair_premature_resolutions(self) -> None:
        """
        One-time fix: reopen positions that were resolved prematurely.
        A position was prematurely resolved if it was closed within 30 minutes
        of opening and the market's deadline is months away.
        Idempotent — does nothing if no suspicious positions found.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        suspicious = conn.execute("""
            SELECT * FROM polymarket_positions
            WHERE status IN ('won', 'lost')
              AND market_type = 'HIT_PRICE'
              AND (julianday(closed_at) - julianday(opened_at)) < 0.021
        """).fetchall()

        if not suspicious:
            conn.close()
            return

        self.logger.info(f"REPAIR: Found {len(suspicious)} prematurely resolved positions")

        for pos in suspicious:
            old_status = pos['status']
            old_pnl = pos['pnl'] or 0

            conn.execute("""
                UPDATE polymarket_positions
                SET status = 'open', exit_price = NULL, pnl = NULL,
                    closed_at = NULL, notes = 'REPAIRED: was prematurely resolved'
                WHERE position_id = ?
            """, (pos['position_id'],))

            self.logger.info(
                f"REPAIR: Reopened '{pos['question'][:50]}' "
                f"(was {old_status}, pnl was ${old_pnl})"
            )

        # Recalculate bankroll from scratch: initial - open_bets + resolved_pnl
        open_bets = conn.execute(
            "SELECT COALESCE(SUM(bet_amount), 0) as total FROM polymarket_positions WHERE status = 'open'"
        ).fetchone()
        resolved_pnl = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) as total FROM polymarket_positions WHERE status IN ('won', 'lost')"
        ).fetchone()

        initial = self.config.get("polymarket", {}).get("initial_bankroll", 500.0)
        self.bankroll = initial - open_bets['total'] + resolved_pnl['total']

        self._log_bankroll("repair_premature_resolution", None, 0)

        conn.commit()
        conn.close()
        self.logger.info(f"REPAIR: Bankroll recalculated to ${self.bankroll:.2f}")

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def execute_cycle(self) -> None:
        """Main cycle: read edges, place bets, check resolutions."""
        if not self.enabled:
            return

        # 1. Check resolution of open positions (every 12 cycles = ~1 hour)
        self._resolution_check_counter += 1
        if self._resolution_check_counter % 12 == 0:
            resolved_count = await self._check_resolutions()
            if resolved_count > 0:
                self.logger.info(f"POLYMARKET EXECUTOR: {resolved_count} positions resolved")

        # 2. Get current open positions
        open_positions = self._get_open_positions()

        # 3. Read edge opportunities from scanner
        opportunities = self._get_scanner_opportunities()
        if not opportunities:
            return

        # 4. Filter and rank
        eligible = self._filter_opportunities(opportunities, open_positions)
        if not eligible:
            return

        # 5. Place bets (up to max_bets_per_cycle)
        bets_placed = 0
        for opp in eligible:
            if bets_placed >= self.max_bets_per_cycle:
                break
            if self.bankroll < 1.0:
                self.logger.warning("POLYMARKET: Bankroll depleted, no more bets")
                break
            if self._place_bet(opp):
                bets_placed += 1

        if bets_placed > 0:
            self.logger.info(
                f"POLYMARKET EXECUTOR: Placed {bets_placed} bets | "
                f"Open: {len(open_positions) + bets_placed} | "
                f"Bankroll: ${self.bankroll:.2f}"
            )

    # ------------------------------------------------------------------
    # Read opportunities from scanner table
    # ------------------------------------------------------------------

    def _get_scanner_opportunities(self) -> List[dict]:
        """Read latest edge opportunities from polymarket_scanner table."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT condition_id, question, slug, market_type, asset,
                   timeframe_minutes, deadline, target_price,
                   yes_price, no_price, volume_24h, liquidity,
                   edge, our_probability, direction, confidence
            FROM polymarket_scanner
            WHERE scan_time = (SELECT MAX(scan_time) FROM polymarket_scanner)
              AND edge IS NOT NULL
              AND edge > 0
              AND direction IS NOT NULL
            ORDER BY edge DESC
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Time remaining helper
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_time_remaining(opp: dict) -> float:
        """
        Compute time remaining until market resolution, in seconds.
        Returns float('inf') if deadline is unknown/unparseable.
        """
        deadline_str = opp.get('deadline') or ''
        if not deadline_str:
            return float('inf')
        try:
            deadline = datetime.fromisoformat(deadline_str.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            remaining = (deadline - now).total_seconds()
            return max(0, remaining)
        except (ValueError, TypeError):
            return float('inf')

    # ------------------------------------------------------------------
    # Filter opportunities (risk limits + time filters)
    # ------------------------------------------------------------------

    def _filter_opportunities(self, opportunities: List[dict],
                              open_positions: List[dict]) -> List[dict]:
        """Apply risk limits, time filters, and dedup to actionable opportunities."""
        eligible = []

        # Build current exposure map
        asset_exposure: Dict[str, float] = {}
        total_exposure = 0.0
        open_slugs: set = set()
        for pos in open_positions:
            asset_exposure[pos['asset']] = asset_exposure.get(pos['asset'], 0) + pos['bet_amount']
            total_exposure += pos['bet_amount']
            if pos.get('slug'):
                open_slugs.add(pos['slug'])

        for opp in opportunities:
            # Skip if already have position with same slug
            slug = opp.get('slug', '')
            if slug and slug in open_slugs:
                continue

            # Time-remaining filter
            market_type = opp.get('market_type', 'OTHER')
            time_remaining = self._compute_time_remaining(opp)
            time_filter = TIME_FILTERS.get(market_type, {'min_seconds': 60, 'max_seconds': float('inf')})

            if time_remaining < time_filter['min_seconds']:
                continue
            if time_remaining > time_filter.get('max_seconds', float('inf')):
                continue

            # Edge threshold (with DIRECTION sweet-spot adjustment)
            min_edge = self.min_edge_hit_price if market_type == 'HIT_PRICE' else self.min_edge_direction

            if market_type == 'DIRECTION':
                sweet = TIME_FILTERS['DIRECTION']
                if sweet['sweet_spot_min'] <= time_remaining <= sweet['sweet_spot_max']:
                    min_edge = max(0.02, min_edge * 0.67)

            if opp.get('edge', 0) < min_edge:
                continue

            # Skip if too many open positions
            if len(open_positions) + len(eligible) >= self.max_positions:
                break

            # Skip if total exposure at limit
            if total_exposure >= self.bankroll * self.max_total_exposure_pct:
                break

            # Skip if asset exposure at limit
            asset = opp.get('asset', '')
            if asset_exposure.get(asset, 0) >= self.bankroll * self.max_per_asset_pct:
                continue

            # Skip low-liquidity markets
            volume = opp.get('volume_24h', 0) or 0
            if volume < 100:
                continue

            eligible.append(opp)

        return eligible

    # ------------------------------------------------------------------
    # Place bet (paper mode)
    # ------------------------------------------------------------------

    def _place_bet(self, opp: dict) -> Optional[dict]:
        """Place a paper bet on a Polymarket opportunity."""
        direction = opp['direction']  # YES/NO or UP/DOWN
        edge = opp['edge']
        our_prob = opp.get('our_probability', 0.5)
        slug = opp.get('slug', '')

        # Duplicate slug check against DB
        conn = sqlite3.connect(self.db_path)
        existing = conn.execute(
            "SELECT COUNT(*) FROM polymarket_positions WHERE slug = ? AND status = 'open'",
            (slug,)
        ).fetchone()
        if existing and existing[0] > 0:
            conn.close()
            return None

        # Determine entry price
        yes_price = opp.get('yes_price', 0.5)
        if direction in ('YES', 'UP'):
            entry_price = yes_price
        else:
            entry_price = 1.0 - yes_price

        if entry_price <= 0.01 or entry_price >= 0.99:
            conn.close()
            return None

        # Kelly sizing for binary outcome
        # f* = (p * b - q) / b  where b = (1/entry_price) - 1
        b = (1.0 / entry_price) - 1.0  # odds
        p = our_prob if direction in ('YES', 'UP') else (1.0 - our_prob)
        q = 1.0 - p

        if b <= 0:
            conn.close()
            return None

        kelly = (p * b - q) / b
        kelly = max(0.0, min(kelly, 0.10))  # Cap at 10%

        # Half-Kelly for safety
        bet_fraction = kelly * 0.5
        bet_amount = round(self.bankroll * bet_fraction, 2)

        # Enforce limits
        bet_amount = max(1.0, min(bet_amount, self.bankroll * self.max_per_market_pct, 100.0))

        if bet_amount > self.bankroll:
            conn.close()
            return None

        shares = round(bet_amount / entry_price, 4)
        position_id = f"pm_{uuid.uuid4().hex[:12]}"

        # Time remaining for logging
        time_remaining = self._compute_time_remaining(opp)
        if time_remaining < 86400:
            time_str = f"{time_remaining / 3600:.1f}h"
        elif time_remaining < 86400 * 30:
            time_str = f"{time_remaining / 86400:.0f}d"
        elif time_remaining == float('inf'):
            time_str = "unknown"
        else:
            time_str = f"{time_remaining / (86400 * 30):.1f}mo"

        self.logger.info(
            f"POLYMARKET BET: {direction} on \"{opp.get('question', '')[:50]}\" | "
            f"Entry: ${entry_price:.3f} | Edge: {edge:.1%} | "
            f"Kelly: {kelly:.1%} | Bet: ${bet_amount:.2f} | "
            f"Time left: {time_str}"
        )

        # Record position
        conn.execute("""
            INSERT INTO polymarket_positions
            (position_id, condition_id, slug, question, market_type, asset,
             direction, entry_price, shares, bet_amount, edge_at_entry,
             our_prob_at_entry, crowd_prob_at_entry, target_price, deadline,
             status, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
        """, (
            position_id, opp.get('condition_id', ''), slug,
            opp.get('question', ''), opp.get('market_type', ''), opp.get('asset', ''),
            direction, entry_price, shares, bet_amount, edge,
            our_prob, yes_price,
            opp.get('target_price'), opp.get('deadline'),
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()
        conn.close()

        # Update bankroll
        self.bankroll -= bet_amount
        self._log_bankroll("bet_placed", position_id, -bet_amount)

        return {"position_id": position_id, "bet_amount": bet_amount}

    # ------------------------------------------------------------------
    # Check resolutions (slug-based, not condition_id)
    # ------------------------------------------------------------------

    async def _check_resolutions(self) -> int:
        """
        Check if any open positions have resolved via Gamma API.

        CRITICAL: A position is only resolved when ALL of these are true:
        1. market.resolved == True (NOT just closed)
        2. One outcome price is >= 0.95 and the other <= 0.05
           (definitive resolution, not live trading prices)
        3. Uses slug-based lookup (condition_id returns unrelated markets)

        If the market is not resolved, we update unrealized P&L only.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        open_positions = conn.execute(
            "SELECT * FROM polymarket_positions WHERE status = 'open'"
        ).fetchall()

        if not open_positions:
            conn.close()
            return 0

        resolved = 0
        timeout = aiohttp.ClientTimeout(total=15)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for pos in open_positions:
                    try:
                        resolved += await self._check_single_resolution(session, dict(pos), conn)
                    except Exception as e:
                        self.logger.debug(f"Resolution check failed for {pos['position_id']}: {e}")
        except Exception as e:
            self.logger.debug(f"Resolution session error: {e}")

        conn.commit()
        conn.close()
        return resolved

    async def _check_single_resolution(self, session: aiohttp.ClientSession,
                                       pos: dict, conn: sqlite3.Connection) -> int:
        """Check a single position for resolution. Returns 1 if resolved, 0 otherwise."""
        slug = pos.get('slug', '')
        if not slug:
            return 0

        # Query by slug — condition_id returns unrelated old markets
        params = {"slug": slug}
        async with session.get(GAMMA_MARKETS_URL, params=params) as resp:
            if resp.status != 200:
                return 0
            markets = await resp.json()

        if not markets:
            return 0

        # Find the correct market — prefer active, non-closed
        market = None
        for m in markets:
            if m.get('active') and not m.get('closed'):
                market = m
                break
        # If all closed, take the one with latest endDate
        if not market:
            market = sorted(
                markets,
                key=lambda m: m.get('endDate', ''),
                reverse=True,
            )[0]

        # Parse outcome prices
        prices = market.get('outcomePrices', '[]')
        if isinstance(prices, str):
            prices = json.loads(prices)
        if not prices or len(prices) < 2:
            return 0

        yes_price = float(prices[0])
        no_price = float(prices[1])

        # ── CHECK 1: Is the market actually RESOLVED? ──
        is_resolved = bool(market.get('resolved'))

        # ── CHECK 2: Are outcome prices definitive? ──
        # A resolved market has one outcome at ~1.0 and the other at ~0.0.
        # Live trading prices (e.g. 0.38/0.62) are NOT resolutions.
        is_definitive = (yes_price >= 0.95 and no_price <= 0.05) or \
                        (yes_price <= 0.05 and no_price >= 0.95)

        if is_resolved and is_definitive:
            # ── RESOLVED: Mark position as won or lost ──
            direction = pos['direction']
            if direction in ('YES', 'UP'):
                won = yes_price >= 0.95
            else:
                won = no_price >= 0.95

            exit_price = 1.0 if won else 0.0
            pnl = (exit_price * pos['shares']) - pos['bet_amount']
            status = 'won' if won else 'lost'

            conn.execute("""
                UPDATE polymarket_positions
                SET status = ?, exit_price = ?, pnl = ?, closed_at = ?,
                    notes = ?
                WHERE position_id = ?
            """, (
                status, exit_price, round(pnl, 2),
                datetime.now(timezone.utc).isoformat(),
                f"resolved=true,definitive=true,yes={yes_price:.3f},no={no_price:.3f}",
                pos['position_id'],
            ))

            if won:
                self.bankroll += pos['bet_amount'] + pnl
            self._log_bankroll("bet_resolved", pos['position_id'], round(pnl, 2))

            self.logger.info(
                f"POLYMARKET RESOLVED: \"{pos['question'][:50]}\" -> {status.upper()} | "
                f"P&L: ${pnl:+.2f} | Bankroll: ${self.bankroll:.2f}"
            )
            return 1

        else:
            # ── NOT RESOLVED: Update unrealized P&L only ──
            direction = pos.get('direction', '')
            if direction in ('YES', 'UP'):
                current_price = yes_price
            elif direction in ('NO', 'DOWN'):
                current_price = no_price
            else:
                current_price = yes_price

            unrealized = (current_price - pos['entry_price']) * pos['shares']
            conn.execute(
                "UPDATE polymarket_positions SET notes = ? WHERE position_id = ?",
                (
                    f"unrealized_pnl={unrealized:.2f},current={current_price:.3f},"
                    f"active={market.get('active')},closed={market.get('closed')},"
                    f"resolved={market.get('resolved')}",
                    pos['position_id'],
                ),
            )
            return 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_open_positions(self) -> List[dict]:
        """Get all open positions from the database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM polymarket_positions WHERE status = 'open'"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_summary(self) -> dict:
        """Get executor summary for dashboard / logging."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        open_pos = conn.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(bet_amount), 0) as exposure "
            "FROM polymarket_positions WHERE status = 'open'"
        ).fetchone()

        resolved = conn.execute(
            "SELECT COUNT(*) as cnt, "
            "SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins, "
            "SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) as losses, "
            "COALESCE(SUM(pnl), 0) as total_pnl "
            "FROM polymarket_positions WHERE status IN ('won', 'lost')"
        ).fetchone()

        pnl_24h = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) as pnl "
            "FROM polymarket_positions "
            "WHERE status IN ('won', 'lost') "
            "AND closed_at >= datetime('now', '-24 hours')"
        ).fetchone()

        conn.close()

        wins = resolved['wins'] or 0
        cnt = resolved['cnt'] or 0

        return {
            "enabled": self.enabled,
            "paper_mode": self.paper_mode,
            "bankroll": round(self.bankroll, 2),
            "open_positions": open_pos['cnt'],
            "total_exposure": round(open_pos['exposure'], 2),
            "resolved_count": cnt,
            "wins": wins,
            "losses": resolved['losses'] or 0,
            "win_rate": round(wins / cnt * 100, 1) if cnt > 0 else 0.0,
            "total_pnl": round(resolved['total_pnl'] or 0, 2),
            "pnl_24h": round(pnl_24h['pnl'] or 0, 2),
        }
