"""
Polymarket Strategy A v3: Confidence-Gated Entry with Active Management

Simple rules:
  1. ML confidence >= 55% AND token cost <= $0.45 -> BUY (half-Kelly sized)
  2. Every cycle, manage open bets:
     - ML flips direction -> SELL immediately
     - ML confidence drops below 50% -> SELL
     - ML confidence >= 55% + token <= $0.45 + under $150 cap -> ADD
     - Otherwise -> HOLD
  3. Rate limit: max 6 bets per hour
  4. Cooldown: 5 min after any loss

ML Source:
  Crash-regime LightGBM (52.9% acc, 0.543 AUC) as primary signal for BTC.
  Calibrated for crash regime: 55-60% confident = 58.3% accurate.
  Half-Kelly sizing: 3-8% of bankroll per bet.

Market Discovery:
  Rolling 15m direction markets via slug pattern: {asset}-updown-15m-{unix_timestamp}
  Discovered via Gamma API (no scanner dependency).
  Instruments: BTC, ETH, SOL, XRP.
"""

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
MARKET_CACHE_TTL = 120  # seconds


# --- Instrument Config ---

@dataclass
class InstrumentConfig:
    asset: str
    ml_pair: str
    price_pair: str
    slug_pattern: str
    enabled: bool = True


INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "btc_15m": InstrumentConfig("BTC", "BTC-USD", "BTC-USD", "btc-updown-15m-{ts}"),
    "eth_15m": InstrumentConfig("ETH", "ETH-USD", "ETH-USD", "eth-updown-15m-{ts}"),
    "sol_15m": InstrumentConfig("SOL", "SOL-USD", "SOL-USD", "sol-updown-15m-{ts}"),
    "doge_15m": InstrumentConfig("DOGE", "DOGE-USD", "DOGE-USD", "doge-updown-15m-{ts}", enabled=False),
    "xrp_15m": InstrumentConfig("XRP", "XRP-USD", "XRP-USD", "xrp-updown-15m-{ts}"),
}


def build_instruments() -> Dict[str, InstrumentConfig]:
    """Return instrument configs (kept for backward compat with dashboard import)."""
    return INSTRUMENTS


# --- Strategy A Executor ---

class StrategyAExecutor:
    """
    v3: Confidence-gated entry with active position management.

    Entry:  ML confidence >= 55% AND token cost <= $0.45
    Sell:   ML flips direction OR confidence < 50%
    Add:    Same direction, >= 55% conf, token <= $0.45, < $150 total
    Limits: 6 bets/hour, 5 min cooldown after loss

    Sizing: Half-Kelly based on model probability and token price.
    """

    # Thresholds — calibrated for crash-regime LightGBM (max conf ~56%)
    CONFIDENCE_THRESHOLD = 53.0       # Model prob >= 0.53 (or <= 0.47)
    MAX_TOKEN_COST = 0.45
    BET_AMOUNT = 50.0                 # Fallback; overridden by Kelly sizing
    EXIT_CONFIDENCE = 50.0            # Exit when model is pure coin-flip
    ADD_CONFIDENCE = 53.0             # Same as entry threshold
    MAX_POSITION_PER_MARKET = 150.0   # dollar cap
    STOP_LOSS_PCT = 0.40              # close if share price drops 40% from avg cost
    MAX_BETS_PER_HOUR = 6
    COOLDOWN_AFTER_LOSS = 300         # 5 min in seconds
    MIN_BET = 5.0                     # Floor for Kelly sizing
    MAX_BET_PCT = 0.15                # Ceiling: 15% of bankroll per bet

    def __init__(self, config: dict, db_path: str, logger: Optional[logging.Logger] = None):
        self.config = config
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)

        pm_cfg = config.get("polymarket", {})
        self.enabled = pm_cfg.get("executor_enabled", True)
        self.initial_bankroll = pm_cfg.get("initial_bankroll", 500.0)

        self.instruments = INSTRUMENTS
        enabled = [k for k, v in self.instruments.items() if v.enabled]
        self.logger.info(f"Strategy A v3: {len(enabled)} instruments enabled: {enabled}")

        self.bankroll = self.initial_bankroll
        self._market_cache: Dict[str, Tuple[dict, float]] = {}

        # Rate limiting & cooldown state
        self._bets_this_hour: List[float] = []
        self._last_loss_time: float = 0.0

        self._ensure_tables()
        self._load_bankroll()

    # -- Database Setup --

    def _ensure_tables(self) -> None:
        conn = sqlite3.connect(self.db_path)

        # Migrate old polymarket_bets (activity log with 'action' column) -> legacy
        cols = []
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(polymarket_bets)").fetchall()]
        except Exception:
            pass

        if cols and "action" in cols and "entry_side" not in cols:
            self.logger.info("Migrating old polymarket_bets -> polymarket_bets_legacy")
            conn.execute("ALTER TABLE polymarket_bets RENAME TO polymarket_bets_legacy")
            conn.commit()

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS polymarket_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL,
                asset TEXT NOT NULL,
                entry_side TEXT NOT NULL,
                entry_token_cost REAL NOT NULL,
                entry_amount REAL NOT NULL,
                entry_tokens REAL NOT NULL,
                entry_confidence REAL NOT NULL,
                adds TEXT DEFAULT '[]',
                total_invested REAL NOT NULL,
                total_tokens REAL NOT NULL,
                avg_cost REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'OPEN',
                exit_price REAL,
                exit_reason TEXT,
                exit_at TEXT,
                pnl REAL,
                return_pct REAL,
                regime TEXT,
                entry_asset_price REAL,
                exit_asset_price REAL,
                opened_at TEXT NOT NULL DEFAULT (datetime('now')),
                question TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_pb3_status ON polymarket_bets(status);
            CREATE INDEX IF NOT EXISTS idx_pb3_asset ON polymarket_bets(asset);

            CREATE TABLE IF NOT EXISTS polymarket_skip_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                asset TEXT NOT NULL,
                slug TEXT,
                reason TEXT NOT NULL,
                ml_confidence REAL,
                token_cost REAL,
                ml_direction TEXT,
                minutes_left REAL
            );
            CREATE INDEX IF NOT EXISTS idx_psl_ts ON polymarket_skip_log(timestamp);

            CREATE TABLE IF NOT EXISTS polymarket_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id TEXT UNIQUE NOT NULL,
                market_id TEXT,
                condition_id TEXT,
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
                notes TEXT,
                registry_key TEXT,
                entry_window TEXT,
                is_contrarian INTEGER DEFAULT 0,
                strategy TEXT DEFAULT 'strategy_a'
            );
            CREATE INDEX IF NOT EXISTS idx_pm_pos_status ON polymarket_positions(status);
            CREATE INDEX IF NOT EXISTS idx_pm_pos_strategy ON polymarket_positions(strategy);

            CREATE TABLE IF NOT EXISTS polymarket_bankroll_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                bankroll REAL NOT NULL,
                event TEXT,
                position_id TEXT,
                amount REAL
            );
        """)
        conn.commit()

        # Migrate: add 'strategy' column if missing on polymarket_positions
        pos_cols = [r[1] for r in conn.execute("PRAGMA table_info(polymarket_positions)").fetchall()]
        if "strategy" not in pos_cols:
            conn.execute("ALTER TABLE polymarket_positions ADD COLUMN strategy TEXT DEFAULT 'legacy'")
            conn.commit()

        # Prune skip log entries older than 7 days
        conn.execute("DELETE FROM polymarket_skip_log WHERE timestamp < datetime('now', '-7 days')")
        conn.commit()

        conn.close()

    def _load_bankroll(self) -> None:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT bankroll FROM polymarket_bankroll_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            self.bankroll = row[0]

    def _log_bankroll(self, event: str, position_id: Optional[str] = None, amount: float = 0) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO polymarket_bankroll_log (timestamp, bankroll, event, position_id, amount) "
            "VALUES (datetime('now'), ?, ?, ?, ?)",
            (self.bankroll, event, position_id, amount),
        )
        conn.commit()
        conn.close()

    def _log_skip(self, asset: str, slug: Optional[str], reason: str,
                  ml_confidence: float = 0, token_cost: float = 0,
                  ml_direction: str = "", minutes_left: float = 0) -> None:
        """Write to polymarket_skip_log table."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO polymarket_skip_log "
            "(asset, slug, reason, ml_confidence, token_cost, ml_direction, minutes_left) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (asset, slug, reason, ml_confidence, token_cost, ml_direction, minutes_left),
        )
        conn.commit()
        conn.close()

    # -- Rate Limiting & Cooldown --

    def _check_rate_limit(self) -> bool:
        """Return True if under rate limit (can bet)."""
        now = time.time()
        cutoff = now - 3600
        self._bets_this_hour = [t for t in self._bets_this_hour if t > cutoff]
        return len(self._bets_this_hour) < self.MAX_BETS_PER_HOUR

    def _check_cooldown(self) -> bool:
        """Return True if cooldown has passed (can bet)."""
        if self._last_loss_time == 0:
            return True
        return (time.time() - self._last_loss_time) >= self.COOLDOWN_AFTER_LOSS

    # -- Main Cycle --

    async def execute_cycle(
        self,
        ml_predictions: dict,
        current_prices: dict,
        current_regime: str = "unknown",
    ) -> None:
        """Called every bot cycle from the main trading loop."""
        if not self.enabled:
            return

        self.logger.info(
            f"Strategy A v3 cycle: regime={current_regime}, "
            f"prices={len(current_prices)}, ml={len(ml_predictions)}, "
            f"bankroll=${self.bankroll:.2f}"
        )

        # 1. Check resolutions on old and new tables
        self._check_resolutions(current_prices)

        # 2. Manage open bets (from new polymarket_bets table)
        self._manage_positions(ml_predictions, current_prices)

        # 3. Look for new entry opportunities
        for inst_key, inst in self.instruments.items():
            if not inst.enabled:
                continue

            market = self._discover_market(inst)
            if not market:
                continue

            slug = market.get("slug", "")
            minutes_left = self._get_minutes_remaining(market)
            if minutes_left is None or minutes_left < 1.0 or minutes_left > 14.0:
                continue

            # ML data
            ml_data = ml_predictions.get(inst.ml_pair, {})
            ml_conf = ml_data.get("confidence", 50.0)
            ml_pred = ml_data.get("prediction", 0)
            ml_direction = "UP" if ml_pred > 0 else "DOWN"
            entry_side = "YES" if ml_direction == "UP" else "NO"

            # Token cost (crowd pricing)
            crowd_up = self._parse_crowd_up(market)
            token_cost = crowd_up if ml_direction == "UP" else (1.0 - crowd_up)

            # Gate: confidence
            if ml_conf < self.CONFIDENCE_THRESHOLD:
                self._log_skip(inst.asset, slug,
                               f"conf {ml_conf:.0f}% < {self.CONFIDENCE_THRESHOLD}%",
                               ml_conf, token_cost, ml_direction, minutes_left)
                continue

            # Gate: token cost
            if token_cost > self.MAX_TOKEN_COST:
                self._log_skip(inst.asset, slug,
                               f"token ${token_cost:.2f} > ${self.MAX_TOKEN_COST}",
                               ml_conf, token_cost, ml_direction, minutes_left)
                continue

            # Gate: rate limit
            if not self._check_rate_limit():
                self._log_skip(inst.asset, slug, "rate_limit",
                               ml_conf, token_cost, ml_direction, minutes_left)
                continue

            # Gate: cooldown
            if not self._check_cooldown():
                remaining = self.COOLDOWN_AFTER_LOSS - (time.time() - self._last_loss_time)
                self._log_skip(inst.asset, slug,
                               f"cooldown {remaining:.0f}s remaining",
                               ml_conf, token_cost, ml_direction, minutes_left)
                continue

            # Gate: not already positioned on this slug
            conn = sqlite3.connect(self.db_path)
            existing = conn.execute(
                "SELECT COUNT(*) FROM polymarket_bets WHERE slug = ? AND status = 'OPEN'",
                (slug,)
            ).fetchone()[0]
            conn.close()

            if existing > 0:
                continue  # Already have a bet, management handles adds

            # Gate: max exposure check
            conn = sqlite3.connect(self.db_path)
            total_open = conn.execute(
                "SELECT COALESCE(SUM(total_invested), 0) FROM polymarket_bets WHERE status = 'OPEN'"
            ).fetchone()[0]
            conn.close()
            if total_open + self.MIN_BET > self.bankroll * 0.8:
                self._log_skip(inst.asset, slug, "max_exposure",
                               ml_conf, token_cost, ml_direction, minutes_left)
                continue

            # Gate: bankroll
            if self.bankroll < self.MIN_BET:
                self._log_skip(inst.asset, slug,
                               f"bankroll ${self.bankroll:.2f} < ${self.MIN_BET}",
                               ml_conf, token_cost, ml_direction, minutes_left)
                continue

            # Get asset price for recording
            asset_price = current_prices.get(inst.price_pair, 0)

            # Place bet
            self._place_bet(inst, market, ml_direction, entry_side, token_cost,
                            ml_conf, current_regime, asset_price)

    # -- Active Position Management --

    def _manage_positions(self, ml_predictions: dict, current_prices: dict) -> None:
        """Check open bets and apply sell/add/hold rules."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        open_bets = conn.execute(
            "SELECT * FROM polymarket_bets WHERE status = 'OPEN'"
        ).fetchall()
        conn.close()

        for bet in open_bets:
            asset = bet["asset"]
            slug = bet["slug"]
            entry_side = bet["entry_side"]
            direction = "UP" if entry_side == "YES" else "DOWN"

            # Find instrument
            inst = self._find_instrument(asset)
            if not inst:
                continue

            # Fetch current market price ONCE (reused by stop-loss and add logic)
            market = self._fetch_market_by_slug(slug)
            current_share = bet["avg_cost"]  # fallback
            if market:
                crowd_up = self._parse_crowd_up(market)
                current_share = crowd_up if direction == "UP" else (1.0 - crowd_up)

            share_change = (current_share / bet["avg_cost"] - 1) if bet["avg_cost"] > 0 else 0

            # Rule 0: STOP LOSS — close if share price dropped too far
            if share_change <= -self.STOP_LOSS_PCT:
                self.logger.warning(
                    f"STOP LOSS [{asset}]: share {current_share:.3f} vs avg {bet['avg_cost']:.3f} "
                    f"({share_change*100:+.1f}%) | Limit: -{self.STOP_LOSS_PCT*100:.0f}%"
                )
                self._close_bet(bet, "stop_loss", current_prices)
                continue

            # Current ML data
            ml_data = ml_predictions.get(inst.ml_pair, {})
            ml_conf = ml_data.get("confidence", 50.0)
            ml_pred = ml_data.get("prediction", 0)
            ml_direction = "UP" if ml_pred > 0 else "DOWN"

            # Rule 1: Direction flipped -> SELL immediately
            if ml_direction != direction:
                self.logger.info(
                    f"SELL FLIP [{asset}]: ML flipped to {ml_direction} "
                    f"({ml_conf:.0f}% conf) | Was {direction}"
                )
                self._close_bet(bet, "direction_flip", current_prices)
                continue

            # Rule 2: Confidence below exit threshold -> SELL
            if ml_conf < self.EXIT_CONFIDENCE:
                self.logger.info(
                    f"SELL LOW CONF [{asset}]: ML confidence {ml_conf:.0f}% "
                    f"< {self.EXIT_CONFIDENCE}%"
                )
                self._close_bet(bet, "low_confidence", current_prices)
                continue

            # Rule 3: Add to position — only if WINNING (current > avg cost)
            if (ml_conf >= self.ADD_CONFIDENCE
                    and bet["total_invested"] < self.MAX_POSITION_PER_MARKET
                    and self.bankroll >= self.MIN_BET
                    and current_share > bet["avg_cost"]):
                if market:
                    token_cost = current_share
                    if token_cost <= self.MAX_TOKEN_COST and self._check_rate_limit():
                        self._add_to_bet(bet, token_cost, ml_conf)

            # Rule 4: Hold (implicit - do nothing)

    def _close_bet(self, bet, reason: str, current_prices: dict) -> None:
        """Close an open bet (active sell)."""
        market = self._fetch_market_by_slug(bet["slug"])
        exit_price = bet["avg_cost"]  # default to break-even
        if market:
            crowd_up = self._parse_crowd_up(market)
            direction = "UP" if bet["entry_side"] == "YES" else "DOWN"
            exit_price = crowd_up if direction == "UP" else (1.0 - crowd_up)

        pnl = round((exit_price - bet["avg_cost"]) * bet["total_tokens"], 2)
        return_pct = round(pnl / bet["total_invested"] * 100, 2) if bet["total_invested"] > 0 else 0

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE polymarket_bets
            SET status = 'CLOSED', exit_price = ?, exit_reason = ?,
                exit_at = datetime('now'), pnl = ?, return_pct = ?
            WHERE id = ?
        """, (exit_price, reason, pnl, return_pct, bet["id"]))
        conn.commit()
        conn.close()

        self.bankroll += bet["total_invested"] + pnl
        self._log_bankroll(f"closed_{reason}", str(bet["id"]), pnl)

        if pnl < 0:
            self._last_loss_time = time.time()

        self.logger.info(
            f"CLOSED [{bet['asset']}]: {reason} | P&L: ${pnl:+.2f} ({return_pct:+.1f}%) | "
            f"Bankroll: ${self.bankroll:.2f}"
        )

    # -- Kelly Sizing --

    def _compute_kelly_bet(self, probability: float, token_cost: float) -> float:
        """Half-Kelly optimal bet size for Polymarket.

        Args:
            probability: Model's P(correct outcome) — e.g. 0.55 means 55% chance we're right.
            token_cost: Price of the token we're buying (0.0 to 1.0).

        Returns:
            Dollar amount to bet (floored at MIN_BET, capped at MAX_BET_PCT of bankroll).
        """
        # Payout ratio: if we buy at token_cost and win, we get $1 per token
        b = (1.0 - token_cost) / (token_cost + 1e-10)

        # Win probability from model (already directional — prob of the side we're betting)
        p = max(0.5, min(0.99, probability))
        q = 1.0 - p

        # Full Kelly fraction
        kelly = (p * b - q) / (b + 1e-10)
        kelly = max(0.0, kelly)

        # Half-Kelly for safety
        half_kelly = kelly * 0.5

        # Dollar bet
        bet = self.bankroll * half_kelly

        # Floor and ceiling
        bet = max(self.MIN_BET, min(bet, self.bankroll * self.MAX_BET_PCT))
        return round(bet, 2)

    # -- Bet Placement --

    def _place_bet(self, inst: InstrumentConfig, market: dict,
                   direction: str, entry_side: str, token_cost: float,
                   ml_confidence: float, regime: str, asset_price: float) -> None:
        """Place a Kelly-sized bet."""
        # Convert confidence (50-100% scale) to probability (0.5-1.0)
        prob = ml_confidence / 100.0
        bet_amount = self._compute_kelly_bet(prob, token_cost)
        tokens = bet_amount / token_cost if token_cost > 0 else 0
        slug = market.get("slug", "")

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO polymarket_bets
            (slug, asset, entry_side, entry_token_cost, entry_amount, entry_tokens,
             entry_confidence, adds, total_invested, total_tokens, avg_cost,
             status, regime, entry_asset_price, question)
            VALUES (?, ?, ?, ?, ?, ?, ?, '[]', ?, ?, ?, 'OPEN', ?, ?, ?)
        """, (
            slug, inst.asset, entry_side, token_cost, bet_amount, tokens,
            ml_confidence, bet_amount, tokens, token_cost,
            regime, asset_price, market.get("question", ""),
        ))
        bet_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        conn.close()

        self.bankroll -= bet_amount
        self._bets_this_hour.append(time.time())
        self._log_bankroll("bet_placed", str(bet_id), -bet_amount)

        self.logger.info(
            f"BET [{inst.asset}]: {direction} ({entry_side}) | "
            f"Conf: {ml_confidence:.1f}% | Token: ${token_cost:.2f} | "
            f"Kelly: ${bet_amount:.2f} | Bankroll: ${self.bankroll:.2f}"
        )

    def _add_to_bet(self, bet, token_cost: float, ml_confidence: float) -> None:
        """Add a Kelly-sized increment to an existing open bet."""
        prob = ml_confidence / 100.0
        add_amount = self._compute_kelly_bet(prob, token_cost)
        new_tokens = add_amount / token_cost if token_cost > 0 else 0

        # Parse existing adds
        adds_raw = bet["adds"] or "[]"
        try:
            adds = json.loads(adds_raw)
        except (json.JSONDecodeError, TypeError):
            adds = []

        adds.append({
            "amount": add_amount,
            "token_cost": token_cost,
            "tokens": new_tokens,
            "confidence": ml_confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        new_total_invested = bet["total_invested"] + add_amount
        new_total_tokens = bet["total_tokens"] + new_tokens
        new_avg_cost = new_total_invested / new_total_tokens if new_total_tokens > 0 else token_cost

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE polymarket_bets
            SET adds = ?, total_invested = ?, total_tokens = ?, avg_cost = ?
            WHERE id = ?
        """, (json.dumps(adds), new_total_invested, new_total_tokens, new_avg_cost, bet["id"]))
        conn.commit()
        conn.close()

        self.bankroll -= add_amount
        self._bets_this_hour.append(time.time())
        self._log_bankroll("add_to_bet", str(bet["id"]), -add_amount)

        self.logger.info(
            f"ADD [{bet['asset']}]: +${add_amount:.2f} | "
            f"Total: ${new_total_invested:.0f}/${self.MAX_POSITION_PER_MARKET:.0f} | "
            f"Avg Cost: ${new_avg_cost:.3f} | Bankroll: ${self.bankroll:.2f}"
        )

    # -- Resolution --

    def _check_resolutions(self, current_prices: Optional[dict] = None) -> None:
        """Check if open bets have resolved via Gamma API or price fallback."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        open_bets = conn.execute(
            "SELECT * FROM polymarket_bets WHERE status = 'OPEN'"
        ).fetchall()

        for bet in open_bets:
            slug = bet["slug"]
            if not slug:
                continue

            try:
                market = self._fetch_market_by_slug(slug)
                if market and market.get("gamma_raw"):
                    raw = market["gamma_raw"]
                    is_resolved = raw.get("resolved", False)
                    prices = raw.get("outcomePrices", "[]")
                    if isinstance(prices, str):
                        try:
                            prices = json.loads(prices)
                        except (json.JSONDecodeError, TypeError):
                            prices = []

                    if isinstance(prices, list) and len(prices) >= 2:
                        yes_price = float(prices[0])
                        no_price = float(prices[1])
                        is_definitive = (yes_price >= 0.95 and no_price <= 0.05) or \
                                        (yes_price <= 0.05 and no_price >= 0.95)

                        if is_resolved or is_definitive:
                            self._resolve_bet(conn, bet, yes_price, no_price, "gamma_api", current_prices)
                            continue

                # Deadline-based resolution
                deadline_str = market.get("deadline", "") if market else ""
                seconds_past = 0
                if deadline_str:
                    try:
                        deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
                        seconds_past = (datetime.now(timezone.utc) - deadline).total_seconds()
                    except (ValueError, TypeError):
                        pass

                # Price-based fallback (>2 min past deadline)
                if seconds_past > 120 and current_prices:
                    inst = self._find_instrument(bet["asset"])
                    if inst:
                        current_asset_price = current_prices.get(inst.price_pair, 0)
                        entry_asset_price = bet["entry_asset_price"] or 0
                        if current_asset_price > 0 and entry_asset_price > 0:
                            went_up = current_asset_price > entry_asset_price
                            yes_price = 1.0 if went_up else 0.0
                            no_price = 1.0 - yes_price
                            self._resolve_bet(conn, bet, yes_price, no_price, "price_fallback", current_prices)
                            continue

                # Force-expire after 30 min
                if seconds_past > 1800:
                    self.logger.info(f"FORCE-EXPIRE [{bet['asset']}]: {slug}")
                    conn.execute("""
                        UPDATE polymarket_bets
                        SET status = 'CLOSED', exit_reason = 'force_expired',
                            exit_at = datetime('now'), pnl = 0, return_pct = 0
                        WHERE id = ?
                    """, (bet["id"],))
                    self.bankroll += bet["total_invested"]
                    self._log_bankroll("force_expired", str(bet["id"]), bet["total_invested"])

            except Exception as e:
                self.logger.debug(f"Resolution check failed for bet {bet['id']}: {e}")

        conn.commit()
        conn.close()

        # Also check old polymarket_positions table for legacy positions
        self._check_legacy_resolutions(current_prices)

    def _resolve_bet(self, conn, bet, yes_price: float, no_price: float,
                     source: str, current_prices: Optional[dict] = None) -> None:
        """Resolve a bet as WON or LOST."""
        side = bet["entry_side"]
        won = (yes_price >= 0.95) if side == "YES" else (no_price >= 0.95)
        exit_price = 1.0 if won else 0.0
        pnl = round((exit_price * bet["total_tokens"]) - bet["total_invested"], 2)
        return_pct = round(pnl / bet["total_invested"] * 100, 2) if bet["total_invested"] > 0 else 0
        status = "WON" if won else "LOST"

        exit_asset_price = 0
        if current_prices:
            inst = self._find_instrument(bet["asset"])
            if inst:
                exit_asset_price = current_prices.get(inst.price_pair, 0)

        conn.execute("""
            UPDATE polymarket_bets
            SET status = ?, exit_price = ?, exit_reason = ?,
                exit_at = datetime('now'), pnl = ?, return_pct = ?,
                exit_asset_price = ?
            WHERE id = ?
        """, (status, exit_price, source, pnl, return_pct, exit_asset_price, bet["id"]))

        if won:
            self.bankroll += bet["total_invested"] + pnl
        if not won:
            self._last_loss_time = time.time()

        self.logger.info(
            f"{'WON' if won else 'LOST'} [{bet['asset']}]: "
            f"{side} | P&L: ${pnl:+.2f} ({return_pct:+.1f}%) | "
            f"Bankroll: ${self.bankroll:.2f} | via {source}"
        )
        self._log_bankroll(f"resolved_{status.lower()}", str(bet["id"]), pnl)

    def _check_legacy_resolutions(self, current_prices: Optional[dict] = None) -> None:
        """Check for any still-open legacy positions in polymarket_positions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            open_positions = conn.execute(
                "SELECT * FROM polymarket_positions WHERE status = 'open' AND strategy = 'strategy_a'"
            ).fetchall()
        except Exception:
            conn.close()
            return

        for pos in open_positions:
            slug = pos["slug"]
            if not slug:
                continue

            deadline_str = pos["deadline"]
            seconds_past = 0
            if deadline_str:
                try:
                    deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
                    seconds_past = (datetime.now(timezone.utc) - deadline).total_seconds()
                except (ValueError, TypeError):
                    pass

            try:
                market = self._fetch_market_by_slug(slug)
                if market and market.get("gamma_raw"):
                    raw = market["gamma_raw"]
                    is_resolved = raw.get("resolved", False)
                    prices = raw.get("outcomePrices", "[]")
                    if isinstance(prices, str):
                        try:
                            prices = json.loads(prices)
                        except (json.JSONDecodeError, TypeError):
                            prices = []

                    if isinstance(prices, list) and len(prices) >= 2:
                        yes_price = float(prices[0])
                        no_price = float(prices[1])
                        is_definitive = (yes_price >= 0.95 and no_price <= 0.05) or \
                                        (yes_price <= 0.05 and no_price >= 0.95)

                        if is_resolved or is_definitive:
                            direction = pos["direction"]
                            won = (yes_price >= 0.95) if direction == "UP" else (no_price >= 0.95)
                            exit_price = 1.0 if won else 0.0
                            pnl = round((exit_price * pos["shares"]) - pos["bet_amount"], 2)
                            status = "won" if won else "lost"

                            conn.execute("""
                                UPDATE polymarket_positions
                                SET status = ?, exit_price = ?, pnl = ?, closed_at = datetime('now'),
                                    notes = COALESCE(notes, '') || ' | resolved_via=gamma_api'
                                WHERE position_id = ?
                            """, (status, exit_price, pnl, pos["position_id"]))

                            if won:
                                self.bankroll += pos["bet_amount"] + pnl
                            self._log_bankroll(f"legacy_resolved_{status}", pos["position_id"], pnl)
                            continue

                # Force-expire legacy positions 30 min past deadline
                if seconds_past > 1800:
                    conn.execute("""
                        UPDATE polymarket_positions
                        SET status = 'expired', pnl = 0, closed_at = datetime('now'),
                            notes = 'Force-expired: 30min past deadline'
                        WHERE position_id = ?
                    """, (pos["position_id"],))
                    self.bankroll += pos["bet_amount"]
                    self._log_bankroll("legacy_force_expired", pos["position_id"], pos["bet_amount"])

            except Exception as e:
                self.logger.debug(f"Legacy resolution check failed for {pos['position_id']}: {e}")

        conn.commit()
        conn.close()

    # -- Market Discovery --

    def _discover_market(self, inst: InstrumentConfig) -> Optional[dict]:
        now_ts = int(time.time())
        window_ts = (now_ts // 900) * 900
        slug = inst.slug_pattern.format(ts=window_ts)

        cached = self._market_cache.get(slug)
        if cached and (now_ts - cached[1] < MARKET_CACHE_TTL):
            return cached[0]

        market = self._fetch_market_by_slug(slug)
        if market:
            self._market_cache[slug] = (market, now_ts)
            return market

        # Try previous window
        prev_slug = inst.slug_pattern.format(ts=window_ts - 900)
        cached_prev = self._market_cache.get(prev_slug)
        if cached_prev and (now_ts - cached_prev[1] < MARKET_CACHE_TTL):
            return cached_prev[0]

        market = self._fetch_market_by_slug(prev_slug)
        if market:
            self._market_cache[prev_slug] = (market, now_ts)
            return market

        return None

    def _fetch_market_by_slug(self, slug: str) -> Optional[dict]:
        try:
            resp = requests.get(GAMMA_MARKETS_URL, params={"slug": slug}, timeout=10)
            if resp.status_code != 200:
                return None
            markets = resp.json()
            if not markets:
                return None

            m = markets[0]
            crowd_up = self._parse_crowd_up_raw(m)

            return {
                "slug": m.get("slug", slug),
                "question": m.get("question", ""),
                "market_type": "DIRECTION",
                "asset": self._extract_asset(m.get("question", "")),
                "condition_id": m.get("conditionId", ""),
                "market_id": m.get("id", ""),
                "crowd_prob_yes": crowd_up,
                "deadline": m.get("endDate", ""),
                "volume_24h": float(m.get("volume24hr", 0) or 0),
                "liquidity": float(m.get("liquidity", 0) or 0),
                "resolved": m.get("resolved", False),
                "gamma_raw": m,
            }
        except Exception as e:
            self.logger.debug(f"Gamma API fetch failed for {slug}: {e}")
            return None

    @staticmethod
    def _parse_crowd_up(market: dict) -> float:
        """Extract YES price from cached market dict."""
        if "crowd_prob_yes" in market and market["crowd_prob_yes"] is not None:
            raw = market.get("gamma_raw")
            if raw is None:
                return float(market["crowd_prob_yes"])

        raw = market.get("gamma_raw", market)
        return StrategyAExecutor._parse_crowd_up_raw(raw)

    @staticmethod
    def _parse_crowd_up_raw(raw: dict) -> float:
        """Extract YES price from raw Gamma API response."""
        prices = raw.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except (json.JSONDecodeError, TypeError):
                prices = []
        if isinstance(prices, list) and len(prices) >= 1:
            try:
                return float(prices[0])
            except (ValueError, TypeError):
                pass
        best_ask = raw.get("bestAsk")
        if best_ask:
            try:
                return float(best_ask)
            except (ValueError, TypeError):
                pass
        return 0.5

    @staticmethod
    def _extract_asset(question: str) -> str:
        q = question.lower()
        for name, symbol in [
            ("bitcoin", "BTC"), ("btc", "BTC"),
            ("ethereum", "ETH"), ("eth", "ETH"),
            ("solana", "SOL"), ("sol", "SOL"),
            ("dogecoin", "DOGE"), ("doge", "DOGE"),
            ("xrp", "XRP"), ("ripple", "XRP"),
        ]:
            if name in q:
                return symbol
        return "UNKNOWN"

    def _get_minutes_remaining(self, market: dict) -> Optional[float]:
        deadline_str = market.get("deadline", "")
        if not deadline_str:
            return None
        try:
            deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
            return (deadline - datetime.now(timezone.utc)).total_seconds() / 60.0
        except (ValueError, TypeError):
            return None

    # -- Helpers --

    def _find_instrument(self, asset: str) -> Optional[InstrumentConfig]:
        for inst in self.instruments.values():
            if inst.asset == asset and inst.enabled:
                return inst
        return None

    def _find_instrument_key(self, asset: str) -> Optional[str]:
        for key, inst in self.instruments.items():
            if inst.asset == asset and inst.enabled:
                return key
        return None

    def get_stats(self) -> dict:
        """Return stats for dashboard (reads from new polymarket_bets table)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            open_count = conn.execute(
                "SELECT COUNT(*) as c FROM polymarket_bets WHERE status = 'OPEN'"
            ).fetchone()["c"]
            total = conn.execute(
                "SELECT COUNT(*) as c, "
                "SUM(CASE WHEN status='WON' THEN 1 ELSE 0 END) as wins, "
                "SUM(CASE WHEN status='LOST' THEN 1 ELSE 0 END) as losses, "
                "COALESCE(SUM(pnl), 0) as pnl "
                "FROM polymarket_bets WHERE status IN ('WON','LOST','CLOSED')"
            ).fetchone()
            return {
                "bankroll": round(self.bankroll, 2),
                "open_positions": open_count,
                "total_resolved": total["c"],
                "wins": total["wins"] or 0,
                "losses": total["losses"] or 0,
                "total_pnl": round(total["pnl"] or 0, 2),
            }
        except Exception:
            return {"bankroll": self.bankroll, "open_positions": 0}
        finally:
            conn.close()
