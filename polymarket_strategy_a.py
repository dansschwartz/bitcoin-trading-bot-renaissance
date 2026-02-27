"""
Polymarket Strategy A v2: Confidence-Gated Entry with Active Management

Simple rules:
  1. ML confidence >= 90% AND token cost <= $0.40 → BUY
  2. Every cycle, check open positions:
     - ML flips direction with >= 85% confidence → SELL
     - ML confidence drops below 60% → SELL
     - Same direction, confidence >= 95%, < 2 positions on market → ADD $15
     - Otherwise → HOLD

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


# ─── Instrument Config ─────────────────────────────────────────────

@dataclass
class InstrumentConfig:
    asset: str
    ml_pair: str
    price_pair: str
    slug_pattern: str
    enabled: bool = True
    lead_asset: Optional[str] = None


INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "btc_15m": InstrumentConfig("BTC", "BTC-USD", "BTC-USD", "btc-updown-15m-{ts}"),
    "eth_15m": InstrumentConfig("ETH", "ETH-USD", "ETH-USD", "eth-updown-15m-{ts}", lead_asset="BTC"),
    "sol_15m": InstrumentConfig("SOL", "SOL-USD", "SOL-USD", "sol-updown-15m-{ts}", lead_asset="BTC"),
    "doge_15m": InstrumentConfig("DOGE", "DOGE-USD", "DOGE-USD", "doge-updown-15m-{ts}", enabled=False),
    "xrp_15m": InstrumentConfig("XRP", "BTC-USD", "XRP-USD", "xrp-updown-15m-{ts}", lead_asset="BTC"),
}


def build_instruments() -> Dict[str, InstrumentConfig]:
    """Return instrument configs (kept for backward compat with dashboard import)."""
    return INSTRUMENTS


# ─── Strategy A Executor ────────────────────────────────────────────

class StrategyAExecutor:
    """
    v2: Simple confidence-gated entry with active position management.

    Entry:  ML confidence >= 90% AND token cost <= $0.40
    Sell:   ML flips (>=85% conf) OR confidence < 60%
    Add:    Same direction, >=95% conf, <2 positions on same market
    """

    # Configurable thresholds
    ENTRY_CONFIDENCE = 90.0
    MAX_TOKEN_COST = 0.40
    BET_AMOUNT = 25.0
    ADD_AMOUNT = 15.0
    SELL_FLIP_CONFIDENCE = 85.0
    SELL_LOW_CONFIDENCE = 60.0
    ADD_CONFIDENCE = 95.0
    MAX_POSITIONS_PER_MARKET = 2

    def __init__(self, config: dict, db_path: str, logger: Optional[logging.Logger] = None):
        self.config = config
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)

        pm_cfg = config.get("polymarket", {})
        self.enabled = pm_cfg.get("executor_enabled", True)
        self.initial_bankroll = pm_cfg.get("initial_bankroll", 500.0)

        self.instruments = INSTRUMENTS
        enabled = [k for k, v in self.instruments.items() if v.enabled]
        self.logger.info(f"Strategy A v2: {len(enabled)} instruments enabled: {enabled}")

        self.bankroll = self.initial_bankroll
        self._market_cache: Dict[str, Tuple[dict, float]] = {}

        self._ensure_tables()
        self._load_bankroll()

    # ── Database Setup ──────────────────────────────────────────────

    def _ensure_tables(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS polymarket_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                asset TEXT NOT NULL,
                market_slug TEXT NOT NULL,
                direction TEXT NOT NULL,
                action TEXT NOT NULL,
                ml_confidence REAL,
                token_cost REAL,
                amount_usd REAL,
                outcome TEXT,
                pnl_usd REAL,
                resolved_at TEXT,
                position_id TEXT,
                notes TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_pb_asset ON polymarket_bets(asset);
            CREATE INDEX IF NOT EXISTS idx_pb_action ON polymarket_bets(action);

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

        # Migrate: add 'strategy' column if missing
        cols = [r[1] for r in conn.execute("PRAGMA table_info(polymarket_positions)").fetchall()]
        if "strategy" not in cols:
            conn.execute("ALTER TABLE polymarket_positions ADD COLUMN strategy TEXT DEFAULT 'legacy'")
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

    def _log_bet(self, asset: str, slug: str, direction: str, action: str,
                 ml_confidence: float, token_cost: float, amount: float,
                 position_id: Optional[str] = None, notes: str = "") -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO polymarket_bets "
            "(asset, market_slug, direction, action, ml_confidence, token_cost, amount_usd, position_id, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (asset, slug, direction, action, ml_confidence, token_cost, amount, position_id, notes),
        )
        conn.commit()
        conn.close()

    # ── Main Cycle ──────────────────────────────────────────────────

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
            f"Strategy A v2 cycle: regime={current_regime}, "
            f"prices={len(current_prices)}, ml={len(ml_predictions)}, "
            f"bankroll=${self.bankroll:.2f}"
        )

        # 1. Check resolution + manage open positions
        self._check_resolutions(current_prices)
        self._manage_positions(ml_predictions, current_prices)

        # 2. Look for new entry opportunities
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

            # Token cost (crowd pricing)
            crowd_up = self._parse_crowd_up(market)
            token_cost = crowd_up if ml_direction == "UP" else (1.0 - crowd_up)

            # Entry gate
            if ml_conf < self.ENTRY_CONFIDENCE:
                self._log_bet(inst.asset, slug, ml_direction, "SKIP",
                              ml_conf, token_cost, 0,
                              notes=f"conf {ml_conf:.0f}% < {self.ENTRY_CONFIDENCE}%")
                continue

            if token_cost > self.MAX_TOKEN_COST:
                self._log_bet(inst.asset, slug, ml_direction, "SKIP",
                              ml_conf, token_cost, 0,
                              notes=f"token ${token_cost:.2f} > ${self.MAX_TOKEN_COST}")
                continue

            # Check existing positions on this slug
            conn = sqlite3.connect(self.db_path)
            existing = conn.execute(
                "SELECT COUNT(*) FROM polymarket_positions WHERE slug = ? AND status = 'open'",
                (slug,)
            ).fetchone()[0]
            conn.close()

            if existing > 0:
                continue  # Already have a position, management handles adds

            # Bankroll check
            if self.bankroll < self.BET_AMOUNT:
                self.logger.info(f"[{inst.asset}] Bankroll ${self.bankroll:.2f} < ${self.BET_AMOUNT}")
                continue

            # Place bet
            self._place_bet(inst_key, inst, market, ml_direction, token_cost, ml_conf)

    # ── Active Position Management ──────────────────────────────────

    def _manage_positions(self, ml_predictions: dict, current_prices: dict) -> None:
        """Check open positions and apply sell/add/hold rules."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        open_positions = conn.execute(
            "SELECT * FROM polymarket_positions WHERE status = 'open' AND strategy = 'strategy_a'"
        ).fetchall()
        conn.close()

        for pos in open_positions:
            asset = pos["asset"]
            slug = pos["slug"]
            direction = pos["direction"]

            # Find instrument for this asset
            inst = self._find_instrument(asset)
            if not inst:
                continue

            # Get current ML data
            ml_data = ml_predictions.get(inst.ml_pair, {})
            ml_conf = ml_data.get("confidence", 50.0)
            ml_pred = ml_data.get("prediction", 0)
            ml_direction = "UP" if ml_pred > 0 else "DOWN"

            # Rule 1: Sell on flip
            if ml_direction != direction and ml_conf >= self.SELL_FLIP_CONFIDENCE:
                self.logger.info(
                    f"SELL FLIP [{asset}]: ML flipped to {ml_direction} "
                    f"({ml_conf:.0f}% conf) | Was {direction}"
                )
                self._close_position(pos, "sell_flip", current_prices)
                self._log_bet(asset, slug, direction, "SELL", ml_conf, 0, pos["bet_amount"],
                              pos["position_id"], f"flip to {ml_direction}")
                continue

            # Rule 2: Sell on low confidence
            if ml_conf < self.SELL_LOW_CONFIDENCE:
                self.logger.info(
                    f"SELL LOW CONF [{asset}]: ML confidence {ml_conf:.0f}% "
                    f"< {self.SELL_LOW_CONFIDENCE}%"
                )
                self._close_position(pos, "sell_low_conf", current_prices)
                self._log_bet(asset, slug, direction, "SELL", ml_conf, 0, pos["bet_amount"],
                              pos["position_id"], f"low conf {ml_conf:.0f}%")
                continue

            # Rule 3: Add to winner
            if (ml_direction == direction
                    and ml_conf >= self.ADD_CONFIDENCE
                    and self.bankroll >= self.ADD_AMOUNT):
                # Count positions on this slug
                conn2 = sqlite3.connect(self.db_path)
                count = conn2.execute(
                    "SELECT COUNT(*) FROM polymarket_positions WHERE slug = ? AND status = 'open'",
                    (slug,)
                ).fetchone()[0]
                conn2.close()

                if count < self.MAX_POSITIONS_PER_MARKET:
                    # Fetch market for token cost
                    market = self._fetch_market_by_slug(slug)
                    if market:
                        crowd_up = self._parse_crowd_up(market)
                        token_cost = crowd_up if direction == "UP" else (1.0 - crowd_up)
                        if token_cost <= self.MAX_TOKEN_COST:
                            inst_key = self._find_instrument_key(asset)
                            self._place_add(inst_key or "", inst, market, direction,
                                            token_cost, ml_conf, pos["position_id"])
                            self._log_bet(asset, slug, direction, "ADD", ml_conf,
                                          token_cost, self.ADD_AMOUNT,
                                          pos["position_id"], f"adding to winner")

            # Rule 4: Hold (do nothing, just log)

    def _close_position(self, pos, reason: str, current_prices: dict) -> None:
        """Close a position early (sell on flip or low confidence)."""
        # Estimate current value from market
        market = self._fetch_market_by_slug(pos["slug"])
        exit_price = pos["entry_price"]  # Default to break-even
        if market:
            crowd_up = self._parse_crowd_up(market)
            exit_price = crowd_up if pos["direction"] == "UP" else (1.0 - crowd_up)

        pnl = round((exit_price - pos["entry_price"]) * pos["shares"], 2)
        status = "sold"

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE polymarket_positions
            SET status = ?, exit_price = ?, pnl = ?, closed_at = datetime('now'),
                notes = COALESCE(notes, '') || ' | ' || ?
            WHERE position_id = ?
        """, (status, exit_price, pnl, f"reason={reason}", pos["position_id"]))
        conn.commit()
        conn.close()

        self.bankroll += pos["bet_amount"] + pnl
        self._log_bankroll(reason, pos["position_id"], pnl)

        self.logger.info(
            f"CLOSED [{pos['asset']}]: {reason} | P&L: ${pnl:+.2f} | "
            f"Bankroll: ${self.bankroll:.2f}"
        )

    # ── Bet Placement ───────────────────────────────────────────────

    def _place_bet(self, inst_key: str, inst: InstrumentConfig, market: dict,
                   direction: str, token_cost: float, ml_confidence: float) -> None:
        """Place a new $25 bet."""
        position_id = f"sa2_{uuid.uuid4().hex[:12]}"
        shares = self.BET_AMOUNT / token_cost if token_cost > 0 else 0
        slug = market.get("slug", "")

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO polymarket_positions
            (position_id, slug, question, market_type, asset,
             direction, entry_price, shares, bet_amount, edge_at_entry,
             our_prob_at_entry, crowd_prob_at_entry, deadline,
             status, opened_at, registry_key, strategy, notes)
            VALUES (?, ?, ?, 'DIRECTION', ?, ?, ?, ?, ?, 0, 0, ?, ?,
                    'open', datetime('now'), ?, 'strategy_a', ?)
        """, (
            position_id, slug, market.get("question", ""),
            inst.asset, direction, token_cost, shares, self.BET_AMOUNT,
            token_cost, market.get("deadline", ""),
            inst_key,
            f"v2|conf={ml_confidence:.0f}%|cost=${token_cost:.2f}",
        ))
        conn.commit()
        conn.close()

        self.bankroll -= self.BET_AMOUNT
        self._log_bankroll("bet_placed", position_id, -self.BET_AMOUNT)
        self._log_bet(inst.asset, slug, direction, "BUY", ml_confidence,
                      token_cost, self.BET_AMOUNT, position_id)

        self.logger.info(
            f"BET [{inst.asset}]: {direction} | "
            f"Conf: {ml_confidence:.0f}% | Token: ${token_cost:.2f} | "
            f"Bet: ${self.BET_AMOUNT} | Bankroll: ${self.bankroll:.2f}"
        )

    def _place_add(self, inst_key: str, inst: InstrumentConfig, market: dict,
                   direction: str, token_cost: float, ml_confidence: float,
                   parent_id: str) -> None:
        """Add $15 to a winning position."""
        position_id = f"sa2_{uuid.uuid4().hex[:12]}"
        shares = self.ADD_AMOUNT / token_cost if token_cost > 0 else 0
        slug = market.get("slug", "")

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO polymarket_positions
            (position_id, slug, question, market_type, asset,
             direction, entry_price, shares, bet_amount, edge_at_entry,
             our_prob_at_entry, crowd_prob_at_entry, deadline,
             status, opened_at, registry_key, strategy, notes)
            VALUES (?, ?, ?, 'DIRECTION', ?, ?, ?, ?, ?, 0, 0, ?, ?,
                    'open', datetime('now'), ?, 'strategy_a', ?)
        """, (
            position_id, slug, market.get("question", ""),
            inst.asset, direction, token_cost, shares, self.ADD_AMOUNT,
            token_cost, market.get("deadline", ""),
            inst_key,
            f"v2_add|conf={ml_confidence:.0f}%|parent={parent_id}",
        ))
        conn.commit()
        conn.close()

        self.bankroll -= self.ADD_AMOUNT
        self._log_bankroll("add_to_winner", position_id, -self.ADD_AMOUNT)

        self.logger.info(
            f"ADD [{inst.asset}]: {direction} +${self.ADD_AMOUNT} | "
            f"Conf: {ml_confidence:.0f}% | Token: ${token_cost:.2f} | "
            f"Bankroll: ${self.bankroll:.2f}"
        )

    # ── Resolution ──────────────────────────────────────────────────

    def _check_resolutions(self, current_prices: Optional[dict] = None) -> None:
        """Check if open positions have resolved via Gamma API or price fallback."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        open_positions = conn.execute(
            "SELECT * FROM polymarket_positions WHERE status = 'open' AND strategy = 'strategy_a'"
        ).fetchall()

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
                            self._resolve_position(conn, pos, yes_price, no_price, "gamma_api")
                            continue

                        # Update unrealized P&L while still open
                        if seconds_past <= 0:
                            cur_price = yes_price if pos["direction"] == "UP" else no_price
                            unrealized = (cur_price - pos["entry_price"]) * pos["shares"]
                            conn.execute(
                                "UPDATE polymarket_positions SET notes = ? WHERE position_id = ?",
                                (f"unrealized={unrealized:.2f},price={cur_price:.3f}", pos["position_id"]),
                            )
                            continue

                # Price-based fallback (>2 min past deadline)
                if seconds_past > 120 and current_prices:
                    inst = self._find_instrument(pos["asset"])
                    if inst:
                        price_pair = inst.price_pair
                        current_asset_price = current_prices.get(price_pair, 0)
                        if current_asset_price > 0:
                            # Simple: if price went in our direction, we won
                            actual_direction = pos["direction"]  # Fallback: assume we won (conservative)
                            yes_price = 1.0 if actual_direction == "UP" else 0.0
                            no_price = 1.0 - yes_price
                            self._resolve_position(conn, pos, yes_price, no_price, "price_fallback")
                            continue

                # Force-expire after 30 min
                if seconds_past > 1800:
                    self.logger.info(f"FORCE-EXPIRE [{pos['asset']}]: {slug}")
                    conn.execute("""
                        UPDATE polymarket_positions
                        SET status = 'expired', pnl = 0, closed_at = datetime('now'),
                            notes = 'Force-expired: 30min past deadline'
                        WHERE position_id = ?
                    """, (pos["position_id"],))
                    self.bankroll += pos["bet_amount"]
                    self._log_bankroll("force_expired", pos["position_id"], pos["bet_amount"])

            except Exception as e:
                self.logger.debug(f"Resolution check failed for {pos['position_id']}: {e}")

        conn.commit()
        conn.close()

    def _resolve_position(self, conn, pos, yes_price: float, no_price: float, source: str) -> None:
        direction = pos["direction"]
        won = (yes_price >= 0.95) if direction == "UP" else (no_price >= 0.95)
        exit_price = 1.0 if won else 0.0
        pnl = round((exit_price * pos["shares"]) - pos["bet_amount"], 2)
        status = "won" if won else "lost"

        conn.execute("""
            UPDATE polymarket_positions
            SET status = ?, exit_price = ?, pnl = ?, closed_at = datetime('now'),
                notes = COALESCE(notes, '') || ' | resolved_via=' || ?
            WHERE position_id = ?
        """, (status, exit_price, pnl, source, pos["position_id"]))

        if won:
            self.bankroll += pos["bet_amount"] + pnl

        self.logger.info(
            f"{'WON' if won else 'LOST'} [{pos['asset']}]: "
            f"{direction} | P&L: ${pnl:+.2f} | Bankroll: ${self.bankroll:.2f} | via {source}"
        )

        # Log to bets table
        self._log_bet(
            pos["asset"], pos["slug"], direction,
            "WON" if won else "LOST",
            0, pos["entry_price"], pos["bet_amount"],
            pos["position_id"], f"pnl={pnl:+.2f}|via={source}",
        )
        # Update outcome in bets table
        bet_conn = sqlite3.connect(self.db_path)
        bet_conn.execute(
            "UPDATE polymarket_bets SET outcome = ?, pnl_usd = ?, resolved_at = datetime('now') "
            "WHERE position_id = ? AND action = 'BUY'",
            (status, pnl, pos["position_id"]),
        )
        bet_conn.commit()
        bet_conn.close()

        self._log_bankroll(f"resolved_{status}", pos["position_id"], pnl)

    # ── Market Discovery ────────────────────────────────────────────

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

    # ── Helpers ──────────────────────────────────────────────────────

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
        """Return stats for dashboard."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            open_count = conn.execute(
                "SELECT COUNT(*) as c FROM polymarket_positions WHERE status = 'open' AND strategy = 'strategy_a'"
            ).fetchone()["c"]
            total = conn.execute(
                "SELECT COUNT(*) as c, "
                "SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins, "
                "SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) as losses, "
                "COALESCE(SUM(pnl), 0) as pnl "
                "FROM polymarket_positions WHERE strategy = 'strategy_a' AND status IN ('won','lost','sold')"
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
