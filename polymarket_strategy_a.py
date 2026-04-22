"""
Polymarket Strategy A v4: Multi-Asset Crash Models with Per-Horizon Routing

Simple rules:
  1. ML confidence >= 52% -> BUY (half-Kelly sized)
  2. Every cycle, manage open bets:
     - ML flips direction -> SELL immediately
     - ML confidence drops below 50% -> SELL
     - ML confidence >= 52% + under $150 cap -> ADD
     - Otherwise -> HOLD
  3. Rate limit: max 6 bets per hour
  4. Cooldown: 5 min after any loss

ML Source:
  Multi-asset crash-regime LightGBMs (52-53% acc, 0.53-0.55 AUC).
  Per-horizon routing: 15-min markets use 2bar model, 5-min markets use 1bar model.
  Half-Kelly sizing: 8-15% of bankroll per bet.

Market Discovery:
  Rolling 15m and 5m direction markets via slug pattern:
    {asset}-updown-15m-{unix_timestamp} (900s alignment)
    {asset}-updown-5m-{unix_timestamp}  (300s alignment)
  Discovered via Gamma API (no scanner dependency).
  Instruments: BTC, ETH, SOL, XRP (both timeframes).
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

from polymarket_timing_features import TimingFeatureEngine

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
    timeframe: int = 15  # minutes (15 or 5)
    kelly_fraction: float = 0.5   # Half-Kelly by default
    max_bet_usd: float = 0.0     # 0 = use global MAX_BET_PCT
    lead_asset: str = ""          # If set, require this asset's ML to agree on direction


INSTRUMENTS: Dict[str, InstrumentConfig] = {
    # 15-minute direction markets (2-bar / 10-min ML horizon, 53.3% acc)
    # Optimizer: SOL profitable, BTC marginal (cheap tokens only), XRP zero edge
    "btc_15m": InstrumentConfig("BTC", "BTC-USD", "BTC-USD", "btc-updown-15m-{ts}",
                                kelly_fraction=0.25, max_bet_usd=20.0, enabled=False),
    "eth_15m": InstrumentConfig("ETH", "ETH-USD", "ETH-USD", "eth-updown-15m-{ts}", enabled=False),
    "sol_15m": InstrumentConfig("SOL", "SOL-USD", "SOL-USD", "sol-updown-15m-{ts}",
                                kelly_fraction=0.65, max_bet_usd=30.0),
    "doge_15m": InstrumentConfig("DOGE", "DOGE-USD", "DOGE-USD", "doge-updown-15m-{ts}", enabled=False),
    "xrp_15m": InstrumentConfig("XRP", "XRP-USD", "XRP-USD", "xrp-updown-15m-{ts}", enabled=False),
    # 5-minute direction markets (1-bar / 5-min ML horizon, 52.4% acc)
    # Optimizer: SOL 5m profitable, BTC 5m zero edge, XRP zero edge
    "btc_5m": InstrumentConfig("BTC", "BTC-USD", "BTC-USD", "btc-updown-5m-{ts}",
                               timeframe=5, kelly_fraction=0.25, max_bet_usd=10.0, enabled=False),
    "eth_5m": InstrumentConfig("ETH", "ETH-USD", "ETH-USD", "eth-updown-5m-{ts}",
                               timeframe=5, kelly_fraction=0.4, max_bet_usd=20.0, lead_asset="BTC", enabled=False),
    "sol_5m": InstrumentConfig("SOL", "SOL-USD", "SOL-USD", "sol-updown-5m-{ts}",
                               timeframe=5, kelly_fraction=0.65, max_bet_usd=20.0, lead_asset="BTC"),
    "xrp_5m": InstrumentConfig("XRP", "XRP-USD", "XRP-USD", "xrp-updown-5m-{ts}",
                               timeframe=5, kelly_fraction=0.4, max_bet_usd=25.0, lead_asset="BTC", enabled=False),
    # DOGE: best calibration accuracy (52.4%), uses BTC crash model as proxy
    "doge_5m": InstrumentConfig("DOGE", "DOGE-USD", "DOGE-USD", "doge-updown-5m-{ts}",
                                timeframe=5, kelly_fraction=0.5, max_bet_usd=15.0, lead_asset="BTC"),
}


def build_instruments() -> Dict[str, InstrumentConfig]:
    """Return instrument configs (kept for backward compat with dashboard import)."""
    return INSTRUMENTS


# --- Strategy A Executor ---

class StrategyAExecutor:
    """
    v3: Confidence-gated entry with active position management.

    Entry:  ML confidence >= 52%
    Sell:   ML flips direction OR confidence < 50%
    Add:    Same direction, >= 52% conf, < $150 total
    Limits: 6 bets/hour, 5 min cooldown after loss

    Sizing: Half-Kelly based on model probability and token price.
    """

    # Thresholds — optimizer-tuned: only 52.0-52.5% confidence band is profitable
    CONFIDENCE_THRESHOLD = 52.0       # Model prob >= 0.52 (or <= 0.48)
    CONFIDENCE_CAP = 52.5             # Max confidence; higher is overfit/destructive (optimizer finding)
    BET_AMOUNT = 50.0                 # Fallback; overridden by Kelly sizing
    EXIT_CONFIDENCE = 50.0            # Exit when model is pure coin-flip
    ADD_CONFIDENCE = 52.0             # Same as entry threshold
    MAX_POSITION_PER_MARKET = 150.0   # dollar cap
    STOP_LOSS_PCT = None              # disabled — direction_flip handles exits; binary options resolve in 15min
    MAX_BETS_PER_HOUR = 16            # 8 instruments × ~2 bets/hr each
    COOLDOWN_AFTER_LOSS = 120         # 2 min (was 5min — too long for 5min markets)
    MIN_BET = 5.0                     # Floor for Kelly sizing
    MAX_BET_PCT = 0.05                # Ceiling: 5% of bankroll per bet (was 15%, reduced after 72% drawdown on 2026-03-02)
    MAX_BET_USD = 20.0                # Hard dollar cap per bet (was $50; optimizer found smaller bets lose less)
    MAX_SIZING_BANKROLL = 1000.0      # Cap effective bankroll for sizing (prevents runaway compounding)
    MIN_BANKROLL_TO_TRADE = 50.0      # Stop betting below this bankroll level
    DAILY_LOSS_LIMIT_PCT = 0.10       # Max 10% daily loss (was 20%; optimizer recommended tighter)
    DAILY_LOSS_LIMIT_5M_PCT = 0.05   # Max 5% daily loss for 5m markets (was 10%)
    MAX_ASSET_CONCENTRATION = 0.50    # Max 50% of open exposure in any single asset
    MAX_OPEN_BETS = 8                 # Max total concurrent open bets across all instruments

    def __init__(self, config: dict, db_path: str, logger: Optional[logging.Logger] = None):
        self.config = config
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)

        pm_cfg = config.get("polymarket", {})
        self.enabled = pm_cfg.get("executor_enabled", True)
        self.initial_bankroll = pm_cfg.get("initial_bankroll", 500.0)

        self.instruments = INSTRUMENTS
        enabled = [k for k, v in self.instruments.items() if v.enabled]
        self.logger.info(f"Strategy A v4: {len(enabled)} instruments enabled: {enabled}")
        self.logger.info(
            f"Strategy A sizing caps: MAX_BET_USD=${self.MAX_BET_USD}, "
            f"MAX_SIZING_BANKROLL=${self.MAX_SIZING_BANKROLL}, "
            f"MAX_BET_PCT={self.MAX_BET_PCT*100:.0f}%"
        )

        self.bankroll = self.initial_bankroll
        self._market_cache: Dict[str, Tuple[dict, float]] = {}

        # Rate limiting & cooldown state
        self._bets_this_hour: List[float] = []
        self._last_loss_time: float = 0.0

        # Daily loss tracking (reset at start of each UTC day)
        self._daily_start_bankroll: float = 0.0
        self._daily_pnl: float = 0.0
        self._daily_pnl_5m: float = 0.0  # Separate 5m loss tracking
        self._last_trading_day: Optional[object] = None

        # Timing features for BTC lead-lag altcoin edge
        self.timing_engine = TimingFeatureEngine()

        # Live executor — set by renaissance_trading_bot after init
        self.live_executor = None
        self._pending_live_bet: Optional[Dict] = None

        self._ensure_tables()
        self._load_bankroll()
        self._reset_daily_if_needed()

    # -- Database Setup --

    def _ensure_tables(self) -> None:
        conn = sqlite3.connect(self.db_path)

        # Migrate old polymarket_bets (activity log with 'action' column) -> legacy
        cols = []
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(polymarket_bets)").fetchall()]
        except Exception as e:
            self.logger.warning(f"Failed to read polymarket_bets schema: {e}")

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
                window_start_price REAL,
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

            CREATE TABLE IF NOT EXISTS polymarket_lifecycle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL,
                question TEXT,
                asset TEXT NOT NULL,
                instrument_key TEXT,
                market_open_time TEXT,
                market_close_time TEXT,
                duration_minutes INTEGER DEFAULT 15,

                t0_timestamp TEXT,
                t0_asset_price REAL,
                t0_crowd_up REAL,
                t0_crowd_down REAL,
                t0_ml_prediction REAL,
                t0_ml_confidence REAL,
                t0_ml_agreement REAL,
                t0_ml_direction TEXT,
                t0_regime TEXT,

                t5_timestamp TEXT,
                t5_asset_price REAL,
                t5_crowd_up REAL,
                t5_crowd_down REAL,
                t5_ml_prediction REAL,
                t5_ml_confidence REAL,
                t5_ml_agreement REAL,
                t5_ml_direction TEXT,
                t5_regime TEXT,
                t5_price_change_from_t0 REAL,
                t5_crowd_change_from_t0 REAL,
                t5_price_direction TEXT,
                t5_ml_agrees_with_t0 INTEGER,

                t10_timestamp TEXT,
                t10_asset_price REAL,
                t10_crowd_up REAL,
                t10_crowd_down REAL,
                t10_ml_prediction REAL,
                t10_ml_confidence REAL,
                t10_ml_agreement REAL,
                t10_ml_direction TEXT,
                t10_regime TEXT,
                t10_price_change_from_t0 REAL,
                t10_price_change_from_t5 REAL,
                t10_crowd_change_from_t0 REAL,
                t10_price_direction TEXT,
                t10_ml_agrees_with_t0 INTEGER,
                t10_ml_agrees_with_t5 INTEGER,

                conviction_score REAL,
                conviction_label TEXT,
                conviction_detail TEXT,
                direction_readings TEXT,

                decision TEXT,
                decision_reason TEXT,
                bet_amount REAL,
                entry_price REAL,
                edge_at_entry REAL,
                position_id TEXT,

                t15_timestamp TEXT,
                t15_asset_price REAL,
                t15_final_result TEXT,
                t15_price_change_from_t0 REAL,
                t15_price_change_from_t10 REAL,
                t15_won INTEGER,
                t15_pnl REAL,

                crowd_lag_t0_to_t5 REAL,
                crowd_lag_t5_to_t10 REAL,
                crowd_total_lag REAL,
                final_5min_reversed INTEGER,
                ml_accuracy INTEGER,

                UNIQUE(slug, market_open_time)
            );
            CREATE INDEX IF NOT EXISTS idx_lifecycle_slug ON polymarket_lifecycle(slug);
            CREATE INDEX IF NOT EXISTS idx_lifecycle_asset ON polymarket_lifecycle(asset);
            CREATE INDEX IF NOT EXISTS idx_lifecycle_decision ON polymarket_lifecycle(decision);
        """)
        conn.commit()

        # Migrate: add 'strategy' column if missing on polymarket_positions
        pos_cols = [r[1] for r in conn.execute("PRAGMA table_info(polymarket_positions)").fetchall()]
        if "strategy" not in pos_cols:
            conn.execute("ALTER TABLE polymarket_positions ADD COLUMN strategy TEXT DEFAULT 'legacy'")
            conn.commit()

        # Add window_start_price column if not exists (migration for existing DBs)
        bet_cols = [r[1] for r in conn.execute("PRAGMA table_info(polymarket_bets)").fetchall()]
        if "window_start_price" not in bet_cols:
            try:
                conn.execute("ALTER TABLE polymarket_bets ADD COLUMN window_start_price REAL")
                conn.commit()
                self.logger.info("Added window_start_price column to polymarket_bets")
            except Exception as e:
                self.logger.warning(f"Failed to add window_start_price column (may already exist): {e}")

        # Add timeframe column if not exists (migration for 5m markets)
        if "timeframe" not in bet_cols:
            try:
                conn.execute("ALTER TABLE polymarket_bets ADD COLUMN timeframe INTEGER DEFAULT 15")
                conn.commit()
                self.logger.info("Added timeframe column to polymarket_bets")
            except Exception as e:
                self.logger.warning(f"Failed to add timeframe column to polymarket_bets (may already exist): {e}")

        skip_cols = [r[1] for r in conn.execute("PRAGMA table_info(polymarket_skip_log)").fetchall()]
        if "timeframe" not in skip_cols:
            try:
                conn.execute("ALTER TABLE polymarket_skip_log ADD COLUMN timeframe INTEGER DEFAULT 15")
                conn.commit()
                self.logger.info("Added timeframe column to polymarket_skip_log")
            except Exception as e:
                self.logger.warning(f"Failed to add timeframe column to polymarket_skip_log (may already exist): {e}")

        # Prune skip log entries older than 7 days
        conn.execute("DELETE FROM polymarket_skip_log WHERE timestamp < datetime('now', '-7 days')")
        conn.commit()

        conn.close()

    def _load_bankroll(self) -> None:
        conn = sqlite3.connect(self.db_path)
        # Reconcile bankroll from P&L data (self-healing after resolution bugs).
        # Correct = initial + sum(resolved PnL) - sum(open exposure).
        try:
            pnl_row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM polymarket_bets "
                "WHERE status IN ('WON', 'LOST', 'CLOSED')"
            ).fetchone()
            open_row = conn.execute(
                "SELECT COALESCE(SUM(total_invested), 0) FROM polymarket_bets "
                "WHERE status = 'OPEN'"
            ).fetchone()
            total_pnl = pnl_row[0] if pnl_row else 0
            open_exposure = open_row[0] if open_row else 0
            raw_bankroll = self.initial_bankroll + total_pnl - open_exposure
            # Cap at MAX_SIZING_BANKROLL to prevent runaway compounding
            self.bankroll = min(raw_bankroll, self.MAX_SIZING_BANKROLL)
            logger.info(f"BANKROLL RECONCILED: ${self.bankroll:.2f} "
                        f"(raw=${raw_bankroll:.2f} = initial=${self.initial_bankroll:.2f} "
                        f"+ pnl=${total_pnl:.2f} - open=${open_exposure:.2f}, "
                        f"capped at ${self.MAX_SIZING_BANKROLL:.0f})")
        except Exception as e:
            # Fallback to bankroll_log if polymarket_bets doesn't exist yet
            row = conn.execute(
                "SELECT bankroll FROM polymarket_bankroll_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                self.bankroll = row[0]
            logger.warning(f"Bankroll fallback to log: ${self.bankroll:.2f} ({e})")
        conn.close()

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
                  ml_direction: str = "", minutes_left: float = 0,
                  timeframe: int = 15) -> None:
        """Write to polymarket_skip_log table."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO polymarket_skip_log "
            "(asset, slug, reason, ml_confidence, token_cost, ml_direction, minutes_left, timeframe) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (asset, slug, reason, ml_confidence, token_cost, ml_direction, minutes_left, timeframe),
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

    def _reset_daily_if_needed(self) -> None:
        """Reset daily loss tracking at start of each UTC day."""
        from datetime import timezone as _tz
        today = datetime.now(_tz.utc).date()
        if self._last_trading_day != today:
            self._last_trading_day = today
            self._daily_start_bankroll = self.bankroll
            self._daily_pnl = 0.0
            self._daily_pnl_5m = 0.0
            self.logger.info(
                f"POLYMARKET DAILY RESET: bankroll=${self.bankroll:.2f}, "
                f"loss_limit=${self.bankroll * self.DAILY_LOSS_LIMIT_PCT:.2f}, "
                f"5m_loss_limit=${self.bankroll * self.DAILY_LOSS_LIMIT_5M_PCT:.2f}"
            )

    def _check_daily_loss_limit(self, timeframe: int = 15) -> bool:
        """Return True if within daily loss limit (can bet).

        Checks both the global daily limit and, for 5m markets,
        the separate 5m-specific daily loss cap.
        """
        self._reset_daily_if_needed()
        if self._daily_start_bankroll <= 0:
            return True
        # Global check
        limit = self._daily_start_bankroll * self.DAILY_LOSS_LIMIT_PCT
        if self._daily_pnl <= -limit:
            return False
        # 5m-specific check
        if timeframe == 5:
            limit_5m = self._daily_start_bankroll * self.DAILY_LOSS_LIMIT_5M_PCT
            if self._daily_pnl_5m <= -limit_5m:
                return False
        return True

    def _check_min_bankroll(self) -> bool:
        """Return True if bankroll is above minimum to trade."""
        return self.bankroll >= self.MIN_BANKROLL_TO_TRADE

    def _check_asset_concentration(self, asset: str, bet_amount: float) -> bool:
        """Ensure no single asset exceeds MAX_ASSET_CONCENTRATION of total open exposure."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT asset, COALESCE(SUM(total_invested), 0) as exposure "
                "FROM polymarket_bets WHERE status = 'OPEN' GROUP BY asset"
            ).fetchall()

            n_assets = len(rows)

            # With fewer than 3 distinct assets open, concentration check
            # is counterproductive — it blocks diversification by preventing
            # new assets from entering. Only enforce once portfolio is built up.
            if n_assets < 3:
                return True

            existing_exposure = sum(r[1] for r in rows)
            total_exposure = existing_exposure + bet_amount
            asset_exposure = sum(r[1] for r in rows if r[0] == asset) + bet_amount

            if total_exposure > 0 and (asset_exposure / total_exposure) > self.MAX_ASSET_CONCENTRATION:
                self.logger.warning(
                    f"POLYMARKET CONCENTRATION: {asset} would be "
                    f"{asset_exposure/total_exposure*100:.0f}% of exposure "
                    f"(limit {self.MAX_ASSET_CONCENTRATION*100:.0f}%)"
                )
                return False
            return True
        finally:
            conn.close()

    # -- Main Cycle --

    async def execute_cycle(
        self,
        ml_predictions: dict,
        current_prices: dict,
        current_regime: str = "unknown",
        cross_data: Optional[dict] = None,
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

        # 1b. Log per-timeframe stats
        self._log_timeframe_stats()

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
            if minutes_left is None:
                continue
            # Timeframe-aware window: 15m -> [1.0, 14.0], 5m -> [0.5, 4.5]
            if inst.timeframe == 5:
                if minutes_left < 0.5 or minutes_left > 4.5:
                    continue
            else:
                if minutes_left < 1.0 or minutes_left > 14.0:
                    continue

            # Per-horizon routing: 5-min markets use 1bar, 15-min markets use 2bar
            ml_data = ml_predictions.get(inst.ml_pair, {})
            if inst.timeframe == 5 and "prediction_1bar" in ml_data:
                ml_conf = ml_data.get("confidence_1bar", ml_data.get("confidence", 50.0))
                ml_pred = ml_data.get("prediction_1bar", 0)
            else:
                ml_conf = ml_data.get("confidence", 50.0)
                ml_pred = ml_data.get("prediction", 0)
            ml_direction = "UP" if ml_pred > 0 else "DOWN"
            entry_side = "YES" if ml_direction == "UP" else "NO"

            # Token cost (crowd pricing)
            crowd_up = self._parse_crowd_up(market)
            token_cost = crowd_up if ml_direction == "UP" else (1.0 - crowd_up)

            # Gate: odds filter — entry-price-first gating
            # 5m data: cheap tokens (<=0.48) are a TRAP (36% wr, -$152).
            #          Edge lives in 0.48-0.52 band (59% wr, +$505).
            # 15m data: cheap tokens work (55% wr), wider range is OK.
            if inst.timeframe == 5:
                if token_cost < 0.47 or token_cost > 0.53:
                    self._log_skip(inst.asset, slug,
                                   f"5m_odds token_cost={token_cost:.3f} outside [0.47, 0.53]",
                                   ml_conf, token_cost, ml_direction, minutes_left,
                                   timeframe=inst.timeframe)
                    continue
            else:
                if token_cost < 0.15 or token_cost > 0.50:
                    self._log_skip(inst.asset, slug,
                                   f"odds_filter token_cost={token_cost:.3f} outside [0.15, 0.50]",
                                   ml_conf, token_cost, ml_direction, minutes_left,
                                   timeframe=inst.timeframe)
                    continue

            # Gate: confidence floor
            if ml_conf < self.CONFIDENCE_THRESHOLD:
                self._log_skip(inst.asset, slug,
                               f"conf {ml_conf:.0f}% < {self.CONFIDENCE_THRESHOLD}%",
                               ml_conf, token_cost, ml_direction, minutes_left,
                               timeframe=inst.timeframe)
                continue

            # Gate: confidence cap — optimizer found >52.5% is destructive
            if ml_conf > self.CONFIDENCE_CAP:
                self._log_skip(inst.asset, slug,
                               f"conf {ml_conf:.1f}% > cap {self.CONFIDENCE_CAP}%",
                               ml_conf, token_cost, ml_direction, minutes_left,
                               timeframe=inst.timeframe)
                continue

            # Gate: altcoin-only filter for 5m markets
            # Calibration shows BTC (49.3%) and ETH (49.4%) have negative edge on 5m,
            # while altcoins SOL (51.7%), XRP (51.8%), DOGE (52.4%) have positive edge.
            if inst.timeframe == 5 and inst.asset in ("BTC", "ETH"):
                self._log_skip(inst.asset, slug,
                               f"5m_altcoin_only (BTC/ETH neg edge on 5m)",
                               ml_conf, token_cost, ml_direction, minutes_left,
                               timeframe=inst.timeframe)
                continue

            # Gate: lead asset agreement (5m altcoins must agree with BTC direction)
            if inst.lead_asset:
                lead_data = ml_predictions.get(f"{inst.lead_asset}-USD", {})
                if "prediction_1bar" in lead_data:
                    lead_pred = lead_data.get("prediction_1bar", 0)
                else:
                    lead_pred = lead_data.get("prediction", 0)
                lead_direction = "UP" if lead_pred > 0 else "DOWN"
                if lead_direction != ml_direction:
                    self._log_skip(inst.asset, slug,
                                   f"lead_{inst.lead_asset}_disagrees",
                                   ml_conf, token_cost, ml_direction, minutes_left,
                                   timeframe=inst.timeframe)
                    continue

            # Timing features: BTC lead-lag boost for altcoin 5m bets
            timing_features = None
            if inst.timeframe == 5 and self.timing_engine.is_follower(inst.asset) and cross_data:
                timing_features = self.timing_engine.compute(
                    asset=inst.asset, cross_data=cross_data,
                    current_prices=current_prices,
                )
                if timing_features.get("has_data"):
                    boost = self.timing_engine.get_direction_boost(timing_features, ml_direction)
                    original_conf = ml_conf
                    ml_conf *= boost
                    if boost != 1.0:
                        self.logger.info(
                            f"[TIMING] {inst.asset} 5m: boost={boost:.3f} "
                            f"conf={original_conf:.1f}→{ml_conf:.1f}% "
                            f"lead_mom={timing_features['lead_momentum']:.3f}"
                        )

            # Gate: rate limit
            if not self._check_rate_limit():
                self._log_skip(inst.asset, slug, "rate_limit",
                               ml_conf, token_cost, ml_direction, minutes_left,
                               timeframe=inst.timeframe)
                continue

            # Gate: cooldown
            if not self._check_cooldown():
                remaining = self.COOLDOWN_AFTER_LOSS - (time.time() - self._last_loss_time)
                self._log_skip(inst.asset, slug,
                               f"cooldown {remaining:.0f}s remaining",
                               ml_conf, token_cost, ml_direction, minutes_left,
                               timeframe=inst.timeframe)
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

            # Gate: max concurrent open bets
            conn = sqlite3.connect(self.db_path)
            open_count = conn.execute(
                "SELECT COUNT(*) FROM polymarket_bets WHERE status = 'OPEN'"
            ).fetchone()[0]
            conn.close()
            if open_count >= self.MAX_OPEN_BETS:
                self._log_skip(inst.asset, slug,
                               f"max_open_bets ({open_count}/{self.MAX_OPEN_BETS})",
                               ml_conf, token_cost, ml_direction, minutes_left,
                               timeframe=inst.timeframe)
                continue

            # Gate: max exposure check
            conn = sqlite3.connect(self.db_path)
            total_open = conn.execute(
                "SELECT COALESCE(SUM(total_invested), 0) FROM polymarket_bets WHERE status = 'OPEN'"
            ).fetchone()[0]
            conn.close()
            max_exposure = min(self.bankroll, self.MAX_SIZING_BANKROLL) * 0.8
            if total_open + self.MIN_BET > max_exposure:
                self._log_skip(inst.asset, slug, f"max_exposure (open=${total_open:.0f} > ${max_exposure:.0f})",
                               ml_conf, token_cost, ml_direction, minutes_left,
                               timeframe=inst.timeframe)
                continue

            # Gate: minimum bankroll
            if not self._check_min_bankroll():
                self._log_skip(inst.asset, slug,
                               f"bankroll ${self.bankroll:.2f} < ${self.MIN_BANKROLL_TO_TRADE}",
                               ml_conf, token_cost, ml_direction, minutes_left,
                               timeframe=inst.timeframe)
                continue

            # Gate: daily loss limit (5m has separate cap)
            if not self._check_daily_loss_limit(timeframe=inst.timeframe):
                pnl_detail = f"pnl=${self._daily_pnl:.2f}"
                if inst.timeframe == 5:
                    pnl_detail += f", 5m_pnl=${self._daily_pnl_5m:.2f}"
                self._log_skip(inst.asset, slug,
                               f"daily loss limit hit ({pnl_detail})",
                               ml_conf, token_cost, ml_direction, minutes_left,
                               timeframe=inst.timeframe)
                continue

            # Gate: asset concentration
            est_bet = max(self.MIN_BET, self.bankroll * self.MAX_BET_PCT)
            if not self._check_asset_concentration(inst.asset, est_bet):
                self._log_skip(inst.asset, slug, "asset_concentration",
                               ml_conf, token_cost, ml_direction, minutes_left,
                               timeframe=inst.timeframe)
                continue

            # Get asset price for recording
            asset_price = current_prices.get(inst.price_pair, 0)

            # Place bet (pass timing features for audit trail)
            self._place_bet(inst, market, ml_direction, entry_side, token_cost,
                            ml_conf, current_regime, asset_price,
                            timing_features=timing_features)

            # Execute live bet if queued by _place_bet
            if self._pending_live_bet and self.live_executor:
                try:
                    lb = self._pending_live_bet
                    should_live, reason = self.live_executor.should_go_live(
                        lb["asset"], lb["bet_amount"],
                    )
                    if should_live:
                        live_result = await self.live_executor.place_live_bet(**lb)
                        if live_result and live_result.get("fill_status") != "error":
                            self.logger.info(f"[LIVE] Bet placed alongside paper: {live_result}")
                        else:
                            self.logger.warning(f"[LIVE] Bet failed: {live_result}")
                    else:
                        self.logger.info(f"[LIVE] Skipped ({reason})")
                except Exception as _live_err:
                    self.logger.warning(f"[LIVE] Live bet error: {_live_err}")
                finally:
                    self._pending_live_bet = None

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

            # Rule 0: STOP LOSS — disabled (direction_flip handles exits;
            # binary options resolve in 15min so stop-loss just crystallizes
            # losses on bets that win 50% of the time if held).
            # Was: STOP_LOSS_PCT = 0.40, but avg actual drop was -60% (20pp
            # overshoot due to 5-min check interval). Cost $45.68 on known bets.

            # Current ML data (per-horizon routing for position management)
            ml_data = ml_predictions.get(inst.ml_pair, {})
            if inst.timeframe == 5 and "prediction_1bar" in ml_data:
                ml_conf = ml_data.get("confidence_1bar", ml_data.get("confidence", 50.0))
                ml_pred = ml_data.get("prediction_1bar", 0)
            else:
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
                    if self._check_rate_limit():
                        self._add_to_bet(bet, token_cost, ml_conf, inst)

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
        self._daily_pnl += pnl  # Track daily P&L for loss limit
        # Track 5m PnL separately for the 5m-specific daily loss cap
        bet_timeframe = bet["timeframe"] if "timeframe" in bet.keys() else 15
        if bet_timeframe == 5:
            self._daily_pnl_5m += pnl
        self._log_bankroll(f"closed_{reason}", str(bet["id"]), pnl)

        if pnl < 0:
            self._last_loss_time = time.time()

        self.logger.info(
            f"CLOSED [{bet['asset']}]: {reason} | P&L: ${pnl:+.2f} ({return_pct:+.1f}%) | "
            f"Bankroll: ${self.bankroll:.2f}"
        )

        # Update lifecycle table for active close
        exit_asset_price = 0
        if current_prices:
            inst = self._find_instrument(bet["asset"])
            if inst:
                exit_asset_price = current_prices.get(inst.price_pair, 0)
        ml_direction = "UP" if bet["entry_side"] == "YES" else "DOWN"
        # For active sells, determine result using window_start_price (asset price
        # at t=0 when the prediction window opened), NOT entry_asset_price (price
        # when bet was placed mid-window). Using entry_asset_price was the root cause
        # of phantom P&L — price_fallback showed 76.5% win rate vs gamma_api 8.3%.
        final_result = "CLOSED"
        ref_price = bet["window_start_price"] if bet["window_start_price"] else bet["entry_asset_price"]
        if exit_asset_price > 0 and ref_price and ref_price > 0:
            final_result = "UP" if exit_asset_price > ref_price else "DOWN"
        self._update_lifecycle_resolution(
            slug=bet["slug"],
            asset=bet["asset"],
            asset_price=exit_asset_price,
            final_result=final_result,
            won=pnl > 0,
            pnl=pnl,
            ml_direction=ml_direction,
            source=f"active_close_{reason}",
        )

    # -- Kelly Sizing --

    def _compute_kelly_bet(self, probability: float, token_cost: float,
                           kelly_fraction: float = 0.5,
                           max_bet_usd: float = 0.0) -> float:
        """Kelly-fraction optimal bet size for Polymarket.

        Args:
            probability: Model's P(correct outcome) — e.g. 0.55 means 55% chance we're right.
            token_cost: Price of the token we're buying (0.0 to 1.0).
            kelly_fraction: Fraction of full Kelly to use (0.5 = half-Kelly, 0.4 = 40% Kelly).
            max_bet_usd: Hard dollar ceiling per bet (0 = use global MAX_BET_PCT).

        Returns:
            Dollar amount to bet (floored at MIN_BET, capped at ceiling).
        """
        # Payout ratio: if we buy at token_cost and win, we get $1 per token
        b = (1.0 - token_cost) / (token_cost + 1e-10)

        # Win probability from model (already directional — prob of the side we're betting)
        p = max(0.5, min(0.99, probability))
        q = 1.0 - p

        # Full Kelly fraction
        kelly = (p * b - q) / (b + 1e-10)
        kelly = max(0.0, kelly)
        if kelly <= 0:
            return 0.0

        # Fractional Kelly for safety
        frac_kelly = kelly * kelly_fraction

        # Use capped bankroll for sizing to prevent runaway compounding
        sizing_bankroll = min(self.bankroll, self.MAX_SIZING_BANKROLL)

        # Dollar bet
        bet = sizing_bankroll * frac_kelly

        # Floor and ceiling
        ceiling = min(sizing_bankroll * self.MAX_BET_PCT, self.MAX_BET_USD)
        if max_bet_usd > 0:
            ceiling = min(ceiling, max_bet_usd)
        bet = max(self.MIN_BET, min(bet, ceiling))
        return round(bet, 2)

    # -- Bet Placement --

    def _place_bet(self, inst: InstrumentConfig, market: dict,
                   direction: str, entry_side: str, token_cost: float,
                   ml_confidence: float, regime: str, asset_price: float,
                   timing_features: Optional[Dict] = None) -> None:
        """Place a Kelly-sized bet."""
        # Convert confidence (50-100% scale) to probability (0.5-1.0)
        prob = ml_confidence / 100.0
        bet_amount = self._compute_kelly_bet(
            prob, token_cost,
            kelly_fraction=inst.kelly_fraction,
            max_bet_usd=inst.max_bet_usd,
        )
        if bet_amount <= 0:
            self.logger.debug(f"SKIP [{inst.asset}]: Kelly returned $0 (no edge at {token_cost:.3f})")
            return

        # 5m entry-price sizing: override Kelly with edge-based flat sizing
        # Data: 0.48-0.52 band has 59% wr (+$505), cheap tokens lose (-$152)
        if inst.timeframe == 5:
            if token_cost > 0.53:
                self.logger.debug(f"SKIP [{inst.asset}] 5m: token_cost={token_cost:.3f} > 0.53, no edge")
                return
            elif token_cost > 0.51:
                bet_amount = self.MIN_BET  # $5 — thin edge, minimum size
            else:
                bet_amount = min(12.0, max(8.0, bet_amount))  # $8-12 — good entry
            self.logger.info(
                f"5m SIZING [{inst.asset}]: entry={token_cost:.3f} → ${bet_amount:.2f}"
            )
        tokens = bet_amount / token_cost if token_cost > 0 else 0
        slug = market.get("slug", "")

        # Get window start price for correct resolution later
        window_start_price = self._get_window_start_price(slug, inst) if slug else 0.0
        if window_start_price <= 0:
            window_start_price = asset_price  # fallback to current price

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO polymarket_bets
            (slug, asset, entry_side, entry_token_cost, entry_amount, entry_tokens,
             entry_confidence, adds, total_invested, total_tokens, avg_cost,
             status, regime, entry_asset_price, window_start_price, question, timeframe)
            VALUES (?, ?, ?, ?, ?, ?, ?, '[]', ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?)
        """, (
            slug, inst.asset, entry_side, token_cost, bet_amount, tokens,
            ml_confidence, bet_amount, tokens, token_cost,
            regime, asset_price, window_start_price, market.get("question", ""),
            inst.timeframe,
        ))
        bet_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        conn.close()

        self.bankroll -= bet_amount
        self._bets_this_hour.append(time.time())
        self._log_bankroll("bet_placed", str(bet_id), -bet_amount)

        # Create lifecycle entry with t0 data
        crowd_up = self._parse_crowd_up(market)
        self._save_lifecycle_entry(
            slug=slug,
            question=market.get("question", ""),
            asset=inst.asset,
            instrument_key=self._find_instrument_key(inst.asset) or "",
            direction=direction,
            ml_confidence=ml_confidence,
            ml_prediction=prob,
            token_cost=token_cost,
            asset_price=asset_price,
            regime=regime,
            bet_amount=bet_amount,
            crowd_up=crowd_up,
            bet_id=str(bet_id),
            deadline=market.get("deadline", ""),
            timeframe=inst.timeframe,
        )

        timing_str = ""
        if timing_features and timing_features.get("has_data"):
            timing_str = (
                f" | lead_mom={timing_features['lead_momentum']:.3f}"
                f" btc_1b={timing_features['btc_1bar_ret']:.4f}"
            )
        self.logger.info(
            f"BET [{inst.asset}]: {direction} ({entry_side}) | "
            f"Conf: {ml_confidence:.1f}% | Token: ${token_cost:.2f} | "
            f"Kelly: ${bet_amount:.2f} | Bankroll: ${self.bankroll:.2f}{timing_str}"
        )

        # Queue live execution if live executor is available
        if self.live_executor:
            token_id = (market.get("token_id_yes") if direction == "UP"
                        else market.get("token_id_no"))
            if token_id:
                # Compute window_start from slug
                ws = 0
                try:
                    ws = int(slug.rsplit("-", 1)[-1])
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Failed to parse window_start from slug '{slug}': {e}")
                self._pending_live_bet = {
                    "asset": inst.asset,
                    "direction": direction,
                    "entry_price": token_cost,
                    "bet_amount": min(bet_amount, 20.0),  # Match Kelly max
                    "token_id": token_id,
                    "slug": slug,
                    "question": market.get("question", ""),
                    "window_start": ws,
                    "timeframe": str(inst.timeframe) + "m",
                    "edge": (prob - token_cost) if prob > 0.5 else (token_cost - (1 - prob)),
                    "confidence": prob,
                    "crowd_price": crowd_up,
                }

    def _add_to_bet(self, bet, token_cost: float, ml_confidence: float,
                    inst: Optional[InstrumentConfig] = None) -> None:
        """Add a Kelly-sized increment to an existing open bet."""
        prob = ml_confidence / 100.0
        kf = inst.kelly_fraction if inst else 0.5
        mbu = inst.max_bet_usd if inst else 0.0
        add_amount = self._compute_kelly_bet(prob, token_cost, kelly_fraction=kf, max_bet_usd=mbu)
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

    # -- Lifecycle Tracking --

    def _save_lifecycle_entry(self, slug: str, question: str, asset: str,
                              instrument_key: str, direction: str,
                              ml_confidence: float, ml_prediction: float,
                              token_cost: float, asset_price: float,
                              regime: str, bet_amount: float, crowd_up: float,
                              bet_id: str, deadline: str,
                              timeframe: int = 15) -> None:
        """Create a lifecycle row with t0 data when a bet is placed."""
        # Extract market_open_time from slug timestamp (e.g. btc-updown-15m-1709312400)
        market_open_time = None
        try:
            slug_parts = slug.rsplit("-", 1)
            if len(slug_parts) == 2 and slug_parts[1].isdigit():
                ts = int(slug_parts[1])
                market_open_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except (ValueError, OSError) as e:
            self.logger.warning(f"Failed to parse market_open_time from slug '{slug}': {e}")

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO polymarket_lifecycle (
                    slug, question, asset, instrument_key,
                    market_open_time, market_close_time, duration_minutes,
                    t0_timestamp, t0_asset_price, t0_crowd_up, t0_crowd_down,
                    t0_ml_prediction, t0_ml_confidence, t0_ml_direction,
                    t0_regime,
                    decision, decision_reason,
                    bet_amount, entry_price, position_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                slug, question, asset, instrument_key,
                market_open_time, deadline, timeframe,
                asset_price, crowd_up, round(1.0 - crowd_up, 4),
                ml_prediction, ml_confidence, direction,
                regime,
                "BET", f"conf={ml_confidence:.1f}%,token=${token_cost:.2f}",
                bet_amount, token_cost, bet_id,
            ))
            conn.commit()
            self.logger.info(
                f"LIFECYCLE T0 [{asset}]: slug={slug}, bet_id={bet_id}, "
                f"price=${asset_price:.2f}, crowd_up={crowd_up:.3f}, "
                f"ml_conf={ml_confidence:.1f}%"
            )
        except Exception as e:
            self.logger.warning(f"LIFECYCLE T0 save failed for {slug}: {e}")
        finally:
            conn.close()

    def _update_lifecycle_resolution(self, slug: str, asset: str,
                                     asset_price: float, final_result: str,
                                     won: bool, pnl: float,
                                     ml_direction: str,
                                     source: str) -> None:
        """Update lifecycle row with t15 resolution data.

        Args:
            slug: Market slug to match on.
            asset: Asset symbol (BTC, ETH, etc.).
            asset_price: Asset price at resolution time.
            final_result: "UP" or "DOWN" -- actual outcome.
            won: Whether our bet won.
            pnl: Dollar P&L.
            ml_direction: Our ML direction at entry ("UP" or "DOWN").
            source: Resolution source (gamma_api, price_fallback, etc.).
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Compute price change from t0
            t0_row = conn.execute(
                "SELECT t0_asset_price, t10_asset_price, t10_price_direction, t0_ml_direction "
                "FROM polymarket_lifecycle WHERE slug = ?",
                (slug,)
            ).fetchone()

            t15_price_change_from_t0 = None
            t15_price_change_from_t10 = None
            final_5min_reversed = None
            ml_accuracy = None

            if t0_row:
                t0_price = t0_row[0]
                t10_price = t0_row[1]
                t10_price_direction = t0_row[2]
                t0_ml_direction = t0_row[3] or ml_direction

                if t0_price and t0_price > 0 and asset_price > 0:
                    t15_price_change_from_t0 = round(
                        ((asset_price - t0_price) / t0_price) * 100, 4
                    )
                if t10_price and t10_price > 0 and asset_price > 0:
                    t15_price_change_from_t10 = round(
                        ((asset_price - t10_price) / t10_price) * 100, 4
                    )
                if t10_price_direction and final_result:
                    final_5min_reversed = 1 if final_result != t10_price_direction else 0
                if t0_ml_direction and final_result:
                    ml_accuracy = 1 if t0_ml_direction == final_result else 0

            rows_updated = conn.execute("""
                UPDATE polymarket_lifecycle
                SET t15_timestamp = datetime('now'),
                    t15_asset_price = ?,
                    t15_final_result = ?,
                    t15_price_change_from_t0 = ?,
                    t15_price_change_from_t10 = ?,
                    t15_won = ?,
                    t15_pnl = ?,
                    final_5min_reversed = COALESCE(?, final_5min_reversed),
                    ml_accuracy = COALESCE(?, ml_accuracy)
                WHERE slug = ?
            """, (
                asset_price,
                final_result,
                t15_price_change_from_t0,
                t15_price_change_from_t10,
                1 if won else 0,
                pnl,
                final_5min_reversed,
                ml_accuracy,
                slug,
            )).rowcount
            conn.commit()

            if rows_updated > 0:
                self.logger.info(
                    f"LIFECYCLE T15 [{asset}]: slug={slug}, result={final_result}, "
                    f"won={won}, pnl=${pnl:+.2f}, ml_acc={ml_accuracy}, "
                    f"price_chg={t15_price_change_from_t0}, via={source}"
                )
            else:
                self.logger.warning(
                    f"LIFECYCLE T15 no matching row for slug={slug} "
                    f"(result={final_result}, won={won}, pnl=${pnl:+.2f}). "
                    f"Creating retroactive lifecycle entry."
                )
                # Create a minimal lifecycle entry retroactively if none exists
                conn.execute("""
                    INSERT OR IGNORE INTO polymarket_lifecycle (
                        slug, asset, t15_timestamp, t15_asset_price,
                        t15_final_result, t15_won, t15_pnl, decision
                    ) VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?)
                """, (
                    slug, asset, asset_price,
                    final_result, 1 if won else 0, pnl,
                    f"retroactive_{source}",
                ))
                conn.commit()
                self.logger.info(
                    f"LIFECYCLE T15 [{asset}]: retroactive entry created for slug={slug}"
                )
        except Exception as e:
            self.logger.warning(
                f"LIFECYCLE T15 update failed for {slug}: {e}"
            )
        finally:
            conn.close()

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

                        # Only resolve if Gamma API explicitly marks as resolved.
                        # DO NOT use is_definitive (price >= 0.95) — these are LIVE
                        # trading prices mid-window, not final outcomes. Resolving
                        # on live prices gave 6% win rate vs 71% on price_fallback.
                        if is_resolved:
                            self._resolve_bet(conn, bet, yes_price, no_price, "gamma_api", current_prices)
                            continue

                # Deadline-based resolution
                deadline_str = market.get("deadline", "") if market else ""
                seconds_past = 0
                if deadline_str:
                    try:
                        deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
                        seconds_past = (datetime.now(timezone.utc) - deadline).total_seconds()
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Failed to parse deadline '{deadline_str}' for bet {bet['id']}: {e}")

                # Price-based fallback: PERMANENTLY DISABLED
                # Resolving via crypto price comparison is unreliable and was the
                # source of the historic $48M phantom bankroll.  Bets stay OPEN
                # until gamma_api resolves them or force-expire after 30 min.
                # (see STRATEGY_MATH_TRUTH_DOCUMENT.md §3.3)

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

                    # Update lifecycle for force-expired bets
                    exit_asset_price = 0
                    if current_prices:
                        inst = self._find_instrument(bet["asset"])
                        if inst:
                            exit_asset_price = current_prices.get(inst.price_pair, 0)
                    ml_direction = "UP" if bet["entry_side"] == "YES" else "DOWN"
                    self._update_lifecycle_resolution(
                        slug=slug,
                        asset=bet["asset"],
                        asset_price=exit_asset_price,
                        final_result="EXPIRED",
                        won=False,
                        pnl=0,
                        ml_direction=ml_direction,
                        source="force_expired",
                    )

            except Exception as e:
                self.logger.warning(f"Resolution check failed for bet {bet['id']}: {e}")

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

        if won and source == "gamma_api":
            self.bankroll += bet["total_invested"] + pnl
        elif won:
            # Non-gamma wins (e.g. price_fallback) — refund investment only,
            # do NOT credit profit (unverified resolution source)
            self.bankroll += bet["total_invested"]
            self.logger.warning(
                f"WON [{bet['asset']}] via {source} — bankroll refunded "
                f"${bet['total_invested']:.2f} but profit ${pnl:.2f} NOT credited"
            )
        if not won:
            self._last_loss_time = time.time()

        self._daily_pnl += pnl  # Track daily P&L for loss limit
        # Track 5m PnL separately for the 5m-specific daily loss cap
        bet_timeframe = bet["timeframe"] if "timeframe" in bet.keys() else 15
        if bet_timeframe == 5:
            self._daily_pnl_5m += pnl

        self.logger.info(
            f"{'WON' if won else 'LOST'} [{bet['asset']}]: "
            f"{side} | P&L: ${pnl:+.2f} ({return_pct:+.1f}%) | "
            f"Bankroll: ${self.bankroll:.2f} | via {source}"
        )
        self._log_bankroll(f"resolved_{status.lower()}", str(bet["id"]), pnl)

        # Update lifecycle table with t15 resolution data
        final_result = "UP" if yes_price >= 0.95 else "DOWN"
        ml_direction = "UP" if side == "YES" else "DOWN"
        self._update_lifecycle_resolution(
            slug=bet["slug"],
            asset=bet["asset"],
            asset_price=exit_asset_price,
            final_result=final_result,
            won=won,
            pnl=pnl,
            ml_direction=ml_direction,
            source=source,
        )

    def _check_legacy_resolutions(self, current_prices: Optional[dict] = None) -> None:
        """Check for any still-open legacy positions in polymarket_positions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            open_positions = conn.execute(
                "SELECT * FROM polymarket_positions WHERE status = 'open' AND strategy = 'strategy_a'"
            ).fetchall()
        except Exception as e:
            self.logger.warning(f"Failed to query legacy polymarket_positions: {e}")
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
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Failed to parse legacy deadline '{deadline_str}' for position {pos['position_id']}: {e}")

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

                        # Only resolve on explicit resolved flag — not live prices
                        if is_resolved:
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

                            # Update lifecycle for legacy position (match on slug or position_id)
                            final_result = "UP" if yes_price >= 0.95 else "DOWN"
                            self._update_lifecycle_resolution(
                                slug=slug,
                                asset=pos["asset"],
                                asset_price=0,
                                final_result=final_result,
                                won=won,
                                pnl=pnl,
                                ml_direction=direction,
                                source="legacy_gamma_api",
                            )
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

                    # Update lifecycle for force-expired legacy position
                    ml_direction = pos["direction"] if pos["direction"] else "UNKNOWN"
                    self._update_lifecycle_resolution(
                        slug=slug,
                        asset=pos["asset"],
                        asset_price=0,
                        final_result="EXPIRED",
                        won=False,
                        pnl=0,
                        ml_direction=ml_direction,
                        source="legacy_force_expired",
                    )

            except Exception as e:
                self.logger.warning(f"Legacy resolution check failed for {pos['position_id']}: {e}")

        conn.commit()
        conn.close()

    # -- Window Start Price Lookup --

    def _get_window_start_price(self, slug: str, inst: InstrumentConfig) -> float:
        """Extract window-start price from slug timestamp by looking up the bar table.

        The slug format is e.g. 'btc-updown-15m-1709312400' where the last segment
        is a unix timestamp for the window open. We look up the closest 5-min bar
        close price at or just before that timestamp to get the reference price
        that the market resolves against.
        """
        try:
            slug_parts = slug.rsplit("-", 1)
            if len(slug_parts) != 2 or not slug_parts[1].isdigit():
                return 0.0
            window_ts = int(slug_parts[1])
            # Convert pair format: "BTC-USD" -> "BTCUSDT" for bar table lookup
            pair = inst.price_pair.replace("-", "").replace("USD", "USDT")
            # Look up the closest bar at or just before window start
            conn = sqlite3.connect(self.db_path)
            row = conn.execute("""
                SELECT close FROM five_minute_bars
                WHERE pair = ? AND bar_start <= ?
                ORDER BY bar_start DESC LIMIT 1
            """, (pair, float(window_ts))).fetchone()
            conn.close()
            if row:
                return float(row[0])
        except Exception as e:
            self.logger.warning(f"Failed to get window start price for {slug}: {e}")
        return 0.0

    # -- Market Discovery --

    def _discover_market(self, inst: InstrumentConfig) -> Optional[dict]:
        now_ts = int(time.time())
        window_sec = inst.timeframe * 60  # 900 for 15m, 300 for 5m
        window_ts = (now_ts // window_sec) * window_sec
        slug = inst.slug_pattern.format(ts=window_ts)

        cached = self._market_cache.get(slug)
        if cached and (now_ts - cached[1] < MARKET_CACHE_TTL):
            return cached[0]

        market = self._fetch_market_by_slug(slug)
        if market:
            self._market_cache[slug] = (market, now_ts)
            return market

        # Try previous window
        prev_slug = inst.slug_pattern.format(ts=window_ts - window_sec)
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

            # Extract CLOB token IDs for live order placement
            token_ids_raw = m.get("clobTokenIds", "[]")
            if isinstance(token_ids_raw, str):
                try:
                    token_ids_raw = json.loads(token_ids_raw)
                except (json.JSONDecodeError, TypeError):
                    token_ids_raw = []
            token_id_yes = token_ids_raw[0] if len(token_ids_raw) >= 1 else None
            token_id_no = token_ids_raw[1] if len(token_ids_raw) >= 2 else None

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
                "token_id_yes": token_id_yes,
                "token_id_no": token_id_no,
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
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse outcomePrices[0] as float: {prices[0]!r}: {e}")
        best_ask = raw.get("bestAsk")
        if best_ask:
            try:
                return float(best_ask)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse bestAsk as float: {best_ask!r}: {e}")
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

    # -- Per-Timeframe Stats --

    def _log_timeframe_stats(self) -> None:
        """Log separate 5m vs 15m performance stats (last 24 hours)."""
        conn = sqlite3.connect(self.db_path)
        try:
            for tf in [5, 15]:
                row = conn.execute("""
                    SELECT COUNT(*) as n,
                           COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) as wins,
                           COALESCE(SUM(pnl), 0) as total_pnl
                    FROM polymarket_bets
                    WHERE timeframe = ? AND status IN ('WON', 'LOST', 'CLOSED')
                    AND opened_at > datetime('now', '-24 hours')
                """, (tf,)).fetchone()
                n, wins, pnl = row
                if n > 0:
                    wr = (wins / n) * 100
                    self.logger.info(
                        f"POLYMARKET {tf}M 24H: {n} bets, {wr:.0f}% win, ${pnl:+.2f}"
                    )
        except Exception as e:
            self.logger.debug(f"Timeframe stats query failed: {e}")
        finally:
            conn.close()

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
        except Exception as e:
            self.logger.warning(f"Failed to query polymarket stats: {e}")
            return {"bankroll": self.bankroll, "open_positions": 0}
        finally:
            conn.close()
