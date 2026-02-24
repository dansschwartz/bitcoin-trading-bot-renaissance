"""
Polymarket Strategy A: Confirmed Momentum

The simplest, highest-frequency, lowest-risk Polymarket strategy.

How it works:
  1. A 15-minute direction market opens (e.g., "Will BTC go Up or Down?")
  2. We WAIT until the 10-minute mark (5 minutes remaining)
  3. We look at what the price ACTUALLY DID over those 10 minutes
  4. If the price moved clearly in one direction (not choppy):
     - And our ML model CONFIRMS the direction
     - And the Polymarket crowd hasn't fully caught up to reality
     - Then we bet that the last 5 minutes continues the same direction
  5. We hold until resolution (5 min later)

Conviction Evolution (Lifecycle Checkpoints):
  Every 15-minute market is tracked across 3 checkpoints (T=0, T=5, T=10)
  plus resolution (T=15). Conviction evolves:
    CONFIRMED: 3/3 agree, price confirms -> full bet
    GROWING:   2/3 agree -> 60% bet
    SIDEWAYS:  no clear direction -> skip
    CONFLICTED: ML flip-flopped -> skip

Market Discovery:
  Rolling 15m direction markets use slug pattern: {asset}-updown-15m-{unix_timestamp}
  where unix_timestamp is aligned to 900-second (15-min) boundaries.
  Discovered via direct Gamma API lookup (no scanner dependency).
  Confirmed assets: BTC, ETH, SOL, XRP.
"""

import json
import logging
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


# ===================================================================
# INSTRUMENT CONFIGURATION
# ===================================================================

@dataclass
class InstrumentConfig:
    """Complete configuration for one instrument under Strategy A."""

    # Identity
    name: str                     # "BTC 15m Confirmed Momentum"
    asset: str                    # "BTC"
    ml_pair: str                  # "BTC-USD" - which ML model to use
    price_pair: str = ""          # "XRP-USD" - which price to use (defaults to ml_pair)
    slug_pattern: str = ""        # Slug template for Polymarket discovery
    enabled: bool = True

    # Timing
    market_duration_minutes: int = 15
    eval_at_minutes_remaining: float = 5.0  # When to evaluate (5 min left)
    eval_window_seconds: float = 60.0       # Tolerance around eval point

    # Price movement thresholds
    min_price_move_pct: float = 0.10     # Min price move over first 10 min
    max_price_chop_ratio: float = 0.50   # Skip if choppy (>50% adverse)

    # ML confirmation thresholds
    min_ml_confidence: float = 50.0
    min_ml_agreement: float = 0.55
    ml_must_agree_with_price: bool = True

    # Edge thresholds
    min_edge: float = 0.03               # Minimum 3% edge over crowd
    max_crowd_efficiency: float = 0.85   # If crowd is at 85%+, skip

    # Position sizing
    kelly_fraction: float = 0.5
    max_bet_pct: float = 0.10
    min_bet_usd: float = 1.0
    max_bet_usd: float = 50.0

    # Risk management
    max_bets_per_hour: int = 4
    cooldown_after_loss_seconds: int = 900
    skip_regimes: list = field(default_factory=lambda: ["high_volatility"])

    # Lead-lag (for altcoins)
    lead_asset: Optional[str] = None
    lead_must_agree: bool = False


# ===================================================================
# THE 5 INSTRUMENTS
# ===================================================================

def build_instruments() -> Dict[str, InstrumentConfig]:
    """The curated universe. Five instruments, five sets of rules."""

    instruments = {}

    # 1. BTC 15m Direction
    # Crowd reprices in ~90s. Confirmation mode: ML is safety check, not primary signal.
    instruments["btc_15m"] = InstrumentConfig(
        name="BTC 15m Confirmed Momentum",
        asset="BTC",
        ml_pair="BTC-USD",
        price_pair="BTC-USD",
        slug_pattern="btc-updown-15m-{ts}",
        min_price_move_pct=0.12,
        max_price_chop_ratio=0.40,
        min_ml_confidence=45.0,
        min_ml_agreement=0.50,
        min_edge=0.04,
        max_crowd_efficiency=0.82,
        kelly_fraction=0.5,
        max_bet_pct=0.10,
        max_bets_per_hour=4,
        cooldown_after_loss_seconds=900,
        skip_regimes=["high_volatility"],
        lead_asset=None,
        lead_must_agree=False,
    )

    # 2. ETH 15m Direction
    # DeFi cascade dynamics make momentum sticky. BTC leads by ~45s.
    instruments["eth_15m"] = InstrumentConfig(
        name="ETH 15m Confirmed Momentum",
        asset="ETH",
        ml_pair="ETH-USD",
        price_pair="ETH-USD",
        slug_pattern="eth-updown-15m-{ts}",
        min_price_move_pct=0.15,
        max_price_chop_ratio=0.45,
        min_ml_confidence=45.0,
        min_ml_agreement=0.45,
        min_edge=0.035,
        max_crowd_efficiency=0.83,
        kelly_fraction=0.5,
        max_bet_pct=0.10,
        max_bets_per_hour=4,
        cooldown_after_loss_seconds=900,
        skip_regimes=["high_volatility"],
        lead_asset="BTC",
        lead_must_agree=True,
    )

    # 3. SOL 15m Direction
    # High beta (2-3x BTC). BTC leads by 60-120s (longest lead-lag).
    instruments["sol_15m"] = InstrumentConfig(
        name="SOL 15m Confirmed Momentum",
        asset="SOL",
        ml_pair="SOL-USD",
        price_pair="SOL-USD",
        slug_pattern="sol-updown-15m-{ts}",
        min_price_move_pct=0.25,
        max_price_chop_ratio=0.45,
        min_ml_confidence=40.0,
        min_ml_agreement=0.45,
        min_edge=0.03,
        max_crowd_efficiency=0.85,
        kelly_fraction=0.55,
        max_bet_pct=0.12,
        max_bets_per_hour=5,
        cooldown_after_loss_seconds=900,
        skip_regimes=["high_volatility"],
        lead_asset="BTC",
        lead_must_agree=True,
    )

    # 4. DOGE 15m Direction — DISABLED (no 15m direction market on Polymarket)
    instruments["doge_15m"] = InstrumentConfig(
        name="DOGE 15m Confirmed Momentum",
        asset="DOGE",
        ml_pair="DOGE-USD",
        price_pair="DOGE-USD",
        slug_pattern="doge-updown-15m-{ts}",
        enabled=False,
        min_price_move_pct=0.15,
        max_price_chop_ratio=0.50,
        min_ml_confidence=45.0,
        min_ml_agreement=0.50,
        min_edge=0.03,
        max_crowd_efficiency=0.85,
        kelly_fraction=0.40,
        max_bet_pct=0.08,
        max_bets_per_hour=4,
        cooldown_after_loss_seconds=1800,
        skip_regimes=["high_volatility", "extreme_volatility"],
        lead_asset="BTC",
        lead_must_agree=False,
    )

    # 5. XRP 15m Direction
    # News-driven. Uses BTC ML model but XRP's actual price for crowd comparison.
    instruments["xrp_15m"] = InstrumentConfig(
        name="XRP 15m Confirmed Momentum",
        asset="XRP",
        ml_pair="BTC-USD",       # Use BTC model for predictions
        price_pair="XRP-USD",    # Use XRP price for crowd comparison
        slug_pattern="xrp-updown-15m-{ts}",
        min_price_move_pct=0.15,
        max_price_chop_ratio=0.35,
        min_ml_confidence=45.0,
        min_ml_agreement=0.45,
        min_edge=0.05,
        max_crowd_efficiency=0.80,
        kelly_fraction=0.35,
        max_bet_pct=0.08,
        max_bets_per_hour=3,
        cooldown_after_loss_seconds=1800,
        skip_regimes=["high_volatility", "extreme_volatility"],
        lead_asset="BTC",
        lead_must_agree=True,
    )

    return instruments


# ===================================================================
# CONVICTION EVOLUTION SYSTEM
# ===================================================================

class ConvictionLevel(Enum):
    """Conviction evolves across 3 checkpoints."""
    CONFIRMED = "CONFIRMED"       # 3/3 agree, price confirms -> BET
    GROWING = "GROWING"           # 2/3 agree, trend building -> BET (smaller)
    TENTATIVE = "TENTATIVE"       # 1/1 so far, too early -> WAIT
    SIDEWAYS = "SIDEWAYS"         # No clear direction -> SKIP
    CONFLICTED = "CONFLICTED"     # ML or price flipped direction -> SKIP
    NO_SIGNAL = "NO_SIGNAL"       # Nothing actionable -> SKIP


# Conviction -> bet size multiplier
CONVICTION_BET_MULTIPLIER = {
    ConvictionLevel.CONFIRMED: 1.0,    # Full Strategy A bet
    ConvictionLevel.GROWING: 0.6,      # 60% of normal bet
    ConvictionLevel.TENTATIVE: 0.0,    # Don't bet
    ConvictionLevel.SIDEWAYS: 0.0,     # Don't bet
    ConvictionLevel.CONFLICTED: 0.0,   # Don't bet
    ConvictionLevel.NO_SIGNAL: 0.0,    # Don't bet
}


class ConvictionTracker:
    """
    Tracks a single market's lifecycle across 3 checkpoints
    and computes an evolving conviction score.

    Factors that INCREASE conviction:
    - ML direction consistent across checkpoints
    - Price moving steadily in one direction
    - ML confidence increasing over time
    - Price and ML agreeing with each other

    Factors that DECREASE conviction:
    - ML direction changed between checkpoints
    - Price reversed or went flat
    - ML confidence dropping
    - Price and ML disagreeing
    """

    def __init__(self, slug: str, asset: str, instrument_key: str):
        self.slug = slug
        self.asset = asset
        self.instrument_key = instrument_key
        self.checkpoints: Dict[int, dict] = {}  # 0, 5, 10

    def record_checkpoint(
        self,
        checkpoint: int,
        asset_price: float,
        crowd_up: float,
        ml_prediction: float,
        ml_confidence: float,
        ml_agreement: float,
        regime: str,
    ) -> None:
        """Record data at a checkpoint (T=0, T=5, or T=10)."""
        ml_direction = "UP" if ml_prediction > 0 else "DOWN"

        price_direction = "SIDEWAYS"
        price_change = 0.0
        if checkpoint > 0 and 0 in self.checkpoints:
            t0_price = self.checkpoints[0]["asset_price"]
            if t0_price > 0:
                price_change = ((asset_price - t0_price) / t0_price) * 100
                if price_change > 0.05:
                    price_direction = "UP"
                elif price_change < -0.05:
                    price_direction = "DOWN"

        self.checkpoints[checkpoint] = {
            "asset_price": asset_price,
            "crowd_up": crowd_up,
            "crowd_down": 1.0 - crowd_up,
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "ml_agreement": ml_agreement,
            "ml_direction": ml_direction,
            "regime": regime,
            "price_change_from_t0": price_change,
            "price_direction": price_direction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def compute_conviction(self) -> Tuple[ConvictionLevel, float, str]:
        """
        Compute conviction at the current state.
        Returns (level, score 0.0-1.0, human-readable detail).
        """
        n_checkpoints = len(self.checkpoints)

        if n_checkpoints == 0:
            return ConvictionLevel.NO_SIGNAL, 0.0, "No data yet"

        if n_checkpoints == 1:
            return ConvictionLevel.TENTATIVE, 0.1, "Only T=0 recorded"

        # Collect ML directions across checkpoints
        directions = [self.checkpoints[t]["ml_direction"] for t in sorted(self.checkpoints.keys())]
        all_same = len(set(directions)) == 1
        majority_direction = max(set(directions), key=directions.count)
        majority_count = directions.count(majority_direction)
        minority_count = len(directions) - majority_count

        # Price movement consistency
        price_confirms = False
        price_sideways = False
        if 0 in self.checkpoints:
            latest = self.checkpoints[max(self.checkpoints.keys())]
            price_dir = latest["price_direction"]
            if price_dir == majority_direction:
                price_confirms = True
            elif price_dir == "SIDEWAYS":
                price_sideways = True

        # ML confidence trend
        confidences = [self.checkpoints[t]["ml_confidence"] for t in sorted(self.checkpoints.keys())]
        confidence_rising = all(b >= a - 5 for a, b in zip(confidences, confidences[1:]))
        avg_confidence = sum(confidences) / len(confidences)

        # Compute score
        score = 0.0
        reasons = []

        # Direction consistency (biggest factor)
        if all_same and n_checkpoints >= 2:
            score += 0.40
            reasons.append(f"{n_checkpoints}/{n_checkpoints} checkpoints agree {majority_direction}")
        elif majority_count >= 2 and n_checkpoints >= 3:
            score += 0.20
            reasons.append(f"{majority_count}/{n_checkpoints} agree {majority_direction} (1 conflicted)")
        elif minority_count >= majority_count:
            return (
                ConvictionLevel.CONFLICTED, 0.05,
                f"ML flip-flopped: {directions}"
            )

        # Price confirmation
        if price_confirms:
            score += 0.25
            price_change = self.checkpoints[max(self.checkpoints.keys())]["price_change_from_t0"]
            reasons.append(f"Price confirms {majority_direction} ({price_change:+.3f}%)")
        elif price_sideways:
            score += 0.05
            reasons.append("Price sideways")
        else:
            score -= 0.10
            reasons.append("Price CONFLICTS with ML direction")

        # ML confidence
        if avg_confidence > 65:
            score += 0.15
            reasons.append(f"Avg ML confidence: {avg_confidence:.0f}")
        elif avg_confidence > 50:
            score += 0.08
            reasons.append(f"Avg ML confidence: {avg_confidence:.0f} (moderate)")

        # Confidence rising over time
        if confidence_rising and n_checkpoints >= 2:
            score += 0.10
            reasons.append("ML confidence rising")

        # Agreement (model consensus)
        if 10 in self.checkpoints:
            agreement = self.checkpoints[10]["ml_agreement"]
            if agreement > 0.70:
                score += 0.10
                reasons.append(f"Model agreement: {agreement:.0%}")

        score = max(0.0, min(1.0, score))

        # Classify conviction level
        if score >= 0.65 and all_same and price_confirms:
            level = ConvictionLevel.CONFIRMED
        elif score >= 0.45 and majority_count >= 2:
            level = ConvictionLevel.GROWING
        elif price_sideways or (all_same and not price_confirms):
            level = ConvictionLevel.SIDEWAYS
        elif minority_count >= majority_count:
            level = ConvictionLevel.CONFLICTED
        else:
            level = ConvictionLevel.TENTATIVE

        detail = " | ".join(reasons)
        return level, score, detail

    def get_majority_direction(self) -> str:
        """Get the direction most checkpoints agree on."""
        directions = [self.checkpoints[t]["ml_direction"] for t in sorted(self.checkpoints.keys())]
        if not directions:
            return "UNKNOWN"
        return max(set(directions), key=directions.count)

    def to_db_dict(self) -> dict:
        """Export all checkpoint data for database storage."""
        d: Dict[str, object] = {
            "slug": self.slug,
            "asset": self.asset,
            "instrument_key": self.instrument_key,
        }
        for t in [0, 5, 10]:
            prefix = f"t{t}_"
            if t in self.checkpoints:
                cp = self.checkpoints[t]
                d[f"{prefix}timestamp"] = cp.get("timestamp")
                d[f"{prefix}asset_price"] = cp["asset_price"]
                d[f"{prefix}crowd_up"] = cp["crowd_up"]
                d[f"{prefix}crowd_down"] = cp["crowd_down"]
                d[f"{prefix}ml_prediction"] = cp["ml_prediction"]
                d[f"{prefix}ml_confidence"] = cp["ml_confidence"]
                d[f"{prefix}ml_agreement"] = cp["ml_agreement"]
                d[f"{prefix}ml_direction"] = cp["ml_direction"]
                d[f"{prefix}regime"] = cp["regime"]
                if t > 0:
                    d[f"{prefix}price_change_from_t0"] = cp.get("price_change_from_t0", 0)
                    d[f"{prefix}price_direction"] = cp.get("price_direction", "")
                    d[f"{prefix}ml_agrees_with_t0"] = 1 if (
                        0 in self.checkpoints and
                        cp["ml_direction"] == self.checkpoints[0]["ml_direction"]
                    ) else 0

        directions = [self.checkpoints[t]["ml_direction"] for t in sorted(self.checkpoints.keys())]
        d["direction_readings"] = json.dumps(directions)
        return d


# ===================================================================
# THE EXECUTOR
# ===================================================================

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
MARKET_CACHE_TTL = 120  # 2-minute cache per slug


class StrategyAExecutor:
    """
    Executes Strategy A (Confirmed Momentum) across 5 instruments.

    Each cycle:
    1. Discover current 15m direction markets via slug construction + Gamma API
    2. Match markets to registered instruments
    3. Record checkpoint data at T=0, T=5, T=10
    4. At T=10: compute conviction, evaluate, place bet if all pass
    5. Check resolution of open positions (API + price-based fallback)
    """

    def __init__(self, config: dict, db_path: str, logger: Optional[logging.Logger] = None):
        self.config = config
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)

        pm_cfg = config.get("polymarket", {})
        self.enabled = pm_cfg.get("executor_enabled", True)
        self.paper_mode = pm_cfg.get("paper_mode", True)
        self.initial_bankroll = pm_cfg.get("initial_bankroll", 500.0)

        # Load instruments
        self.instruments = build_instruments()
        enabled = {k: v for k, v in self.instruments.items() if v.enabled}
        self.logger.info(f"Strategy A: {len(enabled)} instruments loaded")
        for key, inst in enabled.items():
            self.logger.info(f"  {key}: {inst.name} | edge>{inst.min_edge:.0%} | kelly={inst.kelly_fraction}")

        # State
        self.bankroll = self.initial_bankroll
        self.cooldowns: Dict[str, float] = {}   # asset -> cooldown_until
        self.hourly_bets: Dict[str, int] = {}   # asset -> count
        self.hourly_reset: float = 0.0

        # Conviction trackers: slug -> ConvictionTracker
        self.trackers: Dict[str, ConvictionTracker] = {}

        # Market cache: slug -> (market_dict, fetch_timestamp)
        self._market_cache: Dict[str, Tuple[dict, float]] = {}

        # Database
        self._ensure_tables()
        self._load_bankroll()
        self._close_non_strategy_a_positions()

    # ----------------------------------------------------------
    # DATABASE SETUP
    # ----------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
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

            CREATE INDEX IF NOT EXISTS idx_pm_pos_status ON polymarket_positions(status);
            CREATE INDEX IF NOT EXISTS idx_lifecycle_asset ON polymarket_lifecycle(asset);
            CREATE INDEX IF NOT EXISTS idx_lifecycle_conviction ON polymarket_lifecycle(conviction_label);
            CREATE INDEX IF NOT EXISTS idx_lifecycle_decision ON polymarket_lifecycle(decision);
        """)
        conn.commit()

        # Migrate: add 'strategy' column to existing polymarket_positions if missing
        cols = [r[1] for r in conn.execute("PRAGMA table_info(polymarket_positions)").fetchall()]
        if "strategy" not in cols:
            conn.execute("ALTER TABLE polymarket_positions ADD COLUMN strategy TEXT DEFAULT 'legacy'")
            conn.commit()
            self.logger.info("Migrated polymarket_positions: added 'strategy' column")

        # Now safe to create index on strategy column
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pm_pos_strategy ON polymarket_positions(strategy)")
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

    def _close_non_strategy_a_positions(self) -> None:
        """On startup: close any open positions from old strategies."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        old_positions = conn.execute("""
            SELECT * FROM polymarket_positions
            WHERE status = 'open'
              AND (strategy IS NULL OR strategy != 'strategy_a')
        """).fetchall()

        returned = 0.0
        for pos in old_positions:
            conn.execute("""
                UPDATE polymarket_positions
                SET status = 'cancelled', pnl = 0, closed_at = datetime('now'),
                    notes = 'Closed: pivot to Strategy A only'
                WHERE position_id = ?
            """, (pos['position_id'],))
            self.bankroll += pos['bet_amount']
            returned += pos['bet_amount']
            self.logger.info(
                f"CLOSED old position: {(pos['question'] or '')[:50]} | "
                f"Returned ${pos['bet_amount']:.2f}"
            )

        if old_positions:
            conn.commit()
            self._log_bankroll("strategy_pivot", None, returned)
            self.logger.info(
                f"Strategy A startup: closed {len(old_positions)} old positions, "
                f"returned ${returned:.2f} to bankroll (${self.bankroll:.2f})"
            )
        conn.close()

    # ----------------------------------------------------------
    # MAIN CYCLE
    # ----------------------------------------------------------

    async def execute_cycle(
        self,
        ml_predictions: dict,
        current_prices: dict,
        current_regime: str = "unknown",
    ) -> None:
        """
        Called every bot cycle from the main trading loop.

        Args:
            ml_predictions: {pair: {prediction, agreement, confidence}}
            current_prices: {pair: current_price}
            current_regime: regime detector label
        """
        if not self.enabled:
            return

        # Reset hourly counters
        now = time.time()
        if now - self.hourly_reset > 3600:
            self.hourly_bets = {}
            self.hourly_reset = now

        # Check resolution of open positions
        self._check_resolutions(current_prices)

        # Discover and process current 15m direction markets
        for inst_key, inst in self.instruments.items():
            if not inst.enabled:
                continue

            # Skip if in cooldown
            if now < self.cooldowns.get(inst.asset, 0):
                continue

            # Skip if hourly limit reached
            if self.hourly_bets.get(inst.asset, 0) >= inst.max_bets_per_hour:
                continue

            # Skip if regime is excluded
            if current_regime in inst.skip_regimes:
                self.logger.debug(f"[{inst.asset}] Skipping: regime '{current_regime}' excluded")
                continue

            # Discover market via slug construction + Gamma API
            market = self._discover_market(inst)
            if not market:
                continue

            minutes_left = self._get_minutes_remaining(market)
            if minutes_left is None:
                continue

            slug = market.get("slug", "")
            price_pair = inst.price_pair or inst.ml_pair
            current_price = current_prices.get(price_pair, 0)

            # Parse crowd pricing from Gamma API response
            crowd_up = self._parse_crowd_up(market)

            # ML data
            ml_data = ml_predictions.get(inst.ml_pair, {})
            ml_pred = ml_data.get("prediction", 0)
            ml_conf = ml_data.get("confidence", 50)
            ml_agree = ml_data.get("agreement", 0)

            # Lead asset check (for altcoins)
            if inst.lead_asset and inst.lead_must_agree:
                leader_data = ml_predictions.get(f"{inst.lead_asset}-USD", {})
                leader_pred = leader_data.get("prediction", 0)
                leader_dir = "UP" if leader_pred > 0 else "DOWN"
                price_dir = "UP" if ml_pred > 0 else "DOWN"
                if leader_dir != price_dir:
                    self.logger.debug(f"[{inst.asset}] Lead {inst.lead_asset} disagrees")
                    continue

            # Determine which checkpoint this is
            checkpoint = None
            if 14.0 <= minutes_left <= 16.0:
                checkpoint = 0
            elif 9.0 <= minutes_left <= 11.0:
                checkpoint = 5
            elif 4.0 <= minutes_left <= 6.0:
                checkpoint = 10

            if checkpoint is None:
                continue

            # Get or create tracker
            if slug not in self.trackers:
                self.trackers[slug] = ConvictionTracker(
                    slug=slug,
                    asset=inst.asset,
                    instrument_key=inst_key,
                )

            tracker = self.trackers[slug]

            # Don't re-record the same checkpoint
            if checkpoint in tracker.checkpoints:
                if checkpoint != 10:
                    continue
                # At T=10 we may need to re-evaluate if already recorded
                # but don't re-record data

            # Record checkpoint
            tracker.record_checkpoint(
                checkpoint=checkpoint,
                asset_price=current_price,
                crowd_up=crowd_up,
                ml_prediction=ml_pred,
                ml_confidence=ml_conf,
                ml_agreement=ml_agree,
                regime=current_regime,
            )

            self.logger.info(
                f"CHECKPOINT [{inst.asset}] T={checkpoint}: "
                f"Price=${current_price:.2f} | "
                f"Crowd UP={crowd_up:.0%} | "
                f"ML={'UP' if ml_pred > 0 else 'DOWN'} conf={ml_conf:.0f} | "
                f"Regime={current_regime} | "
                f"Slug={slug}"
            )

            # DECISION AT T=10 ONLY
            if checkpoint == 10:
                self._make_decision(inst_key, inst, market, tracker, ml_predictions, current_prices, current_regime)
                # Clean up tracker after decision
                self.trackers.pop(slug, None)

        # Clean up stale trackers (older than 20 min)
        self._cleanup_stale_trackers()

    # ----------------------------------------------------------
    # DECISION AT T=10
    # ----------------------------------------------------------

    def _make_decision(
        self,
        inst_key: str,
        inst: InstrumentConfig,
        market: dict,
        tracker: ConvictionTracker,
        ml_predictions: dict,
        current_prices: dict,
        current_regime: str,
    ) -> None:
        """Evaluate conviction and decide whether to bet at T=10."""

        conviction_level, conviction_score, conviction_detail = tracker.compute_conviction()

        self.logger.info(
            f"CONVICTION [{tracker.asset}]: {conviction_level.value} "
            f"(score={conviction_score:.2f}) -- {conviction_detail}"
        )

        multiplier = CONVICTION_BET_MULTIPLIER.get(conviction_level, 0)
        should_bet = multiplier > 0
        skip_reason = None

        if should_bet:
            # Run the full evaluation
            ref_price = tracker.checkpoints.get(0, {}).get("asset_price", 0)
            price_pair = inst.price_pair or inst.ml_pair
            current_price = current_prices.get(price_pair, 0)

            if ref_price <= 0:
                ref_price = current_price  # Fallback if T=0 missing

            decision = self._evaluate(
                inst_key, inst, market,
                ref_price, current_price,
                ml_predictions, current_regime,
            )

            if decision["enter"]:
                # Adjust bet by conviction multiplier
                decision["bet_amount"] = round(decision["bet_amount"] * multiplier, 2)
                decision["bet_amount"] = max(inst.min_bet_usd, decision["bet_amount"])

                result = self._place_bet(inst_key, inst, market, decision)
                if result:
                    self.hourly_bets[inst.asset] = self.hourly_bets.get(inst.asset, 0) + 1
                    self.logger.info(
                        f"STRATEGY A [{inst.asset}]: {decision['direction']} | "
                        f"Conviction: {conviction_level.value} ({conviction_score:.2f}) | "
                        f"Edge: {decision['edge']:.1%} | "
                        f"Price: {decision['price_move_pct']:+.2f}% over 10min | "
                        f"Bet: ${decision['bet_amount']:.2f}"
                    )
                    self._save_lifecycle(
                        tracker, market, conviction_level, conviction_score,
                        conviction_detail, decision, result.get("position_id"),
                    )
                    return

            skip_reason = decision.get("reason", "evaluation failed")
            should_bet = False

        if not should_bet:
            skip_reason = skip_reason or f"{conviction_level.value}: {conviction_detail}"
            self.logger.info(f"[{tracker.asset}] SKIP: {skip_reason}")
            self._save_lifecycle(
                tracker, market, conviction_level, conviction_score,
                conviction_detail,
                {"enter": False, "direction": tracker.get_majority_direction(), "reason": skip_reason},
                None,
            )

    # ----------------------------------------------------------
    # EVALUATION - Heart of Strategy A
    # ----------------------------------------------------------

    def _evaluate(
        self,
        inst_key: str,
        inst: InstrumentConfig,
        market: dict,
        ref_price: float,
        current_price: float,
        ml_predictions: dict,
        current_regime: str,
    ) -> dict:
        """Core evaluation. All conditions must pass."""

        # 1. PRICE MOVEMENT
        if ref_price <= 0 or current_price <= 0:
            return {"enter": False, "reason": "Missing price data"}

        price_move_pct = ((current_price - ref_price) / ref_price) * 100
        price_direction = "UP" if price_move_pct > 0 else "DOWN"

        if abs(price_move_pct) < inst.min_price_move_pct:
            return {"enter": False, "reason": f"Price move {abs(price_move_pct):.3f}% < {inst.min_price_move_pct}%"}

        # 2. ML CONFIRMATION
        ml_data = ml_predictions.get(inst.ml_pair, {})
        ml_pred = ml_data.get("prediction", 0)
        ml_conf = ml_data.get("confidence", 0)
        ml_agree = ml_data.get("agreement", 0)
        ml_direction = "UP" if ml_pred > 0 else "DOWN"

        if inst.ml_must_agree_with_price and ml_direction != price_direction:
            return {"enter": False, "reason": f"ML says {ml_direction} but price went {price_direction}"}

        if ml_conf < inst.min_ml_confidence:
            return {"enter": False, "reason": f"ML confidence {ml_conf:.0f} < {inst.min_ml_confidence}"}

        if ml_agree < inst.min_ml_agreement:
            return {"enter": False, "reason": f"ML agreement {ml_agree:.0%} < {inst.min_ml_agreement:.0%}"}

        # 3. LEAD ASSET CHECK
        if inst.lead_asset and inst.lead_must_agree:
            leader_pair = f"{inst.lead_asset}-USD"
            leader_pred = ml_predictions.get(leader_pair, {}).get("prediction", 0)
            leader_direction = "UP" if leader_pred > 0 else "DOWN"
            if leader_direction != price_direction:
                return {"enter": False, "reason": f"Lead {inst.lead_asset} disagrees: {leader_direction} vs {price_direction}"}

        # 4. CROWD PRICING
        crowd_up = market.get("crowd_prob_yes", 0.5)
        crowd_down = 1.0 - crowd_up

        if price_direction == "UP":
            our_direction = "UP"
            crowd_for_us = crowd_up
            entry_price = crowd_up
        else:
            our_direction = "DOWN"
            crowd_for_us = crowd_down
            entry_price = crowd_down

        # Convert ML prediction + price confirmation to our probability
        base_prob = 0.65 + abs(price_move_pct) * 5.0
        ml_boost = (ml_conf - 50) * 0.003
        our_prob = min(0.88, max(0.55, base_prob + ml_boost))

        edge = our_prob - crowd_for_us

        self.logger.info(
            f"[{inst.asset}] EVAL: price_move={price_move_pct:+.3f}% | "
            f"ML={ml_direction} conf={ml_conf:.0f} agree={ml_agree:.0%} | "
            f"Crowd={crowd_for_us:.0%} | Edge={edge:.1%} | "
            f"Regime={current_regime}"
        )

        if edge < inst.min_edge:
            return {"enter": False, "reason": f"Edge {edge:.1%} < {inst.min_edge:.1%}"}

        # 5. CROWD EFFICIENCY
        if crowd_for_us > inst.max_crowd_efficiency:
            return {"enter": False, "reason": f"Crowd at {crowd_for_us:.0%} -- no edge"}

        # 6. EXISTING POSITION CHECK
        slug = market.get("slug", "")
        conn = sqlite3.connect(self.db_path)
        existing = conn.execute(
            "SELECT COUNT(*) FROM polymarket_positions WHERE slug = ? AND status = 'open'",
            (slug,)
        ).fetchone()[0]
        conn.close()
        if existing > 0:
            return {"enter": False, "reason": "Already have position"}

        # 7. BANKROLL CHECK
        if self.bankroll < inst.min_bet_usd:
            return {"enter": False, "reason": f"Bankroll ${self.bankroll:.2f} below minimum"}

        # COMPUTE BET SIZE
        if entry_price <= 0 or entry_price >= 1:
            return {"enter": False, "reason": f"Invalid entry price: {entry_price}"}

        b = (1.0 / entry_price) - 1.0
        p = our_prob
        q = 1.0 - p

        kelly_raw = max(0, (p * b - q) / b)
        kelly_adj = min(kelly_raw * inst.kelly_fraction, inst.max_bet_pct)

        bet_amount = round(self.bankroll * kelly_adj, 2)
        bet_amount = max(inst.min_bet_usd, min(bet_amount, inst.max_bet_usd))

        if bet_amount > self.bankroll:
            return {"enter": False, "reason": "Insufficient bankroll"}

        return {
            "enter": True,
            "direction": our_direction,
            "edge": edge,
            "entry_price": entry_price,
            "our_prob": our_prob,
            "crowd_prob": crowd_for_us,
            "bet_amount": bet_amount,
            "kelly_raw": kelly_raw,
            "kelly_adjusted": kelly_adj,
            "price_move_pct": price_move_pct,
            "price_direction": price_direction,
            "ml_prediction": ml_pred,
            "ml_confidence": ml_conf,
            "ml_agreement": ml_agree,
            "current_regime": current_regime,
        }

    # ----------------------------------------------------------
    # MARKET DISCOVERY (direct slug construction + Gamma API)
    # ----------------------------------------------------------

    def _discover_market(self, inst: InstrumentConfig) -> Optional[dict]:
        """
        Discover the current 15m direction market for an instrument.
        Constructs the slug from current time, fetches from Gamma API with cache.
        """
        now_ts = int(time.time())
        window_ts = (now_ts // 900) * 900
        slug = inst.slug_pattern.format(ts=window_ts)

        # Check cache
        cached = self._market_cache.get(slug)
        if cached:
            market_data, cache_time = cached
            if now_ts - cache_time < MARKET_CACHE_TTL:
                return market_data

        # Fetch from Gamma API
        market = self._fetch_market_by_slug(slug)
        if market:
            self._market_cache[slug] = (market, now_ts)
            return market

        # If current window not found, try previous window (may still be open)
        prev_slug = inst.slug_pattern.format(ts=window_ts - 900)
        cached_prev = self._market_cache.get(prev_slug)
        if cached_prev:
            market_data, cache_time = cached_prev
            if now_ts - cache_time < MARKET_CACHE_TTL:
                return market_data

        market = self._fetch_market_by_slug(prev_slug)
        if market:
            self._market_cache[prev_slug] = (market, now_ts)
            return market

        return None

    def _fetch_market_by_slug(self, slug: str) -> Optional[dict]:
        """Fetch a single market from Gamma API by slug. Returns normalized dict."""
        try:
            resp = requests.get(
                GAMMA_MARKETS_URL,
                params={"slug": slug},
                timeout=10,
            )
            if resp.status_code != 200:
                return None

            markets = resp.json()
            if not markets:
                return None

            m = markets[0]

            # Parse end date
            end_date = m.get("endDate", "")
            # Parse outcome prices
            crowd_up = self._parse_crowd_up(m)

            return {
                "slug": m.get("slug", slug),
                "question": m.get("question", ""),
                "market_type": "DIRECTION",
                "asset": self._extract_asset(m.get("question", "")),
                "condition_id": m.get("conditionId", ""),
                "market_id": m.get("id", ""),
                "crowd_prob_yes": crowd_up,
                "deadline": end_date,
                "volume_24h": float(m.get("volume24hr", 0) or 0),
                "liquidity": float(m.get("liquidity", 0) or 0),
                "end_date_source": m.get("endDateIso", end_date),
                "resolved": m.get("resolved", False),
                "gamma_raw": m,  # Keep raw for resolution checks
            }
        except Exception as e:
            self.logger.debug(f"Gamma API fetch failed for {slug}: {e}")
            return None

    @staticmethod
    def _parse_crowd_up(market: dict) -> float:
        """Extract YES price (crowd probability of UP) from market data."""
        # If already parsed (from cache)
        if "crowd_prob_yes" in market and market["crowd_prob_yes"] is not None:
            raw = market.get("gamma_raw")
            if raw is None:
                return float(market["crowd_prob_yes"])

        # Parse from Gamma API raw response
        raw = market.get("gamma_raw", market)
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

        # Fallback: bestAsk/bestBid
        best_ask = raw.get("bestAsk")
        if best_ask:
            try:
                return float(best_ask)
            except (ValueError, TypeError):
                pass

        return 0.5

    @staticmethod
    def _extract_asset(question: str) -> str:
        """Extract asset name from question like 'Bitcoin Up or Down - ...'."""
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
        """Get minutes remaining until market resolution."""
        deadline_str = market.get("deadline", "")
        if not deadline_str:
            return None
        try:
            deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
            now_utc = datetime.now(timezone.utc)
            remaining = (deadline - now_utc).total_seconds() / 60.0
            return max(0, remaining)
        except (ValueError, TypeError):
            return None

    def _cleanup_stale_trackers(self) -> None:
        """Remove trackers older than 20 minutes."""
        stale = []
        for slug, tracker in self.trackers.items():
            if 0 in tracker.checkpoints:
                t0_ts = tracker.checkpoints[0].get("timestamp", "")
                if t0_ts:
                    try:
                        t0 = datetime.fromisoformat(t0_ts)
                        age = (datetime.now(timezone.utc) - t0).total_seconds()
                        if age > 1200:
                            stale.append(slug)
                    except (ValueError, TypeError):
                        stale.append(slug)
        for slug in stale:
            del self.trackers[slug]

        # Also clean up old cache entries
        now_ts = int(time.time())
        stale_cache = [k for k, (_, ts) in self._market_cache.items() if now_ts - ts > 1800]
        for k in stale_cache:
            del self._market_cache[k]

    # ----------------------------------------------------------
    # BET PLACEMENT
    # ----------------------------------------------------------

    def _place_bet(self, inst_key: str, inst: InstrumentConfig,
                   market: dict, decision: dict) -> Optional[dict]:
        """Place a paper bet."""
        position_id = f"sa_{uuid.uuid4().hex[:12]}"

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO polymarket_positions
            (position_id, slug, question, market_type, asset,
             direction, entry_price, shares, bet_amount, edge_at_entry,
             our_prob_at_entry, crowd_prob_at_entry, deadline,
             status, opened_at, registry_key, entry_window, strategy,
             notes)
            VALUES (?, ?, ?, 'DIRECTION', ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    'open', datetime('now'), ?, 'confirmed_momentum', 'strategy_a', ?)
        """, (
            position_id,
            market.get("slug", ""),
            market.get("question", ""),
            inst.asset,
            decision["direction"],
            decision["entry_price"],
            decision["bet_amount"] / decision["entry_price"],
            decision["bet_amount"],
            decision["edge"],
            decision["our_prob"],
            decision["crowd_prob"],
            market.get("deadline", ""),
            inst_key,
            f"price_move={decision['price_move_pct']:+.3f}%,regime={decision['current_regime']}",
        ))
        conn.commit()
        conn.close()

        self.bankroll -= decision["bet_amount"]
        self._log_bankroll("bet_placed", position_id, -decision["bet_amount"])

        return {"position_id": position_id}

    # ----------------------------------------------------------
    # RESOLUTION
    # ----------------------------------------------------------

    def _check_resolutions(self, current_prices: Optional[dict] = None) -> None:
        """
        Check if open Strategy A positions have resolved.
        Uses Gamma API + price-based fallback for stale positions.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        open_positions = conn.execute(
            "SELECT * FROM polymarket_positions WHERE status = 'open' AND strategy = 'strategy_a'"
        ).fetchall()

        for pos in open_positions:
            slug = pos["slug"]
            if not slug:
                continue

            # Check if deadline has passed — use price-based resolution if stale
            deadline_str = pos["deadline"]
            deadline_passed = False
            seconds_past_deadline = 0
            if deadline_str:
                try:
                    deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
                    seconds_past_deadline = (datetime.now(timezone.utc) - deadline).total_seconds()
                    if seconds_past_deadline > 0:
                        deadline_passed = True
                except (ValueError, TypeError):
                    pass

            try:
                # Try Gamma API resolution
                market = self._fetch_market_by_slug(slug)

                if market and market.get("gamma_raw"):
                    raw = market["gamma_raw"]
                    is_resolved = raw.get("resolved", False)
                    prices = raw.get("outcomePrices", "[]")
                    if isinstance(prices, str):
                        prices = json.loads(prices)

                    if isinstance(prices, list) and len(prices) >= 2:
                        yes_price = float(prices[0])
                        no_price = float(prices[1])

                        is_definitive = (yes_price >= 0.95 and no_price <= 0.05) or \
                                       (yes_price <= 0.05 and no_price >= 0.95)

                        if is_resolved or is_definitive:
                            self._resolve_position(conn, pos, yes_price, no_price, current_prices, "gamma_api")
                            continue

                        if not deadline_passed:
                            # Update unrealized P&L
                            direction = pos["direction"]
                            cur_price = yes_price if direction == "UP" else no_price
                            unrealized = (cur_price - pos["entry_price"]) * pos["shares"]
                            conn.execute(
                                "UPDATE polymarket_positions SET notes = ? WHERE position_id = ?",
                                (f"unrealized={unrealized:.2f},price={cur_price:.3f}", pos["position_id"]),
                            )
                            continue

                # Price-based fallback for stale positions (>2 min past deadline)
                if deadline_passed and seconds_past_deadline > 120 and current_prices:
                    inst_key = pos.get("registry_key", "")
                    inst = self.instruments.get(inst_key)
                    if inst:
                        price_pair = inst.price_pair or inst.ml_pair
                        current_asset_price = current_prices.get(price_pair, 0)
                        t0_price = self._get_lifecycle_t0_price(slug)

                        if current_asset_price > 0 and t0_price > 0:
                            price_change = ((current_asset_price - t0_price) / t0_price) * 100
                            actual_direction = "UP" if price_change > 0 else "DOWN"
                            direction = pos["direction"]

                            won = (actual_direction == direction)
                            yes_price = 1.0 if actual_direction == "UP" else 0.0
                            no_price = 1.0 - yes_price

                            self.logger.info(
                                f"PRICE-BASED RESOLUTION [{pos['asset']}]: "
                                f"T0=${t0_price:.2f} → T15=${current_asset_price:.2f} "
                                f"({price_change:+.3f}%) → {actual_direction} | "
                                f"Bet was {direction} → {'WON' if won else 'LOST'}"
                            )
                            self._resolve_position(conn, pos, yes_price, no_price, current_prices, "price_fallback")
                            continue

                # If >30 min past deadline and no resolution, force-expire
                if deadline_passed and seconds_past_deadline > 1800:
                    self.logger.info(f"FORCE-EXPIRE [{pos['asset']}]: {slug} (30min past deadline)")
                    conn.execute("""
                        UPDATE polymarket_positions
                        SET status = 'expired', pnl = 0, closed_at = datetime('now'),
                            notes = 'Force-expired: 30min past deadline, no resolution data'
                        WHERE position_id = ?
                    """, (pos["position_id"],))
                    self.bankroll += pos["bet_amount"]
                    self._log_bankroll("force_expired", pos["position_id"], pos["bet_amount"])

            except Exception as e:
                self.logger.debug(f"Resolution check failed for {pos['position_id']}: {e}")

        conn.commit()
        conn.close()

    def _resolve_position(
        self, conn, pos, yes_price: float, no_price: float,
        current_prices: Optional[dict], source: str,
    ) -> None:
        """Resolve a position and record T=15 lifecycle data."""
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
            f"{'WON' if won else 'LOST'} RESOLVED [{pos['asset']}]: "
            f"{status.upper()} {pos['direction']} | "
            f"P&L: ${pnl:+.2f} | Bankroll: ${self.bankroll:.2f} | via {source}"
        )

        # Set cooldown on loss
        if not won:
            inst = self.instruments.get(pos.get("registry_key", ""), None)
            cooldown_secs = inst.cooldown_after_loss_seconds if inst else 900
            self.cooldowns[pos["asset"]] = time.time() + cooldown_secs

        # Record T=15 in lifecycle
        final_result = "UP" if yes_price >= 0.95 else "DOWN"
        t15_price = None
        if current_prices:
            inst = self.instruments.get(pos.get("registry_key", ""))
            if inst:
                price_pair = inst.price_pair or inst.ml_pair
                t15_price = current_prices.get(price_pair, 0)

        t0_price = self._get_lifecycle_t0_price(pos["slug"])
        t10_price = self._get_lifecycle_t10_price(pos["slug"])

        try:
            conn.execute("""
                UPDATE polymarket_lifecycle
                SET t15_timestamp = datetime('now'),
                    t15_asset_price = ?,
                    t15_final_result = ?,
                    t15_price_change_from_t0 = CASE WHEN ? > 0 AND ? > 0
                        THEN ((? - ?) / ? * 100) ELSE NULL END,
                    t15_price_change_from_t10 = CASE WHEN ? > 0 AND ? > 0
                        THEN ((? - ?) / ? * 100) ELSE NULL END,
                    t15_won = ?,
                    t15_pnl = ?,
                    final_5min_reversed = CASE
                        WHEN t10_price_direction IS NOT NULL AND ? != t10_price_direction THEN 1
                        ELSE 0 END,
                    ml_accuracy = CASE
                        WHEN t10_ml_direction = ? THEN 1
                        ELSE 0 END
                WHERE position_id = ?
            """, (
                t15_price,
                final_result,
                t15_price, t0_price, t15_price, t0_price, t0_price,     # t15 vs t0
                t15_price, t10_price, t15_price, t10_price, t10_price,   # t15 vs t10
                1 if won else 0,
                pnl,
                final_result,
                final_result,
                pos["position_id"],
            ))
        except Exception as e:
            self.logger.debug(f"Lifecycle T=15 update failed: {e}")

        self._log_bankroll(f"resolved_{status}", pos["position_id"], pnl)

    def _get_lifecycle_t0_price(self, slug: str) -> float:
        """Get T=0 asset price from lifecycle table."""
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT t0_asset_price FROM polymarket_lifecycle WHERE slug = ? ORDER BY id DESC LIMIT 1",
                (slug,)
            ).fetchone()
            conn.close()
            return float(row[0]) if row and row[0] else 0.0
        except Exception:
            return 0.0

    def _get_lifecycle_t10_price(self, slug: str) -> float:
        """Get T=10 asset price from lifecycle table."""
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT t10_asset_price FROM polymarket_lifecycle WHERE slug = ? ORDER BY id DESC LIMIT 1",
                (slug,)
            ).fetchone()
            conn.close()
            return float(row[0]) if row and row[0] else 0.0
        except Exception:
            return 0.0

    # ----------------------------------------------------------
    # LIFECYCLE PERSISTENCE
    # ----------------------------------------------------------

    def _save_lifecycle(
        self,
        tracker: ConvictionTracker,
        market: dict,
        conviction_level: ConvictionLevel,
        conviction_score: float,
        conviction_detail: str,
        decision: dict,
        position_id: Optional[str] = None,
    ) -> None:
        """Persist the full lifecycle record to the audit table."""
        data = tracker.to_db_dict()

        # Compute crowd lag metrics
        t5_crowd_change = None
        t10_crowd_change = None
        if 0 in tracker.checkpoints and 5 in tracker.checkpoints:
            t5_crowd_change = tracker.checkpoints[5]["crowd_up"] - tracker.checkpoints[0]["crowd_up"]
        if 0 in tracker.checkpoints and 10 in tracker.checkpoints:
            t10_crowd_change = tracker.checkpoints[10]["crowd_up"] - tracker.checkpoints[0]["crowd_up"]

        # T=10 price change from T=5
        t10_price_from_t5 = None
        if 5 in tracker.checkpoints and 10 in tracker.checkpoints:
            t5_price = tracker.checkpoints[5]["asset_price"]
            t10_price = tracker.checkpoints[10]["asset_price"]
            if t5_price > 0:
                t10_price_from_t5 = ((t10_price - t5_price) / t5_price) * 100

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO polymarket_lifecycle (
                    slug, question, asset, instrument_key,
                    market_open_time,
                    t0_timestamp, t0_asset_price, t0_crowd_up, t0_crowd_down,
                    t0_ml_prediction, t0_ml_confidence, t0_ml_agreement,
                    t0_ml_direction, t0_regime,
                    t5_timestamp, t5_asset_price, t5_crowd_up, t5_crowd_down,
                    t5_ml_prediction, t5_ml_confidence, t5_ml_agreement,
                    t5_ml_direction, t5_regime,
                    t5_price_change_from_t0, t5_crowd_change_from_t0,
                    t5_price_direction, t5_ml_agrees_with_t0,
                    t10_timestamp, t10_asset_price, t10_crowd_up, t10_crowd_down,
                    t10_ml_prediction, t10_ml_confidence, t10_ml_agreement,
                    t10_ml_direction, t10_regime,
                    t10_price_change_from_t0, t10_price_change_from_t5,
                    t10_crowd_change_from_t0,
                    t10_price_direction, t10_ml_agrees_with_t0, t10_ml_agrees_with_t5,
                    conviction_score, conviction_label, conviction_detail,
                    direction_readings,
                    decision, decision_reason,
                    bet_amount, entry_price, edge_at_entry, position_id
                ) VALUES (
                    ?, ?, ?, ?,
                    ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?,
                    ?, ?,
                    ?, ?, ?, ?
                )
            """, (
                data["slug"], market.get("question"), data["asset"], data["instrument_key"],
                data.get("t0_timestamp", datetime.now(timezone.utc).isoformat()),
                # T=0
                data.get("t0_timestamp"), data.get("t0_asset_price"), data.get("t0_crowd_up"), data.get("t0_crowd_down"),
                data.get("t0_ml_prediction"), data.get("t0_ml_confidence"), data.get("t0_ml_agreement"),
                data.get("t0_ml_direction"), data.get("t0_regime"),
                # T=5
                data.get("t5_timestamp"), data.get("t5_asset_price"), data.get("t5_crowd_up"), data.get("t5_crowd_down"),
                data.get("t5_ml_prediction"), data.get("t5_ml_confidence"), data.get("t5_ml_agreement"),
                data.get("t5_ml_direction"), data.get("t5_regime"),
                data.get("t5_price_change_from_t0"), t5_crowd_change,
                data.get("t5_price_direction"), data.get("t5_ml_agrees_with_t0"),
                # T=10
                data.get("t10_timestamp"), data.get("t10_asset_price"), data.get("t10_crowd_up"), data.get("t10_crowd_down"),
                data.get("t10_ml_prediction"), data.get("t10_ml_confidence"), data.get("t10_ml_agreement"),
                data.get("t10_ml_direction"), data.get("t10_regime"),
                data.get("t10_price_change_from_t0"), t10_price_from_t5,
                t10_crowd_change,
                data.get("t10_price_direction"), data.get("t10_ml_agrees_with_t0"), data.get("t10_ml_agrees_with_t5", 0),
                # Conviction
                conviction_score, conviction_level.value, conviction_detail,
                data.get("direction_readings", "[]"),
                # Decision
                f"BET_{decision.get('direction', '?')}" if decision.get("enter") else "SKIP",
                decision.get("reason", ""),
                decision.get("bet_amount") if decision.get("enter") else None,
                decision.get("entry_price") if decision.get("enter") else None,
                decision.get("edge") if decision.get("enter") else None,
                position_id,
            ))
            conn.commit()
        except Exception as e:
            self.logger.debug(f"Lifecycle save failed: {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------
    # SUMMARY (for dashboard)
    # ----------------------------------------------------------

    def get_summary(self) -> dict:
        """Summary stats for dashboard display."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        open_pos = conn.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(bet_amount),0) as exp "
            "FROM polymarket_positions WHERE status='open' AND strategy='strategy_a'"
        ).fetchone()

        resolved = conn.execute(
            "SELECT COUNT(*) as cnt, "
            "SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as w, "
            "SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) as l, "
            "COALESCE(SUM(pnl),0) as pnl "
            "FROM polymarket_positions WHERE status IN ('won','lost') AND strategy='strategy_a'"
        ).fetchone()

        pnl_24h = conn.execute(
            "SELECT COALESCE(SUM(pnl),0) as pnl FROM polymarket_positions "
            "WHERE status IN ('won','lost') AND strategy='strategy_a' "
            "AND closed_at >= datetime('now','-24 hours')"
        ).fetchone()

        per_asset = conn.execute(
            "SELECT asset, COUNT(*) as n, "
            "SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as w, "
            "COALESCE(SUM(pnl),0) as pnl "
            "FROM polymarket_positions WHERE status IN ('won','lost') AND strategy='strategy_a' "
            "GROUP BY asset ORDER BY pnl DESC"
        ).fetchall()

        conn.close()

        total = resolved["cnt"] or 0
        wins = resolved["w"] or 0

        return {
            "strategy": "Strategy A: Confirmed Momentum",
            "bankroll": round(self.bankroll, 2),
            "open_count": open_pos["cnt"],
            "open_exposure": round(open_pos["exp"], 2),
            "total_bets": total,
            "wins": wins,
            "losses": resolved["l"] or 0,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "total_pnl": round(resolved["pnl"] or 0, 2),
            "pnl_24h": round(pnl_24h["pnl"] or 0, 2),
            "per_asset": [dict(r) for r in per_asset],
            "instruments": {k: v.name for k, v in self.instruments.items() if v.enabled},
        }
