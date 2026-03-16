"""
Temporal Pattern Mining — Medallion's "188th five-minute period" approach.

Analyzes historical arb trades to find time-of-day, day-of-week, and
combined temporal patterns in profitability. Outputs bias weights that
tell the arb detector when to be more/less aggressive.

Key question: "Is the 14:00 UTC hour on Wednesdays consistently more
profitable for BTC/USDT cross-exchange arb than the 03:00 hour on Sundays?"

If yes, we can:
1. Increase confidence/sizing during profitable windows
2. Decrease activity during unprofitable windows
3. Prioritize scanning pairs that historically perform well at this time
"""

import sqlite3
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger("arb.temporal")


@dataclass
class TemporalBucket:
    """Statistics for a single temporal bucket (e.g., hour=14, day=Wednesday)."""
    bucket_key: str              # e.g. "hour_14", "dow_3", "hour_14_dow_3"
    total_trades: int = 0
    filled_trades: int = 0
    profitable_trades: int = 0
    total_profit_usd: float = 0.0
    total_gross_spread_bps: float = 0.0
    total_net_spread_bps: float = 0.0
    avg_profit_usd: float = 0.0
    avg_spread_bps: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0    # gross_wins / gross_losses
    sample_count: int = 0         # For significance testing
    is_significant: bool = False  # >= min_sample_size trades
    bias_weight: float = 1.0     # Multiplier for arb detector (0.0-2.0)


@dataclass
class TemporalProfile:
    """Complete temporal profile for a strategy or strategy+pair combination."""
    strategy: str
    pair: Optional[str]           # None = aggregate across all pairs
    by_hour: Dict[int, TemporalBucket] = field(default_factory=dict)      # 0-23
    by_dow: Dict[int, TemporalBucket] = field(default_factory=dict)       # 0=Mon, 6=Sun
    by_hour_dow: Dict[str, TemporalBucket] = field(default_factory=dict)  # "14_3" = 14:00 Wed
    by_funding_window: Dict[str, TemporalBucket] = field(default_factory=dict)  # pre/post settlement
    updated_at: Optional[datetime] = None


class TemporalAnalyzer:
    """
    Mines arb trade history for temporal patterns.

    Runs on startup and then refreshes every REFRESH_INTERVAL_MINUTES.
    Results are cached in memory and optionally persisted to a JSON file
    for fast startup on restart.
    """

    MIN_SAMPLE_SIZE = 20            # Minimum trades in a bucket to consider significant
    REFRESH_INTERVAL_MINUTES = 60   # Re-analyze every hour

    # Funding settlement times (UTC) — Binance/MEXC settle every 8 hours
    FUNDING_SETTLEMENT_HOURS = [0, 8, 16]
    FUNDING_WINDOW_MINUTES = 30     # Minutes before/after settlement to flag

    def __init__(self, db_path: str = "data/arbitrage.db",
                 cache_path: str = "data/temporal_profiles.json"):
        self.db_path = db_path
        self.cache_path = cache_path
        self._profiles: Dict[str, TemporalProfile] = {}
        self._last_refresh: Optional[datetime] = None
        self._trade_count_at_last_refresh: int = 0

    async def initialize(self):
        """Load cached profiles or compute from scratch."""
        import asyncio
        cache = Path(self.cache_path)
        if cache.exists():
            try:
                self._load_cache()
                logger.info(
                    "Loaded temporal profiles from cache — %d profiles",
                    len(self._profiles),
                )
            except Exception as e:
                logger.warning("Cache load failed, recomputing: %s", e)
                await asyncio.to_thread(self._compute_all_profiles)
        else:
            await asyncio.to_thread(self._compute_all_profiles)

    def _compute_all_profiles(self):
        """
        Full recomputation from database.

        Computes profiles for:
        1. Each strategy (aggregate across all pairs)
        2. Each strategy + pair combination
        3. Global aggregate (all strategies, all pairs)
        """
        logger.info("Computing temporal profiles from trade history...")

        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
        except Exception as e:
            logger.error("Cannot open DB for temporal analysis: %s", e)
            return

        try:
            rows = conn.execute(
                "SELECT strategy, symbol, actual_profit_usd, gross_spread_bps, "
                "net_spread_bps, timestamp FROM arb_trades WHERE status = 'filled'"
            ).fetchall()
        except Exception as e:
            logger.error("Query failed: %s", e)
            conn.close()
            return

        conn.close()

        if not rows:
            logger.warning("No filled trades found — temporal analysis empty")
            return

        logger.info("Analyzing %d filled trades for temporal patterns", len(rows))

        # Group trades into buckets
        grouped: Dict[str, List[dict]] = defaultdict(list)

        for row in rows:
            trade = dict(row)
            try:
                ts = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                continue
            trade['_hour'] = ts.hour
            trade['_dow'] = ts.weekday()  # 0=Monday
            trade['_minute'] = ts.minute
            trade['_funding_window'] = self._classify_funding_window(ts)

            strategy = trade['strategy']
            pair = trade['symbol']

            grouped[f"{strategy}:*"].append(trade)
            grouped[f"{strategy}:{pair}"].append(trade)
            grouped["*:*"].append(trade)

        # Compute profiles
        self._profiles = {}
        for key, trades in grouped.items():
            strategy_part, pair_part = key.split(":", 1)
            strategy = strategy_part if strategy_part != "*" else "all"
            pair = pair_part if pair_part != "*" else None

            profile = self._build_profile(strategy, pair, trades)
            self._profiles[key] = profile

        self._last_refresh = datetime.utcnow()
        self._trade_count_at_last_refresh = len(rows)
        self._save_cache()

        logger.info(
            "Temporal analysis complete — %d profiles, %d total trades",
            len(self._profiles), len(rows),
        )

    def _build_profile(
        self, strategy: str, pair: Optional[str], trades: List[dict]
    ) -> TemporalProfile:
        """Build a TemporalProfile from a list of trades."""
        profile = TemporalProfile(strategy=strategy, pair=pair)

        # By hour (0-23)
        by_hour: Dict[int, list] = defaultdict(list)
        for t in trades:
            by_hour[t['_hour']].append(t)
        for hour in range(24):
            profile.by_hour[hour] = self._compute_bucket(
                f"hour_{hour}", by_hour.get(hour, [])
            )

        # By day of week (0=Mon, 6=Sun)
        by_dow: Dict[int, list] = defaultdict(list)
        for t in trades:
            by_dow[t['_dow']].append(t)
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for dow in range(7):
            profile.by_dow[dow] = self._compute_bucket(
                f"dow_{dow}_{dow_names[dow]}", by_dow.get(dow, [])
            )

        # By hour x day combination
        by_hour_dow: Dict[str, list] = defaultdict(list)
        for t in trades:
            key = f"{t['_hour']}_{t['_dow']}"
            by_hour_dow[key].append(t)
        for key, bucket_trades in by_hour_dow.items():
            profile.by_hour_dow[key] = self._compute_bucket(
                f"hour_dow_{key}", bucket_trades
            )

        # By funding settlement window
        by_funding: Dict[str, list] = defaultdict(list)
        for t in trades:
            by_funding[t['_funding_window']].append(t)
        for window, bucket_trades in by_funding.items():
            profile.by_funding_window[window] = self._compute_bucket(
                f"funding_{window}", bucket_trades
            )

        self._compute_bias_weights(profile, trades)
        profile.updated_at = datetime.utcnow()
        return profile

    def _compute_bucket(self, bucket_key: str, trades: List[dict]) -> TemporalBucket:
        """Compute statistics for a single temporal bucket."""
        bucket = TemporalBucket(bucket_key=bucket_key)
        if not trades:
            return bucket

        bucket.total_trades = len(trades)
        bucket.filled_trades = len(trades)
        bucket.sample_count = len(trades)

        profits = [t['actual_profit_usd'] for t in trades if t.get('actual_profit_usd') is not None]
        if not profits:
            return bucket

        bucket.profitable_trades = sum(1 for p in profits if p > 0)
        bucket.total_profit_usd = sum(profits)
        bucket.avg_profit_usd = bucket.total_profit_usd / len(profits)
        bucket.win_rate = bucket.profitable_trades / len(profits)

        spreads = [t['net_spread_bps'] for t in trades if t.get('net_spread_bps') is not None]
        bucket.avg_spread_bps = sum(spreads) / len(spreads) if spreads else 0.0

        gross_wins = sum(p for p in profits if p > 0)
        gross_losses = abs(sum(p for p in profits if p < 0))
        bucket.profit_factor = (
            gross_wins / gross_losses if gross_losses > 0
            else float('inf') if gross_wins > 0
            else 0.0
        )

        bucket.is_significant = len(profits) >= self.MIN_SAMPLE_SIZE
        return bucket

    def _compute_bias_weights(
        self, profile: TemporalProfile, all_trades: List[dict]
    ):
        """
        Compute bias weights for each bucket.

        Weight = bucket_avg_profit / overall_avg_profit, clamped to [0.2, 2.0].
        Only significant buckets get non-default weights.
        """
        if not all_trades:
            return

        profits = [t['actual_profit_usd'] for t in all_trades if t.get('actual_profit_usd') is not None]
        if not profits:
            return

        overall_avg = sum(profits) / len(profits)
        use_win_rate = overall_avg <= 0

        if use_win_rate:
            overall_win_rate = sum(1 for p in profits if p > 0) / len(profits) if profits else 0

        for bucket_collection in [
            profile.by_hour, profile.by_dow,
            profile.by_hour_dow, profile.by_funding_window,
        ]:
            for _key, bucket in bucket_collection.items():
                if not bucket.is_significant:
                    bucket.bias_weight = 1.0
                    continue

                if use_win_rate:
                    if overall_win_rate > 0:
                        raw = bucket.win_rate / overall_win_rate
                    else:
                        raw = 1.0
                else:
                    if overall_avg != 0:
                        raw = bucket.avg_profit_usd / overall_avg
                    else:
                        raw = 1.0

                bucket.bias_weight = max(0.2, min(2.0, raw))

    def _classify_funding_window(self, ts: datetime) -> str:
        """Classify a timestamp relative to funding settlement."""
        for settle_hour in self.FUNDING_SETTLEMENT_HOURS:
            minutes_in_day = ts.hour * 60 + ts.minute
            settle_minutes = settle_hour * 60
            diff = settle_minutes - minutes_in_day

            if diff < -720:
                diff += 1440
            elif diff > 720:
                diff -= 1440

            if 0 < diff <= self.FUNDING_WINDOW_MINUTES:
                return "pre_settlement_30m"
            if -self.FUNDING_WINDOW_MINUTES <= diff <= 0:
                return "post_settlement_30m"

        return "normal"

    async def maybe_refresh(self):
        """Refresh if enough time has passed or new trades have arrived."""
        import asyncio
        if self._last_refresh is None:
            await asyncio.to_thread(self._compute_all_profiles)
            return

        elapsed = (datetime.utcnow() - self._last_refresh).total_seconds() / 60
        if elapsed < self.REFRESH_INTERVAL_MINUTES:
            return

        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            count = conn.execute(
                "SELECT COUNT(*) FROM arb_trades WHERE status = 'filled'"
            ).fetchone()[0]
            conn.close()
        except Exception:
            return

        if count > self._trade_count_at_last_refresh:
            logger.info(
                "Refreshing temporal profiles — %d new trades since last refresh",
                count - self._trade_count_at_last_refresh,
            )
            await asyncio.to_thread(self._compute_all_profiles)

    def get_current_bias(
        self, strategy: str, pair: Optional[str] = None
    ) -> float:
        """
        Get the temporal bias weight for the current moment.

        Returns a float in [0.2, 2.0].
        Combines hour-of-day, day-of-week, and funding window signals.
        """
        now = datetime.utcnow()
        hour = now.hour
        dow = now.weekday()
        hour_dow_key = f"{hour}_{dow}"
        funding_window = self._classify_funding_window(now)

        profile_key = f"{strategy}:{pair}" if pair else f"{strategy}:*"
        profile = self._profiles.get(profile_key)
        if profile is None:
            profile = self._profiles.get(f"{strategy}:*")
        if profile is None:
            profile = self._profiles.get("*:*")
        if profile is None:
            return 1.0

        weights = []

        hour_bucket = profile.by_hour.get(hour)
        if hour_bucket and hour_bucket.is_significant:
            weights.append(hour_bucket.bias_weight)

        dow_bucket = profile.by_dow.get(dow)
        if dow_bucket and dow_bucket.is_significant:
            weights.append(dow_bucket.bias_weight)

        # Hour x day gets double weight if significant
        hour_dow_bucket = profile.by_hour_dow.get(hour_dow_key)
        if hour_dow_bucket and hour_dow_bucket.is_significant:
            weights.append(hour_dow_bucket.bias_weight)
            weights.append(hour_dow_bucket.bias_weight)

        funding_bucket = profile.by_funding_window.get(funding_window)
        if funding_bucket and funding_bucket.is_significant:
            weights.append(funding_bucket.bias_weight)

        if not weights:
            return 1.0

        # Geometric mean
        product = 1.0
        for w in weights:
            product *= w
        combined = product ** (1.0 / len(weights))

        return max(0.2, min(2.0, combined))

    def get_report(self) -> dict:
        """Generate a report for the dashboard."""
        report = {
            "last_updated": self._last_refresh.isoformat() if self._last_refresh else None,
            "total_trades_analyzed": self._trade_count_at_last_refresh,
            "strategies": {},
        }

        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        for key, profile in self._profiles.items():
            strategy_part, pair_part = key.split(":", 1)
            if pair_part != "*":
                continue

            strategy_report = {
                "best_hours": [],
                "worst_hours": [],
                "best_days": [],
                "worst_days": [],
                "funding_effects": {},
                "current_bias": self.get_current_bias(strategy_part),
            }

            sig_hours = [
                (h, b) for h, b in profile.by_hour.items() if b.is_significant
            ]
            sig_hours.sort(key=lambda x: x[1].avg_profit_usd, reverse=True)

            for hour, bucket in sig_hours[:3]:
                strategy_report["best_hours"].append({
                    "hour_utc": hour,
                    "avg_profit_usd": round(bucket.avg_profit_usd, 4),
                    "win_rate": round(bucket.win_rate, 4),
                    "trades": bucket.total_trades,
                    "bias": round(bucket.bias_weight, 3),
                })
            for hour, bucket in sig_hours[-3:]:
                strategy_report["worst_hours"].append({
                    "hour_utc": hour,
                    "avg_profit_usd": round(bucket.avg_profit_usd, 4),
                    "win_rate": round(bucket.win_rate, 4),
                    "trades": bucket.total_trades,
                    "bias": round(bucket.bias_weight, 3),
                })

            sig_days = [
                (d, b) for d, b in profile.by_dow.items() if b.is_significant
            ]
            sig_days.sort(key=lambda x: x[1].avg_profit_usd, reverse=True)

            for dow, bucket in sig_days[:2]:
                strategy_report["best_days"].append({
                    "day": dow_names[dow],
                    "avg_profit_usd": round(bucket.avg_profit_usd, 4),
                    "win_rate": round(bucket.win_rate, 4),
                    "trades": bucket.total_trades,
                    "bias": round(bucket.bias_weight, 3),
                })
            for dow, bucket in sig_days[-2:]:
                strategy_report["worst_days"].append({
                    "day": dow_names[dow],
                    "avg_profit_usd": round(bucket.avg_profit_usd, 4),
                    "win_rate": round(bucket.win_rate, 4),
                    "trades": bucket.total_trades,
                    "bias": round(bucket.bias_weight, 3),
                })

            for window, bucket in profile.by_funding_window.items():
                if bucket.is_significant:
                    strategy_report["funding_effects"][window] = {
                        "avg_profit_usd": round(bucket.avg_profit_usd, 4),
                        "win_rate": round(bucket.win_rate, 4),
                        "trades": bucket.total_trades,
                        "bias": round(bucket.bias_weight, 3),
                    }

            report["strategies"][strategy_part] = strategy_report

        return report

    def _save_cache(self):
        """Persist profiles to JSON for fast startup."""
        try:
            cache_data = {
                "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
                "trade_count": self._trade_count_at_last_refresh,
                "profiles": {},
            }

            for key, profile in self._profiles.items():
                cache_data["profiles"][key] = {
                    "strategy": profile.strategy,
                    "pair": profile.pair,
                    "by_hour": {
                        str(k): self._bucket_to_dict(v) for k, v in profile.by_hour.items()
                    },
                    "by_dow": {
                        str(k): self._bucket_to_dict(v) for k, v in profile.by_dow.items()
                    },
                    "by_hour_dow": {
                        k: self._bucket_to_dict(v) for k, v in profile.by_hour_dow.items()
                    },
                    "by_funding_window": {
                        k: self._bucket_to_dict(v) for k, v in profile.by_funding_window.items()
                    },
                }

            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)

            logger.info("Temporal profiles cached to %s", self.cache_path)
        except Exception as e:
            logger.warning("Failed to cache temporal profiles: %s", e)

    def _load_cache(self):
        """Load profiles from JSON cache."""
        with open(self.cache_path) as f:
            data = json.load(f)

        self._last_refresh = (
            datetime.fromisoformat(data["last_refresh"])
            if data.get("last_refresh") else None
        )
        self._trade_count_at_last_refresh = data.get("trade_count", 0)

        self._profiles = {}
        for key, pdata in data.get("profiles", {}).items():
            profile = TemporalProfile(
                strategy=pdata["strategy"],
                pair=pdata.get("pair"),
            )
            for hour_str, bdata in pdata.get("by_hour", {}).items():
                profile.by_hour[int(hour_str)] = self._dict_to_bucket(bdata)
            for dow_str, bdata in pdata.get("by_dow", {}).items():
                profile.by_dow[int(dow_str)] = self._dict_to_bucket(bdata)
            for key_str, bdata in pdata.get("by_hour_dow", {}).items():
                profile.by_hour_dow[key_str] = self._dict_to_bucket(bdata)
            for window, bdata in pdata.get("by_funding_window", {}).items():
                profile.by_funding_window[window] = self._dict_to_bucket(bdata)
            self._profiles[key] = profile

    @staticmethod
    def _bucket_to_dict(bucket: TemporalBucket) -> dict:
        pf = bucket.profit_factor
        if pf == float('inf'):
            pf = 999999.0
        return {
            "bucket_key": bucket.bucket_key,
            "total_trades": bucket.total_trades,
            "filled_trades": bucket.filled_trades,
            "profitable_trades": bucket.profitable_trades,
            "total_profit_usd": bucket.total_profit_usd,
            "avg_profit_usd": bucket.avg_profit_usd,
            "avg_spread_bps": bucket.avg_spread_bps,
            "win_rate": bucket.win_rate,
            "profit_factor": pf,
            "sample_count": bucket.sample_count,
            "is_significant": bucket.is_significant,
            "bias_weight": bucket.bias_weight,
        }

    @staticmethod
    def _dict_to_bucket(d: dict) -> TemporalBucket:
        return TemporalBucket(
            bucket_key=d.get("bucket_key", ""),
            total_trades=d.get("total_trades", 0),
            filled_trades=d.get("filled_trades", 0),
            profitable_trades=d.get("profitable_trades", 0),
            total_profit_usd=d.get("total_profit_usd", 0.0),
            avg_profit_usd=d.get("avg_profit_usd", 0.0),
            avg_spread_bps=d.get("avg_spread_bps", 0.0),
            win_rate=d.get("win_rate", 0.0),
            profit_factor=d.get("profit_factor", 0.0),
            sample_count=d.get("sample_count", 0),
            is_significant=d.get("is_significant", False),
            bias_weight=d.get("bias_weight", 1.0),
        )
