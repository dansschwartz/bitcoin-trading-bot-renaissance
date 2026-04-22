"""
Unified Strategy Allocator — determines optimal capital allocation
across arbitrage strategies using velocity, edge health, and Sharpe ratio.

Starts in observation mode for first 2 weeks: reports targets but
doesn't enforce allocation changes.
"""
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger("arb.analytics.allocator")


class StrategyAllocator:

    # Scoring weights
    VELOCITY_WEIGHT = 0.50
    HEALTH_WEIGHT = 0.30
    SHARPE_WEIGHT = 0.20

    # Allocation bounds
    MIN_ALLOCATION = 0.10   # 10% floor
    MAX_ALLOCATION = 0.60   # 60% ceiling
    MAX_SHIFT_PER_WEEK = 0.20   # Max 20% shift per week

    # Health scores (mapped from edge decay health classification)
    HEALTH_SCORES = {
        "healthy": 1.0,
        "warning": 0.6,
        "critical": 0.3,
        "dead": 0.0,
        "insufficient_data": 0.5,
    }

    def __init__(self, db_path: str = "data/arbitrage.db"):
        self.db_path = db_path
        self._start_time = time.time()
        self._observation_period_sec = 14 * 86400  # 2 weeks
        self._current_allocation: Dict[str, float] = {}
        self._target_allocation: Dict[str, float] = {}
        self._last_report: Optional[Dict] = None

    @property
    def observation_mode(self) -> bool:
        return (time.time() - self._start_time) < self._observation_period_sec

    def rebalance(self, velocity_report: Dict, decay_report: Dict) -> Dict:
        """Compute optimal allocation from velocity + edge decay data."""
        strategies = ["cross_exchange", "triangular", "funding_rate"]

        # Compute composite score per strategy
        scores = {}
        for strat in strategies:
            # Velocity score (normalized to 0-1)
            vel_data = velocity_report.get("strategies", {}).get(strat, {})
            vel_24h = vel_data.get("24h", {}).get("velocity", 0.0)

            # Health score
            decay_data = decay_report.get("strategies", {}).get(strat, {})
            health_label = decay_data.get("health", "insufficient_data")
            health_score = self.HEALTH_SCORES.get(health_label, 0.5)

            # Sharpe ratio from DB
            sharpe = self._compute_sharpe(strat)

            scores[strat] = {
                "velocity": vel_24h,
                "health_score": health_score,
                "sharpe": sharpe,
            }

        # Normalize velocity scores (0-1 range)
        max_vel = max(abs(s["velocity"]) for s in scores.values()) or 1.0
        for s in scores.values():
            s["velocity_norm"] = max(0.0, s["velocity"] / max_vel)

        # Normalize Sharpe (0-1 range)
        max_sharpe = max(abs(s["sharpe"]) for s in scores.values()) or 1.0
        for s in scores.values():
            s["sharpe_norm"] = max(0.0, (s["sharpe"] + 1.0) / (max_sharpe + 1.0))

        # Composite score
        for strat, s in scores.items():
            s["composite"] = (
                self.VELOCITY_WEIGHT * s["velocity_norm"]
                + self.HEALTH_WEIGHT * s["health_score"]
                + self.SHARPE_WEIGHT * s["sharpe_norm"]
            )

        # Convert scores to allocation percentages
        total_score = sum(s["composite"] for s in scores.values()) or 1.0
        raw_alloc = {
            strat: s["composite"] / total_score
            for strat, s in scores.items()
        }

        # Apply bounds
        target = {}
        for strat, alloc in raw_alloc.items():
            target[strat] = max(self.MIN_ALLOCATION, min(self.MAX_ALLOCATION, alloc))

        # Re-normalize to sum to 1.0
        total = sum(target.values()) or 1.0
        target = {k: round(v / total, 4) for k, v in target.items()}

        # Apply max shift constraint (only if we have previous allocation)
        if self._current_allocation:
            for strat in strategies:
                current = self._current_allocation.get(strat, 1.0 / len(strategies))
                diff = target[strat] - current
                if abs(diff) > self.MAX_SHIFT_PER_WEEK:
                    target[strat] = current + (self.MAX_SHIFT_PER_WEEK if diff > 0 else -self.MAX_SHIFT_PER_WEEK)

            # Re-normalize after shift constraint
            total = sum(target.values()) or 1.0
            target = {k: round(v / total, 4) for k, v in target.items()}

        self._target_allocation = target

        # In observation mode, don't update current allocation
        if not self.observation_mode:
            self._current_allocation = target.copy()

        self._last_report = {
            "observation_mode": self.observation_mode,
            "current_allocation": self._current_allocation or {s: round(1.0 / len(strategies), 4) for s in strategies},
            "target_allocation": target,
            "scores": scores,
            "generated_at": time.time(),
        }
        return self._last_report

    def get_allocation_report(self) -> Dict:
        """Return latest report."""
        if self._last_report is None:
            return {
                "observation_mode": self.observation_mode,
                "current_allocation": {},
                "target_allocation": {},
                "note": "Allocator has not run yet — waiting for first weekly cycle.",
            }
        return self._last_report

    def _compute_sharpe(self, strategy: str, days: int = 7) -> float:
        """Compute annualized Sharpe ratio for a strategy over N days."""
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                """SELECT actual_profit_usd FROM arb_trades
                   WHERE strategy = ? AND status = 'filled'
                     AND timestamp >= date('now', ?)
                   ORDER BY timestamp""",
                (strategy, f"-{days} days"),
            ).fetchall()
            conn.close()
        except Exception:
            return 0.0

        if len(rows) < 5:
            return 0.0

        profits = [float(r[0]) for r in rows]
        mean = sum(profits) / len(profits)
        variance = sum((p - mean) ** 2 for p in profits) / len(profits)
        std = variance ** 0.5

        if std == 0:
            return 0.0

        # Annualize: assume ~100 trades/day
        daily_sharpe = mean / std
        return round(daily_sharpe * (365 ** 0.5), 4)
