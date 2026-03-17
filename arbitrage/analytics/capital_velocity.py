"""
Capital Velocity Tracker — measures how efficiently capital is deployed.

velocity = profit / (trade_size × hold_hours)

Higher velocity = same capital earns more per unit time.
"""
import logging
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

logger = logging.getLogger("arb.analytics.velocity")


class CapitalVelocityTracker:

    def __init__(self, db_path: str = "data/arbitrage.db"):
        self.db_path = db_path
        # Ring buffer: (timestamp, strategy, trade_size_usd, hold_seconds, profit_usd)
        self._trades: deque = deque(maxlen=5000)

    def record_trade(self, strategy: str, trade_size_usd: float,
                     hold_seconds: float, profit_usd: float):
        """Record a completed trade for velocity calculation."""
        self._trades.append({
            "ts": time.time(),
            "strategy": strategy,
            "trade_size_usd": trade_size_usd,
            "hold_seconds": hold_seconds,
            "profit_usd": profit_usd,
        })

    def get_velocity_report(self) -> Dict:
        """Compute per-strategy velocity over 1h/6h/24h windows."""
        now = time.time()
        windows = {
            "1h": 3600,
            "6h": 21600,
            "24h": 86400,
        }

        # Group trades by strategy
        by_strategy: Dict[str, List[dict]] = defaultdict(list)
        for t in self._trades:
            by_strategy[t["strategy"]].append(t)

        report = {}
        for strategy, trades in by_strategy.items():
            strategy_report = {}
            for window_name, window_sec in windows.items():
                cutoff = now - window_sec
                window_trades = [t for t in trades if t["ts"] >= cutoff]

                if not window_trades:
                    strategy_report[window_name] = {
                        "velocity": 0.0,
                        "trades": 0,
                        "total_profit": 0.0,
                        "total_capital_hours": 0.0,
                    }
                    continue

                total_profit = sum(t["profit_usd"] for t in window_trades)
                total_capital_hours = sum(
                    t["trade_size_usd"] * (t["hold_seconds"] / 3600.0)
                    for t in window_trades
                )

                velocity = (
                    total_profit / total_capital_hours
                    if total_capital_hours > 0 else 0.0
                )

                strategy_report[window_name] = {
                    "velocity": round(velocity, 6),
                    "trades": len(window_trades),
                    "total_profit": round(total_profit, 4),
                    "total_capital_hours": round(total_capital_hours, 2),
                }

            report[strategy] = strategy_report

        return {
            "strategies": report,
            "total_tracked_trades": len(self._trades),
            "generated_at": time.time(),
        }
