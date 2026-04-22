"""
Edge Decay Monitor — tracks whether strategy profitability is eroding over time.

Uses linear regression on 7-day rolling windows of daily average profit
per strategy. Classifies health as healthy/warning/critical/dead.
"""
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("arb.analytics.edge_decay")


class EdgeDecayMonitor:

    # Slope thresholds ($ per day change in avg profit)
    HEALTHY_FLOOR = 0.0       # slope >= 0
    WARNING_FLOOR = -0.5      # -0.5 to 0
    CRITICAL_FLOOR = -1.0     # -1.0 to -0.5
    # Below -1.0 = dead

    def __init__(self, db_path: str = "data/arbitrage.db"):
        self.db_path = db_path
        self._last_report: Optional[Dict] = None

    def compute_daily(self) -> Dict:
        """Query arb_trades for daily avg profit per strategy, run regression."""
        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy not available — edge decay disabled")
            return {"error": "numpy not available"}

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        strategies = ["cross_exchange", "triangular", "funding_rate"]
        report = {}

        for strategy in strategies:
            rows = conn.execute(
                """SELECT date(timestamp) as day,
                          AVG(actual_profit_usd) as avg_profit,
                          COUNT(*) as trade_count
                   FROM arb_trades
                   WHERE strategy = ? AND status = 'filled'
                     AND timestamp >= date('now', '-7 days')
                   GROUP BY date(timestamp)
                   ORDER BY date(timestamp)""",
                (strategy,),
            ).fetchall()

            if len(rows) < 3:
                report[strategy] = {
                    "health": "insufficient_data",
                    "slope": 0.0,
                    "days_analyzed": len(rows),
                    "avg_daily_profit": 0.0,
                }
                continue

            # X = day index (0, 1, 2, ...), Y = avg profit per trade
            x = np.arange(len(rows), dtype=float)
            y = np.array([float(r["avg_profit"]) for r in rows])

            # Linear regression: y = slope * x + intercept
            slope, intercept = np.polyfit(x, y, 1)

            avg_daily_profit = float(np.mean(y))

            # Classify health
            if slope >= self.HEALTHY_FLOOR:
                health = "healthy"
            elif slope >= self.WARNING_FLOOR:
                health = "warning"
            elif slope >= self.CRITICAL_FLOOR:
                health = "critical"
            else:
                health = "dead"

            report[strategy] = {
                "health": health,
                "slope": round(float(slope), 6),
                "intercept": round(float(intercept), 6),
                "days_analyzed": len(rows),
                "avg_daily_profit": round(avg_daily_profit, 6),
                "latest_avg_profit": round(float(y[-1]), 6),
                "daily_data": [
                    {
                        "date": r["day"],
                        "avg_profit": round(float(r["avg_profit"]), 6),
                        "trade_count": r["trade_count"],
                    }
                    for r in rows
                ],
            }

        conn.close()

        self._last_report = {
            "strategies": report,
            "generated_at": time.time(),
            "window_days": 7,
        }
        return self._last_report

    def get_decay_report(self) -> Dict:
        """Return latest report (compute if none exists)."""
        if self._last_report is None:
            return self.compute_daily()
        return self._last_report
