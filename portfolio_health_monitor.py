"""
Portfolio Health Monitor — Rolling Sharpe & Auto-Sizing
========================================================
Computes rolling risk-adjusted performance metrics and automatically
adjusts position sizing when the strategy is underperforming.

"When the Sharpe drops, you're not trading alpha — you're trading noise."
"""

from __future__ import annotations

import logging
import math
import numpy as np
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List


class PortfolioHealthMonitor:
    """
    Monitors rolling portfolio performance and returns a size_multiplier
    that the position sizer should apply.

    Usage:
        monitor = PortfolioHealthMonitor(config, logger)
        # After each trade completes:
        monitor.record_trade(pnl_pct=0.003, product_id="BTC-USD")
        # At start of each cycle:
        multiplier = monitor.get_size_multiplier()
        # Apply: max_position_usd *= multiplier
    """

    def __init__(self, config: Dict[str, Any] = None, logger: Optional[logging.Logger] = None):
        config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Window sizes for rolling calculations
        self._short_window = int(config.get("short_window", 20))    # 20 trades
        self._medium_window = int(config.get("medium_window", 50))   # 50 trades
        self._long_window = int(config.get("long_window", 100))     # 100 trades

        # Sharpe thresholds for auto-sizing
        self._full_size_sharpe = float(config.get("full_size_sharpe", 1.0))     # Above = full size
        self._half_size_sharpe = float(config.get("half_size_sharpe", 0.5))     # Below = half size
        self._exits_only_sharpe = float(config.get("exits_only_sharpe", 0.0))   # Below = exits only

        # Trade history: deque of pnl_pct values
        self._trade_returns: deque = deque(maxlen=self._long_window)

        # Current state
        self._size_multiplier = 1.0
        self._exits_only = False
        self._last_sharpe = 0.0
        self._total_trades = 0
        self._total_wins = 0

        # Alert tracking
        self._last_alert_level = "normal"

        self.logger.info(
            f"PortfolioHealthMonitor initialized: "
            f"full>{self._full_size_sharpe}, half>{self._half_size_sharpe}, "
            f"exits-only>{self._exits_only_sharpe}"
        )

    def record_trade(self, pnl_pct: float, product_id: str = ""):
        """Record a completed trade's return percentage."""
        self._trade_returns.append(pnl_pct)
        self._total_trades += 1
        if pnl_pct > 0:
            self._total_wins += 1

        # Recompute after each trade
        self._update_state()

    def _update_state(self):
        """Recompute Sharpe and update size multiplier."""
        returns = list(self._trade_returns)
        if len(returns) < self._short_window:
            # Not enough data — stay at full size
            self._size_multiplier = 1.0
            self._exits_only = False
            return

        # Compute rolling Sharpe on medium window (or whatever we have)
        window = returns[-self._medium_window:] if len(returns) >= self._medium_window else returns
        sharpe = self._compute_sharpe(window)
        self._last_sharpe = sharpe

        old_multiplier = self._size_multiplier
        old_exits_only = self._exits_only

        # Apply auto-sizing rules
        if sharpe >= self._full_size_sharpe:
            self._size_multiplier = 1.0
            self._exits_only = False
            alert_level = "healthy"
        elif sharpe >= self._half_size_sharpe:
            self._size_multiplier = 0.5
            self._exits_only = False
            alert_level = "caution"
        elif sharpe >= self._exits_only_sharpe:
            self._size_multiplier = 0.25
            self._exits_only = False
            alert_level = "warning"
        else:
            self._size_multiplier = 0.0
            self._exits_only = True
            alert_level = "critical"

        # Log state changes
        if alert_level != self._last_alert_level:
            self.logger.warning(
                f"HEALTH MONITOR: State changed {self._last_alert_level} -> {alert_level} | "
                f"Sharpe={sharpe:.2f} | Size multiplier: {old_multiplier:.2f} -> {self._size_multiplier:.2f}"
                f"{' | EXITS ONLY MODE' if self._exits_only else ''}"
            )
            self._last_alert_level = alert_level

    def _compute_sharpe(self, returns: List[float]) -> float:
        """Compute Sharpe ratio from a list of trade returns."""
        if len(returns) < 5:
            return 0.0

        arr = np.array(returns)
        mean_return = float(np.mean(arr))
        std_return = float(np.std(arr, ddof=1))

        if std_return < 1e-10:
            return 0.0 if mean_return <= 0 else 5.0  # Cap at 5.0 for near-zero vol

        # Annualize assuming ~100 trades per day (crypto, multiple products)
        # But since we're using per-trade returns, just compute raw Sharpe
        sharpe = mean_return / std_return

        # Scale to roughly match annualized convention
        # Assuming ~20 trades/day, ~252 trading days -> ~5040 trades/year
        trades_per_year = 5040
        annualized = sharpe * math.sqrt(trades_per_year)

        return float(annualized)

    def get_size_multiplier(self) -> float:
        """Return current position size multiplier [0.0, 1.0]."""
        return self._size_multiplier

    def is_exits_only(self) -> bool:
        """Return whether we're in exits-only mode."""
        return self._exits_only

    def get_metrics(self) -> Dict[str, Any]:
        """Return comprehensive health metrics for dashboard/logging."""
        returns = list(self._trade_returns)

        metrics = {
            "total_trades": self._total_trades,
            "win_rate": self._total_wins / max(self._total_trades, 1),
            "size_multiplier": self._size_multiplier,
            "exits_only": self._exits_only,
            "alert_level": self._last_alert_level,
        }

        if len(returns) >= self._short_window:
            short = returns[-self._short_window:]
            metrics["sharpe_short"] = self._compute_sharpe(short)
            metrics["win_rate_short"] = sum(1 for r in short if r > 0) / len(short)
            metrics["avg_return_short"] = float(np.mean(short))

        if len(returns) >= self._medium_window:
            medium = returns[-self._medium_window:]
            metrics["sharpe_medium"] = self._compute_sharpe(medium)
            metrics["win_rate_medium"] = sum(1 for r in medium if r > 0) / len(medium)

        if len(returns) >= self._long_window:
            long_r = returns[-self._long_window:]
            metrics["sharpe_long"] = self._compute_sharpe(long_r)

        # Sortino ratio (penalize downside only)
        if len(returns) >= self._short_window:
            arr = np.array(returns[-self._medium_window:] if len(returns) >= self._medium_window else returns)
            mean_r = float(np.mean(arr))
            downside = arr[arr < 0]
            if len(downside) >= 3:
                downside_std = float(np.std(downside, ddof=1))
                if downside_std > 1e-10:
                    metrics["sortino"] = mean_r / downside_std * math.sqrt(5040)
                else:
                    metrics["sortino"] = 5.0 if mean_r > 0 else 0.0
            else:
                metrics["sortino"] = 5.0 if mean_r > 0 else 0.0

        return metrics
