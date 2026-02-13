"""Trading strategies and backtest engine.

Includes:
- SimMeanReversionStrategy  (z-score based)
- SimContrarianScanner      (non-intuitive signals)
- SimBacktestEngine         (walk-forward P&L with costs)
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from sim_config import BacktestResult, Trade, DEFAULT_CONFIG
from sim_transaction_costs import SimTransactionCostModel


# ======================================================================
# Mean Reversion Strategy
# ======================================================================

class SimMeanReversionStrategy:
    """Z-score mean reversion: enter at +/- entry_z, exit at exit_z."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or DEFAULT_CONFIG.get("strategies", {}).get("mean_reversion", {})
        self.entry_z = cfg.get("entry_z", 2.0)
        self.exit_z = cfg.get("exit_z", 0.0)
        self.lookback = cfg.get("lookback", 60)

    def generate_signals(self, prices: np.ndarray) -> np.ndarray:
        """Return signal array: +1 (buy), -1 (sell/short), 0 (hold).

        Uses rolling z-score of price relative to its lookback mean/std.
        """
        n = len(prices)
        signals = np.zeros(n)
        if n < self.lookback + 1:
            return signals

        s = pd.Series(prices)
        rolling_mean = s.rolling(self.lookback, min_periods=self.lookback).mean().values
        rolling_std = s.rolling(self.lookback, min_periods=self.lookback).std().values

        position = 0  # 0=flat, 1=long, -1=short
        for i in range(self.lookback, n):
            std = rolling_std[i]
            if std < 1e-12:
                continue
            z = (prices[i] - rolling_mean[i]) / std

            if position == 0:
                if z < -self.entry_z:
                    signals[i] = 1.0   # buy (mean will revert up)
                    position = 1
                elif z > self.entry_z:
                    signals[i] = -1.0  # sell / short (mean will revert down)
                    position = -1
            elif position == 1:
                if z >= -self.exit_z:
                    signals[i] = -1.0  # close long
                    position = 0
            elif position == -1:
                if z <= self.exit_z:
                    signals[i] = 1.0   # close short
                    position = 0

        return signals


# ======================================================================
# Contrarian Scanner
# ======================================================================

class SimContrarianScanner:
    """Non-intuitive signal scanner: buy after N consecutive down days,
    sell after N consecutive up days."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or DEFAULT_CONFIG.get("strategies", {}).get("contrarian_scanner", {})
        self.min_consecutive = cfg.get("min_consecutive", 3)

    def generate_signals(self, prices: np.ndarray) -> np.ndarray:
        n = len(prices)
        signals = np.zeros(n)
        if n < self.min_consecutive + 2:
            return signals

        returns = np.diff(np.log(np.maximum(prices, 1e-9)))

        for i in range(self.min_consecutive, len(returns)):
            recent = returns[i - self.min_consecutive: i]
            if np.all(recent < 0):
                signals[i + 1] = 1.0   # contrarian buy
            elif np.all(recent > 0):
                signals[i + 1] = -1.0  # contrarian sell

        return signals


# ======================================================================
# Backtest Engine
# ======================================================================

class SimBacktestEngine:
    """Walk-forward backtest with transaction costs and per-regime breakdown."""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 cost_model: Optional[SimTransactionCostModel] = None,
                 logger: Optional[logging.Logger] = None):
        cfg = config or DEFAULT_CONFIG.get("backtest", {})
        self.logger = logger or logging.getLogger(__name__)
        self.initial_capital = cfg.get("initial_capital", 100_000.0)
        self.position_fraction = cfg.get("position_fraction", 0.25)
        self.cost_model = cost_model or SimTransactionCostModel(
            DEFAULT_CONFIG.get("transaction_costs", {})
        )

    def run_backtest(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        asset: str = "SIM",
        strategy_name: str = "strategy",
        volumes: Optional[np.ndarray] = None,
        regimes: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """Execute backtest and return full result with metrics."""
        n = len(prices)
        equity = np.zeros(n)
        equity[0] = self.initial_capital
        cash = self.initial_capital
        position = 0.0          # units of asset held
        trades: List[Trade] = []

        if volumes is None:
            volumes = np.full(n, 1e9)

        for i in range(1, n):
            # Mark to market
            equity[i] = cash + position * prices[i]

            sig = signals[i] if i < len(signals) else 0.0
            if sig == 0.0:
                continue

            trade_value = equity[i] * self.position_fraction

            if sig > 0 and position <= 0:
                # Buy / close short
                units = trade_value / prices[i]
                vol = self._rolling_vol(prices, i)
                regime = self._regime_label(regimes, i)
                cost = self.cost_model.calculate_cost(
                    trade_size_usd=trade_value,
                    price=prices[i],
                    volatility=vol,
                    daily_volume=volumes[i],
                    regime=regime,
                )
                cash -= (units * prices[i] + cost.total)
                position += units
                trades.append(Trade(
                    timestamp_idx=i, side="buy", price=prices[i],
                    size_usd=trade_value, cost=cost.total,
                    signal_value=sig,
                ))

            elif sig < 0 and position >= 0:
                # Sell / open short
                if position > 0:
                    sell_value = position * prices[i]
                    vol = self._rolling_vol(prices, i)
                    regime = self._regime_label(regimes, i)
                    cost = self.cost_model.calculate_cost(
                        trade_size_usd=sell_value,
                        price=prices[i],
                        volatility=vol,
                        daily_volume=volumes[i],
                        regime=regime,
                    )
                    cash += sell_value - cost.total
                    trades.append(Trade(
                        timestamp_idx=i, side="sell", price=prices[i],
                        size_usd=sell_value, cost=cost.total,
                        signal_value=sig,
                    ))
                    position = 0.0

        # Final mark-to-market
        equity[-1] = cash + position * prices[-1]

        # Compute returns
        returns = np.diff(equity) / np.maximum(equity[:-1], 1e-9)

        # Fill PnL on trades
        self._fill_trade_pnl(trades, prices)

        metrics = self.compute_metrics(equity, trades, returns)

        # Per-regime breakdown
        regime_perf: Dict[str, Dict[str, float]] = {}
        if regimes is not None:
            regime_perf = self._regime_breakdown(returns, regimes[1:], equity)

        return BacktestResult(
            strategy_name=strategy_name,
            asset=asset,
            equity_curve=equity,
            returns=returns,
            trades=trades,
            metrics=metrics,
            regime_performance=regime_perf,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, equity: np.ndarray,
                        trades: List[Trade],
                        returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        if returns is None:
            returns = np.diff(equity) / np.maximum(equity[:-1], 1e-9)

        total_return = (equity[-1] / equity[0]) - 1.0 if equity[0] > 0 else 0.0
        ann_return = (1 + total_return) ** (252 / max(len(returns), 1)) - 1.0

        std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 1e-9
        sharpe = float(np.mean(returns)) / max(std, 1e-9) * np.sqrt(252)

        downside = returns[returns < 0]
        down_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-9
        sortino = float(np.mean(returns)) / max(down_std, 1e-9) * np.sqrt(252)

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / np.maximum(running_max, 1e-9)
        max_dd = float(np.min(drawdowns))

        calmar = ann_return / max(abs(max_dd), 1e-9) if max_dd != 0 else 0.0

        # Trade stats
        n_trades = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / max(n_trades, 1)
        avg_win = float(np.mean([w.pnl for w in wins])) if wins else 0.0
        avg_loss = float(np.mean([l.pnl for l in losses])) if losses else 0.0
        gross_profit = sum(w.pnl for w in wins)
        gross_loss = abs(sum(l.pnl for l in losses))
        profit_factor = gross_profit / max(gross_loss, 1e-9)
        total_costs = sum(t.cost for t in trades)

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_costs": total_costs,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_vol(prices: np.ndarray, i: int, window: int = 20) -> float:
        start = max(0, i - window)
        if i - start < 2:
            return 0.02
        rets = np.diff(np.log(np.maximum(prices[start:i + 1], 1e-9)))
        return float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.02

    @staticmethod
    def _regime_label(regimes: Optional[np.ndarray], i: int) -> str:
        if regimes is None or i >= len(regimes):
            return "normal"
        r = regimes[i]
        if isinstance(r, str):
            return r
        return "normal"

    @staticmethod
    def _fill_trade_pnl(trades: List[Trade], prices: np.ndarray) -> None:
        """Fill PnL on each trade pair (buy then sell)."""
        buy_price = None
        for t in trades:
            if t.side == "buy":
                buy_price = t.price
            elif t.side == "sell" and buy_price is not None:
                t.pnl = (t.price - buy_price) * (t.size_usd / t.price) - t.cost
                buy_price = None

    def _regime_breakdown(self, returns: np.ndarray,
                          regimes: np.ndarray,
                          equity: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute per-regime performance metrics."""
        result: Dict[str, Dict[str, float]] = {}
        unique_regimes = np.unique(regimes[:len(returns)])
        for r in unique_regimes:
            label = str(r)
            mask = regimes[:len(returns)] == r
            r_returns = returns[mask]
            if len(r_returns) < 2:
                continue
            std = float(np.std(r_returns, ddof=1))
            result[label] = {
                "mean_return": float(np.mean(r_returns)),
                "std": std,
                "sharpe": float(np.mean(r_returns)) / max(std, 1e-9) * np.sqrt(252),
                "n_periods": int(mask.sum()),
                "total_return": float(np.sum(r_returns)),
            }
        return result
