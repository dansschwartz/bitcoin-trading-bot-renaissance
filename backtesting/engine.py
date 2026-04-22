#!/usr/bin/env python3
"""
Renaissance Trading Bot - Backtesting Engine

Comprehensive backtesting framework with:
  1. Historical Replay     - replay SQLite / CSV data through signal generators
  2. Walk-Forward Validation - rolling train/test windows
  3. Monte Carlo Simulation  - shuffled-trade equity curve analysis

Uses only stdlib + numpy + pandas.  All public methods are async.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root (one level up from this file's directory)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Try to import the project's own technical-indicator engine.  If that fails
# (missing transitive deps when running standalone) we fall back to a
# self-contained signal generator embedded below.
# ---------------------------------------------------------------------------
_NATIVE_INDICATORS = False
try:
    from enhanced_technical_indicators import (
        EnhancedTechnicalIndicators,
        PriceData,
    )
    _NATIVE_INDICATORS = True
except Exception:
    pass

logger = logging.getLogger("backtesting.engine")


# ===================================================================
#  Data classes
# ===================================================================

@dataclass
class BacktestResult:
    """Complete results from a single backtest run."""

    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit_per_trade: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    total_fees_paid: float
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    monthly_returns: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo trade-order shuffling."""

    n_simulations: int
    median_final_equity: float
    percentile_5_equity: float
    percentile_95_equity: float
    median_max_drawdown: float
    worst_max_drawdown: float
    pct_profitable: float
    pct_survive_15pct_dd: float


# ===================================================================
#  Built-in signal generator (fallback when native indicators
#  are unavailable or the user has no external deps).
# ===================================================================

class _BuiltinSignalGenerator:
    """Lightweight RSI / MACD / Bollinger signal generator using pandas."""

    def __init__(
        self,
        rsi_period: int = 7,
        macd_fast: int = 5,
        macd_slow: int = 13,
        macd_signal: int = 4,
        bb_period: int = 14,
        bb_std: float = 2.0,
    ):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std

    # ------------------------------------------------------------------
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add signal columns to *df* (must contain 'close' and 'volume').

        Returns the dataframe with an added ``signal`` column whose values
        are one of ``{'BUY', 'SELL', 'HOLD'}`` and a ``signal_strength``
        column in [-1, 1].
        """
        df = df.copy()
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger(df)
        df = self._combine_signals(df)
        return df

    # ------------------------------------------------------------------
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # Signal in [-1, 1]
        df["rsi_signal"] = 0.0
        df.loc[df["rsi"] < 25, "rsi_signal"] = 1.0
        df.loc[df["rsi"] < 35, "rsi_signal"] = df.loc[df["rsi"] < 35, "rsi_signal"].where(
            df["rsi"] < 25, (35 - df["rsi"]) / 10.0 * 0.5
        )
        df.loc[df["rsi"] > 75, "rsi_signal"] = -1.0
        df.loc[df["rsi"] > 65, "rsi_signal"] = df.loc[df["rsi"] > 65, "rsi_signal"].where(
            df["rsi"] > 75, -(df["rsi"] - 65) / 10.0 * 0.5
        )
        return df

    # ------------------------------------------------------------------
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.macd_slow, adjust=False).mean()
        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal_line"] = df["macd_line"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd_line"] - df["macd_signal_line"]

        # Normalise histogram to roughly [-1, 1] via z-score clipping
        hist_std = df["macd_hist"].rolling(window=50, min_periods=10).std().replace(0, np.nan)
        df["macd_signal_val"] = (df["macd_hist"] / hist_std).clip(-2, 2) / 2.0
        df["macd_signal_val"] = df["macd_signal_val"].fillna(0.0)
        return df

    # ------------------------------------------------------------------
    def _add_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        sma = df["close"].rolling(window=self.bb_period).mean()
        std = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = sma + self.bb_std * std
        df["bb_lower"] = sma - self.bb_std * std
        df["bb_mid"] = sma

        band_width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_signal"] = (df["bb_mid"] - df["close"]) / (band_width / 2.0)
        df["bb_signal"] = df["bb_signal"].clip(-1, 1).fillna(0.0)
        return df

    # ------------------------------------------------------------------
    def _combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Weighted combination: RSI 0.35, MACD 0.35, BB 0.30."""
        rsi = df.get("rsi_signal", pd.Series(0.0, index=df.index))
        macd = df.get("macd_signal_val", pd.Series(0.0, index=df.index))
        bb = df.get("bb_signal", pd.Series(0.0, index=df.index))

        df["signal_strength"] = 0.35 * rsi + 0.35 * macd + 0.30 * bb
        df["signal_strength"] = df["signal_strength"].fillna(0.0)

        df["signal"] = "HOLD"
        df.loc[df["signal_strength"] > 0.15, "signal"] = "BUY"
        df.loc[df["signal_strength"] < -0.15, "signal"] = "SELL"
        return df


# ===================================================================
#  Backtest Engine
# ===================================================================

class BacktestEngine:
    """Comprehensive backtesting engine for the Renaissance trading bot.

    Capabilities
    ------------
    1. ``run_backtest``         -- historical data replay with fee/slippage
    2. ``walk_forward_test``    -- rolling train/test window validation
    3. ``monte_carlo_simulation`` -- shuffled-trade equity curve analysis
    """

    DB_PATH_DEFAULT = str(_PROJECT_ROOT / "data" / "renaissance_bot.db")

    def __init__(
        self,
        db_path: Optional[str] = None,
        strategy_name: str = "RenaissanceTechnical",
    ):
        self.db_path: str = db_path or self.DB_PATH_DEFAULT
        self.strategy_name = strategy_name
        self._signal_gen = _BuiltinSignalGenerator()
        logger.info(
            "BacktestEngine initialised  db=%s  native_indicators=%s",
            self.db_path,
            _NATIVE_INDICATORS,
        )

    # ------------------------------------------------------------------
    #  SQLite helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_backtest_runs_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS backtest_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                config_json     TEXT,
                total_trades    INTEGER,
                realized_pnl    REAL,
                sharpe_ratio    REAL,
                max_drawdown    REAL,
                win_rate        REAL,
                duration_seconds REAL,
                notes           TEXT
            )"""
        )
        conn.commit()

    # ------------------------------------------------------------------
    #  Data loading
    # ------------------------------------------------------------------

    async def _load_data_sqlite(
        self,
        start_date: str,
        end_date: str,
        pairs: List[str],
    ) -> pd.DataFrame:
        """Load market data from the project's SQLite database."""
        if not os.path.isfile(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        with self._connect() as conn:
            placeholders = ",".join("?" for _ in pairs)
            query = (
                f"SELECT id, price, volume, bid, ask, spread, timestamp, source, product_id "
                f"FROM market_data "
                f"WHERE product_id IN ({placeholders}) "
                f"  AND timestamp >= ? AND timestamp <= ? "
                f"ORDER BY timestamp ASC"
            )
            params: list = list(pairs) + [start_date, end_date]
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            logger.warning("No rows returned from market_data for %s [%s .. %s]", pairs, start_date, end_date)
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    async def _load_data_csv(self, csv_path: str) -> pd.DataFrame:
        """Load OHLCV data from a CSV file.

        Expected columns (case-insensitive):
            timestamp, open, high, low, close, volume
        Optional: product_id
        """
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]

        required = {"timestamp", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # Fill in optional OHLCV columns if missing
        for col in ("open", "high", "low"):
            if col not in df.columns:
                df[col] = df["close"]
        if "volume" not in df.columns:
            df["volume"] = 0.0
        if "product_id" not in df.columns:
            df["product_id"] = "BTC-USD"
        if "price" not in df.columns:
            df["price"] = df["close"]
        return df

    # ------------------------------------------------------------------
    #  Signal generation
    # ------------------------------------------------------------------

    def _generate_signals_native(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use the project's EnhancedTechnicalIndicators (if available)."""
        indicators = EnhancedTechnicalIndicators()
        signals_list: List[str] = []
        strengths: List[float] = []

        for _, row in df.iterrows():
            pd_ts = row["timestamp"]
            ts = pd_ts.to_pydatetime() if hasattr(pd_ts, "to_pydatetime") else pd_ts
            price_data = PriceData(
                timestamp=ts,
                open=float(row.get("open", row["price"])),
                high=float(row.get("high", row["price"])),
                low=float(row.get("low", row["price"])),
                close=float(row.get("close", row["price"])),
                volume=float(row.get("volume", 0)),
            )
            mtf = indicators.update_price_data(price_data)
            combined = mtf.combined_signal
            if combined > 0.15:
                sig = "BUY"
            elif combined < -0.15:
                sig = "SELL"
            else:
                sig = "HOLD"
            signals_list.append(sig)
            strengths.append(float(combined))

        df = df.copy()
        df["signal"] = signals_list
        df["signal_strength"] = strengths
        return df

    def _generate_signals_builtin(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: use the built-in RSI/MACD/Bollinger generator."""
        if "close" not in df.columns and "price" in df.columns:
            df = df.copy()
            df["close"] = df["price"]
        return self._signal_gen.generate(df)

    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if _NATIVE_INDICATORS:
            try:
                return self._generate_signals_native(df)
            except Exception as exc:
                logger.warning("Native signal generation failed (%s), falling back to built-in", exc)
        return self._generate_signals_builtin(df)

    # ------------------------------------------------------------------
    #  Core simulation loop
    # ------------------------------------------------------------------

    def _simulate_trades(
        self,
        df: pd.DataFrame,
        initial_capital: float,
        fee_rate: float,
        slippage_bps: float,
    ) -> Tuple[List[Dict], List[Dict], float]:
        """Walk through data chronologically, executing signals.

        Returns (trades, equity_curve, total_fees).
        """
        cash = initial_capital
        position_size = 0.0       # units of asset held
        entry_price = 0.0
        total_fees = 0.0
        trades: List[Dict] = []
        equity_curve: List[Dict] = []

        for idx, row in df.iterrows():
            price = float(row.get("close", row.get("price", 0)))
            if price <= 0:
                continue

            signal = row.get("signal", "HOLD")
            strength = float(row.get("signal_strength", 0))
            ts = row["timestamp"]
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

            equity = cash + position_size * price
            equity_curve.append({"timestamp": ts_str, "equity": round(equity, 4)})

            # --- BUY signal while flat ----------------------------------
            if signal == "BUY" and position_size == 0.0:
                allocation_frac = min(0.2 + abs(strength) * 0.6, 0.95)
                invest_amount = cash * allocation_frac

                slip = price * (slippage_bps / 10_000)
                fill_price = price + slip
                fee = invest_amount * (fee_rate / 2)  # half roundtrip
                units = (invest_amount - fee) / fill_price

                if units > 0:
                    position_size = units
                    entry_price = fill_price
                    cash -= invest_amount
                    total_fees += fee
                    trades.append({
                        "timestamp": ts_str,
                        "side": "BUY",
                        "price": round(fill_price, 4),
                        "units": round(units, 8),
                        "value": round(invest_amount, 4),
                        "fee": round(fee, 4),
                    })

            # --- SELL signal while holding a position -------------------
            elif signal == "SELL" and position_size > 0:
                slip = price * (slippage_bps / 10_000)
                fill_price = price - slip
                gross = position_size * fill_price
                fee = gross * (fee_rate / 2)
                net = gross - fee

                pnl = net - (position_size * entry_price)
                trades.append({
                    "timestamp": ts_str,
                    "side": "SELL",
                    "price": round(fill_price, 4),
                    "units": round(position_size, 8),
                    "value": round(gross, 4),
                    "fee": round(fee, 4),
                    "pnl": round(pnl, 4),
                })

                cash += net
                total_fees += fee
                position_size = 0.0
                entry_price = 0.0

        # Close any remaining position at last price
        if position_size > 0 and len(df) > 0:
            last_price = float(df.iloc[-1].get("close", df.iloc[-1].get("price", 0)))
            if last_price > 0:
                slip = last_price * (slippage_bps / 10_000)
                fill_price = last_price - slip
                gross = position_size * fill_price
                fee = gross * (fee_rate / 2)
                net = gross - fee

                pnl = net - (position_size * entry_price)
                last_ts = df.iloc[-1]["timestamp"]
                trades.append({
                    "timestamp": last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
                    "side": "SELL",
                    "price": round(fill_price, 4),
                    "units": round(position_size, 8),
                    "value": round(gross, 4),
                    "fee": round(fee, 4),
                    "pnl": round(pnl, 4),
                    "note": "forced_close",
                })
                cash += net
                total_fees += fee
                position_size = 0.0

        return trades, equity_curve, total_fees

    # ------------------------------------------------------------------
    #  Metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        trades: List[Dict],
        equity_curve: List[Dict],
        initial_capital: float,
        total_fees: float,
        strategy_name: str,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """Derive all performance metrics from trades and equity curve."""

        final_capital = equity_curve[-1]["equity"] if equity_curve else initial_capital
        total_return_pct = ((final_capital - initial_capital) / initial_capital * 100) if initial_capital else 0.0

        # Separate winning / losing trades (only completed round-trips)
        sell_trades = [t for t in trades if t["side"] == "SELL"]
        pnls = [t.get("pnl", 0.0) for t in sell_trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]
        total_trades = len(sell_trades)
        win_rate = (len(winning) / total_trades * 100) if total_trades else 0.0
        avg_profit = (sum(pnls) / total_trades) if total_trades else 0.0

        # Max drawdown
        max_dd, max_dd_pct = BacktestEngine._max_drawdown(equity_curve)

        # Daily returns for Sharpe / Sortino
        daily_returns = BacktestEngine._daily_returns(equity_curve)
        sharpe = BacktestEngine._sharpe_ratio(daily_returns)
        sortino = BacktestEngine._sortino_ratio(daily_returns)

        # Profit factor
        gross_profit = sum(winning) if winning else 0.0
        gross_loss = abs(sum(losing)) if losing else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        # Monthly returns
        monthly = BacktestEngine._monthly_returns(equity_curve)

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=round(initial_capital, 2),
            final_capital=round(final_capital, 2),
            total_return_pct=round(total_return_pct, 4),
            total_trades=total_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=round(win_rate, 2),
            avg_profit_per_trade=round(avg_profit, 4),
            max_drawdown=round(max_dd, 4),
            max_drawdown_pct=round(max_dd_pct, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            profit_factor=round(profit_factor, 4) if profit_factor != float("inf") else 9999.0,
            total_fees_paid=round(total_fees, 4),
            trades=trades,
            equity_curve=equity_curve,
            monthly_returns=monthly,
        )

    # -- drawdown -------------------------------------------------------

    @staticmethod
    def _max_drawdown(equity_curve: List[Dict]) -> Tuple[float, float]:
        if not equity_curve:
            return 0.0, 0.0
        equities = np.array([e["equity"] for e in equity_curve], dtype=float)
        running_max = np.maximum.accumulate(equities)
        drawdowns = running_max - equities
        max_dd = float(np.max(drawdowns))
        idx = int(np.argmax(drawdowns))
        peak = running_max[idx]
        max_dd_pct = (max_dd / peak * 100) if peak > 0 else 0.0
        return max_dd, max_dd_pct

    # -- daily returns --------------------------------------------------

    @staticmethod
    def _daily_returns(equity_curve: List[Dict]) -> np.ndarray:
        """Convert an equity curve into daily log-returns."""
        if len(equity_curve) < 2:
            return np.array([], dtype=float)

        df = pd.DataFrame(equity_curve)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["date"] = df["timestamp"].dt.date
        daily = df.groupby("date")["equity"].last()

        if len(daily) < 2:
            return np.array([], dtype=float)

        returns = daily.pct_change().dropna().values
        return np.array(returns, dtype=float)

    # -- Sharpe ---------------------------------------------------------

    @staticmethod
    def _sharpe_ratio(daily_returns: np.ndarray, risk_free_annual: float = 0.0) -> float:
        if len(daily_returns) < 2:
            return 0.0
        rf_daily = risk_free_annual / 252
        excess = daily_returns - rf_daily
        mean = float(np.mean(excess))
        std = float(np.std(excess, ddof=1))
        if std == 0:
            return 0.0
        return mean / std * math.sqrt(252)

    # -- Sortino --------------------------------------------------------

    @staticmethod
    def _sortino_ratio(daily_returns: np.ndarray, risk_free_annual: float = 0.0) -> float:
        if len(daily_returns) < 2:
            return 0.0
        rf_daily = risk_free_annual / 252
        excess = daily_returns - rf_daily
        mean = float(np.mean(excess))
        downside = excess[excess < 0]
        if len(downside) == 0:
            return 0.0 if mean <= 0 else 9999.0
        downside_std = float(np.std(downside, ddof=1))
        if downside_std == 0:
            return 0.0
        return mean / downside_std * math.sqrt(252)

    # -- monthly returns ------------------------------------------------

    @staticmethod
    def _monthly_returns(equity_curve: List[Dict]) -> List[Dict]:
        if len(equity_curve) < 2:
            return []
        df = pd.DataFrame(equity_curve)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["month"] = df["timestamp"].dt.to_period("M")
        monthly = df.groupby("month")["equity"].agg(["first", "last"])
        results = []
        for period, row in monthly.iterrows():
            ret = ((row["last"] - row["first"]) / row["first"] * 100) if row["first"] != 0 else 0.0
            results.append({"month": str(period), "return_pct": round(ret, 4)})
        return results

    # ------------------------------------------------------------------
    #  1.  Historical Replay
    # ------------------------------------------------------------------

    async def run_backtest(
        self,
        start_date: str,
        end_date: str,
        pairs: List[str],
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.006,
        slippage_bps: float = 5.0,
        csv_path: Optional[str] = None,
    ) -> BacktestResult:
        """Run a full historical replay backtest.

        Parameters
        ----------
        start_date : str
            ISO-format start date, e.g. ``"2025-01-01"``.
        end_date : str
            ISO-format end date, e.g. ``"2025-12-31"``.
        pairs : list[str]
            Trading pairs to include, e.g. ``["BTC-USD", "ETH-USD"]``.
        initial_capital : float
            Starting cash balance.
        fee_rate : float
            Round-trip fee rate (default 0.6 %).
        slippage_bps : float
            Slippage in basis points per side.
        csv_path : str, optional
            Path to a CSV file.  If provided the SQLite database is skipped.

        Returns
        -------
        BacktestResult
        """
        t0 = time.monotonic()
        logger.info(
            "Starting backtest  pairs=%s  %s -> %s  capital=%.2f  fee=%.4f  slip=%.1f bps",
            pairs, start_date, end_date, initial_capital, fee_rate, slippage_bps,
        )

        # --- Load data --------------------------------------------------
        if csv_path:
            df = await self._load_data_csv(csv_path)
            # Filter by date range
            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
            if pairs:
                df = df[df["product_id"].isin(pairs)]
        else:
            df = await self._load_data_sqlite(start_date, end_date, pairs)

        if df.empty:
            logger.warning("No data available for the requested period.")
            return BacktestResult(
                strategy_name=self.strategy_name,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=initial_capital,
                total_return_pct=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_profit_per_trade=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                profit_factor=0.0,
                total_fees_paid=0.0,
            )

        # Ensure close column exists
        if "close" not in df.columns and "price" in df.columns:
            df["close"] = df["price"]

        # If multiple pairs, backtest each and aggregate
        unique_pairs = df["product_id"].unique() if "product_id" in df.columns else ["BTC-USD"]
        if len(unique_pairs) > 1:
            return await self._multi_pair_backtest(
                df, unique_pairs, initial_capital, fee_rate, slippage_bps, start_date, end_date, t0,
            )

        # --- Single-pair fast path --------------------------------------
        df = self._generate_signals(df)

        trades, equity_curve, total_fees = self._simulate_trades(
            df, initial_capital, fee_rate, slippage_bps,
        )

        result = self._compute_metrics(
            trades, equity_curve, initial_capital, total_fees,
            self.strategy_name, start_date, end_date,
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "Backtest complete in %.2fs  trades=%d  return=%.2f%%  sharpe=%.2f  maxDD=%.2f%%",
            elapsed, result.total_trades, result.total_return_pct, result.sharpe_ratio, result.max_drawdown_pct,
        )

        # Persist to database
        await self._save_result(result, elapsed)
        return result

    # ------------------------------------------------------------------

    async def _multi_pair_backtest(
        self,
        df: pd.DataFrame,
        pairs: np.ndarray,
        initial_capital: float,
        fee_rate: float,
        slippage_bps: float,
        start_date: str,
        end_date: str,
        t0: float,
    ) -> BacktestResult:
        """Run separate backtests per pair with equal capital allocation, then merge."""
        per_pair_capital = initial_capital / len(pairs)
        all_trades: List[Dict] = []
        all_equity: List[Dict] = []
        total_fees = 0.0

        for pair in pairs:
            pair_df = df[df["product_id"] == pair].copy().reset_index(drop=True)
            if pair_df.empty:
                continue
            pair_df = self._generate_signals(pair_df)
            trades, eq, fees = self._simulate_trades(pair_df, per_pair_capital, fee_rate, slippage_bps)
            for t in trades:
                t["product_id"] = str(pair)
            all_trades.extend(trades)
            all_equity.extend(eq)
            total_fees += fees

        # Sort combined equity by timestamp for unified curve
        all_equity.sort(key=lambda e: e["timestamp"])

        # Build unified equity curve by summing per-pair equities at each timestamp
        if all_equity:
            eq_df = pd.DataFrame(all_equity)
            eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], utc=True, errors="coerce")
            eq_df = eq_df.dropna(subset=["timestamp"]).sort_values("timestamp")
            # Use the last equity snapshot per timestamp as approximation
            unified = eq_df.groupby("timestamp")["equity"].sum().reset_index()
            unified_curve = [
                {"timestamp": row["timestamp"].isoformat(), "equity": round(row["equity"], 4)}
                for _, row in unified.iterrows()
            ]
        else:
            unified_curve = []

        result = self._compute_metrics(
            all_trades, unified_curve, initial_capital, total_fees,
            self.strategy_name, start_date, end_date,
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "Multi-pair backtest complete in %.2fs  pairs=%d  trades=%d  return=%.2f%%",
            elapsed, len(pairs), result.total_trades, result.total_return_pct,
        )
        await self._save_result(result, elapsed)
        return result

    # ------------------------------------------------------------------
    #  2.  Walk-Forward Validation
    # ------------------------------------------------------------------

    async def walk_forward_test(
        self,
        pairs: List[str],
        total_months: int = 12,
        train_months: int = 8,
        test_months: int = 2,
        step_months: int = 2,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.006,
        slippage_bps: float = 5.0,
        csv_path: Optional[str] = None,
    ) -> List[BacktestResult]:
        """Rolling walk-forward validation.

        Starting from *total_months* ago, the engine creates overlapping
        windows of (train_months + test_months), stepping forward by
        step_months each iteration.  Each window's backtest is run on the
        **test** slice only (the train slice exists for future optimisation
        hooks).

        The strategy is considered robust if the *majority* of windows
        are profitable.

        Parameters
        ----------
        pairs : list[str]
            Trading pairs.
        total_months : int
            How far back (in months) to start from today.
        train_months, test_months, step_months : int
            Window geometry.
        initial_capital, fee_rate, slippage_bps :
            Forwarded to ``run_backtest``.
        csv_path : str, optional
            Optional CSV override.

        Returns
        -------
        list[BacktestResult]
            One result per test window.
        """
        logger.info(
            "Walk-forward test  total=%d  train=%d  test=%d  step=%d months",
            total_months, train_months, test_months, step_months,
        )

        now = datetime.now(timezone.utc)
        origin = now - timedelta(days=total_months * 30)

        window_size = train_months + test_months
        results: List[BacktestResult] = []
        window_idx = 0

        while True:
            win_start = origin + timedelta(days=window_idx * step_months * 30)
            test_start = win_start + timedelta(days=train_months * 30)
            test_end = test_start + timedelta(days=test_months * 30)

            if test_end > now:
                break

            start_str = test_start.strftime("%Y-%m-%d")
            end_str = test_end.strftime("%Y-%m-%d")

            logger.info("  Window %d: test %s -> %s", window_idx, start_str, end_str)

            result = await self.run_backtest(
                start_date=start_str,
                end_date=end_str,
                pairs=pairs,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                slippage_bps=slippage_bps,
                csv_path=csv_path,
            )
            results.append(result)
            window_idx += 1

        # Summary
        if results:
            profitable = sum(1 for r in results if r.total_return_pct > 0)
            total = len(results)
            logger.info(
                "Walk-forward complete: %d/%d windows profitable (%.0f%%)",
                profitable, total, profitable / total * 100,
            )
            if profitable > total / 2:
                logger.info("PASS - strategy profitable in majority of windows")
            else:
                logger.warning("FAIL - strategy NOT profitable in majority of windows")
        else:
            logger.warning("Walk-forward produced zero windows (insufficient date range?)")

        return results

    # ------------------------------------------------------------------
    #  3.  Monte Carlo Simulation
    # ------------------------------------------------------------------

    async def monte_carlo_simulation(
        self,
        trades: List[Dict],
        n_simulations: int = 1000,
        initial_capital: float = 10_000.0,
    ) -> MonteCarloResult:
        """Shuffle trade order and replay to build equity-curve distributions.

        Parameters
        ----------
        trades : list[dict]
            Trade records from a backtest (must contain ``pnl`` key on SELL
            trades).
        n_simulations : int
            Number of random permutations.
        initial_capital : float
            Starting equity for each simulation.

        Returns
        -------
        MonteCarloResult
        """
        logger.info("Monte Carlo simulation  n=%d  trades=%d  capital=%.2f", n_simulations, len(trades), initial_capital)

        # Extract PnLs from completed round-trips (sell trades)
        pnls = [t.get("pnl", 0.0) for t in trades if t.get("side") == "SELL"]
        if not pnls:
            logger.warning("No completed trades to simulate")
            return MonteCarloResult(
                n_simulations=n_simulations,
                median_final_equity=initial_capital,
                percentile_5_equity=initial_capital,
                percentile_95_equity=initial_capital,
                median_max_drawdown=0.0,
                worst_max_drawdown=0.0,
                pct_profitable=0.0,
                pct_survive_15pct_dd=100.0,
            )

        final_equities: List[float] = []
        max_drawdowns: List[float] = []

        rng = random.Random(42)  # deterministic seed for reproducibility

        for _ in range(n_simulations):
            shuffled = pnls.copy()
            rng.shuffle(shuffled)

            equity = initial_capital
            peak = equity
            worst_dd_pct = 0.0

            for pnl in shuffled:
                equity += pnl
                if equity > peak:
                    peak = equity
                dd_pct = ((peak - equity) / peak * 100) if peak > 0 else 0.0
                if dd_pct > worst_dd_pct:
                    worst_dd_pct = dd_pct

            final_equities.append(equity)
            max_drawdowns.append(worst_dd_pct)

        fe = np.array(final_equities)
        dd = np.array(max_drawdowns)

        result = MonteCarloResult(
            n_simulations=n_simulations,
            median_final_equity=round(float(np.median(fe)), 2),
            percentile_5_equity=round(float(np.percentile(fe, 5)), 2),
            percentile_95_equity=round(float(np.percentile(fe, 95)), 2),
            median_max_drawdown=round(float(np.median(dd)), 4),
            worst_max_drawdown=round(float(np.max(dd)), 4),
            pct_profitable=round(float(np.sum(fe > initial_capital) / n_simulations * 100), 2),
            pct_survive_15pct_dd=round(float(np.sum(dd < 15) / n_simulations * 100), 2),
        )

        logger.info(
            "Monte Carlo results:  median_equity=%.2f  5th=%.2f  95th=%.2f  "
            "pct_profitable=%.1f%%  survive_15dd=%.1f%%",
            result.median_final_equity, result.percentile_5_equity,
            result.percentile_95_equity, result.pct_profitable,
            result.pct_survive_15pct_dd,
        )

        return result

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    async def _save_result(self, result: BacktestResult, duration_seconds: float) -> None:
        """Persist a backtest result to the ``backtest_runs`` table."""
        try:
            with self._connect() as conn:
                self._ensure_backtest_runs_table(conn)

                realized_pnl = result.final_capital - result.initial_capital

                config_json = json.dumps({
                    "strategy": result.strategy_name,
                    "start_date": result.start_date,
                    "end_date": result.end_date,
                    "initial_capital": result.initial_capital,
                    "total_return_pct": result.total_return_pct,
                    "winning_trades": result.winning_trades,
                    "losing_trades": result.losing_trades,
                    "sortino_ratio": result.sortino_ratio,
                    "profit_factor": result.profit_factor,
                    "total_fees_paid": result.total_fees_paid,
                })

                conn.execute(
                    """INSERT INTO backtest_runs
                       (timestamp, config_json, total_trades, realized_pnl,
                        sharpe_ratio, max_drawdown, win_rate, duration_seconds, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        config_json,
                        result.total_trades,
                        round(realized_pnl, 4),
                        result.sharpe_ratio,
                        result.max_drawdown_pct,
                        result.win_rate,
                        round(duration_seconds, 2),
                        f"Return: {result.total_return_pct:.2f}%",
                    ),
                )
                conn.commit()
                logger.info("Backtest result saved to database")
        except Exception as exc:
            logger.error("Failed to save backtest result: %s", exc)

    # ------------------------------------------------------------------
    #  Pretty-print summary
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(result: BacktestResult) -> None:
        """Print a formatted results summary to stdout."""
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  BACKTEST RESULTS  --  {result.strategy_name}")
        print(sep)
        print(f"  Period            : {result.start_date}  ->  {result.end_date}")
        print(f"  Initial Capital   : ${result.initial_capital:>12,.2f}")
        print(f"  Final Capital     : ${result.final_capital:>12,.2f}")
        print(f"  Total Return      :  {result.total_return_pct:>+11.2f}%")
        print(f"  {'-' * 58}")
        print(f"  Total Trades      :  {result.total_trades:>11d}")
        print(f"  Winning Trades    :  {result.winning_trades:>11d}")
        print(f"  Losing Trades     :  {result.losing_trades:>11d}")
        print(f"  Win Rate          :  {result.win_rate:>11.2f}%")
        print(f"  Avg Profit/Trade  : ${result.avg_profit_per_trade:>12.4f}")
        print(f"  {'-' * 58}")
        print(f"  Max Drawdown      : ${result.max_drawdown:>12.4f}")
        print(f"  Max Drawdown %    :  {result.max_drawdown_pct:>11.4f}%")
        print(f"  Sharpe Ratio      :  {result.sharpe_ratio:>11.4f}")
        print(f"  Sortino Ratio     :  {result.sortino_ratio:>11.4f}")
        print(f"  Profit Factor     :  {result.profit_factor:>11.4f}")
        print(f"  Total Fees Paid   : ${result.total_fees_paid:>12.4f}")
        print(sep)

        if result.monthly_returns:
            print(f"\n  Monthly Returns:")
            for mr in result.monthly_returns:
                bar_len = int(abs(mr['return_pct']) / 2)
                bar_char = "+" if mr['return_pct'] >= 0 else "-"
                bar = bar_char * min(bar_len, 40)
                print(f"    {mr['month']:>7s}  {mr['return_pct']:>+8.2f}%  {bar}")
            print()

    @staticmethod
    def print_monte_carlo_summary(mc: MonteCarloResult) -> None:
        """Print Monte Carlo simulation results."""
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  MONTE CARLO SIMULATION  ({mc.n_simulations:,} runs)")
        print(sep)
        print(f"  Median Final Equity  : ${mc.median_final_equity:>12,.2f}")
        print(f"  5th Percentile       : ${mc.percentile_5_equity:>12,.2f}")
        print(f"  95th Percentile      : ${mc.percentile_95_equity:>12,.2f}")
        print(f"  {'-' * 58}")
        print(f"  Median Max Drawdown  :  {mc.median_max_drawdown:>11.2f}%")
        print(f"  Worst Max Drawdown   :  {mc.worst_max_drawdown:>11.2f}%")
        print(f"  {'-' * 58}")
        print(f"  % Profitable         :  {mc.pct_profitable:>11.2f}%")
        print(f"  % Survive <15% DD    :  {mc.pct_survive_15pct_dd:>11.2f}%")
        print(sep)
        print()


# ===================================================================
#  CLI entry point
# ===================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Renaissance Trading Bot - Backtesting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m backtesting.engine --start 2025-01-01 --end 2025-06-30 --pairs BTC-USD ETH-USD
  python -m backtesting.engine --csv data/btc_ohlcv.csv --start 2025-01-01 --end 2025-12-31
  python -m backtesting.engine --walk-forward --pairs BTC-USD --total-months 12
  python -m backtesting.engine --start 2025-01-01 --end 2025-12-31 --pairs BTC-USD --monte-carlo
""",
    )

    parser.add_argument("--start", default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--pairs", nargs="+", default=["BTC-USD"], help="Trading pairs")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--fee-rate", type=float, default=0.006, help="Round-trip fee rate")
    parser.add_argument("--slippage", type=float, default=5.0, help="Slippage in basis points")
    parser.add_argument("--csv", default=None, help="Path to CSV file (overrides SQLite)")
    parser.add_argument("--db", default=None, help="Path to SQLite database")
    parser.add_argument("--strategy", default="RenaissanceTechnical", help="Strategy name label")

    # Walk-forward
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--total-months", type=int, default=12, help="Walk-forward: total months back")
    parser.add_argument("--train-months", type=int, default=8, help="Walk-forward: training window months")
    parser.add_argument("--test-months", type=int, default=2, help="Walk-forward: test window months")
    parser.add_argument("--step-months", type=int, default=2, help="Walk-forward: step size months")

    # Monte Carlo
    parser.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo simulation after backtest")
    parser.add_argument("--mc-simulations", type=int, default=1000, help="Number of Monte Carlo simulations")

    # Verbosity
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")

    return parser.parse_args()


async def _async_main() -> None:
    args = _parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    engine = BacktestEngine(db_path=args.db, strategy_name=args.strategy)

    if args.walk_forward:
        # ---- Walk-Forward Validation -----------------------------------
        results = await engine.walk_forward_test(
            pairs=args.pairs,
            total_months=args.total_months,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            initial_capital=args.capital,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage,
            csv_path=args.csv,
        )
        for i, r in enumerate(results):
            print(f"\n--- Window {i} ---")
            BacktestEngine.print_summary(r)

    else:
        # ---- Standard Historical Replay --------------------------------
        result = await engine.run_backtest(
            start_date=args.start,
            end_date=args.end,
            pairs=args.pairs,
            initial_capital=args.capital,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage,
            csv_path=args.csv,
        )
        BacktestEngine.print_summary(result)

        # ---- Optional Monte Carlo --------------------------------------
        if args.monte_carlo and result.trades:
            mc = await engine.monte_carlo_simulation(
                trades=result.trades,
                n_simulations=args.mc_simulations,
                initial_capital=args.capital,
            )
            BacktestEngine.print_monte_carlo_summary(mc)


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
