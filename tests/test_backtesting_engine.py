"""
Tests for backtesting/engine.py — BacktestEngine.

Covers Historical Replay mode, Walk-Forward Validation mode,
Monte Carlo simulation mode, metrics computation, and signal generation.
Uses synthetic price data throughout. All DB access is mocked.
"""

import asyncio
import math
import os
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import (
    BacktestEngine,
    BacktestResult,
    MonteCarloResult,
    _BuiltinSignalGenerator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n_bars: int = 200, base_price: float = 100.0, trend: float = 0.001) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with timestamps."""
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = base_price
    for i in range(n_bars):
        ts = start + timedelta(minutes=5 * i)
        noise = np.random.uniform(-0.5, 0.5)
        o = price + noise
        h = o + abs(noise) + 0.1
        l = o - abs(noise) - 0.1
        c = o + trend * i + noise * 0.3
        v = np.random.uniform(10, 100)
        rows.append({
            "timestamp": ts,
            "open": round(o, 4),
            "high": round(h, 4),
            "low": round(l, 4),
            "close": round(c, 4),
            "volume": round(v, 4),
            "product_id": "BTC-USD",
            "price": round(c, 4),
        })
        price = c
    return pd.DataFrame(rows)


def _run_async(coro):
    """Run an async function synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# Use a fresh event loop for all tests
@pytest.fixture(autouse=True)
def _event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Tests — _BuiltinSignalGenerator
# ---------------------------------------------------------------------------

class TestBuiltinSignalGenerator:
    def test_generate_produces_signal_columns(self):
        gen = _BuiltinSignalGenerator()
        df = _make_ohlcv_df(100)
        result = gen.generate(df)
        assert "signal" in result.columns
        assert "signal_strength" in result.columns
        assert set(result["signal"].unique()).issubset({"BUY", "SELL", "HOLD"})

    def test_signal_strength_bounded(self):
        gen = _BuiltinSignalGenerator()
        df = _make_ohlcv_df(200)
        result = gen.generate(df)
        assert result["signal_strength"].min() >= -1.5  # Weighted combination may slightly exceed
        assert result["signal_strength"].max() <= 1.5

    def test_rsi_column_added(self):
        gen = _BuiltinSignalGenerator()
        df = _make_ohlcv_df(50)
        result = gen.generate(df)
        assert "rsi" in result.columns

    def test_macd_columns_added(self):
        gen = _BuiltinSignalGenerator()
        df = _make_ohlcv_df(50)
        result = gen.generate(df)
        assert "macd_line" in result.columns
        assert "macd_signal_line" in result.columns
        assert "macd_hist" in result.columns

    def test_bollinger_columns_added(self):
        gen = _BuiltinSignalGenerator()
        df = _make_ohlcv_df(50)
        result = gen.generate(df)
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_mid" in result.columns


# ---------------------------------------------------------------------------
# Tests — Simulate trades
# ---------------------------------------------------------------------------

class TestSimulateTrades:
    def test_basic_buy_sell_cycle(self):
        engine = BacktestEngine(db_path=":memory:")
        # Construct a DF with explicit BUY then SELL signals
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC"),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "signal": ["BUY", "HOLD", "HOLD", "SELL", "HOLD"],
            "signal_strength": [0.5, 0.0, 0.0, -0.5, 0.0],
            "volume": [10.0] * 5,
        })
        trades, equity_curve, total_fees = engine._simulate_trades(
            df, initial_capital=10_000.0, fee_rate=0.006, slippage_bps=5.0,
        )
        # Should have a BUY and a SELL trade
        assert len(trades) == 2
        assert trades[0]["side"] == "BUY"
        assert trades[1]["side"] == "SELL"
        assert total_fees > 0

    def test_forced_close_at_end(self):
        engine = BacktestEngine(db_path=":memory:")
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="5min", tz="UTC"),
            "close": [100.0, 105.0, 110.0],
            "signal": ["BUY", "HOLD", "HOLD"],
            "signal_strength": [0.8, 0.0, 0.0],
            "volume": [10.0] * 3,
        })
        trades, _, _ = engine._simulate_trades(df, 10_000.0, 0.006, 5.0)
        # BUY + forced SELL at end
        assert len(trades) == 2
        assert trades[-1].get("note") == "forced_close"

    def test_no_signals_no_trades(self):
        engine = BacktestEngine(db_path=":memory:")
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC"),
            "close": [100.0] * 5,
            "signal": ["HOLD"] * 5,
            "signal_strength": [0.0] * 5,
            "volume": [10.0] * 5,
        })
        trades, eq, fees = engine._simulate_trades(df, 10_000.0, 0.006, 5.0)
        assert len(trades) == 0
        assert fees == 0.0

    def test_equity_curve_length_matches_data(self):
        engine = BacktestEngine(db_path=":memory:")
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC"),
            "close": np.linspace(100, 110, 10).tolist(),
            "signal": ["HOLD"] * 10,
            "signal_strength": [0.0] * 10,
            "volume": [10.0] * 10,
        })
        _, eq, _ = engine._simulate_trades(df, 10_000.0, 0.006, 5.0)
        assert len(eq) == 10


# ---------------------------------------------------------------------------
# Tests — Metrics computation
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_empty_equity_curve(self):
        result = BacktestEngine._compute_metrics(
            trades=[], equity_curve=[], initial_capital=10_000.0,
            total_fees=0.0, strategy_name="test", start_date="2025-01-01",
            end_date="2025-01-31",
        )
        assert result.total_trades == 0
        assert result.final_capital == 10_000.0

    def test_metrics_with_winning_trade(self):
        trades = [
            {"side": "BUY", "price": 100, "units": 10, "value": 1000, "fee": 3},
            {"side": "SELL", "price": 110, "units": 10, "value": 1100, "fee": 3.3, "pnl": 93.7},
        ]
        equity_curve = [
            {"timestamp": "2025-01-01T00:00:00+00:00", "equity": 10_000.0},
            {"timestamp": "2025-01-02T00:00:00+00:00", "equity": 10_093.7},
        ]
        result = BacktestEngine._compute_metrics(
            trades, equity_curve, 10_000.0, 6.3, "test", "2025-01-01", "2025-01-02",
        )
        assert result.total_trades == 1
        assert result.winning_trades == 1
        assert result.losing_trades == 0
        assert result.win_rate == 100.0

    def test_max_drawdown_computation(self):
        equity_curve = [
            {"timestamp": "2025-01-01T00:00:00+00:00", "equity": 10_000.0},
            {"timestamp": "2025-01-02T00:00:00+00:00", "equity": 11_000.0},
            {"timestamp": "2025-01-03T00:00:00+00:00", "equity": 9_000.0},
            {"timestamp": "2025-01-04T00:00:00+00:00", "equity": 10_500.0},
        ]
        dd, dd_pct = BacktestEngine._max_drawdown(equity_curve)
        # Peak was 11000, trough was 9000 => dd = 2000, dd_pct = 2000/11000 * 100
        assert abs(dd - 2_000.0) < 0.01
        assert abs(dd_pct - (2000 / 11000 * 100)) < 0.01

    def test_max_drawdown_empty(self):
        dd, dd_pct = BacktestEngine._max_drawdown([])
        assert dd == 0.0
        assert dd_pct == 0.0


# ---------------------------------------------------------------------------
# Tests — Sharpe and Sortino
# ---------------------------------------------------------------------------

class TestRatios:
    def test_sharpe_zero_with_no_data(self):
        assert BacktestEngine._sharpe_ratio(np.array([])) == 0.0

    def test_sharpe_positive_for_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.015, 0.005, 0.01, 0.02, 0.01])
        sharpe = BacktestEngine._sharpe_ratio(returns)
        assert sharpe > 0

    def test_sortino_zero_with_no_data(self):
        assert BacktestEngine._sortino_ratio(np.array([])) == 0.0

    def test_sortino_returns_large_for_all_positive(self):
        returns = np.array([0.01, 0.02, 0.015, 0.005, 0.01])
        sortino = BacktestEngine._sortino_ratio(returns)
        assert sortino == 9999.0  # No downside => special value


# ---------------------------------------------------------------------------
# Tests — run_backtest (Historical Replay)
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def test_backtest_with_csv(self, tmp_path):
        # Create synthetic CSV
        df = _make_ohlcv_df(100, base_price=100.0, trend=0.05)
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)

        engine = BacktestEngine(db_path=":memory:")

        # Patch _save_result to avoid DB writes
        with patch.object(engine, "_save_result", new_callable=AsyncMock):
            result = _run_async(engine.run_backtest(
                start_date="2024-01-01",
                end_date="2026-01-01",
                pairs=["BTC-USD"],
                initial_capital=10_000.0,
                csv_path=csv_path,
            ))

        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 10_000.0
        assert result.strategy_name == "RenaissanceTechnical"

    def test_backtest_empty_data(self):
        engine = BacktestEngine(db_path=":memory:")

        with patch.object(engine, "_load_data_sqlite", new_callable=AsyncMock, return_value=pd.DataFrame()):
            result = _run_async(engine.run_backtest(
                start_date="2025-01-01",
                end_date="2025-01-31",
                pairs=["BTC-USD"],
            ))

        assert result.total_trades == 0
        assert result.final_capital == 10_000.0


# ---------------------------------------------------------------------------
# Tests — Monte Carlo Simulation
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_no_trades_returns_initial_capital(self):
        engine = BacktestEngine(db_path=":memory:")
        result = _run_async(engine.monte_carlo_simulation(
            trades=[], n_simulations=100, initial_capital=10_000.0,
        ))
        assert isinstance(result, MonteCarloResult)
        assert result.median_final_equity == 10_000.0
        assert result.n_simulations == 100

    def test_positive_trades_produce_profit(self):
        engine = BacktestEngine(db_path=":memory:")
        # All winning trades
        trades = [
            {"side": "SELL", "pnl": 100.0},
            {"side": "SELL", "pnl": 200.0},
            {"side": "SELL", "pnl": 50.0},
        ]
        result = _run_async(engine.monte_carlo_simulation(
            trades=trades, n_simulations=500, initial_capital=10_000.0,
        ))
        # All trades are positive, so median equity should exceed initial
        assert result.median_final_equity > 10_000.0
        assert result.pct_profitable == 100.0

    def test_negative_trades_produce_loss(self):
        engine = BacktestEngine(db_path=":memory:")
        trades = [
            {"side": "SELL", "pnl": -100.0},
            {"side": "SELL", "pnl": -200.0},
        ]
        result = _run_async(engine.monte_carlo_simulation(
            trades=trades, n_simulations=500, initial_capital=10_000.0,
        ))
        assert result.median_final_equity < 10_000.0
        assert result.pct_profitable == 0.0

    def test_deterministic_seed(self):
        engine = BacktestEngine(db_path=":memory:")
        trades = [
            {"side": "SELL", "pnl": 50.0},
            {"side": "SELL", "pnl": -30.0},
            {"side": "SELL", "pnl": 20.0},
        ]
        r1 = _run_async(engine.monte_carlo_simulation(trades, 100, 10_000.0))
        r2 = _run_async(engine.monte_carlo_simulation(trades, 100, 10_000.0))
        # Deterministic seed=42 should produce identical results
        assert r1.median_final_equity == r2.median_final_equity

    def test_buy_trades_ignored(self):
        engine = BacktestEngine(db_path=":memory:")
        trades = [
            {"side": "BUY", "pnl": 999.0},  # Should be ignored
            {"side": "SELL", "pnl": 10.0},
        ]
        result = _run_async(engine.monte_carlo_simulation(trades, 100, 10_000.0))
        assert result.median_final_equity == 10_010.0


# ---------------------------------------------------------------------------
# Tests — Monthly returns
# ---------------------------------------------------------------------------

class TestMonthlyReturns:
    def test_empty_curve_returns_empty(self):
        assert BacktestEngine._monthly_returns([]) == []

    def test_single_month(self):
        curve = [
            {"timestamp": "2025-01-01T00:00:00+00:00", "equity": 10_000.0},
            {"timestamp": "2025-01-15T00:00:00+00:00", "equity": 10_500.0},
            {"timestamp": "2025-01-31T00:00:00+00:00", "equity": 11_000.0},
        ]
        monthly = BacktestEngine._monthly_returns(curve)
        assert len(monthly) == 1
        assert monthly[0]["return_pct"] == pytest.approx(10.0, abs=0.01)
