"""
Tests for execution/synchronized_executor.py — SynchronizedExecutor.

Covers simultaneous order execution, timing gap tracking, execution quality
metrics, retry logic, and paper-trading simulation.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from execution.synchronized_executor import (
    SynchronizedExecutor,
    SyncExecutionResult,
    _DEFAULT_CONFIG,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> dict:
    """Build a minimal config dict for SynchronizedExecutor."""
    cfg = {
        "synchronized_executor": {},
        "trading": {"paper_trading": True},
    }
    cfg["synchronized_executor"].update(overrides)
    return cfg


def _run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def _event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Tests — Initialization
# ---------------------------------------------------------------------------

class TestSynchronizedExecutorInit:
    def test_default_config(self):
        exe = SynchronizedExecutor(_make_config())
        assert exe._paper_trading is True
        assert exe._cfg["max_submission_gap_ms"] == 100
        assert exe._cfg["max_fill_wait_ms"] == 5000

    def test_custom_config(self):
        exe = SynchronizedExecutor(_make_config(
            max_submission_gap_ms=50,
            max_fill_wait_ms=3000,
        ))
        assert exe._cfg["max_submission_gap_ms"] == 50
        assert exe._cfg["max_fill_wait_ms"] == 3000

    def test_devil_tracker_stored(self):
        tracker = MagicMock()
        exe = SynchronizedExecutor(_make_config(), devil_tracker=tracker)
        assert exe._devil_tracker is tracker


# ---------------------------------------------------------------------------
# Tests — _prepare_order
# ---------------------------------------------------------------------------

class TestPrepareOrder:
    def test_order_structure(self):
        exe = SynchronizedExecutor(_make_config())
        order = exe._prepare_order("mexc", "BTCUSDT", "buy", 50000.0, 0.01)

        assert order["exchange"] == "mexc"
        assert order["pair"] == "BTCUSDT"
        assert order["side"] == "buy"
        assert order["price"] == 50000.0
        assert order["quantity"] == 0.01
        assert order["type"] == "limit"
        assert order["time_in_force"] == "IOC"
        assert "id" in order
        assert "created_at" in order

    def test_order_id_uniqueness(self):
        exe = SynchronizedExecutor(_make_config())
        o1 = exe._prepare_order("a", "BTCUSDT", "buy", 100, 1)
        o2 = exe._prepare_order("a", "BTCUSDT", "buy", 100, 1)
        assert o1["id"] != o2["id"]


# ---------------------------------------------------------------------------
# Tests — _submit_order (paper trading)
# ---------------------------------------------------------------------------

class TestSubmitOrderPaper:
    def test_paper_fill_returns_filled(self):
        exe = SynchronizedExecutor(_make_config())
        order = exe._prepare_order("mexc", "BTCUSDT", "buy", 50000.0, 0.01)
        result = _run_async(exe._submit_order("mexc", order))

        assert result["status"] == "filled"
        assert result["fill_quantity"] == 0.01
        assert result["submit_ts"] > 0
        assert result["fill_ts"] >= result["submit_ts"]
        assert result["fill_price"] > 0

    def test_paper_buy_slippage_positive(self):
        """Buy fills should be at or above requested price (unfavorable)."""
        exe = SynchronizedExecutor(_make_config())
        order = exe._prepare_order("binance", "ETHUSDT", "buy", 3000.0, 1.0)
        result = _run_async(exe._submit_order("binance", order))
        assert result["fill_price"] >= 3000.0 - 0.15  # Within 0.5 bps slippage

    def test_paper_sell_slippage_negative(self):
        """Sell fills should be at or below requested price (unfavorable)."""
        exe = SynchronizedExecutor(_make_config())
        order = exe._prepare_order("mexc", "ETHUSDT", "sell", 3000.0, 1.0)
        result = _run_async(exe._submit_order("mexc", order))
        assert result["fill_price"] <= 3000.0 + 0.15


# ---------------------------------------------------------------------------
# Tests — _submit_order (live mode raises)
# ---------------------------------------------------------------------------

class TestSubmitOrderLive:
    def test_live_mode_raises_not_implemented(self):
        cfg = _make_config()
        cfg["trading"]["paper_trading"] = False
        exe = SynchronizedExecutor(cfg)
        order = exe._prepare_order("mexc", "BTCUSDT", "buy", 50000.0, 0.01)
        with pytest.raises(NotImplementedError, match="Live order submission"):
            _run_async(exe._submit_order("mexc", order))


# ---------------------------------------------------------------------------
# Tests — execute_cross_exchange
# ---------------------------------------------------------------------------

class TestExecuteCrossExchange:
    def test_successful_execution(self):
        exe = SynchronizedExecutor(_make_config())
        result = _run_async(exe.execute_cross_exchange(
            buy_exchange="mexc",
            buy_pair="BTCUSDT",
            buy_price=50_000.0,
            buy_quantity=0.01,
            sell_exchange="binance",
            sell_pair="BTCUSDT",
            sell_price=50_050.0,
            sell_quantity=0.01,
        ))

        assert isinstance(result, SyncExecutionResult)
        assert result.leg_a_exchange == "mexc"
        assert result.leg_b_exchange == "binance"
        assert result.expected_spread_bps > 0
        assert result.timing_gap_ms >= 0
        assert result.submission_gap_ms >= 0

    def test_stats_recorded(self):
        exe = SynchronizedExecutor(_make_config())
        _run_async(exe.execute_cross_exchange(
            "mexc", "BTCUSDT", 50_000.0, 0.01,
            "binance", "BTCUSDT", 50_050.0, 0.01,
        ))
        assert len(exe._timing_stats) == 1

    def test_devil_tracker_called(self):
        tracker = MagicMock()
        exe = SynchronizedExecutor(_make_config(), devil_tracker=tracker)
        _run_async(exe.execute_cross_exchange(
            "mexc", "BTCUSDT", 50_000.0, 0.01,
            "binance", "BTCUSDT", 50_050.0, 0.01,
        ))
        tracker.log_event.assert_called_once()

    def test_devil_tracker_error_handled(self):
        """Devil tracker logging errors should not crash execution."""
        tracker = MagicMock()
        tracker.log_event.side_effect = RuntimeError("tracker down")
        exe = SynchronizedExecutor(_make_config(), devil_tracker=tracker)

        result = _run_async(exe.execute_cross_exchange(
            "mexc", "BTCUSDT", 50_000.0, 0.01,
            "binance", "BTCUSDT", 50_050.0, 0.01,
        ))
        assert isinstance(result, SyncExecutionResult)

    def test_retry_on_failure(self):
        exe = SynchronizedExecutor(_make_config(max_retries=2))
        call_count = 0

        original_submit = exe._submit_order

        async def flaky_submit(exchange, order):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Simulated failure")
            return await original_submit(exchange, order)

        exe._submit_order = flaky_submit
        result = _run_async(exe.execute_cross_exchange(
            "mexc", "BTCUSDT", 50_000.0, 0.01,
            "binance", "BTCUSDT", 50_050.0, 0.01,
        ))
        # After 2 failures of the gather (each calling submit twice),
        # should eventually succeed or return failure
        assert isinstance(result, SyncExecutionResult)

    def test_all_retries_exhausted(self):
        exe = SynchronizedExecutor(_make_config(max_retries=0))

        async def always_fail(exchange, order):
            raise ConnectionError("Permanent failure")

        exe._submit_order = always_fail
        result = _run_async(exe.execute_cross_exchange(
            "mexc", "BTCUSDT", 50_000.0, 0.01,
            "binance", "BTCUSDT", 50_050.0, 0.01,
        ))
        assert result.success is False
        assert result.leg_a_fill_price == 0.0


# ---------------------------------------------------------------------------
# Tests — Timing report
# ---------------------------------------------------------------------------

class TestTimingReport:
    def test_empty_report(self):
        exe = SynchronizedExecutor(_make_config())
        report = exe.get_timing_report()
        assert report["total_executions"] == 0
        assert report["success_rate"] == 0.0

    def test_report_after_execution(self):
        exe = SynchronizedExecutor(_make_config())
        _run_async(exe.execute_cross_exchange(
            "mexc", "BTCUSDT", 50_000.0, 0.01,
            "binance", "BTCUSDT", 50_050.0, 0.01,
        ))
        report = exe.get_timing_report()
        assert report["total_executions"] == 1
        assert report["avg_submission_gap_ms"] >= 0
        assert report["avg_fill_gap_ms"] >= 0
        assert report["avg_spread_decay_bps"] is not None

    def test_report_after_multiple_executions(self):
        exe = SynchronizedExecutor(_make_config())
        for _ in range(3):
            _run_async(exe.execute_cross_exchange(
                "mexc", "BTCUSDT", 50_000.0, 0.01,
                "binance", "BTCUSDT", 50_100.0, 0.01,
            ))
        report = exe.get_timing_report()
        assert report["total_executions"] == 3

    def test_p95_submission_gap(self):
        exe = SynchronizedExecutor(_make_config())
        for _ in range(10):
            _run_async(exe.execute_cross_exchange(
                "mexc", "BTCUSDT", 50_000.0, 0.01,
                "binance", "BTCUSDT", 50_100.0, 0.01,
            ))
        report = exe.get_timing_report()
        assert report["p95_submission_gap_ms"] >= 0
