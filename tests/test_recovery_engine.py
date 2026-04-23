"""
Tests for recovery/state_manager.py and recovery/recovery_engine.py.

Tests state lifecycle management, trade registration/update, heartbeat,
and startup reconciliation logic using in-memory/temp SQLite databases.
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from recovery.state_manager import (
    ActiveTrade,
    StateManager,
    SystemState,
    TradeLifecycleState,
)
from recovery.recovery_engine import RecoveryEngine, RecoveryResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp()
    yield d
    # Cleanup
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def state_manager(tmpdir) -> StateManager:
    db_path = os.path.join(tmpdir, "test_recovery.db")
    hb_path = os.path.join(tmpdir, ".heartbeat")
    return StateManager(db_path=db_path, heartbeat_path=hb_path)


@pytest.fixture
def sample_trade() -> ActiveTrade:
    return ActiveTrade(
        trade_id="test_trade_001",
        signal_type="directional",
        symbol="BTCUSDT",
        state=TradeLifecycleState.PENDING,
        buy_exchange="MEXC",
        sell_exchange="MEXC",
        buy_quantity=0.001,
        buy_price=50000.0,
        expected_profit_usd=5.0,
    )


# ---------------------------------------------------------------------------
# Tests: System State Lifecycle
# ---------------------------------------------------------------------------

class TestSystemStateLifecycle:
    def test_initial_state_is_none(self, state_manager):
        assert state_manager.get_system_state() is None

    def test_set_and_get_state(self, state_manager):
        state_manager.set_system_state(SystemState.STARTING, "test startup")
        assert state_manager.get_system_state() == SystemState.STARTING

    def test_state_transitions(self, state_manager):
        state_manager.set_system_state(SystemState.STARTING, "init")
        state_manager.set_system_state(SystemState.RECOVERING, "recovery")
        state_manager.set_system_state(SystemState.RUNNING, "ready")

        assert state_manager.get_system_state() == SystemState.RUNNING

    def test_state_history_recorded(self, state_manager):
        state_manager.set_system_state(SystemState.STARTING, "boot")
        state_manager.set_system_state(SystemState.RUNNING, "ready")

        history = state_manager.get_state_history(limit=10)
        assert len(history) == 2
        # Newest first
        assert history[0]["state"] == "RUNNING"
        assert history[1]["state"] == "STARTING"

    def test_state_history_includes_previous(self, state_manager):
        state_manager.set_system_state(SystemState.STARTING, "init")
        state_manager.set_system_state(SystemState.RUNNING, "ready")

        history = state_manager.get_state_history()
        running_entry = history[0]
        assert running_entry["previous_state"] == "STARTING"

    def test_halted_state(self, state_manager):
        state_manager.set_system_state(SystemState.HALTED, "critical failure")
        assert state_manager.get_system_state() == SystemState.HALTED


# ---------------------------------------------------------------------------
# Tests: Trade Registration & Updates
# ---------------------------------------------------------------------------

class TestTradeLifecycle:
    def test_register_trade(self, state_manager, sample_trade):
        state_manager.register_trade(sample_trade)
        trade = state_manager.get_trade("test_trade_001")

        assert trade is not None
        assert trade.trade_id == "test_trade_001"
        assert trade.signal_type == "directional"
        assert trade.symbol == "BTCUSDT"
        assert trade.state == TradeLifecycleState.PENDING

    def test_update_trade_state(self, state_manager, sample_trade):
        state_manager.register_trade(sample_trade)
        state_manager.update_trade_state(
            "test_trade_001",
            TradeLifecycleState.BUY_SUBMITTED,
            buy_order_id="order_123",
        )

        trade = state_manager.get_trade("test_trade_001")
        assert trade.state == TradeLifecycleState.BUY_SUBMITTED
        assert trade.buy_order_id == "order_123"

    def test_complete_trade_success(self, state_manager, sample_trade):
        state_manager.register_trade(sample_trade)
        state_manager.complete_trade(
            "test_trade_001", actual_profit_usd=4.50,
        )

        trade = state_manager.get_trade("test_trade_001")
        assert trade.state == TradeLifecycleState.COMPLETED
        assert trade.actual_profit_usd == 4.50

    def test_complete_trade_with_error(self, state_manager, sample_trade):
        state_manager.register_trade(sample_trade)
        state_manager.complete_trade(
            "test_trade_001",
            actual_profit_usd=0.0,
            error_message="exchange timeout",
        )

        trade = state_manager.get_trade("test_trade_001")
        assert trade.state == TradeLifecycleState.FAILED
        assert trade.error_message == "exchange timeout"

    def test_get_active_trades_excludes_terminal(self, state_manager):
        # Register 3 trades
        for i, state in enumerate([
            TradeLifecycleState.PENDING,
            TradeLifecycleState.BUY_SUBMITTED,
            TradeLifecycleState.COMPLETED,
        ]):
            trade = ActiveTrade(
                trade_id=f"trade_{i}",
                signal_type="directional",
                symbol="BTCUSDT",
                state=state,
            )
            state_manager.register_trade(trade)

        active = state_manager.get_active_trades()
        active_ids = {t.trade_id for t in active}

        assert "trade_0" in active_ids   # PENDING is non-terminal
        assert "trade_1" in active_ids   # BUY_SUBMITTED is non-terminal
        assert "trade_2" not in active_ids  # COMPLETED is terminal

    def test_get_incomplete_trades_same_as_active(self, state_manager, sample_trade):
        state_manager.register_trade(sample_trade)
        active = state_manager.get_active_trades()
        incomplete = state_manager.get_incomplete_trades()
        assert len(active) == len(incomplete)

    def test_get_nonexistent_trade_returns_none(self, state_manager):
        assert state_manager.get_trade("nonexistent") is None

    def test_trade_timestamps_set_automatically(self, state_manager, sample_trade):
        state_manager.register_trade(sample_trade)
        trade = state_manager.get_trade("test_trade_001")
        assert trade.created_at != ""
        assert trade.updated_at != ""


# ---------------------------------------------------------------------------
# Tests: Heartbeat
# ---------------------------------------------------------------------------

class TestHeartbeat:
    def test_send_heartbeat_creates_file(self, state_manager, tmpdir):
        state_manager.send_heartbeat()
        hb_path = os.path.join(tmpdir, ".heartbeat")
        assert os.path.exists(hb_path)

    def test_heartbeat_age_decreases_after_send(self, state_manager):
        state_manager.send_heartbeat()
        age = state_manager.heartbeat_age_seconds()
        assert age < 2.0  # Should be very recent

    def test_stale_heartbeat_returns_inf_when_missing(self, tmpdir):
        sm = StateManager(
            db_path=os.path.join(tmpdir, "sm.db"),
            heartbeat_path=os.path.join(tmpdir, "nonexistent_hb"),
        )
        # Remove the heartbeat file if it exists
        hb = os.path.join(tmpdir, "nonexistent_hb")
        if os.path.exists(hb):
            os.unlink(hb)
        age = sm.heartbeat_age_seconds()
        assert age == float("inf")


# ---------------------------------------------------------------------------
# Tests: Recovery Engine
# ---------------------------------------------------------------------------

class TestRecoveryEngine:
    def test_recovery_with_no_clients_succeeds(self, state_manager):
        """Recovery with no exchange clients configured should succeed."""
        engine = RecoveryEngine(state_manager=state_manager)
        result = asyncio.run(engine.run())

        assert isinstance(result, RecoveryResult)
        assert result.success is True
        assert result.trades_found == 0

    def test_recovery_sets_running_state(self, state_manager):
        """After successful recovery, system state should be RUNNING."""
        engine = RecoveryEngine(state_manager=state_manager)
        asyncio.run(engine.run())

        assert state_manager.get_system_state() == SystemState.RUNNING

    def test_recovery_finds_incomplete_trades(self, state_manager, sample_trade):
        """Recovery should find and count incomplete trades."""
        state_manager.register_trade(sample_trade)
        engine = RecoveryEngine(state_manager=state_manager)
        result = asyncio.run(engine.run())

        assert result.trades_found == 1

    def test_recovery_cancels_pending_trades(self, state_manager):
        """PENDING trades should be cancelled during recovery."""
        trade = ActiveTrade(
            trade_id="pending_trade",
            signal_type="directional",
            symbol="ETHUSDT",
            state=TradeLifecycleState.PENDING,
        )
        state_manager.register_trade(trade)

        engine = RecoveryEngine(state_manager=state_manager)
        result = asyncio.run(engine.run())

        recovered_trade = state_manager.get_trade("pending_trade")
        assert recovered_trade.state == TradeLifecycleState.CANCELLED

    def test_recovery_with_connectivity_failure(self, state_manager):
        """If exchange connectivity fails, recovery should halt."""
        mock_coinbase = MagicMock()
        mock_coinbase.get_accounts = MagicMock(side_effect=ConnectionError("timeout"))

        engine = RecoveryEngine(
            state_manager=state_manager,
            coinbase_client=mock_coinbase,
        )
        result = asyncio.run(engine.run())

        assert result.success is False
        assert state_manager.get_system_state() == SystemState.HALTED

    def test_recovery_result_dataclass(self):
        """RecoveryResult should have sensible defaults."""
        r = RecoveryResult()
        assert r.success is False
        assert r.trades_found == 0
        assert r.trades_reconciled == 0
        assert r.trades_emergency_closed == 0
        assert r.orphaned_orders == 0
        assert r.errors == []
        assert r.warnings == []


# ---------------------------------------------------------------------------
# Tests: Trade State Enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_system_state_values(self):
        assert SystemState.STARTING.value == "STARTING"
        assert SystemState.RUNNING.value == "RUNNING"
        assert SystemState.HALTED.value == "HALTED"

    def test_trade_lifecycle_terminal_states(self):
        """COMPLETED, FAILED, CANCELLED should be considered terminal."""
        terminal = {
            TradeLifecycleState.COMPLETED,
            TradeLifecycleState.FAILED,
            TradeLifecycleState.CANCELLED,
        }
        non_terminal = {
            TradeLifecycleState.PENDING,
            TradeLifecycleState.BUY_SUBMITTED,
            TradeLifecycleState.BUY_FILLED,
            TradeLifecycleState.SELL_SUBMITTED,
            TradeLifecycleState.SELL_FILLED,
            TradeLifecycleState.PARTIALLY_FILLED,
            TradeLifecycleState.CANCELLING,
        }
        assert len(terminal) == 3
        assert len(non_terminal) == 7
        assert terminal.isdisjoint(non_terminal)
