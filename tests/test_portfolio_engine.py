"""
Unit tests for core/portfolio_engine.py
========================================
Tests PortfolioEngine: signal ingestion, target computation, drift,
correction generation, execution, reconciliation, position registration,
and re-evaluation action execution.

All external dependencies (position manager, devil tracker, exchanges)
are mocked.
"""

import asyncio
import time
from decimal import Decimal
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from core.portfolio_engine import (
    PortfolioEngine,
    PortfolioTarget,
    PortfolioActual,
    CorrectionOrder,
    _DEFAULT_CONFIG,
)
from core.data_structures import PositionContext, ReEvalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(**overrides) -> PortfolioEngine:
    """Create a PortfolioEngine with sensible defaults."""
    kwargs = dict(
        config=None,
        cost_model=None,
        devil_tracker=None,
        position_manager=None,
        kelly_sizer=None,
        regime_detector=None,
    )
    kwargs.update(overrides)
    return PortfolioEngine(**kwargs)


def _make_position_context(**overrides) -> PositionContext:
    defaults = dict(
        position_id="pos-001",
        pair="BTC-USD",
        exchange="mexc",
        side="long",
        strategy="stat_arb",
        entry_price=Decimal("60000"),
        entry_size=Decimal("0.01"),
        entry_size_usd=Decimal("600"),
        entry_timestamp=time.time() - 120,
        entry_confidence=0.72,
        entry_expected_move_bps=15.0,
        entry_cost_estimate_bps=3.0,
        entry_net_edge_bps=12.0,
        entry_regime="trending_up",
        entry_volatility=0.02,
        entry_book_depth_usd=Decimal("50000"),
        entry_spread_bps=1.5,
    )
    defaults.update(overrides)
    return PositionContext(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_default_config(self):
        engine = _make_engine()
        assert engine.cfg["reconciliation_interval_seconds"] == _DEFAULT_CONFIG["reconciliation_interval_seconds"]
        assert engine.cfg["drift_threshold_pct"] == _DEFAULT_CONFIG["drift_threshold_pct"]
        assert engine.cfg["max_leverage"] == _DEFAULT_CONFIG["max_leverage"]

    def test_custom_config(self):
        engine = _make_engine(config={
            "medallion_portfolio_engine": {
                "drift_threshold_pct": 5.0,
                "max_leverage": 5.0,
            }
        })
        assert engine.cfg["drift_threshold_pct"] == 5.0
        assert engine.cfg["max_leverage"] == 5.0
        # Non-overridden keys keep defaults
        assert engine.cfg["max_corrections_per_cycle"] == _DEFAULT_CONFIG["max_corrections_per_cycle"]

    def test_initial_state(self):
        engine = _make_engine()
        assert engine._signals == {}
        assert engine._last_actual is None
        assert engine._corrections_this_cycle == 0
        assert engine._running is False
        assert engine.open_positions == {}
        assert engine.reevaluator is None


# ---------------------------------------------------------------------------
# Signal ingestion
# ---------------------------------------------------------------------------

class TestSignalIngestion:

    def test_ingest_valid_signal(self):
        engine = _make_engine()
        result = engine.ingest_signal({
            "pair": "BTC-USD",
            "signal_type": "stat_arb",
            "side": "BUY",
            "strength": 0.12,
            "confidence": 0.7,
        })
        assert result is True
        assert len(engine._signals) == 1

    def test_ingest_sets_expiry(self):
        engine = _make_engine()
        engine.ingest_signal({
            "pair": "BTC-USD",
            "signal_type": "stat_arb",
            "strength": 0.1,
        })
        key = ("BTC-USD", "stat_arb")
        sig = engine._signals[key]
        assert "_expires_at" in sig
        assert sig["_expires_at"] > time.time()

    def test_ingest_custom_ttl(self):
        engine = _make_engine()
        now = time.time()
        engine.ingest_signal({
            "pair": "BTC-USD",
            "signal_type": "stat_arb",
            "ttl_seconds": 120,
            "strength": 0.1,
        })
        key = ("BTC-USD", "stat_arb")
        expires = engine._signals[key]["_expires_at"]
        assert expires >= now + 119  # within margin

    def test_ingest_missing_pair_rejected(self):
        engine = _make_engine()
        result = engine.ingest_signal({"signal_type": "stat_arb", "strength": 0.1})
        assert result is False
        assert len(engine._signals) == 0

    def test_ingest_replaces_same_key(self):
        engine = _make_engine()
        engine.ingest_signal({
            "pair": "BTC-USD", "signal_type": "stat_arb",
            "strength": 0.1, "confidence": 0.5,
        })
        engine.ingest_signal({
            "pair": "BTC-USD", "signal_type": "stat_arb",
            "strength": 0.5, "confidence": 0.9,
        })
        assert len(engine._signals) == 1
        key = ("BTC-USD", "stat_arb")
        assert engine._signals[key]["strength"] == 0.5

    def test_ingest_different_signals_coexist(self):
        engine = _make_engine()
        engine.ingest_signal({
            "pair": "BTC-USD", "signal_type": "stat_arb", "strength": 0.1,
        })
        engine.ingest_signal({
            "pair": "BTC-USD", "signal_type": "momentum", "strength": 0.2,
        })
        engine.ingest_signal({
            "pair": "ETH-USD", "signal_type": "stat_arb", "strength": 0.3,
        })
        assert len(engine._signals) == 3


# ---------------------------------------------------------------------------
# Signal pruning
# ---------------------------------------------------------------------------

class TestSignalPruning:

    def test_prune_expired_signals(self):
        engine = _make_engine()
        engine.ingest_signal({
            "pair": "BTC-USD", "signal_type": "stat_arb",
            "strength": 0.1, "ttl_seconds": 0,  # expires immediately
        })
        time.sleep(0.01)
        pruned = engine._prune_expired_signals()
        assert pruned == 1
        assert len(engine._signals) == 0

    def test_prune_keeps_active_signals(self):
        engine = _make_engine()
        engine.ingest_signal({
            "pair": "BTC-USD", "signal_type": "stat_arb",
            "strength": 0.1, "ttl_seconds": 3600,
        })
        pruned = engine._prune_expired_signals()
        assert pruned == 0
        assert len(engine._signals) == 1


# ---------------------------------------------------------------------------
# Target computation
# ---------------------------------------------------------------------------

class TestComputeTarget:

    def test_empty_signals(self):
        engine = _make_engine()
        target = engine.compute_target()
        assert isinstance(target, PortfolioTarget)
        assert target.positions == {}
        assert target.net_exposure == 0.0

    def test_single_buy_signal(self):
        engine = _make_engine()
        engine.ingest_signal({
            "pair": "BTC-USD", "signal_type": "stat_arb",
            "side": "BUY", "strength": 0.1, "confidence": 1.0,
        })
        target = engine.compute_target()
        assert "BTC-USD" in target.positions
        # strength=0.1, confidence=1.0, equity=10000 -> 0.1 * 1.0 * 10000 = 1000
        assert target.positions["BTC-USD"] == pytest.approx(1000.0)

    def test_sell_signal_produces_negative_position(self):
        engine = _make_engine()
        engine.ingest_signal({
            "pair": "BTC-USD", "signal_type": "stat_arb",
            "side": "SELL", "strength": 0.2, "confidence": 1.0,
        })
        target = engine.compute_target()
        assert target.positions["BTC-USD"] < 0

    def test_notional_usd_takes_precedence(self):
        """notional_usd overrides strength*confidence sizing, but still capped."""
        engine = _make_engine()
        engine.ingest_signal({
            "pair": "BTC-USD", "signal_type": "stat_arb",
            "side": "BUY", "strength": 0.1, "confidence": 0.5,
            "notional_usd": 5000.0,
        })
        target = engine.compute_target()
        # notional_usd=5000, but equity=10000, max_single_position_pct=15%
        # -> cap at 1500. The notional is used instead of strength*conf*equity.
        assert target.positions["BTC-USD"] == pytest.approx(1500.0)

    def test_position_capped_at_max(self):
        engine = _make_engine(config={
            "medallion_portfolio_engine": {"max_single_position_pct": 10.0}
        })
        engine.ingest_signal({
            "pair": "BTC-USD", "signal_type": "stat_arb",
            "side": "BUY", "notional_usd": 50000.0,
        })
        target = engine.compute_target()
        # equity=10000, max=10% -> cap at 1000
        assert target.positions["BTC-USD"] <= 1000.0 + 1

    def test_leverage_capped(self):
        engine = _make_engine(config={
            "medallion_portfolio_engine": {"max_leverage": 1.0}
        })
        engine.ingest_signal({
            "pair": "BTC-USD", "signal_type": "s1",
            "side": "BUY", "notional_usd": 5000.0,
        })
        engine.ingest_signal({
            "pair": "ETH-USD", "signal_type": "s2",
            "side": "BUY", "notional_usd": 5000.0,
        })
        target = engine.compute_target()
        total_abs = sum(abs(v) for v in target.positions.values())
        # equity=10000, max_leverage=1.0 -> total_abs <= 10000
        assert total_abs <= 10001


# ---------------------------------------------------------------------------
# Fetch actual (no position manager)
# ---------------------------------------------------------------------------

class TestFetchActual:

    def test_no_position_manager(self):
        engine = _make_engine()
        actual = engine.fetch_actual()
        assert isinstance(actual, PortfolioActual)
        assert actual.positions == {}
        assert actual.equity == 10_000.0

    def test_with_mock_position_manager(self):
        pm = MagicMock()
        pos = MagicMock()
        pos.product_id = "BTC-USD"
        pos.size = 0.1
        pos.entry_price = 60000.0
        pos.side = MagicMock(value="LONG")
        pos.unrealized_pnl = 50.0
        pm.get_all_positions.return_value = [pos]
        pm.get_position_summary.return_value = MagicMock(total_exposure_usd=20000.0)

        engine = _make_engine(position_manager=pm)
        actual = engine.fetch_actual()
        assert "BTC-USD" in actual.positions
        assert actual.unrealized_pnl == 50.0


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------

class TestComputeDrift:

    def test_no_signals_no_drift(self):
        engine = _make_engine()
        drift = engine.compute_drift()
        assert drift == {}

    def test_drift_exceeds_threshold(self):
        engine = _make_engine(config={
            "medallion_portfolio_engine": {"drift_threshold_pct": 1.0}
        })
        engine.ingest_signal({
            "pair": "BTC-USD", "signal_type": "stat_arb",
            "side": "BUY", "notional_usd": 500.0,
        })
        drift = engine.compute_drift()
        # Target: 500, Actual: 0, threshold: 10000*0.01=100 -> drift=500 > 100
        assert "BTC-USD" in drift
        assert drift["BTC-USD"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Correction generation
# ---------------------------------------------------------------------------

class TestGenerateCorrections:

    def test_empty_drift(self):
        engine = _make_engine()
        corrections = engine.generate_corrections({})
        assert corrections == []

    def test_generates_buy_for_positive_drift(self):
        engine = _make_engine()
        corrections = engine.generate_corrections({"BTC-USD": 500.0})
        assert len(corrections) == 1
        assert corrections[0].side == "BUY"
        assert corrections[0].quantity == 500.0
        assert corrections[0].pair == "BTC-USD"

    def test_generates_sell_for_negative_drift(self):
        engine = _make_engine()
        corrections = engine.generate_corrections({"ETH-USD": -300.0})
        assert len(corrections) == 1
        assert corrections[0].side == "SELL"
        assert corrections[0].quantity == 300.0

    def test_capped_at_max_corrections(self):
        engine = _make_engine(config={
            "medallion_portfolio_engine": {"max_corrections_per_cycle": 2}
        })
        drift = {f"PAIR-{i}": 100.0 * (i + 1) for i in range(5)}
        corrections = engine.generate_corrections(drift)
        assert len(corrections) <= 2

    def test_prioritized_by_magnitude(self):
        engine = _make_engine()
        drift = {"A": 100.0, "B": 500.0, "C": 200.0}
        corrections = engine.generate_corrections(drift)
        # Sorted by abs magnitude descending
        assert corrections[0].pair == "B"
        assert corrections[1].pair == "C"
        assert corrections[2].pair == "A"

    def test_skip_if_cost_too_high(self):
        cost_model = MagicMock()
        cost_model.estimate_round_trip_cost.return_value = 0.01  # 100 bps
        engine = _make_engine(
            cost_model=cost_model,
            config={"medallion_portfolio_engine": {"max_correction_cost_bps": 3.0}}
        )
        corrections = engine.generate_corrections({"BTC-USD": 500.0})
        assert len(corrections) == 0  # Skipped due to high cost


# ---------------------------------------------------------------------------
# Correction execution
# ---------------------------------------------------------------------------

class TestExecuteCorrections:

    def test_execute_with_devil_tracker(self):
        dt = MagicMock()
        dt.record_signal_detection.return_value = "trade-123"
        engine = _make_engine(devil_tracker=dt)

        corrections = [
            CorrectionOrder(pair="BTC-USD", side="BUY", quantity=500.0,
                            reason="drift"),
        ]
        logged = engine.execute_corrections(corrections)
        assert logged == 1
        dt.record_signal_detection.assert_called_once()

    def test_execute_without_devil_tracker(self):
        engine = _make_engine()
        corrections = [
            CorrectionOrder(pair="BTC-USD", side="BUY", quantity=500.0,
                            reason="drift"),
        ]
        logged = engine.execute_corrections(corrections)
        assert logged == 1

    def test_execute_empty_list(self):
        engine = _make_engine()
        logged = engine.execute_corrections([])
        assert logged == 0


# ---------------------------------------------------------------------------
# Position registration
# ---------------------------------------------------------------------------

class TestPositionRegistration:

    def test_register_position(self):
        engine = _make_engine()
        pos = _make_position_context()
        engine.register_position(pos)
        assert pos.position_id in engine.open_positions
        assert engine.open_positions[pos.position_id] is pos

    def test_register_multiple_positions(self):
        engine = _make_engine()
        pos1 = _make_position_context(position_id="p1")
        pos2 = _make_position_context(position_id="p2", pair="ETH-USD")
        engine.register_position(pos1)
        engine.register_position(pos2)
        assert len(engine.open_positions) == 2


# ---------------------------------------------------------------------------
# Re-evaluation action execution
# ---------------------------------------------------------------------------

class TestExecuteReevalAction:

    def test_hold_action(self):
        engine = _make_engine()
        pos = _make_position_context()
        pos.current_size_usd = Decimal("600")
        engine.open_positions[pos.position_id] = pos

        result = ReEvalResult(
            position_id=pos.position_id,
            timestamp=time.time(),
            action="hold",
            reason="Within tolerance",
            reason_code="WITHIN_TOLERANCE",
            rescored_confidence=0.65,
            remaining_edge_bps=5.0,
            optimal_size_usd=Decimal("580"),
            current_size_usd=Decimal("600"),
            size_ratio=1.03,
        )
        engine._execute_reeval_action(result)
        # Position should still exist
        assert pos.position_id in engine.open_positions

    def test_close_action(self):
        dt = MagicMock()
        engine = _make_engine(devil_tracker=dt)
        pos = _make_position_context()
        pos.current_size_usd = Decimal("600")
        pos.current_price = Decimal("61000")
        pos.current_size = Decimal("0.01")
        engine.open_positions[pos.position_id] = pos

        result = ReEvalResult(
            position_id=pos.position_id,
            timestamp=time.time(),
            action="close",
            reason="Edge consumed",
            reason_code="EDGE_CONSUMED",
            rescored_confidence=0.52,
            remaining_edge_bps=0.5,
            optimal_size_usd=Decimal("0"),
            current_size_usd=Decimal("600"),
            size_ratio=0.0,
        )
        engine._execute_reeval_action(result)
        assert pos.position_id not in engine.open_positions
        dt.record_exit.assert_called_once()

    def test_trim_action(self):
        engine = _make_engine()
        pos = _make_position_context()
        pos.current_size_usd = Decimal("600")
        pos.current_price = Decimal("60000")
        pos.current_size = Decimal("0.01")
        engine.open_positions[pos.position_id] = pos

        result = ReEvalResult(
            position_id=pos.position_id,
            timestamp=time.time(),
            action="trim",
            reason="Trim to optimal",
            reason_code="PROFIT_TRIM",
            rescored_confidence=0.60,
            remaining_edge_bps=3.0,
            optimal_size_usd=Decimal("400"),
            current_size_usd=Decimal("600"),
            size_ratio=0.67,
            trim_amount_usd=Decimal("200"),
        )
        engine._execute_reeval_action(result)
        assert pos.current_size_usd == Decimal("400")
        assert pos.total_trimmed_usd == Decimal("200")
        assert len(pos.adjustments) == 1
        assert pos.adjustments[0]["action"] == "trim"

    def test_add_action(self):
        engine = _make_engine()
        pos = _make_position_context()
        pos.current_size_usd = Decimal("600")
        pos.current_price = Decimal("60000")
        pos.current_size = Decimal("0.01")
        engine.open_positions[pos.position_id] = pos

        result = ReEvalResult(
            position_id=pos.position_id,
            timestamp=time.time(),
            action="add",
            reason="Conditions improved",
            reason_code="REEVAL_ADD",
            rescored_confidence=0.75,
            remaining_edge_bps=10.0,
            optimal_size_usd=Decimal("1000"),
            current_size_usd=Decimal("600"),
            size_ratio=1.67,
            add_amount_usd=Decimal("400"),
        )
        engine._execute_reeval_action(result)
        assert pos.current_size_usd == Decimal("1000")
        assert pos.total_added_usd == Decimal("400")
        assert len(pos.adjustments) == 1
        assert pos.adjustments[0]["action"] == "add"

    def test_missing_position_no_crash(self):
        engine = _make_engine()
        result = ReEvalResult(
            position_id="nonexistent",
            timestamp=time.time(),
            action="close",
            reason="test",
            reason_code="TEST",
            rescored_confidence=0.5,
            remaining_edge_bps=0.0,
            optimal_size_usd=Decimal("0"),
            current_size_usd=Decimal("0"),
            size_ratio=0.0,
        )
        # Should not raise
        engine._execute_reeval_action(result)


# ---------------------------------------------------------------------------
# Status / introspection
# ---------------------------------------------------------------------------

class TestGetStatus:

    def test_status_basic(self):
        engine = _make_engine()
        status = engine.get_status()
        assert status["active_signals"] == 0
        assert status["running"] is False
        assert status["open_positions"] == 0
        assert status["reevaluator_active"] is False

    def test_status_with_signals(self):
        engine = _make_engine()
        engine.ingest_signal({
            "pair": "BTC-USD", "signal_type": "stat_arb", "strength": 0.1,
        })
        status = engine.get_status()
        assert status["active_signals"] == 1


# ---------------------------------------------------------------------------
# Stop method
# ---------------------------------------------------------------------------

class TestStop:

    def test_stop_sets_running_false(self):
        engine = _make_engine()
        engine._running = True
        engine.stop()
        assert engine._running is False
