"""
Unit tests for core/data_structures.py
========================================
Tests PositionContext, ReEvalResult, REASON_CODES, HorizonEstimate,
ProbabilityCone, and ConeAnalysis dataclasses.
"""

import time
from decimal import Decimal

import pytest

from core.data_structures import (
    ConeAnalysis,
    HorizonEstimate,
    ProbabilityCone,
    PositionContext,
    ReEvalResult,
    REASON_CODES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_position(**overrides) -> PositionContext:
    """Create a PositionContext with sensible defaults, overridable."""
    defaults = dict(
        position_id="pos-001",
        pair="BTC-USD",
        exchange="mexc",
        side="long",
        strategy="stat_arb",
        entry_price=Decimal("60000"),
        entry_size=Decimal("0.01"),
        entry_size_usd=Decimal("600"),
        entry_timestamp=time.time() - 120,  # 2 min ago
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


def _make_horizon(**overrides) -> HorizonEstimate:
    """Create a HorizonEstimate with sensible defaults."""
    defaults = dict(
        horizon_seconds=30,
        horizon_label="30s",
        p_profit=0.55,
        p_loss=0.45,
        e_favorable_bps=5.0,
        e_adverse_bps=4.0,
        e_net_bps=1.0,
        sigma_bps=3.0,
        p_adverse_10bps=0.1,
        p_adverse_25bps=0.05,
        p_adverse_50bps=0.01,
        estimate_confidence=0.8,
        dominant_signal="imbalance",
    )
    defaults.update(overrides)
    return HorizonEstimate(**defaults)


# ---------------------------------------------------------------------------
# PositionContext tests
# ---------------------------------------------------------------------------

class TestPositionContext:
    """Tests for the PositionContext dataclass."""

    def test_basic_construction(self):
        """PositionContext can be constructed with required fields."""
        pos = _make_position()
        assert pos.position_id == "pos-001"
        assert pos.pair == "BTC-USD"
        assert pos.side == "long"
        assert pos.entry_price == Decimal("60000")

    def test_default_optional_fields(self):
        """Default values are correctly set for optional/current-state fields."""
        pos = _make_position()
        assert pos.current_size == Decimal("0")
        assert pos.current_confidence == 0.5
        assert pos.adjustments == []
        assert pos.total_trimmed_usd == Decimal("0")
        assert pos.times_reevaluated == 0
        assert pos.entry_funding_rate is None
        assert pos.signal_ttl_seconds == 300
        assert pos.signal_id == "unknown"

    def test_age_seconds_property(self):
        """age_seconds returns the time elapsed since entry."""
        pos = _make_position(entry_timestamp=time.time() - 60)
        age = pos.age_seconds
        assert 59 <= age <= 62  # allow a small window

    def test_time_elapsed_pct(self):
        """time_elapsed_pct returns fraction of TTL consumed."""
        pos = _make_position(
            entry_timestamp=time.time() - 150,
            signal_ttl_seconds=300,
        )
        pct = pos.time_elapsed_pct
        assert 0.48 <= pct <= 0.52

    def test_time_elapsed_pct_zero_ttl(self):
        """time_elapsed_pct returns 1.0 when TTL is zero or negative."""
        pos = _make_position(signal_ttl_seconds=0)
        assert pos.time_elapsed_pct == 1.0

        pos2 = _make_position(signal_ttl_seconds=-10)
        assert pos2.time_elapsed_pct == 1.0

    def test_move_completion_pct(self):
        """move_completion_pct returns fraction of expected move captured."""
        pos = _make_position(
            entry_expected_move_bps=20.0,
        )
        pos.realized_move_bps = 10.0
        assert pos.move_completion_pct == pytest.approx(0.5)

    def test_move_completion_pct_zero_expected(self):
        """move_completion_pct returns 0.0 when expected move is zero."""
        pos = _make_position(entry_expected_move_bps=0.0)
        pos.realized_move_bps = 5.0
        assert pos.move_completion_pct == 0.0

    def test_edge_consumed_pct(self):
        """edge_consumed_pct returns how much of the original edge is consumed."""
        pos = _make_position(entry_net_edge_bps=10.0)
        pos.remaining_edge_bps = 4.0
        assert pos.edge_consumed_pct == pytest.approx(0.6)

    def test_edge_consumed_pct_zero_net_edge(self):
        """edge_consumed_pct returns 1.0 when entry_net_edge_bps is 0."""
        pos = _make_position(entry_net_edge_bps=0.0)
        assert pos.edge_consumed_pct == 1.0

    def test_is_profitable(self):
        """is_profitable is True when unrealized_pnl_bps > 0."""
        pos = _make_position()
        pos.unrealized_pnl_bps = 5.0
        assert pos.is_profitable is True

        pos.unrealized_pnl_bps = -1.0
        assert pos.is_profitable is False

        pos.unrealized_pnl_bps = 0.0
        assert pos.is_profitable is False

    def test_is_expired(self):
        """is_expired is True when time_elapsed_pct >= 1.0."""
        pos = _make_position(
            entry_timestamp=time.time() - 400,
            signal_ttl_seconds=300,
        )
        assert pos.is_expired is True

        pos2 = _make_position(
            entry_timestamp=time.time() - 100,
            signal_ttl_seconds=300,
        )
        assert pos2.is_expired is False

    def test_size_vs_optimal_ratio(self):
        """size_vs_optimal_ratio returns current / optimal ratio."""
        pos = _make_position()
        pos.current_size = Decimal("0.008")
        pos.current_optimal_size = Decimal("0.01")
        assert pos.size_vs_optimal_ratio == pytest.approx(0.8)

    def test_size_vs_optimal_ratio_zero_optimal(self):
        """size_vs_optimal_ratio returns 0.0 when optimal is zero."""
        pos = _make_position()
        pos.current_size = Decimal("0.01")
        pos.current_optimal_size = Decimal("0")
        assert pos.size_vs_optimal_ratio == 0.0

    def test_adjustments_mutable_default(self):
        """Each PositionContext gets its own adjustments list."""
        pos1 = _make_position()
        pos2 = _make_position()
        pos1.adjustments.append({"action": "trim"})
        assert len(pos1.adjustments) == 1
        assert len(pos2.adjustments) == 0


# ---------------------------------------------------------------------------
# ReEvalResult tests
# ---------------------------------------------------------------------------

class TestReEvalResult:
    """Tests for the ReEvalResult dataclass."""

    def test_basic_construction(self):
        result = ReEvalResult(
            position_id="pos-001",
            timestamp=time.time(),
            action="hold",
            reason="Within tolerance",
            reason_code="WITHIN_TOLERANCE",
            rescored_confidence=0.65,
            remaining_edge_bps=5.0,
            optimal_size_usd=Decimal("500"),
            current_size_usd=Decimal("480"),
            size_ratio=0.96,
        )
        assert result.action == "hold"
        assert result.urgency == "normal"
        assert result.trim_amount_usd is None
        assert result.add_amount_usd is None
        assert result.new_stop_price is None

    def test_trim_result(self):
        result = ReEvalResult(
            position_id="pos-002",
            timestamp=time.time(),
            action="trim",
            reason="Optimal less than current",
            reason_code="PROFIT_TRIM",
            rescored_confidence=0.60,
            remaining_edge_bps=3.0,
            optimal_size_usd=Decimal("300"),
            current_size_usd=Decimal("600"),
            size_ratio=0.5,
            trim_amount_usd=Decimal("300"),
            urgency="normal",
        )
        assert result.trim_amount_usd == Decimal("300")

    def test_close_result_with_urgency(self):
        result = ReEvalResult(
            position_id="pos-003",
            timestamp=time.time(),
            action="close",
            reason="Risk budget exhausted",
            reason_code="HARD_RISK_BUDGET",
            rescored_confidence=0.50,
            remaining_edge_bps=-2.0,
            optimal_size_usd=Decimal("0"),
            current_size_usd=Decimal("500"),
            size_ratio=0.0,
            urgency="critical",
        )
        assert result.urgency == "critical"
        assert result.action == "close"


# ---------------------------------------------------------------------------
# REASON_CODES tests
# ---------------------------------------------------------------------------

class TestReasonCodes:
    """Tests for the REASON_CODES dictionary."""

    def test_reason_codes_is_dict(self):
        assert isinstance(REASON_CODES, dict)

    def test_all_keys_are_strings(self):
        for key in REASON_CODES:
            assert isinstance(key, str), f"Key {key!r} is not a string"

    def test_all_values_are_strings(self):
        for key, value in REASON_CODES.items():
            assert isinstance(value, str), f"Value for {key!r} is not a string"

    def test_critical_codes_present(self):
        """Ensure all documented reason codes exist."""
        expected_codes = [
            "EDGE_CONSUMED", "EDGE_NEGATIVE", "EDGE_COST_EXCEEDED",
            "CONFIDENCE_DECAYED", "CONFIDENCE_TIME",
            "HARD_TARGET_HIT", "HARD_TIME_EXPIRED", "HARD_RISK_BUDGET",
            "HARD_DAILY_LOSS", "HARD_SYSTEM_HALT",
            "CHURN_PREVENTION", "WITHIN_TOLERANCE",
            "CONE_CLOSE_NOW", "CONE_TAIL_RISK",
            "PROFIT_TRIM", "PROFIT_CLOSE",
        ]
        for code in expected_codes:
            assert code in REASON_CODES, f"Missing reason code: {code}"

    def test_no_empty_descriptions(self):
        for key, value in REASON_CODES.items():
            assert len(value.strip()) > 0, f"Empty description for {key}"


# ---------------------------------------------------------------------------
# HorizonEstimate tests
# ---------------------------------------------------------------------------

class TestHorizonEstimate:
    """Tests for the HorizonEstimate dataclass."""

    def test_basic_construction(self):
        h = _make_horizon()
        assert h.horizon_seconds == 30
        assert h.horizon_label == "30s"
        assert h.p_profit == 0.55
        assert h.p_loss == 0.45

    def test_all_fields_accessible(self):
        h = _make_horizon(
            p_adverse_10bps=0.12,
            p_adverse_25bps=0.06,
            p_adverse_50bps=0.02,
        )
        assert h.p_adverse_10bps == 0.12
        assert h.p_adverse_25bps == 0.06
        assert h.p_adverse_50bps == 0.02
        assert h.estimate_confidence == 0.8
        assert h.dominant_signal == "imbalance"


# ---------------------------------------------------------------------------
# ProbabilityCone tests
# ---------------------------------------------------------------------------

class TestProbabilityCone:
    """Tests for the ProbabilityCone dataclass."""

    def _make_cone(self, horizons=None, **overrides):
        if horizons is None:
            horizons = [
                _make_horizon(horizon_seconds=1, e_net_bps=2.0, e_favorable_bps=5.0, e_adverse_bps=3.0),
                _make_horizon(horizon_seconds=30, e_net_bps=1.5, e_favorable_bps=6.0, e_adverse_bps=4.0),
                _make_horizon(horizon_seconds=300, e_net_bps=0.5, e_favorable_bps=8.0, e_adverse_bps=7.0),
            ]
        defaults = dict(
            position_id="pos-001",
            pair="BTC-USD",
            side="long",
            timestamp=time.time(),
            computation_time_ms=1.5,
            horizons=horizons,
            peak_ev_horizon_seconds=1,
            peak_ev_bps=2.0,
            ev_zero_crossing_seconds=600,
            optimal_hold_remaining_seconds=120,
            recommended_action="hold_to_peak",
            action_urgency="none",
            max_horizon_with_positive_ev=300,
            worst_case_5min_bps=-15.0,
            worst_case_15min_bps=-25.0,
        )
        defaults.update(overrides)
        return ProbabilityCone(**defaults)

    def test_basic_construction(self):
        cone = self._make_cone()
        assert cone.position_id == "pos-001"
        assert len(cone.horizons) == 3
        assert cone.peak_ev_bps == 2.0

    def test_is_ev_positive_short_term(self):
        """Short-term EV is positive when first horizon has positive e_net_bps."""
        cone = self._make_cone()
        assert cone.is_ev_positive_short_term is True

        negative_h = [_make_horizon(e_net_bps=-1.0)]
        cone2 = self._make_cone(horizons=negative_h)
        assert cone2.is_ev_positive_short_term is False

    def test_is_ev_positive_short_term_empty(self):
        cone = self._make_cone(horizons=[])
        assert cone.is_ev_positive_short_term is False

    def test_is_ev_decaying(self):
        """EV is decaying when last horizon EV < first horizon EV."""
        cone = self._make_cone()
        assert cone.is_ev_decaying is True  # 0.5 < 2.0

    def test_is_ev_decaying_increasing(self):
        horizons = [
            _make_horizon(horizon_seconds=1, e_net_bps=1.0),
            _make_horizon(horizon_seconds=300, e_net_bps=5.0),
        ]
        cone = self._make_cone(horizons=horizons)
        assert cone.is_ev_decaying is False

    def test_is_ev_decaying_single_horizon(self):
        cone = self._make_cone(horizons=[_make_horizon()])
        assert cone.is_ev_decaying is False

    def test_is_risk_accelerating(self):
        """Risk accelerates when adverse/favorable ratio grows significantly."""
        # short: adverse/fav = 3/5 = 0.6
        # long_ (second to last): adverse/fav = 7/8 = 0.875
        # 0.875 > 0.6 * 1.5 = 0.9 -> False (barely under)
        cone = self._make_cone()
        # Let's create a clearer case
        horizons = [
            _make_horizon(e_favorable_bps=10.0, e_adverse_bps=2.0),  # ratio 0.2
            _make_horizon(e_favorable_bps=5.0, e_adverse_bps=4.0),   # ratio 0.8
            _make_horizon(e_favorable_bps=3.0, e_adverse_bps=6.0),   # ratio 2.0
        ]
        cone2 = self._make_cone(horizons=horizons)
        # short_ratio = 2/10 = 0.2, long_ratio (2nd-to-last) = 4/5 = 0.8
        # 0.8 > 0.2*1.5 = 0.3 -> True
        assert cone2.is_risk_accelerating is True

    def test_is_risk_accelerating_too_few_horizons(self):
        cone = self._make_cone(horizons=[_make_horizon(), _make_horizon()])
        assert cone.is_risk_accelerating is False

    def test_is_risk_accelerating_zero_favorable(self):
        """Returns True when short favorable is zero (division by zero guard)."""
        horizons = [
            _make_horizon(e_favorable_bps=0.0, e_adverse_bps=1.0),
            _make_horizon(e_favorable_bps=5.0, e_adverse_bps=4.0),
            _make_horizon(e_favorable_bps=3.0, e_adverse_bps=6.0),
        ]
        cone = self._make_cone(horizons=horizons)
        assert cone.is_risk_accelerating is True


# ---------------------------------------------------------------------------
# ConeAnalysis tests
# ---------------------------------------------------------------------------

class TestConeAnalysis:
    """Tests for the ConeAnalysis dataclass."""

    def test_basic_construction(self):
        ca = ConeAnalysis(
            position_id="pos-001",
            optimal_hold_seconds=120,
            close_by_seconds=300,
            urgency="low",
            size_multiplier=0.8,
            reason="edge decaying",
            tail_risk_warning=False,
            volatility_expanding=True,
            regime_transition_risk=False,
            edge_front_loaded=True,
            edge_back_loaded=False,
            edge_uniformly_distributed=False,
        )
        assert ca.position_id == "pos-001"
        assert ca.size_multiplier == 0.8
        assert ca.tail_risk_warning is False
        assert ca.edge_front_loaded is True

    def test_all_risk_flags(self):
        """All three risk flags can be set independently."""
        ca = ConeAnalysis(
            position_id="pos-002",
            optimal_hold_seconds=60,
            close_by_seconds=120,
            urgency="high",
            size_multiplier=0.0,
            reason="tail risk detected",
            tail_risk_warning=True,
            volatility_expanding=True,
            regime_transition_risk=True,
            edge_front_loaded=False,
            edge_back_loaded=False,
            edge_uniformly_distributed=True,
        )
        assert ca.tail_risk_warning is True
        assert ca.volatility_expanding is True
        assert ca.regime_transition_risk is True
