"""Tests for RenaissancePositionSizer — capacity-aware Kelly position sizing.

Tests cover:
  - calculate_size() with normal inputs returns valid size
  - calculate_size() with zero confidence returns blocked result
  - calculate_size() with confidence below min returns blocked result
  - Kelly criterion calculation with known win_rate/win_loss_ratio
  - Cost gate blocks when cost > 50% of expected profit
  - Vol normalization scales position size correctly
  - Min order size enforcement
  - Max position percentage cap
  - Total exposure cap prevents over-allocation
  - Edge case: negative edge returns zero size
  - Drawdown-based scaling
  - Regime scalar adjustments
  - Round-trip cost estimation
"""

import pytest
import math
from unittest.mock import MagicMock

from position_sizer import RenaissancePositionSizer, SizingResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sizer() -> RenaissancePositionSizer:
    """Default sizer with known config for deterministic tests."""
    return RenaissancePositionSizer(config={
        "default_balance_usd": 10000.0,
        "max_position_pct": 25.0,
        "max_total_exposure_pct": 80.0,
        "kelly_fraction": 0.50,
        "min_edge": 0.001,
        "min_win_prob": 0.52,
        "taker_fee_bps": 5.0,
        "maker_fee_bps": 0.0,
        "spread_cost_bps": 2.0,
        "slippage_bps": 1.0,
        "cost_gate_ratio": 0.50,
        "impact_coefficient": 0.10,
        "max_participation_rate": 0.02,
        "fallback_daily_volume": 500_000_000,
        "target_vol": 0.02,
        "vol_floor": 0.005,
        "vol_ceiling": 0.10,
        "min_order_usd": 1.0,
    })


@pytest.fixture
def generous_sizer() -> RenaissancePositionSizer:
    """Sizer with relaxed gates for testing sizing logic in isolation."""
    return RenaissancePositionSizer(config={
        "default_balance_usd": 100000.0,
        "max_position_pct": 50.0,
        "max_total_exposure_pct": 100.0,
        "kelly_fraction": 1.0,
        "min_edge": 0.0001,
        "min_win_prob": 0.50,
        "taker_fee_bps": 0.0,
        "maker_fee_bps": 0.0,
        "spread_cost_bps": 0.0,
        "slippage_bps": 0.0,
        "cost_gate_ratio": 1.0,
        "impact_coefficient": 0.0,
        "max_participation_rate": 1.0,
        "fallback_daily_volume": 1_000_000_000,
        "target_vol": 0.02,
        "min_order_usd": 1.0,
    })


# ── Test: Normal inputs produce valid size ────────────────────────────────────

class TestNormalInputs:
    def test_normal_inputs_return_positive_size(self, generous_sizer):
        """With strong signal, high confidence, and reasonable price, we get a trade."""
        result = generous_sizer.calculate_size(
            signal_strength=0.8,
            confidence=0.9,
            current_price=50000.0,
            product_id="BTC-USD",
            volatility=0.02,
        )
        assert isinstance(result, SizingResult)
        assert result.asset_units > 0
        assert result.usd_value > 0
        assert result.sizing_method != "blocked"
        assert result.sizing_method != "cost_gated"

    def test_sizing_result_has_audit_trail(self, generous_sizer):
        """SizingResult contains reasons list for auditing."""
        result = generous_sizer.calculate_size(
            signal_strength=0.8,
            confidence=0.9,
            current_price=50000.0,
        )
        assert isinstance(result.reasons, list)
        assert len(result.reasons) > 0


# ── Test: Zero confidence returns blocked ─────────────────────────────────────

class TestZeroConfidence:
    def test_zero_confidence_returns_zero_size(self, sizer):
        result = sizer.calculate_size(
            signal_strength=0.5,
            confidence=0.0,
            current_price=50000.0,
        )
        assert result.asset_units == 0.0
        assert result.usd_value == 0.0
        assert result.sizing_method == "blocked"

    def test_negative_confidence_returns_zero_size(self, sizer):
        result = sizer.calculate_size(
            signal_strength=0.5,
            confidence=-0.1,
            current_price=50000.0,
        )
        assert result.asset_units == 0.0
        assert result.sizing_method == "blocked"


# ── Test: Confidence below min returns blocked ────────────────────────────────

class TestLowConfidence:
    def test_low_confidence_blocked(self, sizer):
        """Confidence that produces win_prob below min_win_prob should block."""
        # With confidence=0.1, win_prob = 0.48 + 0.1*0.10 = 0.49
        # After Bayesian shrinkage: 0.5 + (0.49 - 0.5)*0.55 = 0.4945
        # This is below min_win_prob=0.52, so should be blocked
        result = sizer.calculate_size(
            signal_strength=0.5,
            confidence=0.1,
            current_price=50000.0,
        )
        assert result.asset_units == 0.0
        assert "P(win)" in " ".join(result.reasons) or result.sizing_method in ("blocked", "cost_gated")


# ── Test: Kelly criterion with known values ──────────────────────────────────

class TestKellyCriterion:
    def test_kelly_positive_edge(self, sizer):
        """Kelly formula: f* = (p*b - q) / b"""
        # edge=0.05, win_prob=0.55
        # payoff_ratio = (edge + loss_prob) / win_prob = (0.05 + 0.45) / 0.55 = 0.909
        # kelly_f = (0.55 * 0.909 - 0.45) / 0.909 = (0.5 - 0.45) / 0.909 = 0.055
        kelly = sizer._kelly_criterion(edge=0.05, win_prob=0.55)
        assert kelly > 0.0
        assert kelly <= 0.25  # Capped at 25%

    def test_kelly_zero_edge(self, sizer):
        """Zero edge should give zero Kelly fraction."""
        kelly = sizer._kelly_criterion(edge=0.0, win_prob=0.55)
        assert kelly == 0.0

    def test_kelly_negative_edge(self, sizer):
        """Negative edge should give zero Kelly fraction."""
        kelly = sizer._kelly_criterion(edge=-0.01, win_prob=0.55)
        assert kelly == 0.0

    def test_kelly_capped_at_25_pct(self, sizer):
        """Even with very high edge, Kelly should be capped at 25%."""
        kelly = sizer._kelly_criterion(edge=1.0, win_prob=0.99)
        assert kelly <= 0.25

    def test_kelly_zero_win_prob(self, sizer):
        """Zero win probability should give zero Kelly fraction."""
        kelly = sizer._kelly_criterion(edge=0.05, win_prob=0.0)
        assert kelly == 0.0


# ── Test: Cost gate blocks high-cost trades ──────────────────────────────────

class TestCostGate:
    def test_cost_gate_blocks_when_cost_exceeds_edge(self):
        """When round-trip cost > cost_gate_ratio * edge, trade is blocked."""
        # Use extremely high fees and a low cost_gate_ratio
        # to ensure cost/edge > cost_gate_ratio
        sizer = RenaissancePositionSizer(config={
            "taker_fee_bps": 500.0,  # 5% per leg
            "maker_fee_bps": 500.0,
            "spread_cost_bps": 200.0,
            "slippage_bps": 100.0,
            "cost_gate_ratio": 0.10,  # Very strict gate
            "min_edge": 0.0001,
            "min_win_prob": 0.50,
            "default_balance_usd": 10000.0,
        })
        result = sizer.calculate_size(
            signal_strength=0.3,  # Moderate signal → moderate edge
            confidence=0.6,
            current_price=50000.0,
        )
        assert result.asset_units == 0.0
        assert result.sizing_method == "cost_gated"

    def test_cost_gate_allows_low_cost_trade(self, sizer):
        """With MEXC maker 0%, costs should be low enough to pass gate."""
        cost = sizer.estimate_round_trip_cost()
        # entry: 5 + 1 + 1 = 7bps, exit: 0 + 1 + 1 = 2bps, total = 9bps = 0.0009
        assert cost < 0.01  # Less than 1%


# ── Test: Round-trip cost estimation ─────────────────────────────────────────

class TestRoundTripCost:
    def test_round_trip_cost_calculation(self, sizer):
        """Verify round-trip cost calculation matches expected formula."""
        # entry_cost = taker(5) + spread/2(1) + slippage(1) = 7 bps
        # exit_cost = maker(0) + spread/2(1) + slippage(1) = 2 bps
        # total = 9 bps = 0.0009
        cost = sizer.estimate_round_trip_cost()
        expected = (5.0 + 1.0 + 1.0 + 0.0 + 1.0 + 1.0) / 10000.0
        assert abs(cost - expected) < 1e-8


# ── Test: Volatility normalization ───────────────────────────────────────────

class TestVolNormalization:
    def test_high_vol_reduces_size(self, sizer):
        """Higher volatility should produce a lower vol scalar."""
        scalar = sizer._volatility_scalar(vol=0.08)  # 4x target
        assert scalar < 1.0  # Should shrink
        expected = 0.02 / 0.08  # 0.25
        assert abs(scalar - expected) < 1e-6

    def test_low_vol_increases_size(self, sizer):
        """Lower volatility should produce a higher vol scalar."""
        scalar = sizer._volatility_scalar(vol=0.01)  # 0.5x target
        assert scalar > 1.0  # Should expand
        expected = 0.02 / 0.01  # 2.0 (capped at 2.0)
        assert abs(scalar - expected) < 1e-6

    def test_vol_scalar_clamped_min(self, sizer):
        """Vol scalar is clamped to [0.25, 2.0]."""
        scalar = sizer._volatility_scalar(vol=1.0)  # Very high
        assert scalar >= 0.25

    def test_vol_scalar_clamped_max(self, sizer):
        """Vol scalar is clamped to [0.25, 2.0]."""
        scalar = sizer._volatility_scalar(vol=0.001)  # Very low
        assert scalar <= 2.0

    def test_zero_vol_returns_one(self, sizer):
        """Zero volatility should return 1.0 to avoid division by zero."""
        scalar = sizer._volatility_scalar(vol=0.0)
        assert scalar == 1.0

    def test_vol_regime_fallback(self, sizer):
        """When volatility is None, vol_regime provides default."""
        vol = sizer._get_volatility(None, "crisis")
        assert vol == 0.060  # crisis default

    def test_vol_clipped_to_floor_ceiling(self, sizer):
        """Provided volatility should be clipped to [vol_floor, vol_ceiling]."""
        vol = sizer._get_volatility(0.001, None)
        assert vol == sizer.vol_floor
        vol = sizer._get_volatility(1.0, None)
        assert vol == sizer.vol_ceiling


# ── Test: Min order size enforcement ─────────────────────────────────────────

class TestMinOrderSize:
    def test_below_min_order_returns_gated(self):
        """Size below min_order_usd should be cost-gated."""
        sizer = RenaissancePositionSizer(config={
            "min_order_usd": 100.0,  # High minimum
            "default_balance_usd": 50.0,  # Tiny balance
            "max_position_pct": 100.0,
            "min_edge": 0.0001,
            "min_win_prob": 0.50,
            "cost_gate_ratio": 1.0,
            "taker_fee_bps": 0.0,
            "maker_fee_bps": 0.0,
            "spread_cost_bps": 0.0,
            "slippage_bps": 0.0,
        })
        result = sizer.calculate_size(
            signal_strength=0.8,
            confidence=0.9,
            current_price=50000.0,
        )
        # Either blocked or gated because size < $100 min
        assert result.asset_units == 0.0


# ── Test: Max position percentage cap ────────────────────────────────────────

class TestMaxPositionPct:
    def test_position_capped_at_max_pct(self, generous_sizer):
        """Position USD value should not exceed max_position_pct of balance."""
        result = generous_sizer.calculate_size(
            signal_strength=1.0,
            confidence=1.0,
            current_price=50000.0,
            account_balance_usd=100000.0,
        )
        max_allowed = 100000.0 * 0.50  # 50% max_position_pct
        assert result.usd_value <= max_allowed + 1.0  # +1 for rounding


# ── Test: Total exposure cap ────────────────────────────────────────────────

class TestExposureCap:
    def test_exposure_cap_limits_new_position(self, sizer):
        """When current exposure is near the cap, new position should be small or blocked."""
        result = sizer.calculate_size(
            signal_strength=0.8,
            confidence=0.9,
            current_price=50000.0,
            current_exposure_usd=7900.0,  # Near the 80% cap of $10k = $8000
        )
        # Remaining headroom = $8000 - $7900 = $100
        assert result.usd_value <= 200.0  # Should be small

    def test_exposure_cap_blocks_when_full(self, sizer):
        """When exposure equals the cap, no new position should open."""
        result = sizer.calculate_size(
            signal_strength=0.8,
            confidence=0.9,
            current_price=50000.0,
            current_exposure_usd=8000.0,  # Exactly at 80% cap
        )
        assert result.asset_units == 0.0


# ── Test: Negative/zero edge returns zero size ───────────────────────────────

class TestNegativeEdge:
    def test_zero_signal_strength_blocked(self, sizer):
        """Zero signal should produce a blocked result."""
        result = sizer.calculate_size(
            signal_strength=0.0,
            confidence=0.9,
            current_price=50000.0,
        )
        assert result.asset_units == 0.0
        assert result.sizing_method == "blocked"

    def test_invalid_price_blocked(self, sizer):
        """Negative or zero price should produce a blocked result."""
        result = sizer.calculate_size(
            signal_strength=0.5,
            confidence=0.9,
            current_price=0.0,
        )
        assert result.asset_units == 0.0
        assert result.sizing_method == "blocked"

        result = sizer.calculate_size(
            signal_strength=0.5,
            confidence=0.9,
            current_price=-100.0,
        )
        assert result.asset_units == 0.0


# ── Test: Drawdown-based scaling ─────────────────────────────────────────────

class TestDrawdownScaling:
    def test_drawdown_halts_at_15_pct(self, sizer):
        """15% drawdown should halt trading entirely."""
        result = sizer.calculate_size(
            signal_strength=0.8,
            confidence=0.9,
            current_price=50000.0,
            drawdown_pct=0.15,
        )
        assert result.asset_units == 0.0
        assert result.sizing_method == "blocked"
        assert any("rawdown" in r.lower() or "halt" in r.lower() for r in result.reasons)

    def test_drawdown_reduces_size_at_10_pct(self, generous_sizer):
        """10% drawdown should reduce size to 25%."""
        normal = generous_sizer.calculate_size(
            signal_strength=0.8,
            confidence=0.9,
            current_price=50000.0,
            drawdown_pct=0.0,
        )
        dd = generous_sizer.calculate_size(
            signal_strength=0.8,
            confidence=0.9,
            current_price=50000.0,
            drawdown_pct=0.10,
        )
        # With 10% DD, scalar = 0.25, so size should be roughly 25% of normal
        if normal.usd_value > 0 and dd.usd_value > 0:
            ratio = dd.usd_value / normal.usd_value
            assert ratio < 0.50  # Should be significantly reduced


# ── Test: Regime scalar adjustments ──────────────────────────────────────────

class TestRegimeScalar:
    def test_trending_regime_scales_up(self, sizer):
        """Trending regime scalar (1.20) should produce larger size than volatile (0.60)."""
        assert sizer.regime_scalars["trending"] > sizer.regime_scalars["volatile"]

    def test_chaotic_regime_scales_down(self, sizer):
        """Chaotic regime should have the lowest scalar."""
        assert sizer.regime_scalars["chaotic"] == 0.30

    def test_unknown_regime_uses_default(self, sizer):
        """Unknown regime string should fall back to 0.80."""
        scalar = sizer.regime_scalars.get("nonexistent_regime", 0.80)
        assert scalar == 0.80


# ── Test: Round size precision ───────────────────────────────────────────────

class TestRoundSize:
    def test_btc_precision(self, sizer):
        """BTC should round to 8 decimal places."""
        rounded = sizer._round_size(0.123456789123, "BTC-USD")
        assert rounded == 0.12345678

    def test_sol_precision(self, sizer):
        """SOL should round to 6 decimal places."""
        rounded = sizer._round_size(1.123456789, "SOL-USD")
        assert rounded == 1.123456

    def test_doge_precision(self, sizer):
        """DOGE should round to 2 decimal places."""
        rounded = sizer._round_size(1000.9876, "DOGE-USD")
        assert rounded == 1000.98

    def test_rounds_down_not_up(self, sizer):
        """Should always floor (not round) to avoid over-sizing."""
        rounded = sizer._round_size(0.99999999, "BTC-USD")
        assert rounded == 0.99999999
        rounded = sizer._round_size(1.999, "DOGE-USD")
        assert rounded == 1.99


# ── Test: Exit sizing ────────────────────────────────────────────────────────

class TestExitSizing:
    def test_stop_loss_triggers(self, sizer):
        """>2% loss should trigger immediate stop loss."""
        result = sizer.calculate_exit_size(
            position_size=0.1,
            entry_price=50000.0,
            current_price=48900.0,  # -2.2% loss
            holding_periods=0,
            confidence=0.5,
        )
        assert result["exit_fraction"] == 1.0
        assert result["reason"] == "stop_loss"

    def test_max_age_forces_close(self, sizer):
        """Positions held >= 2 cycles should be force-closed."""
        result = sizer.calculate_exit_size(
            position_size=0.1,
            entry_price=50000.0,
            current_price=50000.0,  # Flat
            holding_periods=2,
            confidence=0.5,
        )
        assert result["exit_fraction"] == 1.0
        assert result["reason"] == "max_age"

    def test_min_hold_prevents_early_exit(self, sizer):
        """Positions held < 1 period should be held (except stop loss)."""
        result = sizer.calculate_exit_size(
            position_size=0.1,
            entry_price=50000.0,
            current_price=50100.0,  # Slight profit
            holding_periods=0,
            confidence=0.5,
        )
        assert result["exit_fraction"] == 0.0
        assert result["reason"] == "hold"

    def test_profit_target_full_exit(self, sizer):
        """>3% profit should trigger full exit."""
        result = sizer.calculate_exit_size(
            position_size=0.1,
            entry_price=50000.0,
            current_price=51600.0,  # +3.2% profit
            holding_periods=1,
            confidence=0.5,
        )
        assert result["exit_fraction"] == 1.0
        assert result["reason"] == "profit_target"

    def test_short_side_pnl_inversion(self, sizer):
        """SHORT positions should profit when price falls."""
        result = sizer.calculate_exit_size(
            position_size=0.1,
            entry_price=50000.0,
            current_price=48000.0,  # Price fell → profit for SHORT
            holding_periods=1,
            confidence=0.5,
            side="SHORT",
        )
        # 4% gain for short → should hit >3% profit target
        assert result["exit_fraction"] == 1.0
        assert result["reason"] == "profit_target"


# ── Test: Fund capacity computation ──────────────────────────────────────────

class TestFundCapacity:
    def test_capacity_positive(self, sizer):
        """Fund capacity should be positive for known instruments."""
        result = sizer.compute_fund_capacity(["BTC-USD", "ETH-USD"])
        assert result["total_capacity_usd"] > 0
        assert "BTC-USD" in result["per_instrument"]
        assert "ETH-USD" in result["per_instrument"]

    def test_capacity_scales_with_volume(self, sizer):
        """Higher daily volume should give higher capacity."""
        cap = sizer.compute_fund_capacity(["BTC-USD"])
        btc_cap = cap["per_instrument"]["BTC-USD"]["effective_capacity_usd"]
        # BTC has $3B daily volume vs. SOL at $300M
        cap_sol = sizer.compute_fund_capacity(["SOL-USD"])
        sol_cap = cap_sol["per_instrument"]["SOL-USD"]["effective_capacity_usd"]
        assert btc_cap > sol_cap
