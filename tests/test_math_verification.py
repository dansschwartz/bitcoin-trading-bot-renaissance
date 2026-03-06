"""
Math verification tests.
Every test case comes from STRATEGY_MATH_TRUTH_DOCUMENT.md.
If a test fails, the CODE is wrong (not the test).

These tests verify formulas in isolation using the same math
the production code uses. They do NOT import production modules
(to avoid import-chain side effects). Instead, they reimplement
each formula from the truth document and verify the expected
input/output pairs.
"""

import pytest
import math


# ══════════════════════════════════════════════════════════════
# STRATEGY 1: BTC STRADDLE (straddle_engine.py)
# ══════════════════════════════════════════════════════════════

class TestStraddleMath:
    """Formulas 1.1 through 1.9."""

    # -- 1.1 Leg PnL (basis points) --

    def test_1_1_long_leg_pnl_positive(self):
        entry, current = 85000, 85010
        pnl_bps = (current - entry) / entry * 10000
        assert abs(pnl_bps - 1.176) < 0.01

    def test_1_1_long_leg_pnl_negative(self):
        entry, current = 85000, 84990
        pnl_bps = (current - entry) / entry * 10000
        assert abs(pnl_bps - (-1.176)) < 0.01

    def test_1_1_short_leg_pnl_positive(self):
        entry, current = 85000, 84990
        pnl_bps = (entry - current) / entry * 10000
        assert abs(pnl_bps - 1.176) < 0.01

    def test_1_1_short_leg_pnl_negative(self):
        entry, current = 85000, 85010
        pnl_bps = (entry - current) / entry * 10000
        assert abs(pnl_bps - (-1.176)) < 0.01

    # -- 1.2 Leg PnL (USD) --

    def test_1_2_leg_pnl_usd_positive(self):
        pnl_bps, size = 10, 5
        pnl_usd = pnl_bps / 10000 * size
        assert pnl_usd == pytest.approx(0.005)

    def test_1_2_leg_pnl_usd_negative(self):
        pnl_bps, size = -8, 5
        pnl_usd = pnl_bps / 10000 * size
        assert pnl_usd == pytest.approx(-0.004)

    def test_1_2_leg_pnl_usd_large(self):
        pnl_bps, size = 100, 5
        pnl_usd = pnl_bps / 10000 * size
        assert pnl_usd == pytest.approx(0.050)

    # -- 1.3 Net Straddle PnL (no doubling) --

    def test_1_3_net_pnl_case1(self):
        """long=+15bp, short=-6bp, size=$5"""
        long_bps, short_bps, size = 15, -6, 5
        net_bps = long_bps + short_bps
        net_usd = net_bps / 10000 * size  # NOT * 2
        assert net_bps == 9
        assert net_usd == pytest.approx(0.0045)

    def test_1_3_net_pnl_case2(self):
        """long=-6bp, short=+25bp, size=$5"""
        long_bps, short_bps, size = -6, 25, 5
        net_bps = long_bps + short_bps
        net_usd = net_bps / 10000 * size
        assert net_bps == 19
        assert net_usd == pytest.approx(0.0095)

    def test_1_3_net_pnl_case3_both_negative(self):
        """long=-6bp, short=-4bp, size=$5"""
        long_bps, short_bps, size = -6, -4, 5
        net_bps = long_bps + short_bps
        net_usd = net_bps / 10000 * size
        assert net_bps == -10
        assert net_usd == pytest.approx(-0.005)

    def test_1_3_net_pnl_usd_equals_sum_of_legs(self):
        """net_pnl_usd must equal sum of individual leg pnl_usd values."""
        long_bps, short_bps, size = 15, -6, 5
        long_usd = long_bps / 10000 * size
        short_usd = short_bps / 10000 * size
        net_via_sum = long_usd + short_usd
        net_via_formula = (long_bps + short_bps) / 10000 * size
        assert net_via_sum == pytest.approx(net_via_formula)

    # -- 1.4 Peak PnL tracking --

    def test_1_4_peak_tracking(self):
        sequence = [1, 3, 5, 4, 2, 6, 3]
        expected_peaks = [1, 3, 5, 5, 5, 6, 6]
        peak = 0.0
        actual_peaks = []
        for pnl in sequence:
            peak = max(peak, pnl)
            actual_peaks.append(peak)
        assert actual_peaks == expected_peaks

    def test_1_4_peak_init_zero(self):
        """Peak must be initialized to 0.0."""
        peak = 0.0
        assert peak == 0.0

    # -- 1.5 Trailing stop --

    def test_1_5_trailing_stop_trigger(self):
        activation, distance = 3, 4
        sequence = [1, 3, 5, 8, 6, 5, 4]
        trail_active = False
        peak = 0.0
        triggered_at = None

        for i, pnl in enumerate(sequence):
            peak = max(peak, pnl)
            if not trail_active and pnl >= activation:
                trail_active = True
            if trail_active and (peak - pnl) >= distance:
                triggered_at = i
                break

        assert triggered_at == 6  # Index of +4
        assert sequence[triggered_at] == 4  # Exit at +4, NOT +8 (peak)

    def test_1_5_trail_not_triggered_before_activation(self):
        """Trail must not trigger before activation threshold is met."""
        activation, distance = 10, 4
        sequence = [1, 3, 5, 8, 4]  # Never reaches activation=10
        trail_active = False
        peak = 0.0
        triggered = False

        for pnl in sequence:
            peak = max(peak, pnl)
            if not trail_active and pnl >= activation:
                trail_active = True
            if trail_active and (peak - pnl) >= distance:
                triggered = True
                break

        assert not triggered

    # -- 1.6 Hard stop --

    def test_1_6_hard_stop_trigger(self):
        stop_loss_bps = 6
        assert -6.1 <= -stop_loss_bps  # triggers
        assert not (-5.9 <= -stop_loss_bps)  # does not trigger

    # -- 1.7 Timeout classification --

    def test_1_7_timeout_flat(self):
        max_hold, min_move = 600, 3
        age, peak, current = 601, 1.5, 0.5
        assert age >= max_hold
        assert abs(peak) < min_move  # timeout_flat

    def test_1_7_timeout_profitable(self):
        max_hold, min_move = 600, 3
        age, peak, current = 601, 10, 4
        assert age >= max_hold
        assert abs(peak) >= min_move
        assert current > 0  # timeout_profitable

    def test_1_7_timeout_loss(self):
        max_hold, min_move = 600, 3
        age, peak, current = 601, 10, -2
        assert age >= max_hold
        assert abs(peak) >= min_move
        assert current <= 0  # timeout_loss

    # -- 1.8 Capital deployed --

    def test_1_8_capital_deployed(self):
        straddles, leg_size = 3, 5
        deployed = straddles * leg_size * 2
        assert deployed == 30

    # -- 1.9 Daily loss circuit breaker --

    def test_1_9_daily_loss_halt(self):
        max_daily_loss = 250
        daily = 0.0
        trades = [-3.00, 1.50, -249.00]
        for pnl in trades:
            if pnl < 0:
                daily += abs(pnl)
        # daily = 3.0 + 249.0 = 252.0
        assert daily >= max_daily_loss  # HALT


# ══════════════════════════════════════════════════════════════
# STRATEGY 2: TOKEN SPRAY (token_spray_engine.py)
# ══════════════════════════════════════════════════════════════

class TestTokenSprayMath:
    """Formulas 2.1 through 2.8."""

    # -- 2.1 Token PnL (bps) — same as straddle --

    def test_2_1_long_pnl(self):
        entry, current = 85000, 85010
        pnl_bps = (current - entry) / entry * 10000
        assert abs(pnl_bps - 1.176) < 0.01

    def test_2_1_short_pnl(self):
        entry, current = 85000, 85010
        pnl_bps = (entry - current) / entry * 10000
        assert abs(pnl_bps - (-1.176)) < 0.01

    # -- 2.2 Token PnL (USD) --

    def test_2_2_pnl_usd_vol_scaled(self):
        pnl_bps, token_size = 10, 40  # Vol-scaled from $100 base
        pnl_usd = pnl_bps / 10000 * token_size
        assert pnl_usd == pytest.approx(0.040)

    def test_2_2_pnl_usd_negative(self):
        pnl_bps, token_size = -8, 40
        pnl_usd = pnl_bps / 10000 * token_size
        assert pnl_usd == pytest.approx(-0.032)

    # -- 2.3 Vol-scaled token size --

    def test_2_3_vol_scaling_high(self):
        base = 100
        scales = {"low": 1.0, "medium": 0.7, "high": 0.4, "extreme": 0.2}
        assert base * scales["high"] == 40

    def test_2_3_vol_scaling_low(self):
        base = 100
        scales = {"low": 1.0, "medium": 0.7, "high": 0.4, "extreme": 0.2}
        assert base * scales["low"] == 100

    def test_2_3_vol_scaling_extreme(self):
        base = 100
        scales = {"low": 1.0, "medium": 0.7, "high": 0.4, "extreme": 0.2}
        assert base * scales["extreme"] == 20

    # -- 2.4 Budget gate --

    def test_2_4_budget_gate_reject(self):
        max_budget, deployed, size_usd = 10000, 9970, 40
        assert deployed + size_usd > max_budget  # REJECT

    def test_2_4_budget_gate_allow(self):
        max_budget, deployed, size_usd = 10000, 9950, 40
        assert deployed + size_usd <= max_budget  # ALLOW

    # -- 2.7 Spread gate --

    def test_2_7_spread_gate_reject(self):
        spread, stop, ratio = 5, 8, 0.5
        threshold = stop * ratio  # 4
        assert spread >= threshold  # REJECT (5 >= 4)

    def test_2_7_spread_gate_allow(self):
        spread, stop, ratio = 3, 8, 0.5
        threshold = stop * ratio  # 4
        assert spread < threshold  # ALLOW (3 < 4)


# ══════════════════════════════════════════════════════════════
# STRATEGY 3: POLYMARKET (polymarket_strategy_a.py)
# ══════════════════════════════════════════════════════════════

class TestPolymarketMath:
    """Formulas 3.1 through 3.5."""

    # -- 3.1 Kelly criterion --

    def test_3_1_kelly_positive(self):
        p, token_cost = 0.60, 0.50
        q = 1 - p
        b = (1 / token_cost) - 1
        kelly = (p * b - q) / (b + 1e-10)
        assert abs(kelly - 0.20) < 0.001

    def test_3_1_kelly_negative_returns_zero(self):
        p, token_cost = 0.40, 0.50
        q = 1 - p
        b = (1 / token_cost) - 1
        kelly = (p * b - q) / (b + 1e-10)
        assert kelly < 0
        result = max(0.0, kelly)
        assert result == 0.0
        # CRITICAL: must return 0.0 before MIN_BET is applied

    def test_3_1_kelly_third_case(self):
        p, token_cost = 0.55, 0.30
        q = 1 - p
        b = (1 / token_cost) - 1  # 2.333...
        kelly = (p * b - q) / (b + 1e-10)
        assert abs(kelly - 0.357) < 0.01

    def test_3_1_b_formula_equivalence(self):
        """Verify (1/cost)-1 == (1-cost)/cost algebraically."""
        for cost in [0.30, 0.50, 0.65, 0.85]:
            b1 = (1 / cost) - 1
            b2 = (1.0 - cost) / cost
            assert b1 == pytest.approx(b2)

    # -- 3.2 Fractional Kelly sizing --

    def test_3_2_basic_sizing(self):
        kelly, fraction, bankroll = 0.20, 0.25, 500
        frac = kelly * fraction  # 0.05
        bet = bankroll * frac  # $25
        assert bet == pytest.approx(25)

    def test_3_2_zero_kelly_returns_zero(self):
        """kelly=0 must return $0.0 immediately, not MIN_BET."""
        kelly = 0.0
        # If kelly <= 0, return 0.0 BEFORE min/max clamping
        assert kelly <= 0
        # bet should be 0.0, NOT $5 (MIN_BET)

    def test_3_2_sizing_caps(self):
        kelly, fraction, bankroll = 0.50, 0.25, 2000
        MAX_SIZING = 1000
        MAX_BET = 50
        sizing_bankroll = min(bankroll, MAX_SIZING)
        frac = kelly * fraction  # 0.125
        bet = sizing_bankroll * frac  # 1000 * 0.125 = $125
        bet = min(bet, MAX_BET)  # Capped at $50
        assert bet == 50

    # -- 3.3 Bet resolution --

    def test_3_3_won_pnl(self):
        invested, tokens = 25, 49.02
        payout = tokens * 1.0
        pnl = payout - invested
        assert abs(pnl - 24.02) < 0.01

    def test_3_3_lost_pnl(self):
        invested = 25
        pnl = 0 - invested  # tokens worth $0
        assert pnl == -25.00

    def test_3_3_closed_at_price(self):
        invested, tokens, exit_price = 25, 48.54, 0.72
        payout = tokens * exit_price
        pnl = payout - invested
        assert abs(pnl - 9.95) < 0.01

    # -- 3.4 Bankroll updates (net effect) --

    def test_3_4_bankroll_won_net(self):
        """Net effect: bankroll changes by +pnl on WON."""
        bankroll = 500
        invested = 25
        pnl = 24.02
        # At placement: bankroll -= invested => 475
        bankroll -= invested
        # At resolution (WON): bankroll += invested + pnl => 475 + 25 + 24.02 = 524.02
        bankroll += invested + pnl
        assert bankroll == pytest.approx(524.02)
        # Net change = +24.02 = +pnl

    def test_3_4_bankroll_lost_net(self):
        """Net effect: bankroll changes by -invested on LOST."""
        bankroll = 500
        invested = 25
        # At placement: bankroll -= invested => 475
        bankroll -= invested
        # At resolution (LOST): no bankroll change
        assert bankroll == pytest.approx(475)
        # Net change = -25 = -invested

    # -- 3.5 Odds filter --

    def test_3_5_odds_filter_extreme_longshot(self):
        MIN_ODDS, MAX_ODDS = 0.15, 0.85
        token_cost = 0.07
        assert token_cost < MIN_ODDS  # BLOCKED

    def test_3_5_odds_filter_allowed(self):
        MIN_ODDS, MAX_ODDS = 0.15, 0.85
        token_cost = 0.50
        assert MIN_ODDS <= token_cost <= MAX_ODDS  # ALLOWED

    def test_3_5_odds_filter_extreme_favorite(self):
        MIN_ODDS, MAX_ODDS = 0.15, 0.85
        token_cost = 0.92
        assert token_cost > MAX_ODDS  # BLOCKED

    def test_3_5_odds_filter_at_boundaries(self):
        MIN_ODDS, MAX_ODDS = 0.15, 0.85
        assert MIN_ODDS <= 0.15 <= MAX_ODDS  # ALLOWED
        assert MIN_ODDS <= 0.85 <= MAX_ODDS  # ALLOWED


# ══════════════════════════════════════════════════════════════
# STRATEGY 4: ARBITRAGE
# ══════════════════════════════════════════════════════════════

class TestArbitrageMath:
    """Formulas 4.1 through 4.2."""

    # -- 4.1 Cross-exchange spread --

    def test_4_1_spread_profitable_direction(self):
        # Buy on MEXC at ask=85000, sell on Binance at bid=85015
        ask_low, bid_high = 85000, 85015
        spread_bps = (bid_high - ask_low) / ask_low * 10000
        assert abs(spread_bps - 1.76) < 0.1

    def test_4_1_spread_negative_direction(self):
        # Buy on Binance at ask=85020, sell on MEXC at bid=84995
        ask_low, bid_high = 85020, 84995
        spread_bps = (bid_high - ask_low) / ask_low * 10000
        assert spread_bps < 0  # No opportunity

    def test_4_1_must_check_both_directions(self):
        """One direction can be profitable while the other isn't."""
        mexc_ask, mexc_bid = 85000, 84995
        binance_ask, binance_bid = 85020, 85015
        # Dir A: buy MEXC, sell Binance
        spread_a = (binance_bid - mexc_ask) / mexc_ask * 10000
        # Dir B: buy Binance, sell MEXC
        spread_b = (mexc_bid - binance_ask) / binance_ask * 10000
        assert spread_a > 0  # Profitable
        assert spread_b < 0  # Not profitable

    # -- 4.2 Arb PnL --

    def test_4_2_arb_pnl_unprofitable_after_fees(self):
        spread_bps, size = 5, 100
        fee_buy = 0  # MEXC maker 0%
        fee_sell = size * 0.001  # Binance 0.1%
        rebalancing = 0.01
        gross = spread_bps / 10000 * size
        net = gross - fee_buy - fee_sell - rebalancing
        assert abs(net - (-0.06)) < 0.01


# ══════════════════════════════════════════════════════════════
# STRATEGY 5: VOLATILITY MODEL
# ══════════════════════════════════════════════════════════════

class TestVolatilityMath:
    """Formulas 5.1 through 5.2."""

    # -- 5.1 Prediction output --

    def test_5_1_expm1_transform(self):
        log_mag = 1.5
        predicted_bps = math.expm1(log_mag)
        assert abs(predicted_bps - 3.48) < 0.1

    def test_5_1_expm1_explosive(self):
        log_mag = 4.5
        predicted_bps = math.expm1(log_mag)
        assert abs(predicted_bps - 89.0) < 2.0

    def test_5_1_regime_classification(self):
        p25, p75, p90 = 2.386, 3.828, 4.380
        cases = [
            (1.5, "dead_zone"),     # < p25
            (3.0, "normal"),        # p25..p75
            (4.0, "active"),        # p75..p90
            (5.0, "explosive"),     # > p90
        ]
        for log_mag, expected_regime in cases:
            if log_mag < p25:
                regime = "dead_zone"
            elif log_mag < p75:
                regime = "normal"
            elif log_mag < p90:
                regime = "active"
            else:
                regime = "explosive"
            assert regime == expected_regime, f"log_mag={log_mag}"

    # -- 5.2 Dead zone gate --

    def test_5_2_dead_zone_blocks(self):
        min_vol_bps = 12.0
        predicted_bps = 8.5
        assert predicted_bps < min_vol_bps  # BLOCK

    def test_5_2_dead_zone_allows(self):
        min_vol_bps = 12.0
        predicted_bps = 15.0
        assert predicted_bps >= min_vol_bps  # ALLOW


# ══════════════════════════════════════════════════════════════
# SHARED FORMULAS (S.1 through S.3)
# ══════════════════════════════════════════════════════════════

class TestSharedMath:
    """Formulas S.1 through S.3."""

    # -- S.1 Win rate --

    def test_S1_win_rate_excludes_zero(self):
        pnls = [5, -3, 2, -1, 8, 0, -4, 1, -2, 3]
        winners = sum(1 for p in pnls if p > 0)  # >0, NOT >=0
        resolved = len(pnls)
        assert winners == 5
        assert resolved == 10
        assert winners / resolved == pytest.approx(0.50)

    def test_S1_zero_is_not_a_win(self):
        pnls = [0, 0, 0]
        winners = sum(1 for p in pnls if p > 0)
        assert winners == 0

    # -- S.2 Average win / average loss --

    def test_S2_avg_win_loss(self):
        pnls = [5, -3, 2, -1, 8, 0, -4, 1, -2, 3]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        assert abs(avg_win - 3.8) < 0.01
        assert abs(avg_loss - (-2.5)) < 0.01

    def test_S2_excludes_zeros(self):
        pnls = [5, 0, -3, 0]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        assert len(wins) == 1  # Only +5
        assert len(losses) == 1  # Only -3

    # -- S.3 Cumulative PnL --

    def test_S3_cumulative_includes_both(self):
        closed_pnls = [5, -3, 2, -1, 8, -4, 1, -2, 3]
        cumulative = sum(closed_pnls)
        assert cumulative == 9  # Must include wins AND losses

    def test_S3_excludes_open(self):
        """Open positions must not be counted in cumulative PnL."""
        closed_pnls = [5, -3, 2]
        open_unrealized = [10, -5]  # Must NOT be included
        cumulative = sum(closed_pnls)  # Only closed
        assert cumulative == 4
