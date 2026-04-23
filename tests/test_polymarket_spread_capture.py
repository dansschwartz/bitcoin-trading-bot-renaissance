"""Tests for SpreadCaptureEngine — 0x8dxd-style dual accumulation strategy.

Tests cover:
  - WindowPosition dataclass computed properties (spread, hedged, arb)
  - Dual accumulation guards (cooldown, price improvement, exposure caps)
  - Early exit conditions (losing side identification, min price, shares threshold)
  - Position limits (per-window, global exposure, daily loss)
  - Scalp detection logic
  - Module-level constants match expected values
"""

import asyncio
import sys
import time
import logging
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# Ensure 'requests' is available as a mock if not installed
if "requests" not in sys.modules:
    sys.modules["requests"] = MagicMock()

from polymarket_spread_capture import (
    WindowPosition,
    SpreadCaptureEngine,
    ASSETS,
    TIMEFRAMES,
    BUY_THRESHOLD,
    FILL_SIZE_USD,
    COOLDOWN_PER_SIDE,
    PRICE_IMPROVEMENT,
    MAX_FILLS_PER_SIDE,
    MAX_EXPOSURE_PER_WINDOW,
    MAX_GLOBAL_EXPOSURE,
    MAX_DAILY_LOSS,
    EARLY_EXIT_SECONDS_BEFORE,
    EARLY_EXIT_END_BEFORE,
    EARLY_EXIT_MIN_PRICE,
    SCALP_MULTIPLIER,
    SCALP_MIN_PROFIT,
    ACCUMULATION_START,
    ACCUMULATION_END_BEFORE,
)


def _run(coro):
    """Run an async coroutine synchronously (no pytest-asyncio dependency)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def basic_position() -> WindowPosition:
    """A basic window position with no shares."""
    return WindowPosition(
        asset="BTC",
        timeframe="15m",
        window_start=int(time.time()) - 100,
        slug="btc-updown-15m-1700000000",
    )


@pytest.fixture
def hedged_position() -> WindowPosition:
    """A hedged position with shares on both sides."""
    return WindowPosition(
        asset="BTC",
        timeframe="15m",
        window_start=int(time.time()) - 100,
        slug="btc-updown-15m-1700000000",
        yes_shares=100.0,
        yes_cost=45.0,
        yes_orders=3,
        no_shares=80.0,
        no_cost=36.0,
        no_orders=3,
    )


@pytest.fixture
def engine():
    """SpreadCaptureEngine with fully mocked external deps."""
    mock_rtds = MagicMock()
    mock_rtds.record_window_start = MagicMock()
    mock_rtds.get_chainlink_price = MagicMock(return_value=95000.0)

    with patch("polymarket_spread_capture.sqlite3") as mock_sqlite, \
         patch.object(SpreadCaptureEngine, "_init_clob"), \
         patch.object(SpreadCaptureEngine, "_load_active_windows"):
        mock_conn = MagicMock()
        mock_sqlite.connect.return_value = mock_conn
        eng = SpreadCaptureEngine(rtds=mock_rtds)

    return eng


# ═══════════════════════════════════════════════════════════════
# WINDOW POSITION PROPERTIES
# ═══════════════════════════════════════════════════════════════

class TestWindowPositionProperties:

    def test_avg_yes_price(self, hedged_position):
        assert abs(hedged_position.avg_yes_price - 0.45) < 0.001

    def test_avg_no_price(self, hedged_position):
        assert abs(hedged_position.avg_no_price - 0.45) < 0.001

    def test_avg_price_zero_shares(self, basic_position):
        assert basic_position.avg_yes_price == 0
        assert basic_position.avg_no_price == 0

    def test_pair_cost(self, hedged_position):
        expected = hedged_position.avg_yes_price + hedged_position.avg_no_price
        assert abs(hedged_position.pair_cost - expected) < 0.001

    def test_pair_cost_no_shares_returns_1(self, basic_position):
        assert basic_position.pair_cost == 1.0

    def test_total_cost(self, hedged_position):
        assert abs(hedged_position.total_cost - 81.0) < 0.01

    def test_total_cost_with_sold_revenue(self, hedged_position):
        hedged_position.yes_sold_revenue = 10.0
        hedged_position.no_sold_revenue = 5.0
        assert abs(hedged_position.total_cost - 66.0) < 0.01

    def test_net_shares(self, hedged_position):
        hedged_position.yes_sold_shares = 20.0
        hedged_position.no_sold_shares = 10.0
        assert abs(hedged_position.net_yes_shares - 80.0) < 0.01
        assert abs(hedged_position.net_no_shares - 70.0) < 0.01

    def test_is_hedged_true(self, hedged_position):
        assert hedged_position.is_hedged is True

    def test_is_hedged_false_one_side_empty(self, basic_position):
        assert basic_position.is_hedged is False

    def test_is_hedged_false_after_full_sell(self, hedged_position):
        hedged_position.yes_sold_shares = hedged_position.yes_shares
        assert hedged_position.is_hedged is False

    def test_guaranteed_profit_hedged(self):
        pos = WindowPosition(
            asset="BTC", timeframe="15m", window_start=1000, slug="s",
            yes_shares=100, yes_cost=40.0,
            no_shares=100, no_cost=40.0,
        )
        assert abs(pos.guaranteed_profit - 20.0) < 0.01

    def test_guaranteed_profit_negative_when_not_arb(self):
        pos = WindowPosition(
            asset="BTC", timeframe="15m", window_start=1000, slug="s",
            yes_shares=10, yes_cost=5.0,
            no_shares=10, no_cost=5.5,
        )
        assert pos.guaranteed_profit < 0

    def test_is_arb(self):
        pos = WindowPosition(
            asset="BTC", timeframe="15m", window_start=1000, slug="s",
            yes_shares=100, yes_cost=40.0,
            no_shares=100, no_cost=40.0,
        )
        assert pos.is_arb is True

    def test_is_not_arb(self):
        pos = WindowPosition(
            asset="BTC", timeframe="15m", window_start=1000, slug="s",
            yes_shares=10, yes_cost=5.0,
            no_shares=10, no_cost=5.5,
        )
        assert pos.is_arb is False


# ═══════════════════════════════════════════════════════════════
# SPREAD CALCULATION
# ═══════════════════════════════════════════════════════════════

class TestSpreadCalculation:

    def test_pair_cost_below_one_is_arb(self):
        pos = WindowPosition(
            asset="ETH", timeframe="5m", window_start=1000, slug="s",
            yes_shares=50, yes_cost=22.5,
            no_shares=50, no_cost=22.5,
        )
        assert pos.pair_cost < 1.0
        assert pos.is_arb is True
        assert abs(pos.guaranteed_profit - 5.0) < 0.01

    def test_pair_cost_above_one_is_loss(self):
        pos = WindowPosition(
            asset="ETH", timeframe="5m", window_start=1000, slug="s",
            yes_shares=50, yes_cost=27.5,
            no_shares=50, no_cost=27.5,
        )
        assert pos.pair_cost > 1.0
        assert pos.is_arb is False


# ═══════════════════════════════════════════════════════════════
# DUAL ACCUMULATION LOGIC
# ═══════════════════════════════════════════════════════════════

class TestDualAccumulation:

    def test_cooldown_blocks_rapid_buys(self, engine, hedged_position):
        """Cannot buy again on same side within COOLDOWN_PER_SIDE seconds."""
        hedged_position.last_yes_buy_time = time.time() - 1

        engine._positions = {hedged_position.slug: hedged_position}
        with patch.object(engine, "_place_order", new_callable=AsyncMock) as mock_order:
            _run(engine._accumulate(hedged_position, "YES", 0.45))
            mock_order.assert_not_called()

    def test_price_improvement_required(self, engine, hedged_position):
        """Must improve price by $0.02 before rebuying same side."""
        hedged_position.last_yes_buy_time = time.time() - 10
        hedged_position.last_yes_buy_price = 0.45

        engine._positions = {hedged_position.slug: hedged_position}
        with patch.object(engine, "_place_order", new_callable=AsyncMock) as mock_order:
            _run(engine._accumulate(hedged_position, "YES", 0.44))
            mock_order.assert_not_called()

    def test_sufficient_price_improvement_allows(self, engine, hedged_position):
        """$0.02+ improvement allows rebuy."""
        hedged_position.last_yes_buy_time = time.time() - 10
        hedged_position.last_yes_buy_price = 0.45

        engine._positions = {hedged_position.slug: hedged_position}
        with patch.object(engine, "_place_order", new_callable=AsyncMock, return_value="order123") as mock_order, \
             patch.object(engine, "_get_global_exposure", return_value=0):
            _run(engine._accumulate(hedged_position, "YES", 0.42))
            mock_order.assert_called_once()

    def test_max_fills_per_side_blocks(self, engine, hedged_position):
        """Cannot exceed MAX_FILLS_PER_SIDE on one side."""
        hedged_position.yes_orders = MAX_FILLS_PER_SIDE
        hedged_position.last_yes_buy_time = time.time() - 10

        engine._positions = {hedged_position.slug: hedged_position}
        with patch.object(engine, "_place_order", new_callable=AsyncMock) as mock_order:
            _run(engine._accumulate(hedged_position, "YES", 0.40))
            mock_order.assert_not_called()

    def test_per_window_exposure_cap(self, engine):
        """Cannot exceed MAX_EXPOSURE_PER_WINDOW per window."""
        pos = WindowPosition(
            asset="BTC", timeframe="15m", window_start=int(time.time()), slug="s",
            yes_shares=500, yes_cost=MAX_EXPOSURE_PER_WINDOW,
            no_shares=0, no_cost=0,
        )

        engine._positions = {pos.slug: pos}
        with patch.object(engine, "_place_order", new_callable=AsyncMock) as mock_order:
            _run(engine._accumulate(pos, "NO", 0.45))
            mock_order.assert_not_called()

    def test_global_exposure_cap(self, engine, hedged_position):
        """Cannot exceed MAX_GLOBAL_EXPOSURE across all windows."""
        hedged_position.last_no_buy_time = time.time() - 10
        hedged_position.last_no_buy_price = 0.50
        engine._positions = {hedged_position.slug: hedged_position}

        with patch.object(engine, "_get_global_exposure", return_value=MAX_GLOBAL_EXPOSURE), \
             patch.object(engine, "_place_order", new_callable=AsyncMock) as mock_order:
            _run(engine._accumulate(hedged_position, "NO", 0.40))
            mock_order.assert_not_called()

    def test_first_buy_on_side_skips_price_improvement(self, engine):
        """First buy on a side (last_buy_price=0) skips price improvement check."""
        pos = WindowPosition(
            asset="BTC", timeframe="15m",
            window_start=int(time.time()), slug="s",
        )
        engine._positions = {pos.slug: pos}

        with patch.object(engine, "_place_order", new_callable=AsyncMock, return_value="ord1") as mock_order, \
             patch.object(engine, "_get_global_exposure", return_value=0):
            _run(engine._accumulate(pos, "YES", 0.45))
            mock_order.assert_called_once()

    def test_buy_amount_capped_by_remaining(self, engine):
        """Buy amount capped; if remaining < $1, no order placed."""
        pos = WindowPosition(
            asset="BTC", timeframe="15m",
            window_start=int(time.time()), slug="s",
            yes_shares=450, yes_cost=MAX_EXPOSURE_PER_WINDOW - 2.0,
            no_shares=0, no_cost=0,
        )
        engine._positions = {pos.slug: pos}

        with patch.object(engine, "_place_order", new_callable=AsyncMock, return_value="ord1") as mock_order, \
             patch.object(engine, "_get_global_exposure", return_value=0):
            _run(engine._accumulate(pos, "YES", 0.45))
            mock_order.assert_not_called()


# ═══════════════════════════════════════════════════════════════
# EARLY EXIT CONDITIONS
# ═══════════════════════════════════════════════════════════════

class TestEarlyExit:

    def test_identifies_yes_as_loser(self, engine, hedged_position):
        """When YES price < NO price, YES is the losing side."""
        engine._positions = {hedged_position.slug: hedged_position}

        with patch.object(engine, "_sell_order", new_callable=AsyncMock, return_value="ord1") as mock_sell, \
             patch.object(engine, "_save_active_window"):
            _run(engine._early_exit(hedged_position, yes_price=0.30, no_price=0.70))

        mock_sell.assert_called_once()
        assert mock_sell.call_args[0][1] == "YES"
        assert hedged_position.early_exit_done is True
        assert hedged_position.early_exit_side == "YES"

    def test_identifies_no_as_loser(self, engine, hedged_position):
        """When NO price < YES price, NO is the losing side."""
        engine._positions = {hedged_position.slug: hedged_position}

        with patch.object(engine, "_sell_order", new_callable=AsyncMock, return_value="ord1") as mock_sell, \
             patch.object(engine, "_save_active_window"):
            _run(engine._early_exit(hedged_position, yes_price=0.70, no_price=0.30))

        assert mock_sell.call_args[0][1] == "NO"
        assert hedged_position.early_exit_side == "NO"

    def test_skips_if_already_done(self, engine, hedged_position):
        """Only one early exit per window."""
        hedged_position.early_exit_done = True

        with patch.object(engine, "_sell_order", new_callable=AsyncMock) as mock_sell:
            _run(engine._early_exit(hedged_position, 0.30, 0.70))
            mock_sell.assert_not_called()

    def test_skips_if_not_hedged(self, engine, basic_position):
        """Cannot early exit if not holding both sides."""
        with patch.object(engine, "_sell_order", new_callable=AsyncMock) as mock_sell:
            _run(engine._early_exit(basic_position, 0.30, 0.70))
            mock_sell.assert_not_called()

    def test_skips_if_price_too_low(self, engine, hedged_position):
        """Don't sell if losing side < EARLY_EXIT_MIN_PRICE."""
        with patch.object(engine, "_sell_order", new_callable=AsyncMock) as mock_sell:
            _run(engine._early_exit(hedged_position, yes_price=0.03, no_price=0.97))
            mock_sell.assert_not_called()
        assert hedged_position.early_exit_done is False

    def test_sells_95_pct_of_shares(self, engine, hedged_position):
        """Should sell ~95% of the losing side's shares."""
        engine._positions = {hedged_position.slug: hedged_position}

        with patch.object(engine, "_sell_order", new_callable=AsyncMock, return_value="ord1") as mock_sell, \
             patch.object(engine, "_save_active_window"):
            _run(engine._early_exit(hedged_position, yes_price=0.30, no_price=0.70))

        sell_shares = mock_sell.call_args[0][3]
        expected = round(hedged_position.yes_shares * 0.95, 2)
        assert abs(sell_shares - expected) < 0.1

    def test_resets_on_failed_sell(self, engine, hedged_position):
        """If sell order fails, early_exit_done should be reset for retry."""
        engine._positions = {hedged_position.slug: hedged_position}

        with patch.object(engine, "_sell_order", new_callable=AsyncMock, return_value=None):
            _run(engine._early_exit(hedged_position, yes_price=0.30, no_price=0.70))

        assert hedged_position.early_exit_done is False


# ═══════════════════════════════════════════════════════════════
# POSITION LIMITS
# ═══════════════════════════════════════════════════════════════

class TestPositionLimits:

    def test_max_exposure_per_window(self):
        assert MAX_EXPOSURE_PER_WINDOW == 100.00

    def test_max_global_exposure(self):
        assert MAX_GLOBAL_EXPOSURE == 500.00

    def test_max_daily_loss(self):
        assert MAX_DAILY_LOSS == 50.00

    def test_max_fills_per_side(self):
        assert MAX_FILLS_PER_SIDE == 15

    def test_fill_size_usd(self):
        assert FILL_SIZE_USD == 4.00

    def test_buy_threshold(self):
        assert BUY_THRESHOLD == 0.48

    def test_global_exposure_calculation(self, engine):
        p1 = WindowPosition(
            asset="BTC", timeframe="15m", window_start=1000, slug="s1",
            yes_shares=50, yes_cost=20, no_shares=50, no_cost=20,
        )
        p2 = WindowPosition(
            asset="ETH", timeframe="5m", window_start=1000, slug="s2",
            yes_shares=30, yes_cost=15, no_shares=30, no_cost=15,
        )
        engine._positions = {"s1": p1, "s2": p2}
        assert abs(engine._get_global_exposure() - 70.0) < 0.01


# ═══════════════════════════════════════════════════════════════
# SCALP DETECTION
# ═══════════════════════════════════════════════════════════════

class TestScalpDetection:

    def test_scalp_constants(self):
        assert SCALP_MULTIPLIER == 11.0
        assert SCALP_MIN_PROFIT == 10.00

    def test_no_scalp_without_fills(self, engine, basic_position):
        with patch.object(engine, "_sell_order", new_callable=AsyncMock) as mock_sell:
            _run(engine._check_scalps(basic_position, 0.50, 0.50))
            mock_sell.assert_not_called()


# ═══════════════════════════════════════════════════════════════
# MODULE CONSTANTS
# ═══════════════════════════════════════════════════════════════

class TestModuleConstants:

    def test_assets_contain_btc_eth_sol(self):
        for asset in ["BTC", "ETH", "SOL"]:
            assert asset in ASSETS

    def test_timeframes(self):
        assert "5m" in TIMEFRAMES
        assert "15m" in TIMEFRAMES
        assert TIMEFRAMES["5m"]["seconds"] == 300
        assert TIMEFRAMES["15m"]["seconds"] == 900

    def test_accumulation_timing(self):
        assert ACCUMULATION_START == 5
        assert ACCUMULATION_END_BEFORE == 30

    def test_early_exit_timing(self):
        assert EARLY_EXIT_SECONDS_BEFORE == 180
        assert EARLY_EXIT_END_BEFORE == 10
        assert EARLY_EXIT_MIN_PRICE == 0.05

    def test_cooldown_and_improvement(self):
        assert COOLDOWN_PER_SIDE == 3.0
        assert PRICE_IMPROVEMENT == 0.02


# ═══════════════════════════════════════════════════════════════
# ENGINE STATS
# ═══════════════════════════════════════════════════════════════

class TestEngineStats:

    def test_get_active_positions_empty(self, engine):
        assert engine.get_active_positions() == []

    def test_get_active_positions_filters_resolved(self, engine):
        p1 = WindowPosition(asset="BTC", timeframe="15m", window_start=1000, slug="s1")
        p2 = WindowPosition(asset="ETH", timeframe="5m", window_start=1000, slug="s2",
                            status="resolved")
        engine._positions = {"s1": p1, "s2": p2}
        assert len(engine.get_active_positions()) == 1
