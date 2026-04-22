"""
Tests for signals/multi_exchange_bridge.py — MultiExchangeBridge
================================================================
Covers: signal computation (momentum, dispersion, imbalance, funding),
symbol conversion, funding rate updates, and error handling.
All exchange/book manager dependencies are mocked.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
import pytest

from signals.multi_exchange_bridge import MultiExchangeBridge


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_level(price: float, quantity: float) -> MagicMock:
    """Create a mock order-book level with .quantity attribute."""
    level = MagicMock()
    level.quantity = Decimal(str(quantity))
    return level


def _make_pair_view(
    mexc_mid: float = 50000.0,
    binance_mid: float = 50010.0,
    is_tradeable: bool = True,
    mexc_bids: list = None,
    mexc_asks: list = None,
    binance_bids: list = None,
    binance_asks: list = None,
) -> MagicMock:
    """Create a mock pair view."""
    pair_view = MagicMock()
    pair_view.is_tradeable = is_tradeable

    pair_view.mexc_book = MagicMock()
    pair_view.mexc_book.mid_price = Decimal(str(mexc_mid)) if mexc_mid else None
    pair_view.mexc_book.bids = mexc_bids or [_make_level(mexc_mid - 1, 1.0) for _ in range(10)]
    pair_view.mexc_book.asks = mexc_asks or [_make_level(mexc_mid + 1, 1.0) for _ in range(10)]

    pair_view.binance_book = MagicMock()
    pair_view.binance_book.mid_price = Decimal(str(binance_mid)) if binance_mid else None
    pair_view.binance_book.bids = binance_bids or [_make_level(binance_mid - 1, 1.5) for _ in range(10)]
    pair_view.binance_book.asks = binance_asks or [_make_level(binance_mid + 1, 1.5) for _ in range(10)]

    return pair_view


def _make_book_manager(pairs: dict = None) -> MagicMock:
    """Create a mock UnifiedBookManager."""
    bm = MagicMock()
    bm.pairs = pairs or {}
    return bm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def book_manager():
    return _make_book_manager({
        "BTC/USDT": _make_pair_view(50000.0, 50010.0),
        "ETH/USDT": _make_pair_view(3000.0, 3001.0),
    })


@pytest.fixture
def bridge(book_manager):
    return MultiExchangeBridge(book_manager)


# ---------------------------------------------------------------------------
# Symbol conversion tests
# ---------------------------------------------------------------------------

class TestSymbolConversion:
    def test_btc_usd_to_usdt(self, bridge):
        assert bridge._to_usdt_symbol("BTC-USD") == "BTC/USDT"

    def test_eth_usd_to_usdt(self, bridge):
        assert bridge._to_usdt_symbol("ETH-USD") == "ETH/USDT"

    def test_sol_usd_to_usdt(self, bridge):
        assert bridge._to_usdt_symbol("SOL-USD") == "SOL/USDT"


# ---------------------------------------------------------------------------
# get_signals tests
# ---------------------------------------------------------------------------

class TestGetSignals:
    def test_returns_all_four_signal_keys(self, bridge):
        result = bridge.get_signals("BTC-USD", 49990.0, 50010.0)
        assert "cross_exchange_momentum" in result
        assert "price_dispersion" in result
        assert "aggregated_book_imbalance" in result
        assert "funding_rate_signal" in result

    def test_cross_exchange_momentum_positive(self, bridge):
        """Binance mid > Coinbase mid -> positive momentum signal."""
        # Binance = 50010, Coinbase bid=49980, ask=49990 -> mid=49985
        result = bridge.get_signals("BTC-USD", 49980.0, 49990.0)
        assert result["cross_exchange_momentum"] > 0.0

    def test_cross_exchange_momentum_negative(self, bridge):
        """Coinbase mid > Binance mid -> negative momentum signal."""
        result = bridge.get_signals("BTC-USD", 50050.0, 50070.0)
        # Coinbase mid = 50060, Binance = 50010 -> spread_pct < 0
        assert result["cross_exchange_momentum"] < 0.0

    def test_price_dispersion_negative_signal(self, bridge):
        """Dispersion is always negative (uncertainty indicator)."""
        result = bridge.get_signals("BTC-USD", 49990.0, 50010.0)
        assert result["price_dispersion"] <= 0.0

    def test_aggregated_book_imbalance_bid_heavy(self):
        """Bid-heavy books across exchanges -> positive imbalance."""
        bm = _make_book_manager({
            "BTC/USDT": _make_pair_view(
                mexc_mid=50000.0, binance_mid=50010.0,
                mexc_bids=[_make_level(49999.0, 10.0) for _ in range(10)],
                mexc_asks=[_make_level(50001.0, 1.0) for _ in range(10)],
                binance_bids=[_make_level(50009.0, 10.0) for _ in range(10)],
                binance_asks=[_make_level(50011.0, 1.0) for _ in range(10)],
            ),
        })
        bridge = MultiExchangeBridge(bm)
        result = bridge.get_signals(
            "BTC-USD",
            coinbase_bid=49990.0, coinbase_ask=50010.0,
            coinbase_bid_vol=100.0, coinbase_ask_vol=10.0,
        )
        assert result["aggregated_book_imbalance"] > 0.0

    def test_aggregated_book_imbalance_ask_heavy(self):
        """Ask-heavy books -> negative imbalance."""
        bm = _make_book_manager({
            "BTC/USDT": _make_pair_view(
                mexc_mid=50000.0, binance_mid=50010.0,
                mexc_bids=[_make_level(49999.0, 1.0) for _ in range(10)],
                mexc_asks=[_make_level(50001.0, 10.0) for _ in range(10)],
                binance_bids=[_make_level(50009.0, 1.0) for _ in range(10)],
                binance_asks=[_make_level(50011.0, 10.0) for _ in range(10)],
            ),
        })
        bridge = MultiExchangeBridge(bm)
        result = bridge.get_signals(
            "BTC-USD",
            coinbase_bid=49990.0, coinbase_ask=50010.0,
            coinbase_bid_vol=10.0, coinbase_ask_vol=100.0,
        )
        assert result["aggregated_book_imbalance"] < 0.0

    def test_funding_rate_contrarian_signal(self, bridge):
        """Positive cached funding -> negative (contrarian) signal."""
        bridge._funding_cache["BTC/USDT"] = 0.005  # 50 bps
        result = bridge.get_signals("BTC-USD", 49990.0, 50010.0)
        assert result["funding_rate_signal"] < 0.0

    def test_negative_funding_rate_signal(self, bridge):
        """Negative cached funding -> positive (contrarian) signal."""
        bridge._funding_cache["BTC/USDT"] = -0.005
        result = bridge.get_signals("BTC-USD", 49990.0, 50010.0)
        assert result["funding_rate_signal"] > 0.0

    def test_no_funding_cached_returns_zero(self, bridge):
        result = bridge.get_signals("BTC-USD", 49990.0, 50010.0)
        assert result["funding_rate_signal"] == 0.0

    def test_unknown_pair_returns_zeros(self, bridge):
        """Unknown product ID -> all zeros (pair not in book manager)."""
        result = bridge.get_signals("UNKNOWN-USD", 100.0, 101.0)
        assert result["cross_exchange_momentum"] == 0.0
        assert result["price_dispersion"] == 0.0
        assert result["aggregated_book_imbalance"] == 0.0
        assert result["funding_rate_signal"] == 0.0

    def test_untradeable_pair_returns_zeros(self):
        """Pair exists but is_tradeable=False -> zeros."""
        bm = _make_book_manager({
            "BTC/USDT": _make_pair_view(is_tradeable=False),
        })
        bridge = MultiExchangeBridge(bm)
        result = bridge.get_signals("BTC-USD", 49990.0, 50010.0)
        assert result == {
            "cross_exchange_momentum": 0.0,
            "price_dispersion": 0.0,
            "aggregated_book_imbalance": 0.0,
            "funding_rate_signal": 0.0,
        }

    def test_signals_bounded_minus_one_to_one(self, bridge):
        """All signal values should be in [-1, 1]."""
        result = bridge.get_signals("BTC-USD", 49990.0, 50010.0)
        for key, val in result.items():
            assert -1.0 <= val <= 1.0, f"{key}={val} out of bounds"


# ---------------------------------------------------------------------------
# update_funding_rates tests
# ---------------------------------------------------------------------------

class TestUpdateFundingRates:
    @pytest.mark.asyncio
    async def test_updates_cache_from_binance(self):
        bm = _make_book_manager({"BTC/USDT": _make_pair_view()})
        mock_binance = AsyncMock()
        mock_fr = MagicMock()
        mock_fr.current_rate = Decimal("0.001")
        mock_binance.get_funding_rate = AsyncMock(return_value=mock_fr)

        bridge = MultiExchangeBridge(bm, binance_client=mock_binance)
        await bridge.update_funding_rates()

        assert "BTC/USDT" in bridge._funding_cache
        assert abs(bridge._funding_cache["BTC/USDT"] - 0.001) < 1e-8

    @pytest.mark.asyncio
    async def test_averages_binance_and_mexc(self):
        bm = _make_book_manager({"BTC/USDT": _make_pair_view()})
        mock_binance = AsyncMock()
        mock_mexc = AsyncMock()

        fr_binance = MagicMock()
        fr_binance.current_rate = Decimal("0.002")
        mock_binance.get_funding_rate = AsyncMock(return_value=fr_binance)

        fr_mexc = MagicMock()
        fr_mexc.current_rate = Decimal("0.004")
        mock_mexc.get_funding_rate = AsyncMock(return_value=fr_mexc)

        bridge = MultiExchangeBridge(bm, mexc_client=mock_mexc, binance_client=mock_binance)
        await bridge.update_funding_rates()

        assert abs(bridge._funding_cache["BTC/USDT"] - 0.003) < 1e-8

    @pytest.mark.asyncio
    async def test_no_clients_noop(self):
        bm = _make_book_manager({"BTC/USDT": _make_pair_view()})
        bridge = MultiExchangeBridge(bm)
        await bridge.update_funding_rates()
        assert len(bridge._funding_cache) == 0

    @pytest.mark.asyncio
    async def test_exception_in_client_handled(self):
        bm = _make_book_manager({"BTC/USDT": _make_pair_view()})
        mock_binance = AsyncMock()
        mock_binance.get_funding_rate = AsyncMock(side_effect=Exception("API error"))

        bridge = MultiExchangeBridge(bm, binance_client=mock_binance)
        # Should not raise
        await bridge.update_funding_rates()
        # No rate cached since it failed
        assert bridge._funding_cache.get("BTC/USDT") is None

    @pytest.mark.asyncio
    async def test_fetch_count_increments(self):
        bm = _make_book_manager({"BTC/USDT": _make_pair_view()})
        mock_binance = AsyncMock()
        fr = MagicMock()
        fr.current_rate = Decimal("0.001")
        mock_binance.get_funding_rate = AsyncMock(return_value=fr)

        bridge = MultiExchangeBridge(bm, binance_client=mock_binance)
        assert bridge._funding_fetch_count == 0
        await bridge.update_funding_rates()
        assert bridge._funding_fetch_count == 1
        await bridge.update_funding_rates()
        assert bridge._funding_fetch_count == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_coinbase_prices(self, bridge):
        """coinbase_bid=0 and coinbase_ask=0 -> coinbase_mid=0."""
        result = bridge.get_signals("BTC-USD", 0.0, 0.0)
        assert result["cross_exchange_momentum"] == 0.0

    def test_none_mid_prices_handled(self):
        """None mid_price in book manager should not crash."""
        pair_view = _make_pair_view()
        pair_view.mexc_book.mid_price = None
        pair_view.binance_book.mid_price = None
        bm = _make_book_manager({"BTC/USDT": pair_view})
        bridge = MultiExchangeBridge(bm)
        result = bridge.get_signals("BTC-USD", 49990.0, 50010.0)
        # Should still return the default dict without crashing
        assert isinstance(result, dict)

    def test_empty_order_books(self):
        """Empty bids/asks lists in books."""
        pair_view = _make_pair_view()
        pair_view.mexc_book.bids = []
        pair_view.mexc_book.asks = []
        pair_view.binance_book.bids = []
        pair_view.binance_book.asks = []
        bm = _make_book_manager({"BTC/USDT": pair_view})
        bridge = MultiExchangeBridge(bm)
        result = bridge.get_signals("BTC-USD", 49990.0, 50010.0)
        assert isinstance(result, dict)
