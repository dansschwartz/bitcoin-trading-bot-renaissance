"""
Tests for arbitrage/orderbook/unified_book.py

Covers:
  - UnifiedPairView freshness, tradeable status, cross-exchange spread
  - UnifiedBookManager initialization, book updates, status reporting
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from arbitrage.exchanges.base import OrderBook, OrderBookLevel, OrderSide
from arbitrage.orderbook.unified_book import (
    UnifiedPairView, UnifiedBookManager, PHASE_1_PAIRS,
)


def _make_book(exchange: str, symbol: str, bid: str, ask: str,
               bid_qty: str = "1.0", ask_qty: str = "1.0"):
    return OrderBook(
        exchange=exchange, symbol=symbol, timestamp=datetime.utcnow(),
        bids=[OrderBookLevel(Decimal(bid), Decimal(bid_qty))],
        asks=[OrderBookLevel(Decimal(ask), Decimal(ask_qty))],
    )


class TestUnifiedPairView:
    def _fresh_view(self, mexc_bid="50000", mexc_ask="50001",
                    binance_bid="50002", binance_ask="50003"):
        view = UnifiedPairView(symbol="BTC/USDT")
        view.mexc_book = _make_book("mexc", "BTC/USDT", mexc_bid, mexc_ask)
        view.binance_book = _make_book("binance", "BTC/USDT", binance_bid, binance_ask)
        view.mexc_last_update = datetime.utcnow()
        view.binance_last_update = datetime.utcnow()
        return view

    def test_is_fresh_when_recent(self):
        view = self._fresh_view()
        assert view.is_fresh is True

    def test_is_not_fresh_when_stale(self):
        view = self._fresh_view()
        view.mexc_last_update = datetime.utcnow() - timedelta(seconds=5)
        assert view.is_fresh is False

    def test_is_tradeable_both_books_fresh(self):
        view = self._fresh_view()
        assert view.is_tradeable is True

    def test_is_not_tradeable_missing_mexc_book(self):
        view = UnifiedPairView(symbol="BTC/USDT")
        view.binance_book = _make_book("binance", "BTC/USDT", "50000", "50001")
        assert view.is_tradeable is False

    def test_is_not_tradeable_missing_binance_book(self):
        view = UnifiedPairView(symbol="BTC/USDT")
        view.mexc_book = _make_book("mexc", "BTC/USDT", "50000", "50001")
        assert view.is_tradeable is False

    def test_is_not_tradeable_stale_data(self):
        view = self._fresh_view()
        view.binance_last_update = datetime.utcnow() - timedelta(seconds=10)
        assert view.is_tradeable is False

    def test_spread_buy_mexc_sell_binance(self):
        """MEXC ask < Binance bid => profitable to buy MEXC, sell Binance."""
        view = self._fresh_view(
            mexc_bid="49990", mexc_ask="50000",
            binance_bid="50050", binance_ask="50060",
        )
        spread = view.get_cross_exchange_spread()
        assert spread is not None
        assert spread["direction"] == "buy_mexc_sell_binance"
        assert spread["buy_exchange"] == "mexc"
        assert spread["sell_exchange"] == "binance"
        assert spread["gross_spread_bps"] > 0

    def test_spread_buy_binance_sell_mexc(self):
        """Binance ask < MEXC bid => profitable to buy Binance, sell MEXC."""
        view = self._fresh_view(
            mexc_bid="50050", mexc_ask="50060",
            binance_bid="49990", binance_ask="50000",
        )
        spread = view.get_cross_exchange_spread()
        assert spread is not None
        assert spread["direction"] == "buy_binance_sell_mexc"
        assert spread["buy_exchange"] == "binance"
        assert spread["sell_exchange"] == "mexc"

    def test_spread_no_opportunity(self):
        """Overlapping books => no profitable spread."""
        view = self._fresh_view(
            mexc_bid="50000", mexc_ask="50005",
            binance_bid="49998", binance_ask="50003",
        )
        spread = view.get_cross_exchange_spread()
        # Neither direction has positive spread
        assert spread is None

    def test_spread_returns_none_when_not_tradeable(self):
        view = UnifiedPairView(symbol="BTC/USDT")
        assert view.get_cross_exchange_spread() is None

    def test_spread_includes_depth_info(self):
        view = self._fresh_view(
            mexc_bid="49990", mexc_ask="50000",
            binance_bid="50050", binance_ask="50060",
        )
        spread = view.get_cross_exchange_spread()
        assert "buy_depth" in spread
        assert "sell_depth" in spread
        assert spread["buy_depth"] >= Decimal('0')


class TestUnifiedBookManager:
    @pytest.fixture
    def mock_clients(self):
        mexc = AsyncMock()
        binance = AsyncMock()
        mexc.subscribe_order_book = AsyncMock()
        binance.subscribe_order_book = AsyncMock()
        mexc.get_order_book = AsyncMock()
        return mexc, binance

    def test_initialization_with_custom_pairs(self, mock_clients):
        mexc, binance = mock_clients
        pairs = ["BTC/USDT", "ETH/USDT"]
        mgr = UnifiedBookManager(mexc, binance, pairs=pairs)
        assert mgr.monitored_pairs == pairs

    def test_initialization_default_pairs(self, mock_clients):
        mexc, binance = mock_clients
        mgr = UnifiedBookManager(mexc, binance)
        assert mgr.monitored_pairs == PHASE_1_PAIRS

    @pytest.mark.asyncio
    async def test_on_mexc_update(self, mock_clients):
        mexc, binance = mock_clients
        mgr = UnifiedBookManager(mexc, binance, pairs=["BTC/USDT"])
        mgr.pairs["BTC/USDT"] = UnifiedPairView(symbol="BTC/USDT")

        book = _make_book("mexc", "BTC/USDT", "50000", "50001")
        await mgr._on_mexc_update("BTC/USDT", book)

        assert mgr.pairs["BTC/USDT"].mexc_book is book
        assert mgr.pairs["BTC/USDT"].mexc_update_count == 1

    @pytest.mark.asyncio
    async def test_on_binance_update(self, mock_clients):
        mexc, binance = mock_clients
        mgr = UnifiedBookManager(mexc, binance, pairs=["BTC/USDT"])
        mgr.pairs["BTC/USDT"] = UnifiedPairView(symbol="BTC/USDT")

        book = _make_book("binance", "BTC/USDT", "50000", "50001")
        await mgr._on_binance_update("BTC/USDT", book)

        assert mgr.pairs["BTC/USDT"].binance_book is book
        assert mgr.pairs["BTC/USDT"].binance_update_count == 1

    @pytest.mark.asyncio
    async def test_on_update_with_bar_aggregator(self, mock_clients):
        mexc, binance = mock_clients
        mock_bar_agg = MagicMock()
        mgr = UnifiedBookManager(mexc, binance, pairs=["BTC/USDT"],
                                 bar_aggregator=mock_bar_agg)
        mgr.pairs["BTC/USDT"] = UnifiedPairView(symbol="BTC/USDT")

        book = _make_book("mexc", "BTC/USDT", "50000", "50001")
        await mgr._on_mexc_update("BTC/USDT", book)

        mock_bar_agg.on_orderbook_snapshot.assert_called_once()

    def test_get_status(self, mock_clients):
        mexc, binance = mock_clients
        mgr = UnifiedBookManager(mexc, binance, pairs=["BTC/USDT", "ETH/USDT"])
        mgr.pairs["BTC/USDT"] = UnifiedPairView(symbol="BTC/USDT")
        mgr.pairs["ETH/USDT"] = UnifiedPairView(symbol="ETH/USDT")
        mgr.pairs["BTC/USDT"].mexc_update_count = 10
        mgr.pairs["BTC/USDT"].binance_update_count = 8

        status = mgr.get_status()
        assert status["total_pairs"] == 2
        assert status["total_updates"] == 18
        assert "BTC/USDT" in status["pairs"]

    @pytest.mark.asyncio
    async def test_stop(self, mock_clients):
        mexc, binance = mock_clients
        mgr = UnifiedBookManager(mexc, binance, pairs=["BTC/USDT"])
        mgr._running = True
        mock_task = MagicMock()
        mgr._validation_task = mock_task

        await mgr.stop()
        assert mgr._running is False
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_update_unknown_pair_ignored(self, mock_clients):
        mexc, binance = mock_clients
        mgr = UnifiedBookManager(mexc, binance, pairs=["BTC/USDT"])
        mgr.pairs["BTC/USDT"] = UnifiedPairView(symbol="BTC/USDT")

        book = _make_book("mexc", "UNKNOWN/PAIR", "100", "101")
        await mgr._on_mexc_update("UNKNOWN/PAIR", book)
        # Should not crash; pair not in mgr.pairs so nothing happens
        assert "UNKNOWN/PAIR" not in mgr.pairs
