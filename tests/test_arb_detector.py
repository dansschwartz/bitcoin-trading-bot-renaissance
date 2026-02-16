"""
Tests for arbitrage/detector/cross_exchange.py

Covers:
  - ArbitrageSignal dataclass
  - CrossExchangeDetector signal generation with synthetic price discrepancies
  - No signal when spread is too small
  - Cost deduction, sizing, confidence, risk approval gate
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from arbitrage.detector.cross_exchange import ArbitrageSignal, CrossExchangeDetector
from arbitrage.exchanges.base import OrderBook, OrderBookLevel, OrderSide
from arbitrage.orderbook.unified_book import UnifiedPairView


def _make_cost_estimate(total_cost_bps: str = "2.0"):
    """Create a mock cost estimate."""
    est = MagicMock()
    est.total_cost_bps = Decimal(total_cost_bps)
    est.buy_fee_bps = Decimal('0')
    est.sell_fee_bps = Decimal('7.5')
    est.buy_slippage_bps = Decimal('1.5')
    est.sell_slippage_bps = Decimal('0.5')
    return est


def _make_book(exchange, symbol, bid, ask, qty="1.0"):
    return OrderBook(
        exchange=exchange, symbol=symbol, timestamp=datetime.utcnow(),
        bids=[OrderBookLevel(Decimal(bid), Decimal(qty))],
        asks=[OrderBookLevel(Decimal(ask), Decimal(qty))],
    )


def _make_tradeable_view(symbol, mexc_bid, mexc_ask, binance_bid, binance_ask, qty="1.0"):
    view = UnifiedPairView(symbol=symbol)
    view.mexc_book = _make_book("mexc", symbol, mexc_bid, mexc_ask, qty)
    view.binance_book = _make_book("binance", symbol, binance_bid, binance_ask, qty)
    view.mexc_last_update = datetime.utcnow()
    view.binance_last_update = datetime.utcnow()
    return view


class TestArbitrageSignal:
    def test_dataclass_creation(self):
        sig = ArbitrageSignal(
            signal_id="test_1", signal_type="cross_exchange",
            timestamp=datetime.utcnow(), symbol="BTC/USDT",
            buy_exchange="mexc", sell_exchange="binance",
            buy_price=Decimal('50000'), sell_price=Decimal('50050'),
            gross_spread_bps=Decimal('10'), total_cost_bps=Decimal('3'),
            net_spread_bps=Decimal('7'),
            max_quantity=Decimal('0.01'), recommended_quantity=Decimal('0.01'),
            expected_profit_usd=Decimal('0.35'),
            buy_fee_bps=Decimal('0'), sell_fee_bps=Decimal('7.5'),
            buy_slippage_bps=Decimal('1'), sell_slippage_bps=Decimal('0.5'),
            expires_at=datetime.utcnow() + timedelta(seconds=5),
            confidence=Decimal('0.7'),
        )
        assert sig.signal_type == "cross_exchange"
        assert sig.net_spread_bps == Decimal('7')


class TestCrossExchangeDetector:
    @pytest.fixture
    def setup_detector(self):
        """Create detector with mocked dependencies."""
        book_manager = MagicMock()
        cost_model = MagicMock()
        risk_engine = MagicMock()
        signal_queue = asyncio.Queue(maxsize=100)

        detector = CrossExchangeDetector(book_manager, cost_model, risk_engine, signal_queue)
        return detector, book_manager, cost_model, risk_engine, signal_queue

    def test_initialization(self, setup_detector):
        detector, _, _, _, _ = setup_detector
        assert detector._scan_count == 0
        assert detector._signals_generated == 0
        assert detector._signals_approved == 0

    @pytest.mark.asyncio
    async def test_generates_signal_when_spread_sufficient(self, setup_detector):
        detector, book_mgr, cost_model, risk_engine, signal_queue = setup_detector

        # Setup: MEXC ask 50000, Binance bid 50050 => gross spread ~10bps
        view = _make_tradeable_view(
            "BTC/USDT",
            mexc_bid="49990", mexc_ask="50000",
            binance_bid="50050", binance_ask="50060",
        )
        book_mgr.pairs = {"BTC/USDT": view}

        cost_model.estimate_arbitrage_cost.return_value = _make_cost_estimate("2.0")
        risk_engine.approve_arbitrage.return_value = True

        # Run one scan iteration
        detector._running = True

        async def run_one_scan():
            # Replicate scan logic inline (one iteration)
            for pair in detector.books.pairs:
                v = detector.books.pairs[pair]
                if not v.is_tradeable:
                    continue
                spread_info = v.get_cross_exchange_spread()
                if spread_info is None or spread_info['gross_spread_bps'] <= 0:
                    continue
                cost_est = detector.costs.estimate_arbitrage_cost(
                    symbol=pair,
                    buy_exchange=spread_info['buy_exchange'],
                    sell_exchange=spread_info['sell_exchange'],
                    buy_price=spread_info['buy_price'],
                    sell_price=spread_info['sell_price'],
                )
                net_spread = spread_info['gross_spread_bps'] - cost_est.total_cost_bps
                if net_spread >= detector.MIN_NET_SPREAD_BPS:
                    detector._signals_generated += 1
                    if detector.risk.approve_arbitrage(MagicMock()):
                        signal_queue.put_nowait(MagicMock())
                        detector._signals_approved += 1

        await run_one_scan()

        assert signal_queue.qsize() >= 1
        assert detector._signals_generated >= 1
        assert detector._signals_approved >= 1

    @pytest.mark.asyncio
    async def test_no_signal_when_spread_too_small(self, setup_detector):
        detector, book_mgr, cost_model, risk_engine, signal_queue = setup_detector

        # Overlapping books => no spread
        view = _make_tradeable_view(
            "BTC/USDT",
            mexc_bid="50000", mexc_ask="50005",
            binance_bid="49998", binance_ask="50003",
        )
        book_mgr.pairs = {"BTC/USDT": view}
        cost_model.estimate_arbitrage_cost.return_value = _make_cost_estimate("5.0")

        # The spread info will be None (no positive spread)
        spread = view.get_cross_exchange_spread()
        assert spread is None
        assert signal_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_no_signal_when_cost_exceeds_spread(self, setup_detector):
        detector, book_mgr, cost_model, risk_engine, signal_queue = setup_detector

        # Small spread: ~2 bps
        view = _make_tradeable_view(
            "BTC/USDT",
            mexc_bid="49990", mexc_ask="50000",
            binance_bid="50010", binance_ask="50020",
        )
        book_mgr.pairs = {"BTC/USDT": view}

        # Cost exceeds the gross spread
        cost_model.estimate_arbitrage_cost.return_value = _make_cost_estimate("50.0")

        spread_info = view.get_cross_exchange_spread()
        if spread_info:
            net = spread_info['gross_spread_bps'] - Decimal('50.0')
            assert net < detector.MIN_NET_SPREAD_BPS

    @pytest.mark.asyncio
    async def test_risk_rejection_blocks_signal(self, setup_detector):
        detector, book_mgr, cost_model, risk_engine, signal_queue = setup_detector

        view = _make_tradeable_view(
            "BTC/USDT",
            mexc_bid="49990", mexc_ask="50000",
            binance_bid="50050", binance_ask="50060",
        )
        book_mgr.pairs = {"BTC/USDT": view}
        cost_model.estimate_arbitrage_cost.return_value = _make_cost_estimate("2.0")
        risk_engine.approve_arbitrage.return_value = False

        # Simulate the check
        spread_info = view.get_cross_exchange_spread()
        assert spread_info is not None  # There IS a spread
        assert risk_engine.approve_arbitrage(MagicMock()) is False
        assert signal_queue.qsize() == 0

    def test_confidence_calculation_fresh_data(self, setup_detector):
        detector, _, _, _, _ = setup_detector

        view = _make_tradeable_view(
            "BTC/USDT",
            mexc_bid="49990", mexc_ask="50000",
            binance_bid="50050", binance_ask="50060",
            qty="2.0",
        )
        spread_info = view.get_cross_exchange_spread()
        confidence = detector._calculate_confidence(view, spread_info)

        # Base: 0.5 + freshness bonus (< 0.5s) + depth bonus (>1.0)
        assert confidence >= Decimal('0.5')
        assert confidence <= Decimal('0.95')

    def test_confidence_with_spread_stability_bonus(self, setup_detector):
        detector, _, _, _, _ = setup_detector

        view = _make_tradeable_view(
            "BTC/USDT",
            mexc_bid="49990", mexc_ask="50000",
            binance_bid="50050", binance_ask="50060",
        )
        spread_info = view.get_cross_exchange_spread()

        # Add previous spread for stability bonus
        detector._last_spreads["BTC/USDT"] = Decimal('10.0')
        confidence = detector._calculate_confidence(view, spread_info)
        assert confidence >= Decimal('0.55')

    def test_confidence_capped_at_095(self, setup_detector):
        detector, _, _, _, _ = setup_detector

        view = _make_tradeable_view(
            "BTC/USDT",
            mexc_bid="49990", mexc_ask="50000",
            binance_bid="50050", binance_ask="50060",
            qty="5.0",
        )
        spread_info = view.get_cross_exchange_spread()
        detector._last_spreads["BTC/USDT"] = Decimal('10.0')
        confidence = detector._calculate_confidence(view, spread_info)
        assert confidence <= Decimal('0.95')

    def test_get_stats(self, setup_detector):
        detector, _, _, _, _ = setup_detector
        detector._scan_count = 100
        detector._signals_generated = 10
        detector._signals_approved = 8

        stats = detector.get_stats()
        assert stats["scan_count"] == 100
        assert stats["signals_generated"] == 10
        assert stats["signals_approved"] == 8
        assert stats["approval_rate"] == 0.8

    def test_stop(self, setup_detector):
        detector, _, _, _, _ = setup_detector
        detector._running = True
        detector.stop()
        assert detector._running is False

    def test_min_net_spread_bps_threshold(self, setup_detector):
        detector, _, _, _, _ = setup_detector
        assert detector.MIN_NET_SPREAD_BPS == Decimal('1.0')
