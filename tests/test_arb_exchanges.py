"""
Tests for arbitrage exchange clients: base.py, mexc_client.py, binance_client.py

Covers:
  - Base data classes, enums, OrderBook properties, liquidity calculation
  - MEXC client: connect, order placement (LIMIT_MAKER), paper fill, 0% maker fee,
    symbol conversion, error handling, balances
  - Binance client: connect, BNB fee discount, order placement, symbol conversion,
    error handling, paper fill, balances
"""
import asyncio
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from arbitrage.exchanges.base import (
    OrderBook, OrderBookLevel, OrderRequest, OrderResult, OrderSide, OrderType,
    OrderStatus, TimeInForce, SpeedTier, Balance, FundingRate, Trade, ExchangeClient,
)
from arbitrage.exchanges.mexc_client import MEXCClient
from arbitrage.exchanges.binance_client import BinanceClient


# ========================================================================
# BASE DATA CLASSES AND ENUMS
# ========================================================================

class TestOrderBookLevel:
    def test_decimal_fields(self):
        level = OrderBookLevel(price=Decimal('50000.50'), quantity=Decimal('0.5'))
        assert level.price == Decimal('50000.50')
        assert level.quantity == Decimal('0.5')


class TestOrderBook:
    _DEFAULT_BIDS = "default"
    _DEFAULT_ASKS = "default"

    def _make_book(self, bids=_DEFAULT_BIDS, asks=_DEFAULT_ASKS):
        if bids is self._DEFAULT_BIDS:
            bids = [
                OrderBookLevel(Decimal('50000'), Decimal('1.0')),
                OrderBookLevel(Decimal('49999'), Decimal('2.0')),
            ]
        if asks is self._DEFAULT_ASKS:
            asks = [
                OrderBookLevel(Decimal('50001'), Decimal('1.5')),
                OrderBookLevel(Decimal('50002'), Decimal('0.5')),
            ]
        return OrderBook(
            exchange="test", symbol="BTC/USDT",
            timestamp=datetime.utcnow(), bids=bids, asks=asks,
        )

    def test_best_bid(self):
        book = self._make_book()
        assert book.best_bid == Decimal('50000')

    def test_best_ask(self):
        book = self._make_book()
        assert book.best_ask == Decimal('50001')

    def test_best_bid_qty(self):
        book = self._make_book()
        assert book.best_bid_qty == Decimal('1.0')

    def test_best_ask_qty(self):
        book = self._make_book()
        assert book.best_ask_qty == Decimal('1.5')

    def test_mid_price(self):
        book = self._make_book()
        expected = (Decimal('50000') + Decimal('50001')) / 2
        assert book.mid_price == expected

    def test_spread_bps(self):
        book = self._make_book()
        mid = (Decimal('50000') + Decimal('50001')) / 2
        expected = (Decimal('50001') - Decimal('50000')) / mid * 10000
        assert book.spread_bps == expected

    def test_empty_book_best_bid_is_none(self):
        book = self._make_book(bids=[], asks=[])
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price is None
        assert book.spread_bps is None

    def test_available_liquidity_buy_within_impact(self):
        book = self._make_book()
        # max_impact_bps = 5 means max_price = 50001 * (1 + 5/10000) = 50026.0005
        liq = book.available_liquidity_at_impact(OrderSide.BUY, Decimal('5'))
        # Both ask levels (50001 and 50002) are within impact
        assert liq == Decimal('2.0')

    def test_available_liquidity_sell_within_impact(self):
        book = self._make_book()
        liq = book.available_liquidity_at_impact(OrderSide.SELL, Decimal('5'))
        # Both bid levels (50000 and 49999) should be within impact
        assert liq == Decimal('3.0')

    def test_available_liquidity_zero_impact(self):
        book = self._make_book()
        liq = book.available_liquidity_at_impact(OrderSide.BUY, Decimal('0'))
        # max_price = 50001 * 1 = 50001, so only first ask at 50001
        assert liq == Decimal('1.5')

    def test_available_liquidity_empty_side(self):
        book = self._make_book(asks=[])
        liq = book.available_liquidity_at_impact(OrderSide.BUY, Decimal('10'))
        assert liq == Decimal('0')


class TestOrderRequest:
    def test_auto_generates_client_order_id(self):
        req = OrderRequest(
            exchange="mexc", symbol="BTC/USDT",
            side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=Decimal('0.01'),
        )
        assert req.client_order_id is not None
        assert req.client_order_id.startswith("arb_")

    def test_custom_client_order_id_preserved(self):
        req = OrderRequest(
            exchange="mexc", symbol="BTC/USDT",
            side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=Decimal('0.01'),
            client_order_id="my_custom_id",
        )
        assert req.client_order_id == "my_custom_id"


class TestEnums:
    def test_order_side_values(self):
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_speed_tier_values(self):
        assert SpeedTier.MAKER_MAKER.value == "maker_maker"
        assert SpeedTier.TAKER_MAKER.value == "taker_maker"
        assert SpeedTier.TAKER_TAKER.value == "taker_taker"

    def test_order_type_limit_maker(self):
        assert OrderType.LIMIT_MAKER.value == "limit_maker"


# ========================================================================
# MEXC CLIENT
# ========================================================================

class TestMEXCClient:
    @pytest.fixture
    def client(self):
        return MEXCClient(api_key="test_key", api_secret="test_secret", paper_trading=True)

    @pytest.fixture
    def live_client(self):
        return MEXCClient(api_key="test_key", api_secret="test_secret", paper_trading=False)

    @pytest.mark.asyncio
    async def test_connect_creates_exchange(self, client):
        with patch('arbitrage.exchanges.mexc_client.ccxt_async') as mock_ccxt:
            mock_exchange = AsyncMock()
            mock_exchange.markets = {"BTC/USDT": {}}
            mock_exchange.load_markets = AsyncMock()
            mock_ccxt.mexc.return_value = mock_exchange

            await client.connect()

            mock_ccxt.mexc.assert_called_once()
            mock_exchange.load_markets.assert_called_once()
            assert client._exchange is mock_exchange

    @pytest.mark.asyncio
    async def test_connect_without_api_key(self):
        client = MEXCClient(api_key="", api_secret="", paper_trading=True)
        with patch('arbitrage.exchanges.mexc_client.ccxt_async') as mock_ccxt:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_ccxt.mexc.return_value = mock_exchange

            await client.connect()

            call_args = mock_ccxt.mexc.call_args[0][0]
            assert 'apiKey' not in call_args

    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        mock_exchange = AsyncMock()
        client._exchange = mock_exchange

        await client.disconnect()

        mock_exchange.close.assert_called_once()
        assert client._exchange is None

    @pytest.mark.asyncio
    async def test_get_order_book(self, client):
        mock_exchange = AsyncMock()
        mock_exchange.fetch_order_book = AsyncMock(return_value={
            'bids': [[50000, 1.0], [49999, 2.0]],
            'asks': [[50001, 1.5], [50002, 0.5]],
            'timestamp': 1700000000000,
        })
        client._exchange = mock_exchange

        book = await client.get_order_book("BTC/USDT")

        assert book.exchange == "mexc"
        assert book.symbol == "BTC/USDT"
        assert book.best_bid == Decimal('50000')
        assert book.best_ask == Decimal('50001')

    @pytest.mark.asyncio
    async def test_trading_fees_zero_maker(self, client):
        fees = await client.get_trading_fees("BTC/USDT")
        assert fees["maker"] == Decimal('0.0000')
        assert fees["taker"] == Decimal('0.0005')

    @pytest.mark.asyncio
    async def test_paper_fill_limit_maker_zero_fee(self, client):
        order = OrderRequest(
            exchange="mexc", symbol="BTC/USDT",
            side=OrderSide.BUY, order_type=OrderType.LIMIT_MAKER,
            quantity=Decimal('0.01'), price=Decimal('50000'),
        )
        with patch('random.random', return_value=0.5):  # Within 85% fill rate
            result = await client.place_order(order)

        assert result.exchange == "mexc"
        assert result.status == OrderStatus.FILLED
        assert result.fee_amount == Decimal('0')  # 0% maker fee

    @pytest.mark.asyncio
    async def test_paper_fill_market_order_has_taker_fee(self, client):
        # Provide a book for price reference
        client._last_books["BTC/USDT"] = OrderBook(
            exchange="mexc", symbol="BTC/USDT", timestamp=datetime.utcnow(),
            bids=[OrderBookLevel(Decimal('49999'), Decimal('1'))],
            asks=[OrderBookLevel(Decimal('50001'), Decimal('1'))],
        )
        order = OrderRequest(
            exchange="mexc", symbol="BTC/USDT",
            side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=Decimal('0.01'),
        )
        result = await client.place_order(order)

        assert result.status == OrderStatus.FILLED
        expected_fee = Decimal('0.01') * Decimal('50001') * Decimal('0.0005')
        assert result.fee_amount == expected_fee

    @pytest.mark.asyncio
    async def test_paper_fill_limit_maker_rejection(self, client):
        order = OrderRequest(
            exchange="mexc", symbol="BTC/USDT",
            side=OrderSide.BUY, order_type=OrderType.LIMIT_MAKER,
            quantity=Decimal('0.01'), price=Decimal('50000'),
        )
        # random > 0.85 causes rejection
        with patch('random.random', return_value=0.90):
            result = await client.place_order(order)
        assert result.status == OrderStatus.CANCELLED
        assert result.filled_quantity == Decimal('0')

    @pytest.mark.asyncio
    async def test_live_order_rejected_on_exception(self, live_client):
        mock_exchange = AsyncMock()
        mock_exchange.create_order = AsyncMock(side_effect=Exception("Insufficient balance"))
        live_client._exchange = mock_exchange

        order = OrderRequest(
            exchange="mexc", symbol="BTC/USDT",
            side=OrderSide.BUY, order_type=OrderType.LIMIT_MAKER,
            quantity=Decimal('0.01'), price=Decimal('50000'),
        )
        result = await live_client.place_order(order)

        assert result.status == OrderStatus.REJECTED
        assert "error" in result.raw_response

    @pytest.mark.asyncio
    async def test_live_order_limit_maker_uses_gtx(self, live_client):
        mock_exchange = AsyncMock()
        mock_exchange.create_order = AsyncMock(return_value={
            'id': '12345', 'status': 'closed', 'filled': 0.01,
            'amount': 0.01, 'side': 'buy', 'average': 50000,
            'fee': {'cost': 0, 'currency': 'USDT'},
        })
        live_client._exchange = mock_exchange

        order = OrderRequest(
            exchange="mexc", symbol="BTC/USDT",
            side=OrderSide.BUY, order_type=OrderType.LIMIT_MAKER,
            quantity=Decimal('0.01'), price=Decimal('50000'),
        )
        result = await live_client.place_order(order)

        call_kwargs = mock_exchange.create_order.call_args
        assert call_kwargs.kwargs['params']['timeInForce'] == 'GTX'
        assert call_kwargs.kwargs['type'] == 'limit'

    def test_symbol_conversion_normalized_to_mexc(self, client):
        assert client._normalized_to_mexc_sym("BTC/USDT") == "BTCUSDT"
        assert client._normalized_to_mexc_sym("ETH/BTC") == "ETHBTC"

    def test_symbol_conversion_mexc_to_normalized(self, client):
        assert client._mexc_sym_to_normalized("BTCUSDT") == "BTC/USDT"
        assert client._mexc_sym_to_normalized("ETHBTC") == "ETH/BTC"
        assert client._mexc_sym_to_normalized("SOLUSDC") == "SOL/USDC"

    @pytest.mark.asyncio
    async def test_paper_balances(self, client):
        balances = await client.get_balances()
        assert "USDT" in balances
        assert balances["USDT"].free == Decimal('5000')
        assert balances["BTC"].total == Decimal('0.15')

    @pytest.mark.asyncio
    async def test_get_balance_missing_currency(self, client):
        balance = await client.get_balance("UNKNOWN_COIN")
        assert balance.free == Decimal('0')
        assert balance.total == Decimal('0')

    @pytest.mark.asyncio
    async def test_get_symbol_info_default(self, client):
        info = await client.get_symbol_info("UNKNOWN/PAIR")
        assert info['price_precision'] == 8
        assert info['min_quantity'] == Decimal('0')

    @pytest.mark.asyncio
    async def test_cancel_order_paper(self, client):
        result = await client.cancel_order("BTC/USDT", "12345")
        assert result is True

    @pytest.mark.asyncio
    async def test_funding_rate_fallback_on_error(self, client):
        mock_exchange = AsyncMock()
        mock_exchange.fetch_funding_rate = AsyncMock(side_effect=Exception("Not supported"))
        client._exchange = mock_exchange

        fr = await client.get_funding_rate("BTC/USDT")
        assert fr.current_rate == Decimal('0')
        assert fr.exchange == "mexc"


# ========================================================================
# BINANCE CLIENT
# ========================================================================

class TestBinanceClient:
    @pytest.fixture
    def client(self):
        return BinanceClient(api_key="test_key", api_secret="test_secret",
                             paper_trading=True, bnb_fee_discount=True)

    @pytest.fixture
    def client_no_bnb(self):
        return BinanceClient(api_key="", api_secret="",
                             paper_trading=True, bnb_fee_discount=False)

    @pytest.mark.asyncio
    async def test_connect_creates_exchange(self, client):
        with patch('arbitrage.exchanges.binance_client.ccxt_async') as mock_ccxt:
            mock_exchange = AsyncMock()
            mock_exchange.markets = {"BTC/USDT": {}}
            mock_exchange.load_markets = AsyncMock()
            mock_ccxt.binance.return_value = mock_exchange

            await client.connect()
            mock_exchange.load_markets.assert_called_once()

    @pytest.mark.asyncio
    async def test_bnb_fee_discount_applied(self, client):
        fees = await client.get_trading_fees("BTC/USDT")
        assert fees["maker"] == Decimal('0.00075')
        assert fees["taker"] == Decimal('0.00075')

    @pytest.mark.asyncio
    async def test_no_bnb_discount_higher_fee(self, client_no_bnb):
        fees = await client_no_bnb.get_trading_fees("BTC/USDT")
        assert fees["maker"] == Decimal('0.001')
        assert fees["taker"] == Decimal('0.001')

    @pytest.mark.asyncio
    async def test_paper_fill_with_bnb_discount(self, client):
        client._last_books["BTC/USDT"] = OrderBook(
            exchange="binance", symbol="BTC/USDT", timestamp=datetime.utcnow(),
            bids=[OrderBookLevel(Decimal('49999'), Decimal('1'))],
            asks=[OrderBookLevel(Decimal('50001'), Decimal('1'))],
        )
        order = OrderRequest(
            exchange="binance", symbol="BTC/USDT",
            side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=Decimal('0.01'),
        )
        with patch('random.random', return_value=0.5):
            result = await client.place_order(order)

        expected_fee = Decimal('0.01') * Decimal('50001') * Decimal('0.00075')
        assert result.fee_amount == expected_fee

    @pytest.mark.asyncio
    async def test_paper_fill_without_bnb_discount(self, client_no_bnb):
        client_no_bnb._last_books["BTC/USDT"] = OrderBook(
            exchange="binance", symbol="BTC/USDT", timestamp=datetime.utcnow(),
            bids=[OrderBookLevel(Decimal('49999'), Decimal('1'))],
            asks=[OrderBookLevel(Decimal('50001'), Decimal('1'))],
        )
        order = OrderRequest(
            exchange="binance", symbol="BTC/USDT",
            side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=Decimal('0.01'),
        )
        with patch('random.random', return_value=0.5):
            result = await client_no_bnb.place_order(order)

        expected_fee = Decimal('0.01') * Decimal('50001') * Decimal('0.001')
        assert result.fee_amount == expected_fee

    def test_binance_symbol_conversion_normalized_to_raw(self, client):
        assert client._normalized_to_binance_sym("BTC/USDT") == "btcusdt"
        assert client._normalized_to_binance_sym("ETH/BNB") == "ethbnb"

    def test_binance_symbol_conversion_raw_to_normalized(self, client):
        assert client._binance_sym_to_normalized("BTCUSDT") == "BTC/USDT"
        assert client._binance_sym_to_normalized("ETHBNB") == "ETH/BNB"
        assert client._binance_sym_to_normalized("SOLUSD") == "SOLUSD"  # No known quote

    @pytest.mark.asyncio
    async def test_paper_balances(self, client):
        balances = await client.get_balances()
        assert "USDT" in balances
        assert balances["USDT"].exchange == "binance"
        assert balances["BNB"].free == Decimal('5')

    @pytest.mark.asyncio
    async def test_live_order_error_returns_rejected(self):
        client = BinanceClient(paper_trading=False)
        mock_exchange = AsyncMock()
        mock_exchange.create_order = AsyncMock(side_effect=Exception("Rate limited"))
        client._exchange = mock_exchange

        order = OrderRequest(
            exchange="binance", symbol="BTC/USDT",
            side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=Decimal('0.01'), price=Decimal('50000'),
        )
        result = await client.place_order(order)
        assert result.status == OrderStatus.REJECTED
        assert result.exchange == "binance"

    @pytest.mark.asyncio
    async def test_parse_order_result_partial_fill(self):
        client = BinanceClient(paper_trading=False)
        raw = {
            'id': '999', 'status': 'open', 'filled': 0.005,
            'amount': 0.01, 'side': 'buy', 'average': 50000,
            'symbol': 'BTC/USDT', 'fee': {'cost': 0.1, 'currency': 'USDT'},
        }
        result = client._parse_order_result(raw, None)
        assert result.status == OrderStatus.PARTIALLY_FILLED
        assert result.filled_quantity == Decimal('0.005')

    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        mock_exchange = AsyncMock()
        client._exchange = mock_exchange
        await client.disconnect()
        mock_exchange.close.assert_called_once()
        assert client._exchange is None

    @pytest.mark.asyncio
    async def test_paper_limit_maker_rejection(self, client):
        order = OrderRequest(
            exchange="binance", symbol="BTC/USDT",
            side=OrderSide.BUY, order_type=OrderType.LIMIT_MAKER,
            quantity=Decimal('0.01'), price=Decimal('50000'),
        )
        with patch('random.random', return_value=0.95):  # > 0.90 threshold
            result = await client.place_order(order)
        assert result.status == OrderStatus.CANCELLED
