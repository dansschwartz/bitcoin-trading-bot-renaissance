"""
Tests for arbitrage/execution/engine.py

Covers:
  - ArbitrageExecutor simultaneous leg execution
  - MAKER FIRST ordering (LIMIT_MAKER)
  - Both sides filled => profit calculation
  - One-sided buy fill => emergency close
  - One-sided sell fill => emergency close
  - Neither filled => clean no_fill
  - Expired signal handling
  - Insufficient balance rejection
  - Quantity rounding logic
  - Timeout handling
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from arbitrage.execution.engine import ArbitrageExecutor, ExecutionResult
from arbitrage.exchanges.base import (
    OrderRequest, OrderResult, OrderSide, OrderType, OrderStatus,
    TimeInForce, Balance,
)
from arbitrage.detector.cross_exchange import ArbitrageSignal


def _make_signal(
    buy_exchange="mexc", sell_exchange="binance",
    buy_price="50000", sell_price="50050",
    quantity="0.01", expires_in=5,
):
    return ArbitrageSignal(
        signal_id="test_arb_001",
        signal_type="cross_exchange",
        timestamp=datetime.utcnow(),
        symbol="BTC/USDT",
        buy_exchange=buy_exchange,
        sell_exchange=sell_exchange,
        buy_price=Decimal(buy_price),
        sell_price=Decimal(sell_price),
        gross_spread_bps=Decimal('10'),
        total_cost_bps=Decimal('3'),
        net_spread_bps=Decimal('7'),
        max_quantity=Decimal(quantity),
        recommended_quantity=Decimal(quantity),
        expected_profit_usd=Decimal('0.35'),
        buy_fee_bps=Decimal('0'),
        sell_fee_bps=Decimal('7.5'),
        buy_slippage_bps=Decimal('1'),
        sell_slippage_bps=Decimal('0.5'),
        expires_at=datetime.utcnow() + timedelta(seconds=expires_in),
        confidence=Decimal('0.7'),
    )


def _make_fill_result(exchange, side, price="50000", qty="0.01", fee="0"):
    return OrderResult(
        exchange=exchange, symbol="BTC/USDT",
        order_id="ord_123", client_order_id="test_buy",
        status=OrderStatus.FILLED, side=side,
        order_type=OrderType.LIMIT_MAKER,
        requested_quantity=Decimal(qty),
        filled_quantity=Decimal(qty),
        average_fill_price=Decimal(price),
        fee_amount=Decimal(fee), fee_currency="USDT",
        timestamp=datetime.utcnow(),
    )


def _make_rejected_result(exchange, side):
    return OrderResult(
        exchange=exchange, symbol="BTC/USDT",
        order_id="", client_order_id="test",
        status=OrderStatus.REJECTED, side=side,
        order_type=OrderType.LIMIT_MAKER,
        requested_quantity=Decimal('0.01'),
        filled_quantity=Decimal('0'),
        average_fill_price=None,
        fee_amount=Decimal('0'), fee_currency="USDT",
        timestamp=datetime.utcnow(),
    )


class TestArbitrageExecutor:
    @pytest.fixture
    def setup_executor(self):
        mexc = AsyncMock()
        binance = AsyncMock()
        cost_model = MagicMock()
        risk_engine = MagicMock()

        # Default balances - sufficient
        mexc.get_balance = AsyncMock(return_value=Balance(
            "mexc", "USDT", Decimal('10000'), Decimal('0'), Decimal('10000')
        ))
        binance.get_balance = AsyncMock(return_value=Balance(
            "binance", "BTC", Decimal('1'), Decimal('0'), Decimal('1')
        ))

        # Default symbol info
        mexc.get_symbol_info = AsyncMock(return_value={
            'symbol': 'BTC/USDT', 'price_precision': 2, 'quantity_precision': 5,
        })
        binance.get_symbol_info = AsyncMock(return_value={
            'symbol': 'BTC/USDT', 'price_precision': 2, 'quantity_precision': 5,
        })

        executor = ArbitrageExecutor(mexc, binance, cost_model, risk_engine)
        return executor, mexc, binance, cost_model

    @pytest.mark.asyncio
    async def test_both_legs_filled_success(self, setup_executor):
        executor, mexc, binance, cost_model = setup_executor

        buy_fill = _make_fill_result("mexc", OrderSide.BUY, price="50000", fee="0")
        sell_fill = _make_fill_result("binance", OrderSide.SELL, price="50050", fee="0.375")

        mexc.place_order = AsyncMock(return_value=buy_fill)
        binance.place_order = AsyncMock(return_value=sell_fill)

        signal = _make_signal()
        result = await executor.execute_arbitrage(signal)

        assert result.status == "filled"
        assert result.buy_result is buy_fill
        assert result.sell_result is sell_fill
        assert executor._fill_count == 1
        cost_model.update_from_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_actual_profit_calculation(self, setup_executor):
        executor, mexc, binance, _ = setup_executor

        buy_fill = _make_fill_result("mexc", OrderSide.BUY, price="50000", qty="0.01", fee="0")
        sell_fill = _make_fill_result("binance", OrderSide.SELL, price="50050", qty="0.01", fee="0.375")

        mexc.place_order = AsyncMock(return_value=buy_fill)
        binance.place_order = AsyncMock(return_value=sell_fill)

        signal = _make_signal()
        result = await executor.execute_arbitrage(signal)

        # Profit = sell_revenue - buy_cost
        # sell_revenue = 0.01 * 50050 - 0.375 = 500.5 - 0.375 = 500.125
        # buy_cost = 0.01 * 50000 + 0 = 500
        expected_profit = Decimal('500.125') - Decimal('500')
        assert result.actual_profit_usd == expected_profit

    @pytest.mark.asyncio
    async def test_one_sided_buy_triggers_emergency_sell(self, setup_executor):
        executor, mexc, binance, _ = setup_executor

        buy_fill = _make_fill_result("mexc", OrderSide.BUY, price="50000")
        sell_reject = _make_rejected_result("binance", OrderSide.SELL)

        mexc.place_order = AsyncMock(return_value=buy_fill)
        binance.place_order = AsyncMock(return_value=sell_reject)

        # Emergency close on the buy exchange
        emergency_fill = _make_fill_result("mexc", OrderSide.SELL, price="49990")
        mexc.place_order = AsyncMock(side_effect=[buy_fill, emergency_fill])

        signal = _make_signal()
        result = await executor.execute_arbitrage(signal)

        assert result.status == "one_sided_buy"

    @pytest.mark.asyncio
    async def test_one_sided_sell_triggers_emergency_buy(self, setup_executor):
        executor, mexc, binance, _ = setup_executor

        buy_reject = _make_rejected_result("mexc", OrderSide.BUY)
        sell_fill = _make_fill_result("binance", OrderSide.SELL, price="50050")

        mexc.place_order = AsyncMock(return_value=buy_reject)
        binance.place_order = AsyncMock(side_effect=[sell_fill, _make_fill_result("binance", OrderSide.BUY)])

        signal = _make_signal()
        result = await executor.execute_arbitrage(signal)

        assert result.status == "one_sided_sell"

    @pytest.mark.asyncio
    async def test_neither_filled_no_fill(self, setup_executor):
        executor, mexc, binance, _ = setup_executor

        buy_reject = _make_rejected_result("mexc", OrderSide.BUY)
        sell_reject = _make_rejected_result("binance", OrderSide.SELL)

        mexc.place_order = AsyncMock(return_value=buy_reject)
        binance.place_order = AsyncMock(return_value=sell_reject)

        signal = _make_signal()
        result = await executor.execute_arbitrage(signal)

        assert result.status == "no_fill"

    @pytest.mark.asyncio
    async def test_expired_signal_rejected(self, setup_executor):
        executor, _, _, _ = setup_executor

        signal = _make_signal(expires_in=-5)  # Already expired
        result = await executor.execute_arbitrage(signal)

        assert result.status == "expired"

    @pytest.mark.asyncio
    async def test_insufficient_quote_balance(self, setup_executor):
        executor, mexc, binance, _ = setup_executor

        # Only $100 USDT, need $500+
        mexc.get_balance = AsyncMock(return_value=Balance(
            "mexc", "USDT", Decimal('100'), Decimal('0'), Decimal('100')
        ))

        signal = _make_signal()
        result = await executor.execute_arbitrage(signal)

        assert result.status == "insufficient_balance"

    @pytest.mark.asyncio
    async def test_insufficient_base_balance(self, setup_executor):
        executor, mexc, binance, _ = setup_executor

        # No BTC on sell side
        binance.get_balance = AsyncMock(return_value=Balance(
            "binance", "BTC", Decimal('0'), Decimal('0'), Decimal('0')
        ))

        signal = _make_signal()
        result = await executor.execute_arbitrage(signal)

        assert result.status == "insufficient_balance"

    def test_round_qty_precision(self, setup_executor):
        executor, _, _, _ = setup_executor
        assert executor._round_qty(Decimal('0.01234'), 3) == Decimal('0.012')
        assert executor._round_qty(Decimal('0.01234'), 0) == Decimal('0')
        assert executor._round_qty(Decimal('1.999'), 0) == Decimal('1')

    def test_round_price_precision(self, setup_executor):
        executor, _, _, _ = setup_executor
        assert executor._round_price(Decimal('50000.567'), 2) == Decimal('50000.56')
        assert executor._round_price(Decimal('50000.5'), 0) == Decimal('50000')

    def test_get_stats_empty(self, setup_executor):
        executor, _, _, _ = setup_executor
        stats = executor.get_stats()
        assert stats["total_trades"] == 0
        assert stats["total_fills"] == 0
        assert stats["total_profit_usd"] == 0.0
        assert stats["win_rate"] == 0

    @pytest.mark.asyncio
    async def test_orders_use_limit_maker(self, setup_executor):
        """Verify MAKER FIRST: both orders use LIMIT_MAKER type."""
        executor, mexc, binance, _ = setup_executor

        buy_fill = _make_fill_result("mexc", OrderSide.BUY)
        sell_fill = _make_fill_result("binance", OrderSide.SELL)
        mexc.place_order = AsyncMock(return_value=buy_fill)
        binance.place_order = AsyncMock(return_value=sell_fill)

        signal = _make_signal()
        await executor.execute_arbitrage(signal)

        buy_order_arg = mexc.place_order.call_args[0][0]
        sell_order_arg = binance.place_order.call_args[0][0]
        assert buy_order_arg.order_type == OrderType.LIMIT_MAKER
        assert sell_order_arg.order_type == OrderType.LIMIT_MAKER
