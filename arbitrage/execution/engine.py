"""
Arbitrage Execution Engine — executes ArbitrageSignal trades
simultaneously on both exchanges.

CRITICAL PRINCIPLES:
1. BOTH LEGS MUST EXECUTE. Partial fill = unhedged directional exposure.
2. MAKER FIRST. LIMIT_MAKER orders for zero fee on MEXC.
3. SPEED. Both orders placed concurrently via asyncio.
4. VERIFY. If one side fills and other doesn't → emergency close.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Optional, List

from ..exchanges.base import (
    OrderRequest, OrderResult, OrderSide, OrderType, OrderStatus, TimeInForce,
)
from ..detector.cross_exchange import ArbitrageSignal

logger = logging.getLogger("arb.execution")


@dataclass
class ExecutionResult:
    trade_id: str
    status: str                      # filled, one_sided_buy, one_sided_sell, no_fill, expired, timeout
    signal: ArbitrageSignal
    buy_result: Optional[OrderResult] = None
    sell_result: Optional[OrderResult] = None
    emergency_close: Optional[OrderResult] = None
    actual_profit_usd: Decimal = Decimal('0')
    realized_cost_bps: Decimal = Decimal('0')
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ArbitrageExecutor:

    FILL_TIMEOUT_SECONDS = 3.0
    EMERGENCY_CLOSE_SECONDS = 5.0

    def __init__(self, mexc_client, binance_client, cost_model, risk_engine):
        self.clients = {
            "mexc": mexc_client,
            "binance": binance_client,
        }
        self.costs = cost_model
        self.risk = risk_engine
        self._active_trades: Dict[str, ExecutionResult] = {}
        self._completed_trades: List[ExecutionResult] = []
        self._total_profit = Decimal('0')
        self._trade_count = 0
        self._fill_count = 0

    async def execute_arbitrage(self, signal: ArbitrageSignal) -> ExecutionResult:
        """Execute a cross-exchange arbitrage trade."""
        trade_id = signal.signal_id

        # Check expiry
        if datetime.utcnow() > signal.expires_at:
            logger.debug(f"Signal expired: {trade_id}")
            return ExecutionResult(trade_id=trade_id, status="expired", signal=signal)

        buy_client = self.clients[signal.buy_exchange]
        sell_client = self.clients[signal.sell_exchange]

        # Verify inventory
        base, quote = signal.symbol.split('/')
        buy_balance = await buy_client.get_balance(quote)
        sell_balance = await sell_client.get_balance(base)

        required_quote = signal.recommended_quantity * signal.buy_price
        required_base = signal.recommended_quantity

        if buy_balance.free < required_quote:
            logger.debug(f"Insufficient {quote} on {signal.buy_exchange}: "
                        f"need {float(required_quote):.2f}, have {float(buy_balance.free):.2f}")
            return ExecutionResult(trade_id=trade_id, status="insufficient_balance", signal=signal)

        if sell_balance.free < required_base:
            logger.debug(f"Insufficient {base} on {signal.sell_exchange}: "
                        f"need {float(required_base):.6f}, have {float(sell_balance.free):.6f}")
            return ExecutionResult(trade_id=trade_id, status="insufficient_balance", signal=signal)

        # Round to exchange precision
        buy_info = await buy_client.get_symbol_info(signal.symbol)
        sell_info = await sell_client.get_symbol_info(signal.symbol)

        buy_qty = self._round_qty(signal.recommended_quantity, buy_info.get('quantity_precision', 8))
        sell_qty = buy_qty  # MUST be same on both sides

        buy_price = self._round_price(signal.buy_price, buy_info.get('price_precision', 8))
        sell_price = self._round_price(signal.sell_price, sell_info.get('price_precision', 8))

        if buy_qty <= 0 or sell_qty <= 0:
            logger.warning(
                f"Rejected {trade_id}: quantity_too_small "
                f"(buy_qty={float(buy_qty):.8f}, sell_qty={float(sell_qty):.8f}, "
                f"recommended={float(signal.recommended_quantity):.8f})"
            )
            return ExecutionResult(trade_id=trade_id, status="quantity_too_small", signal=signal)

        logger.info(
            f"EXECUTING: {signal.symbol} | "
            f"BUY {float(buy_qty):.6f} @ {float(buy_price):.2f} on {signal.buy_exchange} | "
            f"SELL {float(sell_qty):.6f} @ {float(sell_price):.2f} on {signal.sell_exchange} | "
            f"Expected profit: ${float(signal.expected_profit_usd):.2f}"
        )

        buy_order = OrderRequest(
            exchange=signal.buy_exchange,
            symbol=signal.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT_MAKER,
            quantity=buy_qty,
            price=buy_price,
            time_in_force=TimeInForce.GTX,
            client_order_id=f"{trade_id}_buy",
        )

        sell_order = OrderRequest(
            exchange=signal.sell_exchange,
            symbol=signal.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT_MAKER,
            quantity=sell_qty,
            price=sell_price,
            time_in_force=TimeInForce.GTX,
            client_order_id=f"{trade_id}_sell",
        )

        # FIRE BOTH SIMULTANEOUSLY
        try:
            buy_result, sell_result = await asyncio.wait_for(
                asyncio.gather(
                    buy_client.place_order(buy_order),
                    sell_client.place_order(sell_order),
                    return_exceptions=True,
                ),
                timeout=self.FILL_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(f"Execution timeout: {trade_id}")
            await self._cancel_both(buy_client, sell_client, buy_order, sell_order)
            return ExecutionResult(trade_id=trade_id, status="timeout", signal=signal)

        self._trade_count += 1
        return await self._process_results(trade_id, signal, buy_result, sell_result)

    async def _process_results(
        self, trade_id: str, signal: ArbitrageSignal,
        buy_result, sell_result,
    ) -> ExecutionResult:
        buy_ok = (
            isinstance(buy_result, OrderResult)
            and buy_result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)
        )
        sell_ok = (
            isinstance(sell_result, OrderResult)
            and sell_result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)
        )

        if buy_ok and sell_ok:
            # SUCCESS
            self._fill_count += 1
            actual_profit = self._calculate_actual_profit(buy_result, sell_result)
            realized_cost = self._calculate_realized_cost(buy_result, sell_result, signal)
            self._total_profit += actual_profit

            # Feed cost model
            self.costs.update_from_execution({
                "symbol": signal.symbol,
                "exchange": signal.buy_exchange,
                "quantity": buy_result.filled_quantity,
                "estimated_cost_bps": signal.total_cost_bps,
                "realized_cost_bps": realized_cost,
            })

            result = ExecutionResult(
                trade_id=trade_id, status="filled", signal=signal,
                buy_result=buy_result, sell_result=sell_result,
                actual_profit_usd=actual_profit, realized_cost_bps=realized_cost,
            )
            self._completed_trades.append(result)

            logger.info(
                f"ARB FILLED: {signal.symbol} | "
                f"Buy: {float(buy_result.filled_quantity):.6f} @ {float(buy_result.average_fill_price or 0):.2f} "
                f"({signal.buy_exchange}) | "
                f"Sell: {float(sell_result.filled_quantity):.6f} @ {float(sell_result.average_fill_price or 0):.2f} "
                f"({signal.sell_exchange}) | "
                f"Profit: ${float(actual_profit):.4f} | "
                f"Buy fee: ${float(buy_result.fee_amount):.4f} | "
                f"Sell fee: ${float(sell_result.fee_amount):.4f}"
            )

            # CRITICAL: Verify MEXC charged 0 maker fee
            if signal.buy_exchange == "mexc" and buy_result.fee_amount > 0:
                logger.warning(f"MEXC CHARGED FEE on maker order! Fee: {buy_result.fee_amount}")
            if signal.sell_exchange == "mexc" and sell_result.fee_amount > 0:
                logger.warning(f"MEXC CHARGED FEE on maker order! Fee: {sell_result.fee_amount}")

            return result

        elif buy_ok and not sell_ok:
            # DANGER: Buy filled, sell didn't
            logger.error(f"ONE-SIDED FILL (BUY ONLY): {trade_id} — emergency selling")
            emergency = await self._emergency_close(
                signal.buy_exchange, signal.symbol,
                OrderSide.SELL, buy_result.filled_quantity
            )
            result = ExecutionResult(
                trade_id=trade_id, status="one_sided_buy", signal=signal,
                buy_result=buy_result, emergency_close=emergency,
            )
            self._completed_trades.append(result)
            return result

        elif sell_ok and not buy_ok:
            # DANGER: Sell filled, buy didn't — we're short
            logger.error(f"ONE-SIDED FILL (SELL ONLY): {trade_id} — emergency buying")
            emergency = await self._emergency_close(
                signal.sell_exchange, signal.symbol,
                OrderSide.BUY, sell_result.filled_quantity
            )
            result = ExecutionResult(
                trade_id=trade_id, status="one_sided_sell", signal=signal,
                sell_result=sell_result, emergency_close=emergency,
            )
            self._completed_trades.append(result)
            return result

        else:
            # Neither filled — clean, no risk
            logger.debug(f"Both orders unfilled: {trade_id}")
            result = ExecutionResult(trade_id=trade_id, status="no_fill", signal=signal)
            self._completed_trades.append(result)
            return result

    async def _emergency_close(
        self, exchange: str, symbol: str, side: OrderSide, quantity: Decimal
    ) -> Optional[OrderResult]:
        """Emergency market order to close unhedged exposure."""
        client = self.clients[exchange]
        emergency_order = OrderRequest(
            exchange=exchange,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            time_in_force=TimeInForce.IOC,
            client_order_id=f"emergency_{int(datetime.utcnow().timestamp()*1000)}",
        )
        try:
            return await asyncio.wait_for(
                client.place_order(emergency_order),
                timeout=self.EMERGENCY_CLOSE_SECONDS,
            )
        except Exception as e:
            logger.critical(f"EMERGENCY CLOSE FAILED: {e}")
            return None

    async def _cancel_both(self, buy_client, sell_client, buy_order, sell_order):
        try:
            await asyncio.gather(
                buy_client.cancel_order(buy_order.symbol, buy_order.client_order_id or ""),
                sell_client.cancel_order(sell_order.symbol, sell_order.client_order_id or ""),
                return_exceptions=True,
            )
        except Exception:
            pass

    def _calculate_actual_profit(self, buy: OrderResult, sell: OrderResult) -> Decimal:
        if not buy.average_fill_price or not sell.average_fill_price:
            return Decimal('0')
        buy_cost = buy.filled_quantity * buy.average_fill_price + buy.fee_amount
        sell_revenue = sell.filled_quantity * sell.average_fill_price - sell.fee_amount
        return sell_revenue - buy_cost

    def _calculate_realized_cost(
        self, buy: OrderResult, sell: OrderResult, signal: ArbitrageSignal
    ) -> Decimal:
        if not buy.average_fill_price or not sell.average_fill_price:
            return Decimal('0')
        mid = (signal.buy_price + signal.sell_price) / 2
        if mid <= 0:
            return Decimal('0')
        buy_slip = abs(buy.average_fill_price - signal.buy_price) / mid * 10000
        sell_slip = abs(sell.average_fill_price - signal.sell_price) / mid * 10000
        buy_fee_bps = (buy.fee_amount / (buy.filled_quantity * buy.average_fill_price) * 10000
                       if buy.filled_quantity > 0 and buy.average_fill_price > 0 else Decimal('0'))
        sell_fee_bps = (sell.fee_amount / (sell.filled_quantity * sell.average_fill_price) * 10000
                        if sell.filled_quantity > 0 and sell.average_fill_price > 0 else Decimal('0'))
        return buy_fee_bps + sell_fee_bps + buy_slip + sell_slip

    def _round_qty(self, qty: Decimal, precision) -> Decimal:
        precision = int(precision)
        if precision <= 0:
            return qty.quantize(Decimal('1'), rounding=ROUND_DOWN)
        quant = Decimal(10) ** -precision
        return qty.quantize(quant, rounding=ROUND_DOWN)

    def _round_price(self, price: Decimal, precision) -> Decimal:
        precision = int(precision)
        if precision <= 0:
            return price.quantize(Decimal('1'), rounding=ROUND_DOWN)
        quant = Decimal(10) ** -precision
        return price.quantize(quant, rounding=ROUND_DOWN)

    def get_stats(self) -> dict:
        wins = [t for t in self._completed_trades if t.actual_profit_usd > 0]
        losses = [t for t in self._completed_trades if t.actual_profit_usd < 0]
        one_sided = [t for t in self._completed_trades if "one_sided" in t.status]
        return {
            "total_trades": self._trade_count,
            "total_fills": self._fill_count,
            "total_profit_usd": float(self._total_profit),
            "wins": len(wins),
            "losses": len(losses),
            "one_sided_events": len(one_sided),
            "win_rate": len(wins) / max(1, len(wins) + len(losses)),
            "avg_profit_per_fill": float(self._total_profit / max(1, self._fill_count)),
        }
