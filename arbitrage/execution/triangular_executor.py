"""
Triangular Arbitrage Executor — executes 3-leg cycles on a single exchange.

CRITICAL PRINCIPLES:
1. ALL 3 LEGS MUST COMPLETE. If any leg fails, unwind completed legs.
2. Sequential execution: Leg1 → Leg2 → Leg3 (each leg depends on previous).
3. LIMIT_MAKER for 0% fee on MEXC.
4. Speed: minimize time between legs to avoid price movement.
5. Paper trading: simulate fills from last known books.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import List, Optional

from ..exchanges.base import (
    OrderRequest, OrderResult, OrderSide, OrderType, OrderStatus, TimeInForce,
)

logger = logging.getLogger("arb.tri_executor")


@dataclass
class TriLegResult:
    """Result of a single leg execution."""
    leg_number: int
    symbol: str
    side: str
    status: str  # filled, failed, unwound
    order_result: Optional[OrderResult] = None
    quantity_in: Decimal = Decimal('0')
    quantity_out: Decimal = Decimal('0')


@dataclass
class TriExecutionResult:
    """Result of a complete triangular arb attempt."""
    trade_id: str
    status: str  # filled, partial_unwind, failed
    legs: List[TriLegResult] = field(default_factory=list)
    start_amount: Decimal = Decimal('0')
    end_amount: Decimal = Decimal('0')
    profit_usd: Decimal = Decimal('0')
    total_fees_usd: Decimal = Decimal('0')
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TriangularExecutor:
    """Executes 3-leg triangular arbitrage on a single exchange."""

    def __init__(self, mexc_client):
        self.client = mexc_client
        self._trade_count = 0
        self._fill_count = 0
        self._total_profit = Decimal('0')
        self._completed: List[TriExecutionResult] = []

    async def execute(self, path, start_currency: str,
                      trade_usd: Decimal) -> TriExecutionResult:
        """
        Execute a triangular arbitrage cycle.

        Args:
            path: List of (symbol, side, intermediate_currency) tuples
            start_currency: Currency we start and end with (e.g., "USDT")
            trade_usd: USD amount to trade
        """
        trade_id = f"tri_{start_currency}_{int(time.time() * 1000)}"
        self._trade_count += 1
        start_time = time.monotonic()

        legs: List[TriLegResult] = []

        # Convert USD amount to start_currency if not already USD-pegged
        if start_currency in ("USDT", "USDC", "BUSD"):
            current_amount = trade_usd
        else:
            # Get price of start_currency in USDT to convert
            usd_price = await self._get_price(f"{start_currency}/USDT", 'bid')
            if usd_price and usd_price > 0:
                current_amount = trade_usd / usd_price
            else:
                logger.error(f"Cannot convert ${float(trade_usd)} to {start_currency}")
                return self._build_result(trade_id, "failed", [], trade_usd,
                                          Decimal('0'), start_time)

        current_currency = start_currency
        initial_amount = current_amount  # Save for profit calculation

        logger.info(
            f"TRI EXECUTE {trade_id}: {' -> '.join(s[0] for s in path)} -> {start_currency} "
            f"| Starting: {float(current_amount):.8f} {start_currency} (${float(trade_usd):.2f})"
        )

        for i, (symbol, side, next_currency) in enumerate(path):
            leg_num = i + 1
            base, quote = symbol.split('/')

            # Determine order parameters based on side
            if side == 'buy':
                # Buying base with quote currency
                # current_amount is in quote currency, we want to buy base
                order_side = OrderSide.BUY
                # Get current price for the pair
                price = await self._get_price(symbol, 'ask')
                if price is None or price <= 0:
                    logger.error(f"TRI LEG {leg_num} FAILED: no price for {symbol}")
                    await self._unwind(legs, trade_id)
                    return self._build_result(trade_id, "failed", legs, trade_usd,
                                              Decimal('0'), start_time)

                quantity = current_amount / price  # How much base we can buy
                quantity = await self._round_quantity(symbol, quantity)

                if quantity <= 0:
                    logger.error(f"TRI LEG {leg_num} FAILED: quantity too small for {symbol}")
                    await self._unwind(legs, trade_id)
                    return self._build_result(trade_id, "failed", legs, trade_usd,
                                              Decimal('0'), start_time)
            else:
                # Selling base for quote currency
                # current_amount is in base currency
                order_side = OrderSide.SELL
                price = await self._get_price(symbol, 'bid')
                if price is None or price <= 0:
                    logger.error(f"TRI LEG {leg_num} FAILED: no price for {symbol}")
                    await self._unwind(legs, trade_id)
                    return self._build_result(trade_id, "failed", legs, trade_usd,
                                              Decimal('0'), start_time)

                quantity = current_amount
                quantity = await self._round_quantity(symbol, quantity)

                if quantity <= 0:
                    logger.error(f"TRI LEG {leg_num} FAILED: quantity too small for {symbol}")
                    await self._unwind(legs, trade_id)
                    return self._build_result(trade_id, "failed", legs, trade_usd,
                                              Decimal('0'), start_time)

            # Place the order — use LIMIT (not LIMIT_MAKER) to avoid rejection
            # when price moves. On MEXC, even taker fee is only 0.05% for spot.
            order = OrderRequest(
                exchange="mexc",
                symbol=symbol,
                side=order_side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=await self._round_price(symbol, price),
                time_in_force=TimeInForce.GTC,
                client_order_id=f"{trade_id}_leg{leg_num}",
            )

            try:
                result = await asyncio.wait_for(
                    self.client.place_order(order),
                    timeout=3.0,
                )
            except asyncio.TimeoutError:
                logger.error(f"TRI LEG {leg_num} TIMEOUT: {symbol} {side}")
                legs.append(TriLegResult(
                    leg_number=leg_num, symbol=symbol, side=side,
                    status="failed", quantity_in=current_amount,
                ))
                await self._unwind(legs, trade_id)
                return self._build_result(trade_id, "failed", legs, trade_usd,
                                          Decimal('0'), start_time)

            if result.status not in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                logger.error(
                    f"TRI LEG {leg_num} NOT FILLED: {symbol} {side} — status={result.status}"
                )
                legs.append(TriLegResult(
                    leg_number=leg_num, symbol=symbol, side=side,
                    status="failed", order_result=result,
                    quantity_in=current_amount,
                ))
                await self._unwind(legs, trade_id)
                return self._build_result(trade_id, "failed", legs, trade_usd,
                                          Decimal('0'), start_time)

            # Calculate output amount
            fill_price = result.average_fill_price or price
            fill_qty = result.filled_quantity

            if side == 'buy':
                quantity_out = fill_qty  # We received base currency
            else:
                quantity_out = fill_qty * fill_price  # We received quote currency

            quantity_out -= result.fee_amount  # Subtract fees

            logger.info(
                f"TRI LEG {leg_num} FILLED: {side.upper()} {symbol} | "
                f"qty={float(fill_qty):.8f} @ {float(fill_price):.6f} | "
                f"in={float(current_amount):.6f} {current_currency} -> "
                f"out={float(quantity_out):.6f} {next_currency} | "
                f"fee={float(result.fee_amount):.6f}"
            )

            legs.append(TriLegResult(
                leg_number=leg_num, symbol=symbol, side=side,
                status="filled", order_result=result,
                quantity_in=current_amount, quantity_out=quantity_out,
            ))

            current_amount = quantity_out
            current_currency = next_currency

        # All 3 legs completed!
        profit = current_amount - initial_amount
        self._fill_count += 1
        self._total_profit += profit

        logger.info(
            f"TRI COMPLETE {trade_id}: "
            f"Started {float(initial_amount):.8f} {start_currency} -> "
            f"Ended {float(current_amount):.8f} {start_currency} | "
            f"Profit: {float(profit):.8f} {start_currency} | "
            f"Time: {(time.monotonic() - start_time) * 1000:.0f}ms"
        )

        result = self._build_result(
            trade_id, "filled", legs, initial_amount, current_amount, start_time
        )
        self._completed.append(result)
        return result

    async def _unwind(self, completed_legs: List[TriLegResult], trade_id: str) -> None:
        """Unwind completed legs in reverse order to recover starting currency."""
        filled_legs = [l for l in completed_legs if l.status == "filled"]
        if not filled_legs:
            return

        logger.warning(f"TRI UNWINDING {trade_id}: {len(filled_legs)} legs to reverse")

        for leg in reversed(filled_legs):
            try:
                # Reverse the trade
                base, quote = leg.symbol.split('/')
                if leg.side == 'buy':
                    # We bought base — sell it back
                    reverse_side = OrderSide.SELL
                    reverse_qty = leg.quantity_out
                else:
                    # We sold base — buy it back
                    reverse_side = OrderSide.BUY
                    reverse_qty = leg.quantity_out

                reverse_qty = await self._round_quantity(leg.symbol, reverse_qty)
                if reverse_qty <= 0:
                    logger.error(f"TRI UNWIND: quantity too small for {leg.symbol}")
                    continue

                # Use market order for speed during unwind
                order = OrderRequest(
                    exchange="mexc",
                    symbol=leg.symbol,
                    side=reverse_side,
                    order_type=OrderType.MARKET,
                    quantity=reverse_qty,
                    time_in_force=TimeInForce.IOC,
                    client_order_id=f"{trade_id}_unwind_leg{leg.leg_number}",
                )

                result = await asyncio.wait_for(
                    self.client.place_order(order),
                    timeout=5.0,
                )
                leg.status = "unwound"
                logger.info(
                    f"TRI UNWIND leg {leg.leg_number}: {leg.symbol} "
                    f"{'SELL' if leg.side == 'buy' else 'BUY'} "
                    f"{float(reverse_qty):.8f} — {result.status.value}"
                )
            except Exception as e:
                logger.error(f"TRI UNWIND FAILED leg {leg.leg_number}: {e}")

    async def _get_price(self, symbol: str, side: str) -> Optional[Decimal]:
        """Get current best price for a symbol."""
        try:
            book = await self.client.get_order_book(symbol, depth=5)
            if side == 'ask':
                return book.best_ask
            return book.best_bid
        except Exception as e:
            logger.debug(f"Price fetch failed for {symbol}: {e}")
            return None

    async def _round_quantity(self, symbol: str, qty: Decimal) -> Decimal:
        """Round quantity to exchange precision."""
        try:
            info = await self.client.get_symbol_info(symbol)
            precision = int(info.get('quantity_precision', 8))
        except Exception:
            precision = 8
        if precision <= 0:
            return qty.quantize(Decimal('1'), rounding=ROUND_DOWN)
        quant = Decimal(10) ** -precision
        return qty.quantize(quant, rounding=ROUND_DOWN)

    async def _round_price(self, symbol: str, price: Decimal) -> Decimal:
        """Round price to exchange precision."""
        try:
            info = await self.client.get_symbol_info(symbol)
            precision = int(info.get('price_precision', 8))
        except Exception:
            precision = 8
        if precision <= 0:
            return price.quantize(Decimal('1'), rounding=ROUND_DOWN)
        quant = Decimal(10) ** -precision
        return price.quantize(quant, rounding=ROUND_DOWN)

    def _build_result(self, trade_id: str, status: str,
                      legs: List[TriLegResult], start_amount: Decimal,
                      end_amount: Decimal, start_time: float) -> TriExecutionResult:
        total_fees = sum(
            (l.order_result.fee_amount if l.order_result else Decimal('0'))
            for l in legs
        )
        return TriExecutionResult(
            trade_id=trade_id,
            status=status,
            legs=legs,
            start_amount=start_amount,
            end_amount=end_amount,
            profit_usd=end_amount - start_amount,
            total_fees_usd=total_fees,
            execution_time_ms=(time.monotonic() - start_time) * 1000,
        )

    def get_stats(self) -> dict:
        return {
            "total_trades": self._trade_count,
            "total_fills": self._fill_count,
            "total_profit_usd": float(self._total_profit),
            "win_rate": self._fill_count / max(1, self._trade_count),
        }
