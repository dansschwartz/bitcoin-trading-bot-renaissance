"""
Triangular Arbitrage Executor — executes 3-leg cycles on a single exchange.

CRITICAL PRINCIPLES:
1. ALL 3 LEGS MUST COMPLETE. If any leg fails, unwind completed legs.
2. Sequential execution: Leg1 → Leg2 → Leg3 (each leg depends on previous).
3. Speed: pre-fetch all prices concurrently, cache precision, minimize REST calls.
4. Paper trading: simulate fills from last known books.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple

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
        self._precision_cache: Dict[str, Tuple[int, int]] = {}  # symbol -> (price_prec, qty_prec)

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
            usd_price = await self._get_price(f"{start_currency}/USDT", 'bid')
            if usd_price and usd_price > 0:
                current_amount = trade_usd / usd_price
            else:
                logger.error(f"Cannot convert ${float(trade_usd)} to {start_currency}")
                return self._build_result(trade_id, "failed", [], trade_usd,
                                          Decimal('0'), start_time)

        current_currency = start_currency
        initial_amount = current_amount

        # === SPEED OPTIMIZATION ===
        # Pre-fetch all order books and precision info concurrently
        symbols = [s[0] for s in path]
        pre_fetch_start = time.monotonic()
        books, precisions = await self._pre_fetch_all(symbols, path)
        pre_fetch_ms = (time.monotonic() - pre_fetch_start) * 1000

        logger.info(
            f"TRI EXECUTE {trade_id}: {' -> '.join(s[0] for s in path)} -> {start_currency} "
            f"| Starting: {float(current_amount):.8f} {start_currency} (${float(trade_usd):.2f}) "
            f"| Pre-fetch: {pre_fetch_ms:.0f}ms"
        )

        for i, (symbol, side, next_currency) in enumerate(path):
            leg_num = i + 1
            base, quote = symbol.split('/')

            # Use pre-fetched price — LIMIT_MAKER rests in book:
            #   BUY at bid (top of bid book), SELL at ask (top of ask book)
            price_side = 'bid' if side == 'buy' else 'ask'
            price = books.get((symbol, price_side))
            if price is None or price <= 0:
                # Fallback: fetch fresh price for this leg
                price = await self._get_price(symbol, price_side)

            if price is None or price <= 0:
                logger.error(f"TRI LEG {leg_num} FAILED: no price for {symbol}")
                await self._unwind(legs, trade_id)
                return self._build_result(trade_id, "failed", legs, trade_usd,
                                          Decimal('0'), start_time)

            # Calculate quantity
            if side == 'buy':
                order_side = OrderSide.BUY
                quantity = current_amount / price
            else:
                order_side = OrderSide.SELL
                quantity = current_amount

            # Round using cached precision
            qty_prec = precisions.get(symbol, (8, 8))[1]
            price_prec = precisions.get(symbol, (8, 8))[0]
            quantity = self._round_decimal(quantity, qty_prec)
            rounded_price = self._round_decimal(price, price_prec)

            if quantity <= 0:
                logger.error(f"TRI LEG {leg_num} FAILED: quantity too small for {symbol}")
                await self._unwind(legs, trade_id)
                return self._build_result(trade_id, "failed", legs, trade_usd,
                                          Decimal('0'), start_time)

            order = OrderRequest(
                exchange="mexc",
                symbol=symbol,
                side=order_side,
                order_type=OrderType.LIMIT_MAKER,  # Post-only → 0% maker fee on MEXC
                quantity=quantity,
                price=rounded_price,
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

        exec_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            f"TRI COMPLETE {trade_id}: "
            f"Started {float(initial_amount):.8f} {start_currency} -> "
            f"Ended {float(current_amount):.8f} {start_currency} | "
            f"Profit: {float(profit):.8f} {start_currency} | "
            f"Time: {exec_ms:.0f}ms"
        )

        result = self._build_result(
            trade_id, "filled", legs, initial_amount, current_amount, start_time
        )
        self._completed.append(result)
        return result

    async def _pre_fetch_all(
        self, symbols: List[str], path: List[Tuple[str, str, str]]
    ) -> Tuple[Dict[Tuple[str, str], Decimal], Dict[str, Tuple[int, int]]]:
        """Pre-fetch all order books and precision info concurrently."""
        books: Dict[Tuple[str, str], Decimal] = {}
        precisions: Dict[str, Tuple[int, int]] = {}

        # Build list of concurrent tasks
        async def fetch_book(symbol: str, side: str):
            try:
                book = await self.client.get_order_book(symbol, depth=5)
                if side == 'ask':
                    return (symbol, side), book.best_ask
                return (symbol, side), book.best_bid
            except Exception as e:
                logger.debug(f"Pre-fetch failed for {symbol} {side}: {e}")
                return (symbol, side), None

        async def fetch_precision(symbol: str):
            if symbol in self._precision_cache:
                return symbol, self._precision_cache[symbol]
            try:
                info = await self.client.get_symbol_info(symbol)
                prec = (
                    int(info.get('price_precision', 8)),
                    int(info.get('quantity_precision', 8)),
                )
                self._precision_cache[symbol] = prec
                return symbol, prec
            except Exception:
                return symbol, (8, 8)

        tasks = []
        for symbol, side, _ in path:
            price_side = 'bid' if side == 'buy' else 'ask'  # LIMIT_MAKER: rest in book
            tasks.append(fetch_book(symbol, price_side))
            if symbol not in self._precision_cache:
                tasks.append(fetch_precision(symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                continue
            if isinstance(r, tuple) and len(r) == 2:
                key, value = r
                if isinstance(key, tuple):
                    # Book result: ((symbol, side), price)
                    books[key] = value
                elif isinstance(key, str):
                    # Precision result: (symbol, (price_prec, qty_prec))
                    precisions[key] = value

        # Fill precisions from cache for any we already had
        for symbol, _, _ in path:
            if symbol not in precisions and symbol in self._precision_cache:
                precisions[symbol] = self._precision_cache[symbol]

        return books, precisions

    @staticmethod
    def _round_decimal(value: Decimal, precision: int) -> Decimal:
        """Round a decimal to given precision."""
        if precision <= 0:
            return value.quantize(Decimal('1'), rounding=ROUND_DOWN)
        quant = Decimal(10) ** -precision
        return value.quantize(quant, rounding=ROUND_DOWN)

    async def _unwind(self, completed_legs: List[TriLegResult], trade_id: str) -> None:
        """Unwind completed legs in reverse order to recover starting currency."""
        filled_legs = [l for l in completed_legs if l.status == "filled"]
        if not filled_legs:
            return

        logger.warning(f"TRI UNWINDING {trade_id}: {len(filled_legs)} legs to reverse")

        for leg in reversed(filled_legs):
            try:
                base, quote = leg.symbol.split('/')
                if leg.side == 'buy':
                    reverse_side = OrderSide.SELL
                    reverse_qty = leg.quantity_out
                else:
                    reverse_side = OrderSide.BUY
                    reverse_qty = leg.quantity_out

                qty_prec = self._precision_cache.get(leg.symbol, (8, 8))[1]
                reverse_qty = self._round_decimal(reverse_qty, qty_prec)
                if reverse_qty <= 0:
                    logger.error(f"TRI UNWIND: quantity too small for {leg.symbol}")
                    continue

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
