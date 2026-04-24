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
import math
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
    fill_time_ms: float = 0.0


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
    leg_fill_times_ms: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    book_depth: Optional[Dict] = None  # Per-leg book depth at time of execution
    bottleneck_depth_usd: float = 0.0  # USD depth of thinnest leg
    sizing_reason: str = ""            # Human-readable sizing explanation


class TriangularExecutor:
    """Executes 3-leg triangular arbitrage on a single exchange."""

    def __init__(self, mexc_client, maker_fill_timeout_s: float = 5.0):
        self.client = mexc_client
        self._maker_fill_timeout_s = maker_fill_timeout_s
        self._trade_count = 0
        self._fill_count = 0
        self._total_profit = Decimal('0')
        self._completed: List[TriExecutionResult] = []
        self._precision_cache: Dict[str, Tuple[int, int]] = {}  # symbol -> (price_prec, qty_prec)
        self.capital_guard = None  # Set by orchestrator via triangular_arb

    async def execute(self, path, start_currency: str,
                      trade_usd: Decimal,
                      min_trade_usd: float = 50.0,
                      depth_fraction: float = 0.15,
                      min_depth_fraction: float = 0.0,
                      edge_bps: Optional[float] = None,
                      edge_thresholds: Optional[Dict[str, float]] = None,
                      ) -> TriExecutionResult:
        """
        Execute a triangular arbitrage cycle.

        Args:
            path: List of (symbol, side, intermediate_currency) tuples
            start_currency: Currency we start and end with (e.g., "USDT")
            trade_usd: USD ceiling for this trade
            min_trade_usd: Floor — skip if optimal is below this
            depth_fraction: Fraction of thinnest leg's depth to use (default 15%)
            edge_bps: Detected edge in bps (for edge-quality scaling)
            edge_thresholds: Custom edge scaling thresholds
        """
        trade_id = f"tri_{start_currency}_{int(time.time() * 1000)}"
        start_time = time.monotonic()

        # Capital guard: check USDT reserve before committing capital
        if self.capital_guard:
            allowed, cur_bal = await self.capital_guard.can_spend(
                self.client, float(trade_usd)
            )
            if not allowed:
                return self._build_result(
                    trade_id, "capital_guard_blocked", [], trade_usd,
                    Decimal('0'), start_time,
                )

        self._trade_count += 1
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
        books, precisions, raw_books = await self._pre_fetch_all(symbols, path)
        pre_fetch_ms = (time.monotonic() - pre_fetch_start) * 1000

        # === BOOK DEPTH LOGGING ===
        # Record depth at top 5 levels for each leg (for scaling analysis)
        book_depth = self._compute_book_depth(path, raw_books)

        # === DYNAMIC SIZING (active) ===
        leg_depths = [l['depth_usd_top5'] for l in book_depth.get('legs', [])]
        optimal_size, sizing_reason = self._optimal_trade_size(
            leg_depths, float(trade_usd),
            min_trade_usd=min_trade_usd,
            depth_fraction=depth_fraction,
            min_depth_fraction=min_depth_fraction,
            edge_bps=edge_bps,
            edge_thresholds=edge_thresholds,
        )
        min_depth = min(leg_depths) if leg_depths and all(d > 0 for d in leg_depths) else 0.0

        # Determine tier for logging — use min_trade_usd as floor, not hardcoded $20
        if optimal_size >= 5000:
            tier = "LARGE"
        elif optimal_size >= 1500:
            tier = "FULL"
        elif optimal_size >= 500:
            tier = "MEDIUM"
        elif optimal_size > 0:
            tier = "SMALL"
        else:
            tier = "SKIP"

        if tier == "SKIP":
            logger.info(
                f"TRI SKIP {trade_id}: {sizing_reason} | {' -> '.join(s[0] for s in path)}"
            )
            self._trade_count -= 1
            return self._build_result(trade_id, "skipped", [], trade_usd,
                                      Decimal('0'), start_time, book_depth,
                                      bottleneck_depth_usd=min_depth,
                                      sizing_reason=sizing_reason)

        # Apply dynamic size — override trade_usd with optimal
        actual_trade_usd = Decimal(str(optimal_size))
        logger.info(
            f"TRI SIZING: {sizing_reason} | tier={tier} | "
            f"path={' -> '.join(s[0] for s in path)}"
        )

        # Re-compute current_amount with the dynamically sized trade
        if start_currency in ("USDT", "USDC", "BUSD"):
            current_amount = actual_trade_usd
        else:
            current_amount = actual_trade_usd / (await self._get_price(f"{start_currency}/USDT", 'bid') or Decimal('1'))
        initial_amount = current_amount
        trade_usd = actual_trade_usd

        # === PRE-EXECUTION ROUNDING CHECK ===
        # Simulate rounding at each leg's precision to estimate worst-case
        # rounding loss. Skip cycle if rounding eats >50% of expected profit.
        sim_amount = current_amount
        for symbol, side, _ in path:
            qty_prec = precisions.get(symbol, (8, 8))[1]
            price_prec = precisions.get(symbol, (8, 8))[0]
            price_side = 'ask' if side == 'buy' else 'bid'  # IOC: cross spread
            price = books.get((symbol, price_side))
            if not price or price <= 0:
                break  # Can't simulate — let execution handle the error
            if side == 'buy':
                raw_qty = sim_amount / price
                rounded_qty = self._round_decimal(raw_qty, qty_prec)
                sim_amount = rounded_qty  # received base
            else:
                rounded_qty = self._round_decimal(sim_amount, qty_prec)
                rounded_price = self._round_decimal(price, price_prec)
                sim_amount = rounded_qty * rounded_price  # received quote
        else:
            rounding_loss = initial_amount - sim_amount
            if rounding_loss > Decimal('0') and rounding_loss > initial_amount * Decimal('0.0005'):
                logger.warning(
                    f"TRI SKIP {trade_id}: rounding loss ${float(rounding_loss):.4f} "
                    f"({float(rounding_loss / initial_amount * 10000):.1f} bps) "
                    f"on {' -> '.join(s[0] for s in path)}"
                )
                self._trade_count -= 1  # Don't count skips in stats
                return self._build_result(trade_id, "skipped", [], initial_amount,
                                          Decimal('0'), start_time, book_depth)

        # Build compact depth string for log: "LEG1:$4200 LEG2:$12000 LEG3:$85000"
        depth_str = " ".join(
            f"L{i+1}:${book_depth['legs'][i]['depth_usd_top5']:.0f}"
            for i in range(len(book_depth.get('legs', [])))
        ) if book_depth.get('legs') else "no-depth"

        logger.info(
            f"TRI EXECUTE {trade_id}: {' -> '.join(s[0] for s in path)} -> {start_currency} "
            f"| Starting: {float(current_amount):.8f} {start_currency} (${float(trade_usd):.2f}) "
            f"| Pre-fetch: {pre_fetch_ms:.0f}ms | Depth: {depth_str}"
        )

        for i, (symbol, side, next_currency) in enumerate(path):
            leg_num = i + 1
            base, quote = symbol.split('/')

            # Hybrid: Leg 1 = IOC taker (cross spread), Legs 2-3 = LIMIT_MAKER (passive)
            is_taker_leg = (leg_num == 1)
            if is_taker_leg:
                price_side = 'ask' if side == 'buy' else 'bid'  # cross spread
            else:
                price_side = 'bid' if side == 'buy' else 'ask'  # rest on book
            price = books.get((symbol, price_side))
            if price is None or price <= 0:
                # Fallback: fetch fresh price for this leg
                price = await self._get_price(symbol, price_side)

            if price is None or price <= 0:
                logger.error(f"TRI LEG {leg_num} FAILED: no price for {symbol}")
                await self._unwind(legs, trade_id)
                return self._build_result(trade_id, "failed", legs, trade_usd,
                                          Decimal('0'), start_time, book_depth,
                                          bottleneck_depth_usd=min_depth,
                                          sizing_reason=sizing_reason)

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
                                          Decimal('0'), start_time, book_depth,
                                          bottleneck_depth_usd=min_depth,
                                          sizing_reason=sizing_reason)

            leg_type_str = 'IOC' if is_taker_leg else 'MAKER'
            fee_label = 'taker 5bps' if is_taker_leg else 'maker 0bps'
            logger.info(
                f"TRI LEG {leg_num} {leg_type_str}: {side.upper()} {symbol} "
                f"qty={float(quantity):.10f} price={float(rounded_price):.10f} "
                f"({price_side}) prec=({price_prec},{qty_prec})"
            )

            if is_taker_leg:
                order = OrderRequest(
                    exchange="mexc",
                    symbol=symbol,
                    side=order_side,
                    order_type=OrderType.LIMIT,
                    quantity=quantity,
                    price=rounded_price,
                    time_in_force=TimeInForce.IOC,
                    client_order_id=f"{trade_id}_leg{leg_num}",
                )
            else:
                order = OrderRequest(
                    exchange="mexc",
                    symbol=symbol,
                    side=order_side,
                    order_type=OrderType.LIMIT_MAKER,
                    quantity=quantity,
                    price=rounded_price,
                    time_in_force=TimeInForce.GTC,
                    client_order_id=f"{trade_id}_leg{leg_num}",
                )

            leg_start = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    self.client.place_order(order),
                    timeout=15.0,
                )
                leg_ms = (time.monotonic() - leg_start) * 1000
            except asyncio.TimeoutError:
                logger.error(f"TRI LEG {leg_num} TIMEOUT: {symbol} {side}")
                # Cancel any outstanding order to prevent stranded assets
                try:
                    await self._cancel_leg_order(symbol, f"{trade_id}_leg{leg_num}", trade_id, leg_num)
                except Exception as e:
                    logger.warning(f"self._cancel_leg_order failed: {e}")
                legs.append(TriLegResult(
                    leg_number=leg_num, symbol=symbol, side=side,
                    status="failed", quantity_in=current_amount,
                ))
                await self._unwind(legs, trade_id)
                return self._build_result(trade_id, "failed", legs, trade_usd,
                                          Decimal('0'), start_time, book_depth,
                                          bottleneck_depth_usd=min_depth,
                                          sizing_reason=sizing_reason)


            # Handle OPEN orders differently based on leg type
            if result.status == OrderStatus.OPEN and result.order_id:
                if is_taker_leg:
                    # IOC returned OPEN — cancel immediately
                    logger.warning(f"TRI LEG {leg_num} IOC returned OPEN — cancelling")
                    try:
                        await self.client.cancel_order(symbol, result.order_id)
                    except Exception as e:
                        logger.warning(f"self.client.cancel_order failed: {e}")
                    result.status = OrderStatus.CANCELLED
                else:
                    # MAKER leg: wait for fill with timeout
                    result = await self._wait_and_cancel(
                        symbol, result.order_id, trade_id, leg_num,
                        max_wait=self._maker_fill_timeout_s,
                    )

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
                                          Decimal('0'), start_time, book_depth,
                                          bottleneck_depth_usd=min_depth,
                                          sizing_reason=sizing_reason)

            # Calculate output amount
            fill_price = result.average_fill_price or price
            fill_qty = result.filled_quantity

            if side == 'buy':
                quantity_out = fill_qty  # We received base currency
            else:
                quantity_out = fill_qty * fill_price  # We received quote currency

            quantity_out -= result.fee_amount  # Subtract fees

            logger.info(
                f"TRI LEG {leg_num} {leg_type_str} FILLED: "
                f"{side.upper()} {symbol} "
                f"qty={float(fill_qty):.8f} @ {float(fill_price):.6f} "
                f"in {leg_ms:.0f}ms ({fee_label}) | "
                f"in={float(current_amount):.6f} {current_currency} -> "
                f"out={float(quantity_out):.6f} {next_currency} | "
                f"fee={float(result.fee_amount):.6f}"
            )

            legs.append(TriLegResult(
                leg_number=leg_num, symbol=symbol, side=side,
                status="filled", order_result=result,
                quantity_in=current_amount, quantity_out=quantity_out,
                fill_time_ms=leg_ms,
            ))

            current_amount = quantity_out
            current_currency = next_currency

        # All 3 legs completed!
        profit = current_amount - initial_amount
        self._fill_count += 1
        self._total_profit += profit

        exec_ms = (time.monotonic() - start_time) * 1000
        total_fees = sum(l.order_result.fee_amount for l in legs if l.order_result and l.status == "filled")
        fee_bps = float(total_fees / initial_amount * 10000) if initial_amount > 0 else 0
        gross_profit = profit + total_fees
        gross_bps = float(gross_profit / initial_amount * 10000) if initial_amount > 0 else 0
        net_bps = float(profit / initial_amount * 10000) if initial_amount > 0 else 0
        path_str = ' -> '.join(s[0] for s in path)
        logger.info(
            f"TRI COMPLETE {trade_id}: {path_str} | "
            f"gross={gross_bps:.1f}bps fees={fee_bps:.1f}bps net={net_bps:.1f}bps "
            f"profit=${float(profit):.4f} | Time: {exec_ms:.0f}ms"
        )

        leg_times = [l.fill_time_ms for l in legs if l.status == "filled"]
        result = self._build_result(
            trade_id, "filled", legs, initial_amount, current_amount, start_time, book_depth,
            bottleneck_depth_usd=min_depth, sizing_reason=sizing_reason,
            leg_fill_times_ms=leg_times,
        )
        self._completed.append(result)
        return result

    async def _pre_fetch_all(
        self, symbols: List[str], path: List[Tuple[str, str, str]]
    ) -> Tuple[Dict[Tuple[str, str], Decimal], Dict[str, Tuple[int, int]], Dict[str, 'OrderBook']]:
        """Pre-fetch all order books and precision info concurrently.

        Returns:
            books: {(symbol, side): best_price}
            precisions: {symbol: (price_prec, qty_prec)}
            raw_books: {symbol: OrderBook} for depth analysis
        """
        books: Dict[Tuple[str, str], Decimal] = {}
        precisions: Dict[str, Tuple[int, int]] = {}
        raw_books: Dict[str, object] = {}

        # Build list of concurrent tasks
        async def fetch_book(symbol: str, side: str):
            try:
                book = await self.client.get_order_book(symbol, depth=5)
                if side == 'ask':
                    return (symbol, side), book.best_ask, book
                return (symbol, side), book.best_bid, book
            except Exception as e:
                logger.debug(f"Pre-fetch failed for {symbol} {side}: {e}")
                return (symbol, side), None, None

        async def fetch_precision(symbol: str):
            if symbol in self._precision_cache:
                return symbol, self._precision_cache[symbol]
            try:
                info = await self.client.get_symbol_info(symbol)
                raw_price = info.get('price_precision', 8)
                raw_qty = info.get('quantity_precision', 8)
                # Safety: convert TICK_SIZE floats (e.g. 0.0001) to decimal
                # place counts.  MEXCClient.get_symbol_info already does this,
                # but guard against any client that still returns raw tick sizes.
                price_prec = self._safe_precision(raw_price)
                qty_prec = self._safe_precision(raw_qty)
                prec = (price_prec, qty_prec)
                self._precision_cache[symbol] = prec
                logger.debug(f"Precision {symbol}: raw=({raw_price},{raw_qty}) → dp=({price_prec},{qty_prec})")
                return symbol, prec
            except Exception:
                return symbol, (8, 8)

        tasks = []
        for i, (symbol, side, _) in enumerate(path):
            # All-IOC: cross spread on every leg
            price_side = 'ask' if side == 'buy' else 'bid'
            tasks.append(fetch_book(symbol, price_side))
            if symbol not in self._precision_cache:
                tasks.append(fetch_precision(symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                continue
            if isinstance(r, tuple) and len(r) == 3:
                # Book result: ((symbol, side), price, book_obj)
                key, value, book_obj = r
                books[key] = value
                if book_obj is not None:
                    raw_books[key[0]] = book_obj
            elif isinstance(r, tuple) and len(r) == 2:
                # Precision result: (symbol, (price_prec, qty_prec))
                key, value = r
                if isinstance(key, str):
                    precisions[key] = value

        # Fill precisions from cache for any we already had
        for symbol, _, _ in path:
            if symbol not in precisions and symbol in self._precision_cache:
                precisions[symbol] = self._precision_cache[symbol]

        return books, precisions, raw_books

    @staticmethod
    def _safe_precision(raw) -> int:
        """Convert a precision value to integer decimal places.

        Handles both TICK_SIZE (float like 0.0001 → 4) and DECIMAL_PLACES
        (int like 4 → 4).  Prevents the catastrophic int(0.0001)=0 bug.
        """
        if isinstance(raw, int) and raw >= 1:
            return raw
        t = float(raw)
        if t >= 1:
            return int(t)
        if t <= 0:
            return 8  # safe fallback
        return max(0, -int(math.floor(math.log10(t))))

    @staticmethod
    def _round_decimal(value: Decimal, precision) -> Decimal:
        """Round a decimal to given precision.

        precision can be:
        - int >= 1: decimal place count (normal case after _safe_precision)
        - float < 1: tick size (safety fallback if raw tick leaked through)
        """
        p = float(precision)
        if 0 < p < 1:
            # TICK_SIZE leaked through — handle it directly
            tick = Decimal(str(precision))
            return (value / tick).to_integral_value(rounding=ROUND_DOWN) * tick
        places = int(p)
        if places <= 0:
            return value.quantize(Decimal('1'), rounding=ROUND_DOWN)
        quant = Decimal(10) ** -places
        return value.quantize(quant, rounding=ROUND_DOWN)


    async def _wait_and_cancel(self, symbol: str, order_id: str, trade_id: str, leg_num: int, max_wait: float = 5.0) -> "OrderResult":
        """Poll an OPEN order for fills, then cancel if still open. Returns final status."""
        poll_interval = 1.0
        elapsed = 0.0
        last_result = None
        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            try:
                last_result = await asyncio.wait_for(
                    self.client.get_order_status(symbol, order_id),
                    timeout=5.0,
                )
                if last_result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                    logger.info(f"TRI LEG {leg_num} filled after {elapsed:.0f}s wait: {last_result.status.value}")
                    return last_result
                if last_result.status in (OrderStatus.CANCELLED,):
                    return last_result
            except Exception as e:
                logger.debug(f"Order status poll failed: {e}")

        # Still open after max_wait — cancel with retry
        logger.warning(f"TRI LEG {leg_num} still OPEN after {max_wait:.0f}s — cancelling order {order_id}")
        for cancel_attempt in range(3):
            cancel_says_filled = False
            try:
                await self.client.cancel_order(symbol, order_id)
            except Exception as e:
                err_str = str(e).lower()
                if 'filled' in err_str or 'order completed' in err_str:
                    logger.info(f"TRI LEG {leg_num} cancel says order already filled — treating as FILLED")
                    cancel_says_filled = True
                else:
                    logger.error(f"TRI LEG {leg_num} cancel failed (attempt {cancel_attempt + 1}): {e}")

            # Check status after cancel attempt
            try:
                last_result = await asyncio.wait_for(
                    self.client.get_order_status(symbol, order_id),
                    timeout=5.0,
                )
                if last_result.filled_quantity and last_result.filled_quantity > 0:
                    fill_status = OrderStatus.FILLED if cancel_says_filled else OrderStatus.PARTIALLY_FILLED
                    logger.info(
                        f"TRI LEG {leg_num} fill detected after cancel: "
                        f"{float(last_result.filled_quantity):.8f} {symbol} — {fill_status.value}"
                    )
                    last_result.status = fill_status
                    return last_result
                if cancel_says_filled:
                    # Cancel said filled but status check shows no qty — force FILLED status
                    logger.warning(
                        f"TRI LEG {leg_num} cancel said filled but status shows no qty — marking FILLED anyway"
                    )
                    last_result.status = OrderStatus.FILLED
                    return last_result
                if last_result.status != OrderStatus.OPEN:
                    return last_result  # Successfully cancelled
                # Still OPEN — retry cancel
                logger.warning(
                    f"TRI LEG {leg_num} still OPEN after cancel attempt {cancel_attempt + 1} — retrying"
                )
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"TRI LEG {leg_num} post-cancel status check failed: {e}")
                if cancel_says_filled:
                    # Cancel said filled, status check failed — build synthetic FILLED result
                    logger.warning(f"TRI LEG {leg_num} cancel said filled, status unavailable — treating as FILLED")
                    from ..exchanges.base import OrderResult as _OR, OrderSide as _OS, OrderType as _OT
                    return _OR(
                        exchange="mexc", symbol=symbol, order_id=order_id,
                        client_order_id=None, status=OrderStatus.FILLED,
                        side=_OS.BUY, order_type=_OT.LIMIT,
                        requested_quantity=Decimal(0), filled_quantity=Decimal(0),
                        average_fill_price=None, fee_amount=Decimal(0),
                        fee_currency="USDT", timestamp=datetime.utcnow(),
                        raw_response={"cancel_said_filled": True},
                    )
                break

        # Exhausted retries
        if last_result:
            logger.error(
                f"TRI LEG {leg_num} cancel exhausted 3 attempts — final status={last_result.status.value}"
            )
            return last_result
        # Return a synthetic cancelled result
        from ..exchanges.base import OrderResult as _OR, OrderSide as _OS, OrderType as _OT
        return _OR(
            exchange="mexc", symbol=symbol, order_id=order_id,
            client_order_id=None, status=OrderStatus.CANCELLED,
            side=_OS.BUY, order_type=_OT.LIMIT,
            requested_quantity=Decimal(0), filled_quantity=Decimal(0),
            average_fill_price=None, fee_amount=Decimal(0),
            fee_currency="USDT", timestamp=datetime.utcnow(),
            raw_response={},
        )

    async def _cancel_leg_order(self, symbol: str, order_id: str, trade_id: str, leg_num: int) -> None:
        """Cancel an outstanding order for a failed leg to prevent stranded assets."""
        try:
            cancelled = await self.client.cancel_order(symbol, order_id)
            if cancelled:
                logger.info(f"TRI LEG {leg_num} order cancelled: {order_id} on {symbol}")
            else:
                logger.warning(f"TRI LEG {leg_num} cancel returned False: {order_id} on {symbol}")
        except Exception as e:
            logger.error(f"TRI LEG {leg_num} cancel error: {order_id} — {e}")

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
                    # Reverse a BUY = SELL the base currency we received
                    reverse_side = OrderSide.SELL
                    reverse_qty = leg.quantity_out  # base currency received
                else:
                    # Reverse a SELL = BUY back the base currency we sold
                    # FIX: was using quantity_out (quote currency), must use
                    # quantity_in (base currency) to avoid catastrophic mispricing
                    reverse_side = OrderSide.BUY
                    reverse_qty = leg.quantity_in  # base currency that was sold

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
                if result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                    leg.status = "unwound"
                    logger.info(
                        f"TRI UNWIND leg {leg.leg_number}: {leg.symbol} "
                        f"{'SELL' if leg.side == 'buy' else 'BUY'} "
                        f"{float(reverse_qty):.8f} — {result.status.value}"
                    )
                else:
                    logger.error(
                        f"TRI UNWIND leg {leg.leg_number} NOT FILLED: {leg.symbol} "
                        f"status={result.status.value} qty={float(reverse_qty):.8f} "
                        f"— TOKENS MAY BE STRANDED, inventory scanner will clean up"
                    )
            except Exception as e:
                logger.error(f"TRI UNWIND FAILED leg {leg.leg_number}: {e} — inventory scanner will clean up")

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
                      end_amount: Decimal, start_time: float,
                      book_depth: Optional[Dict] = None,
                      bottleneck_depth_usd: float = 0.0,
                      sizing_reason: str = "",
                      leg_fill_times_ms: Optional[List[float]] = None) -> TriExecutionResult:
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
            leg_fill_times_ms=leg_fill_times_ms or [],
            book_depth=book_depth,
            bottleneck_depth_usd=bottleneck_depth_usd,
            sizing_reason=sizing_reason,
        )

    @staticmethod
    def _optimal_trade_size(depths: List[float], max_trade_usd: float,
                            min_trade_usd: float = 50.0,
                            depth_fraction: float = 0.15,
                            min_depth_fraction: float = 0.0,
                            edge_bps: Optional[float] = None,
                            edge_thresholds: Optional[Dict[str, float]] = None,
                            ) -> Tuple[float, str]:
        """Optimal trade size = min(max_trade_usd, depth_fraction * min_leg_depth) × edge_factor.

        We never want to consume more than depth_fraction of the thinnest
        leg's visible depth at the top 5 price levels.

        Edge-quality scaling (edge_bps → multiplier):
          <thin_bps  → 0.5x   (small edge = conservative)
          thin-moderate → 0.75x
          >=full_bps → 1.0x   (big edge = full size)

        Returns (trade_size, sizing_reason). size=0.0 means skip.
        """
        thresholds = edge_thresholds or {
            'thin_bps': 5.0, 'moderate_bps': 10.0, 'full_bps': 20.0,
        }

        if not depths:
            return 0.0, "no_depth_data|skip"

        min_depth = min(depths)
        if min_depth <= 0:
            return 0.0, "zero_depth|skip"

        # Depth-based size
        depth_size = depth_fraction * min_depth
        base_size = min(max_trade_usd, depth_size)

        # Edge-quality scaling
        edge_factor = 1.0
        edge_label = "full"
        if edge_bps is not None:
            if edge_bps < thresholds['thin_bps']:
                edge_factor = 0.5
                edge_label = "thin"
            elif edge_bps < thresholds['moderate_bps']:
                edge_factor = 0.75
                edge_label = "moderate"
            # else edge_factor stays 1.0

        optimal = round(base_size * edge_factor, 2)

        # Max depth fraction guard: skip if trade would consume too much of the
        # book (market impact risk). The depth_fraction cap above already limits
        # this, but min_depth_fraction acts as a second safety net.
        # NOTE: previously this was inverted (blocked SMALL fractions), which
        # killed all trades since $20 / $500+ depth = <5%. Fixed 2026-03-23.
        actual_frac = optimal / min_depth if min_depth > 0 else 0.0

        if optimal < min_trade_usd:
            reason = (
                f"too_thin|depth=${min_depth:.0f}|{depth_fraction:.0%}=${depth_size:.0f}"
                f"|edge={edge_label}({edge_factor}x)|final=${optimal:.0f}<floor${min_trade_usd:.0f}"
            )
            return 0.0, reason

        reason = (
            f"depth=${min_depth:.0f}|{depth_fraction:.0%}=${depth_size:.0f}"
            f"|cap=${max_trade_usd:.0f}|edge={edge_label}({edge_factor}x)|final=${optimal:.0f}"
        )
        return optimal, reason

    @staticmethod
    def _compute_book_depth(
        path: List[Tuple[str, str, str]],
        raw_books: Dict[str, object],
    ) -> Dict:
        """Compute USD depth at top 5 levels for each leg's relevant side."""
        legs_depth = []
        for symbol, side, _ in path:
            book = raw_books.get(symbol)
            if not book:
                legs_depth.append({
                    "symbol": symbol, "side": side,
                    "depth_usd_top5": 0.0, "levels": 0,
                })
                continue

            # For IOC: BUY consumes asks, SELL consumes bids
            # Depth we care about = the side we're taking from
            levels = book.asks[:5] if side == 'buy' else book.bids[:5]
            depth_usd = sum(
                float(lvl.price * lvl.quantity) for lvl in levels
            )
            legs_depth.append({
                "symbol": symbol, "side": side,
                "depth_usd_top5": round(depth_usd, 2),
                "levels": len(levels),
            })

        return {"legs": legs_depth}

    def get_stats(self) -> dict:
        return {
            "total_trades": self._trade_count,
            "total_fills": self._fill_count,
            "total_profit_usd": float(self._total_profit),
            "win_rate": self._fill_count / max(1, self._trade_count),
        }
