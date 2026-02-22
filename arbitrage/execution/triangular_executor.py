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
    book_depth: Optional[Dict] = None  # Per-leg book depth at time of execution


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
        books, precisions, raw_books = await self._pre_fetch_all(symbols, path)
        pre_fetch_ms = (time.monotonic() - pre_fetch_start) * 1000

        # === BOOK DEPTH LOGGING ===
        # Record depth at top 5 levels for each leg (for scaling analysis)
        book_depth = self._compute_book_depth(path, raw_books)

        # === DYNAMIC SIZING (active) ===
        leg_depths = [l['depth_usd_top5'] for l in book_depth.get('legs', [])]
        optimal_size = self._optimal_trade_size(leg_depths, float(trade_usd))
        min_depth = min(leg_depths) if leg_depths and all(d > 0 for d in leg_depths) else 0.0

        # Determine tier for logging
        if optimal_size >= 1500:
            tier = "FULL"
        elif optimal_size >= 500:
            tier = "MEDIUM"
        elif optimal_size >= 50:
            tier = "SMALL"
        else:
            tier = "SKIP"

        if tier == "SKIP":
            logger.info(
                f"TRI SKIP {trade_id}: min_depth=${min_depth:.0f} too thin, "
                f"optimal=$0 (floor=$50) | {' -> '.join(s[0] for s in path)}"
            )
            self._trade_count -= 1
            return self._build_result(trade_id, "skipped", [], trade_usd,
                                      Decimal('0'), start_time, book_depth)

        # Apply dynamic size — override trade_usd with optimal
        actual_trade_usd = Decimal(str(optimal_size))
        logger.info(
            f"TRI SIZING: config=${float(trade_usd):.0f} | optimal=${optimal_size:.0f} | "
            f"min_depth=${min_depth:.0f} | tier={tier} | using=${optimal_size:.0f}"
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
            price_side = 'bid' if side == 'buy' else 'ask'
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
                                          Decimal('0'), start_time, book_depth)

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
                                          Decimal('0'), start_time, book_depth)

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
                                          Decimal('0'), start_time, book_depth)

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
                                          Decimal('0'), start_time, book_depth)

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
            trade_id, "filled", legs, initial_amount, current_amount, start_time, book_depth
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
                      end_amount: Decimal, start_time: float,
                      book_depth: Optional[Dict] = None) -> TriExecutionResult:
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
            book_depth=book_depth,
        )

    @staticmethod
    def _optimal_trade_size(depths: List[float], max_trade_usd: float,
                            min_trade_usd: float = 50.0,
                            depth_fraction: float = 0.25) -> float:
        """Optimal trade size = min(max_trade_usd, depth_fraction * min_leg_depth).

        We never want to consume more than depth_fraction of the thinnest
        leg's visible depth at the top 5 price levels.

        Returns 0.0 if optimal is below min_trade_usd (signal to skip).
        """
        if not depths:
            return max_trade_usd  # no depth data at all — fallback to config
        min_depth = min(depths)
        if min_depth <= 0:
            return 0.0  # at least one leg has zero depth — skip (unsafe to trade)
        optimal = min(max_trade_usd, depth_fraction * min_depth)
        if optimal < min_trade_usd:
            return 0.0  # Signal to skip — depth too thin
        return round(optimal, 2)

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

            # For LIMIT_MAKER: BUY rests at bid, SELL rests at ask
            # Depth we care about = the side we're joining
            levels = book.bids[:5] if side == 'buy' else book.asks[:5]
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
