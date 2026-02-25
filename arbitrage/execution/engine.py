"""
Arbitrage Execution Engine — executes ArbitrageSignal trades
simultaneously on both exchanges.

CRITICAL PRINCIPLES:
1. BOTH LEGS MUST EXECUTE. Partial fill = unhedged directional exposure.
2. MAKER FIRST. LIMIT_MAKER orders for zero fee on MEXC.
3. SPEED. Both orders placed concurrently via asyncio.
4. VERIFY. If one side fills and other doesn't → emergency close.
5. PRE-CHECK. Verify book depth and freshness before firing.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Optional, List, TYPE_CHECKING

from ..exchanges.base import (
    OrderRequest, OrderResult, OrderSide, OrderType, OrderStatus, TimeInForce,
)
from ..detector.cross_exchange import ArbitrageSignal
from .realistic_fill import RealisticCrossExchangeFill, RealisticCostBreakdown

if TYPE_CHECKING:
    from ..orderbook.unified_book import UnifiedBookManager

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
    realistic_costs: Optional[RealisticCostBreakdown] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ArbitrageExecutor:

    # Paper fills are instant — no need to wait long.
    # Live fills need more time for exchange round-trip.
    FILL_TIMEOUT_PAPER = 2.0
    FILL_TIMEOUT_LIVE = 5.0
    EMERGENCY_CLOSE_SECONDS = 5.0

    # Pre-execution gates
    MAX_BOOK_AGE_SEC = 30.0  # Reject if either book is older than this (REST polls every ~15s)
    MIN_DEPTH_RATIO = 0.8    # Need at least 80% of target qty available
    PRICE_TOLERANCE = Decimal('0.002')  # 0.2% tolerance for depth check

    def __init__(self, mexc_client, binance_client, cost_model, risk_engine,
                 config: Optional[dict] = None,
                 book_manager: Optional['UnifiedBookManager'] = None):
        self.clients = {
            "mexc": mexc_client,
            "binance": binance_client,
        }
        self.costs = cost_model
        self.risk = risk_engine
        self.book_manager = book_manager
        self._paper_mode = getattr(mexc_client, 'paper_trading', True)
        self._realistic_fill = RealisticCrossExchangeFill(config=config)
        self._active_trades: Dict[str, ExecutionResult] = {}
        self._completed_trades: List[ExecutionResult] = []
        self._total_profit = Decimal('0')
        self._realistic_profit = Decimal('0')
        self._trade_count = 0
        self._fill_count = 0
        self._depth_rejects = 0
        self._freshness_rejects = 0

    # ── Pre-execution gates ────────────────────────────────────────

    def _check_book_freshness(self, signal: ArbitrageSignal) -> tuple:
        """Reject if either book is stale. Returns (ok, reason)."""
        if not self.book_manager:
            return True, "no_book_manager"

        view = self.book_manager.pairs.get(signal.symbol)
        if not view:
            return False, "pair_not_monitored"

        now = datetime.utcnow()
        mexc_age = (now - view.mexc_last_update).total_seconds()
        binance_age = (now - view.binance_last_update).total_seconds()

        if mexc_age > self.MAX_BOOK_AGE_SEC:
            return False, f"mexc_stale:{mexc_age:.1f}s"
        if binance_age > self.MAX_BOOK_AGE_SEC:
            return False, f"binance_stale:{binance_age:.1f}s"
        return True, f"fresh:M={mexc_age:.1f}s,B={binance_age:.1f}s"

    def _check_book_depth(self, signal: ArbitrageSignal) -> tuple:
        """Reject if either book lacks depth at target price. Returns (ok, reason)."""
        if not self.book_manager:
            return True, "no_book_manager"

        view = self.book_manager.pairs.get(signal.symbol)
        if not view or not view.mexc_book or not view.binance_book:
            return False, "missing_book"

        buy_book = view.mexc_book if signal.buy_exchange == "mexc" else view.binance_book
        sell_book = view.mexc_book if signal.sell_exchange == "mexc" else view.binance_book

        target_qty = signal.recommended_quantity

        # Check buy side: asks at/below our buy price (+tolerance)
        max_buy = signal.buy_price * (1 + self.PRICE_TOLERANCE)
        available_buy = Decimal('0')
        for level in buy_book.asks:
            if level.price <= max_buy:
                available_buy += level.quantity
            else:
                break

        if available_buy < target_qty * Decimal(str(self.MIN_DEPTH_RATIO)):
            return False, (
                f"buy_depth:{float(available_buy):.4f}<"
                f"{float(target_qty * Decimal(str(self.MIN_DEPTH_RATIO))):.4f}"
            )

        # Check sell side: bids at/above our sell price (-tolerance)
        min_sell = signal.sell_price * (1 - self.PRICE_TOLERANCE)
        available_sell = Decimal('0')
        for level in sell_book.bids:
            if level.price >= min_sell:
                available_sell += level.quantity
            else:
                break

        if available_sell < target_qty * Decimal(str(self.MIN_DEPTH_RATIO)):
            return False, (
                f"sell_depth:{float(available_sell):.4f}<"
                f"{float(target_qty * Decimal(str(self.MIN_DEPTH_RATIO))):.4f}"
            )

        return True, "depth_ok"

    # ── Main execution ─────────────────────────────────────────────

    async def execute_arbitrage(self, signal: ArbitrageSignal) -> ExecutionResult:
        """Execute a cross-exchange arbitrage trade."""
        trade_id = signal.signal_id

        # Check expiry
        if datetime.utcnow() > signal.expires_at:
            logger.debug(f"Signal expired: {trade_id}")
            return ExecutionResult(trade_id=trade_id, status="expired", signal=signal)

        # LAYER 1: Book freshness gate
        fresh_ok, fresh_reason = self._check_book_freshness(signal)
        if not fresh_ok:
            self._freshness_rejects += 1
            if self._freshness_rejects <= 5 or self._freshness_rejects % 100 == 0:
                logger.info(f"GATE: freshness reject #{self._freshness_rejects} {signal.symbol}: {fresh_reason}")
            return ExecutionResult(trade_id=trade_id, status="book_stale", signal=signal)

        # LAYER 2: Book depth gate
        depth_ok, depth_reason = self._check_book_depth(signal)
        if not depth_ok:
            self._depth_rejects += 1
            if self._depth_rejects <= 5 or self._depth_rejects % 100 == 0:
                logger.info(f"GATE: depth reject #{self._depth_rejects} {signal.symbol}: {depth_reason}")
            return ExecutionResult(trade_id=trade_id, status="depth_insufficient", signal=signal)

        buy_client = self.clients[signal.buy_exchange]
        sell_client = self.clients[signal.sell_exchange]

        # Verify inventory
        base, quote = signal.symbol.split('/')
        buy_balance = await buy_client.get_balance(quote)
        sell_balance = await sell_client.get_balance(base)

        required_quote = signal.recommended_quantity * signal.buy_price
        required_base = signal.recommended_quantity

        if buy_balance.free < required_quote:
            logger.info(f"Insufficient {quote} on {signal.buy_exchange}: "
                       f"need {float(required_quote):.2f}, have {float(buy_balance.free):.2f}")
            return ExecutionResult(trade_id=trade_id, status="insufficient_balance", signal=signal)

        if sell_balance.free < required_base:
            logger.info(f"Insufficient {base} on {signal.sell_exchange}: "
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

        # Sanity check: reject if rounding moved price >1% from signal
        for label, rounded, original in [
            ("buy", buy_price, signal.buy_price),
            ("sell", sell_price, signal.sell_price),
        ]:
            if original > 0:
                drift_pct = abs(float(rounded - original) / float(original)) * 100
                if drift_pct > 1.0:
                    logger.error(
                        f"PRICE ROUNDING ABORT {trade_id}: {label} price drifted "
                        f"{drift_pct:.1f}% ({float(original):.6f} → {float(rounded):.6f}). "
                        f"Precision mode bug? raw_precision={buy_info.get('price_precision') if label == 'buy' else sell_info.get('price_precision')}"
                    )
                    return ExecutionResult(trade_id=trade_id, status="price_rounding_abort", signal=signal)

        logger.info(
            f"EXECUTING: {signal.symbol} | "
            f"BUY {float(buy_qty):.6f} @ {float(buy_price):.2f} on {signal.buy_exchange} | "
            f"SELL {float(sell_qty):.6f} @ {float(sell_price):.2f} on {signal.sell_exchange} | "
            f"Expected profit: ${float(signal.expected_profit_usd):.2f}"
        )

        # LAYER 4: Order type — MEXC LIMIT_MAKER (0% fee), Binance IOC (fill-or-kill)
        buy_order = OrderRequest(
            exchange=signal.buy_exchange,
            symbol=signal.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT_MAKER if signal.buy_exchange == "mexc" else OrderType.LIMIT,
            quantity=buy_qty,
            price=buy_price,
            time_in_force=TimeInForce.GTX if signal.buy_exchange == "mexc" else TimeInForce.IOC,
            client_order_id=f"{trade_id}_buy",
        )

        sell_order = OrderRequest(
            exchange=signal.sell_exchange,
            symbol=signal.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT_MAKER if signal.sell_exchange == "mexc" else OrderType.LIMIT,
            quantity=sell_qty,
            price=sell_price,
            time_in_force=TimeInForce.GTX if signal.sell_exchange == "mexc" else TimeInForce.IOC,
            client_order_id=f"{trade_id}_sell",
        )

        # FIRE BOTH SIMULTANEOUSLY — LAYER 3: tight timeout
        fill_timeout = self.FILL_TIMEOUT_PAPER if self._paper_mode else self.FILL_TIMEOUT_LIVE
        try:
            buy_result, sell_result = await asyncio.wait_for(
                asyncio.gather(
                    buy_client.place_order(buy_order),
                    sell_client.place_order(sell_order),
                    return_exceptions=True,
                ),
                timeout=fill_timeout,
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
            paper_profit = self._calculate_actual_profit(buy_result, sell_result)
            realized_cost = self._calculate_realized_cost(buy_result, sell_result, signal)
            self._total_profit += paper_profit

            # Apply realistic cross-exchange costs (Binance taker fee + rebalancing)
            trade_size_usd = buy_result.filled_quantity * (buy_result.average_fill_price or signal.buy_price)
            realistic = self._realistic_fill.calculate_realistic_costs(
                symbol=signal.symbol,
                trade_size_usd=trade_size_usd,
                paper_profit_usd=paper_profit,
                buy_exchange=signal.buy_exchange,
                sell_exchange=signal.sell_exchange,
            )
            self._realistic_profit += realistic.realistic_profit_usd

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
                actual_profit_usd=paper_profit, realized_cost_bps=realized_cost,
                realistic_costs=realistic,
            )
            self._completed_trades.append(result)

            edge_tag = "EDGE_SURVIVED" if realistic.edge_survived else "EDGE_LOST"
            logger.info(
                f"ARB FILLED: {signal.symbol} | "
                f"Buy: {float(buy_result.filled_quantity):.6f} @ {float(buy_result.average_fill_price or 0):.2f} "
                f"({signal.buy_exchange}) | "
                f"Sell: {float(sell_result.filled_quantity):.6f} @ {float(sell_result.average_fill_price or 0):.2f} "
                f"({signal.sell_exchange}) | "
                f"Paper: ${float(paper_profit):.4f} | "
                f"Realistic: ${float(realistic.realistic_profit_usd):.4f} [{edge_tag}] | "
                f"Costs: wdraw=${float(realistic.withdrawal_fee_usd):.3f} "
                f"taker=${float(realistic.taker_fee_usd):.3f} "
                f"adverse=${float(realistic.adverse_move_usd):.3f}"
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
            buy_detail = f"status={buy_result.status.value}" if isinstance(buy_result, OrderResult) else f"error={type(buy_result).__name__}"
            sell_detail = f"status={sell_result.status.value}" if isinstance(sell_result, OrderResult) else f"error={type(sell_result).__name__}"
            logger.info(f"NO FILL: {trade_id} — buy:{buy_detail}, sell:{sell_detail}")
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

        # BUY fee is in base currency — convert to quote (USDT) for correct accounting.
        # SELL fee is already in quote currency.
        buy_fee_usd = buy.fee_amount * buy.average_fill_price
        sell_fee_usd = sell.fee_amount

        buy_cost = buy.filled_quantity * buy.average_fill_price + buy_fee_usd
        sell_revenue = sell.filled_quantity * sell.average_fill_price - sell_fee_usd
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

        # BUY fee is in base — convert to quote for bps calculation
        buy_fee_usd = buy.fee_amount * buy.average_fill_price if buy.average_fill_price else Decimal('0')
        buy_fee_bps = (buy_fee_usd / (buy.filled_quantity * buy.average_fill_price) * 10000
                       if buy.filled_quantity > 0 and buy.average_fill_price > 0 else Decimal('0'))
        sell_fee_bps = (sell.fee_amount / (sell.filled_quantity * sell.average_fill_price) * 10000
                        if sell.filled_quantity > 0 and sell.average_fill_price > 0 else Decimal('0'))
        return buy_fee_bps + sell_fee_bps + buy_slip + sell_slip

    def _round_qty(self, qty: Decimal, precision) -> Decimal:
        """Round quantity to exchange precision.

        precision can be:
        - int >= 1: decimal place count (DECIMAL_PLACES mode, e.g. MEXC returns 4)
        - float < 1: step size (TICK_SIZE mode, e.g. Binance returns 0.001)
        """
        p = float(precision)
        if 0 < p < 1:
            # TICK_SIZE mode: precision IS the step size (e.g. 0.001)
            step = Decimal(str(precision))
            return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step
        places = int(p)
        if places <= 0:
            return qty.quantize(Decimal('1'), rounding=ROUND_DOWN)
        quant = Decimal(10) ** -places
        return qty.quantize(quant, rounding=ROUND_DOWN)

    def _round_price(self, price: Decimal, precision) -> Decimal:
        """Round price to exchange precision.

        precision can be:
        - int >= 1: decimal place count (DECIMAL_PLACES mode, e.g. MEXC returns 4)
        - float < 1: tick size (TICK_SIZE mode, e.g. Binance returns 0.01)
        """
        p = float(precision)
        if 0 < p < 1:
            # TICK_SIZE mode: precision IS the tick size
            tick = Decimal(str(precision))
            return (price / tick).to_integral_value(rounding=ROUND_DOWN) * tick
        places = int(p)
        if places <= 0:
            return price.quantize(Decimal('1'), rounding=ROUND_DOWN)
        quant = Decimal(10) ** -places
        return price.quantize(quant, rounding=ROUND_DOWN)

    def get_stats(self) -> dict:
        wins = [t for t in self._completed_trades if t.actual_profit_usd > 0]
        losses = [t for t in self._completed_trades if t.actual_profit_usd < 0]
        one_sided = [t for t in self._completed_trades if "one_sided" in t.status]
        # Realistic wins = trades where edge survived after realistic costs
        realistic_wins = [
            t for t in self._completed_trades
            if t.realistic_costs and t.realistic_costs.edge_survived
        ]
        return {
            "total_trades": self._trade_count,
            "total_fills": self._fill_count,
            "paper_profit_usd": float(self._total_profit),
            "realistic_profit_usd": float(self._realistic_profit),
            "total_profit_usd": float(self._realistic_profit),  # Use realistic as canonical
            "wins": len(wins),
            "losses": len(losses),
            "realistic_wins": len(realistic_wins),
            "one_sided_events": len(one_sided),
            "win_rate": len(wins) / max(1, len(wins) + len(losses)),
            "edge_survival_rate": len(realistic_wins) / max(1, self._fill_count),
            "avg_profit_per_fill": float(self._realistic_profit / max(1, self._fill_count)),
            "depth_rejects": self._depth_rejects,
            "freshness_rejects": self._freshness_rejects,
            "realistic_fill_stats": self._realistic_fill.get_stats(),
        }
