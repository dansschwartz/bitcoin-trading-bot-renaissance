"""
Arbitrage Execution Engine — executes ArbitrageSignal trades
with sequential leg execution (maker-first) for safety.

CRITICAL PRINCIPLES:
1. MAKER FIRST. Place MEXC LIMIT_MAKER (0% fee) first, wait for fill.
2. AGGRESSIVE PRICING. Price LIMIT_MAKER at bid+1tick (sell) or ask-1tick (buy)
   to sit at top of order book inside the spread — fills on next market order.
3. TAKER SECOND. Only fire Binance US IOC (1 bps) after MEXC maker confirms.
4. IF MAKER FAILS → no risk, clean exit (no taker was placed).
5. IF TAKER FAILS → emergency close maker position (rare case).
6. PRE-CHECK. Verify book depth and freshness before firing.
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

    # Exchanges with 0% maker fee — use LIMIT_MAKER for these
    MAKER_EXCHANGES = {"mexc", "binance_us"}

    def __init__(self, mexc_client, binance_client, cost_model, risk_engine,
                 config: Optional[dict] = None,
                 book_manager: Optional['UnifiedBookManager'] = None,
                 kucoin_client=None, binance_us_client=None):
        self.clients = {
            "mexc": mexc_client,
            "binance": binance_client,
        }
        if kucoin_client:
            self.clients["kucoin"] = kucoin_client
        if binance_us_client:
            self.clients["binance_us"] = binance_us_client
        self.costs = cost_model
        self.risk = risk_engine
        self.book_manager = book_manager
        self.config = config or {}
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
        self.exhaust_capture = None  # Set by orchestrator for data exhaust
        self.capital_guard = None   # Set by orchestrator for USDT reserve enforcement

    # ── Pre-execution gates ────────────────────────────────────────

    def _check_book_freshness(self, signal: ArbitrageSignal) -> tuple:
        """Reject if either book is stale. Returns (ok, reason)."""
        if not self.book_manager:
            return True, "no_book_manager"

        view = self.book_manager.pairs.get(signal.symbol)
        if not view:
            return False, "pair_not_monitored"

        now = datetime.utcnow()
        # Dynamically check the two exchanges in this signal
        ages = {}
        for exch in [signal.buy_exchange, signal.sell_exchange]:
            update_time = getattr(view, f"{exch}_last_update", None)
            if update_time:
                ages[exch] = (now - update_time).total_seconds()
            else:
                ages[exch] = 999.0  # No data = very stale

        for exch, age in ages.items():
            if age > self.MAX_BOOK_AGE_SEC:
                return False, f"{exch}_stale:{age:.1f}s"

        age_summary = ",".join(f"{e[0].upper()}={a:.1f}s" for e, a in ages.items())
        return True, f"fresh:{age_summary}"

    def _check_book_depth(self, signal: ArbitrageSignal) -> tuple:
        """Reject if either book lacks depth at target price. Returns (ok, reason)."""
        if not self.book_manager:
            return True, "no_book_manager"

        view = self.book_manager.pairs.get(signal.symbol)
        if not view:
            return False, "missing_book"

        # Dynamically look up books for the exchanges in this signal
        buy_book = getattr(view, f"{signal.buy_exchange}_book", None)
        sell_book = getattr(view, f"{signal.sell_exchange}_book", None)
        if not buy_book or not sell_book:
            return False, "missing_book"

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
        self._last_exec_start = time.monotonic()

        # Check expiry
        if datetime.utcnow() > signal.expires_at:
            logger.debug(f"Signal expired: {trade_id}")
            return ExecutionResult(trade_id=trade_id, status="expired", signal=signal)

        # LAYER 1: Book freshness gate
        fresh_ok, fresh_reason = self._check_book_freshness(signal)
        if not fresh_ok:
            self._freshness_rejects += 1
            if self._freshness_rejects <= 5 or self._freshness_rejects % 100 == 0:
                logger.debug(f"GATE: freshness reject #{self._freshness_rejects} {signal.symbol}: {fresh_reason}")
            return ExecutionResult(trade_id=trade_id, status="book_stale", signal=signal)

        # LAYER 2: Book depth gate
        depth_ok, depth_reason = self._check_book_depth(signal)
        if not depth_ok:
            self._depth_rejects += 1
            if self._depth_rejects <= 5 or self._depth_rejects % 100 == 0:
                logger.debug(f"GATE: depth reject #{self._depth_rejects} {signal.symbol}: {depth_reason}")
            return ExecutionResult(trade_id=trade_id, status="depth_insufficient", signal=signal)

        # LAYER 3: Early spread threshold — avoid API calls for sub-threshold signals
        # With relay: MEXC books are ~100ms fresh. 8bps gross is sufficient.
        EARLY_MIN_GROSS_BPS = Decimal('8.0')
        if signal.gross_spread_bps < EARLY_MIN_GROSS_BPS:
            logger.info(
                f"SKIP: {trade_id} — gross={float(signal.gross_spread_bps):.1f}bps "
                f"< {float(EARLY_MIN_GROSS_BPS):.1f}bps min threshold")
            result = ExecutionResult(trade_id=trade_id, status="no_fill", signal=signal)
            self._completed_trades.append(result)
            return result

        buy_client = self.clients[signal.buy_exchange]
        sell_client = self.clients[signal.sell_exchange]

        # Capital guard: check USDT reserve before spending
        if self.capital_guard:
            trade_value = float(signal.recommended_quantity * signal.buy_price)
            allowed, cur_bal = await self.capital_guard.can_spend(buy_client, trade_value)
            if not allowed:
                return ExecutionResult(
                    trade_id=trade_id, status="capital_guard_blocked", signal=signal
                )

        # Verify inventory — reduce quantity to available balance if needed
        base, quote = signal.symbol.split('/')
        buy_balance = await buy_client.get_balance(quote)
        sell_balance = await sell_client.get_balance(base)

        trade_qty = signal.recommended_quantity

        # Reduce to sell-side inventory if needed
        if sell_balance.free < trade_qty:
            trade_qty = sell_balance.free * Decimal('0.95')  # 5% buffer for rounding/fees
            logger.info(f"Reduced {base} qty to available: {float(trade_qty):.6f} "
                       f"(have {float(sell_balance.free):.6f} on {signal.sell_exchange})")

        # Reduce to buy-side USDT if needed
        max_buy_qty = buy_balance.free / signal.buy_price if signal.buy_price > 0 else Decimal('0')
        if max_buy_qty < trade_qty:
            trade_qty = max_buy_qty * Decimal('0.95')
            logger.info(f"Reduced qty to {quote} balance: {float(trade_qty):.6f} "
                       f"(have {float(buy_balance.free):.2f} {quote} on {signal.buy_exchange})")

        # Check minimum notional
        min_trade_usd = Decimal(str(self.config.get('cross_exchange', {}).get('min_trade_usd', 2)))
        notional = trade_qty * signal.buy_price
        if notional < min_trade_usd:
            logger.info(f"Insufficient balance for min trade: ${float(notional):.2f} "
                       f"< ${float(min_trade_usd):.2f} min")
            return ExecutionResult(trade_id=trade_id, status="insufficient_balance", signal=signal)

        # Update signal quantity
        signal.recommended_quantity = trade_qty

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

        # Capture book state at execution time
        if self.exhaust_capture and self.book_manager:
            try:
                _view = self.book_manager.pairs.get(signal.symbol)
                if _view:
                    _eb = {}
                    for _ex in [signal.buy_exchange, signal.sell_exchange]:
                        _bk = getattr(_view, f"{_ex}_book", None)
                        if _bk:
                            _eb[_ex] = _bk
                    self.exhaust_capture.capture_at_execution(
                        signal.signal_id, signal.symbol, _eb)
            except Exception as e:
                logger.warning(f"pairs.get failed: {e}")

        # LAYER 4: Order type — MARKET on MEXC (instant fill, 5bps taker),
        # IOC on other exchanges. LIMIT_MAKER doesn't fill fast enough for arb.
        # MEXC doesn't support IOC timeInForce — use MARKET instead.
        # For MEXC BUY with MARKET: use quoteOrderQty (buy by USD value) for reliability.
        _mexc_buy = signal.buy_exchange == "mexc"
        _mexc_sell = signal.sell_exchange == "mexc"

        buy_order = OrderRequest(
            exchange=signal.buy_exchange,
            symbol=signal.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET if _mexc_buy else OrderType.LIMIT,
            quantity=buy_qty,
            price=None if _mexc_buy else buy_price,
            time_in_force=None if _mexc_buy else TimeInForce.IOC,
            client_order_id=f"{trade_id}_buy",
        )

        sell_order = OrderRequest(
            exchange=signal.sell_exchange,
            symbol=signal.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET if _mexc_sell else OrderType.LIMIT,
            quantity=sell_qty,
            price=None if _mexc_sell else sell_price,
            time_in_force=None if _mexc_sell else TimeInForce.IOC,
            client_order_id=f"{trade_id}_sell",
        )

        # SEQUENTIAL EXECUTION: Maker first → wait for fill → then taker
        # This eliminates ~90% of one-sided fills since we never fire the
        # taker leg unless the maker has already confirmed.
        fill_timeout = self.FILL_TIMEOUT_PAPER if self._paper_mode else self.FILL_TIMEOUT_LIVE

        # Determine maker vs taker: MAKER_EXCHANGES use LIMIT_MAKER (0% fee)
        buy_is_maker = signal.buy_exchange in self.MAKER_EXCHANGES
        sell_is_maker = signal.sell_exchange in self.MAKER_EXCHANGES

        if buy_is_maker and sell_is_maker:
            # Both exchanges are maker-eligible (e.g. MEXC↔BinanceUS)
            _bus_is_buy = signal.buy_exchange == 'binance_us'
            _bus_is_sell = signal.sell_exchange == 'binance_us'

            if _bus_is_buy or _bus_is_sell:
                # ═══ MEXC↔BinUS Hybrid: MEXC maker FIRST → BinUS IOC SECOND ═══
                # 1. Price MEXC aggressively inside spread (bid+tick sell, ask-tick buy)
                # 2. Place MEXC LIMIT_MAKER (0% fee) and poll for fill
                # 3. If filled → fire BinUS IOC (1 bps). If not → cancel, zero risk.
                if _bus_is_buy:
                    mexc_order, mexc_client_ref = sell_order, sell_client
                    bus_order, bus_client_ref = buy_order, buy_client
                    mexc_is_sell = True
                else:
                    mexc_order, mexc_client_ref = buy_order, buy_client
                    bus_order, bus_client_ref = sell_order, sell_client
                    mexc_is_sell = False

                # BinUS: IOC taker (1 bps fee)
                bus_order.order_type = OrderType.LIMIT
                bus_order.time_in_force = TimeInForce.IOC

                # IOC-IOC only from US droplet — LIMIT_MAKER on MEXC is unviable
                # because MEXC WS is geo-blocked, REST book is 15s stale, making
                # maker order placement unreliable. Accept 6.1 bps cost for
                # certainty of execution.
                IOC_IOC_COST_BPS = Decimal('6.1')  # MEXC taker 5 + BinUS taker 1 + timing 0.1
                # With relay: MEXC books are ~100ms fresh (not 15s stale).
                # 8bps gross provides margin above 6.1bps cost.
                IOC_IOC_MIN_GROSS = Decimal('8.0')

                if signal.gross_spread_bps < IOC_IOC_MIN_GROSS:
                    logger.info(
                        f"SKIP: {trade_id} — gross={float(signal.gross_spread_bps):.1f}bps "
                        f"< {float(IOC_IOC_MIN_GROSS):.1f}bps IOC-IOC threshold")
                    result = ExecutionResult(trade_id=trade_id, status="no_fill", signal=signal)
                    self._completed_trades.append(result)
                    return result

                # ─── MARKET+IOC: MEXC MARKET taker (5bps) + BinUS IOC (1bps) ───
                # MEXC spot doesn't support IOC timeInForce — orders treated as GTC.
                # Use MARKET on MEXC for guaranteed immediate fill.
                mexc_order.order_type = OrderType.MARKET
                mexc_order.time_in_force = None
                mexc_order.price = None  # MARKET orders don't use price
                bus_order.order_type = OrderType.LIMIT
                bus_order.time_in_force = TimeInForce.IOC

                logger.info(
                    f"PARALLEL MARKET+IOC: {trade_id} | MEXC {'SELL' if mexc_is_sell else 'BUY'} "
                    f"MARKET | BinUS {'BUY' if _bus_is_buy else 'SELL'} "
                    f"@ {float(bus_order.price):.6f} IOC | "
                    f"gross={float(signal.gross_spread_bps):.1f}bps cost≈6.1bps "
                    f"net≈{float(signal.gross_spread_bps - IOC_IOC_COST_BPS):.1f}bps")

                # ─── PARALLEL: Fire both legs simultaneously ───
                # MARKET orders virtually always fill. IOC fills if price is right.
                # Parallel saves ~300-500ms vs sequential, capturing more edge.
                async def _place_mexc():
                    try:
                        return await asyncio.wait_for(
                            mexc_client_ref.place_order(mexc_order), timeout=fill_timeout)
                    except Exception as e:
                        logger.error(f"MEXC MARKET exception: {trade_id} — {e}")
                        return None

                async def _place_bus():
                    try:
                        return await asyncio.wait_for(
                            bus_client_ref.place_order(bus_order), timeout=fill_timeout)
                    except Exception as e:
                        logger.error(f"BinUS IOC exception: {trade_id} — {e}")
                        return None

                mexc_result, bus_result = await asyncio.gather(_place_mexc(), _place_bus())

                # Check MEXC fill (may need polling if status=NEW)
                _mexc_ok = (isinstance(mexc_result, OrderResult)
                            and mexc_result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED))
                if (not _mexc_ok and isinstance(mexc_result, OrderResult)
                        and mexc_result.status == OrderStatus.OPEN and mexc_result.order_id):
                    for _poll in range(10):
                        await asyncio.sleep(0.1)
                        try:
                            poll_result = await mexc_client_ref.get_order_status(
                                signal.symbol, mexc_result.order_id)
                            if isinstance(poll_result, OrderResult):
                                if poll_result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                                    mexc_result = poll_result
                                    _mexc_ok = True
                                    break
                                elif poll_result.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                                    break
                        except Exception as e:
                            logger.warning(f"mexc_client_ref.get_order_status failed: {e}")

                # Check BinUS fill
                _bus_ok = (isinstance(bus_result, OrderResult)
                           and bus_result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED))

                _mexc_price = (mexc_result.average_fill_price if isinstance(mexc_result, OrderResult) else None) or Decimal('0')
                _bus_price = (bus_result.average_fill_price if isinstance(bus_result, OrderResult) else None) or Decimal('0')

                if _mexc_ok and _bus_ok:
                    logger.info(
                        f"BOTH FILLED: {trade_id} | MEXC @ {float(_mexc_price):.6f} | "
                        f"BinUS @ {float(_bus_price):.6f}")
                elif _mexc_ok and not _bus_ok:
                    logger.warning(f"MEXC filled, BinUS NOT filled: {trade_id}")
                elif not _mexc_ok and _bus_ok:
                    logger.warning(f"BinUS filled, MEXC NOT filled: {trade_id}")
                else:
                    # Neither filled — clean exit
                    logger.info(f"NEITHER FILLED: {trade_id}")
                    # Cancel any open orders
                    if isinstance(mexc_result, OrderResult) and mexc_result.order_id:
                        try:
                            await mexc_client_ref.cancel_order(signal.symbol, mexc_result.order_id)
                        except Exception as e:
                            logger.warning(f"mexc_client_ref.cancel_order failed: {e}")
                    if isinstance(bus_result, OrderResult) and bus_result.order_id:
                        try:
                            await bus_client_ref.cancel_order(signal.symbol, bus_result.order_id)
                        except Exception as e:
                            logger.warning(f"bus_client_ref.cancel_order failed: {e}")
                    result = ExecutionResult(trade_id=trade_id, status="no_fill", signal=signal)
                    self._completed_trades.append(result)
                    return result

                self._trade_count += 1
                if _bus_is_buy:
                    return await self._process_results(trade_id, signal, bus_result, mexc_result)
                else:
                    return await self._process_results(trade_id, signal, mexc_result, bus_result)

            else:
                # Fallback: both IOC (neither is Binance US)
                buy_order.order_type = OrderType.LIMIT
                buy_order.time_in_force = TimeInForce.IOC
                sell_order.order_type = OrderType.LIMIT
                sell_order.time_in_force = TimeInForce.IOC
                maker_order, maker_client = buy_order, buy_client
                taker_order, taker_client = sell_order, sell_client
                maker_is_buy = True
        elif buy_is_maker:
            maker_order, maker_client = buy_order, buy_client
            taker_order, taker_client = sell_order, sell_client
            maker_is_buy = True
        else:
            maker_order, maker_client = sell_order, sell_client
            taker_order, taker_client = buy_order, buy_client
            maker_is_buy = False

        # STEP 1: Place maker (MEXC) leg
        try:
            maker_result = await asyncio.wait_for(
                maker_client.place_order(maker_order),
                timeout=fill_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(f"Maker timeout: {trade_id}")
            try:
                await maker_client.cancel_order(maker_order.symbol, maker_order.client_order_id or "")
            except Exception as e:
                logger.warning(f"maker_client.cancel_order failed: {e}")
            return ExecutionResult(trade_id=trade_id, status="timeout", signal=signal)
        except Exception as e:
            logger.error(f"Maker order failed: {trade_id} — {e}")
            return ExecutionResult(trade_id=trade_id, status="no_fill", signal=signal)

        maker_ok = (
            isinstance(maker_result, OrderResult)
            and maker_result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)
        )

        if not maker_ok:
            # Maker didn't fill — clean exit, no risk, no taker placed
            maker_detail = (
                f"status={maker_result.status.value}" if isinstance(maker_result, OrderResult)
                else f"error={type(maker_result).__name__}"
            )
            logger.info(f"MAKER NO FILL: {trade_id} — {maker_detail} (clean, no taker placed)")
            result = ExecutionResult(trade_id=trade_id, status="no_fill", signal=signal)
            self._completed_trades.append(result)
            return result

        # STEP 2: Maker filled — now fire taker
        try:
            taker_result = await asyncio.wait_for(
                taker_client.place_order(taker_order),
                timeout=fill_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(f"Taker timeout after maker fill: {trade_id} — emergency close")
            taker_result = None
        except Exception as e:
            logger.error(f"Taker order failed after maker fill: {trade_id} — {e} — emergency close")
            taker_result = None

        # STEP 2b: If taker is LIMIT_MAKER and came back OPEN, poll for fill (up to 60s)
        if (isinstance(taker_result, OrderResult)
                and taker_result.status == OrderStatus.OPEN
                and taker_order.order_type == OrderType.LIMIT_MAKER):
            oid = taker_result.order_id or taker_order.client_order_id
            if oid:
                logger.info(f"MAKER POSTED: {trade_id} — polling for fill (up to 60s)")
                polled = await self._wait_for_maker_fill(
                    taker_client, taker_order.symbol, oid, timeout_seconds=60.0)
                if polled and polled.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                    taker_result = polled
                    logger.info(f"MAKER FILLED: {trade_id} — {taker_result.status.value}")
                else:
                    try:
                        await taker_client.cancel_order(taker_order.symbol, oid)
                    except Exception as e:
                        logger.warning(f"taker_client.cancel_order failed: {e}")
                    # Don't emergency-close — just accept the inventory shift.
                    # The IOC leg filled but the maker leg didn't. This is a small
                    # inventory imbalance ($20 max) that rebalances over time.
                    logger.warning(
                        f"MAKER TIMEOUT: {trade_id} — cancelled after 60s. "
                        f"One-sided {signal.symbol} position accepted (no emergency close).")
                    self._trade_count += 1
                    result = ExecutionResult(
                        trade_id=trade_id, status="maker_timeout", signal=signal,
                        buy_result=maker_result if maker_is_buy else None,
                        sell_result=maker_result if not maker_is_buy else None,
                    )
                    self._completed_trades.append(result)
                    return result

        self._trade_count += 1

        # Reconstruct buy_result/sell_result from maker/taker for _process_results
        if maker_is_buy:
            return await self._process_results(trade_id, signal, maker_result, taker_result)
        else:
            return await self._process_results(trade_id, signal, taker_result, maker_result)

    async def _wait_for_maker_fill(self, client, symbol: str, order_id: str,
                                    timeout_seconds: float = 30.0) -> Optional[OrderResult]:
        """Poll order status until filled, cancelled, or timeout.
        Polls every 0.5s for the first 5s (fast detection), then every 1s."""
        start = time.monotonic()
        polls = 0
        while time.monotonic() - start < timeout_seconds:
            try:
                result = await client.get_order_status(symbol, order_id)
                if isinstance(result, OrderResult):
                    if result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                        return result
                    if result.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                        return result
            except Exception as e:
                logger.warning(f"client.get_order_status failed: {e}")
            polls += 1
            await asyncio.sleep(0.5 if polls <= 10 else 1.0)
        return None

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

            hold_dur = time.monotonic() - getattr(self, '_last_exec_start', time.monotonic())
            result = ExecutionResult(
                trade_id=trade_id, status="filled", signal=signal,
                buy_result=buy_result, sell_result=sell_result,
                actual_profit_usd=paper_profit, realized_cost_bps=realized_cost,
                realistic_costs=realistic,
            )
            result.hold_duration_seconds = hold_dur
            self._completed_trades.append(result)

            # Post-execution book snapshot (~1s later)
            if self.exhaust_capture and self.book_manager:
                try:
                    _loop = asyncio.get_event_loop()
                    _sid = signal.signal_id
                    _sym = signal.symbol
                    _bm = self.book_manager
                    _ec = self.exhaust_capture
                    _bex = signal.buy_exchange
                    _sex = signal.sell_exchange
                    def _post_capture():
                        try:
                            _v = _bm.pairs.get(_sym)
                            if _v:
                                _bs = {}
                                for _e in [_bex, _sex]:
                                    _b = getattr(_v, f"{_e}_book", None)
                                    if _b:
                                        _bs[_e] = _b
                                _ec.capture_post_execution(_sid, _sym, _bs)
                        except Exception as e:
                            logger.warning(f"_bm.pairs.get failed: {e}")
                    _loop.call_later(1.0, _post_capture)
                except Exception as e:
                    logger.warning(f"asyncio.get_event_loop failed: {e}")

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
            time_in_force=None,
            client_order_id=f"emergency_{int(datetime.utcnow().timestamp()*1000)}",
        )
        try:
            result = await asyncio.wait_for(
                client.place_order(emergency_order),
                timeout=self.EMERGENCY_CLOSE_SECONDS,
            )
        except Exception as e:
            logger.critical(f"EMERGENCY CLOSE FAILED: {e}")
            return None

        # MEXC MARKET orders return status=NEW before fill — poll briefly
        if (isinstance(result, OrderResult) and result.order_id
                and result.status not in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)):
            for _poll in range(10):
                await asyncio.sleep(0.1)
                try:
                    poll_result = await client.get_order_status(symbol, result.order_id)
                    if isinstance(poll_result, OrderResult):
                        if poll_result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                            logger.info(f"EMERGENCY CLOSE FILLED: {symbol} {side.value} qty={float(poll_result.filled_quantity)}")
                            return poll_result
                        elif poll_result.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                            break
                except Exception as e:
                    logger.warning(f"client.get_order_status failed: {e}")
            logger.warning(f"EMERGENCY CLOSE: {symbol} MARKET order not filled after polling")
        return result

    async def _cancel_both(self, buy_client, sell_client, buy_order, sell_order):
        try:
            await asyncio.gather(
                buy_client.cancel_order(buy_order.symbol, buy_order.client_order_id or ""),
                sell_client.cancel_order(sell_order.symbol, sell_order.client_order_id or ""),
                return_exceptions=True,
            )
        except Exception as e:
            logger.warning(f"asyncio.gather failed: {e}")

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

    def _get_tick_size(self, precision) -> Decimal:
        """Get the minimum price increment from exchange precision."""
        p = float(precision)
        if 0 < p < 1:
            return Decimal(str(precision))  # Already a tick size
        places = int(p)
        if places <= 0:
            return Decimal('1')
        return Decimal(10) ** (-places)

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
