"""
Recovery Engine — startup reconciliation for incomplete trades.

On every bot start this engine:
1. Sets the system state to RECOVERING.
2. Verifies exchange connectivity (Coinbase for directional, MEXC + Binance for
   arbitrage).
3. Loads incomplete trades from the state store.
4. Reconciles each trade against exchange order status.
5. Handles one-sided fills with emergency close logic.
6. Checks for orphaned orders that are not tracked locally.
7. Alerts on anomalies via the project's AlertManager (with graceful fallback
   to logging when Telegram/Slack is not configured).
8. Transitions to RUNNING on success, or HALTED on critical failure.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from recovery.state_manager import (
    ActiveTrade,
    StateManager,
    SystemState,
    TradeLifecycleState,
)

logger = logging.getLogger("recovery.engine")

# ---------------------------------------------------------------------------
# Project root — used to import exchange clients & alerter dynamically
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Recovery result
# ---------------------------------------------------------------------------

@dataclass
class RecoveryResult:
    """Summary returned by :meth:`RecoveryEngine.run`."""
    success: bool = False
    trades_found: int = 0
    trades_reconciled: int = 0
    trades_emergency_closed: int = 0
    orphaned_orders: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RecoveryEngine
# ---------------------------------------------------------------------------

class RecoveryEngine:
    """
    Runs once at startup to bring the system into a consistent state.

    Parameters
    ----------
    state_manager : StateManager
        The shared state manager instance.
    coinbase_client : object | None
        An ``EnhancedCoinbaseClient`` (from ``coinbase_client.py``) for
        directional trading recovery.  May be ``None`` if not configured.
    mexc_client : object | None
        A ``MEXCClient`` (from ``arbitrage.exchanges.mexc_client``) for
        arbitrage recovery.
    binance_client : object | None
        A ``BinanceClient`` (from ``arbitrage.exchanges.binance_client``) for
        arbitrage recovery.
    alert_manager : object | None
        An ``AlertManager`` instance for sending alerts.  Falls back to
        logging if ``None``.
    max_emergency_close_age_seconds : float
        Maximum age (in seconds) of a one-sided fill before emergency close
        is attempted.  Default: 300 (5 minutes).
    """

    def __init__(
        self,
        state_manager: StateManager,
        coinbase_client: Any = None,
        mexc_client: Any = None,
        binance_client: Any = None,
        alert_manager: Any = None,
        max_emergency_close_age_seconds: float = 300.0,
    ) -> None:
        self._sm = state_manager
        self._coinbase = coinbase_client
        self._mexc = mexc_client
        self._binance = binance_client
        self._alerter = alert_manager
        self._max_age = max_emergency_close_age_seconds

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> RecoveryResult:
        """Execute the full recovery sequence.

        Returns a :class:`RecoveryResult` summarising what happened.
        """
        result = RecoveryResult()

        try:
            # 1. Mark system as recovering
            await self._sm.aset_system_state(SystemState.RECOVERING, "startup recovery")
            logger.info("=== Recovery Engine starting ===")

            # 2. Verify exchange connectivity
            connectivity_ok = await self._verify_connectivity(result)
            if not connectivity_ok:
                await self._alert(
                    "CRITICAL",
                    "Exchange Connectivity Failed",
                    "Recovery aborted — could not reach one or more exchanges.",
                )
                await self._sm.aset_system_state(
                    SystemState.HALTED, "exchange connectivity failed"
                )
                result.success = False
                return result

            # 3. Load incomplete trades
            incomplete = await self._sm.aget_incomplete_trades()
            result.trades_found = len(incomplete)
            logger.info("Found %d incomplete trade(s) to reconcile", len(incomplete))

            # 4. Reconcile each trade
            for trade in incomplete:
                try:
                    await self._reconcile_trade(trade, result)
                    result.trades_reconciled += 1
                except Exception as exc:
                    msg = f"Failed to reconcile trade {trade.trade_id}: {exc}"
                    logger.error(msg)
                    result.errors.append(msg)
                    await self._sm.aupdate_trade_state(
                        trade.trade_id,
                        TradeLifecycleState.FAILED,
                        error_message=str(exc),
                    )

            # 5. Check for orphaned orders
            orphaned = await self._check_orphaned_orders(result)
            result.orphaned_orders = orphaned

            # 6. Summarise
            if result.errors:
                await self._alert(
                    "WARNING",
                    "Recovery Completed With Errors",
                    f"Trades found={result.trades_found}, reconciled={result.trades_reconciled}, "
                    f"emergency_closed={result.trades_emergency_closed}, errors={len(result.errors)}",
                )
            else:
                logger.info(
                    "Recovery complete: found=%d reconciled=%d emergency_closed=%d orphaned=%d",
                    result.trades_found,
                    result.trades_reconciled,
                    result.trades_emergency_closed,
                    result.orphaned_orders,
                )

            # 7. Transition to RUNNING
            await self._sm.aset_system_state(SystemState.RUNNING, "recovery complete")
            result.success = True

        except Exception as exc:
            tb = traceback.format_exc()
            logger.critical("Recovery engine failed: %s\n%s", exc, tb)
            result.errors.append(str(exc))
            result.success = False
            await self._sm.aset_system_state(
                SystemState.HALTED, f"recovery engine exception: {exc}"
            )
            await self._alert(
                "CRITICAL",
                "Recovery Engine Failure",
                f"Unhandled exception during recovery: {exc}",
            )

        return result

    # ------------------------------------------------------------------
    # Connectivity verification
    # ------------------------------------------------------------------

    async def _verify_connectivity(self, result: RecoveryResult) -> bool:
        """Check that we can reach the exchanges we need.

        Returns ``True`` if at least one exchange is reachable.
        """
        any_ok = False

        # Coinbase (directional)
        if self._coinbase is not None:
            try:
                # EnhancedCoinbaseClient exposes get_accounts() or get_product()
                if hasattr(self._coinbase, "get_accounts"):
                    await asyncio.to_thread(self._coinbase.get_accounts)
                elif hasattr(self._coinbase, "get_product"):
                    await asyncio.to_thread(self._coinbase.get_product, "BTC-USD")
                logger.info("Coinbase connectivity OK")
                any_ok = True
            except Exception as exc:
                msg = f"Coinbase connectivity check failed: {exc}"
                logger.warning(msg)
                result.warnings.append(msg)

        # MEXC (arbitrage)
        if self._mexc is not None:
            try:
                if hasattr(self._mexc, "connect"):
                    await self._mexc.connect()
                if hasattr(self._mexc, "get_all_tickers"):
                    await self._mexc.get_all_tickers()
                logger.info("MEXC connectivity OK")
                any_ok = True
            except Exception as exc:
                msg = f"MEXC connectivity check failed: {exc}"
                logger.warning(msg)
                result.warnings.append(msg)

        # Binance (arbitrage)
        if self._binance is not None:
            try:
                if hasattr(self._binance, "connect"):
                    await self._binance.connect()
                if hasattr(self._binance, "get_all_tickers"):
                    await self._binance.get_all_tickers()
                logger.info("Binance connectivity OK")
                any_ok = True
            except Exception as exc:
                msg = f"Binance connectivity check failed: {exc}"
                logger.warning(msg)
                result.warnings.append(msg)

        # If no exchange clients were provided at all, that is fine — the bot
        # may be running in a mode that does not need recovery.
        if (
            self._coinbase is None
            and self._mexc is None
            and self._binance is None
        ):
            logger.info("No exchange clients configured — skipping connectivity check")
            return True

        return any_ok

    # ------------------------------------------------------------------
    # Trade reconciliation (6 scenarios)
    # ------------------------------------------------------------------

    async def _reconcile_trade(
        self,
        trade: ActiveTrade,
        result: RecoveryResult,
    ) -> None:
        """Reconcile a single incomplete trade against exchange state.

        Scenarios handled:
            1. PENDING — order was never submitted.  Mark as CANCELLED.
            2. BUY_SUBMITTED — check exchange for fill status.
            3. BUY_FILLED — buy side done, sell never submitted.  Emergency close.
            4. SELL_SUBMITTED — check exchange for fill status.
            5. SELL_FILLED / PARTIALLY_FILLED — compute actual P&L, complete.
            6. CANCELLING — verify cancellation on exchange, finalise.
        """
        logger.info(
            "Reconciling trade %s  state=%s  type=%s  symbol=%s",
            trade.trade_id, trade.state.value, trade.signal_type, trade.symbol,
        )

        if trade.state == TradeLifecycleState.PENDING:
            # Scenario 1: never submitted — just cancel
            await self._sm.aupdate_trade_state(
                trade.trade_id,
                TradeLifecycleState.CANCELLED,
                error_message="Recovered at startup: order was never submitted",
            )
            result.warnings.append(f"Trade {trade.trade_id}: cancelled (was PENDING)")
            return

        if trade.state == TradeLifecycleState.BUY_SUBMITTED:
            # Scenario 2: check whether the buy was filled on the exchange
            filled, fill_qty, fill_price = await self._check_order_status(
                trade.buy_exchange, trade.symbol, trade.buy_order_id
            )
            if filled:
                # Buy filled — but sell was never started.  Emergency close.
                await self._sm.aupdate_trade_state(
                    trade.trade_id,
                    TradeLifecycleState.BUY_FILLED,
                    buy_filled_qty=fill_qty,
                    buy_fill_price=fill_price,
                )
                await self._emergency_close(trade, "buy", fill_qty, result)
            else:
                # Buy still open or was not filled — cancel it
                await self._try_cancel_order(
                    trade.buy_exchange, trade.symbol, trade.buy_order_id
                )
                await self._sm.aupdate_trade_state(
                    trade.trade_id,
                    TradeLifecycleState.CANCELLED,
                    error_message="Recovered: buy order cancelled at startup",
                )
                result.warnings.append(
                    f"Trade {trade.trade_id}: buy order cancelled (unfilled)"
                )
            return

        if trade.state == TradeLifecycleState.BUY_FILLED:
            # Scenario 3: buy filled, sell never submitted — emergency close
            fill_qty = trade.buy_filled_qty or trade.buy_quantity
            await self._emergency_close(trade, "buy", fill_qty, result)
            return

        if trade.state == TradeLifecycleState.SELL_SUBMITTED:
            # Scenario 4: check sell order
            filled, fill_qty, fill_price = await self._check_order_status(
                trade.sell_exchange, trade.symbol, trade.sell_order_id
            )
            if filled:
                profit = self._compute_profit(trade, fill_qty, fill_price)
                await self._sm.acomplete_trade(
                    trade.trade_id, actual_profit_usd=profit,
                )
                logger.info(
                    "Trade %s sell confirmed filled. Profit=%.4f",
                    trade.trade_id, profit,
                )
            else:
                # Sell not filled — try to cancel and emergency close
                await self._try_cancel_order(
                    trade.sell_exchange, trade.symbol, trade.sell_order_id
                )
                fill_qty = trade.buy_filled_qty or trade.buy_quantity
                await self._emergency_close(trade, "buy", fill_qty, result)
            return

        if trade.state in (
            TradeLifecycleState.SELL_FILLED,
            TradeLifecycleState.PARTIALLY_FILLED,
        ):
            # Scenario 5: sell side is at least partially done
            profit = self._compute_profit(
                trade, trade.sell_filled_qty, trade.sell_fill_price,
            )
            await self._sm.acomplete_trade(
                trade.trade_id, actual_profit_usd=profit,
            )
            result.warnings.append(
                f"Trade {trade.trade_id}: completed from {trade.state.value} "
                f"with profit={profit:.4f}"
            )
            return

        if trade.state == TradeLifecycleState.CANCELLING:
            # Scenario 6: was mid-cancel
            for oid, exch in [
                (trade.buy_order_id, trade.buy_exchange),
                (trade.sell_order_id, trade.sell_exchange),
            ]:
                if oid:
                    await self._try_cancel_order(exch, trade.symbol, oid)
            await self._sm.aupdate_trade_state(
                trade.trade_id,
                TradeLifecycleState.CANCELLED,
                error_message="Recovered: cancellation finalised at startup",
            )
            return

        # Fallback: unknown / unexpected state
        msg = f"Trade {trade.trade_id} in unexpected state {trade.state.value}"
        logger.warning(msg)
        result.warnings.append(msg)
        await self._sm.aupdate_trade_state(
            trade.trade_id,
            TradeLifecycleState.FAILED,
            error_message=f"Recovered: unexpected state {trade.state.value}",
        )

    # ------------------------------------------------------------------
    # Exchange helpers
    # ------------------------------------------------------------------

    def _get_exchange_client(self, exchange_name: str) -> Any:
        """Return the exchange client for the given name, or None."""
        name = (exchange_name or "").lower()
        if name in ("coinbase", "coinbase_advanced", "cb", ""):
            return self._coinbase
        if name in ("mexc", "mexc_global"):
            return self._mexc
        if name in ("binance", "binance_spot"):
            return self._binance
        return None

    async def _check_order_status(
        self,
        exchange_name: str,
        symbol: str,
        order_id: str,
    ) -> Tuple[bool, float, float]:
        """Query the exchange for an order's fill status.

        Returns (is_filled, filled_quantity, average_fill_price).
        If the exchange client is unavailable or the query fails, returns
        ``(False, 0.0, 0.0)``.
        """
        if not order_id:
            return (False, 0.0, 0.0)

        client = self._get_exchange_client(exchange_name)
        if client is None:
            logger.warning(
                "No client for exchange '%s' — cannot check order %s",
                exchange_name, order_id,
            )
            return (False, 0.0, 0.0)

        try:
            # Arbitrage clients (MEXC/Binance) use async get_order_status
            if hasattr(client, "get_order_status"):
                result = await client.get_order_status(symbol, order_id)
                is_filled = result.status.value in ("filled",)
                return (
                    is_filled,
                    float(result.filled_quantity),
                    float(result.average_fill_price or 0),
                )
            # Coinbase client uses sync get_order
            if hasattr(client, "get_order"):
                order_data = await asyncio.to_thread(client.get_order, order_id)
                if isinstance(order_data, dict):
                    status = order_data.get("status", "").lower()
                    is_filled = status in ("filled", "done", "completed")
                    filled_qty = float(order_data.get("filled_size", 0) or 0)
                    fill_price = float(
                        order_data.get("executed_value", 0) or 0
                    )
                    if filled_qty > 0 and fill_price > 0:
                        fill_price = fill_price / filled_qty
                    return (is_filled, filled_qty, fill_price)
        except Exception as exc:
            logger.warning(
                "Failed to check order %s on %s: %s", order_id, exchange_name, exc,
            )
        return (False, 0.0, 0.0)

    async def _try_cancel_order(
        self,
        exchange_name: str,
        symbol: str,
        order_id: str,
    ) -> bool:
        """Attempt to cancel an order on an exchange.  Returns True on success."""
        if not order_id:
            return False

        client = self._get_exchange_client(exchange_name)
        if client is None:
            return False

        try:
            if hasattr(client, "cancel_order"):
                # Arbitrage clients: async cancel_order(symbol, order_id)
                if asyncio.iscoroutinefunction(client.cancel_order):
                    await client.cancel_order(symbol, order_id)
                else:
                    # Coinbase client: sync cancel_order(order_id)
                    await asyncio.to_thread(client.cancel_order, order_id)
                logger.info("Cancelled order %s on %s", order_id, exchange_name)
                return True
        except Exception as exc:
            logger.warning(
                "Failed to cancel order %s on %s: %s", order_id, exchange_name, exc,
            )
        return False

    async def _emergency_close(
        self,
        trade: ActiveTrade,
        filled_side: str,
        filled_qty: float,
        result: RecoveryResult,
    ) -> None:
        """Place an emergency market sell to close a one-sided fill.

        For a buy-filled trade this sells the bought quantity.  For more complex
        arbitrage the logic is analogous but on the opposite exchange.

        If the emergency close cannot be executed (client unavailable), the
        trade is marked FAILED with a descriptive error and an alert is sent.
        """
        logger.warning(
            "Emergency close for trade %s — %s filled qty=%.8f",
            trade.trade_id, filled_side, filled_qty,
        )
        await self._alert(
            "WARNING",
            "Emergency Close",
            f"Trade {trade.trade_id} ({trade.signal_type} {trade.symbol}): "
            f"one-sided fill on {filled_side}, qty={filled_qty:.8f}. "
            f"Attempting emergency close.",
        )

        # Determine which exchange to sell on
        if filled_side == "buy":
            sell_exchange = trade.buy_exchange or trade.sell_exchange
        else:
            sell_exchange = trade.sell_exchange or trade.buy_exchange

        client = self._get_exchange_client(sell_exchange)

        if client is None:
            error_msg = (
                f"Emergency close FAILED — no client for exchange '{sell_exchange}'. "
                f"Manual intervention required!"
            )
            logger.critical(error_msg)
            await self._sm.aupdate_trade_state(
                trade.trade_id,
                TradeLifecycleState.FAILED,
                error_message=error_msg,
            )
            await self._alert("CRITICAL", "Emergency Close Failed", error_msg)
            result.errors.append(error_msg)
            return

        try:
            # Attempt a market sell
            if hasattr(client, "place_order"):
                # Arbitrage-style async client
                from arbitrage.exchanges.base import (
                    OrderRequest,
                    OrderSide,
                    OrderType as ArbOrderType,
                )
                order = OrderRequest(
                    exchange=sell_exchange,
                    symbol=trade.symbol,
                    side=OrderSide.SELL,
                    order_type=ArbOrderType.MARKET,
                    quantity=__import__("decimal").Decimal(str(filled_qty)),
                )
                order_result = await client.place_order(order)
                logger.info(
                    "Emergency sell placed: order_id=%s status=%s",
                    order_result.order_id, order_result.status.value,
                )
                await self._sm.acomplete_trade(
                    trade.trade_id,
                    actual_profit_usd=0.0,
                    error_message=f"Emergency closed via market sell (order {order_result.order_id})",
                )
            elif hasattr(client, "place_market_order"):
                # Coinbase-style sync client
                sell_result = await asyncio.to_thread(
                    client.place_market_order, "sell", trade.symbol, filled_qty,
                )
                logger.info("Emergency sell placed via Coinbase: %s", sell_result)
                await self._sm.acomplete_trade(
                    trade.trade_id,
                    actual_profit_usd=0.0,
                    error_message=f"Emergency closed via Coinbase market sell",
                )
            else:
                raise RuntimeError(
                    f"Exchange client for '{sell_exchange}' has no place_order method"
                )
            result.trades_emergency_closed += 1

        except Exception as exc:
            error_msg = (
                f"Emergency close FAILED for trade {trade.trade_id}: {exc}. "
                f"Manual intervention required!"
            )
            logger.critical(error_msg)
            await self._sm.aupdate_trade_state(
                trade.trade_id,
                TradeLifecycleState.FAILED,
                error_message=error_msg,
            )
            await self._alert("CRITICAL", "Emergency Close Failed", error_msg)
            result.errors.append(error_msg)

    # ------------------------------------------------------------------
    # Orphaned order detection
    # ------------------------------------------------------------------

    async def _check_orphaned_orders(self, result: RecoveryResult) -> int:
        """Look for open orders on each exchange that are not tracked locally.

        Returns the count of orphaned orders found.
        """
        orphaned_count = 0

        # Gather locally-tracked order IDs
        active = await self._sm.aget_active_trades()
        known_ids: set = set()
        for t in active:
            if t.buy_order_id:
                known_ids.add(t.buy_order_id)
            if t.sell_order_id:
                known_ids.add(t.sell_order_id)

        # Check each exchange
        for name, client in [
            ("mexc", self._mexc),
            ("binance", self._binance),
        ]:
            if client is None:
                continue
            try:
                open_orders = await self._get_open_orders(client)
                for oid in open_orders:
                    if oid not in known_ids:
                        orphaned_count += 1
                        msg = f"Orphaned order {oid} on {name} — not tracked locally"
                        logger.warning(msg)
                        result.warnings.append(msg)
            except Exception as exc:
                logger.warning("Could not check open orders on %s: %s", name, exc)

        if orphaned_count:
            await self._alert(
                "WARNING",
                "Orphaned Orders Detected",
                f"Found {orphaned_count} order(s) on exchanges not tracked locally.",
            )

        return orphaned_count

    async def _get_open_orders(self, client: Any) -> List[str]:
        """Retrieve open order IDs from an exchange client."""
        # ccxt-based clients typically don't have a list-all-open-orders method
        # on our wrapper, so we do a best-effort check.
        if hasattr(client, "_exchange") and client._exchange is not None:
            try:
                exchange = client._exchange
                if hasattr(exchange, "fetch_open_orders"):
                    raw = await exchange.fetch_open_orders()
                    return [o.get("id", "") for o in raw if o.get("id")]
            except Exception:
                pass
        return []

    # ------------------------------------------------------------------
    # Profit calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_profit(
        trade: ActiveTrade,
        sell_qty: float,
        sell_price: float,
    ) -> float:
        """Compute simple profit in USD for a completed trade."""
        buy_cost = trade.buy_filled_qty * trade.buy_fill_price
        sell_revenue = sell_qty * sell_price
        return sell_revenue - buy_cost

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    async def _alert(self, level: str, title: str, message: str) -> None:
        """Send an alert, falling back to logging if no alerter is configured."""
        log_fn = logger.critical if level == "CRITICAL" else logger.warning
        log_fn("ALERT [%s] %s: %s", level, title, message)

        if self._alerter is not None:
            try:
                if asyncio.iscoroutinefunction(
                    getattr(self._alerter, "send_alert", None)
                ):
                    await self._alerter.send_alert(level, title, message)
                elif hasattr(self._alerter, "send_alert"):
                    await asyncio.to_thread(
                        self._alerter.send_alert, level, title, message,
                    )
            except Exception as exc:
                logger.warning("AlertManager.send_alert failed: %s", exc)
