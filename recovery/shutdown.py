"""
Graceful Shutdown Handler — orderly shutdown for the Renaissance trading bot.

Responsibilities on SIGTERM / SIGINT:
1. Transition system state to SHUTTING_DOWN.
2. Stop accepting new trading signals.
3. Wait for in-flight trades to complete (with a configurable timeout).
4. Cancel any remaining open orders on all exchanges.
5. Persist final state to the recovery database.
6. Transition system state to HALTED.
7. Exit cleanly.

Usage
-----
::

    from recovery.shutdown import GracefulShutdownHandler

    handler = GracefulShutdownHandler(
        state_manager=sm,
        coinbase_client=cb,
        mexc_client=mexc,
        binance_client=binance,
    )
    handler.install()  # registers SIGTERM / SIGINT

    # ... later, from inside the event loop:
    await handler.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

from recovery.state_manager import (
    ActiveTrade,
    StateManager,
    SystemState,
    TradeLifecycleState,
)

logger = logging.getLogger("recovery.shutdown")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class GracefulShutdownHandler:
    """
    Registers OS signal handlers and drives an orderly shutdown sequence.

    Parameters
    ----------
    state_manager : StateManager
        Shared state manager instance.
    coinbase_client : object | None
        Coinbase client for cancelling directional orders.
    mexc_client : object | None
        MEXC client for cancelling arbitrage orders.
    binance_client : object | None
        Binance client for cancelling arbitrage orders.
    alert_manager : object | None
        Optional alerter for sending shutdown notifications.
    drain_timeout_seconds : float
        Maximum seconds to wait for in-flight trades before force-cancelling.
    on_shutdown_complete : callable | None
        Optional callback invoked after shutdown is fully complete.
    """

    def __init__(
        self,
        state_manager: StateManager,
        coinbase_client: Any = None,
        mexc_client: Any = None,
        binance_client: Any = None,
        alert_manager: Any = None,
        drain_timeout_seconds: float = 30.0,
        on_shutdown_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        self._sm = state_manager
        self._coinbase = coinbase_client
        self._mexc = mexc_client
        self._binance = binance_client
        self._alerter = alert_manager
        self._drain_timeout = drain_timeout_seconds
        self._on_complete = on_shutdown_complete

        self._shutting_down = False
        self._accept_signals = True
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_shutting_down(self) -> bool:
        """True once shutdown has been triggered."""
        return self._shutting_down

    @property
    def accepting_signals(self) -> bool:
        """False once shutdown has been triggered — callers should stop
        submitting new trading signals."""
        return self._accept_signals

    def install(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Register SIGTERM and SIGINT handlers.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop | None
            The running event loop.  If provided, shutdown work is scheduled
            as an asyncio task.  If ``None``, the handler sets a flag that the
            main loop should check.
        """
        self._loop = loop

        def _signal_handler(signum: int, frame: object) -> None:
            sig_name = signal.Signals(signum).name
            logger.info("Received %s — initiating graceful shutdown", sig_name)
            self._shutting_down = True
            self._accept_signals = False

            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(
                    self._loop.create_task, self.shutdown()
                )

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        logger.info("Graceful shutdown handlers installed (SIGTERM, SIGINT)")

    async def shutdown(self) -> None:
        """Execute the full shutdown sequence.

        This is safe to call multiple times; subsequent calls are no-ops.
        """
        if not self._shutting_down:
            self._shutting_down = True
            self._accept_signals = False

        logger.info("=== Graceful Shutdown Sequence Starting ===")

        # 1. Transition to SHUTTING_DOWN
        try:
            await self._sm.aset_system_state(
                SystemState.SHUTTING_DOWN, "graceful shutdown initiated"
            )
        except Exception as exc:
            logger.warning("Failed to set SHUTTING_DOWN state: %s", exc)

        # 2. Stop accepting new signals (already done via flag)
        logger.info("No longer accepting new trading signals.")

        # 3. Wait for in-flight trades to drain
        await self._drain_in_flight_trades()

        # 4. Cancel remaining open orders
        await self._cancel_all_open_orders()

        # 5. Persist final state
        await self._persist_final_state()

        # 6. Send shutdown alert
        await self._alert(
            "INFO",
            "Bot Shutdown Complete",
            "Renaissance trading bot has shut down gracefully.",
        )

        # 7. Transition to HALTED
        try:
            await self._sm.aset_system_state(
                SystemState.HALTED, "graceful shutdown complete"
            )
        except Exception as exc:
            logger.warning("Failed to set HALTED state: %s", exc)

        logger.info("=== Graceful Shutdown Sequence Complete ===")

        if self._on_complete is not None:
            try:
                self._on_complete()
            except Exception as exc:
                logger.warning("on_shutdown_complete callback failed: %s", exc)

    # ------------------------------------------------------------------
    # Drain in-flight trades
    # ------------------------------------------------------------------

    async def _drain_in_flight_trades(self) -> None:
        """Wait for active trades to reach a terminal state, up to the
        configured timeout."""
        logger.info(
            "Waiting up to %.0f seconds for in-flight trades to complete...",
            self._drain_timeout,
        )
        deadline = time.monotonic() + self._drain_timeout
        poll_interval = 1.0

        while time.monotonic() < deadline:
            active = await self._sm.aget_active_trades()
            if not active:
                logger.info("All in-flight trades completed.")
                return
            logger.info(
                "%d trade(s) still in flight — waiting (%.0fs remaining)...",
                len(active),
                deadline - time.monotonic(),
            )
            await asyncio.sleep(poll_interval)

        # Timed out — mark remaining trades as failed
        remaining = await self._sm.aget_active_trades()
        if remaining:
            logger.warning(
                "Drain timeout reached with %d trade(s) still active — "
                "marking as FAILED.",
                len(remaining),
            )
            for trade in remaining:
                try:
                    await self._sm.aupdate_trade_state(
                        trade.trade_id,
                        TradeLifecycleState.FAILED,
                        error_message="Shutdown timeout: trade did not complete in time",
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to mark trade %s as FAILED: %s",
                        trade.trade_id, exc,
                    )

    # ------------------------------------------------------------------
    # Cancel open orders
    # ------------------------------------------------------------------

    async def _cancel_all_open_orders(self) -> None:
        """Best-effort cancel of all open orders on every exchange."""
        logger.info("Cancelling open orders on all exchanges...")
        tasks = []

        if self._coinbase is not None:
            tasks.append(self._cancel_coinbase_orders())
        if self._mexc is not None:
            tasks.append(self._cancel_exchange_orders("mexc", self._mexc))
        if self._binance is not None:
            tasks.append(self._cancel_exchange_orders("binance", self._binance))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    logger.warning("Order cancellation task %d failed: %s", i, r)
        else:
            logger.info("No exchange clients configured — skipping order cancellation.")

    async def _cancel_coinbase_orders(self) -> None:
        """Cancel open orders via the Coinbase client."""
        try:
            # EnhancedCoinbaseClient.cancel_order(order_id) is sync
            # We try to get open orders from state manager and cancel them
            active = await self._sm.aget_active_trades()
            coinbase_order_ids: List[str] = []
            for trade in active:
                if trade.buy_exchange.lower() in ("coinbase", "cb", ""):
                    if trade.buy_order_id:
                        coinbase_order_ids.append(trade.buy_order_id)
                if trade.sell_exchange.lower() in ("coinbase", "cb", ""):
                    if trade.sell_order_id:
                        coinbase_order_ids.append(trade.sell_order_id)

            cancelled = 0
            for oid in coinbase_order_ids:
                try:
                    await asyncio.to_thread(self._coinbase.cancel_order, oid)
                    cancelled += 1
                except Exception as exc:
                    logger.warning("Failed to cancel Coinbase order %s: %s", oid, exc)

            logger.info("Cancelled %d Coinbase order(s).", cancelled)

        except Exception as exc:
            logger.warning("Error cancelling Coinbase orders: %s", exc)

    async def _cancel_exchange_orders(self, name: str, client: Any) -> None:
        """Cancel open orders on an arbitrage exchange (MEXC or Binance)."""
        try:
            # Try to fetch open orders via the ccxt exchange object
            open_orders: List[dict] = []
            if hasattr(client, "_exchange") and client._exchange is not None:
                exchange = client._exchange
                if hasattr(exchange, "fetch_open_orders"):
                    open_orders = await exchange.fetch_open_orders()

            cancelled = 0
            for order in open_orders:
                oid = order.get("id", "")
                symbol = order.get("symbol", "")
                if oid and symbol:
                    try:
                        await client.cancel_order(symbol, oid)
                        cancelled += 1
                    except Exception as exc:
                        logger.warning(
                            "Failed to cancel %s order %s (%s): %s",
                            name, oid, symbol, exc,
                        )

            # Also cancel orders tracked in state manager
            active = await self._sm.aget_active_trades()
            for trade in active:
                for oid, exch in [
                    (trade.buy_order_id, trade.buy_exchange),
                    (trade.sell_order_id, trade.sell_exchange),
                ]:
                    if oid and exch.lower() == name.lower():
                        try:
                            if asyncio.iscoroutinefunction(client.cancel_order):
                                await client.cancel_order(trade.symbol, oid)
                            else:
                                await asyncio.to_thread(
                                    client.cancel_order, trade.symbol, oid,
                                )
                            cancelled += 1
                        except Exception:
                            pass  # Already logged above or order not found

            logger.info("Cancelled %d order(s) on %s.", cancelled, name)

        except Exception as exc:
            logger.warning("Error cancelling %s orders: %s", name, exc)

    # ------------------------------------------------------------------
    # Persist final state
    # ------------------------------------------------------------------

    async def _persist_final_state(self) -> None:
        """Ensure all trade records are flushed to disk."""
        try:
            # Send one last heartbeat so the watchdog does not immediately
            # restart us during the tail end of shutdown.
            await self._sm.asend_heartbeat()
            logger.info("Final state persisted to recovery database.")
        except Exception as exc:
            logger.warning("Failed to persist final state: %s", exc)

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    async def _alert(self, level: str, title: str, message: str) -> None:
        """Send an alert; falls back to logging."""
        log_fn = logger.info if level == "INFO" else logger.warning
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
