"""
Synchronized Cross-Exchange Executor (C1)
Fires buy and sell orders on different exchanges simultaneously using asyncio.gather().
Tracks timing gaps, spread decay, and execution quality metrics.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SyncExecutionResult:
    """Result of a synchronized cross-exchange execution."""
    leg_a_exchange: str
    leg_a_fill_price: float
    leg_a_fill_time: float
    leg_a_order_submit_time: float
    leg_b_exchange: str
    leg_b_fill_price: float
    leg_b_fill_time: float
    leg_b_order_submit_time: float
    timing_gap_ms: float
    submission_gap_ms: float
    realized_spread_bps: float
    expected_spread_bps: float
    spread_decay_bps: float
    success: bool


_DEFAULT_CONFIG = {
    "max_submission_gap_ms": 100,
    "max_fill_wait_ms": 5000,
    "retry_on_partial": True,
    "max_retries": 2,
    "pre_warm_connections": True,
    "use_websocket_orders": True,
}


class SynchronizedExecutor:
    """
    Simultaneously fires buy and sell legs across two exchanges using
    asyncio.gather() so that both order submissions happen in the same
    event-loop tick, minimising the timing gap that erodes arbitrage
    spread.
    """

    def __init__(self, config: Dict[str, Any], devil_tracker=None):
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = json.load(f)

        se_cfg = config.get("synchronized_executor", {})
        self._cfg = {**_DEFAULT_CONFIG, **se_cfg}

        self._paper_trading = config.get("trading", {}).get("paper_trading", True)
        self._devil_tracker = devil_tracker
        self._timing_stats: List[SyncExecutionResult] = []

        logger.info(
            "SynchronizedExecutor initialised  paper=%s  max_sub_gap=%dms  max_fill_wait=%dms",
            self._paper_trading,
            self._cfg["max_submission_gap_ms"],
            self._cfg["max_fill_wait_ms"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_cross_exchange(
        self,
        buy_exchange: str,
        buy_pair: str,
        buy_price: float,
        buy_quantity: float,
        sell_exchange: str,
        sell_pair: str,
        sell_price: float,
        sell_quantity: float,
    ) -> SyncExecutionResult:
        """
        Execute a cross-exchange arb: buy on one venue, sell on another,
        firing both legs as close to simultaneously as possible.
        """
        execution_id = uuid.uuid4().hex[:12]
        expected_spread_bps = ((sell_price - buy_price) / buy_price) * 10_000

        logger.info(
            "[%s] Cross-exchange exec  BUY %s@%s %.6f@%.2f  SELL %s@%s %.6f@%.2f  expected=%.1f bps",
            execution_id,
            buy_pair, buy_exchange, buy_quantity, buy_price,
            sell_pair, sell_exchange, sell_quantity, sell_price,
            expected_spread_bps,
        )

        # 1. Pre-build orders before entering the critical section
        buy_order = self._prepare_order(buy_exchange, buy_pair, "buy", buy_price, buy_quantity)
        sell_order = self._prepare_order(sell_exchange, sell_pair, "sell", sell_price, sell_quantity)

        # 2. Fire both legs simultaneously
        retries_left = self._cfg["max_retries"]
        result: Optional[SyncExecutionResult] = None

        while retries_left >= 0:
            try:
                buy_result, sell_result = await asyncio.gather(
                    self._submit_order(buy_exchange, buy_order),
                    self._submit_order(sell_exchange, sell_order),
                )

                # 3. Calculate timing metrics
                submission_gap_ms = abs(
                    buy_result["submit_ts"] - sell_result["submit_ts"]
                ) * 1000

                timing_gap_ms = abs(
                    buy_result["fill_ts"] - sell_result["fill_ts"]
                ) * 1000

                realized_spread_bps = (
                    (sell_result["fill_price"] - buy_result["fill_price"])
                    / buy_result["fill_price"]
                ) * 10_000

                spread_decay_bps = expected_spread_bps - realized_spread_bps

                success = (
                    buy_result["status"] == "filled"
                    and sell_result["status"] == "filled"
                    and submission_gap_ms <= self._cfg["max_submission_gap_ms"]
                )

                result = SyncExecutionResult(
                    leg_a_exchange=buy_exchange,
                    leg_a_fill_price=buy_result["fill_price"],
                    leg_a_fill_time=buy_result["fill_ts"],
                    leg_a_order_submit_time=buy_result["submit_ts"],
                    leg_b_exchange=sell_exchange,
                    leg_b_fill_price=sell_result["fill_price"],
                    leg_b_fill_time=sell_result["fill_ts"],
                    leg_b_order_submit_time=sell_result["submit_ts"],
                    timing_gap_ms=timing_gap_ms,
                    submission_gap_ms=submission_gap_ms,
                    realized_spread_bps=realized_spread_bps,
                    expected_spread_bps=expected_spread_bps,
                    spread_decay_bps=spread_decay_bps,
                    success=success,
                )
                break  # success path

            except Exception as exc:
                retries_left -= 1
                if retries_left < 0:
                    logger.error("[%s] All retries exhausted: %s", execution_id, exc)
                    result = SyncExecutionResult(
                        leg_a_exchange=buy_exchange,
                        leg_a_fill_price=0.0,
                        leg_a_fill_time=0.0,
                        leg_a_order_submit_time=0.0,
                        leg_b_exchange=sell_exchange,
                        leg_b_fill_price=0.0,
                        leg_b_fill_time=0.0,
                        leg_b_order_submit_time=0.0,
                        timing_gap_ms=0.0,
                        submission_gap_ms=0.0,
                        realized_spread_bps=0.0,
                        expected_spread_bps=expected_spread_bps,
                        spread_decay_bps=expected_spread_bps,
                        success=False,
                    )
                else:
                    logger.warning(
                        "[%s] Attempt failed (%s), retrying (%d left)",
                        execution_id, exc, retries_left,
                    )
                    await asyncio.sleep(0.05)

        # 4. Record stats & optional devil-tracker logging
        self._timing_stats.append(result)

        if self._devil_tracker is not None:
            try:
                self._devil_tracker.log_event(
                    "cross_exchange_execution",
                    asdict(result),
                )
            except Exception as dt_err:
                logger.debug("Devil tracker logging failed: %s", dt_err)

        log_fn = logger.info if result.success else logger.warning
        log_fn(
            "[%s] Result  success=%s  sub_gap=%.1fms  fill_gap=%.1fms  "
            "expected=%.1f bps  realised=%.1f bps  decay=%.1f bps",
            execution_id,
            result.success,
            result.submission_gap_ms,
            result.timing_gap_ms,
            result.expected_spread_bps,
            result.realized_spread_bps,
            result.spread_decay_bps,
        )

        return result

    def get_timing_report(self) -> Dict[str, Any]:
        """Aggregate timing / execution-quality statistics."""
        stats = self._timing_stats
        if not stats:
            return {
                "avg_submission_gap_ms": 0.0,
                "avg_fill_gap_ms": 0.0,
                "p95_submission_gap_ms": 0.0,
                "avg_spread_decay_bps": 0.0,
                "total_executions": 0,
                "success_rate": 0.0,
            }

        sub_gaps = [s.submission_gap_ms for s in stats]
        fill_gaps = [s.timing_gap_ms for s in stats]
        decays = [s.spread_decay_bps for s in stats]
        successes = sum(1 for s in stats if s.success)

        sub_gaps_sorted = sorted(sub_gaps)
        p95_idx = max(0, int(len(sub_gaps_sorted) * 0.95) - 1)

        return {
            "avg_submission_gap_ms": sum(sub_gaps) / len(sub_gaps),
            "avg_fill_gap_ms": sum(fill_gaps) / len(fill_gaps),
            "p95_submission_gap_ms": sub_gaps_sorted[p95_idx],
            "avg_spread_decay_bps": sum(decays) / len(decays),
            "total_executions": len(stats),
            "success_rate": successes / len(stats),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_order(
        self,
        exchange: str,
        pair: str,
        side: str,
        price: float,
        quantity: float,
    ) -> Dict[str, Any]:
        """Pre-build an order dict ready for rapid submission."""
        order = {
            "id": uuid.uuid4().hex[:16],
            "exchange": exchange,
            "pair": pair,
            "side": side,
            "price": price,
            "quantity": quantity,
            "type": "limit",
            "time_in_force": "IOC",  # immediate-or-cancel for arb legs
            "use_websocket": self._cfg["use_websocket_orders"],
            "created_at": time.time(),
        }
        logger.debug("Prepared order: %s", order)
        return order

    async def _submit_order(
        self, exchange: str, order: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Submit an order to the given exchange.

        In paper-trading mode this simulates a fill at the requested price
        with a small random latency.  In live mode the caller would replace
        this with real exchange adapter calls.
        """
        submit_ts = time.time()

        if self._paper_trading:
            # Simulate network + matching-engine latency (1-15 ms)
            import random
            sim_latency = random.uniform(0.001, 0.015)
            await asyncio.sleep(sim_latency)

            fill_ts = time.time()
            # Small simulated slippage: up to 0.5 bps
            slippage_factor = 1.0 + random.uniform(-0.00005, 0.00005)
            if order["side"] == "buy":
                fill_price = order["price"] * (1.0 + abs(slippage_factor - 1.0))
            else:
                fill_price = order["price"] * (1.0 - abs(slippage_factor - 1.0))

            logger.debug(
                "Paper fill  %s %s %s  req=%.2f  fill=%.2f  latency=%.1fms",
                exchange, order["pair"], order["side"],
                order["price"], fill_price, sim_latency * 1000,
            )

            return {
                "order_id": order["id"],
                "exchange": exchange,
                "status": "filled",
                "fill_price": fill_price,
                "fill_quantity": order["quantity"],
                "submit_ts": submit_ts,
                "fill_ts": fill_ts,
            }

        # --- Live order path (placeholder for real exchange adapters) ---
        max_wait = self._cfg["max_fill_wait_ms"] / 1000.0
        logger.info(
            "LIVE order submission  %s %s %s@%.2f qty=%.6f  (max_wait=%.1fs)",
            exchange, order["pair"], order["side"],
            order["price"], order["quantity"], max_wait,
        )

        # In a production system this would dispatch to the appropriate
        # exchange client (Coinbase, Kraken, Binance, etc.) via an adapter
        # registry.  For now raise so callers know live mode needs wiring.
        raise NotImplementedError(
            f"Live order submission for {exchange} is not yet wired. "
            "Implement exchange adapters and register them in SynchronizedExecutor."
        )
