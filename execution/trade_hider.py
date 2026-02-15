"""
Trade Hiding Module (C2)
Adds randomised timing jitter and execution variation to prevent pattern
detection by other market participants or exchange surveillance.
"""

import asyncio
import copy
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_DEFAULT_CONFIG = {
    "enabled": True,
    "max_jitter_seconds": 3.0,
    "size_variance_pct": 10.0,
    "order_type_weights": {"limit": 0.7, "market": 0.3},
    "chunk_orders_above_usd": 500,
    "min_chunks": 2,
    "max_chunks": 5,
}


class TradeHider:
    """
    Applies randomised modifications to outgoing trade signals so that
    execution footprints are harder for adversaries to fingerprint:

    * **Timing jitter** -- random delay before each order.
    * **Size variance** -- slight random adjustment to quantity.
    * **Order-type randomisation** -- probabilistic selection between
      limit and market orders.
    * **Order splitting** -- large orders are broken into smaller chunks
      whose sizes are randomly distributed.
    """

    def __init__(self, config: Dict[str, Any]):
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = json.load(f)

        th_cfg = config.get("trade_hider", {})
        self._cfg: Dict[str, Any] = {**_DEFAULT_CONFIG, **th_cfg}

        self._enabled: bool = self._cfg["enabled"]
        self._max_jitter: float = self._cfg["max_jitter_seconds"]
        self._size_var_pct: float = self._cfg["size_variance_pct"]
        self._order_weights: Dict[str, float] = self._cfg["order_type_weights"]
        self._chunk_threshold: float = self._cfg["chunk_orders_above_usd"]
        self._min_chunks: int = self._cfg["min_chunks"]
        self._max_chunks: int = self._cfg["max_chunks"]

        logger.info(
            "TradeHider initialised  enabled=%s  max_jitter=%.1fs  "
            "size_var=%.1f%%  chunk_above=$%.0f",
            self._enabled,
            self._max_jitter,
            self._size_var_pct,
            self._chunk_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def apply_jitter(self, signal_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply randomised jitter to *signal_dict* in-place (a shallow copy
        is made so the caller's original is not mutated).

        Modifications applied:
        1. Random delay of 0 .. max_jitter_seconds.
        2. Quantity adjusted by +/- size_variance_pct.
        3. Order type probabilistically chosen according to weights.
        4. ``jitter_delay_ms`` field recorded in the returned dict.

        Returns the modified signal dict.
        """
        sig = copy.deepcopy(signal_dict)

        if not self._enabled:
            sig["jitter_delay_ms"] = 0.0
            logger.debug("TradeHider disabled, passing signal through unchanged")
            return sig

        # 1. Random timing delay
        jitter_s = random.uniform(0, self._max_jitter)
        sig["jitter_delay_ms"] = round(jitter_s * 1000, 2)

        logger.debug("Applying jitter delay of %.1f ms", sig["jitter_delay_ms"])
        await asyncio.sleep(jitter_s)

        # 2. Size variance
        if "quantity" in sig and sig["quantity"] is not None:
            original_qty = sig["quantity"]
            variance = random.uniform(
                -self._size_var_pct / 100.0,
                self._size_var_pct / 100.0,
            )
            sig["quantity"] = round(original_qty * (1.0 + variance), 8)
            logger.debug(
                "Size variance applied: %.8f -> %.8f (%.2f%%)",
                original_qty, sig["quantity"], variance * 100,
            )

        # 3. Random order type selection
        types = list(self._order_weights.keys())
        weights = list(self._order_weights.values())
        chosen_type = random.choices(types, weights=weights, k=1)[0]
        sig["order_type"] = chosen_type
        logger.debug("Order type selected: %s", chosen_type)

        return sig

    def split_order(
        self,
        total_quantity: float,
        min_chunks: Optional[int] = None,
        max_chunks: Optional[int] = None,
    ) -> List[float]:
        """
        Split *total_quantity* into a random number of chunks whose sizes
        are randomly distributed but **always sum exactly to
        total_quantity**.

        Parameters
        ----------
        total_quantity : float
            The full order size to split.
        min_chunks : int, optional
            Minimum number of child orders (defaults to config value).
        max_chunks : int, optional
            Maximum number of child orders (defaults to config value).

        Returns
        -------
        list[float]
            A list of quantities that sum to *total_quantity*.
        """
        lo = min_chunks if min_chunks is not None else self._min_chunks
        hi = max_chunks if max_chunks is not None else self._max_chunks

        # Clamp to sensible values
        lo = max(1, lo)
        hi = max(lo, hi)

        n_chunks = random.randint(lo, hi)

        if n_chunks == 1:
            return [total_quantity]

        # Generate n_chunks - 1 random cut-points in (0, total_quantity),
        # then derive the chunk sizes from the gaps between them.
        cuts = sorted(random.uniform(0, total_quantity) for _ in range(n_chunks - 1))

        chunks: List[float] = []
        prev = 0.0
        for cut in cuts:
            chunks.append(round(cut - prev, 8))
            prev = cut
        chunks.append(round(total_quantity - prev, 8))

        # Correct any floating-point drift so the sum is exact
        drift = total_quantity - sum(chunks)
        if drift != 0.0:
            chunks[-1] = round(chunks[-1] + drift, 8)

        logger.debug(
            "Order split: total=%.8f -> %d chunks %s",
            total_quantity, n_chunks, chunks,
        )
        return chunks

    def should_split(self, quantity_usd: float) -> bool:
        """
        Determine whether an order of the given USD value should be split
        into smaller chunks based on the configured threshold.
        """
        do_split = quantity_usd >= self._chunk_threshold
        logger.debug(
            "should_split(%.2f USD) -> %s  (threshold=%.2f)",
            quantity_usd, do_split, self._chunk_threshold,
        )
        return do_split
