"""
Active Pair Set Manager — maintains the live set of pairs being scanned,
handles tier-based resource allocation, and rotates pairs based on
ongoing performance.

Resource allocation by tier:
- Tier 1 (10 pairs): WebSocket depth + trades + REST order book refresh
- Tier 2 (20 pairs): WebSocket depth only (no trade stream)
- Tier 3 (70 pairs): REST ticker only (used for triangular scanning)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

from arbitrage.pairs.discovery import MexcPairDiscovery, PairScore

logger = logging.getLogger("arb.pairs")


class ExpandedPairManager:
    """
    Manages the active pair set lifecycle for triangular arb expansion.

    Lifecycle:
    1. On startup: MexcPairDiscovery.scan() → initial tier assignments
    2. Every 6 hours: MexcPairDiscovery.maybe_rescan() → update tiers
    3. Every 24 hours: performance-based rotation (promote/demote pairs)
    """

    ROTATION_INTERVAL_HOURS = 24
    MIN_TRADES_FOR_DEMOTION = 20

    def __init__(
        self,
        discovery: MexcPairDiscovery,
        orchestrator=None,
    ):
        self.discovery = discovery
        self.orchestrator = orchestrator

        self._tier_1: List[str] = []
        self._tier_2: List[str] = []
        self._tier_3: List[str] = []
        self._locked_pairs: Set[str] = set()
        self._last_rotation: Optional[datetime] = None
        self._initialized = False

    async def initialize(self, locked_pairs: Optional[List[str]] = None):
        """
        Initialize pair sets from discovery scan.

        locked_pairs: pairs that must always remain in Tier 1 (e.g., the
        original pairs from config). These are never demoted.
        """
        if locked_pairs:
            self._locked_pairs = set(locked_pairs)

        try:
            scores = await self.discovery.scan()
        except Exception as e:
            logger.error("Pair discovery scan failed: %s", e)
            scores = []

        if not scores:
            logger.warning("No pairs discovered, using locked pairs only")
            self._tier_1 = list(self._locked_pairs)
            self._initialized = True
            return

        self._tier_1 = self.discovery.get_tier_pairs(1)
        self._tier_2 = self.discovery.get_tier_pairs(2)
        self._tier_3 = self.discovery.get_tier_pairs(3)

        # Ensure locked pairs are always in Tier 1
        for pair in self._locked_pairs:
            if pair not in self._tier_1:
                self._tier_1.append(pair)

        self._initialized = True
        self._last_rotation = datetime.utcnow()

        logger.info(
            "ExpandedPairManager initialized — Tier 1: %d, Tier 2: %d, Tier 3: %d, Locked: %d",
            len(self._tier_1), len(self._tier_2),
            len(self._tier_3), len(self._locked_pairs),
        )

    async def maybe_update(self):
        """Check if pair sets need updating (rescan or rotation)."""
        if not self._initialized:
            return

        new_scores = await self.discovery.maybe_rescan()
        if new_scores:
            await self._apply_new_scores()

        if self._last_rotation:
            elapsed = (datetime.utcnow() - self._last_rotation).total_seconds() / 3600
            if elapsed >= self.ROTATION_INTERVAL_HOURS:
                await self._rotate_pairs()

    async def _apply_new_scores(self):
        """Apply newly discovered scores while preserving locked pairs."""
        new_tier_1 = self.discovery.get_tier_pairs(1)
        new_tier_2 = self.discovery.get_tier_pairs(2)
        new_tier_3 = self.discovery.get_tier_pairs(3)

        for pair in self._locked_pairs:
            if pair not in new_tier_1:
                new_tier_1.append(pair)

        added_t1 = set(new_tier_1) - set(self._tier_1)
        removed_t1 = set(self._tier_1) - set(new_tier_1) - self._locked_pairs

        if added_t1:
            logger.info("Tier 1 additions: %s", added_t1)
        if removed_t1:
            logger.info("Tier 1 removals: %s", removed_t1)

        self._tier_1 = new_tier_1
        self._tier_2 = new_tier_2
        self._tier_3 = new_tier_3

    async def _rotate_pairs(self):
        """Performance-based pair rotation."""
        logger.info("Running pair rotation...")
        self._last_rotation = datetime.utcnow()

        if not self.orchestrator or not hasattr(self.orchestrator, 'tracker'):
            return

        try:
            summary = self.orchestrator.tracker.get_summary()
        except Exception:
            return

        pair_stats = summary.get('top_pairs', {})

        # Find underperforming Tier 1 pairs (not locked)
        demote_candidates = []
        for pair in self._tier_1:
            if pair in self._locked_pairs:
                continue
            stats = pair_stats.get(pair, {})
            if stats.get('trades', 0) >= self.MIN_TRADES_FOR_DEMOTION:
                if stats.get('profit_usd', 0) < 0:
                    demote_candidates.append((pair, stats.get('profit_usd', 0)))

        # Find promising Tier 2 pairs for promotion
        promote_candidates = []
        for pair in self._tier_2:
            stats = pair_stats.get(pair, {})
            if stats.get('trades', 0) >= 10 and stats.get('profit_usd', 0) > 0:
                promote_candidates.append((pair, stats.get('profit_usd', 0)))

        promote_candidates.sort(key=lambda x: x[1], reverse=True)

        # Rotate: demote worst Tier 1, promote best Tier 2
        rotations = min(len(demote_candidates), len(promote_candidates), 3)
        for i in range(rotations):
            demoted = demote_candidates[i][0]
            promoted = promote_candidates[i][0]

            self._tier_1.remove(demoted)
            self._tier_2.append(demoted)

            self._tier_2.remove(promoted)
            self._tier_1.append(promoted)

            logger.info(
                "Pair rotation: demoted %s (T1→T2, P&L $%.2f), promoted %s (T2→T1, P&L $%.2f)",
                demoted, demote_candidates[i][1],
                promoted, promote_candidates[i][1],
            )

    @property
    def tier_1_pairs(self) -> List[str]:
        return list(self._tier_1)

    @property
    def tier_2_pairs(self) -> List[str]:
        return list(self._tier_2)

    @property
    def tier_3_pairs(self) -> List[str]:
        return list(self._tier_3)

    @property
    def all_active_pairs(self) -> List[str]:
        return self._tier_1 + self._tier_2 + self._tier_3

    @property
    def triangular_pairs(self) -> List[str]:
        """All pairs eligible for triangular scanning (all tiers)."""
        return self.all_active_pairs

    @property
    def cross_exchange_pairs(self) -> List[str]:
        """Pairs eligible for cross-exchange arb (Tier 1+2, on Binance)."""
        return self.discovery.get_cross_exchange_eligible()

    def get_report(self) -> dict:
        return {
            "initialized": self._initialized,
            "tier_1": self._tier_1,
            "tier_1_count": len(self._tier_1),
            "tier_2": self._tier_2,
            "tier_2_count": len(self._tier_2),
            "tier_3_count": len(self._tier_3),
            "locked_pairs": list(self._locked_pairs),
            "last_rotation": self._last_rotation.isoformat() if self._last_rotation else None,
            "total_active": len(self.all_active_pairs),
        }
