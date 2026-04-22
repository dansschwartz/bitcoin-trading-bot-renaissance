"""
Capital allocation between arbitrage modules.

Prevents any single module from starving the others.
All modules share the same MEXC account, so this is a
software-level allocation, not exchange-level.

Default split: 40% triangular arb, 50% market maker, 10% reserve.
"""

import logging
import time
from typing import Callable, Awaitable, Dict, Optional

logger = logging.getLogger("arb.capital")


class CapitalAllocator:
    """
    Manages capital budgets across arb modules.

    Each module queries its available budget before taking action.
    The allocator reads live balances and computes remaining budget
    based on allocation percentages.
    """

    # Allocation percentages (must sum to <= 1.0)
    ALLOCATIONS = {
        "triangular": 0.40,
        "market_maker": 0.50,
        "reserve": 0.10,
    }

    # Minimum balance that must always remain free (emergency reserve)
    ABSOLUTE_MIN_FREE_USD = 50.0

    # Cache TTL for balance lookups (seconds)
    BALANCE_CACHE_TTL = 60.0

    def __init__(self, balance_getter: Callable[[], Awaitable[Dict[str, float]]]):
        """
        Args:
            balance_getter: async callable that returns
                {"USDT": float, "USDC": float, ...} of free balances
        """
        self._get_balances = balance_getter
        self._deployed: Dict[str, float] = {
            "triangular": 0.0,
            "market_maker": 0.0,
        }
        self._cached_balances: Dict[str, float] = {}
        self._cache_ts: float = 0.0

    async def _fetch_balances(self) -> Dict[str, float]:
        """Fetch balances with caching."""
        now = time.monotonic()
        if (now - self._cache_ts) < self.BALANCE_CACHE_TTL and self._cached_balances:
            return self._cached_balances
        try:
            balances = await self._get_balances()
            self._cached_balances = balances
            self._cache_ts = now
            return balances
        except Exception as e:
            logger.warning(f"CAPITAL: balance fetch failed: {e}")
            return self._cached_balances

    async def get_total_usd(self) -> float:
        """Total stablecoin balance (USDT + USDC)."""
        balances = await self._fetch_balances()
        return sum(balances.get(c, 0) for c in ("USDT", "USDC"))

    async def get_available_budget(self, module: str) -> Dict[str, float]:
        """
        How much capital a module can currently use.

        Returns dict of {currency: available_usd} for each stablecoin.
        """
        balances = await self._fetch_balances()

        total_usd = sum(balances.get(c, 0) for c in ("USDT", "USDC"))

        if total_usd <= self.ABSOLUTE_MIN_FREE_USD:
            return {"USDT": 0.0, "USDC": 0.0}

        allocation_pct = self.ALLOCATIONS.get(module, 0)
        budget_usd = total_usd * allocation_pct

        # Return per-currency availability capped by budget
        result = {}
        for currency in ("USDT", "USDC"):
            free = balances.get(currency, 0)
            # Module can use up to its budget, capped by actual free balance
            result[currency] = min(free, budget_usd)

        return result

    def record_deployment(self, module: str, amount_usd: float):
        """Record that a module deployed capital."""
        self._deployed[module] = self._deployed.get(module, 0) + amount_usd

    def record_return(self, module: str, amount_usd: float):
        """Record that a module returned capital."""
        self._deployed[module] = max(0, self._deployed.get(module, 0) - amount_usd)

    def get_summary(self) -> Dict:
        """Dashboard summary of capital allocation."""
        return {
            "allocations": dict(self.ALLOCATIONS),
            "deployed": dict(self._deployed),
            "absolute_min_free": self.ABSOLUTE_MIN_FREE_USD,
        }

    async def log_status(self):
        """Log current capital allocation state."""
        try:
            balances = await self._fetch_balances()
            total = sum(balances.get(c, 0) for c in ("USDT", "USDC"))

            tri_budget = await self.get_available_budget("triangular")
            mm_budget = await self.get_available_budget("market_maker")

            logger.info(
                f"CAPITAL: total=${total:.0f} "
                f"(USDT=${balances.get('USDT', 0):.0f} USDC=${balances.get('USDC', 0):.0f}) | "
                f"tri_budget=USDT${tri_budget.get('USDT', 0):.0f}+USDC${tri_budget.get('USDC', 0):.0f} | "
                f"mm_budget=USDC${mm_budget.get('USDC', 0):.0f}"
            )
        except Exception as e:
            logger.warning(f"CAPITAL: status log failed: {e}")
