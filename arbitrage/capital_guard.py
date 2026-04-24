"""
Capital Guard — hard USDT reserve enforcement across all arbitrage subsystems.

Prevents the arb bot from spending all USDT on random token purchases.
Every subsystem must call `can_spend()` before placing any buy order.
"""
import logging
import time
from typing import Tuple

logger = logging.getLogger("arb.capital_guard")


class CapitalGuard:
    """Enforces minimum USDT reserve before any trade.

    Args:
        min_usdt_reserve: Minimum free USDT that must remain after the trade.
        config: Optional dict with 'risk.min_usdt_reserve' override.
    """

    def __init__(self, min_usdt_reserve: float = 300.0, config: dict | None = None):
        if config:
            self.min_usdt_reserve = float(
                config.get("risk", {}).get("min_usdt_reserve", min_usdt_reserve)
            )
        else:
            self.min_usdt_reserve = min_usdt_reserve

        self._blocked_count = 0
        self._allowed_count = 0
        self._last_balance: float = 0.0
        self._last_check_time: float = 0.0

        logger.info(f"CapitalGuard: min_usdt_reserve=${self.min_usdt_reserve:.2f}")

    async def can_spend(self, exchange_client, amount_usd: float) -> Tuple[bool, float]:
        """Check if spending `amount_usd` would violate the USDT reserve.

        Args:
            exchange_client: Exchange client with `get_balance('USDT')` or
                             `get_balances()` method.
            amount_usd: USD amount the caller wants to spend.

        Returns:
            (allowed, current_balance): Whether the trade is allowed,
            and the current free USDT balance.
        """
        try:
            usdt_free = await self._get_usdt_balance(exchange_client)
        except Exception as e:
            logger.warning(f"CapitalGuard: balance fetch failed: {e} — blocking trade")
            self._blocked_count += 1
            return False, 0.0

        self._last_balance = usdt_free
        self._last_check_time = time.time()

        remaining = usdt_free - amount_usd
        if remaining < self.min_usdt_reserve:
            self._blocked_count += 1
            logger.warning(
                f"CAPITAL GUARD BLOCKED: want to spend ${amount_usd:.2f} "
                f"but USDT_free=${usdt_free:.2f}, "
                f"remaining=${remaining:.2f} < reserve=${self.min_usdt_reserve:.2f}"
            )
            return False, usdt_free

        self._allowed_count += 1
        return True, usdt_free

    async def _get_usdt_balance(self, exchange_client) -> float:
        """Extract free USDT balance from an exchange client."""
        # Try get_balance('USDT') first (ArbitrageExecutor style)
        if hasattr(exchange_client, "get_balance"):
            bal = await exchange_client.get_balance("USDT")
            if bal is not None:
                free = getattr(bal, "free", None)
                if free is not None:
                    return float(free)
                # dict-style balance
                if isinstance(bal, dict):
                    return float(bal.get("free", 0))

        # Fallback: get_balances() (SpotRebalancer style)
        if hasattr(exchange_client, "get_balances"):
            balances = await exchange_client.get_balances()
            if balances:
                usdt = balances.get("USDT")
                if usdt is not None:
                    free = getattr(usdt, "free", None)
                    if free is not None:
                        return float(free)
                    if isinstance(usdt, dict):
                        return float(usdt.get("free", 0))

        logger.warning("CapitalGuard: could not read USDT balance from client")
        return 0.0

    def get_stats(self) -> dict:
        """Return stats for dashboard/monitoring."""
        return {
            "min_usdt_reserve": self.min_usdt_reserve,
            "blocked_count": self._blocked_count,
            "allowed_count": self._allowed_count,
            "last_balance": round(self._last_balance, 2),
            "last_check_time": self._last_check_time,
        }
