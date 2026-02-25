"""
Concentration Limiter — prevents over-trading any single pair and
auto-blocks tokens with high one-sided fill rates.

Cross-exchange arb without concentration limits leads to:
1. VANRY spam: same pair traded 20x/minute, ~$4 "profit" each, all phantom
2. Inventory imbalance: one token piles up on one exchange
3. One-sided fills: some pairs consistently fail on one leg

This module enforces:
- Max N trades per pair per hour (default 5)
- Cooldown period after each fill (default 60s)
- Auto-block tokens with >20% one-sided fill rate (after 10+ samples)
"""
import logging
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger("arb.concentration")


class ConcentrationLimiter:
    """Per-pair concentration limits for cross-exchange arbitrage.

    Tracks trade frequency, cooldowns, and one-sided fill rates per symbol.
    Called by the risk engine before approving a signal.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = (config or {}).get('concentration', {})

        # Per-pair trade rate limit
        self.max_trades_per_pair_per_hour = int(cfg.get('max_trades_per_pair_per_hour', 5))

        # Cooldown after a fill (seconds)
        self.cooldown_after_fill_sec = int(cfg.get('cooldown_after_fill_sec', 60))

        # One-sided fill auto-blocking
        self.one_sided_block_threshold = float(cfg.get('one_sided_block_threshold', 0.20))
        self.one_sided_min_samples = int(cfg.get('one_sided_min_samples', 10))

        # State tracking
        self._trade_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._last_fill_time: Dict[str, float] = {}  # symbol -> epoch
        self._fill_count: Dict[str, int] = defaultdict(int)
        self._one_sided_count: Dict[str, int] = defaultdict(int)
        self._blocked_pairs: Set[str] = set()
        self._block_reasons: Dict[str, str] = {}

        # Stats
        self._rejections_rate_limit = 0
        self._rejections_cooldown = 0
        self._rejections_blocked = 0

        logger.info(
            f"ConcentrationLimiter: max {self.max_trades_per_pair_per_hour}/pair/hr, "
            f"{self.cooldown_after_fill_sec}s cooldown, "
            f"auto-block at {self.one_sided_block_threshold*100:.0f}% one-sided rate"
        )

    def check(self, symbol: str) -> Tuple[bool, str]:
        """Check if a trade on this symbol is allowed.

        Returns:
            (allowed, reason) — True if allowed, False + reason if blocked.
        """
        now = time.time()

        # 1. Check if pair is auto-blocked
        if symbol in self._blocked_pairs:
            self._rejections_blocked += 1
            return False, f"blocked: {self._block_reasons.get(symbol, 'one-sided fills')}"

        # 2. Cooldown check — must wait N seconds after last fill
        last_fill = self._last_fill_time.get(symbol, 0)
        elapsed = now - last_fill
        if elapsed < self.cooldown_after_fill_sec:
            remaining = self.cooldown_after_fill_sec - elapsed
            self._rejections_cooldown += 1
            return False, f"cooldown: {remaining:.0f}s remaining"

        # 3. Per-pair rate limit — max N trades per hour
        hour_ago = now - 3600
        pair_times = self._trade_times[symbol]
        recent = sum(1 for t in pair_times if t > hour_ago)
        if recent >= self.max_trades_per_pair_per_hour:
            self._rejections_rate_limit += 1
            return False, f"rate limit: {recent}/{self.max_trades_per_pair_per_hour} per hour"

        return True, "ok"

    def record_trade(self, symbol: str, status: str) -> None:
        """Record a trade outcome for concentration tracking.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            status: Trade outcome — "filled", "one_sided_buy", "one_sided_sell", "no_fill", etc.
        """
        now = time.time()
        self._trade_times[symbol].append(now)

        if status == "filled":
            self._last_fill_time[symbol] = now
            self._fill_count[symbol] += 1

        elif "one_sided" in status:
            self._one_sided_count[symbol] += 1
            self._last_fill_time[symbol] = now  # Cooldown applies to one-sided too

            # Check if one-sided rate exceeds threshold → auto-block
            total = self._fill_count[symbol] + self._one_sided_count[symbol]
            if total >= self.one_sided_min_samples:
                one_sided_rate = self._one_sided_count[symbol] / total
                if one_sided_rate > self.one_sided_block_threshold:
                    self._blocked_pairs.add(symbol)
                    self._block_reasons[symbol] = (
                        f"one-sided rate {one_sided_rate:.0%} "
                        f"({self._one_sided_count[symbol]}/{total} trades)"
                    )
                    logger.warning(
                        f"CONCENTRATION: Auto-blocked {symbol} — "
                        f"{self._block_reasons[symbol]}"
                    )

    def unblock(self, symbol: str) -> bool:
        """Manually unblock a pair. Returns True if it was blocked."""
        if symbol in self._blocked_pairs:
            self._blocked_pairs.discard(symbol)
            self._block_reasons.pop(symbol, None)
            # Reset one-sided counters
            self._one_sided_count[symbol] = 0
            self._fill_count[symbol] = 0
            logger.info(f"CONCENTRATION: Manually unblocked {symbol}")
            return True
        return False

    def get_pair_status(self, symbol: str) -> dict:
        """Get concentration status for a specific pair."""
        now = time.time()
        hour_ago = now - 3600
        pair_times = self._trade_times.get(symbol, deque())
        recent = sum(1 for t in pair_times if t > hour_ago)
        total = self._fill_count[symbol] + self._one_sided_count[symbol]
        one_sided_rate = (
            self._one_sided_count[symbol] / total if total > 0 else 0.0
        )
        last_fill = self._last_fill_time.get(symbol, 0)
        cooldown_remaining = max(0, self.cooldown_after_fill_sec - (now - last_fill))

        return {
            "symbol": symbol,
            "trades_last_hour": recent,
            "max_per_hour": self.max_trades_per_pair_per_hour,
            "cooldown_remaining_sec": round(cooldown_remaining, 0),
            "fills": self._fill_count[symbol],
            "one_sided": self._one_sided_count[symbol],
            "one_sided_rate": round(one_sided_rate, 3),
            "blocked": symbol in self._blocked_pairs,
            "block_reason": self._block_reasons.get(symbol, ""),
        }

    def get_stats(self) -> dict:
        """Return overall concentration limiter statistics."""
        now = time.time()
        hour_ago = now - 3600

        # Active pairs (traded in last hour)
        active_pairs = {}
        for symbol, times in self._trade_times.items():
            recent = sum(1 for t in times if t > hour_ago)
            if recent > 0:
                active_pairs[symbol] = recent

        # Top concentrated pairs
        top_pairs = sorted(active_pairs.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "active_pairs": len(active_pairs),
            "blocked_pairs": sorted(self._blocked_pairs),
            "blocked_count": len(self._blocked_pairs),
            "block_reasons": dict(self._block_reasons),
            "rejections": {
                "rate_limit": self._rejections_rate_limit,
                "cooldown": self._rejections_cooldown,
                "blocked": self._rejections_blocked,
                "total": (
                    self._rejections_rate_limit
                    + self._rejections_cooldown
                    + self._rejections_blocked
                ),
            },
            "top_pairs_by_frequency": [
                {"symbol": sym, "trades_last_hour": cnt}
                for sym, cnt in top_pairs
            ],
            "config": {
                "max_trades_per_pair_per_hour": self.max_trades_per_pair_per_hour,
                "cooldown_after_fill_sec": self.cooldown_after_fill_sec,
                "one_sided_block_threshold": self.one_sided_block_threshold,
                "one_sided_min_samples": self.one_sided_min_samples,
            },
        }
