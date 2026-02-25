"""
Volume Participation Limiter — scales trading capacity to actual market volume.

Replaces the flat ConcentrationLimiter (5 trades/pair/hr) with a model that
limits our volume to a configurable percentage of actual market volume.

Key principle from Renaissance: Edge × Capacity = Real P&L.
VANRY has huge edge (154 bps) but zero capacity ($180K/day).
The profitable middle ground is pairs with BOTH decent spread AND enough
volume to actually trade without moving the market.

This module enforces:
- Max participation rate (default 2% of hourly market volume)
- Minimum daily volume threshold ($500K) — below this, excluded entirely
- Fill rate degradation based on participation (models market impact)
- One-sided fill auto-blocking (kept from ConcentrationLimiter — real risk)
"""
import logging
import time
from collections import defaultdict, deque
from decimal import Decimal
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger("arb.volume_limiter")


class VolumeParticipationLimiter:
    """Limits trading volume per pair based on actual market volume.

    Instead of a flat "N trades per hour", this checks what percentage
    of the market's volume we're consuming. If we're taking more than
    max_participation_rate of the market, we stop — because in live
    trading, we'd be moving the price and the edge would vanish.
    """

    # Fill rate degrades as participation increases (models market impact)
    FILL_DEGRADATION = {
        0.005: 0.85,   # 0-0.5% participation: 85% fill rate (invisible)
        0.010: 0.75,   # 0.5-1%: 75% (minimal impact)
        0.020: 0.60,   # 1-2%: 60% (noticeable, edge eroding)
        0.050: 0.35,   # 2-5%: 35% (significant impact)
        1.000: 0.10,   # 5%+: 10% (we ARE the market)
    }

    def __init__(self, config: Optional[dict] = None):
        cfg = (config or {}).get('volume_participation', {})

        # Maximum percentage of market hourly volume we'll consume
        self.max_participation_rate = float(cfg.get('max_participation_rate', 0.02))

        # Pairs with less than this daily volume are excluded entirely
        self.min_daily_volume_usd = float(cfg.get('min_daily_volume_usd', 500_000))

        # One-sided fill auto-blocking (kept from ConcentrationLimiter)
        self.one_sided_block_threshold = float(cfg.get('one_sided_block_threshold', 0.20))
        self.one_sided_min_samples = int(cfg.get('one_sided_min_samples', 50))

        # State tracking
        self._volume_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=5000)
        )  # symbol -> deque of (timestamp, volume_usd)
        self._pair_volumes: Dict[str, float] = {}  # symbol -> daily volume USD
        self._fill_count: Dict[str, int] = defaultdict(int)
        self._one_sided_count: Dict[str, int] = defaultdict(int)
        self._blocked_pairs: Set[str] = set()
        self._block_reasons: Dict[str, str] = {}

        # Stats counters
        self._rejections_volume = 0
        self._rejections_participation = 0
        self._rejections_blocked = 0

        logger.info(
            f"VolumeParticipationLimiter: max {self.max_participation_rate:.0%} participation, "
            f"min ${self.min_daily_volume_usd:,.0f}/day, "
            f"auto-block at {self.one_sided_block_threshold*100:.0f}% one-sided rate"
        )

    def update_pair_volumes(self, ticker_data: Dict[str, dict]) -> None:
        """Update cached daily volumes from ticker data.

        Called by PairDiscoveryEngine after fetching tickers.
        ticker_data: {symbol: {volume_24h: Decimal, last_price: Decimal, ...}}
        """
        updated = 0
        for symbol, data in ticker_data.items():
            vol_24h = data.get('volume_24h', Decimal('0'))
            last_price = data.get('last_price', Decimal('0'))
            # volume_24h is in base units; we need USD
            try:
                vol_usd = float(vol_24h) * float(last_price) if last_price > 0 else 0.0
            except (TypeError, ValueError):
                vol_usd = 0.0
            if vol_usd > 0:
                self._pair_volumes[symbol] = vol_usd
                updated += 1
        if updated > 0:
            logger.debug(f"Volume limiter: updated {updated} pair volumes")

    def get_pair_hourly_volume(self, symbol: str) -> float:
        """Get estimated hourly market volume for a pair."""
        daily = self._pair_volumes.get(symbol, 0)
        return daily / 24.0

    def is_pair_liquid_enough(self, symbol: str) -> bool:
        """Check if pair meets minimum volume threshold."""
        daily = self._pair_volumes.get(symbol, 0)
        return daily >= self.min_daily_volume_usd

    def record_trade(self, symbol: str, volume_usd: float, status: str) -> None:
        """Record a completed trade's volume and outcome.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            volume_usd: Trade notional value in USD
            status: Trade outcome — "filled", "one_sided_buy", "one_sided_sell", etc.
        """
        now = time.time()

        # Always record volume for participation tracking
        if volume_usd > 0:
            self._volume_history[symbol].append((now, volume_usd))

        if status == "filled":
            self._fill_count[symbol] += 1

        elif "one_sided" in status:
            self._one_sided_count[symbol] += 1

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
                        f"VOLUME_LIMITER: Auto-blocked {symbol} — "
                        f"{self._block_reasons[symbol]}"
                    )

    def get_participation_rate(self, symbol: str) -> float:
        """Get our current participation rate for a pair.

        Returns fraction (e.g., 0.015 = 1.5% of hourly volume).
        """
        hourly_vol = self.get_pair_hourly_volume(symbol)
        if hourly_vol <= 0:
            return 1.0  # Unknown volume = assume we're 100% of market

        now = time.time()
        cutoff = now - 3600
        our_volume = sum(
            v for t, v in self._volume_history.get(symbol, deque()) if t > cutoff
        )
        return our_volume / hourly_vol

    def get_fill_rate_modifier(self, symbol: str) -> float:
        """Get degraded fill rate based on current participation.

        Higher participation = worse fills (market impact).
        Returns the effective fill rate (not a multiplier).
        """
        rate = self.get_participation_rate(symbol)
        for threshold, fill_rate in sorted(self.FILL_DEGRADATION.items()):
            if rate <= threshold:
                return fill_rate
        return 0.10  # Beyond all thresholds

    def check(self, symbol: str, trade_size_usd: float = 500) -> Tuple[bool, str]:
        """Check if a trade on this pair is allowed.

        Returns:
            (allowed, reason) — True if allowed, False + reason if blocked.
        """
        # 1. Check if pair is auto-blocked (one-sided fills)
        if symbol in self._blocked_pairs:
            self._rejections_blocked += 1
            return False, f"blocked: {self._block_reasons.get(symbol, 'one-sided fills')}"

        # 2. Check minimum volume
        if not self.is_pair_liquid_enough(symbol):
            daily = self._pair_volumes.get(symbol, 0)
            self._rejections_volume += 1
            return False, (
                f"insufficient_volume: ${daily:,.0f}/day "
                f"< ${self.min_daily_volume_usd:,.0f} minimum"
            )

        # 3. Check participation rate
        hourly_vol = self.get_pair_hourly_volume(symbol)
        if hourly_vol <= 0:
            self._rejections_volume += 1
            return False, "no_volume_data"

        now = time.time()
        cutoff = now - 3600
        our_volume = sum(
            v for t, v in self._volume_history.get(symbol, deque()) if t > cutoff
        )
        projected_rate = (our_volume + trade_size_usd) / hourly_vol

        if projected_rate > self.max_participation_rate:
            self._rejections_participation += 1
            return False, (
                f"participation_exceeded: {projected_rate:.2%} > "
                f"{self.max_participation_rate:.0%} "
                f"(hourly_vol=${hourly_vol:,.0f}, "
                f"our_vol=${our_volume:,.0f})"
            )

        return True, f"ok: participation={our_volume / hourly_vol:.3%}"

    def get_remaining_capacity(self, symbol: str) -> float:
        """How much more USD volume can we trade this hour on this pair?"""
        hourly_vol = self.get_pair_hourly_volume(symbol)
        if hourly_vol <= 0:
            return 0

        max_volume = hourly_vol * self.max_participation_rate
        now = time.time()
        cutoff = now - 3600
        current_volume = sum(
            v for t, v in self._volume_history.get(symbol, deque()) if t > cutoff
        )
        return max(0, max_volume - current_volume)

    def unblock(self, symbol: str) -> bool:
        """Manually unblock a pair. Returns True if it was blocked."""
        if symbol in self._blocked_pairs:
            self._blocked_pairs.discard(symbol)
            self._block_reasons.pop(symbol, None)
            self._one_sided_count[symbol] = 0
            self._fill_count[symbol] = 0
            logger.info(f"VOLUME_LIMITER: Manually unblocked {symbol}")
            return True
        return False

    def get_stats(self) -> dict:
        """Dashboard-friendly stats for all tracked pairs."""
        now = time.time()
        cutoff = now - 3600

        pair_stats = {}
        for symbol in sorted(self._pair_volumes.keys()):
            daily_vol = self._pair_volumes.get(symbol, 0)
            hourly_vol = daily_vol / 24
            our_vol = sum(
                v for t, v in self._volume_history.get(symbol, deque())
                if t > cutoff
            )
            participation = our_vol / hourly_vol if hourly_vol > 0 else 0
            remaining = self.get_remaining_capacity(symbol)
            is_liquid = daily_vol >= self.min_daily_volume_usd

            # Only include pairs with activity or that are liquid
            if our_vol > 0 or is_liquid:
                pair_stats[symbol] = {
                    'daily_volume_usd': round(daily_vol, 0),
                    'hourly_volume_usd': round(hourly_vol, 0),
                    'our_hourly_volume_usd': round(our_vol, 2),
                    'participation_rate': round(participation, 6),
                    'participation_pct': f"{participation:.2%}",
                    'fill_rate_modifier': self.get_fill_rate_modifier(symbol),
                    'remaining_capacity_usd': round(remaining, 0),
                    'is_liquid': is_liquid,
                    'max_trades_remaining': int(remaining / 500) if remaining > 0 else 0,
                }

        return {
            "config": {
                "max_participation_rate": self.max_participation_rate,
                "min_daily_volume_usd": self.min_daily_volume_usd,
                "one_sided_block_threshold": self.one_sided_block_threshold,
                "one_sided_min_samples": self.one_sided_min_samples,
            },
            "pairs_tracked": len(self._pair_volumes),
            "pairs_liquid": sum(
                1 for v in self._pair_volumes.values()
                if v >= self.min_daily_volume_usd
            ),
            "pairs_excluded": sum(
                1 for v in self._pair_volumes.values()
                if v < self.min_daily_volume_usd
            ),
            "blocked_pairs": sorted(self._blocked_pairs),
            "blocked_count": len(self._blocked_pairs),
            "block_reasons": dict(self._block_reasons),
            "rejections": {
                "volume": self._rejections_volume,
                "participation": self._rejections_participation,
                "blocked": self._rejections_blocked,
                "total": (
                    self._rejections_volume
                    + self._rejections_participation
                    + self._rejections_blocked
                ),
            },
            "total_remaining_capacity_usd": round(
                sum(
                    self.get_remaining_capacity(s)
                    for s in self._pair_volumes
                    if self._pair_volumes[s] >= self.min_daily_volume_usd
                ), 0
            ),
            "pair_details": pair_stats,
        }
