"""
Tick Size Validator — prevents trades where price rounding
destroys the profit.

Usage:
    from arbitrage.utils.tick_check import check_tick_viability, get_tick_size_cached

    result = check_tick_viability(
        symbol='XCN/USDC',
        price=0.0032,
        tick_size=0.0001,
        expected_profit_bps=24.0,
        num_legs=3,
    )
    if not result['viable']:
        logger.info(f"TICK REJECT: {result['reason']}")
"""
import logging
import time

logger = logging.getLogger("arb.tick_check")

# Cache: symbol -> (tick_size, fetched_at)
_tick_cache = {}
TICK_CACHE_TTL = 3600  # 1 hour


def get_tick_size(markets, symbol):
    """
    Fetch tick size (price precision) for a symbol from market info.

    Args:
        markets: dict of market info (e.g. exchange.markets)
        symbol: normalized symbol like 'XCN/USDC'

    Returns:
        tick_size as float, or None if not found
    """
    try:
        market = markets.get(symbol)
        if not market:
            return None

        price_precision = market.get('precision', {}).get('price')
        if price_precision is not None:
            if isinstance(price_precision, int) and price_precision >= 1:
                # Decimal places: 4 -> tick = 0.0001
                tick = 10 ** (-price_precision)
            elif isinstance(price_precision, float) and price_precision < 1:
                # Already a tick size
                tick = price_precision
            elif isinstance(price_precision, (int, float)):
                tick = 10 ** (-int(price_precision))
            else:
                return None
            return tick

        # Fallback: check limits
        min_price = market.get('limits', {}).get('price', {}).get('min')
        if min_price:
            return float(min_price)

        return None
    except Exception as e:
        logger.warning(f"Could not get tick size for {symbol}: {e}")
        return None


def get_tick_size_cached(markets, symbol):
    """Get tick size with 1-hour cache."""
    cached = _tick_cache.get(symbol)
    if cached and (time.time() - cached[1]) < TICK_CACHE_TTL:
        return cached[0]

    tick = get_tick_size(markets, symbol)
    if tick is not None:
        _tick_cache[symbol] = (tick, time.time())
    return tick


def check_tick_viability(
    symbol,
    price,
    tick_size,
    expected_profit_bps,
    num_legs=2,
    max_rounding_ratio=0.5,
):
    """
    Check if tick size rounding could eat the profit.

    Args:
        symbol: pair name for logging
        price: current price of the asset
        tick_size: minimum price increment
        expected_profit_bps: expected gross profit in bps
        num_legs: number of legs where rounding applies
        max_rounding_ratio: max allowed rounding as fraction of profit

    Returns:
        dict with viable, tick_bps, max_rounding_bps, rounding_ratio, reason
    """
    if price <= 0 or tick_size <= 0:
        return {
            'viable': False,
            'tick_bps': 0,
            'max_rounding_bps': 0,
            'rounding_ratio': 999,
            'reason': f'{symbol}: invalid price ({price}) or tick ({tick_size})',
        }

    tick_bps = (tick_size / price) * 10000
    max_rounding_bps = tick_bps * num_legs

    if expected_profit_bps <= 0:
        rounding_ratio = 999
    else:
        rounding_ratio = max_rounding_bps / expected_profit_bps

    viable = rounding_ratio <= max_rounding_ratio

    if not viable:
        reason = (
            f"{symbol}: tick={tick_size} ({tick_bps:.1f}bps of ${price:.6f}) "
            f"x {num_legs} legs = {max_rounding_bps:.1f}bps rounding "
            f"> {max_rounding_ratio*100:.0f}% of {expected_profit_bps:.1f}bps profit"
        )
    else:
        reason = (
            f"{symbol}: tick={tick_bps:.1f}bps x {num_legs} = "
            f"{max_rounding_bps:.1f}bps rounding, "
            f"{rounding_ratio*100:.0f}% of {expected_profit_bps:.1f}bps — OK"
        )

    return {
        'viable': viable,
        'tick_bps': round(tick_bps, 2),
        'max_rounding_bps': round(max_rounding_bps, 2),
        'rounding_ratio': round(rounding_ratio, 3),
        'reason': reason,
    }
