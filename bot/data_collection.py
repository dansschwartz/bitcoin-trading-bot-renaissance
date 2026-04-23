"""
bot/data_collection.py — Data fetching and bar management extracted from RenaissanceTradingBot.

Functions accept `bot` (the RenaissanceTradingBot instance) as first argument
so they can access config, providers, and caches without tight coupling.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from renaissance_trading_bot import RenaissanceTradingBot

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Pair Scheduling
# ──────────────────────────────────────────────

def get_pairs_for_cycle(bot: "RenaissanceTradingBot", cycle_number: int) -> list:
    """Return pairs to scan this cycle based on 4-tier volume schedule.

    Tier 1 (top 15 by volume):   every cycle
    Tier 2 (16-50):              every 2nd cycle
    Tier 3 (51-100):             every 3rd cycle
    Tier 4 (101-150):            every 4th cycle
    """
    if not bot._pair_tiers:
        # Fallback: scan all product_ids if universe not built yet
        return list(bot.product_ids)

    pairs = []
    for pid in bot.product_ids:
        tier = bot._pair_tiers.get(pid, 1)
        if tier == 1:
            pairs.append(pid)
        elif tier == 2 and cycle_number % 2 == 0:
            pairs.append(pid)
        elif tier == 3 and cycle_number % 3 == 0:
            pairs.append(pid)
        elif tier == 4 and cycle_number % 4 == 0:
            pairs.append(pid)
    return pairs


async def build_and_apply_universe(bot: "RenaissanceTradingBot") -> None:
    """Build dynamic trading universe from Binance and apply it."""
    try:
        universe_cfg = bot.config.get('universe', {})
        min_vol = float(universe_cfg.get('min_volume_usd', 2_000_000))
        max_pairs = int(universe_cfg.get('max_pairs', 150))

        max_spread = float(universe_cfg.get('max_spread_bps', 10.0))
        universe = await bot.binance_spot.build_trading_universe(
            min_volume_usd=min_vol, max_pairs=max_pairs,
            max_spread_bps=max_spread,
        )
        if not universe:
            bot.logger.warning("UNIVERSE: Binance returned empty — keeping existing product_ids")
            return

        bot.trading_universe = universe
        bot.product_ids = [c['product_id'] for c in universe]
        bot._pair_tiers = {c['product_id']: c['tier'] for c in universe}
        bot._pair_binance_symbols = {
            c['product_id']: c['binance_symbol'] for c in universe
        }
        bot._universe_built = True
        bot._universe_last_refresh = time.time()
        bot.logger.info(f"UNIVERSE BUILT: {len(bot.product_ids)} pairs")
    except Exception as e:
        bot.logger.error(f"UNIVERSE BUILD FAILED: {e} — keeping existing product_ids")


# ──────────────────────────────────────────────
#  ML Z-Score Normalization
# ──────────────────────────────────────────────

def ml_zscore_rescale(bot: "RenaissanceTradingBot", pair: str, raw_pred: float) -> float:
    """Council #12: Per-pair Z-score normalization of ML predictions.

    Uses Welford online algorithm with exponential decay (lookback window).
    Maps z-score via tanh to [-1, 1] range matching traditional signals.
    """
    stats = bot._ml_pred_stats.get(pair)
    if stats is None:
        stats = {'mean': 0.0, 'var': 0.0, 'n': 0}
        bot._ml_pred_stats[pair] = stats

    n = stats['n']
    old_mean = stats['mean']

    if n < bot._ml_zscore_lookback:
        # Growing phase: standard Welford
        n += 1
        delta = raw_pred - old_mean
        new_mean = old_mean + delta / n
        delta2 = raw_pred - new_mean
        new_var = stats['var'] + delta * delta2
        stats['n'] = n
        stats['mean'] = new_mean
        stats['var'] = new_var
    else:
        # Steady-state: exponentially weighted update
        alpha = 1.0 / bot._ml_zscore_lookback
        delta = raw_pred - old_mean
        stats['mean'] = old_mean + alpha * delta
        stats['var'] = (1 - alpha) * (stats['var'] + alpha * delta * delta * bot._ml_zscore_lookback)
        stats['n'] = n + 1  # track total observations

    # Need at least 20 observations for stable stats
    if n < 20:
        return raw_pred  # pass through during warmup

    std = math.sqrt(stats['var'] / min(n, bot._ml_zscore_lookback))
    if std < 1e-10:
        return raw_pred  # avoid division by zero

    z = (raw_pred - stats['mean']) / std
    return float(math.tanh(z * bot._ml_zscore_tanh_scale))


# ──────────────────────────────────────────────
#  Historical Bar Gap-Fill
# ──────────────────────────────────────────────

async def gap_fill_bars_on_startup(bot: "RenaissanceTradingBot") -> None:
    """Council proposal #1: Fill missing 5-min bars from Binance on startup.

    For each pair, find the latest bar in DB and fetch any missing bars
    up to now from Binance API (max 1000 bars per pair).
    """
    from binance_spot_provider import to_binance_symbol

    if not bot.bar_aggregator or not bot._universe_built:
        return

    db_path = getattr(bot.bar_aggregator, '_db_path', None) or getattr(bot.bar_aggregator, 'db_path', None)
    if not db_path:
        bot.logger.warning("GAP-FILL: No db_path found on bar_aggregator, skipping")
        return

    bot.logger.info("GAP-FILL: Checking for missing bars across universe...")
    total_filled = 0
    sem = asyncio.Semaphore(10)

    async def _fill_pair(pid: str) -> int:
        async with sem:
            try:
                bsym = bot._pair_binance_symbols.get(pid, to_binance_symbol(pid))

                # Find latest bar in DB for this pair
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                row = conn.execute(
                    "SELECT MAX(bar_start) FROM five_minute_bars WHERE pair = ?",
                    (pid,),
                ).fetchone()
                conn.close()

                latest_bar_ts = row[0] if row and row[0] else None
                now_ts = time.time()

                if latest_bar_ts is not None:
                    # Calculate how many bars are missing
                    if isinstance(latest_bar_ts, str):
                        from datetime import datetime as _dt
                        latest_bar_ts = _dt.fromisoformat(latest_bar_ts).timestamp()
                    gap_bars = int((now_ts - latest_bar_ts) / 300)
                    if gap_bars <= 1:
                        return 0  # No gap
                    fetch_count = min(gap_bars + 5, 1000)
                else:
                    fetch_count = 1000  # No bars at all — fetch max

                candles = await bot.binance_spot.fetch_candles(bsym, '5m', fetch_count)
                if not candles:
                    return 0

                # Insert into five_minute_bars (deduplicate via INSERT OR IGNORE)
                conn = sqlite3.connect(db_path, timeout=30.0)
                inserted = 0
                for c in candles:
                    try:
                        bar_start = c['timestamp']
                        conn.execute(
                            """INSERT OR IGNORE INTO five_minute_bars
                               (pair, exchange, bar_start, bar_end, open, high, low, close, volume,
                                num_trades, vwap, log_return, avg_spread_bps, buy_sell_ratio, funding_rate)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (pid, 'binance', bar_start, bar_start + 300,
                             c['open'], c['high'], c['low'], c['close'], c['volume'],
                             0, c['close'], 0.0, 0.0, 0.5, 0.0),
                        )
                        if conn.total_changes > inserted:
                            inserted += 1
                    except Exception as e:
                        bot.logger.warning(f"Gap-fill bar insert failed for {pid}: {e}")
                conn.commit()
                conn.close()

                if inserted > 0:
                    bot.logger.info(f"GAP-FILL: {pid} — inserted {inserted} bars from Binance")
                return inserted
            except Exception as e:
                bot.logger.debug(f"GAP-FILL failed for {pid}: {e}")
                return 0

    # Run tail gap-fill for all pairs in parallel
    results = await asyncio.gather(
        *[_fill_pair(pid) for pid in bot.product_ids],
        return_exceptions=True,
    )
    total_filled = sum(r for r in results if isinstance(r, int))
    pairs_with_gaps = sum(1 for r in results if isinstance(r, int) and r > 0)
    bot.logger.info(
        f"GAP-FILL (tail): {total_filled} bars inserted across "
        f"{pairs_with_gaps}/{len(bot.product_ids)} pairs with gaps"
    )

    # ── Interior gap detection and backfill ──────────────────────
    bot.logger.info("GAP-FILL: Scanning for interior gaps (last 7 days)...")
    interior_sem = asyncio.Semaphore(10)
    seven_days_ago = time.time() - (7 * 24 * 3600)

    async def _fill_interior_gaps(pid: str) -> tuple:
        """Returns (gaps_found, bars_inserted) for one pair."""
        async with interior_sem:
            try:
                bsym = bot._pair_binance_symbols.get(pid, to_binance_symbol(pid))

                # Query all bar_start timestamps in the last 7 days
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                rows = conn.execute(
                    "SELECT bar_start FROM five_minute_bars "
                    "WHERE pair = ? AND exchange = 'binance' AND bar_start > ? "
                    "ORDER BY bar_start",
                    (pid, seven_days_ago),
                ).fetchall()
                conn.close()

                if len(rows) < 2:
                    return (0, 0)

                # Parse bar_start values to float seconds
                timestamps = []
                for r in rows:
                    ts = r[0]
                    if isinstance(ts, str):
                        from datetime import datetime as _dt
                        ts = _dt.fromisoformat(ts).timestamp()
                    timestamps.append(float(ts))
                timestamps.sort()

                # Find interior gaps: consecutive bar_start delta > 600s
                gaps = []
                for i in range(len(timestamps) - 1):
                    delta = timestamps[i + 1] - timestamps[i]
                    if delta > 600:  # More than 2 bar-widths apart → missing bars
                        gap_start = timestamps[i] + 300  # First missing bar
                        gap_end = timestamps[i + 1]
                        gap_bars = int((gap_end - gap_start) / 300)
                        if gap_bars > 0:
                            gaps.append((gap_start, gap_end, gap_bars))

                if not gaps:
                    return (0, 0)

                # Fetch and insert missing bars for each gap
                pair_inserted = 0
                for gap_start_ts, gap_end_ts, gap_bars_count in gaps:
                    try:
                        fetch_limit = min(gap_bars_count + 2, 1000)
                        candles = await bot.binance_spot.fetch_candles(
                            bsym, '5m', limit=fetch_limit,
                            start_time=int(gap_start_ts * 1000),
                            end_time=int(gap_end_ts * 1000),
                        )
                        if not candles:
                            continue

                        conn = sqlite3.connect(db_path, timeout=30.0)
                        for c in candles:
                            try:
                                bar_start = c['timestamp']
                                conn.execute(
                                    """INSERT OR IGNORE INTO five_minute_bars
                                       (pair, exchange, bar_start, bar_end, open, high, low, close, volume,
                                        num_trades, vwap, log_return, avg_spread_bps, buy_sell_ratio, funding_rate)
                                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                    (pid, 'binance', bar_start, bar_start + 300,
                                     c['open'], c['high'], c['low'], c['close'], c['volume'],
                                     0, c['close'], 0.0, 0.0, 0.5, 0.0),
                                )
                                pair_inserted += 1
                            except Exception as e:
                                bot.logger.warning(f"Interior gap-fill bar insert failed for {pid}: {e}")
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        bot.logger.debug(f"GAP-FILL interior fetch failed for {pid} gap {gap_start_ts}-{gap_end_ts}: {e}")
                        continue

                if pair_inserted > 0:
                    bot.logger.info(
                        f"GAP-FILL interior: {pid} — {len(gaps)} gaps found, "
                        f"{pair_inserted} bars inserted"
                    )
                return (len(gaps), pair_inserted)
            except Exception as e:
                bot.logger.debug(f"GAP-FILL interior failed for {pid}: {e}")
                return (0, 0)

    interior_results = await asyncio.gather(
        *[_fill_interior_gaps(pid) for pid in bot.product_ids],
        return_exceptions=True,
    )
    total_interior_gaps = sum(r[0] for r in interior_results if isinstance(r, tuple))
    total_interior_bars = sum(r[1] for r in interior_results if isinstance(r, tuple))
    pairs_with_interior = sum(1 for r in interior_results if isinstance(r, tuple) and r[0] > 0)

    # Log completeness improvement
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        completeness_rows = conn.execute(
            "SELECT pair, COUNT(*) as cnt FROM five_minute_bars "
            "WHERE exchange = 'binance' AND bar_start > ? GROUP BY pair",
            (seven_days_ago,),
        ).fetchall()
        conn.close()
        if completeness_rows:
            expected_bars = int((time.time() - seven_days_ago) / 300)
            avg_pct = sum(min(r[1] / expected_bars * 100, 100) for r in completeness_rows) / len(completeness_rows)
            bot.logger.info(
                f"GAP-FILL interior COMPLETE: {total_interior_gaps} gaps found across "
                f"{pairs_with_interior}/{len(bot.product_ids)} pairs, "
                f"{total_interior_bars} bars inserted. "
                f"Avg completeness (7d): {avg_pct:.1f}%"
            )
        else:
            bot.logger.info(
                f"GAP-FILL interior COMPLETE: {total_interior_gaps} gaps, "
                f"{total_interior_bars} bars inserted"
            )
    except Exception:
        bot.logger.info(
            f"GAP-FILL interior COMPLETE: {total_interior_gaps} gaps, "
            f"{total_interior_bars} bars inserted"
        )

    grand_total = total_filled + total_interior_bars
    bot.logger.info(f"GAP-FILL TOTAL: {grand_total} bars inserted (tail: {total_filled}, interior: {total_interior_bars})")


# ──────────────────────────────────────────────
#  Per-Pair Data Collection
# ──────────────────────────────────────────────

async def collect_from_binance(bot: "RenaissanceTradingBot", product_id: str) -> Dict[str, Any]:
    """Collect market data from Binance for a single pair.

    Returns a market_data dict compatible with the existing pipeline.
    """
    from binance_spot_provider import to_binance_symbol
    from enhanced_technical_indicators import EnhancedTechnicalIndicators

    binance_sym = bot._pair_binance_symbols.get(product_id)
    if not binance_sym:
        binance_sym = to_binance_symbol(product_id)

    try:
        ticker = await bot.binance_spot.fetch_ticker(binance_sym)
        if not ticker or ticker.get('price', 0) <= 0:
            return {}

        # Build market_data dict compatible with existing pipeline
        tech = get_tech(bot, product_id)

        # Fetch latest candle for tech indicator feed
        candles = await bot.binance_spot.fetch_candles(binance_sym, '5m', 2)
        if candles:
            from enhanced_technical_indicators import PriceData
            latest = candles[-1]
            price_data = PriceData(
                timestamp=datetime.utcfromtimestamp(latest['timestamp']),
                open=latest['open'],
                high=latest['high'],
                low=latest['low'],
                close=latest['close'],
                volume=latest['volume'],
            )
            tech.update_price_data(price_data)
        else:
            price_data = None

        technical_signals = tech.get_latest_signals()

        # Build orderbook snapshot for microstructure signals
        order_book_snapshot = None
        try:
            ob = await bot.binance_spot.fetch_orderbook(binance_sym, 20)
            if ob and ob.get('bids') and ob.get('asks'):
                from microstructure_engine import OrderBookSnapshot, OrderBookLevel
                bids = [OrderBookLevel(price=p, size=s) for p, s in ob['bids']]
                asks = [OrderBookLevel(price=p, size=s) for p, s in ob['asks']]
                order_book_snapshot = OrderBookSnapshot(
                    timestamp=datetime.utcnow(),
                    bids=bids,
                    asks=asks,
                    last_price=ticker['price'],
                    last_size=0.0,
                )
        except Exception as e:
            bot.logger.warning(f"Order book snapshot construction failed: {e}")

        # Compute bid_ask_spread
        bid = ticker.get('bid', 0)
        ask = ticker.get('ask', 0)
        spread = ask - bid if bid > 0 and ask > 0 else 0.0

        return {
            'order_book_snapshot': order_book_snapshot,
            'price_data': price_data,
            'technical_signals': technical_signals,
            'alternative_signals': {},  # Filled later in sequential phase
            'ticker': {
                'price': ticker['price'],
                'bid': bid,
                'ask': ask,
                'best_bid': bid,
                'best_ask': ask,
                'volume': ticker.get('volume_24h', 0),
                'volume_24h': ticker.get('volume_24h', 0),
                'quote_volume_24h': ticker.get('quote_volume_24h', 0),
                'bid_ask_spread': spread,
            },
            'product_id': product_id,
            'timestamp': datetime.now(),
            'recent_trades': [],
            '_data_source': 'binance',
        }
    except Exception as e:
        bot.logger.debug(f"Binance collect failed for {product_id}: {e}")
        return {}


def get_tech(bot: "RenaissanceTradingBot", product_id: str):
    """Get per-asset technical indicators instance (creates on-demand for new assets)."""
    from enhanced_technical_indicators import EnhancedTechnicalIndicators
    if product_id not in bot._tech_indicators:
        bot._tech_indicators[product_id] = EnhancedTechnicalIndicators()
    return bot._tech_indicators[product_id]


def load_price_df_from_db(bot: "RenaissanceTradingBot", product_id: str, limit: int = 300):
    """Load recent OHLCV bars from DB for ML inference when tech indicators are sparse."""
    try:
        db_path = bot.config.get('database', {}).get('path', 'data/renaissance_bot.db')
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        rows = conn.execute(
            "SELECT bar_start, open, high, low, close, volume "
            "FROM five_minute_bars WHERE pair=? ORDER BY bar_start DESC LIMIT ?",
            (product_id, limit)
        ).fetchall()
        conn.close()
        if len(rows) < 30:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        bot.logger.debug(f"DB bar load failed for {product_id}: {e}")
        return pd.DataFrame()


def load_candles_from_db(bot: "RenaissanceTradingBot", product_id: str, limit: int = 200) -> List:
    """Load historical bars from five_minute_bars as PriceData objects for tech indicator bootstrap."""
    try:
        from enhanced_technical_indicators import PriceData
        db_path = bot.config.get('database', {}).get('path', 'data/renaissance_bot.db')
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        rows = conn.execute(
            "SELECT bar_start, open, high, low, close, volume "
            "FROM five_minute_bars WHERE pair=? ORDER BY bar_start ASC LIMIT ?",
            (product_id, limit)
        ).fetchall()
        conn.close()
        if not rows:
            return []
        candles = []
        for row in rows:
            ts, o, h, l, c, v = row
            candles.append(PriceData(
                timestamp=datetime.fromtimestamp(ts),
                open=float(o), high=float(h), low=float(l),
                close=float(c), volume=float(v or 0),
            ))
        bot.logger.info(f"Loaded {len(candles)} bars from DB for {product_id}")
        return candles
    except Exception as e:
        bot.logger.warning(f"DB candle load failed for {product_id}: {e}")
        return []
