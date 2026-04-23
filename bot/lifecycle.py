"""
bot/lifecycle.py — Background loops, lifecycle, and kill switch extracted from RenaissanceTradingBot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from renaissance_trading_bot import RenaissanceTradingBot

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Kill Switch
# ──────────────────────────────────────────────


def trigger_kill_switch(bot: "RenaissanceTradingBot", reason: str) -> None:
    """Activate kill switch: close all positions, halt trading loop."""
    bot.logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
    bot._killed = True
    # Stop token spray exit loop
    if bot.token_spray:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(bot.token_spray.stop_exit_loop())
        except Exception as e:
            bot.logger.warning(f"Token spray exit loop stop failed during kill switch: {e}")
    # Stop straddle exit loops (all assets)
    for _s_asset, _s_engine in bot.straddle_engines.items():
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_s_engine.stop_exit_loop())
        except Exception as e:
            bot.logger.warning(f"Straddle exit loop stop failed for {_s_asset} during kill switch: {e}")
    try:
        bot.position_manager.set_emergency_stop(True, reason)
    except Exception as e:
        bot.logger.error(f"Emergency stop failed: {e}")
    # Fire alert asynchronously (best-effort)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(
                bot.alert_manager.send_alert("CRITICAL", "Kill Switch", reason)
            )
    except Exception as e:
        bot.logger.warning(f"Kill switch alert dispatch failed: {e}")


def check_kill_file(bot: "RenaissanceTradingBot") -> None:
    """Check for file-based kill switch (touch KILL_SWITCH to halt)."""
    if bot.KILL_FILE.exists():
        reason = bot.KILL_FILE.read_text().strip() or "Kill file detected"
        trigger_kill_switch(bot, reason)


# ──────────────────────────────────────────────
#  WebSocket Feed
# ──────────────────────────────────────────────


async def run_websocket_feed(bot: "RenaissanceTradingBot") -> None:
    """Background WebSocket feed for real-time market data.

    Uses exponential backoff (5s -> 10s -> 20s ... capped at 300s) so
    persistent Coinbase WebSocket failures don't flood the logs or
    consume CPU in a tight reconnect loop.  This is a best-effort data
    source; the bot works fine via REST polling when the WS is down.
    """
    backoff = 5
    max_backoff = 300
    while not bot._killed:
        try:
            await bot._ws_client.connect_websocket()
            backoff = 5  # Reset backoff on successful connect
            await bot._ws_client.listen_for_messages(bot._ws_queue)
        except Exception as e:
            bot.logger.warning(
                f"Coinbase WebSocket error (retry in {backoff}s): {e}"
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)


# ──────────────────────────────────────────────
#  Multi-Exchange Arbitrage Engine
# ──────────────────────────────────────────────


async def run_arbitrage_engine(bot: "RenaissanceTradingBot") -> None:
    """Run the arbitrage engine as a background task."""
    try:
        bot.logger.info("Arbitrage engine starting...")
        await bot.arbitrage_orchestrator.start()
    except asyncio.CancelledError:
        bot.logger.info("Arbitrage engine cancelled — shutting down")
        await bot.arbitrage_orchestrator.stop()
    except Exception as e:
        bot.logger.error(f"Arbitrage engine error: {e}")
        try:
            await bot.arbitrage_orchestrator.stop()
        except Exception as e:
            bot.logger.warning(f"Arbitrage orchestrator stop failed after error: {e}")


# ──────────────────────────────────────────────
#  BTC Price Relay (Binance WS → Reversal Strategy)
# ──────────────────────────────────────────────


async def run_btc_price_relay(bot: "RenaissanceTradingBot") -> None:
    """Feed BTC price to reversal strategy every 10s from Binance WS."""
    relay_count = 0
    while True:
        try:
            await asyncio.sleep(10)
            if not bot._unified_price_feed:
                continue
            btc = bot._unified_price_feed.get_ticker("BTC/USDT")
            if btc and btc.get("last_price"):
                price = float(btc["last_price"])
                if price > 0:
                    bot.reversal_strategy.update_btc_price(price)
                    relay_count += 1
                    if relay_count % 60 == 1:  # Log every ~10 minutes
                        bot.logger.debug(
                            f"BTC price relay: ${price:,.2f} "
                            f"(relay #{relay_count}, feed age {bot._unified_price_feed.get_age_ms():.0f}ms)"
                        )
        except asyncio.CancelledError:
            return
        except Exception as e:
            bot.logger.debug(f"BTC price relay error: {e}")


# ──────────────────────────────────────────────
#  Strategy A Independent Loop (60s cycle)
# ──────────────────────────────────────────────


async def run_strategy_a_loop(bot: "RenaissanceTradingBot") -> None:
    """Run Strategy A on its own 60s timer, decoupled from main pair scanning."""
    cycle_count = 0
    while True:
        try:
            await asyncio.sleep(60)
            cycle_count += 1

            # 1. Fetch fresh prices from Binance (cheap, ~200ms)
            sa_prices = dict(bot._last_prices) if hasattr(bot, '_last_prices') else {}
            sa_needed = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT",
                         "SOL-USD": "SOLUSDT", "XRP-USD": "XRPUSDT",
                         "DOGE-USD": "DOGEUSDT"}
            try:
                import requests as _sa_req
                _resp = _sa_req.get(
                    "https://api.binance.com/api/v3/ticker/price",
                    timeout=5,
                )
                if _resp.status_code == 200:
                    _tickers = {t['symbol']: float(t['price']) for t in _resp.json()}
                    for _pair, _sym in sa_needed.items():
                        _px = _tickers.get(_sym, 0)
                        if _px > 0:
                            sa_prices[_pair] = _px
            except Exception as e:
                bot.logger.warning(f"Strategy A Binance price fetch failed: {e}")

            # Need at least SOL price for any instrument
            if "SOL-USD" not in sa_prices or sa_prices["SOL-USD"] <= 0:
                continue

            # 2. Get regime
            sa_regime = "unknown"
            if bot.regime_overlay and bot.regime_overlay.enabled:
                sa_regime = bot.regime_overlay.get_hmm_regime_label() or "unknown"

            # 3. Get cached ML predictions (populated by main cycle)
            if not hasattr(bot, '_sa_ml_cache'):
                bot._sa_ml_cache = {}

            # 4. Cross-data for timing features
            sa_cross = getattr(bot, '_latest_cross_data', None)

            # 5. Execute cycle
            await bot.polymarket_executor.execute_cycle(
                ml_predictions=bot._sa_ml_cache,
                current_prices=sa_prices,
                current_regime=sa_regime,
                cross_data=sa_cross,
            )

            # 6. Check live resolutions
            if bot.polymarket_live_executor:
                try:
                    bot.polymarket_live_executor.check_live_resolutions()
                except Exception as e:
                    bot.logger.warning(f"Polymarket live resolution check failed: {e}")

            if cycle_count % 10 == 1:
                bot.logger.info(
                    f"Strategy A loop: cycle #{cycle_count}, "
                    f"prices={len(sa_prices)}, ml_cache={len(bot._sa_ml_cache)}, "
                    f"regime={sa_regime}"
                )

        except asyncio.CancelledError:
            return
        except Exception as e:
            bot.logger.warning(f"Strategy A loop error: {e}")
            await asyncio.sleep(10)


# ──────────────────────────────────────────────
#  Liquidation Cascade Detector (Module D)
# ──────────────────────────────────────────────


async def run_liquidation_detector(bot: "RenaissanceTradingBot") -> None:
    """Run the liquidation cascade detector as a background task."""
    try:
        bot.logger.info("Liquidation cascade detector starting...")
        await bot.liquidation_detector.start()
    except asyncio.CancelledError:
        bot.logger.info("Liquidation detector cancelled — shutting down")
        await bot.liquidation_detector.stop()
    except Exception as e:
        bot.logger.error(f"Liquidation detector error: {e}")
        try:
            await bot.liquidation_detector.stop()
        except Exception as e:
            bot.logger.warning(f"Liquidation detector stop failed after error: {e}")


# ──────────────────────────────────────────────
#  Fast Mean Reversion Scanner
# ──────────────────────────────────────────────


async def run_fast_reversion_scanner(bot: "RenaissanceTradingBot") -> None:
    """Run the fast mean reversion scanner as a background task."""
    try:
        await bot.fast_reversion_scanner.run_loop()
    except asyncio.CancelledError:
        bot.fast_reversion_scanner.stop()
    except Exception as e:
        bot.logger.error(f"Fast reversion scanner error: {e}")


# ──────────────────────────────────────────────
#  Sub-Bar Early Exit Scanner (10s loop)
# ──────────────────────────────────────────────


async def run_sub_bar_scanner(bot: "RenaissanceTradingBot") -> None:
    """Run the sub-bar scanner as a background task."""
    try:
        async def _get_positions():
            """Get open positions for sub-bar scanner."""
            positions = []
            with bot.position_manager._lock:
                for pos in bot.position_manager.positions.values():
                    if pos.status.value == 'OPEN':
                        # Look up predicted_magnitude from pending predictions
                        pred = bot._pending_predictions.get(pos.product_id, {})
                        positions.append({
                            'product_id': pos.product_id,
                            'position_id': pos.position_id,
                            'side': pos.side.value.upper(),
                            'entry_price': float(pos.entry_price),
                            'size_usd': float(pos.size * pos.entry_price),
                            'open_timestamp': pos.entry_time.timestamp() if pos.entry_time else 0,
                            'predicted_magnitude_bps': pred.get('predicted_magnitude_bps', 100.0),
                        })
            return positions

        async def _get_price(symbol: str) -> float:
            """Get current price for a symbol."""
            # Try cached prices from BinanceSpotProvider
            try:
                if hasattr(bot, 'binance_spot_provider') and bot.binance_spot_provider:
                    ticker = bot.binance_spot_provider.get_cached_ticker(symbol)
                    if ticker and ticker.get('last_price', 0) > 0:
                        return float(ticker['last_price'])
            except Exception as e:
                bot.logger.warning(f"Binance spot provider price fetch failed for {symbol}: {e}")
            # Fallback to position current_price
            with bot.position_manager._lock:
                for pos in bot.position_manager.positions.values():
                    if pos.product_id == symbol and pos.current_price > 0:
                        return float(pos.current_price)
            return 0.0

        async def _exit_callback(product_id: str, reason: str, details: dict):
            """Handle sub-bar exit trigger (only when observation_mode=False)."""
            bot.logger.warning(
                f"SUB-BAR EXIT: {product_id} trigger={reason} "
                f"pnl={details.get('pnl_bps', 0):.1f}bps"
            )
            try:
                with bot.position_manager._lock:
                    for pos in list(bot.position_manager.positions.values()):
                        if pos.product_id == product_id and pos.status.value == 'OPEN':
                            ok, msg = bot.position_manager.close_position(
                                pos.position_id, reason=f"sub_bar_{reason.lower()}"
                            )
                            if ok:
                                _cpx = await bot._resolve_close_price(pos)
                                _side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                                _rpnl = bot._compute_realized_pnl(
                                    pos.entry_price, _cpx, pos.size, _side
                                )
                                bot._track_task(
                                    bot.db_manager.close_position_record(
                                        pos.position_id,
                                        close_price=float(_cpx),
                                        realized_pnl=float(_rpnl),
                                        exit_reason=f"sub_bar_{reason.lower()}",
                                    )
                                )
                            break
            except Exception as e:
                bot.logger.error(f"Sub-bar exit execution failed: {e}")

        await bot.sub_bar_scanner.start(
            position_getter=_get_positions,
            price_getter=_get_price,
            exit_callback=_exit_callback,
        )
    except asyncio.CancelledError:
        await bot.sub_bar_scanner.stop()
    except Exception as e:
        bot.logger.error(f"Sub-bar scanner error: {e}")


# ──────────────────────────────────────────────
#  Heartbeat Writer (Multi-Bot Coordination)
# ──────────────────────────────────────────────


async def run_heartbeat_writer(bot: "RenaissanceTradingBot", interval: float = 5.0) -> None:
    """Run the heartbeat writer as a background task."""
    try:
        await bot.heartbeat_writer.start(bot, interval=interval)
    except asyncio.CancelledError:
        bot.heartbeat_writer.stop()
    except Exception as e:
        bot.logger.error(f"Heartbeat writer error: {e}")


# ──────────────────────────────────────────────
#  Phase 2 Observation Loops
# ──────────────────────────────────────────────


async def run_portfolio_drift_logger(bot: "RenaissanceTradingBot") -> None:
    """Log target vs actual portfolio drift every 60s (observation mode -- no corrections)."""
    try:
        while not bot._killed:
            try:
                engine = bot.medallion_portfolio_engine
                drift = engine.compute_drift()
                if drift:
                    pairs = ", ".join(f"{p}={d:+.0f}$" for p, d in drift.items())
                    bot.logger.info(f"PORTFOLIO DRIFT (obs): {pairs}")
            except Exception as e:
                bot.logger.debug(f"Portfolio drift logger error: {e}")
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        bot.logger.warning("Portfolio drift logger loop cancelled")


async def run_insurance_scanner_loop(bot: "RenaissanceTradingBot") -> None:
    """Scan for insurance premiums every 30 minutes (observation mode)."""
    try:
        while not bot._killed:
            for pair in bot.product_ids[:3]:  # Top 3 products only
                try:
                    result = bot.insurance_scanner.get_all_premiums(pair)
                    if result.get("any_premium_detected"):
                        count = result.get("total_premiums_found", 0)
                        rec = result.get("combined_recommendation", "none")
                        bot.logger.info(
                            f"INSURANCE PREMIUM (obs): {pair} — {count} premiums detected, rec={rec}"
                        )
                except Exception as e:
                    bot.logger.debug(f"Insurance scanner error for {pair}: {e}")
            await asyncio.sleep(1800)  # 30 minutes
    except asyncio.CancelledError:
        bot.logger.warning("Insurance scanner loop cancelled")


async def run_daily_signal_review_loop(bot: "RenaissanceTradingBot") -> None:
    """Run daily signal P&L review at midnight UTC."""
    try:
        while not bot._killed:
            now = datetime.now(timezone.utc)
            # Sleep until next midnight UTC
            tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
            wait_seconds = (tomorrow - now).total_seconds()
            await asyncio.sleep(wait_seconds)
            if bot._killed:
                break
            try:
                summary = bot.daily_signal_review.update_daily()
                if summary:
                    for sig_type, stats in summary.items():
                        status = stats.get("status", "active")
                        pnl = stats.get("pnl", 0)
                        bot.logger.info(
                            f"DAILY SIGNAL REVIEW: {sig_type} — P&L=${pnl:.2f}, status={status}"
                        )
            except Exception as e:
                bot.logger.error(f"Daily signal review error: {e}")
    except asyncio.CancelledError:
        bot.logger.warning("Daily signal review loop cancelled")


# ──────────────────────────────────────────────
#  Phase 2 Monitor Loops (BUG 6 fix — orphaned monitors)
# ──────────────────────────────────────────────


async def run_beta_monitor_loop(bot: "RenaissanceTradingBot") -> None:
    """Periodic beta computation (every 60 min, observation mode)."""
    try:
        while not bot._killed:
            try:
                report = bot.beta_monitor.get_report()
                beta = report.get("current_beta", 0.0)
                status = report.get("current_status", "ok")
                trend = report.get("trend", "unknown")
                bot.logger.info(
                    f"BETA MONITOR (obs): beta={beta:+.4f} status={status} trend={trend}"
                )
                if bot.beta_monitor.should_alert() and bot.monitoring_alert_manager:
                    hedge = bot.beta_monitor.get_hedge_recommendation()
                    bot._track_task(bot.monitoring_alert_manager.send_warning(
                        f"Beta alert: {hedge.get('rationale', 'high beta deviation')}"
                    ))
            except Exception as e:
                bot.logger.debug(f"Beta monitor loop error: {e}")
            await asyncio.sleep(3600)  # 60 minutes
    except asyncio.CancelledError:
        bot.logger.warning("Beta monitor loop cancelled")


async def run_sharpe_monitor_loop(bot: "RenaissanceTradingBot") -> None:
    """Periodic Sharpe health check (every 60 min, observation mode)."""
    try:
        while not bot._killed:
            try:
                report = bot.sharpe_monitor_medallion.get_report()
                s7 = report.get("sharpe_7d", 0.0)
                s30 = report.get("sharpe_30d", 0.0)
                status = report.get("status", "unknown")
                mult = report.get("exposure_multiplier", 1.0)
                bot.logger.info(
                    f"SHARPE MONITOR (obs): 7d={s7:.2f} 30d={s30:.2f} "
                    f"status={status} exposure_mult={mult:.2f}"
                )
            except Exception as e:
                bot.logger.debug(f"Sharpe monitor loop error: {e}")
            await asyncio.sleep(3600)  # 60 minutes
    except asyncio.CancelledError:
        bot.logger.warning("Sharpe monitor loop cancelled")


async def run_capacity_monitor_loop(bot: "RenaissanceTradingBot") -> None:
    """Periodic capacity analysis (every 60 min, observation mode)."""
    try:
        while not bot._killed:
            try:
                caps = bot.capacity_monitor.get_all_capacities()
                constrained = [p for p, r in caps.items() if r.get("capacity_status") == "constrained"]
                warning = [p for p, r in caps.items() if r.get("capacity_status") == "warning"]
                bot.logger.info(
                    f"CAPACITY MONITOR (obs): {len(caps)} pairs analysed, "
                    f"{len(constrained)} constrained, {len(warning)} warning"
                )
                if constrained and bot.monitoring_alert_manager:
                    bot._track_task(bot.monitoring_alert_manager.send_warning(
                        f"Capacity constrained pairs: {', '.join(constrained)}"
                    ))
            except Exception as e:
                bot.logger.debug(f"Capacity monitor loop error: {e}")
            await asyncio.sleep(3600)  # 60 minutes
    except asyncio.CancelledError:
        bot.logger.warning("Capacity monitor loop cancelled")


async def run_regime_detector_loop(bot: "RenaissanceTradingBot") -> None:
    """Periodic regime retraining + prediction (every 5 min, observation mode)."""
    try:
        while not bot._killed:
            try:
                if bot.medallion_regime.needs_retrain():
                    trained = bot.medallion_regime.train()
                    if trained:
                        bot.medallion_regime.save_model()
                        bot.logger.info("REGIME DETECTOR (obs): Model retrained and saved")
                pred = bot.medallion_regime.predict_current_regime()
                regime = pred.get("regime_name", "unknown")
                conf = pred.get("confidence", 0.0)
                bot.logger.info(
                    f"REGIME DETECTOR (obs): regime={regime} confidence={conf:.2f}"
                )
            except Exception as e:
                bot.logger.debug(f"Regime detector loop error: {e}")
            await asyncio.sleep(300)  # 5 minutes
    except asyncio.CancelledError:
        bot.logger.warning("Regime detector loop cancelled")


# ──────────────────────────────────────────────
#  Unified Telegram Reporting (Gap 5 fix)
# ──────────────────────────────────────────────


async def run_telegram_report_loop(bot: "RenaissanceTradingBot") -> None:
    """Send a consolidated hourly status report via Telegram."""
    try:
        await asyncio.sleep(300)  # Wait 5 min after startup before first report
        while not bot._killed:
            try:
                stats = {
                    "uptime": str(datetime.now(timezone.utc) - bot._start_time).split('.')[0],
                    "trades_1h": 0,
                    "pnl_1h": 0.0,
                    "open_positions": len(bot.position_manager.positions),
                    "exchanges_healthy": "coinbase",
                }
                # Count recent trades from DB
                if bot.db_enabled:
                    try:
                        import sqlite3
                        cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
                        conn = sqlite3.connect(bot.db_path, timeout=30.0)
                        row = conn.execute(
                            "SELECT COUNT(*), COALESCE(SUM(CASE WHEN UPPER(side)='SELL' "
                            "THEN size*price WHEN UPPER(side)='BUY' THEN -size*price ELSE 0 END), 0) "
                            "FROM trades WHERE timestamp >= ? AND status != 'FAILED'",
                            (cutoff,)
                        ).fetchone()
                        conn.close()
                        if row:
                            stats["trades_1h"] = row[0] or 0
                            stats["pnl_1h"] = float(row[1] or 0)
                    except Exception as e:
                        bot.logger.warning(f"Telegram report trade stats DB query failed: {e}")

                await bot.monitoring_alert_manager._telegram.send_hourly_heartbeat(stats)
            except Exception as e:
                bot.logger.debug(f"Telegram report loop error: {e}")
            await asyncio.sleep(3600)  # Every hour
    except asyncio.CancelledError:
        bot.logger.warning("Telegram report loop cancelled")


# ──────────────────────────────────────────────
#  Data Pruning
# ──────────────────────────────────────────────


def prune_old_data(bot: "RenaissanceTradingBot") -> None:
    """Prune old database rows and bound in-memory collections."""
    try:
        import sqlite3
        db_path = bot.db_manager.db_path if hasattr(bot.db_manager, 'db_path') else "data/renaissance_bot.db"
        conn = sqlite3.connect(db_path, timeout=30.0)

        # Prune tables older than retention period
        pruned = {}
        prune_rules = [
            ("polymarket_skip_log", "timestamp", "7 days"),
            ("ml_predictions", "timestamp", "7 days"),
            ("breakout_scans", "scan_time", "3 days"),
            ("market_data", "timestamp", "3 days"),
            ("five_minute_bars", "bar_end", None),  # Keep last 7 days by epoch
        ]

        for table, col, retention in prune_rules:
            try:
                if retention:
                    cur = conn.execute(
                        f"DELETE FROM [{table}] WHERE [{col}] < datetime('now', '-{retention}')"
                    )
                else:
                    # Epoch-based: bar_end is Unix timestamp
                    import time
                    cutoff = time.time() - 7 * 86400
                    cur = conn.execute(
                        f"DELETE FROM [{table}] WHERE [{col}] < ?", (cutoff,)
                    )
                if cur.rowcount > 0:
                    pruned[table] = cur.rowcount
            except Exception as e:
                bot.logger.debug(f"Prune {table} skipped: {e}")

        # ── Remove bars for pairs no longer in the active universe ──
        # product_ids use "BTC-USD" format; bars may also be stored as "BTC/USDT".
        # Build a set of base assets (e.g. {"BTC","ETH",...}) from the active universe
        # and keep bars whose base asset matches any active pair.
        if hasattr(bot, 'product_ids') and bot.product_ids:
            active_bases = set()
            for pid in bot.product_ids:
                # "BTC-USD" → "BTC", "ETH/USDT" → "ETH"
                base = pid.split('-')[0].split('/')[0].upper()
                active_bases.add(base)

            # Find distinct pairs currently stored in five_minute_bars
            try:
                stored_pairs = [
                    r[0] for r in conn.execute(
                        "SELECT DISTINCT pair FROM five_minute_bars"
                    ).fetchall()
                ]
                orphan_pairs = []
                for sp in stored_pairs:
                    sp_base = sp.split('-')[0].split('/')[0].upper()
                    if sp_base not in active_bases:
                        orphan_pairs.append(sp)

                if orphan_pairs:
                    placeholders = ','.join('?' * len(orphan_pairs))
                    cur = conn.execute(
                        f"DELETE FROM five_minute_bars WHERE pair IN ({placeholders})",
                        orphan_pairs,
                    )
                    if cur.rowcount > 0:
                        pruned['five_minute_bars_orphan'] = cur.rowcount
                        bot.logger.info(
                            f"DB PRUNE: removed {cur.rowcount} bars for "
                            f"{len(orphan_pairs)} orphan pairs no longer in "
                            f"universe: {orphan_pairs}"
                        )
            except Exception as e:
                bot.logger.debug(f"Orphan bar prune skipped: {e}")

        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
        conn.close()

        if pruned:
            bot.logger.info(f"DB PRUNE: {pruned}")

        # Bound in-memory collections
        if hasattr(bot, '_tech_indicators') and len(bot._tech_indicators) > 200:
            # Keep only pairs we've seen recently
            active = set(bot.product_ids) if hasattr(bot, 'product_ids') else set()
            stale = [k for k in bot._tech_indicators if k not in active]
            for k in stale[:50]:  # Remove 50 at a time
                del bot._tech_indicators[k]
            bot.logger.debug(f"Evicted {len(stale[:50])} stale tech indicator instances")

    except Exception as e:
        bot.logger.warning(f"Prune error: {e}")


# ──────────────────────────────────────────────
#  Shutdown
# ──────────────────────────────────────────────


async def shutdown(bot: "RenaissanceTradingBot") -> None:
    """Cancel background tasks and cleanup resources."""
    bot.logger.info("Shutting down - cancelling background tasks...")
    # Stop cascade collector (thread-based, not asyncio)
    if bot.cascade_collector:
        try:
            bot.cascade_collector.stop()
        except Exception as e:
            bot.logger.warning(f"Cascade collector stop failed during shutdown: {e}")
    # Stop sub-bar scanner
    if bot.sub_bar_scanner:
        try:
            await bot.sub_bar_scanner.stop()
        except Exception as e:
            bot.logger.warning(f"Sub-bar scanner stop failed during shutdown: {e}")
    for task in bot._background_tasks:
        if not task.done():
            task.cancel()
    if bot._background_tasks:
        await asyncio.gather(*bot._background_tasks, return_exceptions=True)
    bot._background_tasks.clear()
    bot.logger.info("Shutdown complete.")


# ──────────────────────────────────────────────
#  Main Trading Loop
# ──────────────────────────────────────────────


async def run_continuous_trading(bot: "RenaissanceTradingBot", cycle_interval: int = 300) -> None:
    """Run continuous Renaissance trading (default 5-minute cycles)"""
    # Import optional dependencies used by this function
    try:
        from recovery.state_manager import StateManager, SystemState
        from recovery.shutdown import GracefulShutdownHandler
        RECOVERY_AVAILABLE = True
    except ImportError:
        RECOVERY_AVAILABLE = False
        SystemState = None  # type: ignore[assignment,misc]
        GracefulShutdownHandler = None  # type: ignore[assignment,misc]

    try:
        from polymarket_spread_capture import ASSETS as SC_ASSETS
    except ImportError:
        SC_ASSETS = []

    bot.logger.info(f"Starting continuous Renaissance trading with {cycle_interval}s cycles")

    # ── Module A: Graceful Shutdown Handler ──
    if bot.state_manager and RECOVERY_AVAILABLE:
        try:
            loop = asyncio.get_event_loop()
            bot.shutdown_handler = GracefulShutdownHandler(
                state_manager=bot.state_manager,
                coinbase_client=bot.coinbase_client,
                alert_manager=bot.monitoring_alert_manager,
                drain_timeout_seconds=30.0,
            )
            bot.shutdown_handler.install(loop=loop)
            await bot.state_manager.aset_system_state(SystemState.STARTING, "bot starting")
            bot.logger.info("Graceful shutdown handlers installed")
        except Exception as e:
            bot.logger.warning(f"Graceful shutdown setup failed: {e}")
            # Fallback to basic signal handlers
            def _handle_shutdown(signum, frame):
                trigger_kill_switch(bot, f"Signal {signum} received")
            signal.signal(signal.SIGINT, _handle_shutdown)
            signal.signal(signal.SIGTERM, _handle_shutdown)
    else:
        # Fallback signal handlers when recovery module is not available
        def _handle_shutdown(signum, frame):
            trigger_kill_switch(bot, f"Signal {signum} received")
        signal.signal(signal.SIGINT, _handle_shutdown)
        signal.signal(signal.SIGTERM, _handle_shutdown)

    # Restore positions from DB so anti-stacking logic works across restarts.
    # In paper mode: restore positions but reset daily PnL (balances reset each start).
    paper_mode = bot.config.get("trading", {}).get("paper_trading", True)
    if not paper_mode:
        await bot._restore_state()
    else:
        # Restore positions only (for anti-stacking), reset PnL
        try:
            open_positions = await bot.db_manager.get_open_positions()
            restored = 0
            for row in open_positions:
                from position_manager import Position, PositionSide, PositionStatus
                pos = Position(
                    position_id=row['position_id'],
                    product_id=row['product_id'],
                    side=PositionSide(row['side']),
                    size=row['size'],
                    entry_price=row['entry_price'],
                    current_price=row['entry_price'],
                    stop_loss_price=row.get('stop_loss_price'),
                    take_profit_price=row.get('take_profit_price'),
                    status=PositionStatus.OPEN,
                    entry_time=datetime.fromisoformat(row['opened_at']),
                )
                bot.position_manager.positions[pos.position_id] = pos
                restored += 1
            bot.logger.info(f"Paper mode: restored {restored} positions from DB (anti-stacking)")
        except Exception as e:
            bot.logger.warning(f"Paper mode position restore failed: {e}")
        bot.position_manager.daily_pnl = 0.0
        bot.daily_pnl = 0.0

    # ── Startup deduplication: close duplicate/opposing positions from DB ──
    await bot._deduplicate_positions_on_startup()

    # ── Start Token Spray exit loop ──
    if bot.token_spray:
        await bot.token_spray.start_exit_loop(bot._get_spray_prices)

    # ── Start Straddle exit loops (all assets) ──
    for _s_asset, _s_engine in bot.straddle_engines.items():
        await _s_engine.start_exit_loop(bot._get_straddle_price)

    # ── Start Oracle 4H prediction loop ──
    if bot.oracle:
        try:
            bot.oracle.predict_now()
            bot.logger.info("Oracle: initial prediction completed")
        except Exception as _e:
            bot.logger.warning(f"Oracle initial prediction failed: {_e}")
        asyncio.create_task(bot.oracle.run_forever())

    # ── Start Oracle Trading Engine loop ──
    if bot.oracle_trader:
        asyncio.create_task(bot.oracle_trader.run_forever())

    # ── Prune old data to reduce DB size and memory pressure ──
    prune_old_data(bot)

    # ── One-time: reset ML evaluations to use corrected 1-bar horizon method ──
    # The old evaluation compared prediction-time vs "latest" price (variable horizon).
    # The corrected method compares prediction-time vs 1-bar-later price (fixed 5min horizon).
    _db_path = bot.db_manager.db_path if hasattr(bot.db_manager, 'db_path') else "data/renaissance_bot.db"
    _reset_flag_file = os.path.join(os.path.dirname(_db_path), '.ml_eval_1bar_reset_done')
    if not os.path.exists(_reset_flag_file):
        try:
            import sqlite3 as _sq
            _conn = _sq.connect(_db_path)
            _reset_count = _conn.execute(
                "SELECT COUNT(*) FROM ml_predictions WHERE evaluated_at IS NOT NULL"
            ).fetchone()[0]
            if _reset_count > 0:
                _conn.execute("""
                    UPDATE ml_predictions
                    SET is_correct = NULL, actual_return_1bar = NULL,
                        actual_direction = NULL, evaluated_at = NULL,
                        price_at_evaluation = NULL
                """)
                _conn.commit()
                bot.logger.info(
                    f"ML EVAL RESET: Cleared {_reset_count} old evaluations — "
                    f"will re-evaluate with corrected 1-bar horizon method"
                )
            _conn.close()
            with open(_reset_flag_file, 'w') as _f:
                _f.write('done')
        except Exception as _e:
            bot.logger.debug(f"ML eval reset check: {_e}")

    # ── Module A: Set RUNNING state ──
    if bot.state_manager:
        try:
            await bot.state_manager.aset_system_state(SystemState.RUNNING, "trading loop started")
        except Exception as e:
            bot.logger.warning(f"State manager set RUNNING state failed: {e}")

    # ── Module C: Startup alert ──
    if bot.monitoring_alert_manager:
        try:
            await bot.monitoring_alert_manager.send_system_event(
                "Bot Started",
                f"Renaissance bot starting with {len(bot.product_ids)} products, "
                f"{'paper' if paper_mode else 'live'} mode"
            )
        except Exception as e:
            bot.logger.warning(f"Startup monitoring alert failed: {e}")

    # ── Build dynamic trading universe from Binance ──
    bot.logger.info("Building dynamic trading universe from Binance...")
    await bot._build_and_apply_universe()
    if bot._universe_built:
        # Re-send startup alert with actual pair count
        if bot.monitoring_alert_manager:
            try:
                await bot.monitoring_alert_manager.send_system_event(
                    "Universe Built",
                    f"Dynamic universe: {len(bot.product_ids)} pairs from Binance"
                )
            except Exception as e:
                bot.logger.warning(f"Universe built monitoring alert failed: {e}")

        # ── Council #1: Gap-fill missing bars from Binance on startup ──
        try:
            await asyncio.wait_for(bot._gap_fill_bars_on_startup(), timeout=120)
        except asyncio.TimeoutError:
            bot.logger.warning("GAP-FILL: Timed out after 120s — continuing startup")
        except Exception as e:
            bot.logger.warning(f"GAP-FILL: Failed — {e}")

    # ── Council S6: Batch-evaluate unevaluated ML predictions on startup ──
    if bot.db_enabled:
        try:
            batch_count = await bot.db_manager.batch_evaluate_ml_outcomes()
            if batch_count > 0:
                bot.logger.info(f"Startup ML batch eval: {batch_count} predictions evaluated")
        except Exception as e:
            bot.logger.warning(f"Startup ML batch eval failed: {e}")

    # Start real-time pipeline if enabled
    if bot.real_time_pipeline.enabled:
        await bot.real_time_pipeline.start()

    # Start Ghost Runner Loop (Step 18)
    bot._track_task(bot.ghost_runner.start_ghost_loop(interval=cycle_interval * 2))

    # Start WebSocket feed for real-time data
    if bot._ws_client:
        bot._track_task(run_websocket_feed(bot))

    # Start Multi-Exchange Arbitrage Engine (runs independently alongside main loop)
    if bot.arbitrage_orchestrator:
        bot.logger.info("Launching arbitrage engine...")
        bot._track_task(run_arbitrage_engine(bot))

    # DISABLED: Old Polymarket strategies — all replaced by spread capture
    # Strategy A, Live Executor, Reversal, Simple UP — all disabled

    # ── 0x8dxd Spread Capture — favorite + underdog strategy ──
    if bot.rtds and bot.spread_capture:
        bot.logger.info("Launching RTDS WebSocket in dedicated thread (Binance + Chainlink prices)...")
        bot.rtds.start_in_thread()
        bot.logger.info(f"Launching Spread Capture Engine (0x8dxd strategy, {len(SC_ASSETS)} assets, 5m+15m)...")
        sc_task = bot._track_task(bot.spread_capture.run())
        def _sc_done(t, log=bot.logger):
            if t.cancelled():
                log.warning("Spread capture task was CANCELLED")
            elif t.exception():
                log.error(f"Spread capture task DIED: {t.exception()!r}", exc_info=t.exception())
            else:
                log.info("Spread capture task finished normally")
        sc_task.add_done_callback(_sc_done)

    # ── Module D: Start Liquidation Cascade Detector ──
    if bot.liquidation_detector:
        bot.logger.info("Launching liquidation cascade detector...")
        bot._track_task(run_liquidation_detector(bot))

    # ── Cascade Data Collector — DISABLED (backtest showed no lead-lag edge) ──
    # if bot.cascade_collector:
    #     bot.cascade_collector.start()
    #     bot.logger.info("Cascade data collector started (30s poll)")

    # ── Fast Mean Reversion Scanner (1s eval) ──
    if bot.fast_reversion_scanner:
        bot.logger.info("Launching fast mean reversion scanner (1s eval)...")
        bot._track_task(run_fast_reversion_scanner(bot))

    # ── Sub-Bar Early Exit Scanner (10s eval) ──
    if bot.sub_bar_scanner:
        bot.logger.info("Launching sub-bar scanner (10s early exit monitor)...")
        bot._track_task(run_sub_bar_scanner(bot))

    # ── Heartbeat Writer (multi-bot coordination) ──
    if bot.heartbeat_writer:
        hb_interval = bot.config.get("orchestrator", {}).get(
            "heartbeat_interval_seconds", 5
        )
        bot.logger.info(f"Launching heartbeat writer (every {hb_interval}s)...")
        bot._track_task(run_heartbeat_writer(bot, hb_interval))

    # ── Phase 2 Observation Loops ──
    if bot.medallion_portfolio_engine:
        bot.logger.info("Launching medallion portfolio drift logger (observation mode)...")
        bot._track_task(run_portfolio_drift_logger(bot))

    if bot.insurance_scanner:
        bot.logger.info("Launching insurance premium scanner (every 30 min)...")
        bot._track_task(run_insurance_scanner_loop(bot))

    if bot.daily_signal_review:
        bot.logger.info("Launching daily signal review (midnight UTC)...")
        bot._track_task(run_daily_signal_review_loop(bot))

    # ── Phase 2 Monitor Loops (BUG 6 fix) ──
    if bot.beta_monitor:
        bot.logger.info("Launching beta monitor loop (every 60 min, observation mode)...")
        bot._track_task(run_beta_monitor_loop(bot))

    if bot.sharpe_monitor_medallion:
        bot.logger.info("Launching sharpe monitor loop (every 60 min, observation mode)...")
        bot._track_task(run_sharpe_monitor_loop(bot))

    if bot.capacity_monitor:
        bot.logger.info("Launching capacity monitor loop (every 60 min, observation mode)...")
        bot._track_task(run_capacity_monitor_loop(bot))

    if bot.medallion_regime:
        bot.logger.info("Launching regime detector loop (every 5 min, observation mode)...")
        bot._track_task(run_regime_detector_loop(bot))

    # ── Doc 15: Agent weekly research loop + deployment loop ──
    if bot.agent_coordinator:
        bot.logger.info("Launching agent weekly research check loop...")
        bot._track_task(bot.agent_coordinator.run_weekly_check_loop())
        bot.logger.info("Launching agent deployment loop...")
        bot._track_task(bot.agent_coordinator.run_deployment_loop())

    # ── Gap 5 fix: Unified Telegram Reporting ──
    if bot.monitoring_alert_manager:
        bot.logger.info("Launching unified Telegram hourly report loop...")
        bot._track_task(run_telegram_report_loop(bot))

    while not bot._killed:
        try:
            # Check file-based kill switch
            check_kill_file(bot)
            if bot._killed:
                break

            # Execute trading cycle
            decision = await bot.execute_trading_cycle()

            bot.logger.info(f"{'LIVE' if not bot.coinbase_client.paper_trading else 'PAPER'} TRADE: "
                           f"{decision.action} - "
                           f"Confidence: {decision.confidence:.3f} - "
                           f"Position Size: {decision.position_size:.3f}")

            # Write heartbeat after each successful cycle
            bot._write_heartbeat()
            # Recovery module heartbeat (file-based for watchdog)
            if bot.state_manager:
                try:
                    await bot.state_manager.asend_heartbeat()
                except Exception as e:
                    bot.logger.warning(f"State manager heartbeat send failed: {e}")

            # Wait for next cycle
            await asyncio.sleep(cycle_interval)

        except KeyboardInterrupt:
            trigger_kill_switch(bot, "KeyboardInterrupt")
            break
        except Exception as e:
            bot.logger.error(f"Unexpected error in trading loop: {e}")
            await asyncio.sleep(60)

    bot.logger.info("Trading loop exited. Shutting down background tasks...")
    await shutdown(bot)
