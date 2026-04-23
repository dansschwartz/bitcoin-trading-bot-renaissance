"""
bot/signals.py — Signal generation logic extracted from RenaissanceTradingBot.

Contains:
  collect_all_data()       — Multi-source data collection per product
  generate_signals()       — Signal computation from all components
  calculate_weighted_signal() — Final weighted signal via Renaissance fusion
"""

import asyncio
import logging
import queue
import numpy as np
from datetime import datetime
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# collect_all_data
# ---------------------------------------------------------------------------

async def collect_all_data(bot: Any, product_id: str = "BTC-USD") -> Dict[str, Any]:
    """Collect data from all sources for a specific product.

    Routes to Binance for expanded-universe pairs, Coinbase for legacy pairs.
    """
    # Use Binance as primary data source when universe is built
    # Also use Binance for breakout-flagged pairs not in the original universe
    _is_breakout_pair = product_id in getattr(bot, '_breakout_scores', {})
    if (bot._universe_built and product_id in bot._pair_binance_symbols) or _is_breakout_pair:
        data = await bot._collect_from_binance(product_id)
        if data:
            return data
        # Fall through to Coinbase if Binance fails

    try:
        # Try WebSocket data first (sub-100ms latency)
        # Drain the queue but only use data matching this product_id
        latest_ws = None
        requeue = []
        while not bot._ws_queue.empty():
            try:
                msg = bot._ws_queue.get_nowait()
                msg_pid = getattr(msg, 'product_id', None) or getattr(msg, 'symbol', None) or ''
                if msg_pid == product_id:
                    latest_ws = msg
                else:
                    requeue.append(msg)
            except queue.Empty:
                break
        # Put back messages for other products
        for msg in requeue:
            try:
                bot._ws_queue.put_nowait(msg)
            except queue.Full as e:
                bot.logger.warning(f"WebSocket queue full, dropping requeued message: {e}")

        # Check WebSocket data freshness
        MAX_DATA_AGE_SECONDS = 30
        if latest_ws and hasattr(latest_ws, 'timestamp') and latest_ws.timestamp:
            data_age = (datetime.now() - latest_ws.timestamp).total_seconds()
            if data_age > MAX_DATA_AGE_SECONDS:
                bot.logger.warning(f"WebSocket data stale ({data_age:.1f}s old), falling back to REST")
                latest_ws = None

        if latest_ws and hasattr(latest_ws, 'price') and latest_ws.price > 0:
            # Use real-time WebSocket data
            ticker = {
                'price': latest_ws.price,
                'volume': latest_ws.volume,
                'bid': latest_ws.bid,
                'ask': latest_ws.ask,
                'bid_ask_spread': latest_ws.spread,
            }
            order_book_snapshot = getattr(latest_ws, 'order_book', None)
            bot.logger.debug(f"Using WebSocket data for {product_id} @ ${latest_ws.price:.2f}")
        else:
            ticker = None
            order_book_snapshot = None

        # Always fetch REST snapshot for candle/price history (needed for technicals)
        snapshot = await asyncio.to_thread(bot.market_data_provider.fetch_snapshot, product_id)
        tech = bot._get_tech(product_id)
        if snapshot.price_data:
            tech.update_price_data(snapshot.price_data)

        # Prefer WS ticker if available, otherwise use REST
        if ticker is None:
            ticker = snapshot.ticker
            order_book_snapshot = snapshot.order_book_snapshot

        technical_signals = tech.get_latest_signals()
        alt_signals = await bot.alternative_data_engine.get_alternative_signals()

        return {
            'order_book_snapshot': order_book_snapshot or snapshot.order_book_snapshot,
            'price_data': snapshot.price_data,
            'technical_signals': technical_signals,
            'alternative_signals': alt_signals,
            'ticker': ticker or snapshot.ticker,
            'product_id': product_id,
            'timestamp': datetime.now(),
            'recent_trades': getattr(latest_ws, 'recent_trades', []) if latest_ws else [],
        }
    except Exception as e:
        bot.logger.error(f"Data collection failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# generate_signals
# ---------------------------------------------------------------------------

async def generate_signals(bot: Any, market_data: Dict[str, Any]) -> Dict[str, float]:
    """Generate signals from all components."""
    # Import module-level helpers from the main module
    import renaissance_trading_bot as _rtb
    _continuous_obv_signal = _rtb._continuous_obv_signal
    _continuous_macd_signal = _rtb._continuous_macd_signal
    _continuous_rsi_signal = _rtb._continuous_rsi_signal
    _continuous_bollinger_signal = _rtb._continuous_bollinger_signal
    _convert_ws_orderbook_to_snapshot = _rtb._convert_ws_orderbook_to_snapshot

    signals = {}

    try:
        # 1. Microstructure signals (Order Flow + Order Book = 53% total weight)
        # Feed recent trades into microstructure engine for flow/VPIN analysis
        try:
            _recent = market_data.get('recent_trades', []) or []
            if _recent:
                from microstructure_engine import TradeData as MSTradeData
                for t in _recent[-20:]:  # last 20 trades
                    if isinstance(t, dict) and t.get('price', 0) > 0:
                        bot.microstructure_engine.update_trade(MSTradeData(
                            timestamp=datetime.now(),
                            price=float(t['price']),
                            size=float(t.get('size', 0)),
                            side=str(t.get('side', 'unknown')),
                            trade_id=str(t.get('trade_id', '')),
                        ))
        except Exception as _tf_err:
            bot.logger.debug(f"Trade feed to microstructure failed: {_tf_err}")

        order_book_snapshot = market_data.get('order_book_snapshot')
        # Convert WebSocket dict format to OrderBookSnapshot if needed
        if order_book_snapshot and isinstance(order_book_snapshot, dict) and ('bids' in order_book_snapshot or 'asks' in order_book_snapshot):
            try:
                current_px = float(market_data.get('current_price', 0) or market_data.get('ticker', {}).get('price', 0))
                order_book_snapshot = _convert_ws_orderbook_to_snapshot(order_book_snapshot, current_px)
                market_data['order_book_snapshot'] = order_book_snapshot  # update for downstream
            except Exception as _conv_err:
                bot.logger.debug(f"Order book conversion failed: {_conv_err}")
                order_book_snapshot = None
        if order_book_snapshot:
            microstructure_signal = bot.microstructure_engine.update_order_book(order_book_snapshot)
            signals['order_flow'] = microstructure_signal.large_trade_flow
            if bot.signal_weights.get('order_book', 0) > 0:
                signals['order_book'] = microstructure_signal.order_book_imbalance
        else:
            signals['order_flow'] = 0.0

        # 2. Technical indicators (38% total weight)
        _pid = market_data.get('product_id', 'BTC-USD')
        technical_signal = market_data.get('technical_signals') or bot._get_tech(_pid).get_latest_signals()
        if technical_signal:
            signals['volume'] = _continuous_obv_signal(technical_signal.obv_momentum)
            signals['macd'] = _continuous_macd_signal(technical_signal.quick_macd)
            signals['rsi'] = _continuous_rsi_signal(technical_signal.fast_rsi)
            signals['bollinger'] = _continuous_bollinger_signal(technical_signal.dynamic_bollinger)
        else:
            signals['volume'] = 0.0
            signals['macd'] = 0.0
            signals['rsi'] = 0.0
            signals['bollinger'] = 0.0

            # 3. Alternative data signals (4.5% total weight)
        if market_data.get('alternative_signals'):
            alt_signal = market_data['alternative_signals']

            # Fetch whale signal
            whale_data = await bot.whale_monitor.get_whale_signals()
            whale_pressure = float(whale_data.get("whale_pressure", 0.0))

            # Deep NLP Reasoning (New)
            news_text = " ".join([str(n) for n in market_data.get('news', [])])
            if news_text.strip():
                nlp_result = await bot.nlp_bridge.analyze_sentiment_with_reasoning(news_text)
                nlp_sentiment = float(nlp_result.get('sentiment', 0.0))
                market_data['nlp_reasoning'] = nlp_result.get('reasoning', 'No deep context')
            else:
                nlp_sentiment = 0.0
                market_data['nlp_reasoning'] = 'No news data available'

            # Combine all alternative signals into one composite score
            # 20% Reddit, 15% News, 15% Twitter, 15% Fear/Greed, 10% Whale, 25% Deep NLP
            alternative_composite = (
                float(alt_signal.reddit_sentiment or 0.0) * 0.20 +
                float(alt_signal.news_sentiment or 0.0) * 0.15 +
                float(alt_signal.social_sentiment or 0.0) * 0.15 +
                float(alt_signal.market_psychology or 0.0) * 0.15 +
                whale_pressure * 0.10 +
                nlp_sentiment * 0.25
            )
            signals['alternative'] = float(alternative_composite)
            market_data['whale_signals'] = whale_data  # Pass along for dashboard
        else:
            signals['alternative'] = 0.0

        # Council #2 diagnostic: track zero-signal rate
        _nonzero = sum(1 for v in signals.values() if abs(float(v)) > 1e-6)
        _total = len(signals)
        if _nonzero == 0:
            bot.logger.debug(
                f"ZERO-SIGNAL: {_pid} all {_total} signals zero "
                f"(ob_snap={'yes' if order_book_snapshot else 'NO'}, "
                f"tech={'yes' if technical_signal else 'NO'})"
            )
        bot.logger.info(f"Generated signals: {_nonzero}/{_total} active for {_pid}")

        # 4. Institutional & High-Dimensional Intelligence (Step 16+)
        # Each signal group is isolated so one failure doesn't kill the rest
        p_id = market_data.get('product_id', 'Unknown')
        df = bot._get_tech(p_id)._to_dataframe()
        cur_price = market_data.get('current_price', 0.0)

        # Volume Profile
        try:
            if not df.empty and cur_price > 0:
                profile = bot.volume_profile_engine.calculate_profile(df)
                if profile:
                    vp_signal = bot.volume_profile_engine.get_profile_signal(cur_price, profile)
                    signals['volume_profile'] = vp_signal['signal']
                    bot._last_vp_status[p_id] = vp_signal['status']
        except Exception as e:
            bot.logger.debug(f"Volume profile signal failed: {e}")

        # Statistical Arbitrage (BTC vs ETH)
        try:
            if p_id in ["BTC-USD", "ETH-USD"]:
                other_id = "ETH-USD" if p_id == "BTC-USD" else "BTC-USD"
                sa_signal = bot.stat_arb_engine.calculate_pair_signal(p_id, other_id)
                signals['stat_arb'] = sa_signal.get('signal', 0.0)
        except Exception as e:
            bot.logger.debug(f"Stat arb signal failed: {e}")

        # Feed price to sub-bar scanner for early exit monitoring
        if bot.sub_bar_scanner and cur_price > 0:
            try:
                bot.sub_bar_scanner.update_price(p_id, cur_price)
            except Exception as e:
                bot.logger.warning(f"Sub-bar scanner price update failed for {p_id}: {e}")

        # Cross-Asset Lead-Lag Alpha (Step 16)
        try:
            if len(bot.product_ids) > 1 and cur_price > 0:
                bot.correlation_engine.update_price(p_id, cur_price)
                base = bot.product_ids[0]
                target = p_id
                if base != target:
                    ll_data = bot.correlation_engine.calculate_lead_lag(base, target)
                    signals['lead_lag'] = ll_data.get('directional_signal', 0.0)
                    market_data['lead_lag_alpha'] = ll_data
        except Exception as e:
            bot.logger.debug(f"Lead-lag signal failed: {e}")

        # Fractal Intelligence (DTW)
        try:
            if not df.empty and len(df) >= 10:
                prices = df['close'].values
                fractal_result = bot.fractal_intelligence.find_best_match(prices)
                signals['fractal'] = fractal_result['signal']
                market_data['fractal_intelligence'] = fractal_result
        except Exception as e:
            bot.logger.debug(f"Fractal signal failed: {e}")

        # Market Entropy (Shannon/ApEn)
        try:
            if not df.empty and len(df) >= 20:
                prices = df['close'].values
                entropy_result = bot.market_entropy.calculate_entropy(prices)
                signals['entropy'] = 0.5 * (entropy_result['predictability'] - 0.5)
                market_data['market_entropy'] = entropy_result
        except Exception as e:
            bot.logger.debug(f"Entropy signal failed: {e}")

        # ML Feature Pipeline & Real-Time Intelligence (Step 12/16 Bridge)
        try:
            if not df.empty and len(df) >= 18 and bot.real_time_pipeline.enabled:
                _cross = market_data.get('_cross_data')
                _pair = market_data.get('_pair_name')
                _deriv = market_data.get('_derivatives_data')
                rt_result = await bot.real_time_pipeline.processor.process_all_models(
                    {'price_df': df},
                    cross_data=_cross, pair_name=_pair,
                    derivatives_data=_deriv,
                    macro_data=bot._macro_cache.get(),
                )
                market_data['real_time_predictions'] = rt_result
                _ml_scale = bot.config.get("ml_signal_scale", 10.0)
                _pair_key = p_id or 'unknown'
                # MetaEnsemble is the key from real_time_pipeline name_map
                _ens_val = rt_result.get('MetaEnsemble') or rt_result.get('Ensemble') or 0.0
                if _ens_val:
                    # Council #12: Z-score rescaling before fusion
                    _ens_rescaled = bot._ml_zscore_rescale(_pair_key + '_ens', float(_ens_val))
                    signals['ml_ensemble'] = float(np.clip(_ens_rescaled, -1.0, 1.0))
                if 'CNN' in rt_result and bot.signal_weights.get('ml_cnn', 0) > 0:
                    _cnn_rescaled = bot._ml_zscore_rescale(_pair_key + '_cnn', float(rt_result['CNN']))
                    signals['ml_cnn'] = float(np.clip(_cnn_rescaled, -1.0, 1.0))
                # Crash-regime LightGBM signal (multi-asset: uses 2bar as primary)
                _crash_val = rt_result.get('CrashRegime_2bar') or rt_result.get('CrashRegime')
                if _crash_val is not None:
                    _crash_rescaled = bot._ml_zscore_rescale(_pair_key + '_crash', float(_crash_val))
                    signals['crash_regime'] = float(np.clip(_crash_rescaled, -1.0, 1.0))
                    bot.logger.info(
                        f"CRASH MODEL [{p_id}]: raw={float(_crash_val):.4f}, "
                        f"rescaled={_crash_rescaled:.4f}, signal={signals['crash_regime']:.4f}"
                    )
                bot.logger.info(
                    f"ML SIGNALS: ensemble={signals.get('ml_ensemble', 0):.4f}, "
                    f"cnn={signals.get('ml_cnn', 0):.4f} (raw: E={_ens_val:.4f}, "
                    f"C={rt_result.get('CNN', 0):.4f}, zscore_rescale=on)"
                )
                # Extract volatility prediction for dead-zone filter + position tracking
                _vol_pred = rt_result.get('_volatility')
                if _vol_pred and isinstance(_vol_pred, dict):
                    market_data['volatility_prediction'] = _vol_pred
                    bot.logger.info(
                        f"VOL PRED [{p_id}]: regime={_vol_pred.get('vol_regime', '?')} "
                        f"mag={_vol_pred.get('predicted_magnitude_bps', 0):.1f}bps "
                        f"mult={_vol_pred.get('vol_multiplier', 1.0):.1f}"
                    )
        except Exception as e:
            bot.logger.warning(f"ML RT pipeline failed: {e}")

        # Quantum Oscillator (QHO)
        try:
            if not df.empty and len(df) >= 30:
                prices = df['close'].values
                quantum_result = bot.quantum_oscillator.calculate_quantum_levels(prices)
                signals['quantum'] = quantum_result['signal']
                market_data['quantum_oscillator'] = quantum_result
        except Exception as e:
            bot.logger.debug(f"Quantum signal failed: {e}")

        # Correlation Network Divergence Signal (skip if weight is 0 — Council S4)
        if bot.signal_weights.get('correlation_divergence', 0) > 0:
            try:
                if bot.correlation_network.enabled:
                    div_signal = bot.correlation_network.get_correlation_divergence_signal(p_id)
                    signals['correlation_divergence'] = div_signal
            except Exception as e:
                bot.logger.debug(f"Correlation divergence signal failed: {e}")

        # GARCH Volatility Signal (vol_ratio as directional bias)
        try:
            if bot.garch_engine.is_available:
                forecast = bot.garch_engine.forecast_volatility(p_id)
                vol_ratio = forecast.get('vol_ratio', 1.0)
                if vol_ratio < 0.8:
                    signals['garch_vol'] = min((1.0 - vol_ratio) * 0.5, 0.5)
                elif vol_ratio > 1.2:
                    signals['garch_vol'] = max(-(vol_ratio - 1.0) * 0.5, -0.5)
                else:
                    signals['garch_vol'] = 0.0
                market_data['garch_forecast'] = forecast
        except Exception as e:
            bot.logger.debug(f"GARCH vol signal failed: {e}")

        return signals

    except Exception as e:
        bot.logger.error(f"Signal generation failed: {e}")
        return {key: 0.0 for key in bot.signal_weights.keys()}


# ---------------------------------------------------------------------------
# calculate_weighted_signal
# ---------------------------------------------------------------------------

def calculate_weighted_signal(bot: Any, signals: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Calculate final weighted signal using Renaissance weights (Institutional hardening)."""

    # We redirect to the new ML-enhanced fusion if possible, or use standard
    # For backward compatibility with tests/backtests that call this directly
    ml_package = signals.get('ml_package')  # Might be injected in some contexts

    # PURE SCALAR TYPE GUARD for all signal inputs
    processed_signals = {}
    for k, v in signals.items():
        if k == 'ml_package':
            processed_signals[k] = v
            continue
        try:
            processed_signals[k] = bot._force_float(v)
        except Exception:
            processed_signals[k] = 0.0

    weighted_signal, confidence, fusion_metadata = bot.signal_fusion.fuse_signals_with_ml(
        processed_signals, bot.signal_weights, ml_package
    )

    # Ensure contributions are also hardened
    contributions = fusion_metadata.get('contributions', {})
    hardened_contribs = {}
    for k, v in contributions.items():
        try:
            hardened_contribs[k] = bot._force_float(v)
        except Exception:
            hardened_contribs[k] = 0.0

    return float(bot._force_float(weighted_signal)), hardened_contribs
