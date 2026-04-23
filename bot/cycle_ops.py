"""
bot/cycle_ops.py — Large trading-cycle helper blocks extracted from execute_trading_cycle().

Functions accept `bot` as first argument (the RenaissanceTradingBot instance).
All functions are called from within the per-pair or per-cycle sections of
execute_trading_cycle() and share context via parameters.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from renaissance_trading_bot import RenaissanceTradingBot

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Drawdown Controls & Exposure Monitor
# ──────────────────────────────────────────────

async def apply_drawdown_controls(bot: "RenaissanceTradingBot", account_balance: float) -> None:
    """Progressive drawdown circuit breaker and continuous exposure monitor."""

    # ── Drawdown Circuit Breaker ──
    bot._drawdown_size_scalar = 1.0
    bot._drawdown_exits_only = False

    if bot._current_drawdown_pct >= 0.15:
        bot.logger.warning("CIRCUIT BREAKER: 15% drawdown — closing all positions")
        try:
            with bot.position_manager._lock:
                all_positions = list(bot.position_manager.positions.values())
            for pos in all_positions:
                ok, _ = bot.position_manager.close_position(pos.position_id, reason="Circuit breaker: 15% drawdown")
                if ok:
                    _cpx = await bot._resolve_close_price(pos)
                    _side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                    _rpnl = bot._compute_realized_pnl(pos.entry_price, _cpx, pos.size, _side)
                    bot._track_task(bot.db_manager.close_position_record(
                        pos.position_id,
                        close_price=float(_cpx),
                        realized_pnl=float(_rpnl),
                        exit_reason="circuit_breaker",
                    ))
        except Exception as cb_err:
            bot.logger.error(f"Circuit breaker close-all failed: {cb_err}")
        bot._drawdown_exits_only = True
    elif bot._current_drawdown_pct >= 0.10:
        bot.logger.warning("CIRCUIT BREAKER: 10% drawdown — exits only mode")
        bot._drawdown_exits_only = True
    elif bot._current_drawdown_pct >= 0.05:
        bot._drawdown_size_scalar = 0.5
        bot.logger.info(f"DRAWDOWN SCALING: {bot._current_drawdown_pct:.1%} — 50% position sizes")

    # ── Continuous Exposure Monitor ──
    try:
        total_exposure = bot.position_manager._calculate_total_exposure()
        max_exposure = account_balance * 0.50
        if total_exposure > max_exposure:
            bot.logger.warning(
                f"EXPOSURE LIMIT: ${total_exposure:,.2f} > ${max_exposure:,.2f} — force-closing worst position"
            )
            with bot.position_manager._lock:
                open_positions = list(bot.position_manager.positions.values())
            if open_positions:
                worst_pos = None
                worst_pnl = float('inf')
                for pos in open_positions:
                    if hasattr(pos, 'unrealized_pnl'):
                        if pos.unrealized_pnl < worst_pnl:
                            worst_pnl = pos.unrealized_pnl
                            worst_pos = pos
                    elif hasattr(pos, 'entry_price') and pos.entry_price > 0:
                        worst_pos = worst_pos or pos
                if worst_pos:
                    ok, _ = bot.position_manager.close_position(
                        worst_pos.position_id, reason="Exposure limit exceeded"
                    )
                    if ok:
                        _cpx = await bot._resolve_close_price(worst_pos)
                        _side = worst_pos.side.value if hasattr(worst_pos.side, 'value') else str(worst_pos.side)
                        _rpnl = bot._compute_realized_pnl(
                            worst_pos.entry_price, _cpx, worst_pos.size, _side
                        )
                        bot._track_task(bot.db_manager.close_position_record(
                            worst_pos.position_id,
                            close_price=float(_cpx),
                            realized_pnl=float(_rpnl),
                            exit_reason="exposure_limit",
                        ))
                    bot.logger.info(f"EXPOSURE CLOSE: {worst_pos.position_id}")
    except Exception as exp_err:
        bot.logger.debug(f"Exposure monitor error: {exp_err}")


# ──────────────────────────────────────────────
#  Candle History Preload (first cycle)
# ──────────────────────────────────────────────

async def preload_candle_history(bot: "RenaissanceTradingBot") -> None:
    """Preload 300 candles per pair to eliminate cold-start. Called once on first cycle."""
    import asyncio
    from binance_spot_provider import to_binance_symbol

    preload_pairs = list(bot.product_ids)
    for pid in preload_pairs:
        try:
            bsym = bot._pair_binance_symbols.get(pid, to_binance_symbol(pid))
            raw_candles = await bot.binance_spot.fetch_candles(bsym, '5m', 300)
            if not raw_candles:
                raw_candles_cb = await asyncio.to_thread(
                    bot.market_data_provider.fetch_candle_history, pid
                )
                if raw_candles_cb:
                    pid_tech = bot._get_tech(pid)
                    for candle in raw_candles_cb:
                        pid_tech.update_price_data(candle)
                        if candle.close > 0:
                            bot.garch_engine.update_returns(pid, candle.close)
                            bot.stat_arb_engine.update_price(pid, candle.close)
                            bot.correlation_network.update_price(pid, candle.close)
                            bot.mean_reversion_engine.update_price(pid, candle.close)
                            bot.correlation_engine.update_price(pid, candle.close)
                    bot.logger.info(f"Preloaded {len(raw_candles_cb)} candles for {pid} (Coinbase)")
                continue

            from enhanced_technical_indicators import PriceData
            pid_tech = bot._get_tech(pid)
            for c in raw_candles:
                pd_obj = PriceData(
                    timestamp=datetime.utcfromtimestamp(c['timestamp']),
                    open=c['open'], high=c['high'],
                    low=c['low'], close=c['close'],
                    volume=c['volume'],
                )
                pid_tech.update_price_data(pd_obj)
                if c['close'] > 0:
                    bot.garch_engine.update_returns(pid, c['close'])
                    bot.stat_arb_engine.update_price(pid, c['close'])
                    bot.correlation_network.update_price(pid, c['close'])
                    bot.mean_reversion_engine.update_price(pid, c['close'])
                    bot.correlation_engine.update_price(pid, c['close'])
            bot.logger.info(
                f"Preloaded {len(raw_candles)} candles for {pid} (Binance) — "
                f"price_history={len(pid_tech.price_history)}"
            )
            if bot.garch_engine.should_refit(pid):
                bot.garch_engine.fit_model(pid)
        except Exception as e:
            bot.logger.warning(f"History preload failed for {pid}: {e}")


# ──────────────────────────────────────────────
#  Hierarchical Regime Classification
# ──────────────────────────────────────────────

def classify_hierarchical_regime(
    bot: "RenaissanceTradingBot",
    cross_data: Dict[str, Any],
    market_data_all: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run macro/crypto/micro regime hierarchy. Returns state dict for dashboard."""
    _cycle_regime_state: Dict[str, Any] = {}
    if not (bot._macro_regime_detector and bot._crypto_regime_detector and bot._model_router):
        return _cycle_regime_state
    try:
        _macro_data = bot._macro_cache.get()
        _macro_snap = bot._macro_regime_detector.classify(_macro_data)
        _macro_snap.confidence = max(_macro_snap.confidence, bot._macro_regime_detector.current_confidence)

        _btc_df = None
        for _btc_key in ['BTCUSDT', 'BTC-USDT', 'BTC/USDT']:
            if _btc_key in cross_data and len(cross_data[_btc_key]) >= 50:
                _btc_df = cross_data[_btc_key]
                break
        if _btc_df is None:
            _btc_df = bot._load_price_df_from_db('BTCUSDT', limit=300)

        _btc_market = market_data_all.get('BTCUSDT') or market_data_all.get('BTC-USDT', {})
        _btc_deriv = _btc_market.get('_derivatives_data') if _btc_market else None
        _crypto_snap = bot._crypto_regime_detector.classify(_btc_df, _btc_deriv)

        _micro_label = "*"
        if bot.regime_overlay and bot.regime_overlay.enabled:
            _hmm_lbl = bot.regime_overlay.get_hmm_regime_label()
            if _hmm_lbl and _hmm_lbl != "unknown":
                _micro_label = _hmm_lbl

        _route_config = bot._model_router.route(
            bot._macro_regime_detector.current_regime,
            bot._crypto_regime_detector.current_regime,
            _micro_label,
        )

        _cycle_regime_state = {
            'macro': bot._macro_regime_detector.get_state(),
            'crypto': bot._crypto_regime_detector.get_state(),
            'router': bot._model_router.get_state(),
        }

        bot.logger.info(
            f"REGIME HIERARCHY: "
            f"Macro={bot._macro_regime_detector.current_regime.value} "
            f"Crypto={bot._crypto_regime_detector.current_regime.value} "
            f"Micro={_micro_label} "
            f"-> {_route_config.model_name} "
            f"(kelly={_route_config.kelly_multiplier:.1f}, obs_mode=True)"
        )

        if hasattr(bot, 'event_emitter') and bot.event_emitter:
            bot.event_emitter.emit('regime_hierarchy', _cycle_regime_state)

    except Exception as _regime_err:
        bot.logger.debug(f"Hierarchical regime classification error: {_regime_err}")

    return _cycle_regime_state


# ──────────────────────────────────────────────
#  Advanced Signal Injection
# ──────────────────────────────────────────────

async def inject_advanced_signals(
    bot: "RenaissanceTradingBot",
    product_id: str,
    signals: Dict[str, float],
    market_data: Dict[str, Any],
    current_price: float,
) -> None:
    """Inject microstructure, liquidation, fast-MR, multi-exchange, and medallion analog signals."""

    # Microstructure (Module F)
    if bot.signal_aggregator:
        try:
            ob_snap = market_data.get('order_book_snapshot')
            bids_list, asks_list = [], []
            if ob_snap is not None:
                if hasattr(ob_snap, 'bids') and hasattr(ob_snap, 'asks'):
                    bids_list = [(float(l.price), float(l.size)) for l in ob_snap.bids[:20]
                                 if hasattr(l, 'price')]
                    asks_list = [(float(l.price), float(l.size)) for l in ob_snap.asks[:20]
                                 if hasattr(l, 'price')]
                elif isinstance(ob_snap, dict):
                    bids_raw = ob_snap.get('bids', {})
                    asks_raw = ob_snap.get('asks', {})
                    if isinstance(bids_raw, dict):
                        bids_list = sorted(
                            [(float(p), float(s)) for p, s in bids_raw.items()],
                            reverse=True
                        )[:20]
                        asks_list = sorted(
                            [(float(p), float(s)) for p, s in asks_raw.items()]
                        )[:20]

                if bids_list and asks_list:
                    bot.signal_aggregator.update_book(bids_list, asks_list)
                    micro_entry = bot.signal_aggregator.get_signal_dict_entry()
                    micro_score = bot._force_float(micro_entry.get('microstructure_advanced', 0.0))
                    if abs(micro_score) > 0.001:
                        signals['microstructure_advanced'] = micro_score
        except Exception as _micro_err:
            bot.logger.debug(f"Advanced microstructure signal failed: {_micro_err}")

    # Liquidation Cascade (Module D)
    if bot.liquidation_detector:
        try:
            binance_sym = product_id.replace("-USD", "USDT").replace("-", "")
            current_risk = await bot.liquidation_detector.get_current_risk()
            sym_risk = current_risk.get(binance_sym, {})
            risk_score = float(sym_risk.get('risk_score', 0.0))
            if risk_score > 0.3:
                direction = sym_risk.get('direction', 'long_liquidation')
                direction_mult = 1.0 if direction == "short_squeeze" else -1.0
                signals['liquidation_cascade'] = bot._force_float(direction_mult * risk_score)
                market_data['cascade_risk'] = {
                    'symbol': binance_sym,
                    'direction': direction,
                    'risk_score': risk_score,
                    'funding_rate': sym_risk.get('funding_rate', 0.0),
                }
        except Exception as _liq_err:
            bot.logger.debug(f"Liquidation cascade signal failed: {_liq_err}")

    # Fast Mean Reversion
    if bot.fast_reversion_scanner:
        try:
            fmr_signal = bot.fast_reversion_scanner.get_latest_signal(product_id)
            if fmr_signal and fmr_signal.confidence > 0.52:
                direction_mult = 1.0 if fmr_signal.direction == "long" else -1.0
                signals['fast_mean_reversion'] = bot._force_float(direction_mult * fmr_signal.confidence)
        except Exception as _fmr_err:
            bot.logger.debug(f"Fast mean reversion signal failed: {_fmr_err}")

    # Multi-Exchange Signal Bridge
    if bot.multi_exchange_bridge:
        try:
            cb_bid_vol, cb_ask_vol, cb_bid, cb_ask = 0.0, 0.0, 0.0, 0.0
            ob_snap = market_data.get('order_book_snapshot')
            if ob_snap is not None:
                if hasattr(ob_snap, 'bids') and ob_snap.bids:
                    cb_bid = float(ob_snap.bids[0].price) if hasattr(ob_snap.bids[0], 'price') else 0.0
                    cb_bid_vol = sum(float(lv.size) for lv in ob_snap.bids[:10] if hasattr(lv, 'size'))
                if hasattr(ob_snap, 'asks') and ob_snap.asks:
                    cb_ask = float(ob_snap.asks[0].price) if hasattr(ob_snap.asks[0], 'price') else 0.0
                    cb_ask_vol = sum(float(lv.size) for lv in ob_snap.asks[:10] if hasattr(lv, 'size'))

            me_signals = bot.multi_exchange_bridge.get_signals(
                product_id=product_id,
                coinbase_bid=cb_bid,
                coinbase_ask=cb_ask,
                coinbase_bid_vol=cb_bid_vol,
                coinbase_ask_vol=cb_ask_vol,
            )
            me_cfg = bot.config.get("multi_exchange_signals", {})
            me_weights = me_cfg.get("weights", {})
            me_composite = 0.0
            me_weight_sum = 0.0
            for sig_name, sig_val in me_signals.items():
                w = me_weights.get(sig_name, 0.025)
                me_composite += sig_val * w
                me_weight_sum += w
            if me_weight_sum > 0:
                me_composite /= me_weight_sum
            signals['multi_exchange'] = bot._force_float(me_composite)
            if abs(me_composite) > 0.01:
                bot.logger.info(
                    f"MULTI-EXCHANGE [{product_id}]: composite={me_composite:+.4f} "
                    f"momentum={me_signals['cross_exchange_momentum']:+.4f} "
                    f"dispersion={me_signals['price_dispersion']:+.4f} "
                    f"imbalance={me_signals['aggregated_book_imbalance']:+.4f} "
                    f"funding={me_signals['funding_rate_signal']:+.4f}"
                )
        except Exception as _me_err:
            bot.logger.debug(f"Multi-exchange bridge error: {_me_err}")

    # Medallion Signal Analogs
    if bot.medallion_analogs:
        try:
            _tech = bot._get_tech(product_id)
            price_hist = list(_tech.price_history) if hasattr(_tech, 'price_history') else []
            _funding = 0.0
            if hasattr(bot, 'multi_exchange_bridge') and bot.multi_exchange_bridge:
                usdt_sym = product_id.split("-")[0] + "/USDT"
                _funding = bot.multi_exchange_bridge._funding_cache.get(usdt_sym, 0.0)

            analog_signals = bot.medallion_analogs.get_signals(
                product_id=product_id,
                current_price=current_price,
                price_history=price_hist,
                funding_rate=_funding,
            )
            analog_weights = bot.config.get('medallion_analogs', {}).get('weights', {})
            analog_composite = 0.0
            analog_w_sum = 0.0
            for sig_name, sig_val in analog_signals.items():
                w = analog_weights.get(sig_name, 0.01)
                analog_composite += sig_val * w
                analog_w_sum += w
            if analog_w_sum > 0:
                analog_composite /= analog_w_sum
            if abs(analog_composite) > 0.001:
                signals['medallion_analog'] = bot._force_float(analog_composite)
        except Exception as _ma_err:
            bot.logger.debug(f"Medallion analogs error: {_ma_err}")


# ──────────────────────────────────────────────
#  Polymarket Bridge & Scanner
# ──────────────────────────────────────────────

async def handle_polymarket_signals(
    bot: "RenaissanceTradingBot",
    product_id: str,
    weighted_signal: float,
    ml_package: Any,
    market_data: Dict[str, Any],
) -> None:
    """Emit BTC signal to Polymarket bridge and run periodic scanner."""
    if product_id != 'BTC-USD':
        return

    try:
        _pm_model_preds: Dict[str, float] = {}
        if ml_package and ml_package.ml_predictions:
            for mp in ml_package.ml_predictions:
                if isinstance(mp, (tuple, list)) and len(mp) >= 2:
                    _pm_model_preds[str(mp[0])] = float(mp[1]) if isinstance(mp[1], (int, float)) else 0.0
                elif isinstance(mp, dict):
                    _pm_model_preds[mp.get('name', 'unknown')] = float(mp.get('prediction', 0.0))

        _pm_agreement = 0.5
        if _pm_model_preds:
            _pm_signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in _pm_model_preds.values()]
            _pm_nonzero = [s for s in _pm_signs if s != 0]
            if _pm_nonzero:
                _pm_agreement = max(_pm_nonzero.count(1), _pm_nonzero.count(-1)) / len(_pm_nonzero)

        _pm_regime = bot.regime_overlay.get_hmm_regime_label() if bot.regime_overlay.enabled else "unknown"

        _pm_btc_breakout = 0.0
        _pm_bo = bot._breakout_scores.get('BTC-USD')
        if _pm_bo:
            _pm_btc_breakout = _pm_bo.breakout_score

        _pm_price = float(market_data.get('ticker', {}).get('price', 0.0))

        bot.polymarket_bridge.generate_signal(
            prediction=weighted_signal,
            agreement=_pm_agreement,
            regime=_pm_regime,
            breakout_score=_pm_btc_breakout,
            btc_price=_pm_price,
            model_confidences=_pm_model_preds,
            scanner_opportunities=bot._latest_scanner_opportunities,
        )
    except Exception as _pm_err:
        bot.logger.debug(f"Polymarket bridge error: {_pm_err}")

    # Scanner — every 5 min
    try:
        _scan_due = bot._last_poly_scan is None or \
            (datetime.now() - bot._last_poly_scan).total_seconds() >= 300
        if _scan_due:
            _pm_agreement_val = 0.5
            _pm_regime_val = bot.regime_overlay.get_hmm_regime_label() if bot.regime_overlay.enabled else "unknown"
            _scan_preds: Dict[str, float] = {}
            _scan_prices: Dict[str, float] = {}
            if ml_package and ml_package.ensemble_score != 0.0:
                _scan_preds['BTC'] = float(weighted_signal)
            _pm_price = float(market_data.get('ticker', {}).get('price', 0.0))
            if _pm_price and _pm_price > 0:
                _scan_prices['BTC'] = float(_pm_price)
            if hasattr(bot, '_last_prices'):
                for _pid, _px in bot._last_prices.items():
                    if _px > 0 and '-' in _pid:
                        _asset_key = _pid.split('-')[0]
                        _scan_prices[_asset_key] = float(_px)
            _scan_opps = await bot.polymarket_scanner.scan(
                ml_predictions=_scan_preds,
                agreement=_pm_agreement_val,
                regime=_pm_regime_val,
                current_prices=_scan_prices,
            )
            bot._last_poly_scan = datetime.now()
            if _scan_opps:
                bot._latest_scanner_opportunities = [
                    {
                        "condition_id": o.market.condition_id,
                        "question": o.market.question[:120],
                        "market_type": o.market.market_type,
                        "asset": o.market.asset,
                        "direction": o.direction,
                        "edge": o.edge,
                        "confidence": o.confidence,
                        "our_probability": o.our_probability,
                        "yes_price": o.market.yes_price,
                        "target_price": o.market.target_price,
                        "timeframe_minutes": o.market.timeframe_minutes,
                        "source": o.source,
                    }
                    for o in _scan_opps[:10]
                ]
            else:
                bot._latest_scanner_opportunities = []
    except Exception as _ps_err:
        bot.logger.debug(f"Polymarket scanner error: {_ps_err}")


# ──────────────────────────────────────────────
#  Strategy A ML Prediction Cache
# ──────────────────────────────────────────────

def cache_strategy_a_predictions(
    bot: "RenaissanceTradingBot",
    product_id: str,
    market_data: Dict[str, Any],
    price_df: Any,
    ml_package: Any,
) -> None:
    """Accumulate per-pair crash model predictions for Strategy A routing."""
    if not bot.polymarket_executor:
        return
    if not hasattr(bot, '_sa_ml_cache'):
        bot._sa_ml_cache = {}

    _rt_preds = market_data.get('real_time_predictions', {})
    _crash_2bar = _rt_preds.get('CrashRegime_2bar') or _rt_preds.get('CrashRegime')
    _crash_1bar = _rt_preds.get('CrashRegime_1bar')

    # Run crash inference directly if RT pipeline didn't produce it
    if _crash_2bar is None:
        try:
            _rtp = bot.real_time_pipeline.processor
            if (hasattr(_rtp, '_crash_loader') and _rtp._crash_loader
                    and hasattr(_rtp, '_crash_builder') and _rtp._crash_builder):
                _pm_asset = product_id.split('-')[0] if '-' in product_id else None
                if _pm_asset in ('BTC', 'ETH', 'SOL', 'XRP', 'DOGE'):
                    _pm_cross = market_data.get('_cross_data')
                    _pm_deriv = market_data.get('_derivatives_data')
                    _pm_macro = bot._macro_cache.get() if hasattr(bot, '_macro_cache') else None
                    for _hz in ('2bar', '1bar'):
                        _cm, _cmeta = _rtp._crash_loader.get_model(_pm_asset, _hz)
                        if _cm is None:
                            continue
                        try:
                            _pm_cross_df = None
                            _pm_lead = _rtp._crash_loader.get_cross_asset(_pm_asset)
                            if _pm_cross and _pm_lead:
                                for _ck in _pm_cross:
                                    if _pm_lead in str(_ck).upper():
                                        _pm_cross_df = _pm_cross[_ck]
                                        break
                            _pm_feats = _rtp._crash_builder.build(
                                asset=_pm_asset,
                                price_df=price_df,
                                cross_price_df=_pm_cross_df,
                                derivatives_data=_pm_deriv,
                                macro_data=_pm_macro,
                            )
                            if _pm_feats is not None:
                                _pm_pred, _pm_conf, _pm_src = _rtp._crash_loader.predict_for_asset(
                                    _pm_asset, _hz, _pm_feats
                                )
                                if _hz == '2bar':
                                    _crash_2bar = _pm_pred
                                else:
                                    _crash_1bar = _pm_pred
                                bot.logger.info(
                                    f"Crash {_pm_asset}_{_hz} (Polymarket direct): "
                                    f"pred={_pm_pred:.4f} conf={_pm_conf:.4f} src={_pm_src}"
                                )
                        except Exception as _pm_err:
                            bot.logger.debug(f"Crash {_pm_asset}_{_hz} Polymarket failed: {_pm_err}")
        except Exception as _crash_infer_err:
            bot.logger.debug(f"Crash direct inference failed: {_crash_infer_err}")

    if _crash_2bar is not None:
        _crash_prob_2 = (float(_crash_2bar) + 1.0) / 2.0
        _dir_conf_2 = max(_crash_prob_2, 1.0 - _crash_prob_2)
        _model_acc_2 = 0.50
        try:
            _rtp2 = bot.real_time_pipeline.processor
            if hasattr(_rtp2, '_crash_loader') and _rtp2._crash_loader:
                _pm_a2 = product_id.split('-')[0] if '-' in product_id else 'BTC'
                _key2 = (_pm_a2.upper(), '2bar')
                _entry2 = _rtp2._crash_loader._models.get(_key2)
                if _entry2:
                    _model_acc_2 = _entry2.accuracy
        except Exception:
            pass
        _conf_2 = max(_dir_conf_2, _model_acc_2) * 100.0
        _entry = {
            "prediction": float(_crash_2bar),
            "agreement": abs(_crash_prob_2 - 0.5) * 2.0,
            "confidence": _conf_2,
            "source": "crash_lgbm_2bar",
        }
        if _crash_1bar is not None:
            _crash_prob_1 = (float(_crash_1bar) + 1.0) / 2.0
            _dir_conf_1 = max(_crash_prob_1, 1.0 - _crash_prob_1)
            _model_acc_1 = 0.50
            try:
                if hasattr(_rtp2, '_crash_loader') and _rtp2._crash_loader:
                    _key1 = (_pm_a2.upper(), '1bar')
                    _entry1 = _rtp2._crash_loader._models.get(_key1)
                    if _entry1:
                        _model_acc_1 = _entry1.accuracy
            except Exception:
                pass
            _conf_1 = max(_dir_conf_1, _model_acc_1) * 100.0
            _entry["prediction_1bar"] = float(_crash_1bar)
            _entry["confidence_1bar"] = _conf_1

        bot._sa_ml_cache[product_id] = _entry
    elif ml_package and hasattr(ml_package, 'ensemble_score'):
        bot._sa_ml_cache[product_id] = {
            "prediction": float(ml_package.ensemble_score),
            "agreement": float(ml_package.confidence_score),
            "confidence": float(ml_package.confidence_score * 100),
            "source": "ensemble",
        }


# ──────────────────────────────────────────────
#  Market Sanity Checks
# ──────────────────────────────────────────────

def apply_market_sanity_checks(
    bot: "RenaissanceTradingBot",
    product_id: str,
    market_data: Dict[str, Any],
    current_price: float,
) -> Optional[str]:
    """Pre-trade sanity gates. Returns skip reason string if pair should be skipped, else None."""

    # 1. Stale data check
    data_ts = market_data.get('timestamp')
    if data_ts:
        if isinstance(data_ts, str):
            data_ts = datetime.fromisoformat(data_ts)
        data_age = (datetime.now() - data_ts).total_seconds()
        if data_age > 60:
            bot.logger.warning(f"SANITY: Market data {data_age:.0f}s old - holding")
            return "stale_data"

    # 2. Flash crash / price spike detection
    if hasattr(bot, '_last_prices') and product_id in bot._last_prices:
        last_px = bot._last_prices[product_id]
        if last_px > 0 and current_price > 0:
            pct_change = abs(current_price - last_px) / last_px
            if pct_change > 0.05:
                bot.logger.warning(
                    f"SANITY: Flash move {pct_change:.1%} on {product_id} "
                    f"(${last_px:,.2f} -> ${current_price:,.2f}) — skipping cycle"
                )
                return "flash_move"
    if not hasattr(bot, '_last_prices'):
        bot._last_prices = {}
    bot._last_prices[product_id] = current_price

    # 3. Abnormal spread + per-pair spread filter
    ticker_data = market_data.get('ticker', {})
    bid = bot._force_float(ticker_data.get('bid', 0))
    ask = bot._force_float(ticker_data.get('ask', 0))
    if bid > 0 and ask > 0:
        spread_bps = ((ask - bid) / ((ask + bid) / 2)) * 10000

        if product_id not in bot._pair_spread_history:
            bot._pair_spread_history[product_id] = []
        bot._pair_spread_history[product_id].append(spread_bps)
        if len(bot._pair_spread_history[product_id]) > bot._spread_lookback:
            bot._pair_spread_history[product_id] = bot._pair_spread_history[product_id][-bot._spread_lookback:]

        if len(bot._pair_spread_history[product_id]) >= 10:
            avg_spread = sum(bot._pair_spread_history[product_id]) / len(bot._pair_spread_history[product_id])
            if avg_spread > bot._max_spread_bps:
                bot.logger.info(
                    f"SPREAD FILTER: {product_id} avg_spread={avg_spread:.1f}bps > {bot._max_spread_bps}bps — skipping"
                )
                return "spread_filter"

        if spread_bps > 50:
            bot.logger.warning(
                f"SANITY: Wide spread {spread_bps:.0f}bps on {product_id} — reducing confidence"
            )
            market_data['_sanity_spread_penalty'] = True

    # 4. Weekly loss limit
    weekly_loss_limit = bot._high_watermark_usd * 0.20 if bot._high_watermark_usd > 0 else 2000
    if bot._weekly_pnl < -weekly_loss_limit:
        bot.logger.warning(
            f"SANITY: Weekly loss ${bot._weekly_pnl:,.2f} exceeds limit ${-weekly_loss_limit:,.2f} — holding"
        )
        return "weekly_loss_limit"

    return None


# ──────────────────────────────────────────────
#  Token Spray Path
# ──────────────────────────────────────────────

async def execute_token_spray(
    bot: "RenaissanceTradingBot",
    product_id: str,
    weighted_signal: float,
    contributions: Dict[str, float],
    ml_package: Any,
    market_data: Dict[str, Any],
    current_price: float,
    rt_result: Optional[Dict[str, Any]],
) -> bool:
    """Execute token spray path. Returns True if spray handled this pair (skip legacy path)."""
    if not bot.token_spray:
        return False

    _spray_has_price_feed = (
        not bot._universe_built
        or product_id in bot._pair_binance_symbols
    )
    _spray_fresh_price = (
        bot.token_spray.last_prices.get(product_id)
        if _spray_has_price_feed else None
    )

    if _spray_has_price_feed and bot.token_spray.wallets:
        _wallet_crash_preds: Dict[str, float] = {}
        if rt_result:
            for _ck in ('CrashRegime_2bar', 'CrashRegime_1bar', 'CrashRegime'):
                _cv = rt_result.get(_ck)
                if _cv is not None:
                    _wallet_crash_preds[_ck] = float(_cv) if not isinstance(_cv, dict) else 0.0

        _spray_tokens = await bot.token_spray.spray_wallets(
            pair=product_id,
            ml_predictions=ml_package.ml_predictions if ml_package else [],
            crash_predictions=_wallet_crash_preds,
            weighted_signal=weighted_signal,
            contributions=contributions,
            market_data=market_data,
            fresh_price=_spray_fresh_price,
        )
        spray_token = _spray_tokens[0] if _spray_tokens else None
    elif _spray_has_price_feed:
        spray_token = await bot.token_spray.spray(
            pair=product_id,
            weighted_signal=weighted_signal,
            contributions=contributions,
            ml_package=ml_package,
            market_data=market_data,
            confidence=None,
            fresh_price=_spray_fresh_price,
        )
    else:
        spray_token = None

    _spray_action = "HOLD"
    if spray_token:
        _spray_action = "BUY" if spray_token.side == "LONG" else "SELL"

    # Persist decision + ML predictions
    if bot.db_enabled:
        hmm_regime_label = bot.regime_overlay.get_hmm_regime_label()
        decision_persist = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'product_id': product_id,
            'action': _spray_action,
            'confidence': spray_token.confidence if spray_token else 0.0,
            'position_size': spray_token.size_usd if spray_token else 0.0,
            'weighted_signal': weighted_signal,
            'reasoning': {'source': 'token_spray', 'rule': spray_token.direction_rule if spray_token else None},
            'hmm_regime': hmm_regime_label,
            'vae_loss': None,
        }
        bot._track_task(bot.db_manager.store_decision(decision_persist))

        _ml_preds: Dict[str, Any] = {}
        if ml_package and ml_package.ml_predictions:
            for mp in ml_package.ml_predictions:
                if isinstance(mp, dict):
                    _ml_preds[mp.get('model', 'unknown')] = (mp.get('prediction', 0.0), mp.get('confidence'))
                elif isinstance(mp, tuple) and len(mp) >= 2:
                    _ml_preds[mp[0]] = (mp[1], mp[2] if len(mp) > 2 else None)
        elif rt_result and 'predictions' in rt_result:
            for mn, pv in rt_result['predictions'].items():
                _ml_preds[mn] = (pv, None)
        for model_name, (pred, conf) in _ml_preds.items():
            bot._track_task(bot.db_manager.store_ml_prediction({
                'product_id': product_id,
                'model_name': model_name,
                'prediction': pred,
                'confidence': conf,
                'price_at_prediction': current_price,
            }))

    # Dashboard emit
    try:
        bot._track_task(bot.dashboard_emitter.emit("cycle", {
            "product_id": product_id,
            "action": _spray_action,
            "confidence": float(spray_token.confidence) if spray_token else 0.0,
            "position_size": float(spray_token.size_usd) if spray_token else 0.0,
            "weighted_signal": float(weighted_signal),
            "hmm_regime": bot.regime_overlay.get_hmm_regime_label() if bot.regime_overlay.enabled else None,
            "price": float(current_price),
            "source": "token_spray",
        }))
    except Exception as e:
        bot.logger.warning(f"Token spray decision persist failed for {product_id}: {e}")

    if not hasattr(bot, '_last_prices'):
        bot._last_prices = {}
    bot._last_prices[product_id] = current_price

    return True


# ──────────────────────────────────────────────
#  Exit Engine
# ──────────────────────────────────────────────

def evaluate_exits(
    bot: "RenaissanceTradingBot",
    product_id: str,
    current_price: float,
    market_data: Dict[str, Any],
    decision: Any,
) -> None:
    """Monitor open positions for alpha decay and trigger exits."""
    try:
        with bot.position_manager._lock:
            open_positions = list(bot.position_manager.positions.values())
        for pos in open_positions:
            if pos.product_id != product_id:
                continue
            holding_periods = int(
                (datetime.now() - pos.entry_time).total_seconds()
                / max(60, bot.config.get("trading", {}).get("cycle_interval_seconds", 300))
            )
            garch_forecast = market_data.get('garch_forecast', {})
            _pos_side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
            exit_decision = bot.position_sizer.calculate_exit_size(
                position_size=pos.size,
                entry_price=pos.entry_price,
                current_price=current_price,
                holding_periods=holding_periods,
                confidence=decision.confidence,
                volatility=garch_forecast.get('forecast_vol'),
                regime=bot.regime_overlay.get_current_regime() if hasattr(bot.regime_overlay, 'get_current_regime') else None,
                side=_pos_side,
            )
            if exit_decision['exit_fraction'] > 0:
                bot.logger.info(
                    f"EXIT ENGINE [{pos.position_id}]: {exit_decision['reason']} — "
                    f"fraction={exit_decision['exit_fraction']:.0%}, urgency={exit_decision['urgency']}"
                )
                close_ok, close_msg = bot.position_manager.close_position(
                    pos.position_id, reason=f"Exit engine: {exit_decision['reason']}"
                )
                if close_ok:
                    _side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                    _rpnl = bot._compute_realized_pnl(pos.entry_price, current_price, pos.size, _side)
                    bot._track_task(bot.db_manager.close_position_record(
                        pos.position_id,
                        close_price=float(current_price),
                        realized_pnl=float(_rpnl),
                        exit_reason=f"exit_engine:{exit_decision['reason']}",
                    ))
                    _hold_min = (datetime.now() - pos.entry_time).total_seconds() / 60
                    bot.logger.info(
                        f"TRADE CLOSED: {product_id} held {_hold_min:.1f} min | "
                        f"reason=exit_engine:{exit_decision['reason']} | P&L=${float(_rpnl):.2f}"
                    )
                    if bot.health_monitor and pos.entry_price > 0:
                        trade_pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                        if pos.side.value.upper() == 'SHORT':
                            trade_pnl_pct = -trade_pnl_pct
                        bot.health_monitor.record_trade(trade_pnl_pct, product_id)
                    bot._track_task(bot.dashboard_emitter.emit("trade", {
                        "product_id": product_id,
                        "side": "EXIT",
                        "price": float(current_price),
                        "size": float(pos.size * exit_decision['exit_fraction']),
                        "reason": exit_decision['reason'],
                    }))
    except Exception as exit_err:
        bot.logger.debug(f"Exit engine error: {exit_err}")


# ──────────────────────────────────────────────
#  Position Re-evaluation (Doc 10)
# ──────────────────────────────────────────────

def reevaluate_positions(
    bot: "RenaissanceTradingBot",
    product_id: str,
    current_price: float,
    market_data: Dict[str, Any],
    decision: Any,
) -> None:
    """Continuous position re-evaluation using PositionReEvaluator."""
    if not bot.position_reevaluator:
        return
    try:
        from decimal import Decimal as _D
        from core.data_structures import PositionContext

        with bot.position_manager._lock:
            _reeval_positions = list(bot.position_manager.positions.values())
        _reeval_positions_for_pid = [p for p in _reeval_positions if p.product_id == product_id]
        if not _reeval_positions_for_pid:
            return

        _regime_label = bot.regime_overlay.get_hmm_regime_label() if bot.regime_overlay.enabled else "unknown"
        _contexts = []
        for _pos in _reeval_positions_for_pid:
            _age_s = (datetime.now() - _pos.entry_time).total_seconds()
            _side = _pos.side.value.lower() if hasattr(_pos.side, 'value') else str(_pos.side).lower()
            _pnl_bps = 0.0
            if _pos.entry_price > 0 and current_price > 0:
                _move = (current_price - _pos.entry_price) / _pos.entry_price * 10000
                _pnl_bps = _move if _side == "long" else -_move
            _signal_ttl = bot.config.get('reevaluation', {}).get('signal_ttl_seconds', 3600)
            _rv_ticker = market_data.get('ticker', {})
            _rv_bid = bot._force_float(_rv_ticker.get('bid', 0))
            _rv_ask = bot._force_float(_rv_ticker.get('ask', 0))
            _rv_mid = (_rv_ask + _rv_bid) / 2.0 if _rv_bid > 0 and _rv_ask > 0 else 0
            _rv_spread_bps = ((_rv_ask - _rv_bid) / _rv_mid * 10000) if _rv_mid > 0 else 1.0
            _rv_cost_bps = _rv_spread_bps
            _ctx = PositionContext(
                position_id=_pos.position_id,
                pair=product_id,
                exchange="mexc" if product_id in bot._pair_binance_symbols else "coinbase",
                side=_side,
                strategy="combined",
                entry_price=_D(str(_pos.entry_price)),
                entry_size=_D(str(_pos.size)),
                entry_size_usd=_D(str(_pos.size * _pos.entry_price)),
                entry_timestamp=_pos.entry_time.timestamp(),
                entry_confidence=decision.confidence,
                entry_expected_move_bps=10.0,
                entry_cost_estimate_bps=_rv_cost_bps,
                entry_net_edge_bps=max(10.0 - _rv_cost_bps, 0),
                entry_regime=_regime_label,
                entry_volatility=0.02,
                entry_book_depth_usd=_D("50000"),
                entry_spread_bps=_rv_spread_bps,
                signal_ttl_seconds=_signal_ttl,
                current_size=_D(str(_pos.size)),
                current_size_usd=_D(str(_pos.size * current_price)),
                current_price=_D(str(current_price)),
                unrealized_pnl_bps=_pnl_bps,
                current_confidence=decision.confidence,
                current_regime=_regime_label,
                current_cost_to_exit_bps=_rv_spread_bps / 2.0,
                current_spread_bps=_rv_spread_bps,
            )
            _contexts.append(_ctx)

        _reeval_results = bot.position_reevaluator.reevaluate_all(
            _contexts,
            portfolio_state={"equity": bot._cached_balance_usd or 10000.0},
            market_state={"regime": _regime_label, "price": current_price},
        )
        for _rr in _reeval_results:
            if _rr.action == "close":
                _close_ok, _close_msg = bot.position_manager.close_position(
                    _rr.position_id, reason=f"ReEval: {_rr.reason_code}"
                )
                if _close_ok:
                    _rr_pos = next((p for p in _reeval_positions_for_pid if p.position_id == _rr.position_id), None)
                    if _rr_pos:
                        _rr_side = _rr_pos.side.value if hasattr(_rr_pos.side, 'value') else str(_rr_pos.side)
                        _rr_rpnl = bot._compute_realized_pnl(
                            _rr_pos.entry_price, current_price, _rr_pos.size, _rr_side
                        )
                    else:
                        _rr_rpnl = 0.0
                    bot._track_task(bot.db_manager.close_position_record(
                        _rr.position_id,
                        close_price=float(current_price),
                        realized_pnl=float(_rr_rpnl),
                        exit_reason=f"reeval:{_rr.reason_code}",
                    ))
                    _rr_hold_min = (datetime.now() - _rr_pos.entry_time).total_seconds() / 60 if _rr_pos else 0.0
                    bot.logger.warning(
                        f"REEVAL CLOSE: {_rr.position_id} — {_rr.reason_code} "
                        f"(edge={_rr.remaining_edge_bps:.1f}bps, urgency={_rr.urgency})"
                    )
                    bot.logger.info(
                        f"TRADE CLOSED: {product_id} held {_rr_hold_min:.1f} min | "
                        f"reason=reeval:{_rr.reason_code} | P&L=${float(_rr_rpnl):.2f}"
                    )
                    if bot.devil_tracker:
                        _rr_pnl_bps = 0.0
                        _rr_hold_s = _rr_hold_min * 60.0
                        _rr_adj_count = 0
                        if _rr_pos:
                            if _rr_pos.entry_price > 0 and current_price > 0:
                                _rr_move = (current_price - _rr_pos.entry_price) / _rr_pos.entry_price * 10000
                                _rr_pnl_bps = _rr_move if _rr_side == "long" else -_rr_move
                            _rr_adj_count = getattr(_rr_pos, 'adjustments', 0)
                            if isinstance(_rr_adj_count, list):
                                _rr_adj_count = len(_rr_adj_count)
                        bot.devil_tracker.record_exit(
                            position_id=_rr.position_id,
                            pair=product_id,
                            side=_rr_side if _rr_pos else "",
                            entry_price=float(_rr_pos.entry_price) if _rr_pos else 0.0,
                            exit_price=float(current_price),
                            size=float(_rr_pos.size) if _rr_pos else 0.0,
                            reason_code=_rr.reason_code,
                            hold_time_seconds=_rr_hold_s,
                            adjustments=_rr_adj_count,
                            pnl_bps=_rr_pnl_bps,
                        )
                else:
                    bot.logger.warning(f"REEVAL CLOSE FAILED: {_rr.position_id} — {_close_msg}")
            elif _rr.action == "trim" and _rr.trim_to_usd > 0:
                bot.logger.warning(
                    f"REEVAL TRIM: {_rr.position_id} to ${_rr.trim_to_usd:.2f} — {_rr.reason_code}"
                )
            elif _rr.action != "hold":
                bot.logger.warning(
                    f"REEVAL {_rr.action.upper()}: {_rr.position_id} — {_rr.reason_code}"
                )
    except Exception as _reeval_err:
        bot.logger.warning(f"Position re-evaluation failed: {_reeval_err}")


# ──────────────────────────────────────────────
#  Dashboard Event Emission
# ──────────────────────────────────────────────

def emit_dashboard_events(
    bot: "RenaissanceTradingBot",
    product_id: str,
    decision: Any,
    weighted_signal: float,
    current_price: float,
    market_data: Dict[str, Any],
) -> None:
    """Emit cycle/regime/trade/risk/confluence events to the dashboard."""
    try:
        bot._track_task(bot.dashboard_emitter.emit("cycle", {
            "product_id": product_id,
            "action": decision.action,
            "confidence": float(decision.confidence),
            "position_size": float(decision.position_size),
            "weighted_signal": float(weighted_signal),
            "hmm_regime": bot.regime_overlay.get_hmm_regime_label() if bot.regime_overlay.enabled else None,
            "price": float(current_price),
        }))
        if bot.regime_overlay.enabled and bot.regime_overlay.current_regime:
            _regime = bot.regime_overlay.current_regime
            bot._track_task(bot.dashboard_emitter.emit("regime", {
                "hmm_regime": _regime.get("hmm_regime", "unknown"),
                "confidence": float(_regime.get("hmm_confidence", 0.0)),
                "classifier": bot.regime_overlay.active_classifier,
                "bar_count": bot.regime_overlay.bar_count,
                "trend_persistence": float(_regime.get("trend_persistence", 0.0)),
                "volatility_acceleration": float(_regime.get("volatility_acceleration", 1.0)),
                "details": _regime.get("bootstrap_details", ""),
            }))
        if decision.action != 'HOLD':
            bot._track_task(bot.dashboard_emitter.emit("trade", {
                "product_id": product_id,
                "side": decision.action,
                "price": float(current_price),
                "size": float(decision.position_size),
            }))
        if hasattr(bot, 'risk_gateway') and bot.risk_gateway:
            bot._track_task(bot.dashboard_emitter.emit("risk.gateway", {
                "product_id": product_id,
                "action": decision.action,
                "vae_loss": float(decision.reasoning.get('vae_loss', 0.0) or 0.0),
                "verdict": decision.reasoning.get('risk_gateway_reason', 'unknown'),
                "pass_count": bot.risk_gateway.pass_count,
                "reject_count": bot.risk_gateway.reject_count,
            }))
        if market_data.get('confluence_data'):
            bot._track_task(bot.dashboard_emitter.emit("confluence", market_data['confluence_data']))
    except Exception as _de:
        bot.logger.debug(f"Dashboard emit error: {_de}")
