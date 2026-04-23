"""
bot/decision.py — Trading decision logic extracted from RenaissanceTradingBot.

Contains:
  make_trading_decision() — Final trading decision with Kelly sizing, regime gates,
                            anti-churn, direction consensus, and risk assessment.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def make_trading_decision(
    bot: Any,
    weighted_signal: float,
    signal_contributions: Dict[str, float],
    current_price: float = 0.0,
    real_time_result: Optional[Dict[str, Any]] = None,
    product_id: str = "BTC-USD",
    ml_package: Optional[Any] = None,
    market_data: Optional[Dict[str, Any]] = None,
    drawdown_pct: float = 0.0,
    audit_logger: Optional[Any] = None,
) -> Any:
    """Make final trading decision with Renaissance methodology + Kelly position sizing."""
    from renaissance_types import TradingDecision
    from position_manager import PositionStatus

    # ── COST PRE-SCREEN: "The edge must exceed the vig" — Medallion Principle ──
    try:
        if bot.paper_trading:
            min_viable_signal = 0.0001
        else:
            round_trip_cost = bot.position_sizer.estimate_round_trip_cost()
            min_viable_signal = round_trip_cost * 1.0
        if abs(weighted_signal) < min_viable_signal:
            bot.logger.debug(
                f"COST PRE-SCREEN: {product_id} signal {weighted_signal:.4f} < "
                f"min viable {min_viable_signal:.4f}"
            )
            if audit_logger:
                audit_logger.record_gate('cost_prescreen', False, f'signal={weighted_signal:.4f}<{min_viable_signal:.4f}')
                audit_logger.record_decision('HOLD', 0.0, blocked_by='cost_prescreen')
            return TradingDecision(
                action='HOLD', confidence=0.0, position_size=0.0,
                reasoning={'blocked_by': 'cost_pre_screen',
                           'signal': weighted_signal,
                           'min_viable': min_viable_signal},
                timestamp=datetime.now()
            )
        else:
            if audit_logger:
                audit_logger.record_gate('cost_prescreen', True)
    except Exception as e:
        bot.logger.warning(f"Cost pre-screen check failed (non-fatal): {e}")

    # Calculate confidence based on MODEL AGREEMENT (calibrated 2026-03-02)
    _max_pred_age_min = bot.config.get('ml_max_prediction_age_minutes', 15)
    if ml_package and hasattr(ml_package, 'timestamp') and ml_package.timestamp:
        _pred_age_sec = (datetime.now() - ml_package.timestamp).total_seconds()
        _pred_age_min = _pred_age_sec / 60.0
        if _pred_age_min > _max_pred_age_min:
            bot.logger.warning(
                f"ML STALE: {product_id} prediction is {_pred_age_min:.1f}min old "
                f"(max={_max_pred_age_min}min) — discarding"
            )
            ml_package = None

    # Step 1: Get ML model agreement if available
    ml_agreement = 0.5
    if ml_package and ml_package.ml_predictions:
        _ml_preds = []
        for mp in ml_package.ml_predictions:
            _val = None
            _name = None
            if isinstance(mp, (tuple, list)) and len(mp) >= 2:
                _name, _val = mp[0], mp[1]
            elif isinstance(mp, dict):
                _name = mp.get('model', mp.get('name', ''))
                _val = mp.get('prediction', mp.get('value', None))
            if _name and isinstance(_val, (int, float)) and not str(_name).startswith('_'):
                _ml_preds.append(float(_val))
        if _ml_preds:
            _bullish = sum(1 for p in _ml_preds if p > 0.001)
            _bearish = sum(1 for p in _ml_preds if p < -0.001)
            _total = len(_ml_preds)
            ml_agreement = max(_bullish, _bearish) / _total if _total > 0 else 0.5

    # Step 2: Get traditional signal consensus
    raw_contribs = [v for v in signal_contributions.values() if abs(v) > 0.0001]
    if raw_contribs and weighted_signal != 0:
        agreeing = sum(1 for v in raw_contribs if np.sign(v) == np.sign(weighted_signal))
        signal_consensus = agreeing / len(raw_contribs)
    else:
        signal_consensus = 0.5

    # Step 3: Combine — ML agreement weighted 60%, signal consensus 40%
    combined_agreement = 0.6 * ml_agreement + 0.4 * signal_consensus

    # Step 4: Map to calibrated confidence
    if combined_agreement >= 0.83:
        confidence = 0.65
    elif combined_agreement >= 0.67:
        confidence = 0.55
    elif combined_agreement >= 0.50:
        confidence = 0.42
    else:
        confidence = 0.30

    # Apply regime-derived confidence boost (max +/-5%)
    confidence = float(np.clip(confidence + bot.regime_overlay.get_confidence_boost(), 0.0, 1.0))

    # Record confidence breakdown for audit
    if audit_logger:
        _regime_boost = bot.regime_overlay.get_confidence_boost()
        audit_logger.record_confidence(
            signal_strength=float(abs(weighted_signal)),
            consensus=signal_consensus,
            raw_conf=combined_agreement,
            regime_boost=_regime_boost,
            ml_boost=0.0,
            final_conf=confidence,
        )

    # ── Regime-biased entry thresholds ──
    _regime_label = None
    try:
        if bot.regime_overlay.enabled:
            _regime_label = bot.regime_overlay.get_hmm_regime_label()
    except Exception as e:
        bot.logger.warning(f"Regime overlay label fetch failed: {e}")

    _BEARISH = {'bear_trending', 'bear_mean_reverting', 'high_volatility'}
    _BULLISH = {'bull_trending', 'bull_mean_reverting'}

    _pred_thresh = abs(bot.buy_threshold)
    _agree_thresh = 0.71

    # Determine action direction
    if confidence < bot.min_confidence:
        action = 'HOLD'
        if audit_logger:
            audit_logger.record_gate('confidence', False, f'conf={confidence:.4f}<min={bot.min_confidence}')
    elif weighted_signal > bot.buy_threshold:
        action = 'BUY'
        if audit_logger:
            audit_logger.record_gate('confidence', True)
    elif weighted_signal < bot.sell_threshold:
        action = 'SELL'
        if audit_logger:
            audit_logger.record_gate('confidence', True)
    else:
        action = 'HOLD'
        if audit_logger:
            audit_logger.record_gate('confidence', False, f'signal={weighted_signal:.4f} in dead zone')

    # Apply regime bias to thresholds based on trade direction
    if action != 'HOLD' and _regime_label and _regime_label not in ('neutral_sideways', 'unknown'):
        _is_bearish = _regime_label in _BEARISH
        _is_bullish = _regime_label in _BULLISH
        _is_low_vol = _regime_label == 'low_volatility'
        _counter_trend = (_is_bearish and action == 'BUY') or (_is_bullish and action == 'SELL')
        _with_trend = (_is_bearish and action == 'SELL') or (_is_bullish and action == 'BUY')

        if _is_low_vol:
            _pred_thresh = 0.04
            _agree_thresh = 0.65
            bot.logger.info(
                f"LOW VOL BOOST: {product_id} {action} in {_regime_label} — "
                f"lowered thresholds to pred>{_pred_thresh} agree>{_agree_thresh}"
            )
        elif _counter_trend:
            _pred_thresh = 0.10
            _agree_thresh = 0.80
            if abs(weighted_signal) < _pred_thresh:
                bot.logger.info(
                    f"REGIME FILTER: {product_id} {action} in {_regime_label} — "
                    f"|signal|={abs(weighted_signal):.4f} < {_pred_thresh} (counter-trend blocked)"
                )
                if audit_logger:
                    audit_logger.record_gate('regime_filter', False, f'counter-trend {action} in {_regime_label}')
                action = 'HOLD'
            else:
                bot.logger.info(
                    f"REGIME FILTER: {product_id} {action} in {_regime_label} — "
                    f"raised thresholds to pred>{_pred_thresh} agree>{_agree_thresh}"
                )
                if audit_logger:
                    audit_logger.record_gate('regime_filter', True, f'counter-trend passed {_regime_label}')
        elif _with_trend:
            _pred_thresh = 0.05
            _agree_thresh = 0.65
            bot.logger.info(
                f"REGIME BOOST: {product_id} {action} in {_regime_label} — "
                f"lowered thresholds to pred>{_pred_thresh} agree>{_agree_thresh}"
            )

    # ── Signal filter stats tracking ──
    bot._signal_filter_stats['total'] += 1
    if action == 'HOLD' and abs(weighted_signal) > 0.001:
        bot._signal_filter_stats['filtered_threshold'] += 1

    # ── Direction Consensus Gate ──
    if action in ('BUY', 'SELL') and ml_package and ml_package.ml_predictions:
        _dir_preds = []
        for mp in ml_package.ml_predictions:
            _name, _val = None, None
            if isinstance(mp, (tuple, list)) and len(mp) >= 2:
                _name, _val = mp[0], mp[1]
            elif isinstance(mp, dict):
                _name = mp.get('model', mp.get('name', ''))
                _val = mp.get('prediction', mp.get('value', None))
            if _name and isinstance(_val, (int, float)) and not str(_name).startswith('_'):
                _dir_preds.append((_name, float(_val)))

        if _dir_preds:
            total_models = len(_dir_preds)
            if action == 'SELL':
                aligned_count = sum(1 for _, v in _dir_preds if v < -0.001)
                label = 'bearish'
            else:
                aligned_count = sum(1 for _, v in _dir_preds if v > 0.001)
                label = 'bullish'
            aligned_pct = aligned_count / total_models if total_models > 0 else 0

            if aligned_pct < 0.50:
                bot.logger.info(
                    f"DIRECTION CONSENSUS GATE: {product_id} {action} blocked — "
                    f"only {aligned_count}/{total_models} models {label} ({aligned_pct:.0%} < 50%)"
                )
                bot._signal_filter_stats['filtered_agreement'] += 1
                if audit_logger:
                    audit_logger.record_gate('direction_consensus', False,
                                             f'{label}={aligned_count}/{total_models}={aligned_pct:.0%}')
                action = 'HOLD'
            else:
                bot.logger.info(
                    f"DIRECTION CONSENSUS GATE: {product_id} {action} PASSED — "
                    f"{aligned_count}/{total_models} models {label} ({aligned_pct:.0%})"
                )
                if audit_logger:
                    audit_logger.record_gate('direction_consensus', True,
                                             f'{label}={aligned_count}/{total_models}={aligned_pct:.0%}')
        else:
            if audit_logger:
                audit_logger.record_gate('direction_consensus', True, 'no_ml_preds')

    # ── ML Agreement Gate ──
    if action != 'HOLD' and ml_package and ml_package.ml_predictions:
        pred_values = []
        for mp in ml_package.ml_predictions:
            if isinstance(mp, dict):
                v = mp.get('prediction', 0.0)
                if isinstance(v, (int, float)):
                    pred_values.append(float(v))
            elif isinstance(mp, (tuple, list)) and len(mp) >= 2:
                v = mp[1]
                if isinstance(v, (int, float)):
                    pred_values.append(float(v))
        if len(pred_values) >= 3:
            signs = [1 if p > 0 else (-1 if p < 0 else 0) for p in pred_values]
            nonzero_signs = [s for s in signs if s != 0]
            if nonzero_signs:
                agreement = max(nonzero_signs.count(1), nonzero_signs.count(-1)) / len(nonzero_signs)
                if agreement < _agree_thresh:
                    bot.logger.info(
                        f"ML AGREEMENT GATE: {product_id} blocked — "
                        f"only {agreement:.0%} model agreement (need >{_agree_thresh:.0%})"
                    )
                    bot._signal_filter_stats['filtered_agreement'] += 1
                    if audit_logger:
                        audit_logger.record_gate('ml_agreement', False, f'agree={agreement:.2f}<{_agree_thresh}')
                    action = 'HOLD'
                else:
                    if audit_logger:
                        audit_logger.record_gate('ml_agreement', True, f'agree={agreement:.2f}')

    # Track traded signals
    if action != 'HOLD':
        bot._signal_filter_stats['traded'] += 1

    # Log filter stats every 20 cycles
    cycle_num = getattr(bot, 'scan_cycle_count', 0)
    if cycle_num > 0 and cycle_num % 20 == 0 and bot._signal_filter_stats['total'] > 0:
        stats = bot._signal_filter_stats
        bot.logger.info(
            f"SIGNAL FILTER STATS: {stats['traded']}/{stats['total']} traded "
            f"({100*stats['traded']/max(stats['total'],1):.0f}%), "
            f"filtered: threshold={stats['filtered_threshold']}, "
            f"confidence={stats['filtered_confidence']}, "
            f"agreement={stats['filtered_agreement']}"
        )

    # ── Anti-Churn Gate (Renaissance: conviction before action) ──
    if not hasattr(bot, '_signal_history'):
        bot._signal_history = {}
        bot._last_trade_cycle = {}

    hist = bot._signal_history.setdefault(product_id, [])
    hist.append(action)
    if len(hist) > 10:
        hist.pop(0)

    if action != 'HOLD':
        cycle_num = getattr(bot, 'scan_cycle_count', 0)
        last_trade = bot._last_trade_cycle.get(product_id, -999)
        min_hold_cycles = 6

        # 1. Minimum hold period
        if cycle_num - last_trade < min_hold_cycles:
            bot.logger.info(
                f"ANTI-CHURN: {product_id} cooldown — {cycle_num - last_trade}/{min_hold_cycles} cycles since last trade"
            )
            if audit_logger:
                audit_logger.record_gate('anti_churn', False, f'cooldown {cycle_num - last_trade}/{min_hold_cycles}')
            action = 'HOLD'

        # 2. Signal persistence
        elif len(hist) >= 2 and hist[-2] != action and hist[-2] != 'HOLD':
            bot.logger.info(
                f"ANTI-CHURN: {product_id} signal flip ({hist[-2]} -> {action}) — waiting for persistence"
            )
            if audit_logger:
                audit_logger.record_gate('anti_churn', False, f'flip {hist[-2]}->{action}')
            action = 'HOLD'
        else:
            if audit_logger:
                audit_logger.record_gate('anti_churn', True)

        # 3. Signal reversal on open position — close existing position
        if action != 'HOLD':
            try:
                with bot.position_manager._lock:
                    matching_positions = [
                        pos for pos in bot.position_manager.positions.values()
                        if pos.product_id == product_id and pos.status == PositionStatus.OPEN
                    ]
                for pos in matching_positions:
                    pos_side = pos.side.value.upper()
                    if (pos_side == 'LONG' and action == 'SELL') or \
                       (pos_side == 'SHORT' and action == 'BUY'):
                        close_ok, close_msg = bot.position_manager.close_position(
                            pos.position_id, reason=f"Signal reversal: {pos_side} -> {action}"
                        )
                        if close_ok:
                            _cpx = current_price
                            _rpnl = bot._compute_realized_pnl(
                                pos.entry_price, _cpx, pos.size, pos_side
                            )
                            bot._track_task(
                                bot.db_manager.close_position_record(
                                    pos.position_id,
                                    close_price=float(_cpx),
                                    realized_pnl=float(_rpnl),
                                    exit_reason="signal_reversal",
                                )
                            )
                        bot.logger.info(
                            f"SIGNAL REVERSAL: {product_id} closed {pos_side} position — {close_msg}"
                        )
                        if audit_logger:
                            audit_logger.record_gate('signal_reversal', False, f'closed {pos_side}')
                        action = 'HOLD'
                        break
                else:
                    if audit_logger:
                        audit_logger.record_gate('signal_reversal', True)
            except Exception as e:
                bot.logger.error(f"NETTING CHECK FAILED for {product_id}: {e} — blocking trade for safety")
                action = 'HOLD'

        # 4. Already positioned — don't stack same-direction positions
        if action != 'HOLD':
            try:
                with bot.position_manager._lock:
                    same_dir = [
                        pos for pos in bot.position_manager.positions.values()
                        if pos.product_id == product_id
                        and pos.status == PositionStatus.OPEN
                        and (
                            (pos.side.value.upper() == 'LONG' and action == 'BUY') or
                            (pos.side.value.upper() == 'SHORT' and action == 'SELL')
                        )
                    ]
                if same_dir:
                    bot.logger.info(
                        f"ALREADY POSITIONED: {product_id} already has {len(same_dir)} "
                        f"{same_dir[0].side.value} position(s) — holding"
                    )
                    if audit_logger:
                        audit_logger.record_gate('anti_stacking', False, f'{len(same_dir)} existing')
                    action = 'HOLD'
                else:
                    if audit_logger:
                        audit_logger.record_gate('anti_stacking', True)
            except Exception as e:
                bot.logger.error(f"ANTI-STACK CHECK FAILED for {product_id}: {e} — blocking trade for safety")
                action = 'HOLD'

    # ML-Enhanced Risk Assessment (Regime Gate)
    risk_assessment = bot.risk_manager.assess_risk_regime(ml_package)
    if risk_assessment['recommended_action'] == 'fallback_mode':
        bot.logger.warning("ML Risk assessment triggered FALLBACK MODE - halting trades")
        if audit_logger:
            audit_logger.record_gate('risk_regime', False, 'fallback_mode')
        action = 'HOLD'
    else:
        if audit_logger:
            audit_logger.record_gate('risk_regime', True)

    # Daily loss limit check
    if abs(bot.daily_pnl) >= bot.daily_loss_limit:
        if audit_logger:
            audit_logger.record_gate('daily_loss', False, f'pnl=${bot.daily_pnl:.2f}>=${bot.daily_loss_limit:.2f}')
        action = 'HOLD'
        bot.logger.warning(f"Daily loss limit reached: ${bot.daily_pnl}")
    else:
        if audit_logger:
            audit_logger.record_gate('daily_loss', True)

    # Gate through VAE Anomaly Detection
    vae_loss: float | None = None
    gate_reason = "not_evaluated"
    feature_vector = ml_package.feature_vector if ml_package else None

    # Always compute VAE loss for monitoring (even on HOLD)
    if feature_vector is not None and bot.risk_gateway.vae_trained and bot.risk_gateway.vae is not None:
        try:
            _, vae_loss = bot.risk_gateway._check_anomaly(feature_vector)
        except Exception as e:
            bot.logger.warning(f"VAE anomaly check failed: {e}")

    if action != 'HOLD':
        portfolio_data = {
            'total_value': bot.position_limit,
            'daily_pnl': bot.daily_pnl,
            'positions': {'BTC': bot.current_position},
            'current_price': current_price
        }
        is_allowed, vae_loss, gate_reason = bot.risk_gateway.assess_trade(
            action=action,
            amount=0,
            current_price=current_price,
            portfolio_data=portfolio_data,
            feature_vector=feature_vector
        )
        if not is_allowed:
            bot.logger.warning(f"Risk Gateway BLOCKED {action} order (reason={gate_reason}, vae_loss={vae_loss:.4f})")
            if audit_logger:
                audit_logger.record_gate('vae', False, f'vae_loss={vae_loss:.4f}')
                audit_logger.record_gate('risk_gateway', False, gate_reason)
            action = 'HOLD'
        else:
            if audit_logger:
                audit_logger.record_gate('vae', True, f'vae_loss={vae_loss:.4f}')
                audit_logger.record_gate('risk_gateway', True)

    # ── Volatility Dead-Zone Gate ──
    _vol_prediction = None
    if market_data:
        _vol_prediction = (market_data or {}).get('volatility_prediction')
    if action != 'HOLD' and _vol_prediction and isinstance(_vol_prediction, dict):
        _vol_regime = _vol_prediction.get('vol_regime', 'normal')
        if _vol_regime == 'dead_zone':
            bot.logger.info(
                f"VOL DEAD-ZONE GATE: {product_id} {action} blocked — "
                f"predicted magnitude={_vol_prediction.get('predicted_magnitude_bps', 0):.1f}bps "
                f"(regime={_vol_regime}, not enough expected move to cover costs)"
            )
            if audit_logger:
                audit_logger.record_gate('vol_dead_zone', False, f'regime={_vol_regime}')
            action = 'HOLD'
        else:
            if audit_logger:
                audit_logger.record_gate('vol_dead_zone', True, f'regime={_vol_regime}')

    # ── Renaissance Position Sizing (Kelly + cost gate + vol normalization) ──
    position_size = 0.0
    sizing_result = None
    if action != 'HOLD':
        # Gather volatility data
        mkt = market_data or {}
        garch_forecast = mkt.get('garch_forecast', {})
        volatility = garch_forecast.get('forecast_vol', None)
        vol_regime = garch_forecast.get('vol_regime', None)

        # Gather regime data
        fractal_regime = None
        if ml_package:
            fractal_regime = ml_package.fractal_insights.get('regime_detection', None)

        # Current exposure from position manager
        current_exposure = bot.position_manager._calculate_total_exposure()

        # Order book depth for liquidity constraint
        order_book_depth = None
        ob = mkt.get('order_book_snapshot')
        if ob:
            try:
                if hasattr(ob, 'bids'):
                    bid_depth = sum(lv.price * lv.size for lv in ob.bids[:10]) if ob.bids else 0
                    ask_depth = sum(lv.price * lv.size for lv in ob.asks[:10]) if ob.asks else 0
                else:
                    bids = ob.get('bids', [])
                    asks = ob.get('asks', [])
                    bid_depth = sum(float(b[1]) * float(b[0]) for b in bids[:10]) if bids else 0
                    ask_depth = sum(float(a[1]) * float(a[0]) for a in asks[:10]) if asks else 0
                order_book_depth = bid_depth + ask_depth
            except Exception as e:
                bot.logger.warning(f"Order book depth calculation failed: {e}")

        # Extract daily volume from market data if available
        daily_volume_usd = None
        try:
            ticker = mkt.get('ticker', {})
            vol_24h = ticker.get('volume_24h') or ticker.get('volume')
            if vol_24h and current_price > 0:
                daily_volume_usd = float(vol_24h) * current_price
        except Exception as e:
            bot.logger.warning(f"Daily volume extraction failed: {e}")

        # Use measured edge from signal scorecard when available
        _measured_edge = bot._get_measured_edge(product_id)

        # ── Signal Confidence Tier ──
        _tier_multiplier = 1.0
        _chain = {"regime": 1.0, "corr": 1.0, "health": 1.0, "tier": 1.0}

        # ── Regime Transition Warning ──
        if bot.regime_overlay.enabled:
            try:
                transition = bot.regime_overlay.get_transition_warning()
                if transition["alert_level"] != "none":
                    _chain["regime"] = transition["size_multiplier"]
                    _tier_multiplier *= transition["size_multiplier"]
                    bot.logger.info(f"REGIME TRANSITION: {transition['message']}")
            except Exception as e:
                bot.logger.warning(f"Regime transition warning check failed: {e}")

        # ── Portfolio Engine: correlation-aware sizing ──
        if bot.portfolio_engine and action != 'HOLD':
            try:
                current_positions = {}
                with bot.position_manager._lock:
                    for pos in bot.position_manager.positions.values():
                        pid = pos.product_id
                        current_positions[pid] = current_positions.get(pid, 0.0) + (pos.size * pos.entry_price)
                product_signals = {product_id: (weighted_signal, confidence)}
                for pid in bot.product_ids:
                    if pid != product_id and pid not in product_signals:
                        product_signals[pid] = (0.0, 0.0)

                port_result = bot.portfolio_engine.optimize(
                    product_signals, current_positions,
                    bot._cached_balance_usd or 10000.0,
                    cycle_count=getattr(bot, 'scan_cycle_count', 0),
                )
                port_adj = port_result.get(product_id, {})
                port_mult = port_adj.get("size_multiplier", 1.0)
                _chain["corr"] = port_mult
                if port_mult < 1.0:
                    _tier_multiplier *= port_mult
                    bot.logger.info(f"PORTFOLIO ENGINE: {product_id} sized to {port_mult:.0%} — {port_adj.get('reason', '')}")
            except Exception as e:
                bot.logger.warning(f"Portfolio engine correlation-aware sizing failed for {product_id}: {e}")

        # ── Health Monitor: apply rolling Sharpe-based size scaling ──
        if bot.health_monitor:
            health_mult = bot.health_monitor.get_size_multiplier()
            _chain["health"] = health_mult
            if health_mult < 1.0:
                bot.logger.info(f"HEALTH MONITOR: Sizing at {health_mult:.0%} (Sharpe-based)")
            _tier_multiplier *= health_mult
            if bot.health_monitor.is_exits_only():
                action = 'HOLD'
                if audit_logger:
                    audit_logger.record_gate('health_monitor', False, 'exits_only')
                bot.logger.warning("HEALTH MONITOR: EXITS-ONLY mode — blocking new entries")
            else:
                if audit_logger:
                    audit_logger.record_gate('health_monitor', True)

        sc = bot._signal_scorecard.get(product_id, {})
        if sc:
            top_signals = sorted(
                [(k, v) for k, v in signal_contributions.items() if abs(v) > 0.01],
                key=lambda x: abs(x[1]), reverse=True
            )[:3]
            tier_scores = []
            for sig_name, _ in top_signals:
                stats = sc.get(sig_name, {})
                total = stats.get('total', 0)
                correct = stats.get('correct', 0)
                if total >= 100 and correct / total > 0.55:
                    tier_scores.append(1.0)
                elif total >= 100 and correct / total >= 0.50:
                    tier_scores.append(0.5)
                elif total < 100:
                    tier_scores.append(0.5)
                else:
                    tier_scores.append(0.25)
            if tier_scores:
                _tier_multiplier = sum(tier_scores) / len(tier_scores)

        # ── Volatility Magnitude Sizing ──
        _chain["vol_mag"] = 1.0
        if _vol_prediction and isinstance(_vol_prediction, dict):
            _vm = _vol_prediction.get('vol_multiplier', 1.0)
            _chain["vol_mag"] = _vm
            _tier_multiplier *= _vm
            if _vm != 1.0:
                bot.logger.info(
                    f"VOL SIZING: {product_id} multiplier={_vm:.1f}x "
                    f"(regime={_vol_prediction.get('vol_regime', '?')}, "
                    f"mag={_vol_prediction.get('predicted_magnitude_bps', 0):.1f}bps)"
                )

        # Cap prediction strength for sizing
        PREDICTION_CAP = 0.06
        sizing_signal = weighted_signal
        if abs(weighted_signal) > PREDICTION_CAP:
            sizing_signal = PREDICTION_CAP * (1.0 if weighted_signal > 0 else -1.0)
            bot.logger.info(
                f"PREDICTION CAPPED: {weighted_signal:+.4f} -> {sizing_signal:+.4f} "
                f"for sizing (direction unchanged)"
            )

        sizing_result = bot.position_sizer.calculate_size(
            signal_strength=sizing_signal,
            confidence=confidence,
            current_price=current_price,
            product_id=product_id,
            volatility=volatility,
            vol_regime=vol_regime,
            fractal_regime=fractal_regime,
            order_book_depth_usd=order_book_depth,
            current_exposure_usd=current_exposure,
            ml_package=ml_package,
            account_balance_usd=bot._cached_balance_usd or None,
            daily_volume_usd=daily_volume_usd,
            drawdown_pct=drawdown_pct,
            measured_edge=_measured_edge,
            tier_size_multiplier=_tier_multiplier,
        )
        position_size = sizing_result.asset_units

        # Record sizing result for audit
        if audit_logger:
            audit_logger.record_sizing(
                sizing_result=sizing_result,
                chain=_chain,
                buy_thresh=bot.buy_threshold,
                sell_thresh=bot.sell_threshold,
                garch_mult=1.0,
            )

        if position_size <= 0:
            action = 'HOLD'
            bot.logger.info(f"Position sizer returned 0: {sizing_result.reasons[-1] if sizing_result.reasons else 'no edge'}")
        else:
            bot.logger.info(
                f"POSITION SIZED: {action} {position_size:.8f} {product_id} "
                f"(${sizing_result.usd_value:.2f}) | "
                f"Kelly={sizing_result.kelly_fraction:.4f} -> {sizing_result.applied_fraction:.4f} | "
                f"Edge={sizing_result.edge:.4f} EffEdge={sizing_result.effective_edge:.4f} "
                f"P(w)={sizing_result.win_probability:.3f} | "
                f"Impact={sizing_result.market_impact_bps:.1f}bps "
                f"Capacity={sizing_result.capacity_used_pct:.1f}% | "
                f"CostRatio={sizing_result.transaction_cost_ratio:.2f} "
                f"VolScalar={sizing_result.volatility_scalar:.2f} "
                f"RegimeScalar={sizing_result.regime_scalar:.2f}"
            )

        # ── SIZING CHAIN SUMMARY (Audit 3) ──
        _chain["tier"] = _tier_multiplier
        kelly_f = sizing_result.kelly_fraction if sizing_result else 0.0
        final_usd = sizing_result.usd_value if sizing_result else 0.0
        bot.logger.info(
            f"SIZING CHAIN {product_id}: "
            f"regime={_chain['regime']:.2f} x corr={_chain['corr']:.2f} x "
            f"health={_chain['health']:.2f} x vol_mag={_chain.get('vol_mag', 1.0):.2f} x "
            f"tier={_chain['tier']:.2f} x "
            f"kelly={kelly_f:.4f} -> final=${final_usd:.2f}"
        )

        # ── Kelly Sizer ACTIVE ──
        if bot.kelly_sizer and position_size > 0 and current_price > 0:
            try:
                dominant_sig = max(signal_contributions, key=lambda k: abs(signal_contributions[k]), default="combined")
                kelly_usd = bot.kelly_sizer.get_position_size(
                    signal_dict={"signal_type": dominant_sig, "pair": product_id, "confidence": confidence},
                    equity=bot._cached_balance_usd or 10000.0,
                )
                if kelly_usd > 0:
                    base_usd = position_size * current_price
                    kelly_capped_usd = min(base_usd, kelly_usd)
                    kelly_ratio = kelly_capped_usd / base_usd if base_usd > 0 else 1.0
                    if kelly_ratio < 0.95:
                        position_size = kelly_capped_usd / current_price
                        bot.logger.info(
                            f"KELLY SIZER: {product_id} sized to {kelly_ratio:.0%} of base "
                            f"(Kelly=${kelly_usd:.2f}, base=${base_usd:.2f}, final=${kelly_capped_usd:.2f})"
                        )
                    else:
                        bot.logger.info(
                            f"KELLY SIZER: {product_id} Kelly=${kelly_usd:.2f} >= base=${base_usd:.2f} — no reduction"
                        )
                elif kelly_usd == 0:
                    kelly_stats = bot.kelly_sizer.get_statistics(dominant_sig, product_id)
                    if kelly_stats.get("sufficient_data") and kelly_stats.get("expectancy_per_trade_bps", 0) <= 0:
                        bot.logger.warning(
                            f"KELLY SIZER: {product_id} negative expectancy — blocking trade"
                        )
                        position_size = 0.0
                        action = 'HOLD'
            except Exception as _kelly_err:
                bot.logger.debug(f"Kelly sizer failed: {_kelly_err}")

        # ── Leverage Manager ACTIVE ──
        if bot.leverage_mgr and position_size > 0:
            try:
                max_safe_lev = bot.leverage_mgr.compute_max_safe_leverage()
                if max_safe_lev > 0 and max_safe_lev < 1.0:
                    position_size *= max_safe_lev
                    bot.logger.info(
                        f"LEVERAGE MGR: {product_id} sized to {max_safe_lev:.0%} "
                        f"(consistency-based leverage cap)"
                    )
                elif max_safe_lev >= 1.0:
                    bot.logger.debug(
                        f"LEVERAGE MGR: {product_id} leverage headroom OK ({max_safe_lev:.2f}x)"
                    )
                if max_safe_lev == 0 and bot.leverage_mgr.should_reduce_leverage():
                    bot.logger.warning(
                        f"LEVERAGE MGR: {product_id} no leverage headroom — blocking"
                    )
                    position_size = 0.0
                    action = 'HOLD'
            except Exception as _lev_err:
                bot.logger.debug(f"Leverage manager failed: {_lev_err}")

        # ── Medallion Regime observation (Audit 1/2) ──
        if bot.medallion_regime:
            try:
                med_regime = bot.medallion_regime.predict_current_regime()
                overlay_regime = bot.regime_overlay.get_hmm_regime_label() if bot.regime_overlay.enabled else "unknown"
                bot.logger.info(
                    f"REGIME COMPARE (obs) {product_id}: "
                    f"overlay={overlay_regime} vs medallion={med_regime.get('regime_name', 'unknown')} "
                    f"(conf={med_regime.get('confidence', 0):.2f})"
                )
            except Exception as e:
                bot.logger.warning(f"Medallion regime comparison failed for {product_id}: {e}")

        # ── FINAL SIZE NORMALIZATION ──
        if action != 'HOLD' and position_size > 0 and current_price > 0:
            balance = bot._cached_balance_usd or 10000.0
            base_usd = balance * 0.03
            sig_scalar = min(abs(weighted_signal) / 0.02, 2.0)
            sig_scalar = max(sig_scalar, 0.5)
            conf_scalar = max(0.5, min((confidence - 0.3) * 3.0, 2.0))
            dd_scalar = getattr(bot, '_drawdown_size_scalar', 1.0)
            normalized_usd = base_usd * sig_scalar * conf_scalar * dd_scalar
            normalized_usd = max(100.0, min(normalized_usd, balance * 0.099))
            normalized_size = normalized_usd / current_price
            original_usd = position_size * current_price
            if abs(normalized_usd - original_usd) > 10:
                bot.logger.info(
                    f"SIZE NORMALIZATION: {product_id} "
                    f"chain=${original_usd:.0f} → normalized=${normalized_usd:.0f} "
                    f"(sig={sig_scalar:.2f}x, conf={conf_scalar:.2f}x, dd={dd_scalar:.2f}x)"
                )
            position_size = normalized_size

    reasoning = {
        'weighted_signal': weighted_signal,
        'confidence': confidence,
        'signal_contributions': signal_contributions,
        'current_price': current_price,
        'ml_risk_assessment': risk_assessment,
        'vae_loss': vae_loss,
        'risk_gateway_reason': gate_reason,
        'risk_check': {
            'daily_pnl': bot.daily_pnl,
            'daily_limit': bot.daily_loss_limit,
            'position_limit': bot.position_limit
        },
    }
    if sizing_result:
        reasoning['position_sizing'] = {
            'method': sizing_result.sizing_method,
            'kelly_fraction': sizing_result.kelly_fraction,
            'applied_fraction': sizing_result.applied_fraction,
            'edge': sizing_result.edge,
            'effective_edge': sizing_result.effective_edge,
            'market_impact_bps': sizing_result.market_impact_bps,
            'capacity_used_pct': sizing_result.capacity_used_pct,
            'win_probability': sizing_result.win_probability,
            'cost_ratio': sizing_result.transaction_cost_ratio,
            'vol_scalar': sizing_result.volatility_scalar,
            'regime_scalar': sizing_result.regime_scalar,
            'liquidity_scalar': sizing_result.liquidity_scalar,
            'usd_value': sizing_result.usd_value,
            'reasons': sizing_result.reasons,
        }

    # Final safety
    if action != 'HOLD' and (confidence <= 0 or confidence < bot.min_confidence):
        bot.logger.warning(
            f"CONF GUARD: {product_id} {action} blocked "
            f"(confidence={confidence:.4f}, min={bot.min_confidence})"
        )
        action = 'HOLD'

    decision = TradingDecision(
        action=action,
        confidence=confidence,
        position_size=position_size,
        reasoning=reasoning,
        timestamp=datetime.now()
    )

    return decision
