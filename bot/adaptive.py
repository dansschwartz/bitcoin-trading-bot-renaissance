"""
bot/adaptive.py — Adaptive learning, weight optimization, and monitoring extracted from RenaissanceTradingBot.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from renaissance_trading_bot import RenaissanceTradingBot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. run_adaptive_learning_cycle  (was _run_adaptive_learning_cycle, async)
# ---------------------------------------------------------------------------

async def run_adaptive_learning_cycle(bot: "RenaissanceTradingBot") -> None:
    """Step 15: Online model calibration and attribution analysis"""
    # Specific log for verification
    with open("logs/adaptive_learning.log", "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} - Adaptive Learning Cycle triggered.\n")

    bot.logger.info("Adaptive Learning Cycle triggered.")
    if not bot.db_enabled:
        return

    try:
        # 1. Fetch recent decisions
        recent_decisions = await bot.db_manager.get_recent_data('decisions', hours=24)
        if len(recent_decisions) < 5:
            bot.logger.info("Insufficient data for adaptive learning. Need at least 5 decisions.")
            return

        bot.logger.info(f"Starting Adaptive Learning Cycle with {len(recent_decisions)} data points.")

        # 2. Run Genetic Weight Optimization (Step 14)
        # Skip genetic optimization when weights are locked
        if bot.config.get('weight_lock', False):
            bot.logger.info("Weight lock enabled — skipping genetic optimization")
        else:
            optimized_weights = await bot.genetic_optimizer.run_optimization_cycle(bot.signal_weights)

            if optimized_weights != bot.signal_weights:
                bot.logger.info("New optimized weights discovered via Evolution!")
                async with bot._weights_lock:
                    old_weights = bot.signal_weights.copy()
                    bot.signal_weights = optimized_weights

                # Log the change
                for k, v in optimized_weights.items():
                    diff = v - old_weights.get(k, 0)
                    if abs(diff) > 0.001:
                        bot.logger.info(f"  {k}: {old_weights.get(k,0):.3f} -> {v:.3f} ({diff:+.3f})")

                # 3. Persist to config.json to close the loop
                save_optimized_weights(bot, optimized_weights)

        # 4. Run Self-Reinforcing Learning Cycle (Step 19)
        if bot.real_time_pipeline.enabled:
            await bot.learning_engine.run_learning_cycle(
                bot.real_time_pipeline.processor.models
            )

        # 5. Trigger meta-learner training if we have an Ensemble model
        processor = bot.real_time_pipeline.processor
        if "Ensemble" in processor.models:
            ensemble = processor.models["Ensemble"]
            bot.logger.info("Calibrating Quantum Ensemble meta-learner with recent experience.")

        bot.logger.info(f"Adaptive calibration complete. Analyzed {len(recent_decisions)} recent data points.")

    except Exception as e:
        bot.logger.error(f"Adaptive learning cycle failed: {e}")


# ---------------------------------------------------------------------------
# 2. save_optimized_weights  (was _save_optimized_weights)
# ---------------------------------------------------------------------------

def save_optimized_weights(bot: "RenaissanceTradingBot", weights: Dict[str, float]) -> None:
    """Persist optimized weights back to config/config.json"""
    try:
        if not bot.config_path.exists():
            return

        with open(bot.config_path, 'r') as f:
            config_data = json.load(f)

        # Respect weight_lock — never overwrite locked weights
        if config_data.get('weight_lock', False):
            bot.logger.info("Weight lock enabled — skipping weight persistence")
            return

        # Ensure ML weights survive genetic optimization
        _ml_required = {'ml_ensemble': 0.20, 'ml_cnn': 0.0}
        for k, v in _ml_required.items():
            if k not in weights:
                weights[k] = v

        config_data['signal_weights'] = weights

        with open(bot.config_path, 'w') as f:
            json.dump(config_data, f, indent=4)

        bot.logger.info(f"Optimized weights persisted to {bot.config_path}")

    except Exception as e:
        bot.logger.error(f"Failed to persist optimized weights: {e}")


# ---------------------------------------------------------------------------
# 3. perform_attribution_analysis  (was _perform_attribution_analysis, async)
# ---------------------------------------------------------------------------

async def perform_attribution_analysis(bot: "RenaissanceTradingBot") -> None:
    """Step 11/13: Comprehensive performance attribution with Factor Analysis"""
    import pandas as pd

    if not bot.db_enabled:
        return

    try:
        # 1. Fetch labels (Realized outcomes) from DB
        # This uses the ground truth created in Step 19
        labels = await bot.db_manager.get_recent_data('labels', hours=72)
        if not labels:
            bot.logger.info("Attribution Analysis: No recent labels available yet.")
            return

        bot.logger.info(f"\U0001f3db\ufe0f RENAISSANCE ATTRIBUTION: Analyzing {len(labels)} realized outcomes.")

        # 2. Prepare Factor Exposures from signal contributions
        # We use the signal_contributions stored in the decisions table (via reasoning JSON)
        decisions = await bot.db_manager.get_recent_data('decisions', hours=72)
        if not decisions:
            bot.logger.info("Attribution Analysis: No recent decisions available.")
            return

        # Map labels to decisions
        label_map = {l['decision_id']: l for l in labels}

        portfolio_returns = []
        benchmark_returns = []

        # Use current signal weights to define factors
        current_factors = list(bot.signal_weights.keys())
        factor_exposures = {k: [] for k in current_factors}

        for d in decisions:
            if d['id'] in label_map:
                l = label_map[d['id']]
                # Portfolio return is based on decision and actual price change
                side_mult = 1.0 if d['action'] == 'BUY' else -1.0 if d['action'] == 'SELL' else 0.0
                portfolio_returns.append(l['ret_pct'] * side_mult)

                # Benchmark (Buy and Hold)
                benchmark_returns.append(l['ret_pct'])

                # Factors (Normalized contributions)
                reasoning = json.loads(d['reasoning'])
                contributions = reasoning.get('signal_contributions', {})
                for k in factor_exposures.keys():
                    factor_exposures[k].append(contributions.get(k, 0.0))

        if len(portfolio_returns) < 5:
            bot.logger.info("Attribution Analysis: Insufficient data samples for factor regression.")
            return

        # Execute Attribution
        attribution = bot.attribution_engine.analyze_performance_attribution(
            pd.Series(portfolio_returns),
            pd.Series(benchmark_returns),
            factor_exposures,
            {'factor_returns': pd.DataFrame()}  # Market data can be enhanced later
        )

        if 'error' not in attribution:
            summary = attribution.get('performance_summary', {})
            bot.logger.info(f"\u2705 ATTRIBUTION COMPLETE: Alpha: {summary.get('alpha', 0):+.4f} | Beta: {summary.get('beta', 0):.4f}")

            # Identify Top Alpha Drivers
            factor_attr = attribution.get('factor_attribution', {})
            if factor_attr:
                # Sort factors by their contribution to return
                drivers = sorted(factor_attr.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
                top_driver = drivers[0][0] if drivers else "None"
                bot.logger.info(f"\U0001f680 TOP ALPHA DRIVER: {top_driver}")

    except Exception as e:
        bot.logger.error(f"Performance attribution failed: {e}")


# ---------------------------------------------------------------------------
# 4. check_bar_liveness  (was _check_bar_liveness)
# ---------------------------------------------------------------------------

def check_bar_liveness(bot: "RenaissanceTradingBot") -> None:
    """Council S6: Check if bar pipeline is alive. Log CRITICAL if newest bar > 15min old."""
    if not bot.db_enabled:
        return
    try:
        db_path = bot.config.get('database', {}).get('path', 'data/renaissance_bot.db')
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        row = conn.execute("SELECT MAX(bar_end) FROM five_minute_bars").fetchone()
        conn.close()
        if row and row[0]:
            max_bar_end = float(row[0])
            age_seconds = time.time() - max_bar_end
            if age_seconds > 900:  # 15 minutes = 3x expected 5-min interval
                bot.logger.critical(
                    f"BAR PIPELINE STALE: newest bar is {age_seconds/60:.0f}min old "
                    f"(threshold: 15min). Data pipeline may be dead!"
                )
            elif age_seconds > 600:  # 10 minutes = soft warning
                bot.logger.warning(
                    f"Bar pipeline lagging: newest bar is {age_seconds/60:.1f}min old"
                )
    except Exception as e:
        bot.logger.debug(f"Bar liveness check failed: {e}")


# ---------------------------------------------------------------------------
# 5. log_ml_accuracy_summary  (was _log_ml_accuracy_summary, async)
# ---------------------------------------------------------------------------

async def log_ml_accuracy_summary(bot: "RenaissanceTradingBot") -> None:
    """Council S2 P3: Log per-model accuracy summary from ml_predictions table.

    Called every 100 cycles (~100 min). Queries the last 24h of evaluated predictions.
    """
    try:
        with bot.db_manager._get_connection() as conn:
            cursor = conn.cursor()
            # Per-model accuracy over last 24h
            cursor.execute('''
                SELECT model_name,
                       COUNT(*) as total,
                       SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM ml_predictions
                WHERE is_correct IS NOT NULL
                  AND evaluated_at > datetime('now', '-24 hours')
                GROUP BY model_name
                ORDER BY total DESC
            ''')
            rows = cursor.fetchall()
            if not rows:
                bot.logger.info("ML ACCURACY (24h): No evaluated predictions yet")
                return

            parts = []
            total_all = 0
            correct_all = 0
            for model_name, total, correct in rows:
                acc = round(correct / max(total, 1) * 100, 1)
                parts.append(f"{model_name}={acc}% (n={total})")
                total_all += total
                correct_all += correct
                # Council S3: Degradation alert — flag models below 48% with 50+ samples
                if total >= 50 and acc < 48.0:
                    bot.logger.warning(
                        f"ML MODEL DEGRADED: {model_name} accuracy={acc}% "
                        f"(n={total}) — below 48% threshold"
                    )

            overall_acc = round(correct_all / max(total_all, 1) * 100, 1)
            bot.logger.info(
                f"ML ACCURACY (24h): {', '.join(parts)} | "
                f"Overall={overall_acc}% (n={total_all})"
            )
    except Exception as e:
        bot.logger.error(f"ML accuracy summary failed: {e}")


# ---------------------------------------------------------------------------
# 6. log_kelly_calibration  (was _log_kelly_calibration, async)
# ---------------------------------------------------------------------------

async def log_kelly_calibration(bot: "RenaissanceTradingBot") -> None:
    """Council S3: Log Kelly calibration — compare estimated vs actual win rates.

    Called every 500 cycles (~8 hours at 60s interval).
    Buckets closed positions by confidence and checks if estimated win probability
    matches actual win rate. Results are stored in kelly_calibration_log table.
    """
    try:
        with bot.db_manager._get_connection() as conn:
            cursor = conn.cursor()
            # Get closed positions with confidence and outcome
            cursor.execute('''
                SELECT d.confidence, op.realized_pnl
                FROM open_positions op
                JOIN decisions d ON d.product_id = op.product_id
                    AND d.timestamp >= op.opened_at
                    AND d.action != 'HOLD'
                WHERE op.status = 'CLOSED'
                  AND op.realized_pnl IS NOT NULL
                  AND d.confidence > 0
                ORDER BY op.closed_at DESC
                LIMIT 2000
            ''')
            rows = cursor.fetchall()
            if len(rows) < 20:
                bot.logger.info(f"KELLY CALIBRATION: Only {len(rows)} closed trades — need 20+")
                return

            # Bucket by confidence
            buckets = {}
            for conf, pnl in rows:
                if conf <= 0.50:
                    bucket = '0.00-0.50'
                elif conf <= 0.55:
                    bucket = '0.50-0.55'
                elif conf <= 0.60:
                    bucket = '0.55-0.60'
                elif conf <= 0.65:
                    bucket = '0.60-0.65'
                elif conf <= 0.70:
                    bucket = '0.65-0.70'
                else:
                    bucket = '0.70-1.00'

                if bucket not in buckets:
                    buckets[bucket] = {'wins': 0, 'total': 0, 'conf_sum': 0.0}
                buckets[bucket]['total'] += 1
                buckets[bucket]['conf_sum'] += conf
                if pnl > 0:
                    buckets[bucket]['wins'] += 1

            ts = datetime.now(timezone.utc).isoformat()
            for bucket, data in sorted(buckets.items()):
                actual_wr = data['wins'] / max(data['total'], 1)
                avg_conf = data['conf_sum'] / max(data['total'], 1)
                # Estimated win prob from the Kelly formula: 0.48 + conf * 0.10, shrunk
                est_raw = 0.48 + avg_conf * 0.10
                est_wp = 0.5 + (min(est_raw, 0.65) - 0.5) * 0.55

                cursor.execute('''
                    INSERT INTO kelly_calibration_log
                    (timestamp, confidence_bucket, estimated_win_prob, actual_win_rate,
                     sample_size, kelly_fraction, avg_position_size_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (ts, bucket, round(est_wp, 4), round(actual_wr, 4),
                      data['total'], None, None))

                bot.logger.info(
                    f"KELLY CALIBRATION: bucket={bucket} est={est_wp*100:.1f}% "
                    f"actual={actual_wr*100:.1f}% (n={data['total']})"
                )
            conn.commit()
    except Exception as e:
        bot.logger.error(f"Kelly calibration logging failed: {e}")


# ---------------------------------------------------------------------------
# 7. check_pipeline_health  (was _check_pipeline_health, async)
# ---------------------------------------------------------------------------

async def check_pipeline_health(bot: "RenaissanceTradingBot") -> None:
    """Council S2 P1: Watchdog — check pipeline_heartbeat for stale components.

    Fires a warning if any component hasn't reported in >15 minutes.
    Called every 30 cycles (~30 min at 60s interval).
    """
    STALENESS_THRESHOLD_MINUTES = 15
    try:
        rows = await bot.db_manager.get_pipeline_health()
        if not rows:
            bot.logger.warning("PIPELINE WATCHDOG: No heartbeat rows yet — first cycle?")
            return
        now = datetime.now(timezone.utc)
        for row in rows:
            component = row.get('component', '?')
            last_beat_str = row.get('last_beat_utc', '')
            if not last_beat_str:
                continue
            try:
                last_beat = datetime.fromisoformat(last_beat_str.replace('Z', '+00:00'))
                if last_beat.tzinfo is None:
                    last_beat = last_beat.replace(tzinfo=timezone.utc)
                age_minutes = (now - last_beat).total_seconds() / 60.0
                if age_minutes > STALENESS_THRESHOLD_MINUTES:
                    msg = (
                        f"PIPELINE WATCHDOG ALERT: '{component}' stale for "
                        f"{age_minutes:.0f}min (threshold={STALENESS_THRESHOLD_MINUTES}min)"
                    )
                    bot.logger.error(msg)
                    if bot.monitoring_alert_manager:
                        bot._track_task(
                            bot.monitoring_alert_manager.send_system_event(
                                "pipeline_stale", msg
                            )
                        )
            except (ValueError, TypeError) as _parse_err:
                bot.logger.debug(f"Watchdog parse error for {component}: {_parse_err}")

        bot.logger.info(
            f"PIPELINE WATCHDOG: {len(rows)} components checked, all within threshold"
        )
    except Exception as e:
        bot.logger.error(f"Pipeline watchdog check failed: {e}")


# ---------------------------------------------------------------------------
# 8. compute_adaptive_weights  (was _compute_adaptive_weights)
# ---------------------------------------------------------------------------

def compute_adaptive_weights(
    bot: "RenaissanceTradingBot",
    product_id: str,
    base_weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Adaptive Weight Engine — Renaissance-style Bayesian signal weight updating.

    Uses the signal scorecard (measured accuracy per signal) to adjust weights:
    - Signals with >55% accuracy get upweighted
    - Signals with <48% accuracy get downweighted
    - Signals with too few samples keep config weights (prior)
    - Blend between config (prior) and measured (posterior) ramps up as data accumulates

    Returns adjusted weights dict (does NOT mutate bot.signal_weights).
    """
    sc = bot._signal_scorecard.get(product_id, {})
    if not sc:
        return base_weights

    # Aggregate across all products for more data
    agg_sc: Dict[str, Dict[str, int]] = {}
    for pid, signals in bot._signal_scorecard.items():
        for sig_name, stats in signals.items():
            entry = agg_sc.setdefault(sig_name, {"correct": 0, "total": 0})
            entry["correct"] += stats["correct"]
            entry["total"] += stats["total"]

    # Find signals with enough data
    eligible = {}
    max_total = 0
    for sig_name, stats in agg_sc.items():
        if stats["total"] >= bot._adaptive_min_samples:
            accuracy = stats["correct"] / stats["total"]
            eligible[sig_name] = accuracy
            max_total = max(max_total, stats["total"])

    if not eligible:
        return base_weights

    # Ramp blend factor: 0 at min_samples, 0.5 at 100+ samples
    blend = min(0.5, (max_total - bot._adaptive_min_samples) / 170.0)
    bot._adaptive_weight_blend = blend

    # Compute accuracy-derived weights
    # Transform accuracy to weight multiplier:
    # 50% (random) -> 0.5x, 55% -> 1.0x, 60% -> 1.5x, 65%+ -> 2.0x
    # <48% (anti-predictive) -> 0.1x
    multipliers = {}
    for sig_name in base_weights:
        if sig_name in eligible:
            acc = eligible[sig_name]
            if acc < 0.48:
                multipliers[sig_name] = 0.1  # actively wrong — near zero
            elif acc < 0.52:
                multipliers[sig_name] = 0.5  # noise
            elif acc < 0.55:
                multipliers[sig_name] = 0.8  # weak
            elif acc < 0.60:
                multipliers[sig_name] = 1.2  # good
            elif acc < 0.65:
                multipliers[sig_name] = 1.5  # strong
            else:
                multipliers[sig_name] = 2.0  # excellent
        else:
            multipliers[sig_name] = 1.0  # no data -> keep as-is

    # Blend: final = (1 - blend) * config_weight + blend * (config_weight * multiplier)
    # Simplifies to: final = config_weight * (1 - blend + blend * multiplier)
    adapted = {}
    for sig_name, w in base_weights.items():
        m = multipliers.get(sig_name, 1.0)
        adapted[sig_name] = w * (1.0 - blend + blend * m)

    # Renormalize so weights sum to 1.0
    total = sum(adapted.values())
    if total > 0:
        adapted = {k: v / total for k, v in adapted.items()}

    return adapted


# ---------------------------------------------------------------------------
# 9. get_measured_edge  (was _get_measured_edge)
# ---------------------------------------------------------------------------

def get_measured_edge(bot: "RenaissanceTradingBot", product_id: str) -> Optional[float]:
    """
    Compute realized edge from signal scorecard + ML prediction accuracy.
    Council S3 #1/#5: Blends scorecard edge with DB-measured ML accuracy.
    Returns None if insufficient data, else a float [0, 0.15].
    """
    # Source 1: Signal scorecard (in-memory, per-signal)
    scorecard_edge = None
    sc = bot._signal_scorecard.get(product_id, {})
    if sc:
        total_correct = sum(s["correct"] for s in sc.values())
        total_total = sum(s["total"] for s in sc.values())
        if total_total >= 20:
            accuracy = total_correct / total_total
            scorecard_edge = max(0.0, accuracy - 0.5)

    # Source 2: ML prediction accuracy (DB-backed, refreshed every 10 cycles)
    ml_edge = None
    ml_info = bot._ml_accuracy_cache.get(product_id)
    if ml_info and ml_info['n'] >= bot._ml_eval_min_predictions:
        ml_edge = max(0.0, ml_info['accuracy'] - 0.5)

    # Blend sources
    if scorecard_edge is not None and ml_edge is not None:
        edge = (bot._ml_eval_blend_measured * ml_edge +
                bot._ml_eval_blend_model * scorecard_edge)
    elif ml_edge is not None:
        edge = ml_edge
    elif scorecard_edge is not None:
        edge = scorecard_edge
    else:
        return None

    return min(edge, 0.15)


# ---------------------------------------------------------------------------
# 10. refresh_ml_accuracy_cache  (was _refresh_ml_accuracy_cache)
# ---------------------------------------------------------------------------

def refresh_ml_accuracy_cache(bot: "RenaissanceTradingBot") -> None:
    """Council S3 #1/#5: Query ML prediction accuracy per pair from DB."""
    try:
        db_path = getattr(bot.db_manager, 'db_path', None) or 'data/renaissance_bot.db'
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        rows = conn.execute('''
            SELECT product_id,
                   COUNT(*) as n,
                   SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
            FROM ml_predictions
            WHERE is_correct IS NOT NULL
              AND is_correct >= 0
              AND model_name IN ('meta_ensemble', 'MetaEnsemble')
              AND timestamp > datetime('now', '-7 days')
            GROUP BY product_id
        ''').fetchall()
        conn.close()
        for pid, n, correct in rows:
            acc = correct / n if n > 0 else 0.0
            bot._ml_accuracy_cache[pid] = {
                'accuracy': acc, 'n': n, 'edge': max(0.0, acc - 0.5)
            }
        if rows:
            total_n = sum(r[1] for r in rows)
            total_c = sum(r[2] for r in rows)
            agg_acc = total_c / total_n if total_n > 0 else 0.0
            bot.logger.info(
                f"ML ACCURACY CACHE refreshed: {len(rows)} pairs, "
                f"{total_n} predictions, {agg_acc:.1%} overall accuracy"
            )
        else:
            bot.logger.info("ML ACCURACY CACHE: no evaluated MetaEnsemble predictions found")
    except Exception as e:
        bot.logger.warning(f"ML accuracy cache refresh failed: {e}")


# ---------------------------------------------------------------------------
# 11. update_dynamic_thresholds  (was _update_dynamic_thresholds)
# ---------------------------------------------------------------------------

def update_dynamic_thresholds(
    bot: "RenaissanceTradingBot",
    product_id: str,
    market_data: Dict[str, Any],
) -> None:
    """Adjusts BUY/SELL thresholds based on volatility and confidence (Step 8)"""
    if not bot.adaptive_thresholds:
        return

    try:
        # Use technical indicators volatility regime
        latest_tech = bot._get_tech(product_id).get_latest_signals()
        vol_regime = latest_tech.volatility_regime if latest_tech else None

        # Base thresholds — from config (default 0.06 after backtest analysis).
        # Backtest proved: only |prediction| > 0.06 has >53% accuracy.
        base_buy = float(bot.config.get('trading', {}).get('buy_threshold', 0.06))
        base_sell = float(bot.config.get('trading', {}).get('sell_threshold', -0.06))
        bot.buy_threshold = base_buy
        bot.sell_threshold = base_sell

        # Adjust based on volatility (scale from higher base)
        if vol_regime == "high_volatility" or vol_regime == "extreme_volatility":
            # Increase thresholds in high volatility to avoid fakeouts
            bot.buy_threshold = base_buy * 1.5
            bot.sell_threshold = base_sell * 1.5
        elif vol_regime == "low_volatility":
            # Decrease thresholds in low volatility to catch smaller moves
            bot.buy_threshold = base_buy * 0.7
            bot.sell_threshold = base_sell * 0.7

        bot.logger.info(f"Dynamic Thresholds updated: Buy {bot.buy_threshold:.2f}, Sell {bot.sell_threshold:.2f} (Regime: {vol_regime})")
    except Exception as e:
        bot.logger.error(f"Failed to update dynamic thresholds: {e}")
