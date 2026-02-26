"""
Decision Audit Logger — captures every stage of the 15-step decision pipeline
as flat, queryable columns for signal attribution and gate analysis.

Usage:
    audit = DecisionAuditLogger(db_manager)
    audit.start_decision("BTC-USD", cycle_num=42)
    audit.record_market_snapshot(price=87000, bid=86999, ask=87001, ...)
    audit.record_raw_signals(signals)
    audit.record_regime("bull_trending", 0.72, "hmm", 0.95, False)
    ...
    await audit.finalize()   # single DB insert
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DecisionAuditLogger:
    """Collects pipeline stage data and produces a flat audit dict for DB insertion."""

    def __init__(self, db_manager: Any):
        self.db_manager = db_manager
        self._current: Dict[str, Any] = {}

    # ── Stage 0: Reset ──────────────────────────────────────────

    def start_decision(self, product_id: str, cycle_num: int) -> None:
        """Reset accumulator for a new decision cycle."""
        self._current = {
            'product_id': product_id,
            'cycle_number': cycle_num,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

    # ── Stage 1: Market Snapshot ─────────────────────────────────

    def record_market_snapshot(
        self,
        price: float,
        bid: float = 0.0,
        ask: float = 0.0,
        volume_24h: float = 0.0,
        ob_depth: float = 0.0,
    ) -> None:
        spread_bps = 0.0
        if bid > 0 and ask > 0:
            spread_bps = ((ask - bid) / ((ask + bid) / 2)) * 10_000
        self._current.update({
            'price': float(price),
            'bid': float(bid),
            'ask': float(ask),
            'spread_bps': float(spread_bps),
            'volume_24h_usd': float(volume_24h),
            'order_book_depth_usd': float(ob_depth),
        })

    # ── Stage 2: Raw Signals ─────────────────────────────────────

    # Map of known signal names → audit column suffix
    _SIGNAL_COLS = {
        'order_flow': 'sig_order_flow',
        'order_book': 'sig_order_book',
        'volume': 'sig_volume',
        'macd': 'sig_macd',
        'rsi': 'sig_rsi',
        'bollinger': 'sig_bollinger',
        'alternative': 'sig_alternative',
        'volume_profile': 'sig_volume_profile',
        'stat_arb': 'sig_stat_arb',
        'lead_lag': 'sig_lead_lag',
        'fractal': 'sig_fractal',
        'entropy': 'sig_entropy',
        'ml_ensemble': 'sig_ml_ensemble',
        'ml_cnn': 'sig_ml_cnn',
        'quantum': 'sig_quantum',
        'correlation_divergence': 'sig_correlation_divergence',
        'garch_vol': 'sig_garch_vol',
        'medallion_analog': 'sig_medallion_analog',
    }

    def record_raw_signals(self, signals: Dict[str, float]) -> None:
        for sig_name, col_name in self._SIGNAL_COLS.items():
            self._current[col_name] = float(signals.get(sig_name, 0.0))
        self._current['raw_signal_count'] = sum(
            1 for v in signals.values() if abs(float(v)) > 1e-6
        )

    # ── Stage 3: Regime ──────────────────────────────────────────

    def record_regime(
        self,
        label: Optional[str],
        confidence: float = 0.0,
        classifier: str = "unknown",
        bar_gap_ratio: float = 1.0,
        gap_poisoned: bool = False,
    ) -> None:
        self._current.update({
            'regime_label': label or "unknown",
            'regime_confidence': float(confidence),
            'regime_classifier': classifier,
            'bar_gap_ratio': float(bar_gap_ratio),
            'gap_poisoned': 1 if gap_poisoned else 0,
        })

    # ── Stage 4: Weights & Fusion ────────────────────────────────

    def record_weights(
        self,
        base_weights: Dict[str, float],
        cycle_weights: Dict[str, float],
        contributions: Dict[str, float],
        weighted_signal: float = 0.0,
    ) -> None:
        diffs = {}
        for k in set(base_weights) | set(cycle_weights):
            bw = base_weights.get(k, 0.0)
            cw = cycle_weights.get(k, 0.0)
            if abs(bw - cw) > 1e-6:
                diffs[k] = round(cw - bw, 6)
        self._current.update({
            'weighted_signal': float(weighted_signal),
            'signal_contributions': json.dumps(
                {k: round(float(v), 6) for k, v in contributions.items()},
                separators=(',', ':'),
            ),
            'weight_adjustments': json.dumps(diffs, separators=(',', ':')) if diffs else None,
        })

    # ── Stage 5: ML Predictions ──────────────────────────────────

    def record_ml(
        self,
        ml_package: Any = None,
        rt_result: Optional[Dict] = None,
        ml_scale: float = 1.0,
    ) -> None:
        if ml_package is None:
            self._current.update({
                'ml_ensemble_score': None,
                'ml_confidence': None,
                'ml_model_count': 0,
                'ml_agreement_pct': None,
                'ml_predictions_json': None,
                'ml_scale_factor': float(ml_scale),
            })
            return

        ensemble = float(ml_package.ensemble_score) if ml_package.ensemble_score else 0.0
        conf = float(ml_package.confidence_score) if ml_package.confidence_score else 0.0

        # Extract individual predictions
        pred_map = {}
        pred_values = []
        if ml_package.ml_predictions:
            for mp in ml_package.ml_predictions:
                if isinstance(mp, dict):
                    name = mp.get('model', mp.get('name', 'unknown'))
                    val = float(mp.get('prediction', 0.0))
                elif isinstance(mp, (tuple, list)) and len(mp) >= 2:
                    name = str(mp[0])
                    val = float(mp[1]) if isinstance(mp[1], (int, float)) else 0.0
                else:
                    continue
                pred_map[name] = round(val, 6)
                pred_values.append(val)

        agreement = None
        if pred_values:
            signs = [1 if p > 0 else (-1 if p < 0 else 0) for p in pred_values]
            nonzero = [s for s in signs if s != 0]
            if nonzero:
                agreement = max(nonzero.count(1), nonzero.count(-1)) / len(nonzero)

        self._current.update({
            'ml_ensemble_score': ensemble,
            'ml_confidence': conf,
            'ml_model_count': len(pred_values),
            'ml_agreement_pct': round(agreement, 4) if agreement is not None else None,
            'ml_predictions_json': json.dumps(pred_map, separators=(',', ':')) if pred_map else None,
            'ml_scale_factor': float(ml_scale),
        })

    # ── Stage 6: Confluence ──────────────────────────────────────

    def record_confluence(self, confluence_data: Optional[Dict]) -> None:
        if not confluence_data:
            return
        self._current.update({
            'confluence_boost': float(confluence_data.get('total_confluence_boost', 0.0)),
            'confluence_active_count': int(confluence_data.get('active_signal_count', 0)),
        })

    # ── Stage 7: Confidence ──────────────────────────────────────

    def record_confidence(
        self,
        signal_strength: float = 0.0,
        consensus: float = 0.0,
        raw_conf: float = 0.0,
        regime_boost: float = 0.0,
        ml_boost: float = 0.0,
        final_conf: float = 0.0,
    ) -> None:
        self._current.update({
            'signal_strength': float(signal_strength),
            'signal_consensus': float(consensus),
            'raw_confidence': float(raw_conf),
            'regime_conf_boost': float(regime_boost),
            'ml_conf_boost': float(ml_boost),
            'final_confidence': float(final_conf),
        })

    # ── Stage 8: Risk Gates ──────────────────────────────────────

    def record_gate(self, gate_name: str, passed: bool, detail: str = "") -> None:
        """Record a single gate result. Column name: gate_{gate_name}."""
        col = f'gate_{gate_name}'
        self._current[col] = 1 if passed else 0
        if detail:
            self._current[f'{col}_detail'] = str(detail)[:200]
        if not passed and 'blocked_by' not in self._current:
            self._current['blocked_by'] = gate_name

    # ── Stage 9: Position Sizing ─────────────────────────────────

    def record_sizing(
        self,
        sizing_result: Any = None,
        chain: Optional[Dict[str, float]] = None,
        buy_thresh: float = 0.0,
        sell_thresh: float = 0.0,
        garch_mult: float = 1.0,
    ) -> None:
        self._current.update({
            'buy_threshold': float(buy_thresh),
            'sell_threshold': float(sell_thresh),
            'garch_pos_multiplier': float(garch_mult),
        })
        if chain:
            self._current['sizing_chain_json'] = json.dumps(
                {k: round(float(v), 4) for k, v in chain.items()},
                separators=(',', ':'),
            )
        if sizing_result:
            self._current.update({
                'kelly_fraction': _safe_float(getattr(sizing_result, 'kelly_fraction', None)),
                'applied_fraction': _safe_float(getattr(sizing_result, 'applied_fraction', None)),
                'edge': _safe_float(getattr(sizing_result, 'edge', None)),
                'effective_edge': _safe_float(getattr(sizing_result, 'effective_edge', None)),
                'win_probability': _safe_float(getattr(sizing_result, 'win_probability', None)),
                'position_usd': _safe_float(getattr(sizing_result, 'usd_value', None)),
                'position_units': _safe_float(getattr(sizing_result, 'asset_units', None)),
                'market_impact_bps': _safe_float(getattr(sizing_result, 'market_impact_bps', None)),
                'capacity_pct': _safe_float(getattr(sizing_result, 'capacity_used_pct', None)),
            })

    # ── Stage 10: Final Decision ─────────────────────────────────

    def record_decision(
        self, action: str, position_size: float, blocked_by: Optional[str] = None
    ) -> None:
        self._current['final_action'] = action
        self._current['final_position_size'] = float(position_size)
        if blocked_by:
            self._current['blocked_by'] = blocked_by

    # ── Stage 11: Execution ──────────────────────────────────────

    def record_execution(
        self, mode: str = "PAPER", devil_trade_id: Optional[str] = None
    ) -> None:
        self._current['execution_mode'] = mode
        if devil_trade_id:
            self._current['devil_trade_id'] = devil_trade_id

    # ── Stage 12: Feature Vector ─────────────────────────────────

    def record_feature_vector(self, feature_vector: Any) -> None:
        if feature_vector is None:
            return
        try:
            if hasattr(feature_vector, 'numpy'):
                arr = feature_vector.numpy().flatten()
            elif isinstance(feature_vector, np.ndarray):
                arr = feature_vector.flatten()
            else:
                arr = np.array(feature_vector, dtype=float).flatten()

            # SHA256 hash for dedup
            self._current['feature_vector_hash'] = hashlib.sha256(
                arr.tobytes()
            ).hexdigest()[:16]

            # Top 5 features by absolute value
            abs_vals = np.abs(arr)
            top_idx = np.argsort(abs_vals)[-5:][::-1]
            top5 = {int(i): round(float(arr[i]), 6) for i in top_idx if abs_vals[i] > 1e-8}
            if top5:
                self._current['feature_top5_json'] = json.dumps(
                    top5, separators=(',', ':')
                )
        except Exception as e:
            logger.debug(f"Feature vector audit failed: {e}")

    # ── Stage 13: System State ───────────────────────────────────

    def record_system_state(
        self,
        drawdown_pct: float = 0.0,
        daily_pnl: float = 0.0,
        balance: float = 0.0,
        open_positions_count: int = 0,
        scan_tier: int = 0,
    ) -> None:
        self._current.update({
            'drawdown_pct': float(drawdown_pct),
            'daily_pnl': float(daily_pnl),
            'account_balance': float(balance),
            'open_positions_count': int(open_positions_count),
            'scan_tier': int(scan_tier),
        })

    # ── Finalize: single DB write ────────────────────────────────

    async def finalize(self) -> None:
        """Flatten _current dict and store to DB via db_manager.store_audit_log()."""
        if not self._current or 'product_id' not in self._current:
            return
        try:
            await self.db_manager.store_audit_log(self._current)
        except Exception as e:
            logger.error(f"Audit log persist failed: {e}")
        finally:
            self._current = {}


def _safe_float(val: Any) -> Optional[float]:
    """Convert to float, returning None if not numeric."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
