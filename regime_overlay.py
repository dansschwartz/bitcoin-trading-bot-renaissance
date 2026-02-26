"""
Regime Overlay Adapter
Bridges the Advanced 5-state HMM Regime Detector with the Renaissance Trading Bot.
Uses Bootstrap ATR/SMA rules when <200 bars available, then switches to HMM.
Reads OHLCV bars from five_minute_bars DB table for proper HMM input.
"""

import logging
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from enhanced_technical_indicators import EnhancedTechnicalIndicators as RenaissanceTechnicalIndicators
from medallion_regime_predictor import MedallionRegimePredictor

# Advanced HMM suite (new)
try:
    from advanced_regime_detector import (
        AdvancedRegimeDetector, RegimeState, MarketRegime,
        REGIME_ALPHA_WEIGHTS, ALPHA_TO_SIGNAL_MAP,
    )
    ADVANCED_HMM_AVAILABLE = True
except ImportError:
    ADVANCED_HMM_AVAILABLE = False

# Minimum bars for each classifier
BOOTSTRAP_MIN_BARS = 20
HMM_MIN_BARS = 200

# Bar gap detection thresholds
MIN_BAR_COMPLETENESS = 0.70   # suppress regime if <70% bars present (Council #7)
BAR_DURATION_S = 300          # 5-minute bars
GAP_THRESHOLD_S = 600         # gaps > 10 minutes are logged


class BootstrapRegimeClassifier:
    """
    Simple ATR/SMA-based regime classifier that works with as few as 20 bars.
    Used as PRIMARY classifier until enough bars accumulate for the HMM.

    Rules:
    - ATR(14) / Close > 2% → high_volatility
    - ATR(14) / Close < 0.5% → low_volatility
    - SMA(10) > SMA(30) and ATR moderate → trending (bullish)
    - SMA(10) < SMA(30) and ATR moderate → trending (bearish)
    - Otherwise → low_volatility (range-bound)
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def classify(self, bars_df: pd.DataFrame) -> Dict[str, Any]:
        """Classify regime from OHLCV bars DataFrame.

        Returns dict with keys: regime, confidence, classifier, details.
        """
        n = len(bars_df)
        if n < BOOTSTRAP_MIN_BARS:
            return {
                "regime": "unknown",
                "confidence": 0.0,
                "classifier": "bootstrap",
                "details": f"Insufficient bars ({n}/{BOOTSTRAP_MIN_BARS})",
            }

        close = bars_df["close"].values.astype(float)
        high = bars_df["high"].values.astype(float)
        low = bars_df["low"].values.astype(float)

        # ATR(14)
        atr_period = min(14, n - 1)
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        atr = float(np.mean(tr[-atr_period:]))
        current_price = float(close[-1])
        atr_pct = atr / (current_price + 1e-9)

        # SMAs
        sma_fast_period = min(10, n)
        sma_slow_period = min(30, n)
        sma_fast = float(np.mean(close[-sma_fast_period:]))
        sma_slow = float(np.mean(close[-sma_slow_period:]))
        sma_diff_pct = (sma_fast - sma_slow) / (sma_slow + 1e-9)

        # Classification
        if atr_pct > 0.02:
            regime = "high_volatility"
            confidence = min(0.9, 0.5 + (atr_pct - 0.02) * 10)
        elif atr_pct < 0.005:
            regime = "low_volatility"
            confidence = min(0.9, 0.5 + (0.005 - atr_pct) * 100)
        elif sma_diff_pct > 0.002:
            regime = "trending"
            confidence = min(0.85, 0.5 + abs(sma_diff_pct) * 50)
        elif sma_diff_pct < -0.002:
            regime = "trending"
            confidence = min(0.85, 0.5 + abs(sma_diff_pct) * 50)
        else:
            regime = "low_volatility"
            confidence = 0.55

        return {
            "regime": regime,
            "confidence": float(confidence),
            "classifier": "bootstrap",
            "details": f"ATR%={atr_pct:.4f} SMA_diff={sma_diff_pct:.4f}",
            "atr_pct": float(atr_pct),
            "sma_diff_pct": float(sma_diff_pct),
            "trend_direction": "bullish" if sma_diff_pct > 0 else "bearish",
        }


def _load_bars_from_db(db_path: str, pair: str = "BTC-USD", limit: int = 500) -> pd.DataFrame:
    """Load OHLCV bars from the five_minute_bars table."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT bar_start, open, high, low, close, volume "
            "FROM five_minute_bars WHERE pair = ? ORDER BY bar_start DESC LIMIT ?",
            (pair, limit),
        ).fetchall()
        conn.close()
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        data = [dict(r) for r in rows]
        df = pd.DataFrame(data)
        df = df.sort_values("bar_start").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


class RegimeOverlay:
    """
    Adapter that provides market regime intelligence to the Renaissance Trading Bot.
    Uses Bootstrap ATR/SMA rules when <200 bars, then switches to trained HMM.
    Reads OHLCV from five_minute_bars DB table for proper HMM input.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None,
                 db_path: Optional[str] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = config.get("enabled", False)
        self.consciousness_boost = config.get("consciousness_boost", 0.0)
        self._db_path = db_path

        # Initialize the experimental regime detector (legacy)
        self.detector = RenaissanceTechnicalIndicators()

        # Legacy 3-state HMM
        self.hmm_predictor = MedallionRegimePredictor(n_regimes=3, logger=self.logger)

        # Bootstrap classifier (works with 20+ bars)
        self._bootstrap = BootstrapRegimeClassifier(logger=self.logger)

        # Active classifier tracking
        self._active_classifier = "bootstrap"  # "bootstrap" or "hmm"
        self._bar_count = 0

        # Advanced 5-state HMM (new)
        self._advanced_detector: Optional[Any] = None
        self._advanced_regime_state: Optional[Any] = None
        self._cycle_count = 0
        if ADVANCED_HMM_AVAILABLE:
            hmm_cfg = {
                "n_regimes": config.get("hmm_regimes", 5),
                "refit_interval": config.get("hmm_refit_interval", 50),
                "min_samples": config.get("hmm_min_samples", HMM_MIN_BARS),
                "covariance_type": config.get("hmm_covariance_type", "full"),
                "n_iter": config.get("hmm_n_iter", 150),
            }
            self._advanced_detector = AdvancedRegimeDetector(hmm_cfg, logger=self.logger)
            self.logger.info("Advanced 5-state HMM regime detector initialized")

        self.current_regime = None
        self._last_valid_regime: Optional[Dict[str, Any]] = None  # fallback for gap-poisoned data
        self._bar_gap_ratio: float = 1.0
        # Transition monitoring
        self._prev_regime_label: Optional[str] = None
        self._transition_count: int = 0
        self._decision_count: int = 0
        self._last_transition_diag_cycle: int = 0
        self._transition_diagnostics: Optional[Dict[str, Any]] = None
        self.logger.info(
            f"RegimeOverlay initialized (Enabled: {self.enabled}, "
            f"Advanced HMM: {ADVANCED_HMM_AVAILABLE}, Bootstrap: ready)"
        )

    def set_db_path(self, db_path: str):
        """Set the DB path for reading five_minute_bars (called after init if not passed)."""
        self._db_path = db_path

    def update(self, price_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Update market regime detection using the latest price data.
        1. Loads OHLCV bars from five_minute_bars table
        2. If >= 200 bars: use HMM (AdvancedRegimeDetector)
        3. Else if >= 20 bars: use Bootstrap ATR/SMA rules
        4. Falls back to legacy signals if no bars available
        """
        if not self.enabled or price_df.empty:
            return None

        self._cycle_count += 1

        try:
            # Detect base regime using legacy signals summary
            signals = self.detector.get_signals_summary()

            self.current_regime = {
                'trend': signals.get('trend', 'unknown'),
                'volatility': signals.get('volatility', 'unknown'),
                'combined_signal': signals.get('combined_signal', 0.0),
                'confidence': signals.get('confidence', 0.0)
            }

            # Load OHLCV bars from DB for regime classification
            # Try BTC-USD first, fall back to BTC/USDT (arbitrage module)
            bars_df = pd.DataFrame()
            if self._db_path:
                bars_df = _load_bars_from_db(self._db_path, pair="BTC-USD", limit=500)
                if len(bars_df) < BOOTSTRAP_MIN_BARS:
                    alt = _load_bars_from_db(self._db_path, pair="BTC/USDT", limit=500)
                    if len(alt) > len(bars_df):
                        bars_df = alt

            self._bar_count = len(bars_df)

            # --- Bar gap detection: suppress regime if data has significant gaps ---
            self._bar_gap_ratio = 1.0
            _gap_poisoned = False
            if len(bars_df) >= BOOTSTRAP_MIN_BARS and 'bar_start' in bars_df.columns:
                try:
                    # bar_start may be Unix epoch (float) or ISO string
                    raw = bars_df['bar_start']
                    if pd.api.types.is_numeric_dtype(raw):
                        timestamps = pd.to_datetime(raw, unit='s').sort_values()
                    else:
                        timestamps = pd.to_datetime(raw).sort_values()
                    time_span_s = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
                    expected_bars = int(time_span_s / BAR_DURATION_S) + 1
                    if expected_bars > 0:
                        self._bar_gap_ratio = len(bars_df) / expected_bars

                    # Log significant gaps (> 10 min)
                    diffs = timestamps.diff().dt.total_seconds().dropna()
                    big_gaps = diffs[diffs > GAP_THRESHOLD_S]
                    if len(big_gaps) > 0:
                        max_gap_min = big_gaps.max() / 60
                        self.logger.debug(
                            f"Bar gaps: {len(big_gaps)} gaps >{GAP_THRESHOLD_S}s "
                            f"(max={max_gap_min:.0f}min, ratio={self._bar_gap_ratio:.2f})"
                        )

                    if self._bar_gap_ratio < MIN_BAR_COMPLETENESS:
                        _gap_poisoned = True
                        self.logger.warning(
                            f"REGIME GUARD: bar_gap_ratio={self._bar_gap_ratio:.2f} < "
                            f"{MIN_BAR_COMPLETENESS} — data too gappy for regime classification"
                        )
                except Exception as gap_err:
                    self.logger.debug(f"Gap detection error: {gap_err}")

            # --- Run HMM diagnostics periodically (every 50 cycles, independent of gap status) ---
            if (self._advanced_detector is not None
                    and self._advanced_detector.is_fitted
                    and self._cycle_count % 50 == 0
                    and self._cycle_count != self._last_transition_diag_cycle
                    and len(bars_df) >= HMM_MIN_BARS):
                self._last_transition_diag_cycle = self._cycle_count
                diag = self._advanced_detector.compute_transition_diagnostics(bars_df)
                if diag:
                    self._transition_diagnostics = diag
                    self.logger.info(
                        f"HMM DIAG: transition_rate={diag['transition_rate']:.3f} "
                        f"({'healthy' if diag['transition_rate_healthy'] else 'UNHEALTHY'}), "
                        f"degenerate_pairs={diag['n_degenerate']}, "
                        f"occupancy={diag['state_occupancy']}"
                    )
                    if diag['n_degenerate'] > 0:
                        for pair in diag['degenerate_pairs']:
                            self.logger.warning(
                                f"HMM DEGENERACY: {pair['state_a']} ↔ {pair['state_b']} "
                                f"KL={pair['kl_divergence']:.4f} (threshold=0.1)"
                            )

            # If data is gap-poisoned, fall back to neutral_sideways (Council #7).
            # Previous behavior used last_valid_regime which was often the corrupt
            # bear_trending itself (54% of decisions). neutral_sideways has the best
            # signal-to-noise ratio (29.8% zero rate vs 81.5% for bear_trending).
            if _gap_poisoned:
                self.current_regime = {
                    'hmm_regime': 'neutral_sideways',
                    'hmm_confidence': 0.50,
                    'gap_poisoned': True,
                    'bar_gap_ratio': self._bar_gap_ratio,
                    'classifier': f"{self._active_classifier}+gap_guard",
                    'trend_persistence': self.current_regime.get('trend_persistence', 0.0),
                }
                self.logger.warning(
                    f"GAP GUARD: bar_gap_ratio={self._bar_gap_ratio:.2f} < {MIN_BAR_COMPLETENESS} — "
                    f"forcing neutral_sideways (was {self._last_valid_regime.get('hmm_regime', 'unknown') if self._last_valid_regime else 'none'})"
                )
                return self.current_regime

            # --- Path A: HMM with proper OHLCV bars (>= 200 bars) ---
            hmm_classified = False
            if (self._advanced_detector is not None
                    and len(bars_df) >= HMM_MIN_BARS):
                # Fit if not already fitted
                if not self._advanced_detector.is_fitted:
                    self._advanced_detector.fit(bars_df)
                    if self._advanced_detector.is_fitted:
                        self.logger.info(
                            f"HMM fitted from five_minute_bars: {len(bars_df)} bars"
                        )

                # Periodically refit
                if self._advanced_detector.is_fitted:
                    self._advanced_detector.maybe_refit(bars_df, self._cycle_count)

                # Predict
                regime_state = self._advanced_detector.predict(bars_df)
                if regime_state is not None:
                    self._advanced_regime_state = regime_state
                    self._active_classifier = "hmm"
                    self.current_regime['hmm_regime'] = regime_state.current_regime.value
                    self.current_regime['hmm_confidence'] = regime_state.confidence
                    self.current_regime['regime_probabilities'] = regime_state.regime_probabilities
                    self.current_regime['next_regime_probs'] = regime_state.next_regime_probs
                    self.current_regime['regime_duration'] = regime_state.regime_duration_estimate
                    self.current_regime['alpha_weights'] = regime_state.alpha_weights
                    self.current_regime['classifier'] = "hmm"
                    hmm_classified = True

            # --- Path B: Bootstrap ATR/SMA rules (20-199 bars) ---
            if not hmm_classified and len(bars_df) >= BOOTSTRAP_MIN_BARS:
                bootstrap_result = self._bootstrap.classify(bars_df)
                self._active_classifier = "bootstrap"
                self.current_regime['hmm_regime'] = bootstrap_result['regime']
                self.current_regime['hmm_confidence'] = bootstrap_result['confidence']
                self.current_regime['classifier'] = "bootstrap"
                self.current_regime['bootstrap_details'] = bootstrap_result.get('details', '')

            # --- Path C: No bars → stay unknown ---
            if not hmm_classified and len(bars_df) < BOOTSTRAP_MIN_BARS:
                self._active_classifier = "none"
                self.current_regime['hmm_regime'] = 'unknown'
                self.current_regime['hmm_confidence'] = 0.0
                self.current_regime['classifier'] = 'none'

            # --- Legacy HMM path (for hmm_forecast field) ---
            if not self.hmm_predictor.is_fitted and len(price_df) > 100:
                self.hmm_predictor.fit(price_df)

            hmm_forecast = self.hmm_predictor.predict_next_regime(price_df)
            self.current_regime['hmm_forecast'] = hmm_forecast

            # Trend Persistence Score
            if len(price_df) >= 30:
                returns = price_df['close'].pct_change().dropna()
                persistence = (returns.rolling(window=10).mean().iloc[-1] /
                               (returns.rolling(window=10).std().iloc[-1] + 1e-6))
                self.current_regime['trend_persistence'] = float(np.clip(persistence, -1.0, 1.0))
            else:
                self.current_regime['trend_persistence'] = 0.0

            # Volatility Clustering
            if len(price_df) >= 20:
                recent_vol = price_df['close'].pct_change().tail(5).std()
                baseline_vol = price_df['close'].pct_change().tail(20).std()
                vol_acceleration = recent_vol / (baseline_vol + 1e-9)
                self.current_regime['volatility_acceleration'] = float(vol_acceleration)
            else:
                self.current_regime['volatility_acceleration'] = 1.0

            regime_label = self.current_regime.get('hmm_regime', 'unknown')
            conf = self.current_regime.get('hmm_confidence', 0.0)

            # Store bar quality metadata
            self.current_regime['bar_gap_ratio'] = self._bar_gap_ratio
            self.current_regime['gap_poisoned'] = False

            # --- Transition rate tracking ---
            self._decision_count += 1
            if self._prev_regime_label is not None and regime_label != self._prev_regime_label:
                self._transition_count += 1
            self._prev_regime_label = regime_label
            live_transition_rate = (
                self._transition_count / self._decision_count
                if self._decision_count > 0 else 0.0
            )
            self.current_regime['transition_rate'] = round(live_transition_rate, 4)
            self.current_regime['transition_count'] = self._transition_count

            # Save this as last valid regime for gap-poisoned fallback
            if regime_label != 'unknown' and conf > 0:
                self._last_valid_regime = dict(self.current_regime)

            self.logger.info(
                f"Market Regime: {regime_label} [{self._active_classifier}] "
                f"(conf={conf:.2f}, bars={self._bar_count}, "
                f"gap_ratio={self._bar_gap_ratio:.2f}, "
                f"Persist={self.current_regime['trend_persistence']:.2f})"
            )

            return self.current_regime

        except Exception as e:
            self.logger.error(f"Regime detection failed in overlay: {e}")
            return None

    @property
    def active_classifier(self) -> str:
        """Return which classifier is currently active: 'bootstrap', 'hmm', or 'none'."""
        return self._active_classifier

    @property
    def bar_count(self) -> int:
        """Return how many five-minute bars are available."""
        return self._bar_count

    def get_regime_adjusted_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply regime-specific alpha weight multipliers to base signal weights.
        Uses the ALPHA_TO_SIGNAL_MAP from AdvancedRegimeDetector.
        Falls back to get_adjusted_weights() if advanced HMM is unavailable.
        """
        if not self.enabled or not self.current_regime:
            return base_weights

        alpha_weights = self.current_regime.get('alpha_weights')
        if not alpha_weights or not ADVANCED_HMM_AVAILABLE:
            return self.get_adjusted_weights(base_weights)

        try:
            adjusted = base_weights.copy()

            for alpha_key, signal_keys in ALPHA_TO_SIGNAL_MAP.items():
                multiplier = alpha_weights.get(alpha_key, 1.0)
                for sk in signal_keys:
                    if sk in adjusted:
                        adjusted[sk] = float(adjusted[sk]) * float(multiplier)

            # Re-normalize
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {k: float(v) / total for k, v in adjusted.items()}

            return adjusted

        except Exception as e:
            self.logger.error(f"Regime weight adjustment failed: {e}")
            return base_weights

    def get_adjusted_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Legacy: Adjust base signal weights based on current market regime.
        """
        if not self.enabled or not self.current_regime:
            return base_weights

        try:
            regime_weights = self.current_regime.get('regime_weights', {})
            vol_w = float(regime_weights.get('volatility_weight', 1.0))
            trend_w = float(regime_weights.get('trend_weight', 1.0))
            liq_w = float(regime_weights.get('liquidity_weight', 1.0))

            adjusted = base_weights.copy()

            if 'macd' in adjusted: adjusted['macd'] = float(adjusted['macd']) * float(trend_w)
            if 'rsi' in adjusted: adjusted['rsi'] = float(adjusted['rsi']) * float(trend_w)
            if 'bollinger' in adjusted: adjusted['bollinger'] = float(adjusted['bollinger']) * float(vol_w)
            if 'order_flow' in adjusted: adjusted['order_flow'] = float(adjusted['order_flow']) * float(liq_w)
            if 'order_book' in adjusted: adjusted['order_book'] = float(adjusted['order_book']) * float(liq_w)
            if 'volume' in adjusted: adjusted['volume'] = float(adjusted['volume']) * float(liq_w)

            total = float(sum(adjusted.values()))
            if total > 0:
                adjusted = {k: float(v) / total for k, v in adjusted.items()}
            else:
                total_base = float(sum(base_weights.values()))
                adjusted = {k: float(v) / total_base for k, v in base_weights.items()}

            return adjusted

        except Exception as e:
            self.logger.error(f"Weight adjustment failed: {e}")
            return base_weights

    def get_hmm_regime_label(self) -> str:
        """Return the current HMM regime label string."""
        if self._active_classifier == "hmm" and self._advanced_regime_state is not None:
            return self._advanced_regime_state.current_regime.value
        if self.current_regime:
            return self.current_regime.get('hmm_regime', 'unknown')
        return 'unknown'

    def get_transition_diagnostics(self) -> Dict[str, Any]:
        """Return HMM transition rate monitoring and degeneracy detection results.

        Used by dashboard and research council for regime health assessment.
        """
        result = {
            "live_transition_rate": round(
                self._transition_count / self._decision_count, 4
            ) if self._decision_count > 0 else 0.0,
            "live_transition_count": self._transition_count,
            "live_decision_count": self._decision_count,
            "bar_gap_ratio": self._bar_gap_ratio,
        }
        if self._transition_diagnostics:
            result.update(self._transition_diagnostics)
        return result

    def get_transition_warning(self) -> Dict[str, Any]:
        """
        Check for regime transition risk. Returns size adjustment and alert info.
        Uses next_regime_probs from the HMM to detect upcoming adverse transitions.
        """
        result = {"size_multiplier": 1.0, "alert_level": "none", "message": ""}

        if not self.enabled or not self.current_regime:
            return result

        next_probs = self.current_regime.get('next_regime_probs', {})
        current_regime = self.current_regime.get('hmm_regime', 'unknown')

        if not next_probs:
            return result

        # Define adverse transitions
        bullish_regimes = {'bull_trending', 'bull_mean_reverting'}
        bearish_regimes = {'bear_trending', 'bear_mean_reverting'}

        # Calculate probability of transitioning to an adverse regime
        adverse_prob = 0.0
        if current_regime in bullish_regimes:
            adverse_prob = sum(next_probs.get(r, 0.0) for r in bearish_regimes)
        elif current_regime in bearish_regimes:
            adverse_prob = sum(next_probs.get(r, 0.0) for r in bullish_regimes)
        else:
            adverse_prob = next_probs.get('bear_trending', 0.0) + next_probs.get('bull_trending', 0.0)

        duration = self.current_regime.get('regime_duration', 0)

        if adverse_prob >= 0.60:
            result["size_multiplier"] = 0.5
            result["alert_level"] = "high"
            result["message"] = (
                f"Regime transition risk HIGH: {adverse_prob:.0%} chance of adverse transition "
                f"from {current_regime} (duration: {duration} cycles)"
            )
        elif adverse_prob >= 0.40:
            result["size_multiplier"] = 0.75
            result["alert_level"] = "medium"
            result["message"] = (
                f"Regime transition risk MEDIUM: {adverse_prob:.0%} chance of adverse transition "
                f"from {current_regime}"
            )

        return result

    def get_confidence_boost(self) -> float:
        """Get confidence boost factor from regime analysis."""
        if not self.enabled or not self.current_regime:
            return 0.0

        # Use advanced HMM confidence if available
        if self._advanced_regime_state is not None:
            conf = self._advanced_regime_state.confidence
            return float((conf - 0.5) * 0.1)  # Max +/- 5% boost

        # Use bootstrap confidence
        hmm_conf = self.current_regime.get('hmm_confidence', 0.5)
        return (hmm_conf - 0.5) * 0.1

    def get_current_regime(self) -> str:
        """Return current regime name for display."""
        if self.current_regime:
            return self.current_regime.get('hmm_regime', 'unknown')
        return 'unknown'
