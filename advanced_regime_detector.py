"""
Advanced Regime Detector — Medallion-Style HMM Suite
5-state Hidden Markov Model for market regime classification
with periodic refitting, regime-specific alpha weights, and duration estimation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Graceful import with fallback
try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

try:
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MarketRegime(Enum):
    BULL_TRENDING = "bull_trending"
    BULL_MEAN_REVERTING = "bull_mean_reverting"
    NEUTRAL_SIDEWAYS = "neutral_sideways"
    BEAR_MEAN_REVERTING = "bear_mean_reverting"
    BEAR_TRENDING = "bear_trending"


@dataclass
class RegimeState:
    current_regime: MarketRegime
    regime_probabilities: Dict[str, float]
    transition_matrix: Optional[np.ndarray]
    next_regime_probs: Dict[str, float]
    regime_duration_estimate: int
    alpha_weights: Dict[str, float]
    confidence: float
    features_used: List[str] = field(default_factory=list)


# Regime-specific signal weight multipliers
REGIME_ALPHA_WEIGHTS = {
    MarketRegime.BULL_TRENDING: {
        "momentum_boost": 1.4,
        "mean_rev_boost": 0.6,
        "volatility_boost": 0.8,
        "flow_boost": 1.2,
    },
    MarketRegime.BULL_MEAN_REVERTING: {
        "momentum_boost": 0.7,
        "mean_rev_boost": 1.5,
        "volatility_boost": 1.0,
        "flow_boost": 1.0,
    },
    MarketRegime.NEUTRAL_SIDEWAYS: {
        "momentum_boost": 0.5,
        "mean_rev_boost": 1.3,
        "volatility_boost": 0.7,
        "flow_boost": 1.1,
    },
    MarketRegime.BEAR_MEAN_REVERTING: {
        "momentum_boost": 0.7,
        "mean_rev_boost": 1.5,
        "volatility_boost": 1.0,
        "flow_boost": 1.0,
    },
    MarketRegime.BEAR_TRENDING: {
        "momentum_boost": 1.4,
        "mean_rev_boost": 0.6,
        "volatility_boost": 1.2,
        "flow_boost": 1.2,
    },
}

# Map alpha weight keys to signal weight keys
ALPHA_TO_SIGNAL_MAP = {
    "momentum_boost": ["macd", "lead_lag", "fractal"],
    "mean_rev_boost": ["stat_arb", "bollinger", "rsi"],
    "volatility_boost": ["entropy", "quantum"],
    "flow_boost": ["order_flow", "order_book", "volume"],
}


class AdvancedRegimeDetector:
    """
    5-state HMM regime detector with rich features, periodic refitting,
    transition probability output, and regime-specific alpha weights.
    Uses hmmlearn with sklearn GaussianMixture as fallback.
    """

    FEATURE_NAMES = [
        "log_returns", "realized_volatility", "volume_change",
        "spread_proxy", "order_flow_proxy",
    ]

    def __init__(self, config: Dict[str, Any] = None, logger: Optional[logging.Logger] = None):
        config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.n_regimes = config.get("n_regimes", 5)
        self.refit_interval = config.get("refit_interval", 50)
        self.min_samples = config.get("min_samples", 200)
        self.covariance_type = config.get("covariance_type", "full")
        self.n_iter = config.get("n_iter", 150)
        self.use_hmmlearn = config.get("use_hmmlearn", True) and HMMLEARN_AVAILABLE
        self.fallback_to_gmm = config.get("fallback_to_gmm", True) and SKLEARN_AVAILABLE

        self._model = None
        self._gmm_model = None
        self.is_fitted = False
        self._regime_map: Dict[int, MarketRegime] = {}
        self._last_fit_cycle = -1

    def _build_features(self, price_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Build 5-feature matrix from OHLCV data."""
        if len(price_df) < 30:
            return None

        close = price_df["close"].values.astype(float)
        high = price_df["high"].values.astype(float)
        low = price_df["low"].values.astype(float)
        open_ = price_df["open"].values.astype(float)
        volume = price_df["volume"].values.astype(float)

        # 1. Log returns
        log_returns = np.diff(np.log(np.maximum(close, 1e-9)))

        # 2. Realized volatility (20-bar rolling std of returns)
        vol_window = 20
        realized_vol = pd.Series(log_returns).rolling(vol_window).std().values

        # 3. Volume change (pct change of volume, clipped)
        vol_safe = np.maximum(volume[1:], 1.0)
        vol_prev = np.maximum(volume[:-1], 1.0)
        volume_change = np.clip((vol_safe - vol_prev) / vol_prev, -5.0, 5.0)

        # 4. Spread proxy: (high - low) / close (intrabar range)
        range_vals = (high[1:] - low[1:]) / np.maximum(close[1:], 1e-9)

        # 5. Order flow proxy: (close - open) / (high - low + epsilon)
        bar_range = high[1:] - low[1:] + 1e-9
        flow_proxy = (close[1:] - open_[1:]) / bar_range

        # Align lengths (realized_vol has NaNs for first vol_window entries)
        min_len = len(log_returns)
        features = np.column_stack([
            log_returns[:min_len],
            realized_vol[:min_len],
            volume_change[:min_len],
            range_vals[:min_len],
            flow_proxy[:min_len],
        ])

        # Drop rows with NaN
        valid_mask = ~np.isnan(features).any(axis=1)
        features = features[valid_mask]

        if len(features) < self.min_samples // 2:
            return None

        return features

    def fit(self, price_df: pd.DataFrame) -> bool:
        """Fit the HMM on the feature matrix."""
        features = self._build_features(price_df)
        if features is None or len(features) < self.min_samples:
            return False

        # Normalize features for numerical stability
        self._feature_means = features.mean(axis=0)
        self._feature_stds = features.std(axis=0) + 1e-9
        normalized = (features - self._feature_means) / self._feature_stds

        fitted = False

        # Try hmmlearn first
        if self.use_hmmlearn:
            try:
                model = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=42,
                )
                model.fit(normalized)
                # Regularize transition matrix: fix zero-sum rows
                tm = model.transmat_.copy()
                row_sums = tm.sum(axis=1)
                for i in range(len(row_sums)):
                    if row_sums[i] < 1e-10:
                        tm[i] = 1.0 / self.n_regimes
                tm = tm / tm.sum(axis=1, keepdims=True)
                model.transmat_ = tm
                self._model = model
                hidden_states = model.predict(normalized)
                self._regime_map = self._label_regimes(hidden_states, features)
                fitted = True
                self.logger.info(f"HMM fitted with hmmlearn: {self.n_regimes} states, {len(features)} samples")
            except Exception as e:
                self.logger.warning(f"hmmlearn fit failed: {e}")

        # Fallback to GMM
        if not fitted and self.fallback_to_gmm:
            try:
                gmm = GaussianMixture(
                    n_components=self.n_regimes,
                    covariance_type=self.covariance_type,
                    n_init=3,
                    random_state=42,
                )
                gmm.fit(normalized)
                self._gmm_model = gmm
                self._model = None
                labels = gmm.predict(normalized)
                self._regime_map = self._label_regimes(labels, features)
                fitted = True
                self.logger.info(f"HMM fallback: GMM fitted with {self.n_regimes} components")
            except Exception as e:
                self.logger.warning(f"GMM fallback fit failed: {e}")

        self.is_fitted = fitted
        return fitted

    def predict(self, price_df: pd.DataFrame) -> Optional[RegimeState]:
        """Predict current regime and return full RegimeState."""
        if not self.is_fitted:
            return None

        features = self._build_features(price_df)
        if features is None or len(features) < 10:
            return None

        normalized = (features - self._feature_means) / self._feature_stds

        try:
            if self._model is not None:
                # hmmlearn path
                hidden_states = self._model.predict(normalized)
                current_state = int(hidden_states[-1])

                # State probabilities
                posteriors = self._model.predict_proba(normalized)
                current_probs = posteriors[-1]

                # Transition matrix
                trans_matrix = self._model.transmat_
                next_probs = trans_matrix[current_state]

                duration = self._estimate_regime_duration(trans_matrix, current_state)
            elif self._gmm_model is not None:
                # GMM fallback (no temporal info)
                labels = self._gmm_model.predict(normalized)
                current_state = int(labels[-1])
                current_probs = self._gmm_model.predict_proba(normalized[-1:]).flatten()
                trans_matrix = None
                next_probs = current_probs  # No temporal info
                duration = 0
            else:
                return None

            current_regime = self._regime_map.get(current_state, MarketRegime.NEUTRAL_SIDEWAYS)
            alpha_weights = self.get_alpha_weights(current_regime)

            # Build regime probability dicts
            regime_probs = {}
            next_regime_probs = {}
            for state_idx, regime in self._regime_map.items():
                if state_idx < len(current_probs):
                    regime_probs[regime.value] = float(current_probs[state_idx])
                    next_regime_probs[regime.value] = float(next_probs[state_idx])

            confidence = float(np.max(current_probs))

            return RegimeState(
                current_regime=current_regime,
                regime_probabilities=regime_probs,
                transition_matrix=trans_matrix,
                next_regime_probs=next_regime_probs,
                regime_duration_estimate=duration,
                alpha_weights=alpha_weights,
                confidence=confidence,
                features_used=self.FEATURE_NAMES,
            )
        except Exception as e:
            self.logger.error(f"Regime prediction failed: {e}")
            return None

    def maybe_refit(self, price_df: pd.DataFrame, cycle_count: int) -> bool:
        """Refit if cycle_count is at a refit interval boundary."""
        if cycle_count <= 0:
            return False
        if cycle_count % self.refit_interval != 0:
            return False
        if cycle_count == self._last_fit_cycle:
            return False
        self._last_fit_cycle = cycle_count
        return self.fit(price_df)

    def _label_regimes(self, hidden_states: np.ndarray, features: np.ndarray) -> Dict[int, MarketRegime]:
        """
        Map HMM state indices to semantic MarketRegime labels
        by analyzing mean returns and volatility per state.
        """
        returns = features[:, 0]  # log_returns
        volatility = features[:, 1]  # realized_volatility

        state_stats = {}
        for state in range(self.n_regimes):
            mask = hidden_states == state
            if mask.sum() < 5:
                state_stats[state] = (0.0, 0.0)
                continue
            state_stats[state] = (
                float(np.nanmean(returns[mask])),
                float(np.nanmean(volatility[mask])),
            )

        # Sort states by mean return
        sorted_states = sorted(state_stats.items(), key=lambda x: x[1][0])

        # Compute median volatility as threshold
        all_vols = [v for _, (_, v) in sorted_states if not np.isnan(v)]
        med_vol = np.median(all_vols) if all_vols else 0.0

        # Assign labels based on return rank and volatility
        regimes = [
            MarketRegime.BEAR_TRENDING,
            MarketRegime.BEAR_MEAN_REVERTING,
            MarketRegime.NEUTRAL_SIDEWAYS,
            MarketRegime.BULL_MEAN_REVERTING,
            MarketRegime.BULL_TRENDING,
        ]

        regime_map = {}
        for rank, (state_idx, (mean_ret, mean_vol)) in enumerate(sorted_states):
            if rank < len(regimes):
                assigned = regimes[rank]
                # Refine: if high vol in a middle rank, swap to trending variant
                if mean_vol > med_vol * 1.2 and assigned in (
                    MarketRegime.BULL_MEAN_REVERTING, MarketRegime.BEAR_MEAN_REVERTING
                ):
                    if mean_ret > 0:
                        assigned = MarketRegime.BULL_TRENDING
                    else:
                        assigned = MarketRegime.BEAR_TRENDING
                regime_map[state_idx] = assigned
            else:
                regime_map[state_idx] = MarketRegime.NEUTRAL_SIDEWAYS

        return regime_map

    def _estimate_regime_duration(self, transition_matrix: np.ndarray, current_state: int) -> int:
        """Expected duration in current state = 1 / (1 - P(i,i))."""
        if transition_matrix is None:
            return 0
        p_stay = transition_matrix[current_state, current_state]
        if p_stay >= 1.0:
            return 999
        return max(1, int(1.0 / (1.0 - p_stay + 1e-9)))

    def get_alpha_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Return regime-specific alpha weight multipliers."""
        return REGIME_ALPHA_WEIGHTS.get(regime, {
            "momentum_boost": 1.0,
            "mean_rev_boost": 1.0,
            "volatility_boost": 1.0,
            "flow_boost": 1.0,
        })
