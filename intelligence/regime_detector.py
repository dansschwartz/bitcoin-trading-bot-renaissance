"""
Intelligence Layer B1: Market Regime Detector
==============================================
Standalone 3-state Hidden Markov Model for market regime classification.

Regimes:
    0 - low_volatility   : Quiet, range-bound markets
    1 - trending          : Directional momentum (bull or bear)
    2 - high_volatility   : Stressed / chaotic markets

Each regime carries per-signal-type weight multipliers so that the
execution layer can scale position sizes and strategy allocations
according to the prevailing market environment.

Uses hmmlearn.hmm.GaussianHMM with graceful fallback when unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sqlite3
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Suppress hmmlearn convergence and covariance warnings (expected for new pairs with <50 bars)
warnings.filterwarnings("ignore", message=".*covars.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Fitting.*converge.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*KMeans.*converge.*", category=UserWarning)
logging.getLogger('hmmlearn').setLevel(logging.ERROR)

import numpy as np

# ---------------------------------------------------------------------------
# Graceful hmmlearn import
# ---------------------------------------------------------------------------
try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    GaussianHMM = None  # type: ignore[assignment,misc]
    HMMLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    3-state Hidden Markov Model regime detector backed by SQLite market data.

    Config key: ``medallion_regime_detector``
    """

    # Semantic labels for each HMM state index
    REGIMES: Dict[int, str] = {
        0: "low_volatility",
        1: "trending",
        2: "high_volatility",
    }

    # Per-regime signal-type weight multipliers.
    # Values > 1 amplify; values < 1 attenuate.
    REGIME_SIGNAL_WEIGHTS: Dict[str, Dict[str, float]] = {
        "low_volatility": {
            "cross_exchange_arb": 1.3,
            "funding_rate_arb": 1.4,
            "triangular_arb": 1.2,
            "mean_reversion": 1.5,
        },
        "trending": {
            "cross_exchange_arb": 0.8,
            "funding_rate_arb": 0.9,
            "triangular_arb": 0.7,
            "mean_reversion": 0.5,
        },
        "high_volatility": {
            "cross_exchange_arb": 1.1,
            "funding_rate_arb": 0.6,
            "triangular_arb": 0.9,
            "mean_reversion": 0.4,
        },
    }

    # Neutral weights returned when the model is unavailable
    _NEUTRAL_WEIGHTS: Dict[str, float] = {
        "cross_exchange_arb": 1.0,
        "funding_rate_arb": 1.0,
        "triangular_arb": 1.0,
        "mean_reversion": 1.0,
    }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, config: Dict[str, Any], db_path: str):
        """
        Parameters
        ----------
        config : dict
            Section ``medallion_regime_detector`` from config.json.
        db_path : str
            Filesystem path to the SQLite database.
        """
        self.logger = logging.getLogger(f"{__name__}.RegimeDetector")

        self.enabled: bool = config.get("enabled", True)
        self.n_states: int = config.get("n_states", 3)
        self.retrain_interval_hours: int = config.get("retrain_interval_hours", 168)
        self.lookback_days: int = config.get("lookback_days", 90)
        self.prediction_interval_seconds: int = config.get("prediction_interval_seconds", 300)
        self.model_path: str = config.get("model_path", "models/hmm_regime.pkl")
        self.confidence_threshold: float = config.get("confidence_threshold", 0.6)
        self.high_vol_action: str = config.get("high_vol_action", "reduce_50pct")

        self.db_path: str = db_path

        # Model state
        self._model: Optional[Any] = None  # GaussianHMM instance
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._last_train_time: Optional[datetime] = None
        self._current_regime_id: int = 0
        self._current_regime_name: str = self.REGIMES.get(0, "low_volatility")
        self._current_probabilities: Optional[np.ndarray] = None

        if not HMMLEARN_AVAILABLE:
            self.logger.warning(
                "hmmlearn is not installed. RegimeDetector will return neutral "
                "weights for all signals. Install with: pip install hmmlearn"
            )

        self.logger.info(
            "RegimeDetector initialized (n_states=%d, lookback=%d days, "
            "retrain_interval=%d hrs, hmmlearn=%s)",
            self.n_states, self.lookback_days, self.retrain_interval_hours,
            HMMLEARN_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _get_connection(self):
        """Yield a SQLite connection with WAL mode and reasonable timeout."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _prepare_features(self, lookback_days: int = 90) -> np.ndarray:
        """
        Build a feature matrix from the ``market_data`` table.

        Features (per row):
            0. log_returns      - log(price_t / price_{t-1})
            1. volatility       - 20-period rolling std of log returns
            2. volume_ratio     - volume_t / rolling_mean(volume, 20)
            3. spread_ratio     - spread / price

        Returns
        -------
        np.ndarray of shape (N, 4) or empty (0, 4) if insufficient data.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT price, volume, spread, timestamp
                    FROM market_data
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                    """,
                    (cutoff,),
                )
                rows = cursor.fetchall()
        except Exception as exc:
            self.logger.error("Failed to query market_data: %s", exc)
            return np.empty((0, 4))

        if len(rows) < 50:
            self.logger.warning(
                "Insufficient market_data rows (%d) for feature preparation.", len(rows)
            )
            return np.empty((0, 4))

        prices = np.array([r[0] for r in rows], dtype=np.float64)
        volumes = np.array([r[1] for r in rows], dtype=np.float64)
        spreads = np.array([r[2] for r in rows], dtype=np.float64)

        # Avoid division by zero / log of zero
        prices = np.maximum(prices, 1e-9)
        volumes = np.maximum(volumes, 1e-9)

        # 1. Log returns
        log_returns = np.diff(np.log(prices))

        # 2. Rolling volatility (20-period std of log returns)
        window = 20
        if len(log_returns) < window:
            self.logger.warning("Not enough data for rolling volatility window.")
            return np.empty((0, 4))

        volatility = np.array([
            np.std(log_returns[max(0, i - window + 1): i + 1])
            for i in range(len(log_returns))
        ], dtype=np.float64)

        # 3. Volume ratio (current / 20-period rolling mean)
        volumes_aligned = volumes[1:]  # align with log_returns (N-1)
        rolling_vol_mean = np.array([
            np.mean(volumes_aligned[max(0, i - window + 1): i + 1])
            for i in range(len(volumes_aligned))
        ], dtype=np.float64)
        rolling_vol_mean = np.maximum(rolling_vol_mean, 1e-9)
        volume_ratio = volumes_aligned / rolling_vol_mean

        # 4. Spread ratio (spread / price)
        spreads_aligned = spreads[1:]
        prices_aligned = prices[1:]
        spread_ratio = spreads_aligned / np.maximum(prices_aligned, 1e-9)

        # Stack into feature matrix
        min_len = min(len(log_returns), len(volatility), len(volume_ratio), len(spread_ratio))
        features = np.column_stack([
            log_returns[:min_len],
            volatility[:min_len],
            volume_ratio[:min_len],
            spread_ratio[:min_len],
        ])

        # Drop rows with NaN or Inf
        valid_mask = np.isfinite(features).all(axis=1)
        features = features[valid_mask]

        self.logger.info(
            "Prepared feature matrix: %d rows x %d cols from %d raw market_data rows.",
            features.shape[0], features.shape[1], len(rows),
        )
        return features

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> bool:
        """
        Train the GaussianHMM on historical market data features.

        Returns True on success, False otherwise.
        """
        if not HMMLEARN_AVAILABLE:
            self.logger.warning("Cannot train: hmmlearn is not installed.")
            return False

        features = self._prepare_features(lookback_days=self.lookback_days)
        if features.shape[0] < 50:
            self.logger.warning(
                "Cannot train HMM: only %d valid samples (need >= 50).", features.shape[0]
            )
            return False

        # Normalize features for numerical stability
        self._feature_means = features.mean(axis=0)
        self._feature_stds = features.std(axis=0) + 1e-9
        normalized = (features - self._feature_means) / self._feature_stds

        try:
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=1000,
                random_state=42,
                tol=0.01,
            )
            model.fit(normalized)

            # Regularise transition matrix — fix degenerate rows
            tm = model.transmat_.copy()
            row_sums = tm.sum(axis=1)
            for i in range(len(row_sums)):
                if row_sums[i] < 1e-10:
                    tm[i] = 1.0 / self.n_states
            tm = tm / tm.sum(axis=1, keepdims=True)
            model.transmat_ = tm

            self._model = model
            self._label_states(normalized)
            self._last_train_time = datetime.now(timezone.utc)

            self.logger.info(
                "HMM trained successfully: %d states, %d samples, score=%.4f",
                self.n_states, features.shape[0], model.score(normalized),
            )
            return True

        except Exception as exc:
            self.logger.error("HMM training failed: %s", exc, exc_info=True)
            return False

    def _label_states(self, normalized: np.ndarray) -> None:
        """
        Map fitted HMM state indices to semantic regime labels by
        examining mean volatility (feature column 1) per state.

        State ordering by volatility:
            lowest  -> low_volatility (0)
            middle  -> trending       (1)
            highest -> high_volatility(2)
        """
        if self._model is None:
            return

        hidden_states = self._model.predict(normalized)
        state_vol: Dict[int, float] = {}
        for state in range(self.n_states):
            mask = hidden_states == state
            if mask.sum() > 0:
                state_vol[state] = float(np.mean(normalized[mask, 1]))
            else:
                state_vol[state] = 0.0

        # Sort states by mean volatility, ascending
        sorted_states = sorted(state_vol.items(), key=lambda x: x[1])

        # Build mapping: raw_state_idx -> semantic_regime_id
        self._state_to_regime: Dict[int, int] = {}
        regime_ids = list(range(self.n_states))  # [0, 1, 2]
        for rank, (raw_state, _) in enumerate(sorted_states):
            if rank < len(regime_ids):
                self._state_to_regime[raw_state] = regime_ids[rank]
            else:
                self._state_to_regime[raw_state] = regime_ids[-1]

        self.logger.info(
            f"State-to-regime mapping: {self._state_to_regime!r} (by volatility ascending)"
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_current_regime(self) -> Dict[str, Any]:
        """
        Predict the current market regime from the latest features.

        Returns
        -------
        dict with keys:
            regime_name, regime_id, probabilities, confidence,
            signal_weights, recommendation, timestamp
        """
        result_template: Dict[str, Any] = {
            "regime_name": "unknown",
            "regime_id": -1,
            "probabilities": {},
            "confidence": 0.0,
            "signal_weights": dict(self._NEUTRAL_WEIGHTS),
            "recommendation": "neutral",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if not HMMLEARN_AVAILABLE or self._model is None:
            self.logger.debug("No trained model available; returning neutral prediction.")
            return result_template

        # Prepare a short recent feature window
        features = self._prepare_features(lookback_days=7)
        if features.shape[0] < 5:
            self.logger.warning("Insufficient recent data for prediction.")
            return result_template

        try:
            normalized = (features - self._feature_means) / self._feature_stds

            # Predict hidden states and posterior probabilities
            hidden_states = self._model.predict(normalized)
            posteriors = self._model.predict_proba(normalized)

            raw_state = int(hidden_states[-1])
            current_probs = posteriors[-1]

            # Map raw HMM state to semantic regime
            regime_id = getattr(self, "_state_to_regime", {}).get(raw_state, raw_state % self.n_states)
            regime_name = self.REGIMES.get(regime_id, "unknown")

            confidence = float(np.max(current_probs))

            # Build probability dict keyed by regime name
            prob_dict: Dict[str, float] = {}
            state_to_regime = getattr(self, "_state_to_regime", {})
            for raw_idx in range(self.n_states):
                mapped_id = state_to_regime.get(raw_idx, raw_idx)
                mapped_name = self.REGIMES.get(mapped_id, f"state_{mapped_id}")
                if raw_idx < len(current_probs):
                    prob_dict[mapped_name] = float(current_probs[raw_idx])

            # Signal weights for the detected regime
            signal_weights = self.REGIME_SIGNAL_WEIGHTS.get(regime_name, dict(self._NEUTRAL_WEIGHTS))

            # Recommendation based on regime
            if regime_name == "high_volatility":
                recommendation = self.high_vol_action
            elif regime_name == "trending":
                recommendation = "follow_trend"
            else:
                recommendation = "normal_operation"

            # Update internal state
            self._current_regime_id = regime_id
            self._current_regime_name = regime_name
            self._current_probabilities = current_probs

            return {
                "regime_name": regime_name,
                "regime_id": regime_id,
                "probabilities": prob_dict,
                "confidence": confidence,
                "signal_weights": signal_weights,
                "recommendation": recommendation,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as exc:
            self.logger.error("Regime prediction failed: %s", exc, exc_info=True)
            return result_template

    # ------------------------------------------------------------------
    # Signal weight accessor
    # ------------------------------------------------------------------

    def get_signal_weight(self, signal_type: str) -> float:
        """
        Return the weight multiplier for *signal_type* under the current regime.

        If no model is trained or the signal type is unknown, returns 1.0
        (neutral).
        """
        if self._model is None or not HMMLEARN_AVAILABLE:
            return 1.0

        regime_name = self._current_regime_name
        weights = self.REGIME_SIGNAL_WEIGHTS.get(regime_name, self._NEUTRAL_WEIGHTS)
        return weights.get(signal_type, 1.0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self) -> bool:
        """
        Pickle the trained HMM model and normalization parameters to disk.

        Saves to the path specified in config (default ``models/hmm_regime.pkl``).
        Returns True on success.
        """
        if self._model is None:
            self.logger.warning("No trained model to save.")
            return False

        model_dir = os.path.dirname(self.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        payload = {
            "model": self._model,
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
            "state_to_regime": getattr(self, "_state_to_regime", {}),
            "n_states": self.n_states,
            "last_train_time": self._last_train_time.isoformat() if self._last_train_time else None,
        }

        try:
            with open(self.model_path, "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info("Model saved to %s", self.model_path)
            return True
        except Exception as exc:
            self.logger.error("Failed to save model: %s", exc)
            return False

    def load_model(self) -> bool:
        """
        Load a previously saved HMM model from disk.

        Returns True on success.
        """
        if not os.path.isfile(self.model_path):
            self.logger.info("No saved model found at %s", self.model_path)
            return False

        try:
            with open(self.model_path, "rb") as fh:
                payload = pickle.load(fh)

            self._model = payload["model"]
            self._feature_means = payload["feature_means"]
            self._feature_stds = payload["feature_stds"]
            self._state_to_regime = payload.get("state_to_regime", {})
            self.n_states = payload.get("n_states", self.n_states)

            last_train_str = payload.get("last_train_time")
            if last_train_str:
                self._last_train_time = datetime.fromisoformat(last_train_str)

            self.logger.info(
                "Model loaded from %s (trained at %s)", self.model_path, last_train_str
            )
            return True
        except Exception as exc:
            self.logger.error("Failed to load model from %s: %s", self.model_path, exc)
            return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def needs_retrain(self) -> bool:
        """Return True if the model has never been trained or is stale."""
        if self._model is None or self._last_train_time is None:
            return True
        elapsed = datetime.now(timezone.utc) - self._last_train_time.replace(tzinfo=timezone.utc)
        return elapsed > timedelta(hours=self.retrain_interval_hours)

    def __repr__(self) -> str:
        return (
            f"<RegimeDetector states={self.n_states} "
            f"current={self._current_regime_name} "
            f"hmmlearn={HMMLEARN_AVAILABLE} "
            f"trained={self._model is not None}>"
        )
