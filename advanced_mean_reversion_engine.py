"""
Advanced Mean Reversion Engine
Implements Ornstein-Uhlenbeck process fitting, Engle-Granger cointegration,
Kalman filter hedge ratios, and multi-pair signal generation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Graceful imports
try:
    from statsmodels.tsa.stattools import coint, adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


@dataclass
class PairState:
    base_id: str
    target_id: str
    is_cointegrated: bool
    adf_pvalue: float
    half_life: float
    hedge_ratio: float
    hedge_ratio_std: float
    z_score: float
    signal: float
    entry_threshold: float
    exit_threshold: float
    spread_mean: float
    spread_std: float


@dataclass
class MeanReversionPortfolio:
    pair_signals: List[PairState]
    composite_signal: float
    n_active_pairs: int
    best_pair: Optional[PairState]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AdvancedMeanReversionEngine:
    """
    Multi-pair mean reversion engine with cointegration testing,
    OU process fitting, and Kalman filter hedge ratios.
    """

    def __init__(self, config: Dict[str, Any] = None, logger: Optional[logging.Logger] = None):
        config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.history: Dict[str, List[float]] = {}
        self.window_size = config.get("window_size", 240)
        self.min_history = config.get("min_history", 60)
        self.coint_pvalue_threshold = config.get("coint_pvalue", 0.05)
        self.retest_interval = config.get("retest_interval", 100)
        self.entry_z = config.get("entry_z", 2.0)
        self.exit_z = config.get("exit_z", 0.5)
        self.max_half_life = config.get("max_half_life", 120)
        self.min_half_life = config.get("min_half_life", 5)
        self.max_pairs = config.get("max_pairs", 10)

        # Kalman filter config
        self._kalman_delta = config.get("kalman_delta", 0.0001)
        self._kalman_ve = config.get("kalman_ve", 0.001)

        # Kalman state per pair: {pair_key: {beta, P, initialized}}
        self._kalman_state: Dict[str, Dict[str, float]] = {}

        # Cointegration cache: {(base, target): {is_coint, pvalue, last_cycle}}
        self._coint_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

        self._active_pairs: List[Tuple[str, str]] = []

    def update_price(self, product_id: str, price: float):
        """Update price history for a product."""
        if price <= 0:
            return
        if product_id not in self.history:
            self.history[product_id] = []
        self.history[product_id].append(price)
        if len(self.history[product_id]) > self.window_size:
            self.history[product_id].pop(0)

    def discover_pairs(self, product_ids: List[str], cycle_count: int) -> List[Tuple[str, str]]:
        """
        Test all combinations of product_ids for cointegration.
        Cache results and only retest every retest_interval cycles.
        """
        pairs = []
        for i in range(len(product_ids)):
            for j in range(i + 1, len(product_ids)):
                base, target = product_ids[i], product_ids[j]
                pair_key = (base, target)

                # Check cache
                cached = self._coint_cache.get(pair_key)
                if cached and (cycle_count - cached.get("last_cycle", 0)) < self.retest_interval:
                    if cached["is_coint"]:
                        pairs.append(pair_key)
                    continue

                # Test cointegration
                is_coint, pvalue = self._test_cointegration(base, target)
                self._coint_cache[pair_key] = {
                    "is_coint": is_coint,
                    "pvalue": pvalue,
                    "last_cycle": cycle_count,
                }
                if is_coint:
                    pairs.append(pair_key)

        self._active_pairs = pairs[:self.max_pairs]
        return self._active_pairs

    def _test_cointegration(self, base_id: str, target_id: str) -> Tuple[bool, float]:
        """
        Run Engle-Granger cointegration test.
        Falls back to ADF on OLS residual if coint() unavailable.
        """
        b_prices = self.history.get(base_id, [])
        t_prices = self.history.get(target_id, [])

        if len(b_prices) < self.min_history or len(t_prices) < self.min_history:
            return False, 1.0

        min_len = min(len(b_prices), len(t_prices))
        b = np.array(b_prices[-min_len:])
        t = np.array(t_prices[-min_len:])

        if STATSMODELS_AVAILABLE:
            try:
                _, pvalue, _ = coint(b, t)
                return pvalue < self.coint_pvalue_threshold, float(pvalue)
            except Exception as e:
                self.logger.debug(f"Cointegration test failed for {base_id}/{target_id}: {e}")
                return False, 1.0
        else:
            # Fallback: ADF test on OLS residual
            try:
                beta = np.cov(np.log(b), np.log(t))[0, 1] / (np.var(np.log(t)) + 1e-9)
                spread = np.log(b) - beta * np.log(t)
                result = adfuller(spread, maxlag=int(len(spread) ** 0.25))
                pvalue = float(result[1])
                return pvalue < self.coint_pvalue_threshold, pvalue
            except Exception:
                return False, 1.0

    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """
        Fit Ornstein-Uhlenbeck process to the spread.
        half_life = -log(2) / log(1 + theta)
        Uses OLS: delta_spread = theta * spread_lagged + intercept
        """
        if len(spread) < 20:
            return float("inf")

        spread_lag = spread[:-1]
        delta_spread = np.diff(spread)

        # OLS regression: delta_S = theta * S_{t-1} + c
        X = np.column_stack([spread_lag, np.ones(len(spread_lag))])
        try:
            theta_vec = np.linalg.lstsq(X, delta_spread, rcond=None)[0]
            theta = theta_vec[0]
        except Exception:
            return float("inf")

        if theta >= 0:
            return float("inf")  # Not mean-reverting

        half_life = -np.log(2) / np.log(1 + theta)
        return max(1.0, float(half_life))

    def _kalman_update(self, pair_key: str, y: float, x: float) -> Tuple[float, float]:
        """
        Kalman filter for dynamic hedge ratio.
        State: beta (hedge ratio), modeled as random walk.
        Observation: y = beta * x + noise
        """
        if pair_key not in self._kalman_state:
            # Initialize with OLS-like estimate
            self._kalman_state[pair_key] = {
                "beta": y / (x + 1e-9),
                "P": 1.0,
                "initialized": True,
            }
            return self._kalman_state[pair_key]["beta"], 1.0

        state = self._kalman_state[pair_key]
        beta_prior = state["beta"]
        P_prior = state["P"] + self._kalman_delta  # Process noise

        # Kalman gain
        S = x * P_prior * x + self._kalman_ve  # Innovation variance
        K = P_prior * x / (S + 1e-12)

        # Update
        innovation = y - beta_prior * x
        beta_post = beta_prior + K * innovation
        P_post = (1 - K * x) * P_prior

        self._kalman_state[pair_key] = {
            "beta": float(beta_post),
            "P": float(P_post),
            "initialized": True,
        }

        return float(beta_post), float(P_post)

    def calculate_pair_signal(self, base_id: str, target_id: str) -> PairState:
        """Calculate mean reversion signal for a single pair."""
        b_prices = self.history.get(base_id, [])
        t_prices = self.history.get(target_id, [])

        # Default inactive state
        inactive = PairState(
            base_id=base_id, target_id=target_id,
            is_cointegrated=False, adf_pvalue=1.0,
            half_life=float("inf"), hedge_ratio=1.0, hedge_ratio_std=1.0,
            z_score=0.0, signal=0.0,
            entry_threshold=self.entry_z, exit_threshold=self.exit_z,
            spread_mean=0.0, spread_std=0.0,
        )

        if len(b_prices) < self.min_history or len(t_prices) < self.min_history:
            return inactive

        min_len = min(len(b_prices), len(t_prices))
        b = np.log(np.array(b_prices[-min_len:]))
        t = np.log(np.array(t_prices[-min_len:]))

        # Kalman-filtered hedge ratio
        pair_key = f"{base_id}_{target_id}"
        hedge_ratio, hedge_std = self._kalman_update(pair_key, b[-1], t[-1])

        # Compute spread
        spread = b - hedge_ratio * t

        # Half-life
        half_life = self._calculate_half_life(spread)
        if half_life < self.min_half_life or half_life > self.max_half_life:
            inactive.hedge_ratio = hedge_ratio
            inactive.half_life = half_life
            return inactive

        # Z-score
        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread))
        if spread_std < 1e-9:
            return inactive

        z_score = float((spread[-1] - spread_mean) / spread_std)

        # Signal: mean reversion — negative z-score / entry_z, clipped
        signal = float(np.clip(-z_score / self.entry_z, -1.0, 1.0))

        # Check cointegration from cache
        cache_key = (base_id, target_id)
        cached = self._coint_cache.get(cache_key, {})

        return PairState(
            base_id=base_id,
            target_id=target_id,
            is_cointegrated=cached.get("is_coint", False),
            adf_pvalue=cached.get("pvalue", 1.0),
            half_life=half_life,
            hedge_ratio=hedge_ratio,
            hedge_ratio_std=hedge_std,
            z_score=z_score,
            signal=signal,
            entry_threshold=self.entry_z,
            exit_threshold=self.exit_z,
            spread_mean=spread_mean,
            spread_std=spread_std,
        )

    def generate_portfolio_signal(self, product_ids: List[str],
                                  cycle_count: int) -> MeanReversionPortfolio:
        """
        Main entry point. Discovers cointegrated pairs, computes signals,
        returns inverse-half-life weighted composite.
        """
        # Discover pairs periodically
        if cycle_count % self.retest_interval == 0 or not self._active_pairs:
            self.discover_pairs(product_ids, cycle_count)

        pair_signals = []
        for base_id, target_id in self._active_pairs:
            ps = self.calculate_pair_signal(base_id, target_id)
            if ps.half_life < self.max_half_life and abs(ps.signal) > 0.01:
                pair_signals.append(ps)

        if not pair_signals:
            return MeanReversionPortfolio(
                pair_signals=[],
                composite_signal=0.0,
                n_active_pairs=0,
                best_pair=None,
            )

        # Inverse half-life weighting: faster mean reversion = higher weight
        weights = np.array([1.0 / max(ps.half_life, 1.0) for ps in pair_signals])
        total_weight = weights.sum()
        if total_weight < 1e-9:
            composite = 0.0
        else:
            signals = np.array([ps.signal for ps in pair_signals])
            composite = float(np.clip(np.dot(weights, signals) / total_weight, -1.0, 1.0))

        # Best pair = highest absolute signal
        best_pair = max(pair_signals, key=lambda ps: abs(ps.signal))

        return MeanReversionPortfolio(
            pair_signals=pair_signals,
            composite_signal=composite,
            n_active_pairs=len(pair_signals),
            best_pair=best_pair,
        )
