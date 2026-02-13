"""
Correlation Network Engine: Multi-Asset Correlation Discovery & Divergence Alpha
Tracks 20-50 crypto assets, computes rolling correlation matrices, identifies
correlation clusters via hierarchical clustering, and generates divergence signals.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from collections import deque

try:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class CorrelationNetworkEngine:
    """
    Multi-asset correlation network that discovers correlation clusters
    and generates divergence-based alpha signals.
    """

    def __init__(self, config: Dict[str, Any] = None, logger: Optional[logging.Logger] = None):
        config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = config.get("enabled", False)
        self.max_assets = config.get("max_assets", 30)
        self.rolling_window = config.get("rolling_window", 1440)
        self.update_interval = config.get("update_interval_cycles", 5)
        self.min_history = config.get("min_history_length", 60)
        self.divergence_threshold = config.get("divergence_zscore_threshold", 2.0)
        self.cluster_threshold = config.get("cluster_distance_threshold", 0.5)

        # Per-asset price history
        self._price_history: Dict[str, deque] = {}

        # Cached results
        self._cached_corr_matrix: Optional[pd.DataFrame] = None
        self._cached_clusters: Dict[int, List[str]] = {}
        self._cached_divergences: Dict[str, float] = {}
        self._last_compute_cycle = -1

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Batch update prices for all tracked assets."""
        if not self.enabled:
            return
        for product_id, price in prices.items():
            if price <= 0:
                continue
            if product_id not in self._price_history:
                self._price_history[product_id] = deque(maxlen=self.rolling_window)
            self._price_history[product_id].append(price)

    def update_price(self, product_id: str, price: float) -> None:
        """Update price for a single asset."""
        if not self.enabled or price <= 0:
            return
        if product_id not in self._price_history:
            self._price_history[product_id] = deque(maxlen=self.rolling_window)
        self._price_history[product_id].append(price)

    def should_recompute(self, cycle_count: int) -> bool:
        """Returns True if matrix should be recomputed this cycle."""
        if not self.enabled:
            return False
        if cycle_count - self._last_compute_cycle >= self.update_interval:
            return True
        return False

    def compute_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Compute NxN Pearson correlation matrix from log-returns."""
        # Filter assets with sufficient history
        eligible = {
            pid: list(hist)
            for pid, hist in self._price_history.items()
            if len(hist) >= self.min_history
        }

        if len(eligible) < 3:
            return None

        # Align to common length
        min_len = min(len(v) for v in eligible.values())
        if min_len < self.min_history:
            return None

        product_ids = sorted(eligible.keys())[:self.max_assets]

        # Build returns matrix
        returns_matrix = []
        for pid in product_ids:
            prices = np.array(eligible[pid][-min_len:])
            log_returns = np.diff(np.log(np.maximum(prices, 1e-9)))
            returns_matrix.append(log_returns)

        returns_array = np.array(returns_matrix)

        # Correlation matrix
        corr_matrix = np.corrcoef(returns_array)

        # Handle NaN
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        self._cached_corr_matrix = pd.DataFrame(
            corr_matrix, index=product_ids, columns=product_ids
        )
        return self._cached_corr_matrix

    def identify_clusters(self, corr_matrix: pd.DataFrame) -> Dict[int, List[str]]:
        """Hierarchical clustering on 1 - abs(correlation) distance matrix."""
        if not SCIPY_AVAILABLE or corr_matrix is None or len(corr_matrix) < 3:
            return {}

        try:
            # Distance = 1 - abs(correlation)
            dist_matrix = 1.0 - np.abs(corr_matrix.values)
            np.fill_diagonal(dist_matrix, 0.0)

            # Ensure symmetry and valid distances
            dist_matrix = np.maximum(dist_matrix, 0.0)
            dist_matrix = (dist_matrix + dist_matrix.T) / 2.0

            condensed = squareform(dist_matrix, checks=False)
            Z = linkage(condensed, method="average")
            labels = fcluster(Z, t=self.cluster_threshold, criterion="distance")

            clusters: Dict[int, List[str]] = {}
            for idx, cluster_id in enumerate(labels):
                cid = int(cluster_id)
                if cid not in clusters:
                    clusters[cid] = []
                clusters[cid].append(corr_matrix.index[idx])

            self._cached_clusters = clusters
            return clusters
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            return {}

    def detect_divergences(self, corr_matrix: pd.DataFrame,
                           clusters: Dict[int, List[str]]) -> Dict[str, float]:
        """
        For each asset, compare its current return to cluster mean return.
        Positive signal = asset underperforming cluster (buy).
        Negative signal = asset outperforming cluster (sell).
        """
        if not clusters:
            self._cached_divergences = {}
            return {}

        # Compute recent returns for each asset
        recent_returns: Dict[str, float] = {}
        lookback = min(20, self.min_history)
        for pid, hist in self._price_history.items():
            if len(hist) >= lookback + 1:
                prices = list(hist)
                ret = np.log(prices[-1] / max(prices[-lookback], 1e-9))
                recent_returns[pid] = float(ret)

        divergences: Dict[str, float] = {}
        for cluster_id, members in clusters.items():
            if len(members) < 2:
                continue

            # Cluster mean return
            member_returns = [recent_returns.get(m, 0.0) for m in members]
            cluster_mean = np.mean(member_returns)
            cluster_std = np.std(member_returns) + 1e-9

            for member in members:
                member_ret = recent_returns.get(member, 0.0)
                z = (member_ret - cluster_mean) / cluster_std

                # Divergence signal: negative of z (mean-reversion logic)
                if abs(z) > self.divergence_threshold:
                    signal = float(np.clip(-z / self.divergence_threshold, -1.0, 1.0))
                else:
                    signal = 0.0
                divergences[member] = signal

        self._cached_divergences = divergences
        return divergences

    def get_correlation_divergence_signal(self, product_id: str) -> float:
        """Return cached divergence signal for a specific product."""
        if not self.enabled:
            return 0.0
        return self._cached_divergences.get(product_id, 0.0)

    def get_network_summary(self) -> Dict[str, Any]:
        """Summary for dashboard logging."""
        return {
            "tracked_assets": len(self._price_history),
            "num_clusters": len(self._cached_clusters),
            "active_divergences": sum(1 for v in self._cached_divergences.values() if abs(v) > 0.01),
            "top_divergences": sorted(
                self._cached_divergences.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5],
        }

    def run_full_update(self, cycle_count: int) -> None:
        """Convenience method to run the full compute pipeline."""
        if not self.should_recompute(cycle_count):
            return
        self._last_compute_cycle = cycle_count
        corr_matrix = self.compute_correlation_matrix()
        if corr_matrix is not None:
            clusters = self.identify_clusters(corr_matrix)
            self.detect_divergences(corr_matrix, clusters)
