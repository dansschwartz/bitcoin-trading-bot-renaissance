"""
Unified Portfolio Engine — Cross-Product Portfolio Optimization
================================================================
Treats all trading products as ONE portfolio, applying correlation-aware
sizing and portfolio-level risk constraints.

"Medallion never looked at a single instrument in isolation.
 Everything was part of the portfolio."

This engine sits between signal generation and execution:
1. Collect signals for ALL products
2. Apply portfolio-level constraints (correlation, risk budget)
3. Output adjusted decisions

Starts simple (correlation limits) — can evolve to full mean-variance optimization.
"""

from __future__ import annotations

import logging
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List, Tuple


class UnifiedPortfolioEngine:
    """
    Cross-product portfolio optimizer.

    Usage:
        engine = UnifiedPortfolioEngine(config, logger)
        adjusted = engine.optimize(
            product_signals={"BTC-USD": (0.15, 0.8), "ETH-USD": (0.10, 0.7)},
            current_positions={"BTC-USD": 500.0, "ETH-USD": 300.0},
            total_balance=10000.0,
        )
        # Returns: {"BTC-USD": {"size_multiplier": 0.7, "reason": "corr_reduction"},
        #           "ETH-USD": {"size_multiplier": 0.5, "reason": "corr_reduction"}}
    """

    def __init__(self, config: Dict[str, Any] = None, logger: Optional[logging.Logger] = None):
        config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Correlation threshold — reduce combined size above this
        self._corr_threshold = float(config.get("correlation_threshold", 0.75))
        self._corr_penalty = float(config.get("correlation_penalty", 0.5))  # Multiply by this when corr > threshold

        # Max portfolio concentration — no single product > this % of total exposure
        self._max_concentration = float(config.get("max_concentration", 0.40))  # 40%

        # Max total exposure as fraction of balance
        self._max_total_exposure = float(config.get("max_total_exposure_pct", 0.80))  # 80%

        # Rolling return history for correlation computation
        self._return_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._last_prices: Dict[str, float] = {}

        # Cached correlation matrix
        self._corr_matrix: Dict[Tuple[str, str], float] = {}
        self._corr_update_cycle = 0
        self._corr_update_interval = int(config.get("corr_update_interval", 20))

        self.logger.info(
            f"UnifiedPortfolioEngine initialized: "
            f"corr_threshold={self._corr_threshold}, "
            f"max_concentration={self._max_concentration:.0%}"
        )

    def update_price(self, product_id: str, price: float):
        """Record a price observation for correlation computation."""
        last = self._last_prices.get(product_id)
        if last and last > 0 and price > 0:
            ret = (price - last) / last
            self._return_history[product_id].append(ret)
        self._last_prices[product_id] = price

    def optimize(
        self,
        product_signals: Dict[str, Tuple[float, float]],
        current_positions: Dict[str, float],
        total_balance: float,
        cycle_count: int = 0,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Apply portfolio-level constraints to per-product signals.

        Args:
            product_signals: {product_id: (weighted_signal, confidence)}
            current_positions: {product_id: current_exposure_usd}
            total_balance: Total account balance USD
            cycle_count: Current cycle number (for periodic correlation recompute)

        Returns:
            {product_id: {"size_multiplier": float, "reason": str}}
        """
        result = {}

        # Update correlation matrix periodically
        if cycle_count - self._corr_update_cycle >= self._corr_update_interval:
            self._update_correlations()
            self._corr_update_cycle = cycle_count

        products = list(product_signals.keys())

        for product_id in products:
            signal, confidence = product_signals[product_id]
            multiplier = 1.0
            reasons = []

            # 1. Concentration check
            current_exp = current_positions.get(product_id, 0.0)
            total_exp = sum(current_positions.values())
            if total_exp > 0 and current_exp / total_exp > self._max_concentration:
                multiplier *= 0.5
                reasons.append(f"concentration={current_exp/total_exp:.0%}")

            # 2. Total exposure check
            if total_balance > 0 and total_exp / total_balance > self._max_total_exposure:
                multiplier *= 0.5
                reasons.append(f"total_exposure={total_exp/total_balance:.0%}")

            # 3. Correlation-aware sizing
            # If another product with same signal direction has high correlation,
            # reduce both to avoid concentrated bets
            corr_penalty_applied = False
            for other_id in products:
                if other_id == product_id:
                    continue
                other_signal, _ = product_signals[other_id]

                # Both same direction?
                if np.sign(signal) == np.sign(other_signal) and abs(signal) > 0.01 and abs(other_signal) > 0.01:
                    corr = self._get_correlation(product_id, other_id)
                    if corr > self._corr_threshold:
                        multiplier *= self._corr_penalty
                        corr_penalty_applied = True
                        reasons.append(f"corr({other_id})={corr:.2f}")

            if corr_penalty_applied and not reasons:
                reasons.append("corr_reduction")

            result[product_id] = {
                "size_multiplier": round(multiplier, 3),
                "reason": ", ".join(reasons) if reasons else "no_adjustment",
            }

        # Log adjustments
        adjusted = {k: v for k, v in result.items() if v["size_multiplier"] < 1.0}
        if adjusted:
            adj_str = ", ".join(f"{k}={v['size_multiplier']:.2f}" for k, v in adjusted.items())
            self.logger.info(f"PORTFOLIO ENGINE: Adjusted sizes: {adj_str}")

        return result

    def _update_correlations(self):
        """Recompute pairwise correlation matrix from recent returns."""
        products = [p for p in self._return_history if len(self._return_history[p]) >= 20]

        if len(products) < 2:
            return

        self._corr_matrix.clear()

        for i, p1 in enumerate(products):
            for j, p2 in enumerate(products):
                if i >= j:
                    continue

                r1 = list(self._return_history[p1])
                r2 = list(self._return_history[p2])

                # Align lengths
                min_len = min(len(r1), len(r2))
                if min_len < 20:
                    continue

                r1 = r1[-min_len:]
                r2 = r2[-min_len:]

                try:
                    corr = float(np.corrcoef(r1, r2)[0, 1])
                    if np.isnan(corr):
                        corr = 0.0
                except Exception:
                    corr = 0.0

                self._corr_matrix[(p1, p2)] = corr
                self._corr_matrix[(p2, p1)] = corr

    def _get_correlation(self, p1: str, p2: str) -> float:
        """Get cached correlation between two products."""
        return self._corr_matrix.get((p1, p2), 0.0)

    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Return the full correlation matrix for dashboard."""
        products = sorted(set(p for pair in self._corr_matrix for p in (pair[0] if isinstance(pair, tuple) else pair,)))
        # Rebuild from cache
        matrix = {}
        all_products = sorted(set(p for pair in self._corr_matrix.keys() for p in pair))
        for p1 in all_products:
            matrix[p1] = {}
            for p2 in all_products:
                if p1 == p2:
                    matrix[p1][p2] = 1.0
                else:
                    matrix[p1][p2] = self._corr_matrix.get((p1, p2), 0.0)
        return matrix

    def get_stats(self) -> Dict[str, Any]:
        """Return portfolio engine statistics."""
        return {
            "correlation_pairs": len(self._corr_matrix) // 2,
            "products_tracked": len(self._return_history),
            "corr_threshold": self._corr_threshold,
            "max_concentration": self._max_concentration,
        }
