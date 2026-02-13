"""N-gram pattern generator: treats discretised returns as a language.

Builds conditional transition probabilities P(bin_t | bin_{t-1}, ..., bin_{t-n+1})
and generates new price paths by sampling from these learned patterns.
"""

import logging
import numpy as np
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from sim_models_base import SimulationModel


class NGramSimulator(SimulationModel):
    """N-gram model for return-sequence generation.

    Calibration:
        1. Quantile-bin daily log returns into *n_bins* buckets.
        2. Build an (n-1)-gram → next-bin transition table.

    Simulation:
        1. Pick a random starting context from empirical n-grams.
        2. At each step, look up the context in the table and sample.
        3. Map sampled bin back to a continuous return (uniform within bin).
        4. Fallback to (n-2)-gram, then uniform, if context unseen.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self._n: int = self.config.get("n", 3)
        self._n_bins: int = self.config.get("n_bins", 20)

        self._bin_edges: Optional[np.ndarray] = None
        self._transition_probs: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self._all_contexts: List[tuple] = []

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, returns: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        n = self._n
        n_bins = self._n_bins

        if len(returns) < n + 5:
            self.logger.warning("Insufficient data for n-gram calibration")
            self._set_fallback(returns)
            return self._parameters

        # Quantile-based binning (edges include -inf / +inf)
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(returns, quantiles)
        edges[0] = -np.inf
        edges[-1] = np.inf
        self._bin_edges = edges

        # Digitise: each return → bin index [0, n_bins-1]
        digitised = np.digitize(returns, edges[1:-1])  # 0..n_bins-1

        # Build transition table
        transitions: Dict[tuple, Counter] = defaultdict(Counter)
        for i in range(n - 1, len(digitised)):
            context = tuple(digitised[i - n + 1: i])
            next_bin = int(digitised[i])
            transitions[context][next_bin] += 1

        self._transition_probs = {}
        for context, counter in transitions.items():
            bins = np.array(list(counter.keys()))
            counts = np.array(list(counter.values()), dtype=float)
            probs = counts / counts.sum()
            self._transition_probs[context] = (bins, probs)

        self._all_contexts = list(self._transition_probs.keys())

        self._parameters = {
            "n": n,
            "n_bins": n_bins,
            "n_unique_contexts": len(self._all_contexts),
            "n_observations": len(returns),
        }
        self._calibrated = True
        self.logger.info(
            f"N-gram(n={n}) calibrated: {len(self._all_contexts)} unique "
            f"contexts from {len(returns)} returns, {n_bins} bins"
        )
        return self._parameters

    def _set_fallback(self, returns: np.ndarray) -> None:
        """Fallback: treat all returns as equally likely bins."""
        self._bin_edges = np.array([-np.inf, np.inf])
        self._transition_probs = {}
        self._all_contexts = []
        self._parameters = {"n": 1, "n_bins": 1, "n_unique_contexts": 0,
                            "n_observations": len(returns), "method": "fallback"}
        self._calibrated = True
        # Store empirical returns for raw bootstrap in simulate
        self._fallback_returns = returns.copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bin_to_return(self, bin_idx: int, rng: np.random.Generator) -> float:
        """Sample a return uniformly within the given bin's range."""
        edges = self._bin_edges
        if edges is None or len(edges) < 2:
            return 0.0
        low = edges[bin_idx] if bin_idx < len(edges) else edges[-2]
        high = edges[bin_idx + 1] if (bin_idx + 1) < len(edges) else edges[-1]
        # Clip infinite edges to reasonable bounds
        low = max(float(low), -0.5)
        high = min(float(high), 0.5)
        if low >= high:
            return float(low)
        return float(rng.uniform(low, high))

    def _sample_next_bin(self, context: tuple,
                         rng: np.random.Generator) -> Optional[int]:
        """Lookup context in transition table; try shorter contexts as fallback."""
        if context in self._transition_probs:
            bins, probs = self._transition_probs[context]
            return int(rng.choice(bins, p=probs))

        # Backoff: try shorter context
        for trim in range(1, len(context)):
            shorter = context[trim:]
            if shorter in self._transition_probs:
                bins, probs = self._transition_probs[shorter]
                return int(rng.choice(bins, p=probs))

        return None  # no match at all

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, S0: float, n_steps: int, n_simulations: int,
                 dt: float = 1.0 / 252, seed: Optional[int] = None) -> np.ndarray:
        if not self._calibrated:
            raise RuntimeError("Model not calibrated — call calibrate() first")

        rng = np.random.default_rng(seed)
        n = self._parameters.get("n", self._n)
        n_bins = self._parameters.get("n_bins", self._n_bins)

        # Fallback path
        if not self._all_contexts:
            fb = getattr(self, "_fallback_returns", None)
            if fb is not None and len(fb) > 0:
                idx = rng.integers(0, len(fb), (n_simulations, n_steps))
                daily = fb[idx]
            else:
                daily = rng.normal(0, 0.02, (n_simulations, n_steps))
            log_paths = np.cumsum(daily, axis=1)
            return S0 * np.exp(
                np.column_stack([np.zeros(n_simulations), log_paths])
            )

        S = np.zeros((n_simulations, n_steps + 1))
        S[:, 0] = S0

        for i in range(n_simulations):
            # Random starting context
            ctx = list(self._all_contexts[rng.integers(len(self._all_contexts))])
            for t in range(n_steps):
                context_tuple = tuple(ctx[-(n - 1):]) if n > 1 else ()
                next_bin = self._sample_next_bin(context_tuple, rng)
                if next_bin is None:
                    next_bin = rng.integers(0, n_bins)
                ret = self._bin_to_return(next_bin, rng)
                S[i, t + 1] = S[i, t] * np.exp(ret)
                ctx.append(next_bin)

        return S
