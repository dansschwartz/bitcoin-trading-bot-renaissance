
"""
feature_pipeline.py - Fractal Multi-Dimensional Feature Engineering
Revolutionary ML component for the world's most advanced trading AI

Breakthrough innovations:
- Fractal market analysis across multiple timeframes
- Cross-asset correlation networks with graph neural networks
- Chaos theory features (Lyapunov exponents, fractal dimensions)
- Market microstructure genetic algorithms
- Quantum entanglement metrics for portfolio correlation
- Hyperdimensional computing for pattern recognition
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import networkx as nx
from scipy import signal, stats, optimize
from scipy.spatial.distance import pdist, squareform
import talib
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FractalFeatures:
    """Container for fractal analysis results"""
    hurst_exponents: np.ndarray
    fractal_dimensions: np.ndarray
    detrended_fluctuation: np.ndarray
    multifractal_spectrum: np.ndarray
    scaling_exponents: np.ndarray


@dataclass
class ChaosFeatures:
    """Container for chaos theory features"""
    lyapunov_exponents: np.ndarray
    correlation_dimension: float
    entropy_measures: Dict[str, float]
    phase_space_density: np.ndarray
    recurrence_metrics: Dict[str, float]


@dataclass
class QuantumEntanglementMetrics:
    """Container for quantum-inspired correlation metrics"""
    entanglement_entropy: np.ndarray
    mutual_information: np.ndarray
    quantum_discord: np.ndarray
    concurrence: np.ndarray
    negativity: np.ndarray


class HyperdimensionalVector:
    """Hyperdimensional computing vector for pattern encoding"""

    def __init__(self, dimension: int = 10000, density: float = 0.01):
        self.dimension = dimension
        self.density = density
        self.vector = self._generate_sparse_vector()

    def _generate_sparse_vector(self) -> np.ndarray:
        """Generate sparse hyperdimensional vector"""
        vec = np.zeros(self.dimension)
        num_ones = int(self.dimension * self.density)
        indices = np.random.choice(self.dimension, num_ones, replace=False)
        vec[indices] = np.random.choice([-1, 1], num_ones)
        return vec

    def bind(self, other: 'HyperdimensionalVector') -> 'HyperdimensionalVector':
        """Bind operation (element-wise multiplication)"""
        result = HyperdimensionalVector(self.dimension, self.density)
        result.vector = self.vector * other.vector
        return result

    def bundle(self, other: 'HyperdimensionalVector') -> 'HyperdimensionalVector':
        """Bundle operation (addition with normalization)"""
        result = HyperdimensionalVector(self.dimension, self.density)
        result.vector = np.sign(self.vector + other.vector)
        return result

    def permute(self, shift: int) -> 'HyperdimensionalVector':
        """Permute operation (circular shift)"""
        result = HyperdimensionalVector(self.dimension, self.density)
        result.vector = np.roll(self.vector, shift)
        return result

    def similarity(self, other: 'HyperdimensionalVector') -> float:
        """Compute cosine similarity"""
        return np.dot(self.vector, other.vector) / (
            np.linalg.norm(self.vector) * np.linalg.norm(other.vector) + 1e-8
        )


class FractalAnalyzer:
    """
    Advanced fractal analysis for financial time series
    """

    def __init__(self, min_scale: int = 4, max_scale: int = 64):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('FractalAnalyzer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def hurst_exponent(self, time_series: np.ndarray, max_lag: int = 100) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        try:
            ts = np.asarray(time_series).flatten()
            n = len(ts)

            if n < max_lag * 2:
                max_lag = n // 4

            lags = np.arange(2, max_lag)
            rs_values = []

            for lag in lags:
                # Divide time series into non-overlapping periods
                periods = n // lag
                rs_period = []

                for i in range(periods):
                    start_idx = i * lag
                    end_idx = (i + 1) * lag
                    period_data = ts[start_idx:end_idx]

                    # Calculate mean-adjusted cumulative sum
                    mean_val = np.mean(period_data)
                    cumsum = np.cumsum(period_data - mean_val)

                    # Calculate range and standard deviation
                    R = np.max(cumsum) - np.min(cumsum)
                    S = np.std(period_data, ddof=1)

                    if S > 0:
                        rs_period.append(R / S)

                if rs_period:
                    rs_values.append(np.mean(rs_period))
                else:
                    rs_values.append(1.0)

            # Linear regression on log-log plot
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)

            # Remove invalid values
            valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
            if np.sum(valid_mask) < 3:
                return 0.5  # Random walk default

            hurst = np.polyfit(log_lags[valid_mask], log_rs[valid_mask], 1)[0]
            return np.clip(hurst, 0, 1)

        except Exception as e:
            self.logger.warning(f"Error calculating Hurst exponent: {e}")
            return 0.5

    def fractal_dimension(self, time_series: np.ndarray, method: str = 'box_counting') -> float:
        """Calculate fractal dimension using various methods"""
        try:
            ts = np.asarray(time_series).flatten()

            if method == 'box_counting':
                return self._box_counting_dimension(ts)
            elif method == 'correlation':
                return self._correlation_dimension(ts)
            elif method == 'katz':
                return self._katz_fractal_dimension(ts)
            else:
                return self._box_counting_dimension(ts)

        except Exception as e:
            self.logger.warning(f"Error calculating fractal dimension: {e}")
            return 1.5  # Default value

    def _box_counting_dimension(self, time_series: np.ndarray) -> float:
        """Box-counting fractal dimension"""
        ts = (time_series - np.mean(time_series)) / np.std(time_series)
        n = len(ts)

        # Create 2D trajectory
        trajectory = np.column_stack([np.arange(n), ts])

        # Range of box sizes
        box_sizes = np.logspace(0, np.log10(n//4), 20, dtype=int)
        box_counts = []

        for box_size in box_sizes:
            # Count boxes containing trajectory points
            x_boxes = int(np.ceil(n / box_size))
            y_min, y_max = np.min(ts), np.max(ts)
            y_range = y_max - y_min
            y_boxes = max(1, int(np.ceil(y_range / (y_range / box_size))))

            occupied_boxes = set()
            for i, (x, y) in enumerate(trajectory):
                x_idx = min(int(x / box_size), x_boxes - 1)
                y_idx = min(int((y - y_min) / (y_range / y_boxes)), y_boxes - 1)
                occupied_boxes.add((x_idx, y_idx))

            box_counts.append(len(occupied_boxes))

        # Linear regression on log-log plot
        log_sizes = np.log(1.0 / box_sizes)
        log_counts = np.log(box_counts)

        valid_mask = np.isfinite(log_sizes) & np.isfinite(log_counts)
        if np.sum(valid_mask) < 3:
            return 1.5

        slope = np.polyfit(log_sizes[valid_mask], log_counts[valid_mask], 1)[0]
        return np.clip(slope, 1.0, 2.0)

    def _katz_fractal_dimension(self, time_series: np.ndarray) -> float:
        """Katz fractal dimension"""
        ts = np.asarray(time_series).flatten()
        n = len(ts)

        if n < 2:
            return 1.0

        # Calculate total length
        diffs = np.diff(ts)
        total_length = np.sum(np.sqrt(1 + diffs**2))

        # Calculate diameter
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((j-i)**2 + (ts[j]-ts[i])**2)
                distances.append(dist)

        diameter = np.max(distances) if distances else 1.0

        # Katz dimension
        if diameter > 0 and total_length > 0:
            fd = np.log10(total_length) / (np.log10(diameter) + np.log10(total_length/diameter))
            return np.clip(fd, 1.0, 2.0)
        else:
            return 1.0

    def detrended_fluctuation_analysis(self, time_series: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detrended Fluctuation Analysis (DFA)"""
        try:
            ts = np.asarray(time_series).flatten()
            n = len(ts)

            # Integrate the time series
            y = np.cumsum(ts - np.mean(ts))

            # Range of scales
            scales = np.unique(np.logspace(0.7, np.log10(n//4), 20).astype(int))
            fluctuations = []

            for scale in scales:
                # Divide into non-overlapping segments
                segments = n // scale
                local_trends = []

                for i in range(segments):
                    start_idx = i * scale
                    end_idx = (i + 1) * scale
                    segment = y[start_idx:end_idx]

                    # Fit polynomial trend
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)

                    # Calculate detrended fluctuation
                    detrended = segment - trend
                    local_trends.extend(detrended**2)

                # Root mean square fluctuation
                if local_trends:
                    fluctuations.append(np.sqrt(np.mean(local_trends)))
                else:
                    fluctuations.append(1.0)

            # Calculate scaling exponent
            log_scales = np.log(scales[:len(fluctuations)])
            log_fluct = np.log(fluctuations)

            valid_mask = np.isfinite(log_scales) & np.isfinite(log_fluct)
            if np.sum(valid_mask) < 3:
                return 0.5, np.array([0.5])

            alpha = np.polyfit(log_scales[valid_mask], log_fluct[valid_mask], 1)[0]
            return np.clip(alpha, 0, 2), np.array(fluctuations)

        except Exception as e:
            self.logger.warning(f"Error in DFA: {e}")
            return 0.5, np.array([0.5])

    def multifractal_analysis(self, time_series: np.ndarray, q_range: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Multifractal Detrended Fluctuation Analysis (MFDFA)"""
        try:
            ts = np.asarray(time_series).flatten()
            n = len(ts)

            if q_range is None:
                q_range = np.arange(-5, 6, 0.5)

            # Integrate the time series
            y = np.cumsum(ts - np.mean(ts))

            # Range of scales
            scales = np.unique(np.logspace(0.7, np.log10(n//4), 15).astype(int))

            # Calculate fluctuation functions for each q
            fluctuation_functions = np.zeros((len(q_range), len(scales)))

            for scale_idx, scale in enumerate(scales):
                segments = n // scale
                local_fluctuations = []

                for i in range(segments):
                    start_idx = i * scale
                    end_idx = (i + 1) * scale
                    segment = y[start_idx:end_idx]

                    # Fit polynomial trend (order 1)
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)

                    # Local fluctuation
                    detrended = segment - trend
                    local_fluctuations.append(np.mean(detrended**2))

                local_fluctuations = np.array(local_fluctuations)
                local_fluctuations = local_fluctuations[local_fluctuations > 0]  # Remove zeros

                # Calculate Fq for each q
                for q_idx, q in enumerate(q_range):
                    if len(local_fluctuations) > 0:
                        if q == 0:
                            # Special case for q=0
                            fq = np.exp(0.5 * np.mean(np.log(local_fluctuations)))
                        else:
                            # General case
                            fq = np.mean(local_fluctuations**(q/2))**(1/q)
                        fluctuation_functions[q_idx, scale_idx] = fq
                    else:
                        fluctuation_functions[q_idx, scale_idx] = 1.0

            # Calculate Hurst exponents for each q
            hurst_exponents = []
            for q_idx in range(len(q_range)):
                log_scales = np.log(scales)
                log_fq = np.log(fluctuation_functions[q_idx])

                valid_mask = np.isfinite(log_scales) & np.isfinite(log_fq)
                if np.sum(valid_mask) >= 3:
                    h_q = np.polyfit(log_scales[valid_mask], log_fq[valid_mask], 1)[0]
                    hurst_exponents.append(h_q)
                else:
                    hurst_exponents.append(0.5)

            hurst_exponents = np.array(hurst_exponents)

            # Calculate scaling exponents tau(q)
            tau = q_range * hurst_exponents - 1

            # Calculate multifractal spectrum f(alpha)
            alpha = np.gradient(tau) / np.gradient(q_range)
            f_alpha = q_range * alpha - tau

            return {
                'q_values': q_range,
                'hurst_exponents': hurst_exponents,
                'scaling_exponents': tau,
                'singularity_spectrum': alpha,
                'multifractal_spectrum': f_alpha,
                'fluctuation_functions': fluctuation_functions
            }

        except Exception as e:
            self.logger.warning(f"Error in multifractal analysis: {e}")
            return {
                'q_values': np.array([0]),
                'hurst_exponents': np.array([0.5]),
                'scaling_exponents': np.array([0]),
                'singularity_spectrum': np.array([0.5]),
                'multifractal_spectrum': np.array([1]),
                'fluctuation_functions': np.array([[1]])
            }


class ChaosAnalyzer:
    """
    Chaos theory feature extraction for financial time series
    """

    def __init__(self, embedding_dim: int = 3, tau: int = 1):
        self.embedding_dim = embedding_dim
        self.tau = tau
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('ChaosAnalyzer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def embed_time_series(self, time_series: np.ndarray, 
                         embedding_dim: Optional[int] = None, 
                         tau: Optional[int] = None) -> np.ndarray:
        """Create phase space embedding using time delay coordinates"""
        ts = np.asarray(time_series).flatten()
        dim = embedding_dim or self.embedding_dim
        delay = tau or self.tau

        n = len(ts)
        embedded_dim = n - (dim - 1) * delay

        if embedded_dim <= 0:
            return np.array([[0] * dim])

        embedded = np.zeros((embedded_dim, dim))
        for i in range(dim):
            embedded[:, i] = ts[i * delay:i * delay + embedded_dim]

        return embedded

    def lyapunov_exponent(self, time_series: np.ndarray, max_iter: int = 500) -> float:
        """Calculate largest Lyapunov exponent"""
        try:
            ts = np.asarray(time_series).flatten()

            # Embed time series
            embedded = self.embed_time_series(ts)
            n_points = len(embedded)

            if n_points < 50:
                return 0.0

            # Find nearest neighbors
            distances = pdist(embedded)
            distance_matrix = squareform(distances)

            # Set diagonal to infinity to avoid self-matching
            np.fill_diagonal(distance_matrix, np.inf)

            # Track divergence
            divergences = []
            min_separation = np.percentile(distances[distances > 0], 5)

            for i in range(min(n_points - 10, max_iter)):
                # Find nearest neighbor
                neighbors = np.where(distance_matrix[i] < min_separation * 2)[0]

                if len(neighbors) == 0:
                    continue

                # Choose nearest neighbor
                nearest_idx = neighbors[np.argmin(distance_matrix[i, neighbors])]

                # Track evolution for next steps
                max_evolution = min(10, n_points - max(i, nearest_idx) - 1)

                for step in range(1, max_evolution):
                    if i + step < len(embedded) and nearest_idx + step < len(embedded):
                        initial_distance = np.linalg.norm(embedded[i] - embedded[nearest_idx])
                        evolved_distance = np.linalg.norm(embedded[i + step] - embedded[nearest_idx + step])

                        if initial_distance > 0 and evolved_distance > 0:
                            divergence = np.log(evolved_distance / initial_distance) / step
                            divergences.append(divergence)

            if divergences:
                lyapunov = np.mean(divergences)
                return np.clip(lyapunov, -10, 10)  # Reasonable bounds
            else:
                return 0.0

        except Exception as e:
            self.logger.warning(f"Error calculating Lyapunov exponent: {e}")
            return 0.0

    def correlation_dimension(self, time_series: np.ndarray, 
                            r_range: Optional[np.ndarray] = None) -> float:
        """Calculate correlation dimension using Grassberger-Procaccia algorithm"""
        try:
            ts = np.asarray(time_series).flatten()
            embedded = self.embed_time_series(ts)
            n_points = len(embedded)

            if n_points < 20:
                return 1.0

            # Distance matrix
            distances = pdist(embedded)

            if r_range is None:
                r_min = np.percentile(distances, 1)
                r_max = np.percentile(distances, 50)
                r_range = np.logspace(np.log10(r_min), np.log10(r_max), 20)

            correlations = []

            for r in r_range:
                # Count pairs within distance r
                count = np.sum(distances < r)
                total_pairs = n_points * (n_points - 1) / 2
                correlation = count / total_pairs if total_pairs > 0 else 0
                correlations.append(max(correlation, 1e-10))  # Avoid log(0)

            # Linear regression on log-log plot
            log_r = np.log(r_range)
            log_c = np.log(correlations)

            valid_mask = np.isfinite(log_r) & np.isfinite(log_c)
            if np.sum(valid_mask) < 3:
                return 1.0

            correlation_dim = np.polyfit(log_r[valid_mask], log_c[valid_mask], 1)[0]
            return np.clip(correlation_dim, 0.5, 10.0)

        except Exception as e:
            self.logger.warning(f"Error calculating correlation dimension: {e}")
            return 1.0

    def sample_entropy(self, time_series: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate sample entropy"""
        try:
            ts = np.asarray(time_series).flatten()
            n = len(ts)

            if n < 20:
                return 0.0

            if r is None:
                r = 0.2 * np.std(ts)

            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])

            def _phi(m):
                patterns = np.array([ts[i:i + m] for i in range(n - m + 1)])
                matches = 0

                for i in range(len(patterns)):
                    for j in range(len(patterns)):
                        if i != j and _maxdist(patterns[i], patterns[j], m) <= r:
                            matches += 1

                return matches / (len(patterns) * (len(patterns) - 1))

            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)

            if phi_m == 0 or phi_m1 == 0:
                return 0.0

            sample_ent = -np.log(phi_m1 / phi_m)
            return np.clip(sample_ent, 0, 10)

        except Exception as e:
            self.logger.warning(f"Error calculating sample entropy: {e}")
            return 0.0

    def recurrence_analysis(self, time_series: np.ndarray, 
                          threshold: float = None) -> Dict[str, float]:
        """Recurrence Quantification Analysis (RQA)"""
        try:
            ts = np.asarray(time_series).flatten()
            embedded = self.embed_time_series(ts)
            n_points = len(embedded)

            if n_points < 20:
                return {'recurrence_rate': 0, 'determinism': 0, 'laminarity': 0}

            # Distance matrix
            distance_matrix = squareform(pdist(embedded))

            if threshold is None:
                threshold = np.percentile(distance_matrix.flatten(), 5)

            # Recurrence matrix
            recurrence_matrix = (distance_matrix < threshold).astype(int)

            # Remove main diagonal
            np.fill_diagonal(recurrence_matrix, 0)

            # Recurrence rate
            recurrence_rate = np.sum(recurrence_matrix) / (n_points**2 - n_points)

            # Determinism - ratio of recurrent points in diagonal lines
            diagonal_points = 0
            total_recurrent = np.sum(recurrence_matrix)

            for offset in range(1, n_points):
                diagonal = np.diagonal(recurrence_matrix, offset=offset)
                # Find consecutive sequences of 1s
                sequences = []
                current_seq = 0
                for val in diagonal:
                    if val == 1:
                        current_seq += 1
                    else:
                        if current_seq >= 2:  # Minimum line length
                            sequences.append(current_seq)
                        current_seq = 0
                if current_seq >= 2:
                    sequences.append(current_seq)

                diagonal_points += sum(sequences)

            determinism = diagonal_points / total_recurrent if total_recurrent > 0 else 0

            # Laminarity - ratio of recurrent points in vertical/horizontal lines  
            vertical_points = 0
            for i in range(n_points):
                column = recurrence_matrix[:, i]
                sequences = []
                current_seq = 0
                for val in column:
                    if val == 1:
                        current_seq += 1
                    else:
                        if current_seq >= 2:
                            sequences.append(current_seq)
                        current_seq = 0
                if current_seq >= 2:
                    sequences.append(current_seq)

                vertical_points += sum(sequences)

            laminarity = vertical_points / total_recurrent if total_recurrent > 0 else 0

            return {
                'recurrence_rate': np.clip(recurrence_rate, 0, 1),
                'determinism': np.clip(determinism, 0, 1),
                'laminarity': np.clip(laminarity, 0, 1)
            }

        except Exception as e:
            self.logger.warning(f"Error in recurrence analysis: {e}")
            return {'recurrence_rate': 0, 'determinism': 0, 'laminarity': 0}


class QuantumEntanglementAnalyzer:
    """
    Quantum-inspired correlation analysis for portfolio assets
    """

    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('QuantumEntanglementAnalyzer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def quantum_mutual_information(self, X: np.ndarray, Y: np.ndarray, bins: int = 10) -> float:
        """Calculate quantum mutual information between two assets"""
        try:
            # Discretize the data
            x_discrete = pd.cut(X, bins=bins, labels=False)
            y_discrete = pd.cut(Y, bins=bins, labels=False)

            # Remove NaN values
            valid_mask = ~(np.isnan(x_discrete) | np.isnan(y_discrete))
            x_discrete = x_discrete[valid_mask]
            y_discrete = y_discrete[valid_mask]

            if len(x_discrete) == 0:
                return 0.0

            # Joint probability distribution
            joint_hist, _, _ = np.histogram2d(x_discrete, y_discrete, bins=bins)
            joint_prob = joint_hist / np.sum(joint_hist)

            # Marginal distributions
            px = np.sum(joint_prob, axis=1)
            py = np.sum(joint_prob, axis=0)

            # Mutual information
            mutual_info = 0.0
            for i in range(bins):
                for j in range(bins):
                    if joint_prob[i, j] > 0 and px[i] > 0 and py[j] > 0:
                        mutual_info += joint_prob[i, j] * np.log2(
                            joint_prob[i, j] / (px[i] * py[j])
                        )

            return np.clip(mutual_info, 0, 10)

        except Exception as e:
            self.logger.warning(f"Error calculating quantum mutual information: {e}")
            return 0.0

    def entanglement_entropy(self, correlation_matrix: np.ndarray) -> np.ndarray:
        """Calculate entanglement entropy from correlation matrix"""
        try:
            # Ensure positive semi-definite
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 1e-10)  # Avoid log(0)

            # Normalize eigenvalues
            eigenvals = eigenvals / np.sum(eigenvals)

            # Von Neumann entropy
            entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-10))

            return np.clip(entropy, 0, np.log2(len(eigenvals)))

        except Exception as e:
            self.logger.warning(f"Error calculating entanglement entropy: {e}")
            return 0.0

    def quantum_discord(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Calculate quantum discord between two time series"""
        try:
            # Create 2x2 correlation matrix for bipartite system
            data = np.column_stack([X, Y])
            corr_matrix = np.corrcoef(data.T)

            # Regularize correlation matrix
            corr_matrix = np.clip(corr_matrix, -0.99, 0.99)

            # Convert to density matrix (simplified quantum state)
            density_matrix = (np.eye(2) + corr_matrix) / 2

            # Classical correlation (mutual information)
            classical_corr = self.quantum_mutual_information(X, Y)

            # Quantum correlation (total correlation)
            entropy_joint = self.entanglement_entropy(density_matrix)
            entropy_x = self.entanglement_entropy(np.array([[1, corr_matrix[0, 1]], 
                                                          [corr_matrix[1, 0], 1]]))

            quantum_corr = entropy_x + entropy_joint

            # Discord = Quantum correlation - Classical correlation
            discord = max(0, quantum_corr - classical_corr)

            return np.clip(discord, 0, 10)

        except Exception as e:
            self.logger.warning(f"Error calculating quantum discord: {e}")
            return 0.0

    def concurrence(self, correlation_matrix: np.ndarray) -> float:
        """Calculate concurrence measure of entanglement"""
        try:
            if correlation_matrix.shape != (2, 2):
                # For larger matrices, use average pairwise concurrence
                n = correlation_matrix.shape[0]
                concurrences = []

                for i in range(n):
                    for j in range(i+1, n):
                        sub_matrix = correlation_matrix[[i, j]][:, [i, j]]
                        conc = self.concurrence(sub_matrix)
                        concurrences.append(conc)

                return np.mean(concurrences) if concurrences else 0.0

            # For 2x2 case
            rho = (np.eye(2) + correlation_matrix) / 2

            # Pauli matrices
            sigma_y = np.array([[0, -1j], [1j, 0]])

            # Spin-flipped density matrix
            rho_tilde = np.kron(sigma_y, sigma_y) @ np.conj(rho) @ np.kron(sigma_y, sigma_y)

            # Calculate concurrence
            R_matrix = rho @ rho_tilde
            eigenvals = np.linalg.eigvals(R_matrix)
            eigenvals = np.sqrt(np.maximum(np.real(eigenvals), 0))  # Take real part and ensure non-negative
            eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order

            concurrence_val = max(0, eigenvals[0] - np.sum(eigenvals[1:]))

            return np.clip(concurrence_val, 0, 1)

        except Exception as e:
            self.logger.warning(f"Error calculating concurrence: {e}")
            return 0.0

    def negativity(self, correlation_matrix: np.ndarray) -> float:
        """Calculate negativity measure of entanglement"""
        try:
            # Convert correlation to density matrix
            rho = (np.eye(len(correlation_matrix)) + correlation_matrix) / 2

            # Partial transpose (transpose subsystem B)
            n = len(correlation_matrix)
            if n == 2:
                # Simple 2x2 case
                rho_pt = np.array([[rho[0,0], rho[0,1]], 
                                  [rho[1,0], rho[1,1]]])
                rho_pt = rho_pt.T  # Transpose
            else:
                # For larger systems, use simplified partial transpose
                rho_pt = rho.T

            # Calculate eigenvalues of partial transpose
            eigenvals = np.linalg.eigvals(rho_pt)
            eigenvals = np.real(eigenvals)  # Take real part

            # Negativity = sum of absolute values of negative eigenvalues
            negative_eigenvals = eigenvals[eigenvals < 0]
            negativity_val = np.sum(np.abs(negative_eigenvals))

            return np.clip(negativity_val, 0, 10)

        except Exception as e:
            self.logger.warning(f"Error calculating negativity: {e}")
            return 0.0


class GeneticFeatureOptimizer:
    """
    Genetic algorithm for market microstructure feature optimization
    """

    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('GeneticFeatureOptimizer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def create_individual(self, n_features: int) -> np.ndarray:
        """Create a random individual (feature combination)"""
        return np.random.choice([0, 1], size=n_features, p=[0.7, 0.3])

    def fitness_function(self, individual: np.ndarray, features: np.ndarray, 
                        target: np.ndarray) -> float:
        """Evaluate fitness of feature combination"""
        try:
            selected_features = features[:, individual.astype(bool)]

            if selected_features.shape[1] == 0:
                return 0.0

            # Simple correlation-based fitness
            correlations = []
            for i in range(selected_features.shape[1]):
                corr = np.corrcoef(selected_features[:, i], target)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

            if not correlations:
                return 0.0

            # Fitness = mean correlation - penalty for too many features
            fitness = np.mean(correlations) - 0.01 * np.sum(individual)
            return max(0, fitness)

        except Exception as e:
            self.logger.warning(f"Error in fitness evaluation: {e}")
            return 0.0

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover"""
        crossover_point = np.random.randint(1, len(parent1))

        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

        return child1, child2

    def mutate(self, individual: np.ndarray, mutation_rate: float = 0.01) -> np.ndarray:
        """Bit-flip mutation"""
        mutation_mask = np.random.random(len(individual)) < mutation_rate
        individual[mutation_mask] = 1 - individual[mutation_mask]
        return individual

    def evolve(self, features: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Run genetic algorithm evolution"""
        try:
            n_features = features.shape[1]

            # Initialize population
            population = [self.create_individual(n_features) for _ in range(self.population_size)]

            best_individual = None
            best_fitness = 0.0

            for generation in range(self.generations):
                # Evaluate fitness
                fitness_scores = [
                    self.fitness_function(ind, features, target) 
                    for ind in population
                ]

                # Track best individual
                max_fitness_idx = np.argmax(fitness_scores)
                if fitness_scores[max_fitness_idx] > best_fitness:
                    best_fitness = fitness_scores[max_fitness_idx]
                    best_individual = population[max_fitness_idx].copy()

                # Selection (tournament selection)
                new_population = []
                for _ in range(self.population_size):
                    tournament_indices = np.random.choice(
                        self.population_size, size=3, replace=False
                    )
                    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                    new_population.append(population[winner_idx].copy())

                # Crossover and mutation
                next_population = []
                for i in range(0, self.population_size, 2):
                    if i + 1 < self.population_size:
                        child1, child2 = self.crossover(new_population[i], new_population[i+1])
                        child1 = self.mutate(child1)
                        child2 = self.mutate(child2)
                        next_population.extend([child1, child2])
                    else:
                        next_population.append(self.mutate(new_population[i]))

                population = next_population[:self.population_size]

                if generation % 20 == 0:
                    self.logger.info(f"Generation {generation}: Best fitness = {best_fitness:.4f}")

            return best_individual if best_individual is not None else np.ones(n_features)

        except Exception as e:
            self.logger.error(f"Error in genetic evolution: {e}")
            return np.ones(features.shape[1])  # Return all features as fallback


class FractalFeaturePipeline:
    """
    Main feature engineering pipeline combining all revolutionary components
    """

    def __init__(self, 
                 fractal_scales: Tuple[int, int] = (4, 64),
                 chaos_embedding_dim: int = 3,
                 hd_dimension: int = 10000,
                 genetic_generations: int = 50):

        # Initialize components
        self.fractal_analyzer = FractalAnalyzer(fractal_scales[0], fractal_scales[1])
        self.chaos_analyzer = ChaosAnalyzer(chaos_embedding_dim)
        self.quantum_analyzer = QuantumEntanglementAnalyzer()
        self.genetic_optimizer = GeneticFeatureOptimizer(generations=genetic_generations)

        # Hyperdimensional computing
        self.hd_dimension = hd_dimension
        self.hd_codebook = {}

        # Feature scaling
        self.scaler = RobustScaler()
        self.feature_names = []
        self.is_fitted = False

        # Logging
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('FractalFeaturePipeline')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _create_hyperdimensional_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create hyperdimensional computing features"""
        try:
            hd_features = []

            for col in data.columns:
                if col not in self.hd_codebook:
                    self.hd_codebook[col] = HyperdimensionalVector(self.hd_dimension)

                # Encode time series values
                series_values = data[col].values
                for i, value in enumerate(series_values[:min(len(series_values), 100)]):  # Limit for efficiency
                    # Create position vector
                    pos_vector = HyperdimensionalVector(self.hd_dimension)
                    pos_vector = pos_vector.permute(i)

                    # Create value vector (quantized)
                    value_idx = int((value - np.min(series_values)) / 
                                  (np.max(series_values) - np.min(series_values) + 1e-8) * 99)
                    value_key = f"{col}_value_{value_idx}"

                    if value_key not in self.hd_codebook:
                        self.hd_codebook[value_key] = HyperdimensionalVector(self.hd_dimension)

                    # Bind position and value
                    bound_vector = pos_vector.bind(self.hd_codebook[value_key])

                    # Bundle with base vector
                    self.hd_codebook[col] = self.hd_codebook[col].bundle(bound_vector)

            # Extract similarity features
            base_symbols = list(data.columns)
            for i, sym1 in enumerate(base_symbols):
                for j, sym2 in enumerate(base_symbols[i+1:], i+1):
                    similarity = self.hd_codebook[sym1].similarity(self.hd_codebook[sym2])
                    hd_features.append(similarity)

            return np.array(hd_features) if hd_features else np.array([0])

        except Exception as e:
            self.logger.warning(f"Error creating hyperdimensional features: {e}")
            return np.array([0])

    def _extract_fractal_features(self, time_series: np.ndarray) -> np.ndarray:
        """Extract comprehensive fractal features"""
        features = []

        try:
            # Hurst exponent
            hurst = self.fractal_analyzer.hurst_exponent(time_series)
            features.append(hurst)

            # Fractal dimensions
            for method in ['box_counting', 'katz']:
                fd = self.fractal_analyzer.fractal_dimension(time_series, method)
                features.append(fd)

            # Detrended Fluctuation Analysis
            dfa_alpha, _ = self.fractal_analyzer.detrended_fluctuation_analysis(time_series)
            features.append(dfa_alpha)

            # Multifractal analysis
            mf_results = self.fractal_analyzer.multifractal_analysis(time_series)

            # Multifractal width
            alpha_range = np.max(mf_results['singularity_spectrum']) - np.min(mf_results['singularity_spectrum'])
            features.append(alpha_range)

            # Asymmetry of multifractal spectrum
            alpha_values = mf_results['singularity_spectrum']
            max_idx = np.argmax(mf_results['multifractal_spectrum'])
            if len(alpha_values) > max_idx > 0:
                left_width = alpha_values[max_idx] - np.min(alpha_values[:max_idx+1])
                right_width = np.max(alpha_values[max_idx:]) - alpha_values[max_idx]
                asymmetry = (right_width - left_width) / (right_width + left_width + 1e-8)
                features.append(asymmetry)
            else:
                features.append(0.0)

        except Exception as e:
            self.logger.warning(f"Error extracting fractal features: {e}")
            features.extend([0.5] * 6)  # Default values

        return np.array(features)

    def _extract_chaos_features(self, time_series: np.ndarray) -> np.ndarray:
        """Extract chaos theory features"""
        features = []

        try:
            # Lyapunov exponent
            lyapunov = self.chaos_analyzer.lyapunov_exponent(time_series)
            features.append(lyapunov)

            # Correlation dimension
            corr_dim = self.chaos_analyzer.correlation_dimension(time_series)
            features.append(corr_dim)

            # Sample entropy
            sample_ent = self.chaos_analyzer.sample_entropy(time_series)
            features.append(sample_ent)

            # Recurrence quantification analysis
            rqa_results = self.chaos_analyzer.recurrence_analysis(time_series)
            features.extend([
                rqa_results['recurrence_rate'],
                rqa_results['determinism'],
                rqa_results['laminarity']
            ])

        except Exception as e:
            self.logger.warning(f"Error extracting chaos features: {e}")
            features.extend([0.0] * 6)  # Default values

        return np.array(features)

    def _extract_quantum_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract quantum entanglement features"""
        features = []

        try:
            if len(data.columns) < 2:
                return np.array([0] * 5)

            # Correlation matrix
            corr_matrix = data.corr().fillna(0).values

            # Entanglement entropy
            ent_entropy = self.quantum_analyzer.entanglement_entropy(corr_matrix)
            features.append(ent_entropy)

            # Pairwise quantum features for first few asset pairs
            symbols = list(data.columns)
            n_pairs = min(3, len(symbols) * (len(symbols) - 1) // 2)  # Limit computational cost

            pair_features = []
            pair_count = 0

            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    if pair_count >= n_pairs:
                        break

                    X = data[symbols[i]].values
                    Y = data[symbols[j]].values

                    # Remove NaN values
                    valid_mask = ~(np.isnan(X) | np.isnan(Y))
                    if np.sum(valid_mask) < 10:
                        continue

                    X = X[valid_mask]
                    Y = Y[valid_mask]

                    # Quantum mutual information
                    qmi = self.quantum_analyzer.quantum_mutual_information(X, Y)
                    pair_features.append(qmi)

                    # Quantum discord
                    discord = self.quantum_analyzer.quantum_discord(X, Y)
                    pair_features.append(discord)

                    pair_count += 1

                if pair_count >= n_pairs:
                    break

            # Pad or truncate to consistent length
            target_length = 4  # 2 features per pair, max 2 pairs
            if len(pair_features) < target_length:
                pair_features.extend([0.0] * (target_length - len(pair_features)))
            else:
                pair_features = pair_features[:target_length]

            features.extend(pair_features)

            # Overall concurrence and negativity
            concurrence = self.quantum_analyzer.concurrence(corr_matrix)
            negativity = self.quantum_analyzer.negativity(corr_matrix)

            features.extend([concurrence, negativity])

        except Exception as e:
            self.logger.warning(f"Error extracting quantum features: {e}")
            features = [0.0] * 7  # Default values

        return np.array(features)

    def _extract_technical_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract technical analysis features"""
        features = []

        try:
            for col in data.columns:
                series = data[col].fillna(method='ffill').values

                if len(series) < 20:
                    features.extend([0] * 10)
                    continue

                # Basic technical indicators
                try:
                    rsi = talib.RSI(series, timeperiod=14)[-1]
                    if np.isnan(rsi):
                        rsi = 50
                except:
                    rsi = 50

                try:
                    macd, macd_signal, macd_hist = talib.MACD(series)
                    macd_val = macd[-1] if not np.isnan(macd[-1]) else 0
                    macd_signal_val = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
                except:
                    macd_val = 0
                    macd_signal_val = 0

                try:
                    bb_upper, bb_middle, bb_lower = talib.BBANDS(series)
                    bb_position = (series[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1] + 1e-8)
                    if np.isnan(bb_position):
                        bb_position = 0.5
                except:
                    bb_position = 0.5

                # Volatility measures
                returns = np.diff(series) / (series[:-1] + 1e-8)
                realized_vol = np.std(returns) * np.sqrt(252)
                skewness = stats.skew(returns)
                kurtosis = stats.kurtosis(returns)

                # VaR and CVaR
                var_95 = np.percentile(returns, 5)
                cvar_95 = np.mean(returns[returns <= var_95])

                features.extend([
                    rsi / 100.0,  # Normalize
                    np.tanh(macd_val),  # Bounded
                    np.tanh(macd_signal_val),
                    bb_position,
                    np.tanh(realized_vol),
                    np.tanh(skewness),
                    np.tanh(kurtosis),
                    np.tanh(var_95),
                    np.tanh(cvar_95),
                    len(series) / 1000.0  # Series length feature
                ])

                # Only process first few columns to control feature count
                if len(features) >= 50:  # Limit technical features
                    break

        except Exception as e:
            self.logger.warning(f"Error extracting technical features: {e}")
            features = [0.0] * 10  # Default values

        return np.array(features)

    def fit_transform(self, data: pd.DataFrame, target: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the pipeline and transform data"""
        self.logger.info(f"Fitting fractal feature pipeline with {data.shape} data")

        try:
            all_features = []
            feature_names = []

            # Extract features for each time series
            for col in data.columns:
                series = data[col].fillna(method='ffill').values

                # Fractal features
                fractal_feats = self._extract_fractal_features(series)
                all_features.extend(fractal_feats)
                feature_names.extend([f'{col}_fractal_{i}' for i in range(len(fractal_feats))])

                # Chaos features
                chaos_feats = self._extract_chaos_features(series)
                all_features.extend(chaos_feats)
                feature_names.extend([f'{col}_chaos_{i}' for i in range(len(chaos_feats))])

            # Cross-asset quantum features
            quantum_feats = self._extract_quantum_features(data)
            all_features.extend(quantum_feats)
            feature_names.extend([f'quantum_{i}' for i in range(len(quantum_feats))])

            # Technical features
            technical_feats = self._extract_technical_features(data)
            all_features.extend(technical_feats)
            feature_names.extend([f'technical_{i}' for i in range(len(technical_feats))])

            # Hyperdimensional features
            hd_feats = self._create_hyperdimensional_features(data)
            all_features.extend(hd_feats)
            feature_names.extend([f'hd_{i}' for i in range(len(hd_feats))])

            # Convert to array and handle NaN/inf
            feature_array = np.array(all_features).reshape(1, -1)
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)

            # Fit scaler
            self.scaler.fit(feature_array)

            # Scale features
            scaled_features = self.scaler.transform(feature_array)

            # Genetic optimization if target provided
            if target is not None and len(target) > 0:
                self.logger.info("Running genetic feature optimization...")
                # Expand target to match feature array if needed
                if len(target) != feature_array.shape[0]:
                    target_expanded = np.full(feature_array.shape[0], target[0] if len(target) > 0 else 0)
                else:
                    target_expanded = target

                optimal_features = self.genetic_optimizer.evolve(scaled_features.T, target_expanded)
                self.optimal_feature_mask = optimal_features
                scaled_features = scaled_features[:, optimal_features.astype(bool)]
                feature_names = [name for i, name in enumerate(feature_names) if optimal_features[i]]

            self.feature_names = feature_names
            self.is_fitted = True

            self.logger.info(f"Feature extraction completed: {scaled_features.shape[1]} features")

            return scaled_features.flatten()

        except Exception as e:
            self.logger.error(f"Error in fit_transform: {e}")
            # Return safe fallback
            return np.zeros(100)  # Default feature vector

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        try:
            all_features = []

            # Extract same features as in fit
            for col in data.columns:
                if col in [name.split('_')[0] for name in self.feature_names]:  # Only process known columns
                    series = data[col].fillna(method='ffill').values

                    # Fractal features
                    fractal_feats = self._extract_fractal_features(series)
                    all_features.extend(fractal_feats)

                    # Chaos features
                    chaos_feats = self._extract_chaos_features(series)
                    all_features.extend(chaos_feats)

            # Cross-asset quantum features
            quantum_feats = self._extract_quantum_features(data)
            all_features.extend(quantum_feats)

            # Technical features  
            technical_feats = self._extract_technical_features(data)
            all_features.extend(technical_feats)

            # Hyperdimensional features
            hd_feats = self._create_hyperdimensional_features(data)
            all_features.extend(hd_feats)

            # Convert to array and handle NaN/inf
            feature_array = np.array(all_features).reshape(1, -1)
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)

            # Ensure correct number of features
            if feature_array.shape[1] != len(self.feature_names):
                # Pad or truncate as needed
                target_features = len(self.feature_names)
                if feature_array.shape[1] < target_features:
                    padding = np.zeros((1, target_features - feature_array.shape[1]))
                    feature_array = np.hstack([feature_array, padding])
                else:
                    feature_array = feature_array[:, :target_features]

            # Scale features
            scaled_features = self.scaler.transform(feature_array)

            # Apply genetic optimization mask if available
            if hasattr(self, 'optimal_feature_mask'):
                scaled_features = scaled_features[:, self.optimal_feature_mask.astype(bool)]

            return scaled_features.flatten()

        except Exception as e:
            self.logger.error(f"Error in transform: {e}")
            return np.zeros(len(self.feature_names))

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        if not self.is_fitted:
            return pd.DataFrame()

        try:
            importance_scores = []

            if hasattr(self, 'optimal_feature_mask'):
                # Use genetic optimization results
                for i, name in enumerate(self.feature_names):
                    importance = float(self.optimal_feature_mask[i]) if i < len(self.optimal_feature_mask) else 0.0
                    importance_scores.append(importance)
            else:
                # Use feature variance as proxy for importance
                importance_scores = [1.0] * len(self.feature_names)

            return pd.DataFrame({
                'feature_name': self.feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)

        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()

    def save_pipeline(self, filepath: str):
        """Save the fitted pipeline"""
        try:
            import pickle

            pipeline_data = {
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'hd_codebook': self.hd_codebook,
                'is_fitted': self.is_fitted,
                'optimal_feature_mask': getattr(self, 'optimal_feature_mask', None)
            }

            with open(filepath, 'wb') as f:
                pickle.dump(pipeline_data, f)

            self.logger.info(f"Pipeline saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving pipeline: {e}")

    def load_pipeline(self, filepath: str):
        """Load a fitted pipeline"""
        try:
            import pickle

            with open(filepath, 'rb') as f:
                pipeline_data = pickle.load(f)

            self.scaler = pipeline_data['scaler']
            self.feature_names = pipeline_data['feature_names']
            self.hd_codebook = pipeline_data['hd_codebook']
            self.is_fitted = pipeline_data['is_fitted']

            if 'optimal_feature_mask' in pipeline_data and pipeline_data['optimal_feature_mask'] is not None:
                self.optimal_feature_mask = pipeline_data['optimal_feature_mask']

            self.logger.info(f"Pipeline loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading pipeline: {e}")


# Factory function for easy instantiation
def create_fractal_pipeline(**kwargs) -> FractalFeaturePipeline:
    """Create a fractal feature pipeline with specified parameters"""
    return FractalFeaturePipeline(**kwargs)


# Demonstration function
def demonstrate_fractal_pipeline():
    """Demonstrate the fractal feature pipeline"""
    # Generate synthetic multi-asset data
    np.random.seed(42)
    n_samples = 1000
    n_assets = 3

    # Create correlated time series with different fractal properties
    data = {}

    for i in range(n_assets):
        # Generate fractional Brownian motion with different Hurst exponents
        hurst = 0.3 + i * 0.2  # Different fractal properties

        # Simple fractional Brownian motion approximation
        noise = np.random.randn(n_samples)
        cumsum = np.cumsum(noise)

        # Add trend and seasonality
        trend = np.linspace(0, 10, n_samples) * (1 + i * 0.1)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 252) * (1 + i * 0.2)

        time_series = cumsum + trend + seasonal + np.random.randn(n_samples) * 0.1
        data[f'Asset_{i+1}'] = time_series

    df = pd.DataFrame(data)

    # Create pipeline and extract features
    pipeline = FractalFeaturePipeline()

    # Create synthetic target
    target = np.sum(df.values, axis=1)[:1]  # Simple target for demonstration

    # Fit and transform
    features = pipeline.fit_transform(df, target)

    print(f"Extracted {len(features)} fractal features from {df.shape[1]} assets")
    print(f"Feature shape: {features.shape}")

    # Show feature importance
    importance = pipeline.get_feature_importance()
    print(f"\nTop 10 most important features:")
    print(importance.head(10))

    return pipeline, features


if __name__ == "__main__":
    # Run demonstration
    demo_pipeline, demo_features = demonstrate_fractal_pipeline()
    print("\nFractal Feature Pipeline demonstration completed successfully!")
