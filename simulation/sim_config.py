"""Shared configuration, dataclasses, and defaults for the simulation system."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AssetConfig:
    """Configuration for a single tradable asset."""
    symbol: str
    exchange_ccxt: str = "coinbase"
    yfinance_ticker: str = ""
    asset_class: str = "crypto"

    def __post_init__(self):
        if not self.yfinance_ticker:
            self.yfinance_ticker = self.symbol


@dataclass
class SimulationResult:
    """Output of a single simulation run."""
    model_name: str
    asset: str
    paths: np.ndarray                        # (n_simulations, n_steps+1)
    timestamps: Optional[pd.DatetimeIndex] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    calibration_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_simulations(self) -> int:
        return self.paths.shape[0]

    @property
    def n_steps(self) -> int:
        return self.paths.shape[1] - 1

    def mean_path(self) -> np.ndarray:
        return self.paths.mean(axis=0)

    def percentile_path(self, q: float) -> np.ndarray:
        return np.percentile(self.paths, q, axis=0)

    def log_returns(self) -> np.ndarray:
        """Flattened log returns across all paths."""
        return np.diff(np.log(self.paths), axis=1).ravel()


@dataclass
class TradeCost:
    """Breakdown of a single trade's transaction costs."""
    maker_fee: float
    taker_fee: float
    slippage: float
    half_spread: float
    funding_cost: float
    total: float
    breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class Trade:
    """A single executed trade."""
    timestamp_idx: int
    side: str               # "buy" or "sell"
    price: float
    size_usd: float
    cost: float
    signal_value: float
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Output of a strategy backtest."""
    strategy_name: str
    asset: str
    equity_curve: np.ndarray
    returns: np.ndarray
    trades: List[Trade]
    metrics: Dict[str, float]
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ValidationScore:
    """Scorecard for one model on one asset."""
    model_name: str
    asset: str
    ks_stat: float
    ks_pvalue: float
    ad_stat: float
    ad_pvalue: float
    acf_rmse: float
    garch_param_distance: float
    vol_clustering_score: float
    composite_score: float              # 0–10

    # Per-metric 0-10 sub-scores
    ks_score: float = 0.0
    ad_score: float = 0.0
    acf_score: float = 0.0
    garch_score: float = 0.0
    vol_clust_sub_score: float = 0.0


@dataclass
class ParameterDistribution:
    """Bootstrap distribution for a single model parameter."""
    param_name: str
    mean: float
    std: float
    ci_lower: float             # 5th percentile
    ci_upper: float             # 95th percentile
    samples: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "assets": [
        {"symbol": "BTC-USD", "exchange_ccxt": "coinbase",
         "yfinance_ticker": "BTC-USD", "asset_class": "crypto"},
        {"symbol": "ETH-USD", "exchange_ccxt": "coinbase",
         "yfinance_ticker": "ETH-USD", "asset_class": "crypto"},
        {"symbol": "SOL-USD", "exchange_ccxt": "coinbase",
         "yfinance_ticker": "SOL-USD", "asset_class": "crypto"},
    ],

    "data": {
        "lookback_days": 730,
        "source_priority": ["ccxt", "yfinance"],
        "outlier_sigma": 3.0,
        "nan_interpolation": "linear",
        "volume_norm_window": 20,
    },

    "simulation": {
        "n_simulations": 1000,
        "n_steps": 252,
        "dt": 1.0 / 252,
        "random_seed": 42,
    },

    "models": {
        "monte_carlo": {"enabled": True},
        "gbm": {"enabled": True},
        "heston": {
            "enabled": True,
            "kappa": 2.0,
            "theta": 0.04,
            "xi": 0.5,
            "rho": -0.7,
            "v0": None,          # None → calibrate from data
        },
        "hmm_regime": {
            "enabled": True,
            "n_regimes": 3,
            "covariance_type": "full",
            "n_iter": 150,
        },
        "ngram": {
            "enabled": True,
            "n": 3,
            "n_bins": 20,
        },
    },

    "transaction_costs": {
        "maker_fee": 0.001,
        "taker_fee": 0.002,
        "base_slippage_bps": 5.0,
        "vol_slippage_coeff": 0.1,
        "volume_slippage_coeff": 0.05,
        "half_spread_bps": 3.0,
        "funding_rate_daily": 0.0001,
    },

    "strategies": {
        "mean_reversion": {
            "entry_z": 2.0,
            "exit_z": 0.0,
            "lookback": 60,
        },
        "contrarian_scanner": {
            "enabled": True,
            "min_consecutive": 3,
        },
    },

    "bootstrap": {
        "n_bootstrap": 200,
        "block_size": 20,
    },

    "stress_test": {
        "flash_crash_pct": -0.30,
        "covid_decline_days": 30,
        "covid_total_decline": -0.50,
        "death_spiral_feedback": 0.02,
        "death_spiral_duration": 20,
        "correlation_crisis_duration": 10,
        "liquidity_crisis_multiplier": 5.0,
    },

    "output": {
        "output_dir": "sim_output",
        "save_csv": True,
        "save_json": True,
        "save_plots": True,
        "save_parquet": True,
        "plot_format": "png",
        "plot_dpi": 150,
    },

    "backtest": {
        "initial_capital": 100_000.0,
        "position_fraction": 0.25,
    },
}


def merge_config(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Deep-merge *user_config* over DEFAULT_CONFIG (one level)."""
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if not user_config:
        return cfg
    for key, val in user_config.items():
        if isinstance(val, dict) and key in cfg and isinstance(cfg[key], dict):
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg
