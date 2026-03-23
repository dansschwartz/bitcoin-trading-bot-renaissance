"""
polymarket_timing_features.py — Intra-window BTC momentum features for altcoin edge.

Core insight: BTC leads altcoins by 10-120 seconds in 5-minute windows.
When BTC moves strongly in the first minute of a window, altcoins tend to follow.

Features computed:
  1. btc_1bar_ret    — BTC return over last completed 5-min bar
  2. btc_3bar_ret    — BTC return over last 3 completed bars (momentum trend)
  3. btc_vol_ratio   — BTC realized vol vs 20-bar average (regime filter)
  4. btc_alt_spread  — BTC return minus altcoin return (convergence signal)
  5. btc_volume_z    — BTC volume z-score (conviction indicator)
  6. lead_momentum   — Composite score combining 1,3-bar momentum + vol

Usage:
    from polymarket_timing_features import TimingFeatureEngine

    engine = TimingFeatureEngine()
    features = engine.compute(
        asset="SOL",
        cross_data=cross_data,         # Dict[pair_name, DataFrame]
        current_prices=current_prices,  # Dict[pair, float]
    )

    if features["lead_momentum"] > 0.5:
        # Strong BTC lead signal → boost altcoin confidence
"""

import logging
import time
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TimingFeatureEngine:
    """Compute BTC lead-lag features for Polymarket altcoin 5m bets."""

    # Assets that serve as market leaders
    LEAD_ASSETS = {"BTC": "BTC", "ETH": "ETH"}

    # Assets that follow leaders (altcoins)
    FOLLOWER_ASSETS = {"SOL", "XRP", "DOGE", "HYPE", "BNB", "AVAX", "ADA"}

    # Minimum bars needed for feature computation
    MIN_BARS = 5

    # Throttle logging to avoid spam
    _LOG_INTERVAL = 300  # seconds

    def __init__(self) -> None:
        self._last_log_time = 0.0
        self._call_count = 0
        self._feature_cache: Dict[str, Dict] = {}
        self._cache_ts: Dict[str, float] = {}
        self._cache_ttl = 30  # seconds

    def compute(
        self,
        asset: str,
        cross_data: Optional[Dict] = None,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Compute timing features for an altcoin bet.

        Args:
            asset: Target altcoin ("SOL", "XRP", etc.)
            cross_data: Dict of pair_name -> DataFrame with OHLCV data
            current_prices: Dict of pair -> latest price

        Returns:
            Dict with timing features and composite score.
        """
        self._call_count += 1
        now = time.time()

        # Cache check
        cache_key = f"{asset}_{int(now // self._cache_ttl)}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        result = self._compute_features(asset, cross_data, current_prices)

        self._feature_cache[cache_key] = result
        self._cache_ts[cache_key] = now

        # Prune old cache entries
        stale = [k for k, ts in self._cache_ts.items() if now - ts > self._cache_ttl * 3]
        for k in stale:
            self._feature_cache.pop(k, None)
            self._cache_ts.pop(k, None)

        # Throttled logging
        if now - self._last_log_time > self._LOG_INTERVAL:
            self._last_log_time = now
            logger.info(
                f"[TIMING] {asset}: lead_momentum={result['lead_momentum']:.3f} "
                f"btc_1bar={result['btc_1bar_ret']:.4f} "
                f"btc_3bar={result['btc_3bar_ret']:.4f} "
                f"vol_ratio={result['btc_vol_ratio']:.2f} "
                f"spread={result['btc_alt_spread']:.4f}"
            )

        return result

    def _compute_features(
        self,
        asset: str,
        cross_data: Optional[Dict],
        current_prices: Optional[Dict[str, float]],
    ) -> Dict:
        """Core feature computation."""
        default = {
            "btc_1bar_ret": 0.0,
            "btc_3bar_ret": 0.0,
            "btc_vol_ratio": 1.0,
            "btc_alt_spread": 0.0,
            "btc_volume_z": 0.0,
            "lead_momentum": 0.0,
            "has_data": False,
            "asset": asset,
            "timestamp": time.time(),
        }

        if not cross_data:
            return default

        # Find BTC data in cross_data
        btc_df = self._find_pair(cross_data, "BTC")
        if btc_df is None or len(btc_df) < self.MIN_BARS:
            return default

        # Find altcoin data
        alt_df = self._find_pair(cross_data, asset)

        # --- Feature 1: BTC 1-bar return ---
        btc_close = btc_df["close"].values
        btc_1bar_ret = 0.0
        if len(btc_close) >= 2 and btc_close[-2] > 0:
            btc_1bar_ret = (btc_close[-1] - btc_close[-2]) / btc_close[-2]

        # --- Feature 2: BTC 3-bar return ---
        btc_3bar_ret = 0.0
        if len(btc_close) >= 4 and btc_close[-4] > 0:
            btc_3bar_ret = (btc_close[-1] - btc_close[-4]) / btc_close[-4]

        # --- Feature 3: BTC volatility ratio ---
        btc_vol_ratio = 1.0
        if len(btc_close) >= 21:
            returns = np.diff(np.log(btc_close[-22:]))
            recent_vol = np.std(returns[-3:]) if len(returns) >= 3 else 0.0
            avg_vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.001
            if avg_vol > 1e-8:
                btc_vol_ratio = recent_vol / avg_vol

        # --- Feature 4: BTC-altcoin return spread ---
        btc_alt_spread = 0.0
        if alt_df is not None and len(alt_df) >= 2:
            alt_close = alt_df["close"].values
            if alt_close[-2] > 0:
                alt_1bar_ret = (alt_close[-1] - alt_close[-2]) / alt_close[-2]
                btc_alt_spread = btc_1bar_ret - alt_1bar_ret

        # --- Feature 5: BTC volume z-score ---
        btc_volume_z = 0.0
        if "volume" in btc_df.columns and len(btc_df) >= 21:
            vol_arr = btc_df["volume"].values
            recent_vol = vol_arr[-1]
            avg = np.mean(vol_arr[-20:])
            std = np.std(vol_arr[-20:])
            if std > 1e-8:
                btc_volume_z = (recent_vol - avg) / std

        # --- Feature 6: Composite lead momentum score ---
        # Weighted combination: directional momentum + conviction
        # Range: roughly [-2, +2]
        lead_momentum = (
            np.sign(btc_1bar_ret) * min(abs(btc_1bar_ret) * 200, 1.0) * 0.50  # 1-bar direction (50%)
            + np.sign(btc_3bar_ret) * min(abs(btc_3bar_ret) * 100, 1.0) * 0.30  # 3-bar trend (30%)
            + np.sign(btc_volume_z) * min(abs(btc_volume_z), 1.0) * 0.20  # Volume conviction (20%)
        )

        return {
            "btc_1bar_ret": round(float(btc_1bar_ret), 6),
            "btc_3bar_ret": round(float(btc_3bar_ret), 6),
            "btc_vol_ratio": round(float(btc_vol_ratio), 4),
            "btc_alt_spread": round(float(btc_alt_spread), 6),
            "btc_volume_z": round(float(btc_volume_z), 4),
            "lead_momentum": round(float(lead_momentum), 4),
            "has_data": True,
            "asset": asset,
            "timestamp": time.time(),
        }

    def _find_pair(self, cross_data: Dict, asset: str):
        """Find a pair's DataFrame in cross_data by asset symbol."""
        # Try common key formats
        candidates = [
            f"{asset}-USD",
            f"{asset}USDT",
            f"{asset}/USDT",
            f"{asset}-USDT",
            asset,
        ]
        for key in candidates:
            if key in cross_data:
                df = cross_data[key]
                if hasattr(df, "columns") and "close" in df.columns:
                    return df
        # Fuzzy match
        asset_upper = asset.upper()
        for key, df in cross_data.items():
            if asset_upper in str(key).upper():
                if hasattr(df, "columns") and "close" in df.columns:
                    return df
        return None

    def is_follower(self, asset: str) -> bool:
        """Check if an asset is a follower (altcoin) vs leader."""
        return asset.upper() in self.FOLLOWER_ASSETS

    def get_direction_boost(self, features: Dict, ml_direction: str) -> float:
        """
        Get confidence boost/penalty based on lead-lag alignment.

        Args:
            features: Output from compute()
            ml_direction: "UP" or "DOWN" from ML model

        Returns:
            Multiplier for confidence: >1.0 means boost, <1.0 means penalty.
            Range: [0.80, 1.15]
        """
        if not features.get("has_data"):
            return 1.0  # No data = no adjustment

        momentum = features["lead_momentum"]
        vol_ratio = features["btc_vol_ratio"]

        # Direction alignment: does BTC momentum agree with ML direction?
        if ml_direction == "UP":
            alignment = momentum  # Positive = aligned
        else:
            alignment = -momentum  # Negative BTC momentum = aligned with DOWN

        # Base boost from alignment
        # alignment > 0 means BTC agrees with ML → boost
        # alignment < 0 means BTC disagrees → penalty
        boost = 1.0 + alignment * 0.10  # +/-10% max from momentum alignment

        # Vol regime filter: high-vol periods have stronger lead signals
        if vol_ratio > 1.5:
            boost = 1.0 + (boost - 1.0) * 1.3  # Amplify during high vol
        elif vol_ratio < 0.5:
            boost = 1.0 + (boost - 1.0) * 0.7  # Dampen during low vol

        # Clamp to safe range
        return max(0.80, min(1.15, boost))

    def get_stats(self) -> Dict:
        """Return engine statistics for dashboard."""
        return {
            "call_count": self._call_count,
            "cache_size": len(self._feature_cache),
            "lead_assets": list(self.LEAD_ASSETS.keys()),
            "follower_assets": sorted(self.FOLLOWER_ASSETS),
        }
