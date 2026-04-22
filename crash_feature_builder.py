"""Crash-regime feature builder — 51 features for any asset.

Computes the same 51-feature vector used by all crash-regime LightGBM models.
Cross-asset direction is handled automatically:
  - BTC model: uses ETH as cross-asset lead
  - Alt models: use BTC as cross-asset lead

Feature groups (by index):
  0-14:  Price/volume technical (15)
  15-24: Daily macro (10)
  25-33: Derivatives (9)
  34-44: Intraday macro placeholders (11, always zero)
  45-50: Cross-asset lead signals (6)
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Cross-asset map: which asset serves as lead indicator
CROSS_ASSET_LEAD = {
    'BTC': 'ETH',
    'ETH': 'BTC',
    'SOL': 'BTC',
    'XRP': 'BTC',
    'DOGE': 'BTC',
}


class CrashFeatureBuilder:
    """Builds 51-feature vectors for crash-regime LightGBM models.

    Usage:
        builder = CrashFeatureBuilder()
        features = builder.build(
            asset='ETH',
            price_df=eth_price_df,
            cross_price_df=btc_price_df,  # BTC leads for ETH model
            derivatives_data=eth_derivatives,
            macro_data=macro_cache.get(),
        )
        # features.shape == (1, 51)
    """

    MIN_ROWS = 50  # Minimum price_df rows needed

    def build(
        self,
        asset: str,
        price_df: Any,
        cross_price_df: Any = None,
        derivatives_data: Optional[Dict[str, Any]] = None,
        macro_data: Optional[Dict[str, float]] = None,
    ) -> Optional[np.ndarray]:
        """Build 51-feature vector for the given asset.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP, DOGE).
            price_df: DataFrame with [open, high, low, close, volume, quote_volume].
                      Needs >= 50 rows.
            cross_price_df: DataFrame for the cross-asset lead indicator.
                           For BTC: pass ETH df. For alts: pass BTC df.
                           If None, cross-asset features computed from cross_data dict.
            derivatives_data: Dict with keys like 'funding_rate', 'open_interest', etc.
            macro_data: Dict with daily macro keys (spx_return_1d, vix_norm, etc.).

        Returns:
            numpy array shape (1, 51) or None if insufficient data.
        """
        if price_df is None or len(price_df) < self.MIN_ROWS:
            return None

        try:
            close = price_df['close'].values.astype(float)
            open_ = price_df['open'].values.astype(float) if 'open' in price_df.columns else close
            high = price_df['high'].values.astype(float) if 'high' in price_df.columns else close
            low = price_df['low'].values.astype(float) if 'low' in price_df.columns else close
            volume = price_df['volume'].values.astype(float) if 'volume' in price_df.columns else np.ones(len(close))

            n = len(close)
            features = np.zeros(51, dtype=np.float64)

            # -- Price/Volume Technical (15 features, indices 0-14) --
            self._compute_technical(features, close, open_, high, low, volume, n)

            # -- Daily Macro (10 features, indices 15-24) --
            self._compute_macro(features, macro_data)

            # -- Derivatives (9 features, indices 25-33) --
            self._compute_derivatives(features, derivatives_data)

            # -- Intraday Macro (11 features, indices 34-44) -- always zero
            # features[34:45] already zero from initialization

            # -- Cross-Asset Lead (6 features, indices 45-50) --
            self._compute_cross_asset(features, asset, close, n, cross_price_df)

            # Clean NaN/Inf
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            return features.reshape(1, 51)

        except Exception as e:
            logger.warning(f"CrashFeatureBuilder.build({asset}) failed: {e}")
            return None

    def _compute_technical(
        self,
        features: np.ndarray,
        close: np.ndarray,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
        n: int,
    ) -> None:
        """Compute 15 price/volume technical features (indices 0-14)."""
        # Returns
        features[0] = (close[-1] / close[-2] - 1.0) if n >= 2 else 0.0
        features[1] = (close[-1] / close[-7] - 1.0) if n >= 7 else 0.0
        features[2] = (close[-1] / close[-13] - 1.0) if n >= 13 else 0.0
        features[3] = (close[-1] / close[-49] - 1.0) if n >= 49 else 0.0
        features[4] = (close[-1] / close[-min(289, n)] - 1.0) if n >= 50 else 0.0

        # Volatility (log returns std)
        log_ret = np.diff(np.log(np.maximum(close, 1e-10)))
        features[5] = np.std(log_ret[-12:]) if len(log_ret) >= 12 else 0.0
        features[6] = np.std(log_ret[-48:]) if len(log_ret) >= 48 else 0.0
        features[7] = (features[5] / features[6]) if features[6] > 1e-10 else 1.0

        # Volume features
        vol_mean_20 = np.mean(volume[-20:]) if n >= 20 else np.mean(volume)
        features[8] = (volume[-1] / vol_mean_20) - 1.0 if vol_mean_20 > 0 else 0.0
        vol_mean_early = np.mean(volume[-20:-10]) if n >= 20 else vol_mean_20
        vol_mean_late = np.mean(volume[-10:]) if n >= 10 else vol_mean_20
        features[9] = (vol_mean_late / vol_mean_early - 1.0) if vol_mean_early > 0 else 0.0

        # Consecutive red candles
        red_count = 0
        for i in range(n - 1, max(n - 20, 0) - 1, -1):
            if close[i] < open_[i]:
                red_count += 1
            else:
                break
        features[10] = float(red_count)

        # Drawdown from 24h high
        high_24h = np.max(high[-288:]) if n >= 288 else np.max(high)
        features[11] = (close[-1] / high_24h - 1.0) if high_24h > 0 else 0.0

        # RSI (14-period, normalized to [-1, 1])
        if len(log_ret) >= 14:
            gains = np.where(log_ret[-14:] > 0, log_ret[-14:], 0)
            losses = np.where(log_ret[-14:] < 0, -log_ret[-14:], 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            rs = avg_gain / avg_loss if avg_loss > 1e-10 else 100.0
            rsi = 100.0 - (100.0 / (1.0 + rs))
            features[12] = (rsi - 50.0) / 50.0

        # Bollinger Band %B
        if n >= 20:
            sma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            if std20 > 1e-10:
                upper = sma20 + 2.0 * std20
                lower = sma20 - 2.0 * std20
                features[13] = (close[-1] - lower) / (upper - lower)
            else:
                features[13] = 0.5

        # VWAP distance
        if n >= 20 and np.sum(volume[-20:]) > 0:
            vwap = np.sum(close[-20:] * volume[-20:]) / np.sum(volume[-20:])
            features[14] = (close[-1] / vwap - 1.0) if vwap > 0 else 0.0

    def _compute_macro(
        self,
        features: np.ndarray,
        macro_data: Optional[Dict[str, float]],
    ) -> None:
        """Compute 10 daily macro features (indices 15-24)."""
        if not macro_data:
            return

        features[15] = float(macro_data.get('spx_return_1d', 0.0))
        features[16] = float(macro_data.get('spx_vs_sma', 0.0))
        features[17] = float(macro_data.get('vix_norm', 0.0))
        features[18] = float(macro_data.get('vix_change', 0.0))
        features[19] = float(macro_data.get('vix_extreme', 0.0))
        features[20] = float(macro_data.get('dxy_return_1d', 0.0))
        features[21] = float(macro_data.get('dxy_trend', 0.0))
        features[22] = float(macro_data.get('yield_level', 0.0))
        features[23] = float(macro_data.get('yield_change', 0.0))
        features[24] = float(macro_data.get('fng_norm', 0.0))

    def _compute_derivatives(
        self,
        features: np.ndarray,
        derivatives_data: Optional[Dict[str, Any]],
    ) -> None:
        """Compute 9 derivatives features (indices 25-33)."""
        if not derivatives_data:
            return

        # Funding rate z-score
        fr = derivatives_data.get('funding_rate')
        if fr is not None:
            fr_val = float(fr.iloc[-1]) if hasattr(fr, 'iloc') else float(fr)
            features[25] = fr_val / 0.0003 if abs(fr_val) > 1e-10 else 0.0
            features[26] = 1.0 if fr_val > 0.0005 else 0.0
            features[27] = 1.0 if fr_val < -0.0005 else 0.0

        # Open interest changes
        oi = derivatives_data.get('open_interest')
        if oi is not None and hasattr(oi, 'iloc') and len(oi) >= 12:
            oi_arr = oi.values.astype(float)
            oi_now = oi_arr[-1]
            if oi_now > 0:
                features[28] = (oi_now / oi_arr[-12] - 1.0)
                features[29] = (oi_now / oi_arr[-min(48, len(oi_arr))] - 1.0)
                oi_mean = np.mean(oi_arr[-48:]) if len(oi_arr) >= 48 else np.mean(oi_arr)
                oi_std = np.std(oi_arr[-48:]) if len(oi_arr) >= 48 else np.std(oi_arr)
                features[30] = (oi_now - oi_mean) / oi_std if oi_std > 1e-10 else 0.0

        # Long/short ratio
        ls = derivatives_data.get('long_short_ratio')
        if ls is not None:
            ls_val = float(ls.iloc[-1]) if hasattr(ls, 'iloc') else float(ls)
            features[31] = (ls_val - 1.0)
            features[32] = 1.0 if ls_val > 2.0 else 0.0

        # Taker buy/sell imbalance
        tbv = derivatives_data.get('taker_buy_volume')
        tsv = derivatives_data.get('taker_sell_volume')
        if tbv is not None and tsv is not None:
            tb = float(tbv.iloc[-1]) if hasattr(tbv, 'iloc') else float(tbv)
            ts = float(tsv.iloc[-1]) if hasattr(tsv, 'iloc') else float(tsv)
            total = tb + ts
            features[33] = (tb - ts) / total if total > 0 else 0.0

    def _compute_cross_asset(
        self,
        features: np.ndarray,
        asset: str,
        close: np.ndarray,
        n: int,
        cross_price_df: Any,
    ) -> None:
        """Compute 6 cross-asset lead features (indices 45-50).

        For BTC: cross = ETH (ETH data goes into slots 45-50)
        For alts: cross = BTC (BTC data goes into slots 45-50)

        The feature column names in training are btc_return_1bar/eth_return_1bar
        etc., but the MODEL only cares about position, not name. Position 45-50
        always contain cross-asset lead data regardless of which asset is the lead.
        """
        asset_upper = asset.upper()
        lead_asset = CROSS_ASSET_LEAD.get(asset_upper, 'BTC')

        if cross_price_df is not None and hasattr(cross_price_df, 'empty') and not cross_price_df.empty:
            if 'close' in cross_price_df.columns:
                cross_close = cross_price_df['close'].values.astype(float)
                cn = len(cross_close)

                # Cross-asset returns
                if cn >= 2:
                    features[45] = cross_close[-1] / cross_close[-2] - 1.0
                if cn >= 7:
                    features[46] = cross_close[-1] / cross_close[-7] - 1.0

                # Asset / cross-asset ratio change
                if cn >= 7 and cross_close[-1] > 0 and cross_close[-7] > 0:
                    ratio_now = close[-1] / cross_close[-1]
                    ratio_prev = close[-7] / cross_close[-7]
                    features[47] = ratio_now / ratio_prev - 1.0

        # Lead signals: lagged returns of the LEAD asset
        # For BTC model: lead = self (BTC leads itself in lagged form)
        # For alt models: lead = BTC (BTC leads alts)
        if asset_upper == 'BTC':
            # BTC model: lead signals are BTC's own lagged returns
            lead_close = close
        elif cross_price_df is not None and hasattr(cross_price_df, 'empty') and not cross_price_df.empty:
            lead_close = cross_price_df['close'].values.astype(float)
        else:
            lead_close = close  # Fallback to self

        ln = len(lead_close)
        if ln >= 3:
            features[48] = lead_close[-2] / lead_close[-3] - 1.0
        if ln >= 4:
            features[49] = lead_close[-3] / lead_close[-4] - 1.0
        if ln >= 5:
            features[50] = lead_close[-4] / lead_close[-5] - 1.0
