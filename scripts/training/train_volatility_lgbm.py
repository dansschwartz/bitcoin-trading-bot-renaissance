#!/usr/bin/env python3
"""Train Volatility LightGBM — predicts price move MAGNITUDE over 6 bars (30min).

This is a regression model that learns to predict how MUCH price will move,
regardless of direction. The label is:

    label = log1p(abs(forward_6bar_return_bps))
    where forward_6bar_return_bps = (close[t+6] / close[t] - 1) * 10000

The predicted magnitude is classified into volatility regimes:
    dead_zone:  predicted < p25
    normal:     p25 <= predicted < p75
    active:     p75 <= predicted < p90
    explosive:  predicted >= p90

Features: 46 base (from _compute_single_pair_features) + 13 volatility-specific = 59

Usage:
    python -m scripts.training.train_volatility_lgbm
    python -m scripts.training.train_volatility_lgbm --days 90 --rounds 3000
    python -m scripts.training.train_volatility_lgbm --assets BTCUSDT ETHUSDT
"""

import argparse
import logging
import math
import os
import pickle
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ── Path setup ────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_model_loader import _compute_single_pair_features

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT",
]
LOOKBACK_DAYS = 180
BAR_INTERVAL = "5m"
BAR_MS = 5 * 60 * 1000          # 5 minutes in milliseconds
FORWARD_BARS = 6                 # 30 minutes
WARMUP = 100                     # Rows needed for indicator warmup (longest rolling window)
BINANCE_BATCH = 1000             # Binance max klines per request
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "volatility_lgbm.pkl")

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "seed": 42,
}

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
NUM_BOOST_ROUND = 3000
EARLY_STOPPING = 100


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_binance_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch historical klines from Binance with reverse-window pagination.

    Uses endTime-based backward pagination to handle the 1000-row limit.
    Fetches from (now - days) to now.

    Args:
        symbol: Binance symbol e.g. "BTCUSDT"
        interval: Kline interval e.g. "5m"
        days: Number of days of history to fetch

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume]
        sorted by timestamp ascending. timestamp is Unix seconds.
    """
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 60 * 60 * 1000)

    all_rows: List[list] = []
    cursor_end_ms = end_ms

    total_bars_expected = (days * 24 * 60) // 5
    batches_expected = math.ceil(total_bars_expected / BINANCE_BATCH)
    batch_num = 0

    while cursor_end_ms > start_ms:
        batch_num += 1
        params = {
            "symbol": symbol,
            "interval": interval,
            "endTime": cursor_end_ms,
            "limit": BINANCE_BATCH,
        }

        # Set startTime so we don't go before our target
        batch_start_ms = max(start_ms, cursor_end_ms - BINANCE_BATCH * BAR_MS)
        params["startTime"] = batch_start_ms

        try:
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"  {symbol}: batch {batch_num} request failed: {e}")
            time.sleep(2.0)
            continue
        except Exception as e:
            logger.warning(f"  {symbol}: batch {batch_num} parse failed: {e}")
            time.sleep(2.0)
            continue

        if not data:
            break

        # Binance kline format:
        # [open_time, open, high, low, close, volume, close_time, ...]
        for row in data:
            all_rows.append([
                int(row[0]) // 1000,    # open_time → Unix seconds
                float(row[1]),          # open
                float(row[2]),          # high
                float(row[3]),          # low
                float(row[4]),          # close
                float(row[5]),          # volume
            ])

        # Move cursor backward: earliest candle's open_time - 1ms
        earliest_open_ms = int(data[0][0])
        cursor_end_ms = earliest_open_ms - 1

        if batch_num % 20 == 0:
            logger.info(f"  {symbol}: batch {batch_num}/{batches_expected} — "
                        f"{len(all_rows):,} bars so far")

        # Binance rate limit: weight=2 per klines request, 2400/min limit
        time.sleep(0.15)

    if not all_rows:
        logger.warning(f"  {symbol}: no klines fetched")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    logger.info(f"  {symbol}: fetched {len(df):,} bars "
                f"({len(df) * 5 / (60 * 24):.1f} days)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Parkinson volatility estimator: mean of ln(H/L)^2 / (4*ln2) over window."""
    log_hl_sq = (np.log(high / (low + 1e-10))) ** 2
    return (log_hl_sq.rolling(window, min_periods=max(1, window // 2)).mean()
            / (4.0 * np.log(2.0)))


def _garman_klass_vol(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    """Garman-Klass volatility estimator over rolling window.

    GK = 0.5 * ln(H/L)^2 - (2*ln2 - 1) * ln(C/O)^2
    Then take rolling mean.
    """
    log_hl_sq = (np.log(high / (low + 1e-10))) ** 2
    log_co_sq = (np.log(close / (open_p + 1e-10))) ** 2
    gk_single = 0.5 * log_hl_sq - (2.0 * np.log(2.0) - 1.0) * log_co_sq
    return gk_single.rolling(window, min_periods=max(1, window // 2)).mean()


def build_volatility_features(
    df: pd.DataFrame,
    btc_df: Optional[pd.DataFrame] = None,
    is_btc: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build features for volatility prediction.

    Uses the base 46 features from _compute_single_pair_features()
    plus 13 volatility-specific additions.

    Args:
        df: OHLCV DataFrame with columns [timestamp, open, high, low, close, volume]
        btc_df: BTC OHLCV DataFrame (for cross-asset vol). Used when predicting alts.
                 When predicting BTC, pass ETH data here instead.
        is_btc: True if this asset IS BTC (will use btc_df as ETH cross-asset data)

    Returns:
        (features_df, feature_names) where features_df has one column per feature
    """
    # ── Base 46 features from the production pipeline ─────────────────────────
    base_feats = _compute_single_pair_features(df)
    base_names = list(base_feats.keys())

    # Combine into a DataFrame aligned on the original index
    feat_df = pd.DataFrame(base_feats, index=df.index)

    close = df["close"].astype(float)
    open_p = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # ── Volatility-specific features (13 additional) ──────────────────────────

    extra_names: List[str] = []

    # 1-2: Realized vol ratios
    pct_ret = close.pct_change()
    vol_5 = pct_ret.rolling(5).std()
    vol_10 = pct_ret.rolling(10).std()
    vol_20 = pct_ret.rolling(20).std()

    feat_df["vol_ratio_5_20"] = vol_5 / (vol_20 + 1e-10)
    feat_df["vol_ratio_10_20"] = vol_10 / (vol_20 + 1e-10)
    extra_names.extend(["vol_ratio_5_20", "vol_ratio_10_20"])

    # 3-5: Parkinson volatility over 5, 10, 20 bars
    feat_df["parkinson_vol_5"] = _parkinson_vol(high, low, 5)
    feat_df["parkinson_vol_10"] = _parkinson_vol(high, low, 10)
    feat_df["parkinson_vol_20"] = _parkinson_vol(high, low, 20)
    extra_names.extend(["parkinson_vol_5", "parkinson_vol_10", "parkinson_vol_20"])

    # 6: Garman-Klass volatility over 10 bars
    feat_df["garman_klass_vol_10"] = _garman_klass_vol(open_p, high, low, close, 10)
    extra_names.append("garman_klass_vol_10")

    # 7-8: Hour of day sin/cos
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    else:
        ts = pd.to_datetime(df.index, unit="s", utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60.0
    feat_df["hour_sin"] = np.sin(2.0 * np.pi * hour.values / 24.0)
    feat_df["hour_cos"] = np.cos(2.0 * np.pi * hour.values / 24.0)
    extra_names.extend(["hour_sin", "hour_cos"])

    # 9-10: Day of week sin/cos
    dow = ts.dt.dayofweek.values.astype(float)
    feat_df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    feat_df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    extra_names.extend(["dow_sin", "dow_cos"])

    # 11: Cross-asset volatility
    # For alts: use BTC 5-bar vol. For BTC: use ETH 5-bar vol.
    if btc_df is not None and len(btc_df) >= len(df):
        cross_close = btc_df["close"].astype(float).values[-len(df):]
        cross_ret = pd.Series(cross_close).pct_change()
        cross_vol = cross_ret.rolling(5).std()
        feat_df["cross_asset_vol"] = cross_vol.values
    else:
        feat_df["cross_asset_vol"] = 0.0
    extra_names.append("cross_asset_vol")

    # 12: OI change pct (not available from spot klines, fill with 0)
    feat_df["oi_change_pct"] = 0.0
    extra_names.append("oi_change_pct")

    # 13: Funding rate z-score (not available from spot klines, fill with 0)
    feat_df["funding_rate_z"] = 0.0
    extra_names.append("funding_rate_z")

    feature_names = base_names + extra_names
    return feat_df[feature_names], feature_names


def build_labels(df: pd.DataFrame, forward_bars: int = FORWARD_BARS) -> pd.Series:
    """Build volatility labels: log1p(abs(forward_return_bps)).

    Args:
        df: DataFrame with 'close' column
        forward_bars: Number of bars to look forward (default 6 = 30min)

    Returns:
        pd.Series of labels, NaN for last forward_bars rows
    """
    close = df["close"].astype(float)
    future_close = close.shift(-forward_bars)
    forward_return_bps = (future_close / close - 1.0) * 10000.0
    label = np.log1p(np.abs(forward_return_bps))
    return label


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_volatility_lgbm(
    assets: List[str] = None,
    days: int = LOOKBACK_DAYS,
    num_boost_round: int = NUM_BOOST_ROUND,
    save_path: str = SAVE_PATH,
) -> dict:
    """Fetch data, build features, train LightGBM volatility model.

    Args:
        assets: List of Binance symbols to train on
        days: Days of historical data per asset
        num_boost_round: Maximum boosting rounds
        save_path: Path to save the pickled model

    Returns:
        Dict with training metadata
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "LightGBM not installed. Run: pip install lightgbm"
        )

    if assets is None:
        assets = ASSETS

    # ── Phase 1: Fetch data ───────────────────────────────────────────────────
    logger.info(f"Fetching {days} days of {BAR_INTERVAL} klines for {len(assets)} assets...")

    raw_dfs: Dict[str, pd.DataFrame] = {}
    for symbol in assets:
        logger.info(f"Fetching {symbol}...")
        df = fetch_binance_klines(symbol, BAR_INTERVAL, days)
        if df is not None and len(df) > WARMUP + FORWARD_BARS + 50:
            raw_dfs[symbol] = df
        else:
            logger.warning(f"  {symbol}: insufficient data ({len(df) if df is not None else 0} bars), skipping")

    if not raw_dfs:
        raise RuntimeError("No data fetched for any asset")

    logger.info(f"Data fetched: {len(raw_dfs)} assets, "
                f"{sum(len(d) for d in raw_dfs.values()):,} total bars")

    # Identify BTC and ETH for cross-asset features
    btc_df = raw_dfs.get("BTCUSDT")
    eth_df = raw_dfs.get("ETHUSDT")

    # ── Phase 2: Build features + labels ──────────────────────────────────────
    logger.info("Building features and labels...")

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    feature_names: Optional[List[str]] = None
    total_samples_per_asset: Dict[str, int] = {}

    for symbol, df in raw_dfs.items():
        is_btc = (symbol == "BTCUSDT")

        # Cross-asset: alts get BTC vol, BTC gets ETH vol
        if is_btc:
            cross_df = eth_df
        else:
            cross_df = btc_df

        # Build features
        feat_df, names = build_volatility_features(df, btc_df=cross_df, is_btc=is_btc)

        if feature_names is None:
            feature_names = names
        else:
            # Verify feature names match across assets
            if names != feature_names:
                logger.warning(f"  {symbol}: feature names mismatch, skipping")
                continue

        # Build labels
        labels = build_labels(df, FORWARD_BARS)

        # Create valid mask: no NaN in features or labels, after warmup
        feat_matrix = feat_df.values.astype(np.float64)
        label_arr = labels.values.astype(np.float64)

        valid_mask = np.ones(len(df), dtype=bool)
        valid_mask[:WARMUP] = False                        # Warmup period
        valid_mask[len(df) - FORWARD_BARS:] = False        # No future data for labels
        valid_mask &= ~np.isnan(label_arr)                 # Valid labels
        valid_mask &= ~np.any(np.isnan(feat_matrix), axis=1)  # Valid features
        valid_mask &= ~np.any(np.isinf(feat_matrix), axis=1)  # No inf

        n_valid = valid_mask.sum()
        if n_valid < 100:
            logger.warning(f"  {symbol}: only {n_valid} valid samples, skipping")
            continue

        all_X.append(feat_matrix[valid_mask])
        all_y.append(label_arr[valid_mask])
        total_samples_per_asset[symbol] = n_valid
        logger.info(f"  {symbol}: {n_valid:,} valid samples "
                    f"(of {len(df):,} bars)")

    if not all_X:
        raise RuntimeError("No valid samples from any asset")

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)

    # Replace any remaining NaN/inf (belt + suspenders)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Total: {len(X):,} samples, {X.shape[1]} features")
    logger.info(f"Label stats: mean={y.mean():.3f}, std={y.std():.3f}, "
                f"min={y.min():.3f}, max={y.max():.3f}")

    # ── Phase 3: Time-based train/val/test split ──────────────────────────────
    # Since data is pooled across assets but each asset's data is time-ordered,
    # we split the concatenated array by position (each asset contributes its
    # time-ordered chunk, and within each chunk earlier bars come first).
    n = len(X)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(f"Split: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")

    # ── Phase 4: Train LightGBM ──────────────────────────────────────────────
    logger.info(f"Training LightGBM: {num_boost_round} max rounds, "
                f"lr={LGB_PARAMS['learning_rate']}, leaves={LGB_PARAMS['num_leaves']}, "
                f"depth={LGB_PARAMS['max_depth']}")

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        LGB_PARAMS,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # ── Phase 5: Evaluate ─────────────────────────────────────────────────────
    logger.info("Evaluating...")

    results = {}
    for split_name, X_s, y_s in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        preds = model.predict(X_s)
        mae = np.mean(np.abs(preds - y_s))
        rmse = np.sqrt(np.mean((preds - y_s) ** 2))

        # Rank correlation (Spearman) — does the model rank volatility correctly?
        from scipy.stats import spearmanr
        try:
            spearman_corr, spearman_p = spearmanr(preds, y_s)
        except Exception:
            spearman_corr, spearman_p = 0.0, 1.0

        results[f"{split_name}_mae"] = float(mae)
        results[f"{split_name}_rmse"] = float(rmse)
        results[f"{split_name}_spearman"] = float(spearman_corr)

        logger.info(f"  {split_name:5s}: MAE={mae:.4f}, RMSE={rmse:.4f}, "
                    f"Spearman={spearman_corr:.4f} (p={spearman_p:.2e})")

    # ── Phase 6: Compute label percentiles for regime thresholds ──────────────
    # Use TRAINING labels to define thresholds (avoids data leakage)
    percentiles = {
        "p25": float(np.percentile(y_train, 25)),
        "p50": float(np.percentile(y_train, 50)),
        "p75": float(np.percentile(y_train, 75)),
        "p90": float(np.percentile(y_train, 90)),
    }
    logger.info(f"Label percentiles (training set):")
    logger.info(f"  p25={percentiles['p25']:.3f} (dead_zone threshold)")
    logger.info(f"  p50={percentiles['p50']:.3f}")
    logger.info(f"  p75={percentiles['p75']:.3f} (active threshold)")
    logger.info(f"  p90={percentiles['p90']:.3f} (explosive threshold)")

    # Show regime distribution on test set
    test_preds = model.predict(X_test)
    n_dead = (test_preds < percentiles["p25"]).sum()
    n_normal = ((test_preds >= percentiles["p25"]) & (test_preds < percentiles["p75"])).sum()
    n_active = ((test_preds >= percentiles["p75"]) & (test_preds < percentiles["p90"])).sum()
    n_explosive = (test_preds >= percentiles["p90"]).sum()
    n_total = len(test_preds)

    logger.info(f"Regime distribution on test set:")
    logger.info(f"  dead_zone:  {n_dead:>6,} ({100*n_dead/n_total:.1f}%)")
    logger.info(f"  normal:     {n_normal:>6,} ({100*n_normal/n_total:.1f}%)")
    logger.info(f"  active:     {n_active:>6,} ({100*n_active/n_total:.1f}%)")
    logger.info(f"  explosive:  {n_explosive:>6,} ({100*n_explosive/n_total:.1f}%)")

    # ── Phase 7: Feature importance ───────────────────────────────────────────
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])

    print(f"\n{'='*65}")
    print(f"{'TOP 20 FEATURE IMPORTANCE (gain)':^65}")
    print(f"{'='*65}")
    max_imp = max(importance) if max(importance) > 0 else 1.0
    for i, (fname, imp) in enumerate(feat_imp[:20]):
        bar = "#" * min(int(imp / max_imp * 40), 40)
        print(f"  {i+1:2d}. {fname:<25s} {imp:>12.1f}  {bar}")

    # ── Phase 8: Save model ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model_artifact = {
        "model": model,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "label_percentiles": percentiles,
        "train_mae": results["train_mae"],
        "val_mae": results["val_mae"],
        "test_mae": results["test_mae"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_samples": len(X_train),
        "assets_used": list(raw_dfs.keys()),
        "results": results,
        "best_iteration": model.best_iteration,
        "regime_thresholds": {
            "dead_zone": f"predicted < {percentiles['p25']:.3f}",
            "normal": f"{percentiles['p25']:.3f} <= predicted < {percentiles['p75']:.3f}",
            "active": f"{percentiles['p75']:.3f} <= predicted < {percentiles['p90']:.3f}",
            "explosive": f"predicted >= {percentiles['p90']:.3f}",
        },
        "feature_importance_top20": [
            {"feature": fname, "gain": float(imp)}
            for fname, imp in feat_imp[:20]
        ],
    }

    with open(save_path, "wb") as f:
        pickle.dump(model_artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = os.path.getsize(save_path) / 1024
    logger.info(f"Saved model to {save_path} ({size_kb:.0f} KB)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'VOLATILITY LGBM TRAINING RESULTS':^65}")
    print(f"{'='*65}")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Model size: {size_kb:.0f} KB")
    print(f"  Assets: {', '.join(raw_dfs.keys())}")
    print()
    print(f"  {'Split':<8} {'MAE':>8} {'RMSE':>8} {'Spearman':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for split_name in ["train", "val", "test"]:
        mae = results[f"{split_name}_mae"]
        rmse = results[f"{split_name}_rmse"]
        sp = results[f"{split_name}_spearman"]
        print(f"  {split_name:<8} {mae:>8.4f} {rmse:>8.4f} {sp:>10.4f}")

    print(f"\n  Regime Thresholds (from training label percentiles):")
    print(f"    dead_zone:  predicted < {percentiles['p25']:.3f}")
    print(f"    normal:     {percentiles['p25']:.3f} <= predicted < {percentiles['p75']:.3f}")
    print(f"    active:     {percentiles['p75']:.3f} <= predicted < {percentiles['p90']:.3f}")
    print(f"    explosive:  predicted >= {percentiles['p90']:.3f}")

    print(f"\n  Key insight: Spearman correlation on TEST measures ranking quality.")
    print(f"  Higher Spearman = better at distinguishing high-vol from low-vol periods.")
    print(f"\n  Saved to: {save_path}")

    return model_artifact


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train Volatility LightGBM — predicts price move magnitude over 30min"
    )
    parser.add_argument(
        "--assets", nargs="+", default=ASSETS,
        help=f"Binance symbols to train on (default: {' '.join(ASSETS)})"
    )
    parser.add_argument(
        "--days", type=int, default=LOOKBACK_DAYS,
        help=f"Days of historical data per asset (default: {LOOKBACK_DAYS})"
    )
    parser.add_argument(
        "--rounds", type=int, default=NUM_BOOST_ROUND,
        help=f"Max boosting rounds (default: {NUM_BOOST_ROUND})"
    )
    parser.add_argument(
        "--save-path", default=SAVE_PATH,
        help=f"Path to save model pickle (default: {SAVE_PATH})"
    )
    args = parser.parse_args()

    result = train_volatility_lgbm(
        assets=args.assets,
        days=args.days,
        num_boost_round=args.rounds,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
