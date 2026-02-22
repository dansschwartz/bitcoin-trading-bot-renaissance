#!/usr/bin/env python3
"""
Backtest ML Pipeline — validates recent pipeline fixes by replaying historical data
through all 7 models and comparing FIXED vs OLD decision logic.

READ-ONLY: Does not modify any production code or databases.

Recent pipeline fixes tested:
  Bug 3: confidence denominator 0.05 -> 0.02
  Bug 7: thresholds +/-0.01 -> +/-0.015
  Bug 5: confidence_floor 0.505 -> 0.48, signal TTL 5min -> 30min (6 bars)
  Bug 1: ML signals now in weight map with 10x scale
  Bug 4: position size normalization ($50-300 range)

Usage:
  cd /Users/danielschwartz/Downloads/bitcoin-trading-bot-renaissance
  .venv/bin/python3 scripts/backtest_ml_pipeline.py
"""

import argparse
import json as _json
import sys
import os
import time
import math
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")

# Suppress noisy torch warnings
logging.getLogger("torch").setLevel(logging.WARNING)

import torch

# Import from project
from ml_model_loader import (
    INPUT_DIM,
    build_feature_sequence,
    build_full_feature_matrix,
    load_trained_models,
    predict_with_models,
    _compute_single_pair_features,
    _build_cross_features,
    _build_derivatives_features,
    LEAD_SIGNALS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML Pipeline Backtest")
    parser.add_argument(
        "--json-config", type=str, default=None,
        help="Path to JSON file with config overrides for all backtest constants",
    )
    parser.add_argument(
        "--json-progress", action="store_true", default=False,
        help="Emit structured JSON progress lines to stdout (for dashboard integration)",
    )
    return parser.parse_args()

_CLI_ARGS = _parse_args()

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS (overridable via --json-config)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_config() -> dict:
    """Load config from JSON file if --json-config was provided, else return empty dict."""
    if _CLI_ARGS.json_config:
        with open(_CLI_ARGS.json_config, "r") as f:
            return _json.load(f)
    return {}

_CFG = _load_config()

TRAINING_DIR = os.path.join(PROJECT_ROOT, "data", "training")
SEQ_LEN = 30           # Feature window length
WARMUP = 200           # Bars to skip for indicator warmup
LOOKAHEAD = int(_CFG.get("lookahead", 6))
COST_BPS = float(_CFG.get("cost_bps", 0.0065))
PROGRESS_INTERVAL = 1000  # Log progress every N candles

# FIXED pipeline params (post-fix)
NEW_DENOM = float(_CFG.get("new_denom", 0.02))
NEW_BUY_THRESH = float(_CFG.get("new_buy_thresh", 0.015))
NEW_SELL_THRESH = float(_CFG.get("new_sell_thresh", -0.015))
NEW_CONF_FLOOR = float(_CFG.get("new_conf_floor", 0.48))
NEW_SIGNAL_SCALE = float(_CFG.get("new_signal_scale", 10.0))
NEW_EXIT_BARS = int(_CFG.get("new_exit_bars", 6))
NEW_POS_MIN = float(_CFG.get("new_pos_min", 50.0))
NEW_POS_MAX = float(_CFG.get("new_pos_max", 300.0))
NEW_POS_BASE = float(_CFG.get("new_pos_base", 100.0))

# OLD pipeline params (pre-fix)
OLD_DENOM = float(_CFG.get("old_denom", 0.05))
OLD_BUY_THRESH = float(_CFG.get("old_buy_thresh", 0.01))
OLD_SELL_THRESH = float(_CFG.get("old_sell_thresh", -0.01))
OLD_CONF_FLOOR = float(_CFG.get("old_conf_floor", 0.505))
OLD_SIGNAL_SCALE = float(_CFG.get("old_signal_scale", 1.0))
OLD_EXIT_BARS = int(_CFG.get("old_exit_bars", 1))
OLD_POS_USD = float(_CFG.get("old_pos_usd", 75.0))

# Pairs to backtest
PAIRS = _CFG.get("pairs", ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "AVAX-USD", "DOGE-USD"])

# JSON progress mode
JSON_PROGRESS = _CLI_ARGS.json_progress


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_pair_data(pair: str) -> Optional[pd.DataFrame]:
    """Load the most recent ~30 days of 5-min OHLCV data for a pair."""
    # Prefer the recent data file (pair.csv) over the massive historical file
    recent_path = os.path.join(TRAINING_DIR, f"{pair}.csv")
    historical_path = os.path.join(TRAINING_DIR, f"{pair}_5m_historical.csv")

    path = recent_path if os.path.exists(recent_path) else historical_path
    if not os.path.exists(path):
        logger.warning(f"No data file for {pair}")
        return None

    logger.info(f"Loading {pair} from {os.path.basename(path)}...")
    df = pd.read_csv(path)

    # Ensure required columns
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"{pair}: missing columns {missing}")
        return None

    # Convert numeric
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])

    # Use last 8640 rows (30 days of 5-min bars)
    if len(df) > 8640:
        df = df.tail(8640).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    logger.info(f"  {pair}: {len(df)} bars loaded")
    return df


def load_all_pairs() -> Dict[str, pd.DataFrame]:
    """Load data for all available pairs."""
    pair_data = {}
    for pair in PAIRS:
        df = load_pair_data(pair)
        if df is not None and len(df) >= WARMUP + SEQ_LEN + LOOKAHEAD:
            pair_data[pair] = df
        else:
            logger.warning(f"  {pair}: insufficient data, skipping")
    return pair_data


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDING (uses build_full_feature_matrix for efficiency)
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_features(
    pair_data: Dict[str, pd.DataFrame],
) -> Dict[str, np.ndarray]:
    """Precompute the full feature matrix for each pair.

    Returns dict of pair_name -> (N, INPUT_DIM) float32 array.
    """
    # Build cross_data dict for cross-asset features
    cross_data = {}
    for pair, df in pair_data.items():
        cross_data[pair] = df[["close", "volume"]].copy()

    feature_matrices = {}
    for pair, df in pair_data.items():
        logger.info(f"Computing features for {pair}...")
        feat_mat = build_full_feature_matrix(
            price_df=df,
            cross_data=cross_data,
            pair_name=pair,
            derivatives_data=None,
        )
        if feat_mat is not None:
            feature_matrices[pair] = feat_mat
            logger.info(f"  {pair}: feature matrix shape {feat_mat.shape}")
        else:
            logger.warning(f"  {pair}: feature build failed")
    return feature_matrices


def get_windowed_features(
    feat_matrix: np.ndarray,
    idx: int,
    seq_len: int = SEQ_LEN,
) -> Optional[np.ndarray]:
    """Extract a (seq_len, INPUT_DIM) window ending at idx (inclusive).

    Applies per-window standardization to match build_feature_sequence().
    """
    start = idx - seq_len + 1
    if start < 0:
        return None
    window = feat_matrix[start:idx + 1].copy()  # (seq_len, INPUT_DIM)

    # Per-window standardization (matches build_feature_sequence)
    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True) + 1e-8
    window = (window - mean) / std

    return window.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# DECISION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def compute_agreement(predictions: Dict[str, float]) -> float:
    """Fraction of models agreeing on direction with the ensemble."""
    if "meta_ensemble" not in predictions:
        return 0.0
    ensemble_pred = predictions["meta_ensemble"]
    if abs(ensemble_pred) < 1e-10:
        return 0.5
    ensemble_sign = 1.0 if ensemble_pred > 0 else -1.0
    base_models = [n for n in predictions if n != "meta_ensemble"]
    if not base_models:
        return 0.5
    agrees = sum(1 for n in base_models
                 if (predictions[n] > 0 and ensemble_sign > 0) or
                    (predictions[n] < 0 and ensemble_sign < 0))
    return agrees / len(base_models)


def new_decision(predictions: Dict[str, float]) -> Tuple[str, float, float, float]:
    """FIXED pipeline decision logic.

    Returns (action, confidence, ml_signal, position_usd).
    """
    ensemble_pred = predictions.get("meta_ensemble", 0.0)
    agreement = compute_agreement(predictions)

    signal_strength = min(abs(ensemble_pred) / NEW_DENOM, 1.0)
    confidence = math.sqrt(signal_strength * agreement) if (signal_strength * agreement) > 0 else 0.0
    ml_signal = ensemble_pred * NEW_SIGNAL_SCALE

    if confidence >= NEW_CONF_FLOOR and ml_signal > NEW_BUY_THRESH:
        action = "BUY"
    elif confidence >= NEW_CONF_FLOOR and ml_signal < NEW_SELL_THRESH:
        action = "SELL"
    else:
        action = "HOLD"

    # Position sizing (Bug 4 fix)
    sig_scalar = min(abs(ml_signal) / 0.1, 1.5)  # signal magnitude scalar
    conf_scalar = confidence  # confidence scalar
    position_usd = max(NEW_POS_MIN, min(NEW_POS_BASE * sig_scalar * conf_scalar, NEW_POS_MAX))

    return action, confidence, ml_signal, position_usd


def old_decision(predictions: Dict[str, float]) -> Tuple[str, float, float, float]:
    """OLD pipeline decision logic (pre-fix).

    Returns (action, confidence, ml_signal, position_usd).
    """
    ensemble_pred = predictions.get("meta_ensemble", 0.0)
    agreement = compute_agreement(predictions)

    signal_strength = min(abs(ensemble_pred) / OLD_DENOM, 1.0)
    confidence = math.sqrt(signal_strength * agreement) if (signal_strength * agreement) > 0 else 0.0
    ml_signal = ensemble_pred * OLD_SIGNAL_SCALE

    if confidence >= OLD_CONF_FLOOR and ml_signal > OLD_BUY_THRESH:
        action = "BUY"
    elif confidence >= OLD_CONF_FLOOR and ml_signal < OLD_SELL_THRESH:
        action = "SELL"
    else:
        action = "HOLD"

    position_usd = OLD_POS_USD  # flat sizing in old pipeline

    return action, confidence, ml_signal, position_usd


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

class TradeResult:
    """Single trade result."""
    def __init__(self, entry_idx: int, exit_idx: int, direction: str,
                 position_usd: float, entry_price: float, exit_price: float,
                 pnl: float):
        self.entry_idx = entry_idx
        self.exit_idx = exit_idx
        self.direction = direction
        self.position_usd = position_usd
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.pnl = pnl

    @property
    def won(self) -> bool:
        return self.pnl > 0


class PipelineSimulator:
    """Simulates trading for a single pipeline version (FIXED or OLD)."""

    def __init__(self, name: str, exit_bars: int, cost_bps: float = COST_BPS):
        self.name = name
        self.exit_bars = exit_bars
        self.cost_bps = cost_bps
        self.trades: List[TradeResult] = []
        self.equity_curve: List[float] = []
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 0}

        # Active position tracking
        self.position_dir: Optional[str] = None  # "BUY" or "SELL"
        self.position_entry_idx: int = 0
        self.position_entry_price: float = 0.0
        self.position_usd: float = 0.0
        self.bars_in_position: int = 0

        self.cumulative_pnl: float = 0.0

    def step(self, idx: int, close_price: float,
             action: str, confidence: float, ml_signal: float, position_usd: float) -> None:
        """Process one bar."""
        self.signals[action] = self.signals.get(action, 0) + 1

        # Check if current position should exit
        if self.position_dir is not None:
            self.bars_in_position += 1
            if self.bars_in_position >= self.exit_bars:
                # Exit position
                if self.position_dir == "BUY":
                    raw_ret = (close_price - self.position_entry_price) / self.position_entry_price
                else:
                    raw_ret = (self.position_entry_price - close_price) / self.position_entry_price
                pnl = self.position_usd * (raw_ret - self.cost_bps)
                self.cumulative_pnl += pnl
                self.trades.append(TradeResult(
                    entry_idx=self.position_entry_idx,
                    exit_idx=idx,
                    direction=self.position_dir,
                    position_usd=self.position_usd,
                    entry_price=self.position_entry_price,
                    exit_price=close_price,
                    pnl=pnl,
                ))
                self.position_dir = None

        # Enter new position if no current position and action is not HOLD
        if self.position_dir is None and action in ("BUY", "SELL"):
            self.position_dir = action
            self.position_entry_idx = idx
            self.position_entry_price = close_price
            self.position_usd = position_usd
            self.bars_in_position = 0

        self.equity_curve.append(self.cumulative_pnl)

    def report(self) -> Dict:
        """Generate summary statistics."""
        n_trades = len(self.trades)
        if n_trades == 0:
            return {
                "name": self.name,
                "n_trades": 0,
                "signals": dict(self.signals),
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "avg_position_usd": 0.0,
            }

        wins = sum(1 for t in self.trades if t.won)
        pnls = [t.pnl for t in self.trades]
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / n_trades
        avg_pos = sum(t.position_usd for t in self.trades) / n_trades

        # Sharpe (annualized from 5-min bars)
        pnl_arr = np.array(pnls)
        if pnl_arr.std() > 0:
            # Bars per year = 288 * 365 = 105120
            sharpe = (pnl_arr.mean() / pnl_arr.std()) * math.sqrt(105120)
        else:
            sharpe = 0.0

        # Max drawdown on equity curve
        eq = np.array(self.equity_curve)
        if len(eq) > 0:
            peak = np.maximum.accumulate(eq)
            drawdown = peak - eq
            max_drawdown = float(drawdown.max()) if len(drawdown) > 0 else 0.0
        else:
            max_drawdown = 0.0

        return {
            "name": self.name,
            "n_trades": n_trades,
            "signals": dict(self.signals),
            "win_rate": wins / n_trades * 100,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "avg_position_usd": avg_pos,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PER-MODEL ACCURACY TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

class ModelAccuracyTracker:
    """Track directional accuracy per model."""

    def __init__(self):
        self.predictions: Dict[str, List[float]] = {}
        self.forward_returns: List[float] = []

    def record(self, predictions: Dict[str, float], forward_return: float) -> None:
        for name, pred in predictions.items():
            if name not in self.predictions:
                self.predictions[name] = []
            self.predictions[name].append(pred)
        self.forward_returns.append(forward_return)

    def compute_accuracy(self) -> Dict[str, Dict]:
        """Compute directional accuracy for each model."""
        results = {}
        fwd = np.array(self.forward_returns)
        fwd_direction = np.sign(fwd)

        for name, preds in self.predictions.items():
            pred_arr = np.array(preds)
            pred_direction = np.sign(pred_arr)

            # Only count where both are non-zero
            mask = (fwd_direction != 0) & (pred_direction != 0)
            if mask.sum() == 0:
                results[name] = {
                    "accuracy": 0.0,
                    "n_predictions": len(preds),
                    "n_nonzero": 0,
                    "mean_pred": float(pred_arr.mean()),
                    "std_pred": float(pred_arr.std()),
                    "abs_mean": float(np.abs(pred_arr).mean()),
                }
                continue

            correct = (pred_direction[mask] == fwd_direction[mask]).sum()
            total = mask.sum()

            results[name] = {
                "accuracy": correct / total * 100,
                "n_predictions": len(preds),
                "n_nonzero": int(total),
                "mean_pred": float(pred_arr.mean()),
                "std_pred": float(pred_arr.std()),
                "abs_mean": float(np.abs(pred_arr).mean()),
            }

        return results

    def compute_correlation_matrix(self) -> pd.DataFrame:
        """Compute pairwise correlation matrix between model predictions."""
        names = sorted(self.predictions.keys())
        if len(names) < 2:
            return pd.DataFrame()
        pred_df = pd.DataFrame({n: self.predictions[n] for n in names})
        return pred_df.corr()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def _merge_simulators(sims: List[PipelineSimulator], name: str, exit_bars: int) -> PipelineSimulator:
    """Merge per-pair simulators into a single aggregate simulator."""
    merged = PipelineSimulator(name, exit_bars=exit_bars)
    for s in sims:
        merged.trades.extend(s.trades)
        merged.equity_curve.extend(s.equity_curve)
        merged.cumulative_pnl += s.cumulative_pnl
        for k, v in s.signals.items():
            merged.signals[k] = merged.signals.get(k, 0) + v
    # Re-build equity curve as cumulative sum of trade P&Ls sorted by exit index
    all_trades_sorted = sorted(merged.trades, key=lambda t: t.exit_idx)
    cum = 0.0
    merged.equity_curve = []
    for t in all_trades_sorted:
        cum += t.pnl
        merged.equity_curve.append(cum)
    merged.cumulative_pnl = cum
    return merged


def run_backtest(pair_data: Dict[str, pd.DataFrame], models: Dict) -> Dict:
    """Run the full backtest across all pairs.

    Each pair gets its own independent simulators to avoid cross-pair contamination.
    Results are aggregated at the end.

    Returns dict with all results.
    """
    logger.info("=" * 70)
    logger.info("STARTING BACKTEST")
    logger.info("=" * 70)

    # Precompute features for all pairs
    feature_matrices = precompute_features(pair_data)

    # Trackers
    accuracy_tracker = ModelAccuracyTracker()
    new_sims: List[PipelineSimulator] = []
    old_sims: List[PipelineSimulator] = []
    per_pair_results: Dict[str, Dict] = {}

    # Per-pair results for CSV output
    raw_results = []

    total_candles = 0
    total_inferences = 0
    start_time = time.time()

    sorted_pairs = sorted(feature_matrices.keys())
    total_bars_all = sum(
        max(0, len(pair_data[p]) - LOOKAHEAD - max(WARMUP, SEQ_LEN))
        for p in sorted_pairs
    )
    global_bars_done = 0

    for pair_idx, pair in enumerate(sorted_pairs):
        feat_matrix = feature_matrices[pair]
        df = pair_data[pair]
        close_prices = df["close"].values
        n_bars = len(df)

        # Each pair gets its own simulators
        new_sim = PipelineSimulator(f"FIXED-{pair}", exit_bars=NEW_EXIT_BARS)
        old_sim = PipelineSimulator(f"OLD-{pair}", exit_bars=OLD_EXIT_BARS)

        # Determine walkable range
        start_idx = max(WARMUP, SEQ_LEN)
        end_idx = n_bars - LOOKAHEAD  # leave room for forward return

        if start_idx >= end_idx:
            logger.warning(f"{pair}: not enough bars after warmup/lookahead, skipping")
            continue

        n_steps = end_idx - start_idx
        logger.info(f"\n--- {pair}: walking {n_steps} candles [{start_idx}..{end_idx-1}] ---")
        pair_start = time.time()
        pair_inferences = 0

        for idx in range(start_idx, end_idx):
            # Build feature window
            features = get_windowed_features(feat_matrix, idx, SEQ_LEN)
            if features is None:
                continue

            # Run inference
            try:
                with torch.no_grad():
                    predictions, confidences = predict_with_models(models, features)
            except Exception as e:
                if pair_inferences == 0:
                    logger.warning(f"  Inference error at idx {idx}: {e}")
                continue

            pair_inferences += 1
            total_inferences += 1

            # Forward return (6-bar lookahead)
            future_close = close_prices[idx + LOOKAHEAD]
            current_close = close_prices[idx]
            forward_return = (future_close - current_close) / current_close

            # Track model accuracy
            accuracy_tracker.record(predictions, forward_return)

            # FIXED pipeline decision
            new_action, new_conf, new_ml_signal, new_pos_usd = new_decision(predictions)
            new_sim.step(idx, current_close, new_action, new_conf, new_ml_signal, new_pos_usd)

            # OLD pipeline decision
            old_action, old_conf, old_ml_signal, old_pos_usd = old_decision(predictions)
            old_sim.step(idx, current_close, old_action, old_conf, old_ml_signal, old_pos_usd)

            # Save raw results
            ensemble_pred = predictions.get("meta_ensemble", 0.0)
            raw_results.append({
                "pair": pair,
                "bar_idx": idx,
                "timestamp": df["timestamp"].iloc[idx] if "timestamp" in df.columns else idx,
                "close": current_close,
                "forward_return_6bar": forward_return,
                "ensemble_pred": ensemble_pred,
                "agreement": compute_agreement(predictions),
                "new_action": new_action,
                "new_confidence": new_conf,
                "new_ml_signal": new_ml_signal,
                "new_pos_usd": new_pos_usd,
                "old_action": old_action,
                "old_confidence": old_conf,
                "old_ml_signal": old_ml_signal,
                "old_pos_usd": old_pos_usd,
                **{f"pred_{k}": v for k, v in predictions.items()},
            })

            total_candles += 1
            global_bars_done += 1

            # Progress logging
            if pair_inferences % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - pair_start
                rate = pair_inferences / elapsed if elapsed > 0 else 0
                if JSON_PROGRESS:
                    pct = (global_bars_done / total_bars_all * 100) if total_bars_all > 0 else 0
                    print(_json.dumps({
                        "type": "progress",
                        "pair": pair,
                        "pair_idx": pair_idx + 1,
                        "total_pairs": len(sorted_pairs),
                        "bars_done": global_bars_done,
                        "bars_total": total_bars_all,
                        "pct": round(pct, 1),
                        "new_pnl": round(new_sim.cumulative_pnl, 2),
                        "old_pnl": round(old_sim.cumulative_pnl, 2),
                    }), flush=True)
                else:
                    logger.info(
                        f"  {pair} [{pair_inferences}/{n_steps}] "
                        f"{rate:.0f} candles/sec, "
                        f"new_pnl=${new_sim.cumulative_pnl:.2f}, "
                        f"old_pnl=${old_sim.cumulative_pnl:.2f}"
                    )

        elapsed = time.time() - pair_start
        rate_str = f"{pair_inferences/elapsed:.0f}/sec" if elapsed > 0 else "N/A"
        logger.info(
            f"  {pair}: {pair_inferences} inferences in {elapsed:.1f}s ({rate_str})"
        )

        # Store per-pair results
        new_sims.append(new_sim)
        old_sims.append(old_sim)
        per_pair_results[pair] = {
            "new": new_sim.report(),
            "old": old_sim.report(),
        }

    # Aggregate all per-pair simulators
    new_merged = _merge_simulators(new_sims, "FIXED", NEW_EXIT_BARS)
    old_merged = _merge_simulators(old_sims, "OLD", OLD_EXIT_BARS)

    total_elapsed = time.time() - start_time
    logger.info(f"\nTotal: {total_inferences} inferences across {len(feature_matrices)} pairs "
                f"in {total_elapsed:.1f}s ({total_inferences/total_elapsed:.0f}/sec)")

    return {
        "accuracy_tracker": accuracy_tracker,
        "new_sim": new_merged,
        "old_sim": old_merged,
        "per_pair_results": per_pair_results,
        "raw_results": raw_results,
        "total_candles": total_candles,
        "total_inferences": total_inferences,
        "elapsed": total_elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results: Dict) -> None:
    """Print the full comparison report."""
    accuracy_tracker: ModelAccuracyTracker = results["accuracy_tracker"]
    new_sim: PipelineSimulator = results["new_sim"]
    old_sim: PipelineSimulator = results["old_sim"]

    print("\n")
    print("=" * 80)
    print("  ML PIPELINE BACKTEST REPORT")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total candles processed: {results['total_candles']:,}")
    print(f"  Total inferences: {results['total_inferences']:,}")
    print(f"  Runtime: {results['elapsed']:.1f}s")
    print("=" * 80)

    # ── Section 1: Per-Model Directional Accuracy ──
    print("\n" + "-" * 80)
    print("  1. PER-MODEL DIRECTIONAL ACCURACY (6-bar lookahead)")
    print("-" * 80)
    print(f"  {'Model':<25} {'Accuracy':>8} {'N_nonzero':>10} {'|Pred| mean':>12} {'Pred std':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*12} {'-'*10}")

    accuracies = accuracy_tracker.compute_accuracy()
    for name in sorted(accuracies.keys()):
        acc = accuracies[name]
        print(f"  {name:<25} {acc['accuracy']:>7.1f}% {acc['n_nonzero']:>10,} "
              f"{acc['abs_mean']:>12.5f} {acc['std_pred']:>10.5f}")

    # ── Section 2: Model Diversity (Pairwise Correlation) ──
    print("\n" + "-" * 80)
    print("  2. MODEL DIVERSITY (Pairwise prediction correlation)")
    print("-" * 80)

    corr_matrix = accuracy_tracker.compute_correlation_matrix()
    if not corr_matrix.empty:
        # Shorten names for display
        short_names = {
            "quantum_transformer": "QT",
            "bidirectional_lstm": "BiLSTM",
            "dilated_cnn": "DilCNN",
            "cnn": "CNN",
            "gru": "GRU",
            "lightgbm": "LGB",
            "meta_ensemble": "MetaEns",
        }
        display_names = [short_names.get(n, n[:8]) for n in corr_matrix.columns]

        # Header
        header = "  " + " " * 10
        for dn in display_names:
            header += f"{dn:>8}"
        print(header)

        for i, (full_name, row) in enumerate(corr_matrix.iterrows()):
            line = f"  {display_names[i]:<10}"
            for val in row.values:
                line += f"{val:>8.3f}"
            print(line)

        # Average off-diagonal correlation
        n = len(corr_matrix)
        if n > 1:
            off_diag = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        off_diag.append(corr_matrix.iloc[i, j])
            avg_corr = np.mean(off_diag)
            print(f"\n  Average off-diagonal correlation: {avg_corr:.3f}")
            print(f"  (Lower = more diverse ensemble, <0.5 is good, <0.3 is excellent)")
    else:
        print("  Insufficient models for correlation analysis.")

    # ── Section 3: Trading Simulation Comparison ──
    print("\n" + "-" * 80)
    print("  3. TRADING SIMULATION: FIXED vs OLD PIPELINE")
    print("-" * 80)

    new_report = new_sim.report()
    old_report = old_sim.report()

    metrics = [
        ("Total Signals (BUY)", new_report["signals"].get("BUY", 0), old_report["signals"].get("BUY", 0)),
        ("Total Signals (SELL)", new_report["signals"].get("SELL", 0), old_report["signals"].get("SELL", 0)),
        ("Total Signals (HOLD)", new_report["signals"].get("HOLD", 0), old_report["signals"].get("HOLD", 0)),
        ("Signal Rate (%)",
         (new_report["signals"].get("BUY", 0) + new_report["signals"].get("SELL", 0)) / max(sum(new_report["signals"].values()), 1) * 100,
         (old_report["signals"].get("BUY", 0) + old_report["signals"].get("SELL", 0)) / max(sum(old_report["signals"].values()), 1) * 100),
        ("Closed Trades", new_report["n_trades"], old_report["n_trades"]),
        ("Win Rate (%)", new_report["win_rate"], old_report["win_rate"]),
        ("Total P&L ($)", new_report["total_pnl"], old_report["total_pnl"]),
        ("Avg P&L/Trade ($)", new_report["avg_pnl"], old_report["avg_pnl"]),
        ("Sharpe Ratio (ann.)", new_report["sharpe"], old_report["sharpe"]),
        ("Max Drawdown ($)", new_report["max_drawdown"], old_report["max_drawdown"]),
        ("Avg Position ($)", new_report["avg_position_usd"], old_report["avg_position_usd"]),
    ]

    print(f"  {'Metric':<25} {'FIXED':>15} {'OLD':>15} {'Delta':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15}")

    for label, new_val, old_val in metrics:
        delta = new_val - old_val
        if isinstance(new_val, int):
            print(f"  {label:<25} {new_val:>15,} {old_val:>15,} {delta:>+15,}")
        else:
            print(f"  {label:<25} {new_val:>15.2f} {old_val:>15.2f} {delta:>+15.2f}")

    # ── Section 3b: Per-Pair Breakdown ──
    per_pair = results.get("per_pair_results", {})
    if per_pair:
        print(f"\n  Per-Pair Breakdown:")
        print(f"  {'Pair':<12} {'FIXED P&L':>12} {'OLD P&L':>12} {'Delta':>12} {'FIXED WR':>10} {'OLD WR':>10} {'FIXED #':>8} {'OLD #':>8}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
        for pair in sorted(per_pair.keys()):
            nr = per_pair[pair]["new"]
            odr = per_pair[pair]["old"]
            delta = nr["total_pnl"] - odr["total_pnl"]
            print(f"  {pair:<12} ${nr['total_pnl']:>10.2f} ${odr['total_pnl']:>10.2f} ${delta:>+10.2f} "
                  f"{nr['win_rate']:>9.1f}% {odr['win_rate']:>9.1f}% {nr['n_trades']:>8,} {odr['n_trades']:>8,}")

    # ── Section 4: Improvement Summary ──
    print("\n" + "-" * 80)
    print("  4. IMPROVEMENT SUMMARY")
    print("-" * 80)

    pnl_diff = new_report["total_pnl"] - old_report["total_pnl"]
    wr_diff = new_report["win_rate"] - old_report["win_rate"]
    sharpe_diff = new_report["sharpe"] - old_report["sharpe"]
    signal_pct_new = (new_report["signals"].get("BUY", 0) + new_report["signals"].get("SELL", 0)) / max(sum(new_report["signals"].values()), 1) * 100
    signal_pct_old = (old_report["signals"].get("BUY", 0) + old_report["signals"].get("SELL", 0)) / max(sum(old_report["signals"].values()), 1) * 100

    print(f"  P&L improvement:    ${pnl_diff:>+.2f} (FIXED: ${new_report['total_pnl']:.2f} vs OLD: ${old_report['total_pnl']:.2f})")
    print(f"  Win rate change:    {wr_diff:>+.1f}pp (FIXED: {new_report['win_rate']:.1f}% vs OLD: {old_report['win_rate']:.1f}%)")
    print(f"  Sharpe change:      {sharpe_diff:>+.2f} (FIXED: {new_report['sharpe']:.2f} vs OLD: {old_report['sharpe']:.2f})")
    print(f"  Signal rate change: {signal_pct_new - signal_pct_old:>+.1f}pp (FIXED: {signal_pct_new:.1f}% vs OLD: {signal_pct_old:.1f}%)")
    print(f"  Trade count change: {new_report['n_trades'] - old_report['n_trades']:>+,} trades")

    if pnl_diff > 0:
        print(f"\n  VERDICT: FIXED pipeline has HIGHER net P&L by ${pnl_diff:.2f}")
    elif pnl_diff < 0:
        print(f"\n  VERDICT: FIXED pipeline has LOWER net P&L by ${abs(pnl_diff):.2f}")
    else:
        print(f"\n  VERDICT: No P&L difference between pipelines")

    # Note on cost
    print(f"\n  Note: Cost assumption = {COST_BPS*100:.2f}% round-trip (spread + slippage)")
    print(f"  FIXED exits after {NEW_EXIT_BARS} bars (30 min), OLD exits after {OLD_EXIT_BARS} bar (5 min)")

    print("\n" + "=" * 80)


def save_results_csv(results: Dict) -> str:
    """Save raw results to CSV."""
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(PROJECT_ROOT, "data", f"backtest_results_{date_str}.csv")

    df = pd.DataFrame(results["raw_results"])
    df.to_csv(csv_path, index=False)
    logger.info(f"Raw results saved to {csv_path} ({len(df)} rows)")
    return csv_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # When in JSON progress mode, redirect logging to stderr so stdout is clean JSON
    if JSON_PROGRESS:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stderr,
        )

    logger.info("ML Pipeline Backtest — Renaissance Trading Bot")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Training data: {TRAINING_DIR}")
    logger.info(f"Sequence length: {SEQ_LEN}, Warmup: {WARMUP}, Lookahead: {LOOKAHEAD}")

    # 1. Load data
    logger.info("\n--- Loading pair data ---")
    pair_data = load_all_pairs()
    if not pair_data:
        logger.error("No pair data loaded. Exiting.")
        sys.exit(1)
    logger.info(f"Loaded {len(pair_data)} pairs: {list(pair_data.keys())}")

    # 2. Load models
    logger.info("\n--- Loading trained models ---")
    models = load_trained_models(base_dir=PROJECT_ROOT)
    if not models:
        logger.error("No models loaded. Exiting.")
        sys.exit(1)
    logger.info(f"Loaded {len(models)} models: {list(models.keys())}")

    # Ensure all PyTorch models are in eval mode
    for name, model in models.items():
        if isinstance(model, torch.nn.Module):
            model.eval()

    # 3. Run backtest
    results = run_backtest(pair_data, models)

    # 4. Print report (skip in JSON mode — summary goes as JSON)
    if not JSON_PROGRESS:
        print_report(results)

    # 5. Save CSV
    csv_path = save_results_csv(results)

    # 6. Emit final JSON or plain text
    if JSON_PROGRESS:
        new_report = results["new_sim"].report()
        old_report = results["old_sim"].report()
        accuracy = results["accuracy_tracker"].compute_accuracy()
        # Sanitize accuracy: ensure all values are JSON-serializable
        clean_accuracy = {}
        for name, acc in accuracy.items():
            clean_accuracy[name] = {
                k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                for k, v in acc.items()
            }
        per_pair_clean = {}
        for pair, pr in results.get("per_pair_results", {}).items():
            per_pair_clean[pair] = {
                "new": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in pr["new"].items()},
                "old": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in pr["old"].items()},
            }
        print(_json.dumps({
            "type": "complete",
            "csv_path": csv_path,
            "summary": {
                "new": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in new_report.items()},
                "old": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in old_report.items()},
                "per_pair": per_pair_clean,
                "accuracy": clean_accuracy,
                "total_candles": results["total_candles"],
                "total_inferences": results["total_inferences"],
                "elapsed": round(results["elapsed"], 1),
            },
        }), flush=True)
    else:
        print(f"\nRaw results saved to: {csv_path}")


if __name__ == "__main__":
    main()
