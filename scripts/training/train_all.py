"""
Orchestrator: download data and train all 7 ML models in the correct order.

Training order:
  Phase 1 — 5 base models (independent, could be parallelized):
    1. Quantum Transformer
    2. Bidirectional LSTM
    3. Dilated CNN
    4. Simple CNN
    5. Bidirectional GRU
  Phase 2 — Meta-Ensemble (depends on all 5 base models):
    6. Meta-Ensemble stacking layer
  Phase 3 — VAE anomaly detector (independent but runs last):
    7. VAE

Usage:
    python -m scripts.training.train_all
    python -m scripts.training.train_all --days 30 --epochs 100
    python -m scripts.training.train_all --epochs 5  # quick smoke test
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.training.fetch_training_data import download_all, DEFAULT_PAIRS
from scripts.training.train_quantum_transformer import train_quantum_transformer
from scripts.training.train_bidirectional_lstm import train_bidirectional_lstm
from scripts.training.train_dilated_cnn import train_dilated_cnn
from scripts.training.train_cnn import train_cnn
from scripts.training.train_gru import train_gru
from scripts.training.train_meta_ensemble import train_meta_ensemble

logger = logging.getLogger(__name__)


def _train_one(name: str, train_fn, epochs: int, results: dict) -> None:
    """Train a single model and record results."""
    try:
        results[name] = train_fn(epochs=epochs)
        results[name]["status"] = "success"
    except Exception as e:
        logger.error(f"{name} training failed: {e}")
        results[name] = {"status": "failed", "error": str(e)}


def _train_vae(epochs: int, results: dict) -> None:
    """Train the VAE using the CSV training data pipeline."""
    try:
        from scripts.training.training_utils import load_training_csvs

        pair_dfs = load_training_csvs()
        if not pair_dfs:
            raise RuntimeError("No training data for VAE")

        # Import VAE training function from existing train_vae.py
        sys.path.insert(0, PROJECT_ROOT)
        from train_vae import generate_training_samples, train_vae as _train_vae_fn
        import torch

        samples = generate_training_samples(pair_dfs)
        if samples.size == 0:
            raise RuntimeError("No VAE training samples generated")

        model = _train_vae_fn(samples, epochs=epochs)

        output_path = os.path.join(PROJECT_ROOT, "models", "trained", "vae_anomaly_detector.pth")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(model.state_dict(), output_path)

        results["vae"] = {
            "status": "success",
            "model": "vae",
            "last_trained": datetime.now(timezone.utc).isoformat(),
            "train_samples": len(samples),
        }
        logger.info(f"VAE trained on {len(samples)} samples, saved to {output_path}")

    except Exception as e:
        logger.error(f"VAE training failed: {e}")
        results["vae"] = {"status": "failed", "error": str(e)}


def train_all(days: int = 30, epochs: int = 100, pairs: list = None) -> dict:
    """Download data and train all 7 models in the correct order.

    Args:
        days: Days of historical data to download
        epochs: Max epochs per model
        pairs: Trading pairs (default: 6 major pairs)

    Returns:
        Dict with per-model results
    """
    if pairs is None:
        pairs = DEFAULT_PAIRS

    start_time = time.time()
    results = {}

    # ── Step 1: Download data ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading training data")
    logger.info("=" * 60)
    pair_dfs = download_all(pairs, days)
    if not pair_dfs:
        raise RuntimeError("No training data downloaded. Check network connectivity.")

    total_bars = sum(len(df) for df in pair_dfs.values())
    logger.info(f"Downloaded {total_bars} total bars across {len(pair_dfs)} pairs\n")

    # ── Step 2: Train 5 base models ───────────────────────────────────────
    base_models = [
        ("quantum_transformer", train_quantum_transformer),
        ("bidirectional_lstm", train_bidirectional_lstm),
        ("dilated_cnn", train_dilated_cnn),
        ("cnn", train_cnn),
        ("gru", train_gru),
    ]

    for i, (name, train_fn) in enumerate(base_models, start=2):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"STEP {i}: Training {name}")
        logger.info(f"{'=' * 60}")
        _train_one(name, train_fn, epochs, results)

    # Check how many base models succeeded
    base_ok = sum(1 for n, _ in base_models if results.get(n, {}).get("status") == "success")
    logger.info(f"\nBase models trained: {base_ok}/{len(base_models)}")

    # ── Step 3: Train Meta-Ensemble (requires all 5 base models) ──────────
    logger.info(f"\n{'=' * 60}")
    logger.info("STEP 7: Training Meta-Ensemble (stacking layer)")
    logger.info(f"{'=' * 60}")
    if base_ok == len(base_models):
        _train_one("meta_ensemble", train_meta_ensemble, epochs, results)
    else:
        failed = [n for n, _ in base_models if results.get(n, {}).get("status") != "success"]
        logger.warning(
            f"Skipping meta-ensemble: {len(failed)} base model(s) failed: {failed}. "
            f"All 5 base models must succeed first."
        )
        results["meta_ensemble"] = {
            "status": "skipped",
            "error": f"Base models failed: {failed}",
        }

    # ── Step 4: Train VAE ─────────────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info("STEP 8: Training VAE Anomaly Detector")
    logger.info(f"{'=' * 60}")
    _train_vae(epochs, results)

    elapsed = time.time() - start_time

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info("TRAINING SUMMARY — ALL 7 MODELS")
    logger.info(f"{'=' * 60}")
    print(f"\n{'Model':<25} {'Status':<10} {'Val Loss':>10} {'Val Acc':>10} {'Test Acc':>10} {'Epochs':>8}")
    print("-" * 75)
    for model_name in [n for n, _ in base_models] + ["meta_ensemble", "vae"]:
        r = results.get(model_name, {})
        status = r.get("status", "unknown")
        val_loss = f"{r['val_loss']:.4f}" if "val_loss" in r else "N/A"
        val_acc = f"{r['val_dir_acc']:.3f}" if "val_dir_acc" in r else "N/A"
        test_acc = f"{r['test_dir_acc']:.3f}" if "test_dir_acc" in r else "N/A"
        ep = str(r.get("epochs_trained", "N/A"))
        print(f"{model_name:<25} {status:<10} {val_loss:>10} {val_acc:>10} {test_acc:>10} {ep:>8}")

    n_success = sum(1 for r in results.values() if r.get("status") == "success")
    print(f"\n{n_success}/{len(results)} models trained successfully")
    print(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train all 7 ML models")
    parser.add_argument("--days", type=int, default=30, help="Days of history to download")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per model")
    parser.add_argument("--pairs", nargs="+", default=None)
    args = parser.parse_args()

    train_all(days=args.days, epochs=args.epochs, pairs=args.pairs)


if __name__ == "__main__":
    main()
