"""
Weekly retraining for all 7 ML models with safety gates.

Trains in the correct order:
  Phase 1: 5 base models (QT, BiLSTM, DilatedCNN, CNN, GRU)
  Phase 2: Meta-Ensemble (stacking layer, depends on 5 base models)
  Phase 3: VAE anomaly detector

Each model has its own safety gate: deploy only if new_accuracy >= old_accuracy - 1%.
Date-stamped backups are kept for 4 weeks.

Usage:
    python -m scripts.training.retrain_weekly           # interactive
    python -m scripts.training.retrain_weekly --auto     # auto-deploy if better
    python -m scripts.training.retrain_weekly --days 30  # custom data window

Exit codes:
    0 = all models retrained successfully
    1 = training failed for one or more models
    2 = new model worse than old (old kept)
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_model_loader import (
    BASE_MODEL_NAMES,
    TrainedBidirectionalLSTM,
    TrainedCNN,
    TrainedDilatedCNN,
    TrainedGRU,
    TrainedMetaEnsemble,
    TrainedQuantumTransformer,
)
from scripts.training.fetch_training_data import download_all, DEFAULT_PAIRS
from scripts.training.training_utils import (
    DirectionalLoss,
    directional_accuracy,
    evaluate_on_dataset,
    generate_sequences,
    load_training_csvs,
    walk_forward_split,
)
from scripts.training.train_quantum_transformer import train_quantum_transformer
from scripts.training.train_bidirectional_lstm import train_bidirectional_lstm
from scripts.training.train_dilated_cnn import train_dilated_cnn
from scripts.training.train_cnn import train_cnn
from scripts.training.train_gru import train_gru
from scripts.training.train_meta_ensemble import train_meta_ensemble

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "trained")
METADATA_PATH = os.path.join(MODELS_DIR, "training_metadata.json")
SAFETY_TOLERANCE = 0.01  # Deploy if new_acc >= old_acc - tolerance
KEEP_WEEKS = 4

# Base models: evaluated on sequence data (X_test, y_test)
BASE_MODEL_CONFIGS = {
    "quantum_transformer": {
        "class": TrainedQuantumTransformer,
        "file": "best_quantum_transformer_model.pth",
        "train_fn": train_quantum_transformer,
    },
    "bidirectional_lstm": {
        "class": TrainedBidirectionalLSTM,
        "file": "best_bidirectional_lstm_model.pth",
        "train_fn": train_bidirectional_lstm,
    },
    "dilated_cnn": {
        "class": TrainedDilatedCNN,
        "file": "best_dilated_cnn_model.pth",
        "train_fn": train_dilated_cnn,
    },
    "cnn": {
        "class": TrainedCNN,
        "file": "best_cnn_model.pth",
        "train_fn": train_cnn,
    },
    "gru": {
        "class": TrainedGRU,
        "file": "best_gru_model.pth",
        "train_fn": train_gru,
    },
}

# Meta-ensemble config (separate because it has different eval logic)
META_ENSEMBLE_CONFIG = {
    "class": TrainedMetaEnsemble,
    "file": "best_meta_ensemble_model.pth",
    "train_fn": train_meta_ensemble,
}

# VAE config
VAE_CONFIG = {
    "file": "vae_anomaly_detector.pth",
}


def load_metadata() -> dict:
    """Load training metadata JSON."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {}


def save_metadata(metadata: dict) -> None:
    """Save training metadata JSON."""
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Updated {METADATA_PATH}")


def evaluate_existing_base_model(
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> Optional[float]:
    """Load a deployed base model and evaluate on test set."""
    config = BASE_MODEL_CONFIGS[model_name]
    model_path = os.path.join(MODELS_DIR, config["file"])

    if not os.path.exists(model_path):
        logger.info(f"No existing {model_name} model at {model_path}")
        return None

    try:
        model = config["class"]()
        sd = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=False)
        model.to(device).eval()
        _, dir_acc = evaluate_on_dataset(model, X_test, y_test, device)
        logger.info(f"Existing {model_name}: test dir_acc = {dir_acc:.4f}")
        return dir_acc
    except Exception as e:
        logger.warning(f"Could not evaluate existing {model_name}: {e}")
        return None


def evaluate_existing_meta_ensemble(
    X_test_seq: np.ndarray,
    y_test: np.ndarray,
    base_models: Dict[str, nn.Module],
    device: torch.device,
) -> Optional[float]:
    """Load the deployed meta-ensemble and evaluate on test set."""
    model_path = os.path.join(MODELS_DIR, META_ENSEMBLE_CONFIG["file"])

    if not os.path.exists(model_path):
        logger.info(f"No existing meta_ensemble at {model_path}")
        return None

    try:
        model = META_ENSEMBLE_CONFIG["class"]()
        sd = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=False)
        model.to(device).eval()

        # Generate meta-inputs from base model predictions
        from scripts.training.train_meta_ensemble import generate_meta_inputs
        meta_X, meta_y = generate_meta_inputs(base_models, X_test_seq, y_test, device)

        # Evaluate
        criterion = DirectionalLoss(0.3)
        ds = TensorDataset(torch.FloatTensor(meta_X), torch.FloatTensor(meta_y))
        loader = DataLoader(ds, batch_size=128, shuffle=False)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                pred, _ = model(X_b.to(device))
                all_preds.append(pred.squeeze(-1).cpu().numpy())
                all_targets.append(y_b.numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        dir_acc = directional_accuracy(preds, targets)
        logger.info(f"Existing meta_ensemble: test dir_acc = {dir_acc:.4f}")
        return dir_acc
    except Exception as e:
        logger.warning(f"Could not evaluate existing meta_ensemble: {e}")
        return None


def backup_model(file_name: str) -> Optional[str]:
    """Create a date-stamped backup."""
    model_path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(model_path):
        return None

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    backup_name = file_name.replace(".pth", f"_{date_str}.pth")
    backup_path = os.path.join(MODELS_DIR, backup_name)
    shutil.copy2(model_path, backup_path)
    logger.info(f"Backed up {file_name} → {backup_name}")
    return backup_path


def restore_from_backup(file_name: str) -> bool:
    """Restore model from today's backup."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    backup_name = file_name.replace(".pth", f"_{date_str}.pth")
    backup_path = os.path.join(MODELS_DIR, backup_name)
    model_path = os.path.join(MODELS_DIR, file_name)

    if os.path.exists(backup_path):
        shutil.copy2(backup_path, model_path)
        logger.info(f"Restored {file_name} from backup")
        return True
    return False


def prune_old_backups(file_name: str, keep: int = KEEP_WEEKS) -> None:
    """Remove old date-stamped backups, keeping the most recent N."""
    base_name = file_name.replace(".pth", "")
    pattern = os.path.join(MODELS_DIR, f"{base_name}_????-??-??.pth")
    backups = sorted(glob.glob(pattern))
    for path in backups[:-keep] if len(backups) > keep else []:
        os.remove(path)
        logger.info(f"Pruned old backup: {os.path.basename(path)}")


def _apply_safety_gate(
    model_name: str,
    file_name: str,
    old_accuracy: Optional[float],
    new_accuracy: float,
    model_result: dict,
) -> int:
    """Apply safety gate logic. Returns exit_code contribution (0 or 2)."""
    if old_accuracy is not None:
        threshold = old_accuracy - SAFETY_TOLERANCE
        if new_accuracy >= threshold:
            improvement = new_accuracy - old_accuracy
            logger.info(
                f"PASSED safety gate: {new_accuracy:.4f} >= {threshold:.4f} "
                f"(improvement: {improvement:+.4f})"
            )
            model_result["deployed"] = True
            model_result["status"] = "deployed"
            return 0
        else:
            logger.warning(
                f"FAILED safety gate: {new_accuracy:.4f} < {threshold:.4f}. "
                f"Keeping old model."
            )
            restore_from_backup(file_name)
            model_result["deployed"] = False
            model_result["status"] = "kept_old"
            return 2
    else:
        logger.info("No old model to compare — deploying new model")
        model_result["deployed"] = True
        model_result["status"] = "deployed"
        return 0


def retrain_weekly(
    days: int = 30,
    epochs: int = 100,
    auto_deploy: bool = False,
    pairs: list = None,
) -> Tuple[dict, int]:
    """Run weekly retraining for all 7 models with safety gates.

    Returns:
        (results_dict, exit_code)
    """
    if pairs is None:
        pairs = DEFAULT_PAIRS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    metadata = load_metadata()
    results = {}
    exit_code = 0

    # ── Download data ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("WEEKLY RETRAINING — Downloading latest data")
    logger.info("=" * 60)
    pair_dfs = download_all(pairs, days)
    if not pair_dfs:
        logger.error("No data downloaded. Aborting.")
        return {}, 1

    # Prepare test set for safety gate evaluation
    logger.info("Preparing walk-forward split for evaluation...")
    _, _, test_dfs = walk_forward_split(pair_dfs)
    X_test_seq, y_test = generate_sequences(test_dfs, stride=1)

    if len(X_test_seq) == 0:
        logger.error("No test sequences generated. Aborting.")
        return {}, 1

    data_window_start = datetime.fromtimestamp(
        min(df["timestamp"].min() for df in pair_dfs.values()), tz=timezone.utc
    ).strftime("%Y-%m-%d")
    data_window_end = datetime.fromtimestamp(
        max(df["timestamp"].max() for df in pair_dfs.values()), tz=timezone.utc
    ).strftime("%Y-%m-%d")

    # ── Phase 1: Retrain 5 base models ────────────────────────────────────
    for model_name, config in BASE_MODEL_CONFIGS.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Retraining base model: {model_name}")
        logger.info(f"{'=' * 60}")

        model_result = {"model": model_name}

        # Evaluate old model
        old_accuracy = evaluate_existing_base_model(
            model_name, X_test_seq, y_test, device
        )
        model_result["old_accuracy"] = old_accuracy

        # Back up
        backup_model(config["file"])

        # Train new model
        try:
            train_result = config["train_fn"](epochs=epochs)
            model_result.update(train_result)
            new_accuracy = train_result.get("test_dir_acc", 0.0)
            model_result["new_accuracy"] = new_accuracy
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}")
            restore_from_backup(config["file"])
            model_result["status"] = "failed"
            model_result["error"] = str(e)
            results[model_name] = model_result
            exit_code = max(exit_code, 1)
            continue

        # Safety gate
        gate_code = _apply_safety_gate(
            model_name, config["file"], old_accuracy, new_accuracy, model_result
        )
        exit_code = max(exit_code, gate_code)

        # Update metadata
        metadata[model_name] = {
            "last_trained": datetime.now(timezone.utc).isoformat(),
            "directional_accuracy": float(model_result.get("new_accuracy", 0)),
            "validation_loss": float(model_result.get("val_loss", 0)),
            "data_window": f"{data_window_start} to {data_window_end}",
            "model_file": config["file"],
            "deployed": model_result.get("deployed", False),
        }

        prune_old_backups(config["file"])
        results[model_name] = model_result

    # ── Phase 2: Retrain Meta-Ensemble ────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info("Retraining: meta_ensemble (stacking layer)")
    logger.info(f"{'=' * 60}")

    base_ok = sum(
        1 for name in BASE_MODEL_CONFIGS
        if results.get(name, {}).get("status") in ("deployed", "kept_old")
    )

    if base_ok == len(BASE_MODEL_CONFIGS):
        model_result = {"model": "meta_ensemble"}

        # Load deployed base models for meta-ensemble evaluation
        deployed_base_models = {}
        for name, config in BASE_MODEL_CONFIGS.items():
            try:
                m = config["class"]()
                sd = torch.load(
                    os.path.join(MODELS_DIR, config["file"]),
                    map_location="cpu", weights_only=False
                )
                if isinstance(sd, dict) and "model_state_dict" in sd:
                    sd = sd["model_state_dict"]
                m.load_state_dict(sd, strict=False)
                m.to(device).eval()
                deployed_base_models[name] = m
            except Exception:
                pass

        # Evaluate old meta-ensemble
        old_accuracy = evaluate_existing_meta_ensemble(
            X_test_seq, y_test, deployed_base_models, device
        )
        model_result["old_accuracy"] = old_accuracy

        # Back up
        backup_model(META_ENSEMBLE_CONFIG["file"])

        # Train
        try:
            train_result = train_meta_ensemble(epochs=epochs)
            model_result.update(train_result)
            new_accuracy = train_result.get("test_dir_acc", 0.0)
            model_result["new_accuracy"] = new_accuracy
        except Exception as e:
            logger.error(f"Meta-ensemble training failed: {e}")
            restore_from_backup(META_ENSEMBLE_CONFIG["file"])
            model_result["status"] = "failed"
            model_result["error"] = str(e)
            results["meta_ensemble"] = model_result
            exit_code = max(exit_code, 1)
        else:
            gate_code = _apply_safety_gate(
                "meta_ensemble", META_ENSEMBLE_CONFIG["file"],
                old_accuracy, new_accuracy, model_result
            )
            exit_code = max(exit_code, gate_code)

            metadata["meta_ensemble"] = {
                "last_trained": datetime.now(timezone.utc).isoformat(),
                "directional_accuracy": float(new_accuracy),
                "validation_loss": float(model_result.get("val_loss", 0)),
                "data_window": f"{data_window_start} to {data_window_end}",
                "model_file": META_ENSEMBLE_CONFIG["file"],
                "deployed": model_result.get("deployed", False),
            }
            prune_old_backups(META_ENSEMBLE_CONFIG["file"])
            results["meta_ensemble"] = model_result
    else:
        failed = [
            n for n in BASE_MODEL_CONFIGS
            if results.get(n, {}).get("status") not in ("deployed", "kept_old")
        ]
        logger.warning(f"Skipping meta-ensemble: base models not ready: {failed}")
        results["meta_ensemble"] = {"status": "skipped", "error": f"Base models failed: {failed}"}

    # ── Phase 3: Retrain VAE ──────────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info("Retraining: VAE Anomaly Detector")
    logger.info(f"{'=' * 60}")

    backup_model(VAE_CONFIG["file"])

    try:
        from train_vae import generate_training_samples, train_vae as _train_vae_fn
        import torch as _torch

        vae_pair_dfs = load_training_csvs()
        if not vae_pair_dfs:
            raise RuntimeError("No data for VAE")

        samples = generate_training_samples(vae_pair_dfs)
        if samples.size == 0:
            raise RuntimeError("No VAE samples generated")

        model = _train_vae_fn(samples, epochs=epochs)
        output_path = os.path.join(MODELS_DIR, VAE_CONFIG["file"])
        _torch.save(model.state_dict(), output_path)

        results["vae"] = {
            "status": "deployed",
            "deployed": True,
            "train_samples": len(samples),
            "last_trained": datetime.now(timezone.utc).isoformat(),
        }
        metadata["vae"] = {
            "last_trained": datetime.now(timezone.utc).isoformat(),
            "train_samples": len(samples),
            "data_window": f"{data_window_start} to {data_window_end}",
            "model_file": VAE_CONFIG["file"],
            "deployed": True,
        }
        logger.info(f"VAE trained on {len(samples)} samples")
    except Exception as e:
        logger.error(f"VAE training failed: {e}")
        restore_from_backup(VAE_CONFIG["file"])
        results["vae"] = {"status": "failed", "error": str(e)}
        exit_code = max(exit_code, 1)

    prune_old_backups(VAE_CONFIG["file"])

    # ── Save metadata & report ────────────────────────────────────────────
    save_metadata(metadata)
    elapsed = time.time() - start_time

    logger.info(f"\n{'=' * 60}")
    logger.info("WEEKLY RETRAINING REPORT — ALL 7 MODELS")
    logger.info(f"{'=' * 60}")

    all_model_names = list(BASE_MODEL_CONFIGS.keys()) + ["meta_ensemble", "vae"]
    print(f"\n{'Model':<25} {'Status':<12} {'Old Acc':>10} {'New Acc':>10} {'Deployed':>10}")
    print("-" * 70)
    for model_name in all_model_names:
        r = results.get(model_name, {})
        status = r.get("status", "unknown")
        old_acc = f"{r['old_accuracy']:.4f}" if r.get("old_accuracy") is not None else "N/A"
        new_acc = f"{r['new_accuracy']:.4f}" if "new_accuracy" in r else "N/A"
        deployed = "Yes" if r.get("deployed") else "No"
        print(f"{model_name:<25} {status:<12} {old_acc:>10} {new_acc:>10} {deployed:>10}")

    n_deployed = sum(1 for r in results.values() if r.get("deployed"))
    print(f"\nData window: {data_window_start} to {data_window_end}")
    print(f"Models deployed: {n_deployed}/{len(results)}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Exit code: {exit_code}")

    return results, exit_code


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Weekly retraining for all 7 models")
    parser.add_argument("--auto", action="store_true", help="Auto-deploy without prompting")
    parser.add_argument("--days", type=int, default=30, help="Days of training data")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per model")
    parser.add_argument("--pairs", nargs="+", default=None)
    args = parser.parse_args()

    _, exit_code = retrain_weekly(
        days=args.days,
        epochs=args.epochs,
        auto_deploy=args.auto,
        pairs=args.pairs,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
