"""
Train the Meta-Ensemble stacking layer.

This must run AFTER all 5 base models are trained. It:
1. Loads all 5 trained base models (QT, BiLSTM, DilatedCNN, CNN, GRU)
2. Runs inference on training data to collect their predictions
3. Builds meta-inputs: [83-dim features | 5 model predictions] = 88-dim
4. Trains the Meta-Ensemble to learn which models to trust in which conditions

The ensemble edge comes from diversity of errors: five models that individually
score 52-54% can combine to 56-58% if they make different kinds of mistakes.

Usage:
    python -m scripts.training.train_meta_ensemble
    python -m scripts.training.train_meta_ensemble --epochs 100 --lr 1e-3
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_model_loader import (
    BASE_MODEL_NAMES,
    INPUT_DIM,
    TrainedBidirectionalLSTM,
    TrainedCNN,
    TrainedDilatedCNN,
    TrainedGRU,
    TrainedMetaEnsemble,
    TrainedQuantumTransformer,
    _detect_input_dim,
)
from scripts.training.training_utils import (
    DirectionalLoss,
    directional_accuracy,
    generate_sequences,
    load_derivatives_csvs,
    load_training_csvs,
    save_model_checkpoint,
    walk_forward_split,
)

logger = logging.getLogger(__name__)

DEFAULTS = {
    "epochs": 80,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 12,
    "directional_weight": 0.4,
    "gradient_clip": 1.0,
}

SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "trained", "best_meta_ensemble_model.pth")

# Map base model names to their classes and weight files
BASE_MODEL_CONFIGS = {
    "quantum_transformer": (
        TrainedQuantumTransformer,
        "models/trained/best_quantum_transformer_model.pth",
    ),
    "bidirectional_lstm": (
        TrainedBidirectionalLSTM,
        "models/trained/best_bidirectional_lstm_model.pth",
    ),
    "dilated_cnn": (
        TrainedDilatedCNN,
        "models/trained/best_dilated_cnn_model.pth",
    ),
    "cnn": (
        TrainedCNN,
        "models/trained/best_cnn_model.pth",
    ),
    "gru": (
        TrainedGRU,
        "models/trained/best_gru_model.pth",
    ),
}


def load_base_models(device: torch.device) -> Dict[str, nn.Module]:
    """Load all 5 trained base models.

    Raises RuntimeError if any base model is missing.
    """
    models = {}
    missing = []

    for name in BASE_MODEL_NAMES:
        cls, weight_path = BASE_MODEL_CONFIGS[name]
        full_path = os.path.join(PROJECT_ROOT, weight_path)

        if not os.path.exists(full_path):
            missing.append(name)
            continue

        try:
            sd = torch.load(full_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            # Auto-detect input_dim from saved weights
            detected_dim = _detect_input_dim(name, sd)
            use_dim = detected_dim if detected_dim is not None else INPUT_DIM
            model = cls(input_dim=use_dim)
            model.load_state_dict(sd, strict=False)
            model.to(device).eval()
            models[name] = model
            logger.info(f"  Loaded base model: {name}")
        except Exception as e:
            logger.error(f"  Failed to load {name}: {e}")
            missing.append(name)

    if missing:
        raise RuntimeError(
            f"Missing base models: {missing}. "
            f"Train all 5 base models first before training the meta-ensemble."
        )

    return models


def generate_meta_inputs(
    base_models: Dict[str, nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run all base models on data and build meta-ensemble inputs.

    For each sample:
    - Extract last-timestep features (83-dim)
    - Get each base model's prediction (5 scalars)
    - Concatenate: [features | predictions] = 88-dim

    Args:
        base_models: Dict of model_name → trained model
        X: (N, seq_len, 83) feature sequences
        y: (N,) labels
        device: torch device
        batch_size: Batch size for inference

    Returns:
        (meta_X, meta_y) where meta_X is (N, 88) and meta_y is (N,)
    """
    n = len(X)
    n_models = len(BASE_MODEL_NAMES)
    all_preds = {name: np.zeros(n) for name in BASE_MODEL_NAMES}

    # Run each base model on all data
    for name in BASE_MODEL_NAMES:
        model = base_models[name]
        model_preds = []

        for i in range(0, n, batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
            with torch.no_grad():
                output = model(batch)
                if isinstance(output, tuple):
                    pred = output[0]
                else:
                    pred = output
                pred = torch.tanh(pred.squeeze(-1))
                model_preds.append(pred.cpu().numpy())

        all_preds[name] = np.concatenate(model_preds)

    # Log individual model accuracies
    for name in BASE_MODEL_NAMES:
        acc = directional_accuracy(all_preds[name], y)
        logger.info(f"  Base model {name}: dir_acc={acc:.3f}")

    # Build meta-inputs: [last-timestep features | 5 predictions]
    last_features = X[:, -1, :]  # (N, INPUT_DIM) — last timestep of each sequence
    pred_matrix = np.column_stack([all_preds[name] for name in BASE_MODEL_NAMES])  # (N, 5)
    meta_X = np.concatenate([last_features, pred_matrix], axis=1).astype(np.float32)  # (N, INPUT_DIM+5)

    logger.info(f"Meta-inputs: {meta_X.shape} ({INPUT_DIM} features + {n_models} predictions)")
    return meta_X, y


def train_meta_ensemble(
    epochs: int = DEFAULTS["epochs"],
    batch_size: int = DEFAULTS["batch_size"],
    lr: float = DEFAULTS["lr"],
    weight_decay: float = DEFAULTS["weight_decay"],
    patience: int = DEFAULTS["patience"],
    data_dir: str = "data/training",
    save_path: str = SAVE_PATH,
) -> dict:
    """Train the Meta-Ensemble stacking layer.

    Returns:
        Dict with training results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Training MetaEnsemble on {device}")

    # Load base models
    logger.info("Loading 5 base models...")
    base_models = load_base_models(device)

    # Load and split data
    pair_dfs = load_training_csvs(data_dir=data_dir)
    if not pair_dfs:
        raise RuntimeError(f"No training data found in {data_dir}.")

    # Load derivatives data (optional — zeros if missing)
    derivatives_dfs, fear_greed_df = load_derivatives_csvs(
        data_dir=os.path.join(data_dir, "derivatives")
    )

    logger.info("Splitting data (walk-forward)...")
    train_dfs, val_dfs, test_dfs = walk_forward_split(pair_dfs)

    logger.info("Generating sequences...")
    X_train_seq, y_train = generate_sequences(
        train_dfs, stride=1,
        derivatives_dfs=derivatives_dfs, fear_greed_df=fear_greed_df,
    )
    X_val_seq, y_val = generate_sequences(
        val_dfs, stride=1,
        derivatives_dfs=derivatives_dfs, fear_greed_df=fear_greed_df,
    )
    X_test_seq, y_test = generate_sequences(
        test_dfs, stride=1,
        derivatives_dfs=derivatives_dfs, fear_greed_df=fear_greed_df,
    )

    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
        raise RuntimeError("Not enough data.")

    # Generate meta-inputs (run base models on data)
    logger.info("Generating meta-inputs from base model predictions...")
    logger.info("Training set base model accuracies:")
    meta_X_train, meta_y_train = generate_meta_inputs(
        base_models, X_train_seq, y_train, device
    )
    logger.info("Validation set base model accuracies:")
    meta_X_val, meta_y_val = generate_meta_inputs(
        base_models, X_val_seq, y_val, device
    )
    logger.info("Test set base model accuracies:")
    meta_X_test, meta_y_test = generate_meta_inputs(
        base_models, X_test_seq, y_test, device
    )

    logger.info(f"Meta dataset sizes: train={len(meta_X_train)}, "
                f"val={len(meta_X_val)}, test={len(meta_X_test)}")

    # Create meta-ensemble model
    model = TrainedMetaEnsemble().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
    )
    criterion = DirectionalLoss(logit_scale=20.0, margin=0.10)

    # Create data loaders (meta-ensemble uses flat 88-dim input, not sequences)
    train_ds = TensorDataset(
        torch.FloatTensor(meta_X_train), torch.FloatTensor(meta_y_train)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(meta_X_val), torch.FloatTensor(meta_y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Training loop
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    logger.info(f"Starting meta-ensemble training: {epochs} max epochs, patience={patience}")

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred, _conf = model(X_batch)
            pred = pred.squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULTS["gradient_clip"])
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred, _conf = model(X_batch)
                pred = pred.squeeze(-1)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
                val_batches += 1
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        val_loss /= max(val_batches, 1)
        preds = np.concatenate(all_preds) if all_preds else np.array([])
        targets = np.concatenate(all_targets) if all_targets else np.array([])
        val_dir_acc = directional_accuracy(preds, targets) if len(preds) > 0 else 0.5

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or patience_counter == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_dir_acc={val_dir_acc:.3f}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}, "
                f"best={best_val_loss:.4f}"
            )

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best weights
    model.load_state_dict(best_state)
    model.eval()

    # Evaluate on test set
    test_dir_acc = 0.5
    test_loss = 0.0
    if len(meta_X_test) > 0:
        test_ds = TensorDataset(
            torch.FloatTensor(meta_X_test), torch.FloatTensor(meta_y_test)
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        test_loss_total = 0.0
        test_batches = 0
        test_preds = []
        test_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred, _conf = model(X_batch)
                pred = pred.squeeze(-1)
                loss = criterion(pred, y_batch)
                test_loss_total += loss.item()
                test_batches += 1
                test_preds.append(pred.cpu().numpy())
                test_targets.append(y_batch.cpu().numpy())

        test_loss = test_loss_total / max(test_batches, 1)
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        test_dir_acc = directional_accuracy(test_preds, test_targets)
        logger.info(f"Test set: loss={test_loss:.4f}, dir_acc={test_dir_acc:.3f}")

        # Compare to simple average
        simple_avg_preds = meta_X_test[:, INPUT_DIM:].mean(axis=1)
        simple_avg_acc = directional_accuracy(simple_avg_preds, meta_y_test)
        logger.info(f"Simple average baseline: dir_acc={simple_avg_acc:.3f}")
        logger.info(
            f"Ensemble improvement over simple avg: "
            f"{(test_dir_acc - simple_avg_acc)*100:+.1f} percentage points"
        )

    # Save
    metadata = {
        "model": "meta_ensemble",
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "epochs_trained": epoch + 1,
        "val_loss": float(best_val_loss),
        "val_dir_acc": float(val_dir_acc),
        "test_loss": float(test_loss),
        "test_dir_acc": float(test_dir_acc),
        "train_samples": len(meta_X_train),
        "val_samples": len(meta_X_val),
        "test_samples": len(meta_X_test),
        "device": str(device),
        "base_models": BASE_MODEL_NAMES,
    }
    save_model_checkpoint(model, save_path, metadata)

    return metadata


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train Meta-Ensemble (stacking layer)")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--data-dir", default="data/training")
    args = parser.parse_args()

    results = train_meta_ensemble(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
    )

    print(f"\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
