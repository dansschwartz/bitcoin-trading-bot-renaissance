"""
Train the Dilated CNN model on historical OHLCV data.

Uses the exact same TrainedDilatedCNN architecture from ml_model_loader,
ensuring weight compatibility between training and inference.

Usage:
    python -m scripts.training.train_dilated_cnn
    python -m scripts.training.train_dilated_cnn --epochs 100 --lr 5e-4
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_model_loader import TrainedDilatedCNN
from scripts.training.training_utils import (
    DirectionalLoss,
    generate_sequences,
    load_derivatives_csvs,
    load_training_csvs,
    make_dataloaders,
    save_model_checkpoint,
    train_epoch,
    validate_epoch,
    walk_forward_split,
)

logger = logging.getLogger(__name__)

DEFAULTS = {
    "epochs": 100,
    "batch_size": 64,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "patience": 15,
    "directional_weight": 0.3,
    "directional_weight_max": 0.5,
    "gradient_clip": 1.0,
}

SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "trained", "best_dilated_cnn_model.pth")


def train_dilated_cnn(
    epochs: int = DEFAULTS["epochs"],
    batch_size: int = DEFAULTS["batch_size"],
    lr: float = DEFAULTS["lr"],
    weight_decay: float = DEFAULTS["weight_decay"],
    patience: int = DEFAULTS["patience"],
    data_dir: str = "data/training",
    save_path: str = SAVE_PATH,
) -> dict:
    """Train the Dilated CNN model.

    Returns:
        Dict with training results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training DilatedCNN on {device}")

    # Load and split data
    pair_dfs = load_training_csvs(data_dir=data_dir)
    if not pair_dfs:
        raise RuntimeError(f"No training data found in {data_dir}. Run fetch_training_data first.")

    # Load derivatives data (optional — zeros if missing)
    derivatives_dfs, fear_greed_df = load_derivatives_csvs(
        data_dir=os.path.join(data_dir, "derivatives")
    )

    logger.info("Splitting data (walk-forward)...")
    train_dfs, val_dfs, test_dfs = walk_forward_split(pair_dfs)

    logger.info("Generating training sequences...")
    X_train, y_train = generate_sequences(
        train_dfs, stride=1,
        derivatives_dfs=derivatives_dfs, fear_greed_df=fear_greed_df,
    )
    logger.info("Generating validation sequences...")
    X_val, y_val = generate_sequences(
        val_dfs, stride=1,
        derivatives_dfs=derivatives_dfs, fear_greed_df=fear_greed_df,
    )
    logger.info("Generating test sequences...")
    X_test, y_test = generate_sequences(
        test_dfs, stride=1,
        derivatives_dfs=derivatives_dfs, fear_greed_df=fear_greed_df,
    )

    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError("Not enough data to generate sequences. Need more historical bars.")

    logger.info(f"Dataset sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Create model and training components
    model = TrainedDilatedCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
    )

    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size)

    # Training loop
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    dir_weight = DEFAULTS["directional_weight"]
    dir_weight_step = (DEFAULTS["directional_weight_max"] - dir_weight) / max(epochs, 1)

    logger.info(f"Starting training: {epochs} max epochs, patience={patience}")

    for epoch in range(epochs):
        dir_weight = min(dir_weight + dir_weight_step, DEFAULTS["directional_weight_max"])
        criterion = DirectionalLoss(logit_scale=20.0, margin=0.10)

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip=DEFAULTS["gradient_clip"]
        )
        val_loss, val_dir_acc = validate_epoch(model, val_loader, criterion, device)
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
    test_loss, test_dir_acc = 0.0, 0.5
    if len(X_test) > 0:
        from scripts.training.training_utils import evaluate_on_dataset
        test_loss, test_dir_acc = evaluate_on_dataset(model, X_test, y_test, device)
        logger.info(f"Test set: loss={test_loss:.4f}, dir_acc={test_dir_acc:.3f}")

    # Save
    metadata = {
        "model": "dilated_cnn",
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "epochs_trained": epoch + 1,
        "val_loss": float(best_val_loss),
        "val_dir_acc": float(val_dir_acc),
        "test_loss": float(test_loss),
        "test_dir_acc": float(test_dir_acc),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "device": str(device),
    }
    save_model_checkpoint(model, save_path, metadata)

    return metadata


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train Dilated CNN")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--data-dir", default="data/training")
    args = parser.parse_args()

    results = train_dilated_cnn(
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
