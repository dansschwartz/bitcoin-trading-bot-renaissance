"""
Train the Quantum Transformer model on historical OHLCV data.

Uses the exact same TrainedQuantumTransformer architecture from ml_model_loader,
ensuring weight compatibility between training and inference.

Usage:
    python -m scripts.training.train_quantum_transformer
    python -m scripts.training.train_quantum_transformer --epochs 100 --lr 5e-4 --batch-size 64
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

from ml_model_loader import TrainedQuantumTransformer
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

# Training hyperparameters
DEFAULTS = {
    "epochs": 100,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 0,  # v7: wd compounds 13.5x on full data (10K+ batches/epoch)
    "patience": 15,
    "gradient_clip": 1.0,
    "warmup_epochs": 3,
    "attn_lr_scale": 0.1,  # Attention params at 0.1x LR (fragile to large updates)
    "max_collapse_recoveries": 3,
}

SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "trained", "best_quantum_transformer_model.pth")


def train_quantum_transformer(
    epochs: int = DEFAULTS["epochs"],
    batch_size: int = DEFAULTS["batch_size"],
    lr: float = DEFAULTS["lr"],
    weight_decay: float = DEFAULTS["weight_decay"],
    patience: int = DEFAULTS["patience"],
    data_dir: str = "data/training",
    save_path: str = SAVE_PATH,
) -> dict:
    """Train the Quantum Transformer model.

    Args:
        epochs: Maximum training epochs
        batch_size: Training batch size
        lr: Initial learning rate
        weight_decay: AdamW weight decay
        patience: Early stopping patience
        data_dir: Directory with training CSVs
        save_path: Path to save best model

    Returns:
        Dict with training results (val_loss, val_dir_acc, test_dir_acc, epochs_trained)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Training QuantumTransformer on {device}")

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

    # Create model with differential LR (v7: attention at 0.1x, wd=0)
    model = TrainedQuantumTransformer().to(device)

    attn_params = []
    other_params = []
    for pname, p in model.named_parameters():
        if any(k in pname for k in ['attention', 'pos_encoding', 'skip_enhancement']):
            attn_params.append(p)
        else:
            other_params.append(p)

    attn_lr = lr * DEFAULTS["attn_lr_scale"]
    optimizer = torch.optim.AdamW([
        {'params': attn_params, 'lr': attn_lr, 'weight_decay': 0},
        {'params': other_params, 'lr': lr, 'weight_decay': 0},
    ])
    logger.info(f"Param groups: attention={sum(p.numel() for p in attn_params):,} "
                f"(lr={attn_lr:.1e}), other={sum(p.numel() for p in other_params):,} (lr={lr:.1e})")

    import math
    warmup_epochs = DEFAULTS["warmup_epochs"]
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size)

    # Training loop with collapse recovery
    criterion = DirectionalLoss(logit_scale=20.0, margin=0.10)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    collapse_recoveries = 0
    max_recoveries = DEFAULTS["max_collapse_recoveries"]

    logger.info(f"Starting training: {epochs} max epochs, patience={patience}")

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip=DEFAULTS["gradient_clip"]
        )
        val_loss, val_dir_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

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

        # Collapse recovery: reload best checkpoint and halve LR
        if val_dir_acc < 0.49 and epoch >= warmup_epochs and best_state is not None:
            if collapse_recoveries < max_recoveries:
                collapse_recoveries += 1
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.5
                logger.info(f"  >>> Collapse recovery #{collapse_recoveries}: "
                            f"reloaded best, halved LR to {optimizer.param_groups[0]['lr']:.2e}")
                patience_counter = 0
                continue
            else:
                logger.info(f"  >>> Max recoveries ({max_recoveries}) reached, stopping")
                break

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
        "model": "quantum_transformer",
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

    parser = argparse.ArgumentParser(description="Train Quantum Transformer")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--data-dir", default="data/training")
    args = parser.parse_args()

    results = train_quantum_transformer(
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
