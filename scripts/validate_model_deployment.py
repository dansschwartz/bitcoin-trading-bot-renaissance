#!/usr/bin/env python3
"""
Validate ML Model Deployment — checks all trained models load correctly
and produce valid predictions.

Usage:
    python scripts/validate_model_deployment.py
    python scripts/validate_model_deployment.py --base-dir /path/to/project
    python scripts/validate_model_deployment.py --verbose
    python scripts/validate_model_deployment.py --skip-db
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Add project root to sys.path so we can import ml_model_loader
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

logger = logging.getLogger("validate_model_deployment")

# ── Constants ─────────────────────────────────────────────────────────────────

SEQ_LEN = 30
INPUT_DIM = 98  # ml_model_loader.INPUT_DIM (current weights may differ)

DB_PATH = os.path.join("data", "renaissance_bot.db")

# Expected model files in models/trained/
PYTORCH_MODEL_FILES = {
    "quantum_transformer": "models/trained/best_quantum_transformer_model.pth",
    "bidirectional_lstm": "models/trained/best_bidirectional_lstm_model.pth",
    "dilated_cnn": "models/trained/best_dilated_cnn_model.pth",
    "cnn": "models/trained/best_cnn_model.pth",
    "gru": "models/trained/best_gru_model.pth",
    "meta_ensemble": "models/trained/best_meta_ensemble_model.pth",
}
LIGHTGBM_MODEL_FILE = "models/trained/lightgbm_model.txt"


# ── Helpers ───────────────────────────────────────────────────────────────────

def md5_file(path: str) -> str:
    """Compute the MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def _open_db_readonly(base_dir: str) -> Optional[sqlite3.Connection]:
    """Open the renaissance_bot.db in read-only mode. Returns None on failure."""
    db_full = os.path.join(base_dir, DB_PATH)
    if not os.path.exists(db_full):
        logger.warning("Database not found: %s", db_full)
        return None
    try:
        uri = f"file:{db_full}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:
        logger.warning("Could not open database: %s", exc)
        return None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


# ── Validation Steps ──────────────────────────────────────────────────────────

class ModelValidationResult:
    """Result for a single model's validation."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.file_exists: bool = False
        self.file_path: str = ""
        self.file_size: int = 0
        self.file_hash: str = ""
        self.load_ok: bool = False
        self.load_error: str = ""
        self.inference_ok: bool = False
        self.inference_error: str = ""
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.output_value: Optional[float] = None
        self.value_in_range: bool = False
        self.db_prediction_count: int = 0
        self.db_accuracy: Optional[float] = None
        self.ledger_accuracy: Optional[float] = None
        self.passed: bool = False

    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        parts = [f"[{status}]  {self.name:<25s}"]
        if self.file_exists:
            parts.append(f"size={human_size(self.file_size):>10s}")
            parts.append(f"md5={self.file_hash[:12]}")
        else:
            parts.append("FILE MISSING")
            return "  ".join(parts)
        if self.load_ok:
            parts.append("load=OK")
        else:
            parts.append(f"load=FAIL({self.load_error[:40]})")
            return "  ".join(parts)
        if self.inference_ok:
            parts.append(f"pred={self.output_value:+.6f}")
            parts.append(f"shape={self.output_shape}")
            parts.append(f"range={'OK' if self.value_in_range else 'BAD'}")
        else:
            parts.append(f"infer=FAIL({self.inference_error[:40]})")
        if self.db_prediction_count > 0:
            acc_str = f"{self.db_accuracy:.1%}" if self.db_accuracy is not None else "N/A"
            parts.append(f"db_preds={self.db_prediction_count}  db_acc={acc_str}")
        if self.ledger_accuracy is not None:
            parts.append(f"ledger_acc={self.ledger_accuracy:.1%}")
        return "  ".join(parts)


def validate_file(result: ModelValidationResult, base_dir: str, rel_path: str) -> None:
    """Check file existence, size, and hash."""
    full = os.path.join(base_dir, rel_path)
    result.file_path = full
    if not os.path.isfile(full):
        result.file_exists = False
        return
    result.file_exists = True
    result.file_size = os.path.getsize(full)
    result.file_hash = md5_file(full)


def _detect_model_input_dim(model: Any) -> int:
    """Detect a model's expected input_dim from its first layer weights.

    Uses ml_model_loader._detect_input_dim() if available, otherwise
    inspects model parameters directly.
    """
    # If the model stores _input_dim (e.g. meta_ensemble), use that
    if hasattr(model, '_input_dim'):
        return model._input_dim

    try:
        from ml_model_loader import _detect_input_dim
        return _detect_input_dim(model)
    except (ImportError, Exception):
        pass

    try:
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                if 'weight_ih_l0' in name:
                    return param.shape[1]
                if 'weight' in name and 'norm' not in name and 'bn' not in name:
                    return param.shape[1]
                if 'conv' in name.lower() and param.dim() == 3:
                    return param.shape[1]
    except Exception:
        pass
    return INPUT_DIM


def validate_pytorch_model(
    result: ModelValidationResult,
    base_dir: str,
) -> None:
    """Load a PyTorch model and run dummy inference."""
    try:
        from ml_model_loader import load_trained_models
    except ImportError as exc:
        result.load_error = f"import error: {exc}"
        return

    try:
        # load_trained_models loads ALL models; we will validate individually
        # but for efficiency we load once and cache
        if not hasattr(validate_pytorch_model, "_cache"):
            validate_pytorch_model._cache = load_trained_models(base_dir=base_dir)
        models = validate_pytorch_model._cache
    except Exception as exc:
        result.load_error = str(exc)
        return

    model = models.get(result.name)
    if model is None:
        result.load_error = f"model '{result.name}' not in loaded dict"
        return
    result.load_ok = True

    # Detect this model's actual input dimension from its weights
    model_input_dim = _detect_model_input_dim(model)

    # Inference
    try:
        import torch

        dummy = np.random.randn(SEQ_LEN, model_input_dim).astype(np.float32)

        if result.name == "meta_ensemble":
            # Meta-ensemble takes (batch, input_dim + 5):
            #   first input_dim dims = market features
            #   last 5 dims = base model predictions
            feat = np.random.randn(model_input_dim).astype(np.float32)
            base_preds = np.zeros(5, dtype=np.float32)
            meta_input = np.concatenate([feat, base_preds])
            x = torch.FloatTensor(meta_input).unsqueeze(0)  # (1, input_dim+5)
        else:
            x = torch.FloatTensor(dummy).unsqueeze(0)  # (1, SEQ_LEN, input_dim)

        with torch.no_grad():
            output = model(x)

        if isinstance(output, tuple):
            pred_tensor = output[0]
        else:
            pred_tensor = output

        result.output_shape = tuple(pred_tensor.shape)
        result.output_value = float(pred_tensor.flatten()[0])
        result.inference_ok = True

        # Check value range: after tanh the prediction should be in [-1, 1],
        # but raw output may exceed that; we check the raw value is finite
        result.value_in_range = (
            np.isfinite(result.output_value) and abs(result.output_value) < 100.0
        )
    except Exception as exc:
        result.inference_error = str(exc)


def validate_lightgbm_model(
    result: ModelValidationResult,
    base_dir: str,
) -> None:
    """Load and validate the LightGBM model."""
    try:
        from ml_model_loader import load_lightgbm_model, predict_lightgbm
    except ImportError as exc:
        result.load_error = f"import error: {exc}"
        return

    try:
        model = load_lightgbm_model(base_dir=base_dir)
    except Exception as exc:
        result.load_error = str(exc)
        return

    if model is None:
        result.load_error = "load_lightgbm_model returned None (missing file or lightgbm not installed)"
        return
    result.load_ok = True

    # Inference
    try:
        dummy_features = np.random.randn(SEQ_LEN, INPUT_DIM).astype(np.float32)
        dummy_prices = np.linspace(100.0, 105.0, 100).astype(np.float32)
        pred_val, conf_val = predict_lightgbm(model, dummy_features, dummy_prices)
        result.inference_ok = True
        result.output_shape = (1,)
        result.output_value = pred_val
        result.value_in_range = (
            np.isfinite(pred_val) and -1.0 <= pred_val <= 1.0
            and np.isfinite(conf_val) and 0.0 <= conf_val <= 1.0
        )
    except Exception as exc:
        result.inference_error = str(exc)


def validate_db_predictions(
    result: ModelValidationResult,
    conn: Optional[sqlite3.Connection],
) -> None:
    """Query ml_predictions table for recent prediction counts and accuracy."""
    if conn is None:
        return
    if not _table_exists(conn, "ml_predictions"):
        return

    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM ml_predictions WHERE model_name = ?",
            (result.name,),
        ).fetchone()
        result.db_prediction_count = row[0] if row else 0
    except Exception as exc:
        logger.debug("ml_predictions query failed for %s: %s", result.name, exc)


def validate_model_ledger(
    result: ModelValidationResult,
    conn: Optional[sqlite3.Connection],
) -> None:
    """Query model_ledger table for stored accuracy metrics."""
    if conn is None:
        return
    if not _table_exists(conn, "model_ledger"):
        return

    try:
        row = conn.execute(
            "SELECT accuracy FROM model_ledger WHERE model_name = ? AND status = 'active' "
            "ORDER BY timestamp DESC LIMIT 1",
            (result.name,),
        ).fetchone()
        if row and row[0] is not None:
            result.ledger_accuracy = row[0]
    except Exception as exc:
        logger.debug("model_ledger query failed for %s: %s", result.name, exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_validation(base_dir: str, skip_db: bool = False, verbose: bool = False) -> List[ModelValidationResult]:
    """Run full validation on all models. Returns list of results."""
    results: List[ModelValidationResult] = []

    # Open DB once (read-only)
    conn: Optional[sqlite3.Connection] = None
    if not skip_db:
        conn = _open_db_readonly(base_dir)

    # ---------- PyTorch models ----------
    for name, rel_path in PYTORCH_MODEL_FILES.items():
        r = ModelValidationResult(name)
        validate_file(r, base_dir, rel_path)
        if r.file_exists:
            validate_pytorch_model(r, base_dir)
        if not skip_db:
            validate_db_predictions(r, conn)
            validate_model_ledger(r, conn)
        # Determine pass/fail
        r.passed = r.file_exists and r.load_ok and r.inference_ok and r.value_in_range
        results.append(r)

    # ---------- LightGBM ----------
    r = ModelValidationResult("lightgbm")
    validate_file(r, base_dir, LIGHTGBM_MODEL_FILE)
    if r.file_exists:
        validate_lightgbm_model(r, base_dir)
    if not skip_db:
        validate_db_predictions(r, conn)
        validate_model_ledger(r, conn)
    r.passed = r.file_exists and r.load_ok and r.inference_ok and r.value_in_range
    results.append(r)

    if conn is not None:
        conn.close()

    return results


def print_summary(results: List[ModelValidationResult]) -> None:
    """Print a formatted summary table."""
    sep = "=" * 120
    print()
    print(sep)
    print("  ML MODEL DEPLOYMENT VALIDATION REPORT")
    print(f"  Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(sep)
    print()

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    # File inventory
    print("  FILE INVENTORY")
    print("  " + "-" * 100)
    print(f"  {'Model':<25s}  {'File Exists':<12s}  {'Size':>10s}  {'MD5 Hash':<34s}")
    print("  " + "-" * 100)
    for r in results:
        exists_str = "YES" if r.file_exists else "MISSING"
        size_str = human_size(r.file_size) if r.file_exists else "-"
        hash_str = r.file_hash if r.file_exists else "-"
        print(f"  {r.name:<25s}  {exists_str:<12s}  {size_str:>10s}  {hash_str:<34s}")
    print()

    # Load & Inference
    print("  LOAD & INFERENCE")
    print("  " + "-" * 100)
    print(f"  {'Model':<25s}  {'Load':<8s}  {'Infer':<8s}  {'Shape':<16s}  {'Output':>12s}  {'In Range':<10s}")
    print("  " + "-" * 100)
    for r in results:
        load_str = "OK" if r.load_ok else (f"FAIL" if r.file_exists else "SKIP")
        infer_str = "OK" if r.inference_ok else ("FAIL" if r.file_exists and r.load_ok else "SKIP")
        shape_str = str(r.output_shape) if r.output_shape else "-"
        out_str = f"{r.output_value:+.6f}" if r.output_value is not None else "-"
        range_str = "OK" if r.value_in_range else ("BAD" if r.inference_ok else "-")
        print(f"  {r.name:<25s}  {load_str:<8s}  {infer_str:<8s}  {shape_str:<16s}  {out_str:>12s}  {range_str:<10s}")

        # Print errors if present
        if r.load_error:
            print(f"    -> Load error: {r.load_error}")
        if r.inference_error:
            print(f"    -> Inference error: {r.inference_error}")
    print()

    # DB Comparison
    has_db_data = any(r.db_prediction_count > 0 or r.ledger_accuracy is not None for r in results)
    if has_db_data:
        print("  DATABASE METRICS")
        print("  " + "-" * 100)
        print(f"  {'Model':<25s}  {'DB Predictions':>15s}  {'DB Accuracy':>12s}  {'Ledger Accuracy':>16s}")
        print("  " + "-" * 100)
        for r in results:
            preds_str = str(r.db_prediction_count) if r.db_prediction_count > 0 else "-"
            db_acc_str = f"{r.db_accuracy:.1%}" if r.db_accuracy is not None else "-"
            led_acc_str = f"{r.ledger_accuracy:.1%}" if r.ledger_accuracy is not None else "-"
            print(f"  {r.name:<25s}  {preds_str:>15s}  {db_acc_str:>12s}  {led_acc_str:>16s}")
        print()
    else:
        print("  DATABASE METRICS: No data found (tables empty or DB unavailable)")
        print()

    # Final verdict
    print(sep)
    if passed == total:
        print(f"  RESULT: ALL {total} MODELS PASSED")
    else:
        print(f"  RESULT: {passed}/{total} PASSED, {total - passed} FAILED")
        for r in results:
            if not r.passed:
                reason = "file missing" if not r.file_exists else (
                    f"load failed: {r.load_error[:60]}" if not r.load_ok else (
                        f"inference failed: {r.inference_error[:60]}" if not r.inference_ok else (
                            "output out of range"
                        )
                    )
                )
                print(f"    FAIL: {r.name} -- {reason}")
    print(sep)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate that all ML models load and produce valid predictions."
    )
    parser.add_argument(
        "--base-dir",
        default=PROJECT_ROOT,
        help="Project root directory (default: auto-detected from script location)",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip database queries (ml_predictions, model_ledger)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.isdir(base_dir):
        print(f"ERROR: base directory does not exist: {base_dir}")
        return 1

    # Change to project root so relative paths in ml_model_loader work
    original_cwd = os.getcwd()
    os.chdir(base_dir)

    try:
        results = run_validation(base_dir, skip_db=args.skip_db, verbose=args.verbose)
        print_summary(results)
    finally:
        os.chdir(original_cwd)

    all_passed = all(r.passed for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
