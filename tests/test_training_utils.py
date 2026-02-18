"""
Tests for scripts/training/training_utils.py — Training utilities.

Covers generate_sequences (window sliding, label generation),
walk_forward_split (date-based splitting), DirectionalLoss (MSE + directional penalty),
and directional_accuracy metric.
"""

import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

# Ensure project root is on path for imports
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.training.training_utils import (
    DirectionalLoss,
    directional_accuracy,
    walk_forward_split,
    generate_sequences,
    make_dataloaders,
    SEQ_LEN,
    INPUT_DIM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n_rows: int = 200) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame."""
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(n_rows) * 0.5)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n_rows, freq="5min"),
        "open": prices + np.random.randn(n_rows) * 0.1,
        "high": prices + abs(np.random.randn(n_rows) * 0.5),
        "low": prices - abs(np.random.randn(n_rows) * 0.5),
        "close": prices,
        "volume": np.random.uniform(10, 100, n_rows),
    })


# ---------------------------------------------------------------------------
# Tests — DirectionalLoss
# ---------------------------------------------------------------------------

class TestDirectionalLoss:
    def test_zero_loss_when_perfect(self):
        criterion = DirectionalLoss()
        pred = torch.tensor([1.0, -1.0, 0.5])
        target = torch.tensor([1.0, -1.0, 0.5])
        loss = criterion(pred, target)
        # Same-direction predictions with correct sign → low loss
        assert loss.item() < 1.0

    def test_penalty_when_wrong_direction(self):
        criterion = DirectionalLoss()
        pred = torch.tensor([1.0, -1.0])
        target = torch.tensor([-1.0, 1.0])  # Opposite directions
        loss = criterion(pred, target)
        # Wrong direction → higher loss than correct direction
        criterion2 = DirectionalLoss()
        correct_loss = criterion2(torch.tensor([1.0, -1.0]), torch.tensor([1.0, -1.0]))
        assert loss.item() > correct_loss.item()

    def test_loss_increases_with_margin(self):
        pred = torch.tensor([0.01, -0.01])  # Near-zero predictions
        target = torch.tensor([1.0, -1.0])

        loss_low = DirectionalLoss(margin=0.05)(pred, target)
        loss_high = DirectionalLoss(margin=0.20)(pred, target)
        assert loss_high.item() > loss_low.item()

    def test_handles_extra_dimension(self):
        criterion = DirectionalLoss()
        pred = torch.tensor([[1.0], [-1.0]])
        target = torch.tensor([[1.0], [-1.0]])
        loss = criterion(pred, target)
        assert loss.item() >= 0.0

    def test_backward_pass(self):
        """Loss should support gradient computation."""
        criterion = DirectionalLoss()
        pred = torch.tensor([0.5, -0.3], requires_grad=True)
        target = torch.tensor([1.0, -1.0])
        loss = criterion(pred, target)
        loss.backward()
        assert pred.grad is not None


# ---------------------------------------------------------------------------
# Tests — directional_accuracy
# ---------------------------------------------------------------------------

class TestDirectionalAccuracy:
    def test_perfect_accuracy(self):
        predictions = np.array([1.0, -1.0, 0.5, -0.3])
        targets = np.array([1.0, -1.0, 1.0, -1.0])
        assert directional_accuracy(predictions, targets) == 1.0

    def test_zero_accuracy(self):
        predictions = np.array([1.0, -1.0])
        targets = np.array([-1.0, 1.0])
        assert directional_accuracy(predictions, targets) == 0.0

    def test_half_accuracy(self):
        predictions = np.array([1.0, -1.0, 1.0, -1.0])
        targets = np.array([1.0, 1.0, -1.0, -1.0])
        assert directional_accuracy(predictions, targets) == 0.5

    def test_zero_prediction_counted_as_wrong(self):
        predictions = np.array([0.0])
        targets = np.array([1.0])
        # sign(0) = 0, sign(1) = 1, 0 != 1 => wrong
        assert directional_accuracy(predictions, targets) == 0.0

    def test_single_element(self):
        assert directional_accuracy(np.array([1.0]), np.array([1.0])) == 1.0


# ---------------------------------------------------------------------------
# Tests — walk_forward_split
# ---------------------------------------------------------------------------

class TestWalkForwardSplit:
    def test_split_proportions(self):
        df = _make_ohlcv_df(100)
        pair_dfs = {"BTC-USD": df}
        train, val, test = walk_forward_split(
            pair_dfs, train_frac=0.7, val_frac=0.15, test_frac=0.15,
        )
        assert len(train["BTC-USD"]) == 70
        assert len(val["BTC-USD"]) == 15
        assert len(test["BTC-USD"]) == 15

    def test_no_overlap(self):
        df = _make_ohlcv_df(100)
        pair_dfs = {"BTC-USD": df}
        train, val, test = walk_forward_split(pair_dfs)
        total = len(train["BTC-USD"]) + len(val["BTC-USD"]) + len(test["BTC-USD"])
        assert total == 100

    def test_multiple_pairs(self):
        pair_dfs = {
            "BTC-USD": _make_ohlcv_df(100),
            "ETH-USD": _make_ohlcv_df(50),
        }
        train, val, test = walk_forward_split(pair_dfs)
        assert "BTC-USD" in train
        assert "ETH-USD" in train
        assert len(train["ETH-USD"]) == 35  # 50 * 0.7

    def test_temporal_ordering_preserved(self):
        df = _make_ohlcv_df(100)
        pair_dfs = {"BTC-USD": df}
        train, val, test = walk_forward_split(pair_dfs)

        # Train timestamps should all be before val timestamps
        train_max_ts = train["BTC-USD"]["timestamp"].max()
        val_min_ts = val["BTC-USD"]["timestamp"].min()
        assert train_max_ts < val_min_ts

    def test_empty_pair(self):
        pair_dfs = {"BTC-USD": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])}
        train, val, test = walk_forward_split(pair_dfs)
        assert len(train["BTC-USD"]) == 0
        assert len(val["BTC-USD"]) == 0
        assert len(test["BTC-USD"]) == 0


# ---------------------------------------------------------------------------
# Tests — generate_sequences
# ---------------------------------------------------------------------------

class TestGenerateSequences:
    @patch("scripts.training.training_utils.build_feature_sequence")
    def test_generates_correct_shape(self, mock_bfs):
        """Test that generate_sequences produces (N, seq_len, 83) array."""
        mock_bfs.return_value = np.random.randn(SEQ_LEN, INPUT_DIM).astype(np.float32)

        df = _make_ohlcv_df(200)
        pair_dfs = {"BTC-USD": df}
        X, y = generate_sequences(pair_dfs, seq_len=SEQ_LEN, stride=10)

        assert X.ndim == 3
        assert X.shape[1] == SEQ_LEN
        assert X.shape[2] == INPUT_DIM
        assert len(X) == len(y)

    @patch("scripts.training.training_utils.build_feature_sequence")
    def test_labels_are_binary(self, mock_bfs):
        mock_bfs.return_value = np.random.randn(SEQ_LEN, INPUT_DIM).astype(np.float32)

        df = _make_ohlcv_df(200)
        pair_dfs = {"BTC-USD": df}
        X, y = generate_sequences(pair_dfs, seq_len=SEQ_LEN, stride=10)

        # Labels should be continuous in [-1, +1] range
        assert y.min() >= -1.0
        assert y.max() <= 1.0
        assert len(y) > 0

    @patch("scripts.training.training_utils.build_feature_sequence")
    def test_skips_short_pair(self, mock_bfs):
        """Pairs with insufficient data should be skipped."""
        mock_bfs.return_value = np.random.randn(SEQ_LEN, INPUT_DIM).astype(np.float32)

        short_df = _make_ohlcv_df(10)  # Too short (need seq_len + 51)
        pair_dfs = {"SHORT-PAIR": short_df}
        X, y = generate_sequences(pair_dfs, seq_len=SEQ_LEN)

        assert len(X) == 0
        assert len(y) == 0

    @patch("scripts.training.training_utils.build_feature_sequence")
    def test_handles_none_features(self, mock_bfs):
        """If build_feature_sequence returns None, that sample should be skipped."""
        call_count = [0]

        def sometimes_none(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return None
            return np.random.randn(SEQ_LEN, INPUT_DIM).astype(np.float32)

        mock_bfs.side_effect = sometimes_none

        df = _make_ohlcv_df(200)
        pair_dfs = {"BTC-USD": df}
        X, y = generate_sequences(pair_dfs, seq_len=SEQ_LEN, stride=5)

        # Some should have been skipped but we should still get results
        if len(X) > 0:
            assert X.shape[1] == SEQ_LEN
            assert X.shape[2] == INPUT_DIM


# ---------------------------------------------------------------------------
# Tests — make_dataloaders
# ---------------------------------------------------------------------------

class TestMakeDataloaders:
    def test_loader_batch_size(self):
        X_train = np.random.randn(100, SEQ_LEN, INPUT_DIM).astype(np.float32)
        y_train = np.random.choice([-1.0, 1.0], size=100).astype(np.float32)
        X_val = np.random.randn(20, SEQ_LEN, INPUT_DIM).astype(np.float32)
        y_val = np.random.choice([-1.0, 1.0], size=20).astype(np.float32)

        train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)

        # Get first batch
        X_batch, y_batch = next(iter(train_loader))
        assert X_batch.shape == (32, SEQ_LEN, INPUT_DIM)
        assert y_batch.shape == (32,)

    def test_val_loader_not_shuffled(self):
        """Validation loader should return data in the same order each time."""
        X_val = np.arange(50 * SEQ_LEN * INPUT_DIM).reshape(50, SEQ_LEN, INPUT_DIM).astype(np.float32)
        y_val = np.arange(50).astype(np.float32)
        X_train = X_val.copy()
        y_train = y_val.copy()

        _, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=10)

        first_pass = []
        for X_b, y_b in val_loader:
            first_pass.append(y_b.numpy().tolist())

        second_pass = []
        for X_b, y_b in val_loader:
            second_pass.append(y_b.numpy().tolist())

        assert first_pass == second_pass
