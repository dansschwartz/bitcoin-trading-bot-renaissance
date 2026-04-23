"""Tests for ml_model_loader — ML model loading, feature pipeline, and prediction.

Tests cover:
  - DISABLED_MODELS correctly loaded from config
  - Disabled models are skipped during prediction
  - Feature pipeline produces correct dimension (98)
  - Prediction output has expected shape and range
  - Stale model detection / per-pair model exclusions
  - Feature normalization handles NaN/inf inputs
  - Empty input data returns safe defaults
  - PredictionDebiaser and ModelPredictionNormalizer
  - should_include_model per-pair exclusions
"""

import os
import math
import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from ml_model_loader import (
    INPUT_DIM,
    INPUT_DIM_LEGACY,
    DISABLED_MODELS,
    MODEL_PAIR_EXCLUSIONS,
    _DEFAULT_DISABLED_MODELS,
    should_include_model,
    PredictionDebiaser,
    ModelPredictionNormalizer,
    TrainedQuantumTransformer,
    TrainedBidirectionalLSTM,
    TrainedDilatedCNN,
    TrainedCNN,
    TrainedGRU,
    TrainedMetaEnsemble,
    predict_with_models,
    _match_feature_dim,
    _prepare_lgb_features,
    predict_lightgbm,
    _detect_input_dim,
    _load_disabled_models,
    BASE_MODEL_NAMES,
    N_BASE_MODELS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def random_features():
    """Random feature sequence (seq_len=30, dim=INPUT_DIM)."""
    return np.random.randn(30, INPUT_DIM).astype(np.float32)


@pytest.fixture
def qt_model():
    """Fresh QuantumTransformer in eval mode."""
    model = TrainedQuantumTransformer(input_dim=INPUT_DIM)
    model.eval()
    return model


@pytest.fixture
def lstm_model():
    """Fresh Bidirectional LSTM in eval mode."""
    model = TrainedBidirectionalLSTM(input_dim=INPUT_DIM)
    model.eval()
    return model


@pytest.fixture
def cnn_model():
    """Fresh CNN in eval mode."""
    model = TrainedCNN(input_dim=INPUT_DIM)
    model.eval()
    return model


@pytest.fixture
def gru_model():
    """Fresh GRU in eval mode."""
    model = TrainedGRU(input_dim=INPUT_DIM)
    model.eval()
    return model


@pytest.fixture
def dilated_cnn_model():
    """Fresh Dilated CNN in eval mode."""
    model = TrainedDilatedCNN(input_dim=INPUT_DIM)
    model.eval()
    return model


@pytest.fixture
def meta_model():
    """Fresh Meta-Ensemble in eval mode."""
    model = TrainedMetaEnsemble(input_dim=INPUT_DIM, n_models=N_BASE_MODELS)
    model.eval()
    return model


@pytest.fixture
def models_dict(qt_model, lstm_model, cnn_model, dilated_cnn_model):
    """Dict of loaded models for predict_with_models (no GRU since it's disabled)."""
    return {
        "quantum_transformer": qt_model,
        "bidirectional_lstm": lstm_model,
        "cnn": cnn_model,
        "dilated_cnn": dilated_cnn_model,
    }


# ── Test: DISABLED_MODELS loaded correctly ──────────────────────────────────

class TestDisabledModels:
    def test_default_disabled_models(self):
        """Default disabled models set should contain 'gru'."""
        assert "gru" in _DEFAULT_DISABLED_MODELS

    def test_disabled_models_is_set(self):
        """DISABLED_MODELS should be a set."""
        assert isinstance(DISABLED_MODELS, set)

    def test_load_disabled_models_returns_set(self):
        """_load_disabled_models should return a set."""
        result = _load_disabled_models()
        assert isinstance(result, set)

    def test_load_disabled_models_fallback(self):
        """When config file doesn't exist, should return defaults."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = _load_disabled_models()
            assert result == _DEFAULT_DISABLED_MODELS


# ── Test: Disabled models skipped during prediction ──────────────────────────

class TestDisabledModelSkip:
    def test_disabled_model_returns_zero(self, random_features, gru_model):
        """Disabled models should be skipped and return 0.0."""
        models = {"gru": gru_model}  # GRU is in DISABLED_MODELS
        preds, confs = predict_with_models(models, random_features)
        # GRU should be skipped because it's in DISABLED_MODELS
        # It won't appear in predictions at all (skipped in the loop)
        assert preds.get("gru", 0.0) == 0.0

    def test_non_disabled_model_runs(self, random_features, qt_model):
        """Non-disabled models should produce predictions."""
        models = {"quantum_transformer": qt_model}
        preds, confs = predict_with_models(models, random_features)
        assert "quantum_transformer" in preds
        # Should have a non-trivial prediction (model is randomly initialized)
        assert isinstance(preds["quantum_transformer"], float)


# ── Test: Feature pipeline produces correct dimension ────────────────────────

class TestFeatureDimension:
    def test_input_dim_is_98(self):
        """INPUT_DIM constant should be 98."""
        assert INPUT_DIM == 98

    def test_legacy_dim_is_83(self):
        """INPUT_DIM_LEGACY should be 83."""
        assert INPUT_DIM_LEGACY == 83

    def test_random_features_correct_shape(self, random_features):
        """Random features fixture should have correct shape."""
        assert random_features.shape == (30, INPUT_DIM)
        assert random_features.dtype == np.float32


# ── Test: Model forward pass shapes ─────────────────────────────────────────

class TestModelForwardPass:
    def test_qt_output_shape(self, qt_model, random_features):
        """QuantumTransformer should output (batch, 1) prediction + uncertainty."""
        x = torch.FloatTensor(random_features).unsqueeze(0)  # (1, 30, 98)
        with torch.no_grad():
            pred, unc = qt_model(x)
        assert pred.shape == (1, 1)
        assert unc.shape == (1, 1)

    def test_lstm_output_shape(self, lstm_model, random_features):
        """BiLSTM should output (batch, 1) prediction + confidence."""
        x = torch.FloatTensor(random_features).unsqueeze(0)
        with torch.no_grad():
            pred, conf = lstm_model(x)
        assert pred.shape == (1, 1)
        assert conf.shape == (1, 1)

    def test_cnn_output_shape(self, cnn_model, random_features):
        """CNN should output (batch, 1) prediction."""
        x = torch.FloatTensor(random_features).unsqueeze(0)
        with torch.no_grad():
            pred = cnn_model(x)
        assert pred.shape == (1, 1)

    def test_gru_output_shape(self, gru_model, random_features):
        """GRU should output (batch, 1) prediction."""
        x = torch.FloatTensor(random_features).unsqueeze(0)
        with torch.no_grad():
            pred = gru_model(x)
        assert pred.shape == (1, 1)

    def test_dilated_cnn_output_shape(self, dilated_cnn_model, random_features):
        """Dilated CNN should output (batch, 1) prediction."""
        x = torch.FloatTensor(random_features).unsqueeze(0)
        with torch.no_grad():
            pred = dilated_cnn_model(x)
        assert pred.shape == (1, 1)

    def test_meta_ensemble_output_shape(self, meta_model):
        """Meta-ensemble should output (batch, 1) prediction + confidence."""
        x = torch.randn(1, INPUT_DIM + N_BASE_MODELS)
        with torch.no_grad():
            pred, conf = meta_model(x)
        assert pred.shape == (1, 1)
        assert conf.shape == (1, 1)


# ── Test: Prediction output range ───────────────────────────────────────────

class TestPredictionRange:
    def test_qt_prediction_bounded(self, qt_model, random_features):
        """QuantumTransformer predictions should be in [-1, 1] (tanh output)."""
        x = torch.FloatTensor(random_features).unsqueeze(0)
        with torch.no_grad():
            pred, _ = qt_model(x)
        assert -1.0 <= float(pred[0, 0]) <= 1.0

    def test_lstm_prediction_bounded(self, lstm_model, random_features):
        """BiLSTM predictions should be in [-1, 1] (tanh output)."""
        x = torch.FloatTensor(random_features).unsqueeze(0)
        with torch.no_grad():
            pred, _ = lstm_model(x)
        assert -1.0 <= float(pred[0, 0]) <= 1.0

    def test_cnn_prediction_bounded(self, cnn_model, random_features):
        """CNN predictions should be in [-1, 1] (tanh output)."""
        x = torch.FloatTensor(random_features).unsqueeze(0)
        with torch.no_grad():
            pred = cnn_model(x)
        assert -1.0 <= float(pred[0, 0]) <= 1.0

    def test_meta_confidence_bounded(self, meta_model):
        """Meta-ensemble confidence should be in [0, 1] (sigmoid output)."""
        x = torch.randn(1, INPUT_DIM + N_BASE_MODELS)
        with torch.no_grad():
            _, conf = meta_model(x)
        assert 0.0 <= float(conf[0, 0]) <= 1.0


# ── Test: predict_with_models full pipeline ─────────────────────────────────

class TestPredictWithModels:
    def test_returns_predictions_and_confidences(self, models_dict, random_features):
        """predict_with_models should return two dicts."""
        preds, confs = predict_with_models(models_dict, random_features)
        assert isinstance(preds, dict)
        assert isinstance(confs, dict)

    def test_all_models_have_predictions(self, models_dict, random_features):
        """Each model should have a prediction entry."""
        preds, confs = predict_with_models(models_dict, random_features)
        for name in models_dict:
            if name not in DISABLED_MODELS:
                assert name in preds
                assert name in confs

    def test_none_features_returns_zeros(self, models_dict):
        """None features should return zero predictions for all models."""
        preds, confs = predict_with_models(models_dict, None)
        for name, val in preds.items():
            assert val == 0.0
        for name, val in confs.items():
            assert val == 0.5

    def test_predictions_are_float(self, models_dict, random_features):
        """All predictions should be Python floats."""
        preds, confs = predict_with_models(models_dict, random_features)
        for val in preds.values():
            assert isinstance(val, float)
        for val in confs.values():
            assert isinstance(val, float)


# ── Test: Per-pair model exclusions ─────────────────────────────────────────

class TestPerPairExclusions:
    def test_qt_excluded_for_near(self):
        """QuantumTransformer should be excluded for NEAR."""
        assert not should_include_model("quantum_transformer", "NEAR-USD")
        assert not should_include_model("quantum_transformer", "NEARUSDT")

    def test_qt_included_for_btc(self):
        """QuantumTransformer should be included for BTC."""
        assert should_include_model("quantum_transformer", "BTC-USD")

    def test_qt_excluded_for_eth(self):
        """QuantumTransformer should be excluded for ETH (28% accuracy)."""
        assert not should_include_model("quantum_transformer", "ETH-USD")

    def test_unknown_model_included(self):
        """Models not in exclusion list should always be included."""
        assert should_include_model("some_new_model", "NEAR-USD")

    def test_exclusion_normalizes_pair_formats(self):
        """Exclusion should work for various pair format strings."""
        assert not should_include_model("quantum_transformer", "AVAX-USD")
        assert not should_include_model("quantum_transformer", "AVAX/USDT")
        assert not should_include_model("quantum_transformer", "AVAXUSDT")

    def test_excluded_model_gets_zero_in_predictions(self, models_dict, random_features):
        """Excluded model should get 0.0 prediction for excluded pair."""
        preds, confs = predict_with_models(models_dict, random_features, pair="NEAR-USD")
        assert preds.get("quantum_transformer", 0.0) == 0.0
        assert confs.get("quantum_transformer", 0.0) == 0.0


# ── Test: Feature normalization handles NaN/inf ─────────────────────────────

class TestFeatureNormalization:
    def test_nan_features_produce_valid_output(self, models_dict):
        """NaN features should not crash prediction (model gets cleaned data)."""
        features = np.full((30, INPUT_DIM), np.nan, dtype=np.float32)
        # Replace NaN with 0 for prediction (as build_feature_sequence does)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        preds, confs = predict_with_models(models_dict, features)
        for val in preds.values():
            assert not math.isnan(val)
            assert not math.isinf(val)

    def test_inf_features_produce_valid_output(self, models_dict):
        """Inf features should not crash prediction."""
        features = np.random.randn(30, INPUT_DIM).astype(np.float32)
        features[0, 0] = np.inf
        features[1, 1] = -np.inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        preds, confs = predict_with_models(models_dict, features)
        for val in preds.values():
            assert not math.isnan(val)
            assert not math.isinf(val)

    def test_all_zero_features(self, models_dict):
        """All-zero features should produce valid (if meaningless) output."""
        features = np.zeros((30, INPUT_DIM), dtype=np.float32)
        preds, confs = predict_with_models(models_dict, features)
        for val in preds.values():
            assert isinstance(val, float)
            assert not math.isnan(val)


# ── Test: Empty input returns safe defaults ─────────────────────────────────

class TestEmptyInput:
    def test_none_features_safe(self, models_dict):
        """None features → zeros and 0.5 confidence."""
        preds, confs = predict_with_models(models_dict, None)
        for v in preds.values():
            assert v == 0.0
        for v in confs.values():
            assert v == 0.5


# ── Test: PredictionDebiaser ────────────────────────────────────────────────

class TestPredictionDebiaser:
    def test_no_debias_during_warmup(self):
        """During warmup, raw prediction should be returned unchanged."""
        debiaser = PredictionDebiaser(alpha=0.01)
        # Default warmup is 200 samples
        result = debiaser.debias("test_model", 0.5)
        assert result == 0.5

    def test_debias_after_warmup(self):
        """After warmup, debiased value should differ from raw."""
        debiaser = PredictionDebiaser(alpha=0.1)
        debiaser._warmup = 10  # Reduce for test

        # Feed 15 constant predictions
        for _ in range(15):
            debiaser.debias("model_a", 0.1)

        # After warmup, should subtract the EMA (which is ~0.1)
        result = debiaser.debias("model_a", 0.1)
        assert abs(result) < 0.1  # Should be near zero after debiasing

    def test_get_bias(self):
        """get_bias should return current EMA."""
        debiaser = PredictionDebiaser(alpha=0.5)
        debiaser.debias("model_x", 1.0)
        bias = debiaser.get_bias("model_x")
        assert bias == 0.5  # alpha * 1.0 + (1-alpha) * 0.0 = 0.5

    def test_get_stats(self):
        """get_stats should return per-model info."""
        debiaser = PredictionDebiaser(alpha=0.01)
        debiaser.debias("model_a", 0.1)
        stats = debiaser.get_stats()
        assert "model_a" in stats
        assert "bias" in stats["model_a"]
        assert "samples" in stats["model_a"]
        assert stats["model_a"]["samples"] == 1


# ── Test: ModelPredictionNormalizer ──────────────────────────────────────────

class TestModelPredictionNormalizer:
    def test_passthrough_during_warmup(self):
        """During warmup, raw prediction should pass through."""
        norm = ModelPredictionNormalizer(alpha=0.05, min_samples=10)
        result = norm.normalize("test", 0.5)
        assert result == 0.5

    def test_normalize_after_warmup(self):
        """After warmup, output should be z-score normalized."""
        norm = ModelPredictionNormalizer(alpha=0.1, min_samples=5)

        # Feed 10 samples of value 1.0
        for _ in range(10):
            norm.normalize("model_b", 1.0)

        # After warmup, mean should be ~1.0, so a 1.0 input → ~0
        result = norm.normalize("model_b", 1.0)
        assert abs(result) < 5.0  # Should be near zero (z-scored around mean)

    def test_get_stats(self):
        """get_stats should return normalization stats."""
        norm = ModelPredictionNormalizer(alpha=0.05, min_samples=100)
        norm.normalize("model_c", 0.5)
        stats = norm.get_stats()
        assert "model_c" in stats
        assert "mean" in stats["model_c"]
        assert "std" in stats["model_c"]
        assert "samples" in stats["model_c"]


# ── Test: _match_feature_dim ────────────────────────────────────────────────

class TestMatchFeatureDim:
    def test_pad_when_features_too_small(self, qt_model):
        """Should pad with zeros when features have fewer dims than model expects."""
        x = torch.randn(1, 30, 50)  # Only 50 features
        result = _match_feature_dim(x, qt_model, "quantum_transformer")
        assert result.shape == (1, 30, INPUT_DIM)  # Padded to 98

    def test_truncate_when_features_too_large(self, qt_model):
        """Should truncate when features have more dims than model expects."""
        x = torch.randn(1, 30, 120)  # 120 > 98
        result = _match_feature_dim(x, qt_model, "quantum_transformer")
        assert result.shape == (1, 30, INPUT_DIM)

    def test_no_change_when_matching(self, qt_model):
        """Should return unchanged tensor when dimensions match."""
        x = torch.randn(1, 30, INPUT_DIM)
        result = _match_feature_dim(x, qt_model, "quantum_transformer")
        assert result.shape == (1, 30, INPUT_DIM)
        assert torch.equal(result, x)


# ── Test: _detect_input_dim ────────────────────────────────────────────────

class TestDetectInputDim:
    def test_detect_qt_dim(self, qt_model):
        """Should detect input dim from QuantumTransformer state dict."""
        sd = qt_model.state_dict()
        dim = _detect_input_dim("quantum_transformer", sd)
        assert dim == INPUT_DIM

    def test_detect_lstm_dim(self, lstm_model):
        """Should detect input dim from BiLSTM state dict."""
        sd = lstm_model.state_dict()
        dim = _detect_input_dim("bidirectional_lstm", sd)
        assert dim == INPUT_DIM

    def test_detect_cnn_dim(self, cnn_model):
        """Should detect input dim from CNN state dict."""
        sd = cnn_model.state_dict()
        dim = _detect_input_dim("cnn", sd)
        assert dim == INPUT_DIM

    def test_detect_gru_dim(self, gru_model):
        """Should detect input dim from GRU state dict."""
        sd = gru_model.state_dict()
        dim = _detect_input_dim("gru", sd)
        assert dim == INPUT_DIM

    def test_unknown_model_returns_none(self):
        """Unknown model name should return None."""
        dim = _detect_input_dim("unknown_model", {})
        assert dim is None


# ── Test: LightGBM feature preparation ──────────────────────────────────────

class TestLightGBMFeaturePrep:
    def test_prepare_lgb_features_shape(self, random_features):
        """_prepare_lgb_features should produce (1, INPUT_DIM*3) output."""
        result = _prepare_lgb_features(random_features)
        assert result.shape == (1, INPUT_DIM * 3)

    def test_prepare_lgb_features_content(self):
        """Output should contain [last, mean, std] of the sequence."""
        features = np.ones((10, 5), dtype=np.float32)
        features[-1, :] = 2.0  # Last row is 2.0
        result = _prepare_lgb_features(features)
        # Last = [2,2,2,2,2], Mean = [1.1,1.1,...], Std = [0.3,0.3,...]
        assert result[0, 0] == 2.0  # First 5 values are the last row
        assert result[0, 5] > 1.0  # Mean > 1 because last row is 2

    def test_predict_lightgbm_none_model(self):
        """None model should return (0.0, 0.0)."""
        pred, conf = predict_lightgbm(None, np.zeros((30, INPUT_DIM)))
        assert pred == 0.0
        assert conf == 0.0

    def test_predict_lightgbm_none_features(self):
        """None features should return (0.0, 0.0)."""
        pred, conf = predict_lightgbm(MagicMock(), None)
        assert pred == 0.0
        assert conf == 0.0


# ── Test: Model architecture constants ──────────────────────────────────────

class TestModelConstants:
    def test_base_model_names(self):
        """BASE_MODEL_NAMES should list all 6 base models."""
        assert len(BASE_MODEL_NAMES) == 6
        assert "quantum_transformer" in BASE_MODEL_NAMES
        assert "lightgbm" in BASE_MODEL_NAMES

    def test_n_base_models(self):
        """N_BASE_MODELS should match BASE_MODEL_NAMES length."""
        assert N_BASE_MODELS == len(BASE_MODEL_NAMES)
