"""
ML Model Loader — loads trained PyTorch model weights into matching architectures.

The trained models (in models/trained/) were trained with specific hyperparameters
that differ from the current neural_network_prediction_engine.py defaults.
This module creates architectures that exactly match the saved weight shapes,
loads them with strict=True to guarantee 100% weight match, and exposes
a simple prediction interface.

Feature dimension: INPUT_DIM=98 (49 single-pair + 15 cross-asset, padded)
  - 49 single-pair features: OHLCV, returns, MAs, RSI, MACD, BB, ATR, volume
  - 15 cross-asset features: lead signals, correlations, spreads, market-wide
"""

import os
import math
import logging
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Disabled models ──────────────────────────────────────────────────────────
# Models disabled from inference. Loaded from config/config.json "disabled_models".
# These are not loaded, not run, and pass 0.0 to the meta-ensemble.

# Default disabled models — overridden by config/config.json "disabled_models" if present
_DEFAULT_DISABLED_MODELS: set = {'gru'}

def _load_disabled_models() -> set:
    """Load disabled model list from config file."""
    try:
        import json
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        disabled = config.get('disabled_models', [])
        if isinstance(disabled, list) and disabled:
            return set(disabled)
    except Exception as e:
        logger.warning(f"Failed to load disabled models from config file: {e}")
    return _DEFAULT_DISABLED_MODELS

DISABLED_MODELS: set = _load_disabled_models()

# ── Per-pair model exclusions (live accuracy audit 2026-03-02) ────────────────
# Models that are actively anti-predictive (<35% accuracy) on specific pairs.
# These models are still loaded/run on other pairs — just excluded from the
# ensemble for these specific pairs where they hurt performance.
MODEL_PAIR_EXCLUSIONS: dict = {
    # QT: 91% on ZEC, 82% on ATOM, 80% on BCH — but 13% NEAR, 18% AVAX, 28% ETH, 32% UNI
    "quantum_transformer": {"NEAR", "AVAX", "ETH", "UNI"},
    # GRU: 33% on NEAR (already globally disabled, but if re-enabled this applies)
    "gru": {"NEAR"},
}


def should_include_model(model_name: str, pair: str) -> bool:
    """Check if a model should be included in ensemble for a specific pair."""
    exclusions = MODEL_PAIR_EXCLUSIONS.get(model_name, set())
    if not exclusions:
        return True
    # Normalize: extract base asset from any pair format (BTC-USD, BTC/USDT, BTCUSDT)
    base = pair.upper().replace("-USD", "").replace("/USDT", "").replace("USDT", "").replace("/USD", "")
    return base not in exclusions


# ── Feature dimension constants ───────────────────────────────────────────────

INPUT_DIM = 98          # Current feature dimension (padded to 98)
INPUT_DIM_LEGACY = 83   # Previous feature dimension (for reference / weight loading)
N_CROSS_FEATURES = 15   # Number of cross-asset features
N_DERIVATIVES_FEATURES = 7  # funding_rate_z, oi_change_pct, long_short_ratio,
                            # taker_buy_sell_ratio, has_derivatives_data,
                            # fear_greed_norm, fear_greed_roc
# Total real features: 46 single-pair + 15 cross-asset + 7 derivatives + 4 liquidation = 72, padded to 98

# Feature audit tracking — logs once per pair per process lifetime
_feature_audit_logged: set = set()

# Cache for raw (pre-standardization) features for LightGBM inference.
# build_feature_sequence stores the last bar's raw features here so
# predict_lightgbm can use them instead of the standardized version.
# See docs/ML_ACCURACY_INVESTIGATION.md Finding 1.
_lgb_raw_feature_cache: Dict[str, np.ndarray] = {}

# Momentum horizons matching training (train_lightgbm.py MOMENTUM_BARS)
_LGB_MOMENTUM_BARS = [1, 3, 6, 12, 24, 72]

# ── Cross-asset lead signal configuration ─────────────────────────────────────

_LEAD_SIGNALS_STATIC = {
    'BTC-USD':  {'primary': 'ETH-USD',  'secondary': 'SOL-USD'},
    'ETH-USD':  {'primary': 'BTC-USD',  'secondary': 'LINK-USD'},
    'SOL-USD':  {'primary': 'BTC-USD',  'secondary': 'ETH-USD'},
    'LINK-USD': {'primary': 'ETH-USD',  'secondary': 'BTC-USD'},
    'AVAX-USD': {'primary': 'ETH-USD',  'secondary': 'BTC-USD'},
    'DOGE-USD': {'primary': 'BTC-USD',  'secondary': 'ETH-USD'},
}

# Sector-based lead signals: BTC leads all; ETH leads L1/DeFi/infra
_SECTOR_LEADS = {
    # L1 / alt-L1 chains → led by ETH + BTC
    'SOL': ('ETH', 'BTC'), 'AVAX': ('ETH', 'BTC'), 'NEAR': ('ETH', 'BTC'),
    'ATOM': ('ETH', 'BTC'), 'DOT': ('ETH', 'BTC'), 'ADA': ('ETH', 'BTC'),
    'ALGO': ('ETH', 'BTC'), 'SEI': ('ETH', 'BTC'), 'SUI': ('ETH', 'BTC'),
    'APT': ('ETH', 'BTC'), 'INJ': ('ETH', 'BTC'), 'TIA': ('ETH', 'BTC'),
    'FTM': ('ETH', 'BTC'), 'MATIC': ('ETH', 'BTC'), 'HBAR': ('ETH', 'BTC'),
    'ICP': ('ETH', 'BTC'), 'FIL': ('ETH', 'BTC'), 'AR': ('ETH', 'BTC'),
    'STX': ('BTC', 'ETH'), 'KAS': ('BTC', 'ETH'),
    # DeFi → led by ETH
    'LINK': ('ETH', 'BTC'), 'UNI': ('ETH', 'BTC'), 'AAVE': ('ETH', 'BTC'),
    'MKR': ('ETH', 'BTC'), 'SNX': ('ETH', 'BTC'), 'CRV': ('ETH', 'BTC'),
    'DYDX': ('ETH', 'BTC'), 'PENDLE': ('ETH', 'BTC'), 'JUP': ('SOL', 'ETH'),
    # L2 → led by ETH
    'ARB': ('ETH', 'BTC'), 'OP': ('ETH', 'BTC'), 'IMX': ('ETH', 'BTC'),
    # Memes → led by BTC + DOGE
    'DOGE': ('BTC', 'ETH'), 'SHIB': ('DOGE', 'BTC'), 'PEPE': ('DOGE', 'BTC'),
    'WIF': ('SOL', 'BTC'), 'BONK': ('SOL', 'BTC'), 'FLOKI': ('DOGE', 'BTC'),
    # Gaming/NFT → led by ETH
    'SAND': ('ETH', 'BTC'), 'MANA': ('ETH', 'BTC'), 'GALA': ('ETH', 'BTC'),
    'AXS': ('ETH', 'BTC'), 'ENJ': ('ETH', 'BTC'), 'APE': ('ETH', 'BTC'),
    'IMX': ('ETH', 'BTC'), 'RNDR': ('ETH', 'BTC'),
    # AI → led by ETH
    'FET': ('ETH', 'BTC'), 'RNDR': ('ETH', 'BTC'), 'TAO': ('ETH', 'BTC'),
    'ARKM': ('ETH', 'BTC'),
    # Exchange tokens → led by BTC
    'BNB': ('BTC', 'ETH'),
    # Privacy/utility → led by BTC
    'XRP': ('BTC', 'ETH'), 'XLM': ('XRP', 'BTC'), 'LTC': ('BTC', 'ETH'),
    'BCH': ('BTC', 'ETH'), 'ETC': ('ETH', 'BTC'), 'TRX': ('BTC', 'ETH'),
    'RUNE': ('BTC', 'ETH'), 'GRT': ('ETH', 'BTC'), 'THETA': ('ETH', 'BTC'),
    'JASMY': ('BTC', 'ETH'), 'CHZ': ('ETH', 'BTC'), 'ENS': ('ETH', 'BTC'),
    'LDO': ('ETH', 'BTC'), 'WLD': ('ETH', 'BTC'), 'JTO': ('SOL', 'ETH'),
    'PYTH': ('SOL', 'ETH'), 'W': ('ETH', 'BTC'), 'STRK': ('ETH', 'BTC'),
    'ZRO': ('ETH', 'BTC'), 'EIGEN': ('ETH', 'BTC'), 'NTRN': ('ATOM', 'ETH'),
}


def _resolve_lead_signals(pair_name: str) -> Dict[str, str]:
    """Resolve lead signals for any pair, static map first then sector-based."""
    if pair_name in _LEAD_SIGNALS_STATIC:
        return _LEAD_SIGNALS_STATIC[pair_name]
    # Extract base symbol: 'SOL-USD' → 'SOL', 'BTCUSDT' → 'BTC'
    base = pair_name.split('-')[0].split('/')[0].replace('USDT', '').replace('USD', '')
    if base in _SECTOR_LEADS:
        pri, sec = _SECTOR_LEADS[base]
        # Convert to pair format matching cross_data keys (e.g. 'BTC-USD')
        return {'primary': f'{pri}-USD', 'secondary': f'{sec}-USD'}
    # Default: BTC primary, ETH secondary
    return {'primary': 'BTC-USD', 'secondary': 'ETH-USD'}


# Backward-compatible alias
LEAD_SIGNALS = _LEAD_SIGNALS_STATIC


# ── Prediction Debiaser ──────────────────────────────────────────────────────

class PredictionDebiaser:
    """EMA-based centering to remove persistent directional bias from ML predictions.

    Tracks a running EMA of each model's predictions and subtracts it,
    so a model that's consistently -0.01 gets shifted up by ~0.01.
    Alpha=0.01 -> ~100 sample half-life (conservative).
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self._ema: Dict[str, float] = {}  # model_name -> running EMA
        self._count: Dict[str, int] = {}  # model_name -> sample count
        self._warmup = 200  # Council S3: 200 for stable bias estimate (was 50)

    def debias(self, model_name: str, raw_prediction: float) -> float:
        """Apply debiasing to a raw prediction. Returns debiased value."""
        if model_name not in self._ema:
            self._ema[model_name] = 0.0
            self._count[model_name] = 0

        # Update EMA
        self._ema[model_name] = (
            self.alpha * raw_prediction + (1.0 - self.alpha) * self._ema[model_name]
        )
        self._count[model_name] = self._count.get(model_name, 0) + 1

        # Only debias after warmup period
        if self._count[model_name] < self._warmup:
            return raw_prediction

        return raw_prediction - self._ema[model_name]

    def get_bias(self, model_name: str) -> float:
        """Get current estimated bias for a model."""
        return self._ema.get(model_name, 0.0)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Return all model biases for dashboard."""
        return {
            name: {
                'bias': round(self._ema.get(name, 0.0), 6),
                'samples': self._count.get(name, 0),
                'warmed_up': self._count.get(name, 0) >= self._warmup,
            }
            for name in self._ema
        }


# ── Per-Model Z-Score Normalization ──────────────────────────────────────────

class ModelPredictionNormalizer:
    """Running z-score normalization to equalize model output scales.

    Models output on very different scales (LightGBM avg|pred|=0.168 vs DL 0.004-0.063).
    This normalizer applies EMA-based z-score normalization so all models contribute
    equally to the ensemble regardless of their raw output magnitude.
    Council S3 proposal #5.
    """

    def __init__(self, alpha: float = 0.05, min_samples: int = 200):
        self.alpha = alpha
        self.min_samples = min_samples
        self.stats: Dict[str, Dict[str, float]] = {}  # model_name -> {mean, var, count}

    def normalize(self, model_name: str, prediction: float) -> float:
        """Apply running z-score normalization to a prediction."""
        if model_name not in self.stats:
            self.stats[model_name] = {'mean': 0.0, 'var': 1.0, 'count': 0}

        s = self.stats[model_name]
        s['count'] += 1
        s['mean'] += self.alpha * (prediction - s['mean'])
        s['var'] += self.alpha * ((prediction - s['mean']) ** 2 - s['var'])

        if s['count'] < self.min_samples:
            return prediction  # Passthrough during warmup

        std = max(s['var'] ** 0.5, 1e-8)
        return (prediction - s['mean']) / std

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Return all model normalization stats for dashboard."""
        return {
            name: {
                'mean': round(s['mean'], 6),
                'std': round(max(s['var'] ** 0.5, 1e-8), 6),
                'samples': int(s['count']),
                'warmed_up': s['count'] >= self.min_samples,
            }
            for name, s in self.stats.items()
        }


# Module-level singletons — shared across all predict_with_models calls
_debiaser = PredictionDebiaser(alpha=0.01)
_normalizer = ModelPredictionNormalizer(alpha=0.05, min_samples=200)


# ── Attention (matching trained weights) ──────────────────────────────────────

class _TrainedAttention(nn.Module):
    """Attention matching the trained weight structure.

    Saved keys per block:
      attention.attention_temperature  (1,)
      attention.quantum_enhancement_scale  (n_heads,)
      attention.w_q.weight/bias  (qkv_dim, d_model)
      attention.w_k.weight/bias
      attention.w_v.weight/bias
      attention.w_o.weight/bias  (d_model, qkv_dim)
    """

    def __init__(self, d_model: int, n_heads: int, qkv_dim: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = qkv_dim // n_heads

        self.w_q = nn.Linear(d_model, qkv_dim)
        self.w_k = nn.Linear(d_model, qkv_dim)
        self.w_v = nn.Linear(d_model, qkv_dim)
        self.w_o = nn.Linear(qkv_dim, d_model)

        self.attention_temperature = nn.Parameter(torch.ones(1))
        self.quantum_enhancement_scale = nn.Parameter(torch.ones(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.w_q(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head) * self.attention_temperature
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(scores, dim=-1)

        # Per-head quantum enhancement
        enhancement = self.quantum_enhancement_scale.view(1, self.n_heads, 1, 1)
        attn = attn * enhancement

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.w_o(out)


# ── Positional Encoding (matching saved keys: pos_encoding.pe, pos_encoding.quantum_phase) ──

class _TrainedPosEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        self.quantum_phase = nn.Parameter(torch.zeros(d_model))
        pe = torch.zeros(1, max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :] * torch.cos(self.quantum_phase)


# ── Transformer Block (matching trained keys) ────────────────────────────────

class _TrainedTransformerBlock(nn.Module):
    """Saved keys per block:
      attention.*
      feed_forward.0.weight/bias  (d_ff, d_model)
      feed_forward.3.weight/bias  (d_model, d_ff)
      norm1.weight/bias  (d_model,)
      norm2.weight/bias  (d_model,)
      skip_enhancement  (1,)
    """

    def __init__(self, d_model: int, n_heads: int, qkv_dim: int, d_ff: int, dropout: float = 0.2):
        super().__init__()
        self.attention = _TrainedAttention(d_model, n_heads, qkv_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.skip_enhancement = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x)
        x = self.norm1(x + self.skip_enhancement * attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: Quantum Transformer  (best_quantum_transformer_model.pth — 92 keys)
# ══════════════════════════════════════════════════════════════════════════════

class TrainedQuantumTransformer(nn.Module):
    """Architecture matching saved weights exactly.

    input_dim=98, hidden=288, 4 blocks, 8 heads (qkv_dim=328), d_ff=1315
    Single output_head → scalar prediction.
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        d_model, n_heads, qkv_dim, d_ff, n_blocks = 288, 8, 328, 1315, 4

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = _TrainedPosEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            _TrainedTransformerBlock(d_model, n_heads, qkv_dim, d_ff)
            for _ in range(n_blocks)
        ])
        # output_head: BN(288) → Linear(288,144) → ... → Linear(72,1) → Tanh
        self.output_head = nn.Sequential(
            nn.BatchNorm1d(d_model),        # 0
            nn.GELU(),                      # 1
            nn.Linear(d_model, 144),        # 2
            nn.GELU(),                      # 3
            nn.Dropout(0.2),                # 4
            nn.Linear(144, 72),             # 5
            nn.GELU(),                      # 6
            nn.Linear(72, 1),               # 7
            nn.Tanh(),                      # 8  v7: bound output to [-1, 1]
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, 72),         # 0
            nn.ReLU(),                      # 1
            nn.Linear(72, 1),              # 2
            nn.Softplus(),                  # 3
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, seq_len, input_dim) → prediction (batch, 1), uncertainty (batch, 1)"""
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        pooled = x.mean(dim=1)  # (batch, d_model)
        pred = self.output_head(pooled)
        unc = self.uncertainty_head(pooled)
        return pred, unc


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: Bidirectional LSTM  (best_bidirectional_lstm_model.pth — 44 keys)
# ══════════════════════════════════════════════════════════════════════════════

class _TrainedLSTMCore(nn.Module):
    """Matches saved keys: lstm.lstm_layers, lstm.skip_projections, lstm.consciousness_gates"""

    def __init__(self, input_size: int = INPUT_DIM, hidden_size: int = 292, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        bidir_dim = hidden_size * 2  # 584

        self.lstm_layers = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        self.consciousness_gates = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_size if i == 0 else bidir_dim
            self.lstm_layers.append(nn.LSTM(
                input_size=in_dim, hidden_size=hidden_size,
                num_layers=1, batch_first=True, bidirectional=True,
            ))
            self.skip_projections.append(nn.Linear(in_dim, bidir_dim))
            self.consciousness_gates.append(nn.Sequential(
                nn.Linear(bidir_dim, bidir_dim),
                nn.Sigmoid(),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lstm_layer in enumerate(self.lstm_layers):
            skip = self.skip_projections[i](x)
            out, _ = lstm_layer(x)
            gate = self.consciousness_gates[i](out)
            x = gate * out + (1 - gate) * skip
        return x


class TrainedBidirectionalLSTM(nn.Module):
    """Architecture matching saved weights exactly.

    input=98, hidden=292 per direction (584 total), 2 LSTM layers.
    prediction_head: BN(584)→Linear→BN→Linear→Linear→1
    confidence_head: Linear(584,73)→Linear(73,1)
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        bidir_dim = 584  # 292 * 2

        self.lstm = _TrainedLSTMCore(input_size=input_dim, hidden_size=292, num_layers=2)

        # prediction_head: indices 0=BN, 2=Linear(584,292), 4=BN, 6=Linear(292,146), 8=Linear(146,1), 9=Tanh
        self.prediction_head = nn.Sequential(
            nn.BatchNorm1d(bidir_dim),      # 0
            nn.GELU(),                      # 1
            nn.Linear(bidir_dim, 292),      # 2
            nn.GELU(),                      # 3
            nn.BatchNorm1d(292),            # 4
            nn.GELU(),                      # 5
            nn.Linear(292, 146),            # 6
            nn.GELU(),                      # 7
            nn.Linear(146, 1),              # 8
            nn.Tanh(),                      # 9  v7: bound output to [-1, 1]
        )

        # confidence_head: Linear(584,73) → ReLU → Linear(73,1) → Sigmoid
        self.confidence_head = nn.Sequential(
            nn.Linear(bidir_dim, 73),       # 0
            nn.ReLU(),                      # 1
            nn.Linear(73, 1),              # 2
            nn.Sigmoid(),                   # 3
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, seq_len, input_dim) → prediction (batch, 1), confidence (batch, 1)"""
        lstm_out = self.lstm(x)
        pooled = lstm_out.mean(dim=1)  # (batch, 584)
        pred = self.prediction_head(pooled)
        conf = self.confidence_head(pooled)
        return pred, conf


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: Dilated CNN  (best_dilated_cnn_model.pth — 113 keys)
# ══════════════════════════════════════════════════════════════════════════════

class _TrainedDilatedBlock(nn.Module):
    """Single dilated conv block: Conv1d(ch, ch, 3, dilation) + BN + ReLU + Conv1d(ch, ch, 1) + BN + residual

    Saved keys use flat numeric indices (conv_blocks.0.0, conv_blocks.0.1, etc.)
    so layers are registered directly via add_module to match.
    """

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        # Register layers with numeric names to match saved key pattern
        self.add_module('0', nn.Conv1d(channels, channels, kernel_size=3,
                                       dilation=dilation, padding=dilation))
        self.add_module('1', nn.BatchNorm1d(channels))
        # indices 2=ReLU, 3=Dropout are functional (no params)
        self.add_module('4', nn.Conv1d(channels, channels, kernel_size=1))
        self.add_module('5', nn.BatchNorm1d(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = getattr(self, '0')(x)
        h = getattr(self, '1')(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = getattr(self, '4')(h)
        h = getattr(self, '5')(h)
        return F.relu(h + x)


class _TrainedDilatedCNNCore(nn.Module):
    """Matches saved keys: dilated_cnn.conv_blocks, dilated_cnn.fusion, dilated_cnn.attention_pool"""

    def __init__(self, channels: int = INPUT_DIM, hidden: int = 332, n_blocks: int = 5):
        super().__init__()
        # 5 dilated conv blocks (channels=83)
        self.conv_blocks = nn.ModuleList([
            _TrainedDilatedBlock(channels, dilation=2 ** i)
            for i in range(n_blocks)
        ])
        # fusion: Conv1d(channels*n_blocks + channels, hidden) → BN → ReLU → Drop → Conv1d(hidden, hidden) → BN
        fusion_in = channels * n_blocks + channels  # 83*5 + 83 = 498? Let me check
        # Actually: from saved weights, fusion.0.weight: (332, 415, 1)
        # So fusion_in = 415 = 83*5 = 415. That means concat of 5 block outputs.
        # Wait: 83*5 = 415. So it's just 5 blocks concatenated.
        fusion_in = channels * n_blocks  # 415
        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_in, hidden, kernel_size=1),    # 0
            nn.BatchNorm1d(hidden),                         # 1
            nn.ReLU(),                                      # 2
            nn.Dropout(0.2),                                # 3
            nn.Conv1d(hidden, hidden, kernel_size=1),       # 4
            nn.BatchNorm1d(hidden),                         # 5
        )
        # attention_pool: Softmax → Conv1d(channels, hidden) → Conv1d(hidden, channels)
        # saved: attention_pool.1.weight: (83, 332, 1), attention_pool.3.weight: (332, 83, 1)
        self.attention_pool = nn.Sequential(
            nn.Softmax(dim=-1),                             # 0
            nn.Conv1d(hidden, channels, kernel_size=1),     # 1
            nn.ReLU(),                                      # 2
            nn.Conv1d(channels, hidden, kernel_size=1),     # 3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels=83, seq_len)"""
        block_outputs = []
        h = x
        for block in self.conv_blocks:
            h = block(h)
            block_outputs.append(h)
        # Concatenate all block outputs
        cat = torch.cat(block_outputs, dim=1)  # (batch, 415, seq_len)
        fused = self.fusion(cat)               # (batch, 332, seq_len)
        fused = F.relu(fused)
        # Attention pooling
        attn = self.attention_pool(fused)       # (batch, 332, seq_len)
        pooled = (fused * F.softmax(attn, dim=-1)).sum(dim=-1)  # (batch, 332)
        return pooled


class TrainedDilatedCNN(nn.Module):
    """Architecture matching saved weights exactly.

    channels=input_dim, 5 dilated blocks, hidden=332.
    classifier: BN(332)→Linear(332,166)→BN(166)→...→Linear(input_dim,1)
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        hidden = 332

        self.dilated_cnn = _TrainedDilatedCNNCore(channels=input_dim, hidden=hidden, n_blocks=5)

        # classifier: BN(332)→Linear→BN→...→Linear(input_dim, 1)→Tanh
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden),         # 0
            nn.GELU(),                      # 1
            nn.Linear(hidden, 166),         # 2
            nn.BatchNorm1d(166),            # 3
            nn.GELU(),                      # 4
            nn.Dropout(0.2),                # 5
            nn.Linear(166, input_dim),      # 6
            nn.BatchNorm1d(input_dim),      # 7
            nn.GELU(),                      # 8
            nn.Linear(input_dim, 1),        # 9
            nn.Tanh(),                      # 10 v7: bound output to [-1, 1]
        )

        # pattern_strength: Linear(332→41) → ReLU → Linear(41→1)
        self.pattern_strength = nn.Sequential(
            nn.Linear(hidden, 41),          # 0
            nn.ReLU(),                      # 1
            nn.Linear(41, 1),              # 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → prediction (batch, 1)

        Transposes to (batch, input_dim, seq_len) for conv1d processing.
        """
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        pooled = self.dilated_cnn(x)  # (batch, 332)
        return self.classifier(pooled)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4: Simple CNN  (best_cnn_model.pth — fresh architecture for 83 features)
# ══════════════════════════════════════════════════════════════════════════════

class TrainedCNN(nn.Module):
    """Simple Conv1d model for input_dim-feature input.

    4 conv layers with increasing then decreasing channels,
    global average pooling, and a small classifier head.
    Input: (batch, seq_len, input_dim)  Output: (batch, 1)
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Tanh(),                      # v7: bound output to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → prediction (batch, 1)"""
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.conv_layers(x)  # (batch, 64, seq_len)
        x = x.mean(dim=-1)  # global avg pool → (batch, 64)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 5: Bidirectional GRU  (best_gru_model.pth — fresh architecture)
# ══════════════════════════════════════════════════════════════════════════════

class TrainedGRU(nn.Module):
    """Bidirectional GRU for input_dim-feature input.

    2-layer BiGRU with hidden_size=134 (268 total), plus a classifier head.
    Input: (batch, seq_len, input_dim)  Output: (batch, 1)
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=134,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        bidir_dim = 268  # 134 * 2
        self.classifier = nn.Sequential(
            nn.Linear(bidir_dim, 134),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(134, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),                      # v7: bound output to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → prediction (batch, 1)"""
        out, _ = self.gru(x)  # (batch, seq_len, 268)
        pooled = out.mean(dim=1)  # (batch, 268)
        return self.classifier(pooled)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 6: Meta-Ensemble  (stacking layer over 5 base models)
# ══════════════════════════════════════════════════════════════════════════════

N_BASE_MODELS = 6  # QT, BiLSTM, DilatedCNN, CNN, GRU, LightGBM
BASE_MODEL_NAMES = [
    'quantum_transformer', 'bidirectional_lstm', 'dilated_cnn', 'cnn', 'gru', 'lightgbm',
]


class TrainedMetaEnsemble(nn.Module):
    """Meta-learning stacking layer that learns which base models to trust.

    Takes input_dim-dim market features + N base model predictions as input.
    Learns context-dependent model weighting: in trending markets, trust
    momentum models more; in choppy markets, trust mean-reversion models.

    Input: (batch, input_dim + n_models)
    Output: (prediction, confidence) tuple

    The n_models parameter auto-detects from saved weights when loading,
    supporting both 5-model (legacy) and 6-model (with LightGBM) ensembles.
    """

    def __init__(self, input_dim: int = INPUT_DIM, n_models: int = N_BASE_MODELS):
        super().__init__()
        self._input_dim = input_dim
        self._n_models = n_models

        # Feature extractor — understands market context
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),  # 0
            nn.BatchNorm1d(128),        # 1
            nn.GELU(),                  # 2
            nn.Dropout(0.2),            # 3
            nn.Linear(128, 64),         # 4
            nn.BatchNorm1d(64),         # 5
            nn.GELU(),                  # 6
        )

        # Weight generator — context-dependent per-model attention weights
        self.weight_generator = nn.Sequential(
            nn.Linear(64, 32),          # 0
            nn.GELU(),                  # 1
            nn.Linear(32, n_models),    # 2  → n_models weights
        )

        # Final predictor — combines context + base predictions
        self.final_predictor = nn.Sequential(
            nn.Linear(64 + n_models, 32),  # 0
            nn.GELU(),                     # 1
            nn.Dropout(0.1),               # 2
            nn.Linear(32, 1),              # 3
            nn.Tanh(),                     # 4  v7: bound output to [-1, 1]
        )

        # Confidence estimator — how sure is the ensemble
        self.confidence_estimator = nn.Sequential(
            nn.Linear(64 + n_models, 16),  # 0
            nn.ReLU(),                     # 1
            nn.Linear(16, 1),              # 2
            nn.Sigmoid(),                  # 3
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, input_dim + n_models) → prediction (batch, 1), confidence (batch, 1)"""
        features = x[:, :self._input_dim]
        model_preds = x[:, self._input_dim:]  # (batch, n_models)

        # Extract market context
        ctx = self.feature_extractor(features)  # (batch, 64)

        # Generate per-model attention weights
        weights = F.softmax(self.weight_generator(ctx), dim=-1)  # (batch, n_models)

        # Combine context + raw model predictions
        combined = torch.cat([ctx, model_preds], dim=-1)  # (batch, 64 + n_models)

        # Final prediction (learns both from weighted combo and direct features)
        pred = self.final_predictor(combined)  # (batch, 1)
        conf = self.confidence_estimator(combined)  # (batch, 1)

        return pred, conf


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — 46 scale-invariant single-pair + 15 cross-asset features
# No raw prices or raw volumes — all returns, ratios, z-scores, bounded indicators.
# Ensures identical feature distributions whether BTC is $3K or $130K.
# ══════════════════════════════════════════════════════════════════════════════


def _compute_single_pair_features(
    df: 'pd.DataFrame',
) -> Dict[str, 'pd.Series']:
    """Compute 46 scale-invariant single-pair features from OHLCV data.

    CRITICAL: No feature depends on absolute price level or raw volume.
    Every feature is a return, ratio, z-score, or bounded indicator.
    This ensures the model sees identical feature distributions whether
    BTC is $3K or $130K.

    Returns:
        Dict of feature_name → pd.Series (46 features)
    """
    close = df['close'].astype(float)
    _open = df['open'].astype(float) if 'open' in df.columns else close
    high = df['high'].astype(float) if 'high' in df.columns else close
    low = df['low'].astype(float) if 'low' in df.columns else close
    vol = df['volume'].astype(float) if 'volume' in df.columns else None

    features: Dict[str, 'pd.Series'] = {}

    # ── Group 1: Candle shape (5 features) — replaces raw OHLCV ──────────
    features['open_gap'] = np.log(_open / (close.shift(1) + 1e-10))
    features['upper_wick'] = (high - np.maximum(_open, close)) / (close + 1e-10)
    features['lower_wick'] = (np.minimum(_open, close) - low) / (close + 1e-10)
    features['body'] = (close - _open) / (close + 1e-10)
    # Volume: rolling z-score (100-bar window)
    if vol is not None:
        vol_mean = vol.rolling(100, min_periods=10).mean()
        vol_std = vol.rolling(100, min_periods=10).std()
        features['volume_z'] = (vol - vol_mean) / (vol_std + 1e-10)
    else:
        features['volume_z'] = close * 0.0

    # ── Group 2: Returns at multiple horizons (7 features) ────────────────
    for w in [1, 2, 3, 5, 10, 20]:
        features[f'ret_{w}'] = close.pct_change(w)
    features['log_ret'] = np.log(close / close.shift(1))

    # ── Group 3: SMA distance + slope (8 features) — replaces raw SMA ────
    for w in [5, 10, 20, 50]:
        sma = close.rolling(w).mean()
        features[f'sma_dist_{w}'] = (close - sma) / (sma + 1e-10)
        features[f'sma_slope_{w}'] = sma.pct_change(3)

    # ── Group 4: EMA distance + slope (6 features) — replaces raw EMA ────
    for w in [5, 10, 20]:
        ema = close.ewm(span=w, adjust=False).mean()
        features[f'ema_dist_{w}'] = (close - ema) / (ema + 1e-10)
        features[f'ema_slope_{w}'] = ema.pct_change(3)

    # ── Group 5: Realized volatility (3 features) ────────────────────────
    pct_ret = close.pct_change()
    for w in [5, 10, 20]:
        features[f'vol_{w}'] = pct_ret.rolling(w).std()

    # ── Group 6: RSI, rescaled to [-1, 1] (1 feature) ────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss_s = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss_s + 1e-10)
    features['rsi_norm'] = (100 - (100 / (1 + rs)) - 50) / 50

    # ── Group 7: MACD normalized by price (3 features) ────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    features['macd_pct'] = macd / (close + 1e-10)
    features['macd_signal_pct'] = macd_signal / (close + 1e-10)
    features['macd_hist_pct'] = (macd - macd_signal) / (close + 1e-10)

    # ── Group 8: Bollinger Bands — z-score + width (4 features) ──────────
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = bb_upper - bb_lower + 1e-10
    features['bb_pct'] = (close - bb_lower) / bb_range
    features['bb_width'] = bb_range / (sma20 + 1e-10)
    features['bb_upper_dist'] = (bb_upper - close) / (close + 1e-10)
    features['bb_lower_dist'] = (close - bb_lower) / (close + 1e-10)

    # ── Group 9: ATR as % of price (1 feature) ───────────────────────────
    import pandas as pd
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    features['atr_pct'] = tr.rolling(14).mean() / (close + 1e-10)

    # ── Group 10: Volume ratios (3 features) ─────────────────────────────
    if vol is not None:
        features['vol_ratio'] = vol / (vol.rolling(10, min_periods=1).mean() + 1e-10)
        features['vol_change'] = vol.pct_change()
        features['vol_trend'] = (vol.rolling(5, min_periods=1).mean()
                                 / (vol.rolling(20, min_periods=1).mean() + 1e-10))
    else:
        features['vol_ratio'] = close * 0.0
        features['vol_change'] = close * 0.0
        features['vol_trend'] = close * 0.0 + 1.0

    # ── Group 11: Price momentum (3 features) ────────────────────────────
    features['momentum_5'] = close / close.shift(5) - 1
    features['momentum_10'] = close / close.shift(10) - 1
    features['momentum_20'] = close / close.shift(20) - 1

    # ── Group 12: Range features (2 features) ────────────────────────────
    features['hl_range'] = (high - low) / (close + 1e-10)
    hl_range = features['hl_range']
    features['hl_range_norm'] = hl_range / (hl_range.rolling(10, min_periods=1).mean() + 1e-10)

    # Total: 5 + 7 + 8 + 6 + 3 + 1 + 3 + 4 + 1 + 3 + 3 + 2 = 46
    return features


def _build_cross_features(
    close: 'pd.Series',
    volume: 'pd.Series',
    cross_data: Dict[str, 'pd.DataFrame'],
    pair_name: str,
) -> Dict[str, 'pd.Series']:
    """Compute 15 cross-asset features from other pairs' data.

    Features are computed using only past data (no future leakage).

    Args:
        close: This pair's close price series (aligned index with cross_data)
        volume: This pair's volume series
        cross_data: Dict of pair_name → DataFrame with at least [close, volume] columns
        pair_name: This pair's name (e.g. 'BTC-USD')

    Returns:
        Dict of feature_name → pd.Series (15 features)
    """
    import pandas as pd

    feats: Dict[str, pd.Series] = {}
    log_ret = np.log(close / close.shift(1))

    # ── Lead signals (5 features) ─────────────────────────────────────────
    lead_cfg = _resolve_lead_signals(pair_name)
    primary_pair = lead_cfg.get('primary')
    secondary_pair = lead_cfg.get('secondary')

    for role, leader_pair, horizons in [
        ('primary', primary_pair, [1, 3, 6]),
        ('secondary', secondary_pair, [1, 3]),
    ]:
        if leader_pair and leader_pair in cross_data:
            leader_close = cross_data[leader_pair]['close'].astype(float)
            leader_log_ret = np.log(leader_close / leader_close.shift(1))
            for h in horizons:
                # h-bar log return of the leader
                feat_name = f'lead_{role}_ret_{h}'
                feats[feat_name] = leader_log_ret.rolling(h).sum()
        else:
            # Fill zeros for missing leader
            for h in ([1, 3, 6] if role == 'primary' else [1, 3]):
                feats[f'lead_{role}_ret_{h}'] = pd.Series(0.0, index=close.index)

    # ── Cross-pair correlations (4 features) ──────────────────────────────
    for ref_name, ref_label in [('BTC-USD', 'btc'), ('ETH-USD', 'eth')]:
        if ref_name in cross_data and ref_name != pair_name:
            ref_close = cross_data[ref_name]['close'].astype(float)
            ref_log_ret = np.log(ref_close / ref_close.shift(1))
            corr_50 = log_ret.rolling(50).corr(ref_log_ret)
            feats[f'corr_{ref_label}_50'] = corr_50
            # Z-score of correlation over 200-bar window
            corr_mean = corr_50.rolling(200).mean()
            corr_std = corr_50.rolling(200).std()
            feats[f'corr_z_{ref_label}'] = (corr_50 - corr_mean) / (corr_std + 1e-10)
        else:
            feats[f'corr_{ref_label}_50'] = pd.Series(0.0, index=close.index)
            feats[f'corr_z_{ref_label}'] = pd.Series(0.0, index=close.index)

    # ── Spread features (2 features) ──────────────────────────────────────
    for ref_name, ref_label in [('BTC-USD', 'btc'), ('ETH-USD', 'eth')]:
        if ref_name in cross_data and ref_name != pair_name:
            ref_close = cross_data[ref_name]['close'].astype(float)
            log_spread = np.log(close / (ref_close + 1e-10))
            spread_mean = log_spread.rolling(100).mean()
            spread_std = log_spread.rolling(100).std()
            feats[f'spread_{ref_label}_z'] = (log_spread - spread_mean) / (spread_std + 1e-10)
        else:
            feats[f'spread_{ref_label}_z'] = pd.Series(0.0, index=close.index)

    # ── Market-wide features (4 features, same for all pairs) ─────────────
    all_rets = []
    all_vol_zs = []
    for p, cdf in cross_data.items():
        c = cdf['close'].astype(float)
        r = np.log(c / c.shift(1))
        all_rets.append(r)
        if 'volume' in cdf.columns:
            v = cdf['volume'].astype(float)
            vol_ma = v.rolling(20).mean()
            all_vol_zs.append(v / (vol_ma + 1e-10) - 1.0)

    # Also include self
    all_rets.append(log_ret)
    if volume is not None:
        vol_ma_self = volume.rolling(20).mean()
        all_vol_zs.append(volume / (vol_ma_self + 1e-10) - 1.0)

    if all_rets:
        ret_df = pd.concat(all_rets, axis=1)
        feats['mkt_avg_ret'] = ret_df.mean(axis=1)
        feats['mkt_dispersion'] = ret_df.std(axis=1)
        feats['mkt_breadth'] = (ret_df > 0).mean(axis=1)
    else:
        feats['mkt_avg_ret'] = pd.Series(0.0, index=close.index)
        feats['mkt_dispersion'] = pd.Series(0.0, index=close.index)
        feats['mkt_breadth'] = pd.Series(0.5, index=close.index)

    if all_vol_zs:
        vol_df = pd.concat(all_vol_zs, axis=1)
        feats['mkt_avg_vol_z'] = vol_df.mean(axis=1)
    else:
        feats['mkt_avg_vol_z'] = pd.Series(0.0, index=close.index)

    return feats


def _build_derivatives_features(
    n_rows: int,
    derivatives_data: Optional[Dict[str, 'pd.Series']] = None,
    df_index=None,
) -> Dict[str, 'pd.Series']:
    """Compute 7 derivatives + sentiment features.

    Args:
        n_rows: Number of rows in the price DataFrame (for index alignment)
        derivatives_data: Dict with optional keys:
            - 'funding_rate': pd.Series of raw funding rates
            - 'open_interest': pd.Series of open interest values
            - 'long_short_ratio': pd.Series of long/short account ratios
            - 'taker_buy_vol': pd.Series of taker buy volume
            - 'taker_sell_vol': pd.Series of taker sell volume
            - 'fear_greed': pd.Series of Fear & Greed index (0-100)
            Series can be shorter than n_rows — they'll be right-aligned
            (most recent at end) and NaN-padded at the start.
        df_index: The index of the price DataFrame — ensures alignment.

    Returns:
        Dict of 7 feature_name → pd.Series
    """
    import pandas as pd

    idx = df_index if df_index is not None else pd.RangeIndex(n_rows)
    feats: Dict[str, pd.Series] = {}

    has_deriv = False

    def _right_align(raw_series: 'pd.Series') -> 'pd.Series':
        """Right-align a shorter Series to the target index.

        Extracts raw values, pads with NaN at the start if shorter than
        n_rows, then assigns the target index so DataFrame alignment works.
        """
        vals = np.array(raw_series.values, dtype=float)
        n = len(vals)
        if n >= n_rows:
            aligned = vals[-n_rows:]
        else:
            aligned = np.concatenate([np.full(n_rows - n, np.nan), vals])
        return pd.Series(aligned, index=idx)

    if derivatives_data is not None:
        # ── Funding rate z-score (50-bar window) ────────────────────────
        fr = derivatives_data.get('funding_rate')
        if fr is not None and len(fr) > 0:
            fr = _right_align(fr)
            fr_mean = fr.rolling(50, min_periods=5).mean()
            fr_std = fr.rolling(50, min_periods=5).std()
            feats['funding_rate_z'] = (fr - fr_mean) / (fr_std + 1e-10)
            has_deriv = True
        else:
            feats['funding_rate_z'] = pd.Series(0.0, index=idx)

        # ── Open interest 5-bar % change ────────────────────────────────
        oi = derivatives_data.get('open_interest')
        if oi is not None and len(oi) > 0:
            oi = _right_align(oi)
            feats['oi_change_pct'] = oi.pct_change(5)
            has_deriv = True
        else:
            feats['oi_change_pct'] = pd.Series(0.0, index=idx)

        # ── Long/short ratio (raw, already scale-invariant) ─────────────
        ls = derivatives_data.get('long_short_ratio')
        if ls is not None and len(ls) > 0:
            feats['long_short_ratio'] = _right_align(ls)
            has_deriv = True
        else:
            feats['long_short_ratio'] = pd.Series(0.0, index=idx)

        # ── Taker buy/sell ratio ────────────────────────────────────────
        buy_vol = derivatives_data.get('taker_buy_vol')
        sell_vol = derivatives_data.get('taker_sell_vol')
        if buy_vol is not None and sell_vol is not None and len(buy_vol) > 0:
            bv = _right_align(buy_vol)
            sv = _right_align(sell_vol)
            feats['taker_buy_sell_ratio'] = bv / (sv + 1e-10)
            has_deriv = True
        else:
            feats['taker_buy_sell_ratio'] = pd.Series(0.0, index=idx)

        # ── Fear & Greed (normalized + 3-day ROC) ──────────────────────
        fg = derivatives_data.get('fear_greed')
        if fg is not None and len(fg) > 0:
            fg = _right_align(fg)
            feats['fear_greed_norm'] = fg / 100.0
            # 3-day ROC: for 5-min bars, 3 days = 3 * 288 = 864 bars
            # But FnG is daily (forward-filled), so diff(864) gives ~3-day change
            feats['fear_greed_roc'] = fg.diff(864) / 100.0
        else:
            feats['fear_greed_norm'] = pd.Series(0.0, index=idx)
            feats['fear_greed_roc'] = pd.Series(0.0, index=idx)

    else:
        # No derivatives data at all — fill zeros
        feats['funding_rate_z'] = pd.Series(0.0, index=idx)
        feats['oi_change_pct'] = pd.Series(0.0, index=idx)
        feats['long_short_ratio'] = pd.Series(0.0, index=idx)
        feats['taker_buy_sell_ratio'] = pd.Series(0.0, index=idx)
        feats['fear_greed_norm'] = pd.Series(0.0, index=idx)
        feats['fear_greed_roc'] = pd.Series(0.0, index=idx)

    # ── Binary flag: model knows when derivatives data is present ────
    feats['has_derivatives_data'] = pd.Series(
        1.0 if has_deriv else 0.0, index=idx
    )

    # ── 4 liquidation features (use padding slots, no INPUT_DIM change) ────
    # These are scalar values from the real-time liquidation feed, broadcast
    # across all rows (they represent the CURRENT state, not historical).
    if derivatives_data is not None:
        liq_long = float(derivatives_data.get('liq_long_usd_5m', 0) or 0)
        liq_short = float(derivatives_data.get('liq_short_usd_5m', 0) or 0)
        liq_total = liq_long + liq_short

        import math
        feats['liq_long_volume_z'] = pd.Series(
            math.log1p(liq_long / 100_000), index=idx
        )
        feats['liq_short_volume_z'] = pd.Series(
            math.log1p(liq_short / 100_000), index=idx
        )
        feats['liq_imbalance'] = pd.Series(
            (liq_long - liq_short) / liq_total if liq_total > 0 else 0.0,
            index=idx,
        )
        feats['liq_cascade_active'] = pd.Series(
            1.0 if derivatives_data.get('liq_cascade_active') else 0.0,
            index=idx,
        )
    else:
        feats['liq_long_volume_z'] = pd.Series(0.0, index=idx)
        feats['liq_short_volume_z'] = pd.Series(0.0, index=idx)
        feats['liq_imbalance'] = pd.Series(0.0, index=idx)
        feats['liq_cascade_active'] = pd.Series(0.0, index=idx)

    return feats


def build_feature_sequence(
    price_df,
    seq_len: int = 30,
    cross_data: Optional[Dict[str, 'pd.DataFrame']] = None,
    pair_name: Optional[str] = None,
    derivatives_data: Optional[Dict[str, 'pd.Series']] = None,
) -> Optional[np.ndarray]:
    """Build a (seq_len, INPUT_DIM) feature matrix from a price DataFrame.

    All features are scale-invariant: returns, ratios, z-scores, bounded
    indicators.  No raw prices or raw volumes.  Per-window standardization
    is applied on top so every sample has zero mean / unit variance.

    Args:
        price_df: DataFrame with OHLCV columns
        seq_len: Number of time steps in the output sequence
        cross_data: Optional dict of pair_name → DataFrame with [close, volume] columns.
        pair_name: This pair's identifier (e.g. 'BTC-USD').
        derivatives_data: Optional dict of feature_name → pd.Series for derivatives
            features (funding_rate, open_interest, long_short_ratio, taker_buy_vol,
            taker_sell_vol, fear_greed). Series must be aligned to price_df index.

    Returns:
        (seq_len, INPUT_DIM) float32 array, or None if insufficient data
    """
    import pandas as pd

    if price_df is None or len(price_df) < seq_len:
        return None

    # Need 250+ rows for rolling-window features (corr_z needs rolling(50).corr
    # then rolling(200).mean = 250 bars).  seq_len + 270 → 300 rows.
    df = price_df.tail(seq_len + 270).copy()

    # Trim derivatives_data to match
    if derivatives_data is not None:
        n_tail = len(df)
        derivatives_data = {
            k: v.tail(n_tail).reset_index(drop=True) if hasattr(v, 'tail') else v
            for k, v in derivatives_data.items()
        }

    if cross_data is not None:
        _cross = {}
        for p, cdf in cross_data.items():
            if p == pair_name:
                continue
            _cross[p] = cdf.tail(seq_len + 270).copy().reset_index(drop=True)
        cross_data = _cross if _cross else None

    df = df.reset_index(drop=True)
    close = df['close'].astype(float) if 'close' in df.columns else None
    if close is None:
        return None
    vol = df['volume'].astype(float) if 'volume' in df.columns else None

    # 46 scale-invariant single-pair features
    features = _compute_single_pair_features(df)

    # 15 cross-asset features (already scale-invariant)
    if cross_data is not None and pair_name is not None:
        cross_feats = _build_cross_features(close, vol, cross_data, pair_name)
        features.update(cross_feats)

    # 7 derivatives + sentiment features
    deriv_feats = _build_derivatives_features(len(df), derivatives_data, df_index=df.index)
    features.update(deriv_feats)

    feat_df = pd.DataFrame(features, index=df.index)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    # ── Feature audit (once per pair) ────────────────────────────────────────
    _pair_key = pair_name or "unknown"
    if _pair_key not in _feature_audit_logged:
        _feature_audit_logged.add(_pair_key)
        n_total = len(feat_df.columns)
        active_cols = [c for c in feat_df.columns if feat_df[c].abs().sum() > 0]
        zero_cols = [c for c in feat_df.columns if c not in active_cols]
        logger.info(
            f"[FEATURE AUDIT] {_pair_key}: {len(active_cols)}/{n_total} active, "
            f"{INPUT_DIM - n_total} zero-padded | "
            f"zeros: {zero_cols[:10]}{'...' if len(zero_cols) > 10 else ''}"
        )

    # ── Feature health diagnostic (throttled: every 100 calls) ────────────
    if not hasattr(build_feature_sequence, '_health_counter'):
        build_feature_sequence._health_counter = 0
    build_feature_sequence._health_counter += 1
    if build_feature_sequence._health_counter % 100 == 1:
        last_row = feat_df.iloc[-1]
        cross_names = [c for c in feat_df.columns if c.startswith(('lead_', 'corr_', 'spread_', 'mkt_'))]
        deriv_names = [c for c in feat_df.columns if c.startswith(('funding_', 'oi_', 'taker_', 'fear_', 'long_short', 'has_deriv'))]
        cross_vals = {c: round(float(last_row.get(c, 0)), 4) for c in cross_names}
        deriv_vals = {c: round(float(last_row.get(c, 0)), 4) for c in deriv_names}
        cross_active = sum(1 for v in cross_vals.values() if abs(v) > 1e-8)
        deriv_active = sum(1 for v in deriv_vals.values() if abs(v) > 1e-8)
        logger.info(
            f"[FEATURE HEALTH] {_pair_key} (call #{build_feature_sequence._health_counter}): "
            f"cross={cross_active}/{len(cross_names)} deriv={deriv_active}/{len(deriv_names)} | "
            f"rows={len(feat_df)} | "
            f"cross={cross_vals} | deriv={deriv_vals}"
        )

    # Take last seq_len rows
    feat_arr = feat_df.tail(seq_len).values.astype(np.float32)

    # Cache raw (pre-standardization) last bar for LightGBM (Finding 1 fix).
    # LightGBM was trained on raw features, not per-window standardized ones.
    _pair_cache_key = pair_name or "unknown"
    _lgb_raw_feature_cache[_pair_cache_key] = feat_arr[-1].copy()

    # Per-window standardization (zero-mean, unit-variance per column)
    mean = feat_arr.mean(axis=0, keepdims=True)
    std = feat_arr.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-4)  # Floor to prevent noise amplification (Finding 3 fix)
    feat_arr = (feat_arr - mean) / std

    # Pad or truncate to exactly INPUT_DIM
    n_feat = feat_arr.shape[1]
    if n_feat < INPUT_DIM:
        pad = np.zeros((seq_len, INPUT_DIM - n_feat), dtype=np.float32)
        feat_arr = np.concatenate([feat_arr, pad], axis=1)
    elif n_feat > INPUT_DIM:
        feat_arr = feat_arr[:, :INPUT_DIM]

    return feat_arr  # (seq_len, INPUT_DIM)


def build_full_feature_matrix(
    price_df,
    cross_data: Optional[Dict[str, 'pd.DataFrame']] = None,
    pair_name: Optional[str] = None,
    derivatives_data: Optional[Dict[str, 'pd.Series']] = None,
) -> Optional[np.ndarray]:
    """Build a (N, INPUT_DIM) feature matrix for the ENTIRE DataFrame at once.

    Uses the same scale-invariant features as build_feature_sequence().
    Does NOT apply per-window standardization — that is done when slicing
    windows in generate_sequences().

    Args:
        price_df: DataFrame with OHLCV columns (entire history for one pair)
        cross_data: Optional dict of pair_name → DataFrame for cross-asset features
        pair_name: This pair's identifier
        derivatives_data: Optional dict of feature_name → pd.Series for derivatives
            features. Series must be aligned to price_df index.

    Returns:
        (N, INPUT_DIM) float32 array where N = len(price_df), or None
    """
    import pandas as pd

    if price_df is None or len(price_df) < 30:
        return None

    df = price_df.copy().reset_index(drop=True)

    if cross_data is not None:
        _cross = {p: cdf.copy().reset_index(drop=True)
                  for p, cdf in cross_data.items() if p != pair_name}
        cross_data = _cross if _cross else None

    # Reset derivatives_data index to match df
    if derivatives_data is not None:
        derivatives_data = {
            k: v.reset_index(drop=True) if hasattr(v, 'reset_index') else v
            for k, v in derivatives_data.items()
        }

    close = df['close'].astype(float) if 'close' in df.columns else None
    if close is None:
        return None
    vol = df['volume'].astype(float) if 'volume' in df.columns else None

    # 46 scale-invariant single-pair features
    features = _compute_single_pair_features(df)

    # 15 cross-asset features
    if cross_data is not None and pair_name is not None:
        cross_feats = _build_cross_features(close, vol, cross_data, pair_name)
        features.update(cross_feats)

    # 7 derivatives + sentiment features
    deriv_feats = _build_derivatives_features(len(df), derivatives_data, df_index=df.index)
    features.update(deriv_feats)

    feat_df = pd.DataFrame(features, index=df.index)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    feat_arr = feat_df.values.astype(np.float32)

    # Pad or truncate to INPUT_DIM
    n_feat = feat_arr.shape[1]
    if n_feat < INPUT_DIM:
        pad = np.zeros((len(feat_arr), INPUT_DIM - n_feat), dtype=np.float32)
        feat_arr = np.concatenate([feat_arr, pad], axis=1)
    elif n_feat > INPUT_DIM:
        feat_arr = feat_arr[:, :INPUT_DIM]

    return feat_arr  # (N, INPUT_DIM)


# ══════════════════════════════════════════════════════════════════════════════
# LOADER — loads weights, validates, returns ready-to-use models
# ══════════════════════════════════════════════════════════════════════════════


def _detect_input_dim(model_name: str, state_dict: dict) -> Optional[int]:
    """Auto-detect the input dimension a saved model was trained with.

    Inspects the first layer's weight shape to infer the original input_dim.
    Returns None if detection fails (caller should use default).
    """
    try:
        # Each model type has a different "first layer" key
        key_map = {
            'quantum_transformer': 'input_projection.weight',
            'bidirectional_lstm': 'lstm.lstm_layers.0.weight_ih_l0',
            'dilated_cnn': 'dilated_cnn.conv_blocks.0.0.weight',
            'cnn': 'conv_layers.0.weight',
            'gru': 'gru.weight_ih_l0',
            'meta_ensemble': 'feature_extractor.0.weight',
        }
        key = key_map.get(model_name)
        if key and key in state_dict:
            w = state_dict[key]
            if model_name == 'quantum_transformer':
                # input_projection: Linear(input_dim, d_model) → weight shape (d_model, input_dim)
                return w.shape[1]
            elif model_name == 'bidirectional_lstm':
                # LSTM weight_ih: shape (4*hidden, input_size)
                return w.shape[1]
            elif model_name == 'dilated_cnn':
                # Conv1d(channels, channels, 3) → weight shape (channels, channels, 3)
                return w.shape[0]
            elif model_name == 'cnn':
                # Conv1d(input_dim, 128, 3) → weight shape (128, input_dim, 3)
                return w.shape[1]
            elif model_name == 'gru':
                # GRU weight_ih: shape (3*hidden, input_size)
                return w.shape[1]
            elif model_name == 'meta_ensemble':
                # Linear(input_dim, 128) → weight shape (128, input_dim)
                return w.shape[1]
    except Exception as e:
        logger.warning(f"Failed to auto-detect input dimension for model '{model_name}': {e}")
    return None


MODEL_REGISTRY = {
    'quantum_transformer': (
        'models/trained/best_quantum_transformer_model.pth',
        TrainedQuantumTransformer,
    ),
    'bidirectional_lstm': (
        'models/trained/best_bidirectional_lstm_model.pth',
        TrainedBidirectionalLSTM,
    ),
    'dilated_cnn': (
        'models/trained/best_dilated_cnn_model.pth',
        TrainedDilatedCNN,
    ),
    'cnn': (
        'models/trained/best_cnn_model.pth',
        TrainedCNN,
    ),
    'gru': (
        'models/trained/best_gru_model.pth',
        TrainedGRU,
    ),
    'meta_ensemble': (
        'models/trained/best_meta_ensemble_model.pth',
        TrainedMetaEnsemble,
    ),
}


def load_trained_models(base_dir: str = '.', input_dim: int = INPUT_DIM,
                        disabled: Optional[set] = None) -> Dict[str, nn.Module]:
    """Load all trained models with strict validation.

    Args:
        base_dir: Base directory for model weight files
        input_dim: Feature dimension for model constructors (default INPUT_DIM=98).
                   Use INPUT_DIM_LEGACY=83 to load old pre-trained weights exactly.
        disabled: Optional set of model names to skip. If None, uses DISABLED_MODELS.

    Returns dict of model_name → nn.Module (in eval mode).
    Only includes models that loaded with 100% weight match.
    """
    skip = disabled if disabled is not None else DISABLED_MODELS
    loaded = {}

    for name, (rel_path, model_cls) in MODEL_REGISTRY.items():
        if name in skip:
            logger.info(f"Model {name}: DISABLED (below 40% live accuracy) — skipping")
            continue
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            logger.warning(f"Model file not found: {full_path}")
            continue

        try:
            saved_sd = torch.load(full_path, map_location='cpu', weights_only=False)
            # Handle wrapped state dicts
            if isinstance(saved_sd, dict) and 'model_state_dict' in saved_sd:
                saved_sd = saved_sd['model_state_dict']

            # Auto-detect input_dim from saved weights to handle legacy models
            detected_dim = _detect_input_dim(name, saved_sd)
            use_dim = detected_dim if detected_dim is not None else input_dim

            # For meta_ensemble, detect n_models from saved weight_generator shape
            if name == 'meta_ensemble':
                wg_key = 'weight_generator.2.weight'
                detected_n_models = saved_sd[wg_key].shape[0] if wg_key in saved_sd else N_BASE_MODELS
                model = model_cls(input_dim=use_dim, n_models=detected_n_models)
                if detected_n_models != N_BASE_MODELS:
                    logger.info(f"Meta-ensemble: detected {detected_n_models} base models from saved weights "
                                f"(current N_BASE_MODELS={N_BASE_MODELS})")
            else:
                model = model_cls(input_dim=use_dim)
            result = model.load_state_dict(saved_sd, strict=False)

            # Check for mismatches
            if result.missing_keys or result.unexpected_keys:
                total = len(saved_sd)
                matched = total - len(result.unexpected_keys)
                model_params = len(dict(model.named_parameters()))
                missing_pct = len(result.missing_keys) / max(model_params, 1) * 100
                unexpected_pct = len(result.unexpected_keys) / max(total, 1) * 100

                logger.info(
                    f"Model {name}: loaded {matched}/{total} saved params, "
                    f"{len(result.missing_keys)} missing ({missing_pct:.0f}%), "
                    f"{len(result.unexpected_keys)} unexpected ({unexpected_pct:.0f}%)"
                )
                if result.missing_keys:
                    logger.debug(f"  Missing keys: {result.missing_keys[:5]}...")
                if result.unexpected_keys:
                    logger.debug(f"  Unexpected keys: {result.unexpected_keys[:5]}...")

                # If more than 20% of model params are missing, warn loudly
                if missing_pct > 20:
                    logger.warning(
                        f"Model {name}: {missing_pct:.0f}% of parameters missing — "
                        f"predictions may be unreliable"
                    )
            else:
                logger.info(f"Model {name}: loaded perfectly ({len(saved_sd)} params, strict match)")

            model.eval()
            loaded[name] = model

        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")

    logger.info(f"ML Model Loader: {len(loaded)}/{len(MODEL_REGISTRY)} models loaded")

    # Also load LightGBM if available (non-PyTorch, separate path)
    lgbm = load_lightgbm_model(base_dir)
    if lgbm is not None:
        loaded['lightgbm'] = lgbm

    return loaded


def predict_with_models(
    models: Dict[str, nn.Module],
    features: np.ndarray,
    price_series: Optional[np.ndarray] = None,
    pair: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Run inference on all loaded models (5 DL + LightGBM + meta-ensemble).

    Args:
        models: dict from load_trained_models()
        features: (seq_len, INPUT_DIM) numpy array from build_feature_sequence()
        price_series: Optional close prices (unused, kept for API compat)
        pair: Optional pair name for per-pair model exclusions (e.g. "ETH-USD")

    Returns:
        (predictions, confidences) where:
          predictions: dict of model_name → prediction (float in ~[-1, 1])
          confidences: dict of model_name → confidence (float in [0, 1])
            For deep learning models, confidence = abs(prediction) (higher signal = more confident).
            For LightGBM, confidence = distance from 0.5 probability (calibrated).
            For meta_ensemble, confidence comes from the confidence head.
    """
    if features is None:
        empty = {name: 0.0 for name in models}
        return empty, {name: 0.5 for name in models}

    predictions = {}
    confidences = {}
    x = torch.FloatTensor(features).unsqueeze(0)  # (1, seq_len, INPUT_DIM)

    # Run base models first (all take sequence input)
    _pair_excluded = []
    for name, model in models.items():
        if name in ('meta_ensemble', 'lightgbm'):
            continue  # Handle separately
        if name in DISABLED_MODELS:
            continue  # Skip models disabled for poor accuracy
        if pair and not should_include_model(name, pair):
            _pair_excluded.append(name)
            predictions[name] = 0.0  # Zero out excluded models (meta-ensemble still sees them)
            confidences[name] = 0.0
            continue
        try:
            with torch.no_grad():
                # Match feature dim to model's expected input
                model_x = _match_feature_dim(x, model, name)
                output = model(model_x)
                if isinstance(output, tuple):
                    pred = output[0]  # (prediction, uncertainty/confidence)
                else:
                    pred = output
                pred_val = float(pred[0, 0])  # Already tanh-bounded by model output layer
                pred_val = _debiaser.debias(name, pred_val)  # Remove systematic bias
                pred_val = _normalizer.normalize(name, pred_val)  # Council S3: z-score normalization
                predictions[name] = pred_val
                confidences[name] = min(abs(pred_val) + 0.5, 0.95)
        except Exception as e:
            logger.warning(f"Inference failed for {name}: {e}")
            predictions[name] = 0.0
            confidences[name] = 0.0

    # Log per-pair exclusions
    if _pair_excluded:
        logger.info(f"MODEL_ROUTING: Excluded {_pair_excluded} for {pair}")

    # Run LightGBM if loaded (uses raw features + momentum, not standardized)
    if 'lightgbm' in models:
        lgbm_pred, lgbm_conf = predict_lightgbm(
            models['lightgbm'], features, price_series, pair=pair
        )
        lgbm_pred = _debiaser.debias('lightgbm', lgbm_pred)  # Remove systematic bias
        lgbm_pred = _normalizer.normalize('lightgbm', lgbm_pred)  # Council S3: z-score normalization
        predictions['lightgbm'] = lgbm_pred
        confidences['lightgbm'] = lgbm_conf

    # Run meta-ensemble if loaded (uses base model predictions + features)
    if 'meta_ensemble' in models:
        try:
            with torch.no_grad():
                meta_model = models['meta_ensemble']
                meta_dim = meta_model._input_dim
                n_models = meta_model._n_models

                # Build base model predictions list matching what the meta-ensemble expects
                # If meta-ensemble was trained with 5 models, only pass the 5 DL predictions
                # If trained with 6, include LightGBM
                if n_models >= N_BASE_MODELS:
                    # New 6-model meta-ensemble: pass all including lightgbm
                    base_pred_list = [predictions.get(name, 0.0) for name in BASE_MODEL_NAMES]
                else:
                    # Legacy 5-model meta-ensemble: pass only DL predictions
                    base_pred_list = [
                        predictions.get(name, 0.0)
                        for name in BASE_MODEL_NAMES if name != 'lightgbm'
                    ]

                # Ensure we have exactly n_models predictions
                while len(base_pred_list) < n_models:
                    base_pred_list.append(0.0)
                base_pred_list = base_pred_list[:n_models]

                feat_vec = torch.FloatTensor(features[-1, :meta_dim]).unsqueeze(0)  # (1, meta_dim)
                base_preds = torch.FloatTensor([base_pred_list])  # (1, n_models)
                meta_input = torch.cat([feat_vec, base_preds], dim=-1)  # (1, meta_dim + n_models)
                pred, conf = meta_model(meta_input)
                meta_pred_val = _debiaser.debias('meta_ensemble', float(pred[0, 0]))
                meta_pred_val = _normalizer.normalize('meta_ensemble', meta_pred_val)  # Council S3: z-score
                predictions['meta_ensemble'] = meta_pred_val  # Debiased + normalized
                confidences['meta_ensemble'] = float(conf[0, 0])
        except Exception as e:
            logger.warning(f"Inference failed for meta_ensemble: {e}")
            predictions['meta_ensemble'] = 0.0
            confidences['meta_ensemble'] = 0.0

    # --- Ensemble agreement modifier (Council proposal #10) ---
    # Compute fraction of base models agreeing on sign with meta-ensemble prediction
    meta_pred = predictions.get('meta_ensemble', 0.0)
    if meta_pred != 0.0 and len(predictions) > 1:
        meta_sign = 1 if meta_pred > 0 else -1
        base_names = [n for n in BASE_MODEL_NAMES if n in predictions]
        if base_names:
            agree_count = sum(
                1 for n in base_names
                if (predictions[n] > 0 and meta_sign > 0) or (predictions[n] < 0 and meta_sign < 0)
            )
            agreement = agree_count / len(base_names)
            predictions['_ensemble_agreement'] = agreement

            # Apply confidence bonus/penalty to meta-ensemble confidence
            if agreement >= 0.83:  # >= 5/6 agree
                confidences['meta_ensemble'] = min(confidences.get('meta_ensemble', 0.5) + 0.05, 0.99)
            elif agreement <= 0.33:  # <= 2/6 agree
                confidences['meta_ensemble'] = max(confidences.get('meta_ensemble', 0.5) - 0.05, 0.0)

    return predictions, confidences


def _match_feature_dim(
    x: torch.Tensor,
    model: nn.Module,
    model_name: str,
) -> torch.Tensor:
    """Pad or truncate feature tensor to match model's expected input dimension.

    This handles the case where a model was trained with INPUT_DIM_LEGACY=83
    but features are now INPUT_DIM=98, or vice versa.
    """
    feat_dim = x.shape[-1]

    # Detect model's expected input dim from its first layer
    expected_dim = None
    try:
        if model_name == 'quantum_transformer':
            expected_dim = model.input_projection.in_features
        elif model_name == 'bidirectional_lstm':
            expected_dim = model.lstm.lstm_layers[0].input_size
        elif model_name == 'dilated_cnn':
            expected_dim = model.dilated_cnn.conv_blocks[0]._modules['0'].in_channels
        elif model_name == 'cnn':
            expected_dim = model.conv_layers[0].in_channels
        elif model_name == 'gru':
            expected_dim = model.gru.input_size
    except Exception as e:
        logger.warning(f"Failed to detect expected input dimension for model '{model_name}': {e}")

    if expected_dim is None or expected_dim == feat_dim:
        return x

    if feat_dim > expected_dim:
        # Truncate extra features
        return x[:, :, :expected_dim]
    else:
        # Pad with zeros
        pad = torch.zeros(x.shape[0], x.shape[1], expected_dim - feat_dim)
        return torch.cat([x, pad], dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# LIGHTGBM — gradient boosting model (non-PyTorch)
# Trained in Colab with [last, mean, std] feature preparation → 294 features.
# Trees find interaction effects and threshold rules that neural nets miss,
# structurally diversifying the ensemble.
# ══════════════════════════════════════════════════════════════════════════════

LIGHTGBM_PKL_PATH = os.path.join('models', 'trained', 'best_lightgbm_model.pkl')
LIGHTGBM_TXT_PATH = os.path.join('models', 'trained', 'lightgbm_model.txt')
LIGHTGBM_META_PATH = os.path.join('models', 'trained', 'lightgbm_meta.json')


def load_lightgbm_model(base_dir: str = '.') -> Optional[object]:
    """Load a trained LightGBM model (pickle or Booster .txt format).

    Tries pickle format first (from Colab training), falls back to .txt Booster.
    Returns the model object, or None if not found/not installed.
    """
    import json as _json
    import pickle as _pickle

    # Try pickle format first (Colab-trained model)
    pkl_path = os.path.join(base_dir, LIGHTGBM_PKL_PATH)
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                model = _pickle.load(f)
            # Load metadata if available
            meta_path = os.path.join(base_dir, LIGHTGBM_META_PATH)
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = _json.load(f)
            logger.info(f"LightGBM model loaded (pkl): {pkl_path} "
                        f"(best_iter={meta.get('best_iteration', '?')}, "
                        f"test_acc={meta.get('test_accuracy', '?')})")
            return model
        except Exception as e:
            logger.error(f"Failed to load LightGBM from pkl: {e}")

    # Fallback: .txt Booster format
    txt_path = os.path.join(base_dir, LIGHTGBM_TXT_PATH)
    if os.path.exists(txt_path):
        try:
            import lightgbm as lgb
            model = lgb.Booster(model_file=txt_path)
            logger.info(f"LightGBM model loaded (txt): {txt_path} "
                        f"({model.num_trees()} trees, {model.num_feature()} features)")
            return model
        except ImportError:
            logger.debug("LightGBM not installed — skipping lightgbm model")
            return None
        except Exception as e:
            logger.error(f"Failed to load LightGBM from txt: {e}")
            return None

    logger.debug(f"LightGBM model not found at {pkl_path} or {txt_path}")
    return None


def _prepare_lgb_features(
    raw_bar: np.ndarray,
    price_series: Optional[np.ndarray],
) -> np.ndarray:
    """Build (1, 104) feature vector matching LightGBM training format exactly.

    Training (train_lightgbm.py) uses:
      - 98 RAW features from build_full_feature_matrix (no standardization)
      - 6 momentum returns: current_close / past_close - 1.0
        for horizons [1, 3, 6, 12, 24, 72] bars back

    This was previously mismatched: inference used [last, mean, std] = 294
    standardized features while training used 104 raw features.
    See docs/ML_ACCURACY_INVESTIGATION.md Finding 1.

    Args:
        raw_bar: (INPUT_DIM,) raw features for the most recent bar
        price_series: Close prices array for momentum computation
    """
    # 1. Raw feature vector (98 features, matching training)
    bar_features = raw_bar[:INPUT_DIM]  # (INPUT_DIM,)

    # 2. Momentum returns matching training MOMENTUM_BARS = [1, 3, 6, 12, 24, 72]
    momentum_feats = []
    if price_series is not None and len(price_series) > 0:
        current_close = float(price_series[-1])
        for h in _LGB_MOMENTUM_BARS:
            if current_close > 0 and len(price_series) > h:
                past_close = float(price_series[-(h + 1)])
                if past_close > 0:
                    momentum_feats.append(current_close / past_close - 1.0)
                else:
                    momentum_feats.append(0.0)
            else:
                momentum_feats.append(0.0)
    else:
        momentum_feats = [0.0] * len(_LGB_MOMENTUM_BARS)

    # Concatenate: 98 raw + 6 momentum = 104 features (matches training)
    row = np.concatenate([bar_features, np.array(momentum_feats, dtype=np.float32)])
    return row.reshape(1, -1)  # (1, 104)


def predict_lightgbm(
    model: object,
    features: Optional[np.ndarray],
    price_series: Optional[np.ndarray] = None,
    pair: Optional[str] = None,
) -> Tuple[float, float]:
    """Run inference on LightGBM model.

    Uses raw (pre-standardization) features + momentum returns to match the
    training format exactly (98 raw + 6 momentum = 104 features).
    See docs/ML_ACCURACY_INVESTIGATION.md Finding 1.

    Args:
        model: LightGBM model from load_lightgbm_model()
        features: (seq_len, INPUT_DIM) array from build_feature_sequence()
        price_series: Close prices for momentum computation
        pair: Pair name for raw feature cache lookup

    Returns:
        (prediction, confidence) where:
          prediction: float in [-1, 1] (mapped from probability)
          confidence: float in [0, 1] (distance from 0.5, doubled)
    """
    if features is None or model is None:
        return 0.0, 0.0

    try:
        # Use raw (pre-standardization) features from cache if available.
        # LightGBM was trained on raw features, not per-window standardized.
        _cache_key = pair or "unknown"
        raw_bar = _lgb_raw_feature_cache.get(_cache_key)
        if raw_bar is None:
            # Fallback: use last bar from standardized features (not ideal but functional)
            raw_bar = features[-1, :]
            logger.debug(f"LightGBM: raw feature cache miss for {_cache_key}, using standardized fallback")

        lgb_input = _prepare_lgb_features(raw_bar, price_series)  # (1, 104)
        lgb_input = np.nan_to_num(lgb_input, nan=0.0, posinf=0.0, neginf=0.0)

        # Auto-detect input format: if model expects more features than we have,
        # pad with zeros; if fewer, truncate (handles legacy vs new format)
        expected_features = None
        if hasattr(model, 'num_feature'):
            expected_features = model.num_feature()
        elif hasattr(model, 'n_features_'):
            expected_features = model.n_features_

        if expected_features is not None:
            n_have = lgb_input.shape[1]
            if n_have < expected_features:
                pad = np.zeros((1, expected_features - n_have), dtype=np.float32)
                lgb_input = np.concatenate([lgb_input, pad], axis=1)
            elif n_have > expected_features:
                lgb_input = lgb_input[:, :expected_features]

        # LightGBM predict returns probability of class 1 (up)
        prob = float(model.predict(lgb_input)[0])

        # Map probability [0, 1] → prediction [-1, 1]
        prediction = float(np.clip((prob - 0.5) * 2.0, -1.0, 1.0))

        # Confidence: aligned with DL model range [0.5, 0.95] (Council proposal #7)
        confidence = min(abs(prob - 0.5) * 2.0 + 0.5, 0.95)

        return prediction, confidence

    except Exception as e:
        logger.warning(f"LightGBM inference failed: {e}")
        return 0.0, 0.0


# ── Crash-Regime LightGBM (v2) ──────────────────────────────────────────────
# Separate model trained on BTC crash periods (2018, 2021-22, 2025-26).
# 51 features, 2-bar (10min) horizon, binary (UP/DOWN).
# Runs ONLY for BTC pairs alongside the main ensemble.

CRASH_LGBM_PKL = os.path.join('models', 'trained', 'crash_lightgbm_model.pkl')
CRASH_LGBM_META = os.path.join('models', 'trained', 'crash_lightgbm_meta.json')

# Ordered feature list (must match training exactly — v3 order)
CRASH_FEATURE_NAMES = [
    'return_1bar', 'return_6bar', 'return_12bar', 'return_48bar', 'return_288bar',
    'vol_12bar', 'vol_48bar', 'vol_ratio',
    'volume_surge', 'volume_trend',
    'consecutive_red', 'drawdown_24h',
    'rsi_14_norm', 'bb_pct_b', 'vwap_distance',
    # Daily macro (10)
    'spx_return_1d', 'spx_vs_sma', 'vix_norm', 'vix_change', 'vix_extreme',
    'dxy_return_1d', 'dxy_trend', 'yield_level', 'yield_change', 'fng_norm',
    # Derivatives (9) — indices 25-33 in v3
    'funding_z', 'funding_extreme_long', 'funding_extreme_short',
    'oi_change_1h', 'oi_change_4h', 'oi_spike',
    'ls_ratio_norm', 'ls_extreme_long', 'taker_imbalance',
    # Intraday macro (11) — indices 34-44, all zero importance, always zero-filled
    'spx_return_5m', 'spx_return_15m', 'spx_return_1h',
    'spx_momentum_5m', 'spx_direction_5m',
    'vix_return_5m', 'vix_return_1h', 'vix_spike_5m',
    'ndx_return_5m', 'ndx_return_1h', 'has_intraday_macro',
    # Cross-asset (6)
    'eth_return_1bar', 'eth_return_6bar', 'eth_btc_ratio_change',
    'btc_lead_1', 'btc_lead_2', 'btc_lead_3',
]


def load_crash_lgbm(base_dir: str = '.') -> Tuple[object, Optional[dict]]:
    """Load the crash-regime LightGBM model and metadata.

    Returns:
        (model, meta_dict) on success, (None, None) on failure.
    """
    import json
    import pickle

    pkl_path = os.path.join(base_dir, CRASH_LGBM_PKL)
    meta_path = os.path.join(base_dir, CRASH_LGBM_META)

    if not os.path.exists(pkl_path):
        logger.warning(f"Crash LightGBM model not found: {pkl_path}")
        return None, None

    try:
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)

        meta = None
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)

        n_feat = model.num_feature() if hasattr(model, 'num_feature') else '?'
        logger.info(f"Crash-regime LightGBM loaded: {n_feat} features")
        return model, meta

    except Exception as e:
        logger.error(f"Failed to load crash LightGBM: {e}")
        return None, None


def build_crash_features(
    price_df,
    derivatives_data: Optional[dict] = None,
    macro_data: Optional[dict] = None,
    cross_data: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Build the 51-feature vector for crash-regime LightGBM inference.

    Args:
        price_df: DataFrame with columns [open, high, low, close, volume, quote_volume].
                  Needs >= 300 rows for rolling windows (vol_288bar).
        derivatives_data: Dict with keys like 'funding_rate', 'open_interest', etc.
        macro_data: Dict with daily macro keys (spx_return_1d, vix_norm, etc.).
        cross_data: Dict of pair_name → DataFrame with [close, volume] columns.

    Returns:
        numpy array shape (1, 51) or None if insufficient data.
    """
    if price_df is None or len(price_df) < 50:
        return None

    try:
        close = price_df['close'].values.astype(float)
        open_ = price_df['open'].values.astype(float) if 'open' in price_df.columns else close
        high = price_df['high'].values.astype(float) if 'high' in price_df.columns else close
        low = price_df['low'].values.astype(float) if 'low' in price_df.columns else close
        volume = price_df['volume'].values.astype(float) if 'volume' in price_df.columns else np.ones(len(close))
        quote_vol = price_df['quote_volume'].values.astype(float) if 'quote_volume' in price_df.columns else volume * close

        n = len(close)
        features = np.zeros(51, dtype=np.float64)

        # ── BTC Technical (15 features, indices 0-14) ─────────────────────
        # Returns
        features[0] = (close[-1] / close[-2] - 1.0) if n >= 2 else 0.0           # return_1bar
        features[1] = (close[-1] / close[-7] - 1.0) if n >= 7 else 0.0           # return_6bar
        features[2] = (close[-1] / close[-13] - 1.0) if n >= 13 else 0.0         # return_12bar
        features[3] = (close[-1] / close[-49] - 1.0) if n >= 49 else 0.0         # return_48bar
        features[4] = (close[-1] / close[-min(289, n)] - 1.0) if n >= 50 else 0.0  # return_288bar

        # Volatility (log returns std)
        log_ret = np.diff(np.log(np.maximum(close, 1e-10)))
        features[5] = np.std(log_ret[-12:]) if len(log_ret) >= 12 else 0.0       # vol_12bar
        features[6] = np.std(log_ret[-48:]) if len(log_ret) >= 48 else 0.0       # vol_48bar
        features[7] = (features[5] / features[6]) if features[6] > 1e-10 else 1.0  # vol_ratio

        # Volume features
        vol_mean_20 = np.mean(volume[-20:]) if n >= 20 else np.mean(volume)
        features[8] = (volume[-1] / vol_mean_20) - 1.0 if vol_mean_20 > 0 else 0.0  # volume_surge
        vol_mean_early = np.mean(volume[-20:-10]) if n >= 20 else vol_mean_20
        vol_mean_late = np.mean(volume[-10:]) if n >= 10 else vol_mean_20
        features[9] = (vol_mean_late / vol_mean_early - 1.0) if vol_mean_early > 0 else 0.0  # volume_trend

        # Consecutive red candles
        red_count = 0
        for i in range(n - 1, max(n - 20, 0) - 1, -1):
            if close[i] < open_[i]:
                red_count += 1
            else:
                break
        features[10] = float(red_count)  # consecutive_red

        # Drawdown from 24h high
        high_24h = np.max(high[-288:]) if n >= 288 else np.max(high)
        features[11] = (close[-1] / high_24h - 1.0) if high_24h > 0 else 0.0  # drawdown_24h

        # RSI (14-period, normalized to [-1, 1])
        if len(log_ret) >= 14:
            gains = np.where(log_ret[-14:] > 0, log_ret[-14:], 0)
            losses = np.where(log_ret[-14:] < 0, -log_ret[-14:], 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            rs = avg_gain / avg_loss if avg_loss > 1e-10 else 100.0
            rsi = 100.0 - (100.0 / (1.0 + rs))
            features[12] = (rsi - 50.0) / 50.0  # rsi_14_norm → [-1, 1]
        # else stays 0.0

        # Bollinger Band %B
        if n >= 20:
            sma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            if std20 > 1e-10:
                upper = sma20 + 2.0 * std20
                lower = sma20 - 2.0 * std20
                features[13] = (close[-1] - lower) / (upper - lower)  # bb_pct_b
            else:
                features[13] = 0.5

        # VWAP distance
        if n >= 20 and np.sum(volume[-20:]) > 0:
            vwap = np.sum(close[-20:] * volume[-20:]) / np.sum(volume[-20:])
            features[14] = (close[-1] / vwap - 1.0) if vwap > 0 else 0.0  # vwap_distance

        # ── Daily Macro (10 features, indices 15-24) ──────────────────────
        if macro_data:
            features[15] = float(macro_data.get('spx_return_1d', 0.0))
            features[16] = float(macro_data.get('spx_vs_sma', 0.0))
            features[17] = float(macro_data.get('vix_norm', 0.0))
            features[18] = float(macro_data.get('vix_change', 0.0))
            features[19] = float(macro_data.get('vix_extreme', 0.0))
            features[20] = float(macro_data.get('dxy_return_1d', 0.0))
            features[21] = float(macro_data.get('dxy_trend', 0.0))
            features[22] = float(macro_data.get('yield_level', 0.0))
            features[23] = float(macro_data.get('yield_change', 0.0))
            features[24] = float(macro_data.get('fng_norm', 0.0))

        # ── Derivatives (9 features, indices 25-33) — v3 order ──────────
        if derivatives_data:
            # Funding rate z-score
            fr = derivatives_data.get('funding_rate')
            if fr is not None:
                fr_val = float(fr.iloc[-1]) if hasattr(fr, 'iloc') else float(fr)
                features[25] = fr_val / 0.0003 if abs(fr_val) > 1e-10 else 0.0  # funding_z
                features[26] = 1.0 if fr_val > 0.0005 else 0.0   # funding_extreme_long
                features[27] = 1.0 if fr_val < -0.0005 else 0.0  # funding_extreme_short

            # Open interest changes
            oi = derivatives_data.get('open_interest')
            if oi is not None and hasattr(oi, 'iloc') and len(oi) >= 12:
                oi_arr = oi.values.astype(float)
                oi_now = oi_arr[-1]
                if oi_now > 0:
                    features[28] = (oi_now / oi_arr[-12] - 1.0) if len(oi_arr) >= 12 else 0.0  # oi_change_1h
                    features[29] = (oi_now / oi_arr[-min(48, len(oi_arr))] - 1.0)  # oi_change_4h
                    oi_mean = np.mean(oi_arr[-48:]) if len(oi_arr) >= 48 else np.mean(oi_arr)
                    oi_std = np.std(oi_arr[-48:]) if len(oi_arr) >= 48 else np.std(oi_arr)
                    features[30] = (oi_now - oi_mean) / oi_std if oi_std > 1e-10 else 0.0  # oi_spike

            # Long/short ratio
            ls = derivatives_data.get('long_short_ratio')
            if ls is not None:
                ls_val = float(ls.iloc[-1]) if hasattr(ls, 'iloc') else float(ls)
                features[31] = (ls_val - 1.0)  # ls_ratio_norm (centered at 1.0)
                features[32] = 1.0 if ls_val > 2.0 else 0.0  # ls_extreme_long

            # Taker buy/sell imbalance
            tbv = derivatives_data.get('taker_buy_volume')
            tsv = derivatives_data.get('taker_sell_volume')
            if tbv is not None and tsv is not None:
                tb = float(tbv.iloc[-1]) if hasattr(tbv, 'iloc') else float(tbv)
                ts = float(tsv.iloc[-1]) if hasattr(tsv, 'iloc') else float(tsv)
                total = tb + ts
                features[33] = (tb - ts) / total if total > 0 else 0.0  # taker_imbalance

        # ── Intraday Macro (11 features, indices 34-44) — always zero ────
        # All 11 have zero importance in the trained model.
        # features[34:45] already zero from initialization.

        # ── Cross-Asset / ETH (6 features, indices 45-50) ────────────────
        if cross_data:
            # Look for ETH data in cross_data dict
            eth_df = None
            for key in cross_data:
                if 'ETH' in str(key).upper():
                    eth_df = cross_data[key]
                    break

            if eth_df is not None and hasattr(eth_df, 'empty') and not eth_df.empty and 'close' in eth_df.columns:
                eth_close = eth_df['close'].values.astype(float)
                en = len(eth_close)
                if en >= 2:
                    features[45] = eth_close[-1] / eth_close[-2] - 1.0  # eth_return_1bar
                if en >= 7:
                    features[46] = eth_close[-1] / eth_close[-7] - 1.0  # eth_return_6bar
                # BTC/ETH ratio change
                if en >= 7 and eth_close[-1] > 0 and eth_close[-7] > 0:
                    btc_eth_now = close[-1] / eth_close[-1]
                    btc_eth_prev = close[-7] / eth_close[-7]
                    features[47] = btc_eth_now / btc_eth_prev - 1.0  # eth_btc_ratio_change
                # Lead signals: lagged BTC returns
                if n >= 4:
                    features[48] = close[-2] / close[-3] - 1.0 if n >= 3 else 0.0  # btc_lead_1
                    features[49] = close[-3] / close[-4] - 1.0 if n >= 4 else 0.0  # btc_lead_2
                    features[50] = close[-4] / close[-5] - 1.0 if n >= 5 else 0.0  # btc_lead_3
            else:
                # Still compute lead signals from BTC even without ETH
                if n >= 5:
                    features[48] = close[-2] / close[-3] - 1.0  # btc_lead_1
                    features[49] = close[-3] / close[-4] - 1.0  # btc_lead_2
                    features[50] = close[-4] / close[-5] - 1.0  # btc_lead_3

        else:
            # No cross data at all — still compute BTC leads
            if n >= 5:
                features[48] = close[-2] / close[-3] - 1.0  # btc_lead_1
                features[49] = close[-3] / close[-4] - 1.0  # btc_lead_2
                features[50] = close[-4] / close[-5] - 1.0  # btc_lead_3

        # Clean NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features.reshape(1, 51)

    except Exception as e:
        logger.warning(f"build_crash_features failed: {e}")
        return None


def predict_crash_lgbm(
    model: object,
    features: Optional[np.ndarray],
) -> Tuple[float, float]:
    """Run inference on crash-regime LightGBM.

    Args:
        model: LightGBM Booster from load_crash_lgbm().
        features: (1, 51) array from build_crash_features().

    Returns:
        (prediction, confidence) where:
          prediction: float in [-1, 1] (mapped from P(UP))
          confidence: float in [0, 1] (distance from 0.5, doubled)
    """
    if features is None or model is None:
        return 0.0, 0.0

    try:
        prob = float(model.predict(features)[0])  # P(UP) in [0, 1]
        prediction = float(np.clip((prob - 0.5) * 2.0, -1.0, 1.0))
        confidence = min(abs(prob - 0.5) * 2.0 + 0.5, 0.95)  # Aligned with DL range
        return prediction, confidence
    except Exception as e:
        logger.warning(f"Crash LightGBM inference failed: {e}")
        return 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# VOLATILITY PREDICTION MODEL — LightGBM regression for magnitude prediction
# Predicts |forward_6bar_return| (magnitude, NOT direction) for Kelly sizing
# and dead-zone filtering.
# ══════════════════════════════════════════════════════════════════════════════

VOLATILITY_LGBM_PKL = os.path.join('models', 'volatility_lgbm.pkl')

# Regime classification from training percentiles
VOL_REGIMES = ('dead_zone', 'normal', 'active', 'explosive')


def load_volatility_model(base_dir: str = '.') -> Tuple[Optional[object], Optional[dict]]:
    """Load the volatility prediction LightGBM model and metadata.

    Returns:
        (model, meta_dict) on success, (None, None) on failure.
        meta_dict has keys: feature_names, n_features, label_percentiles,
        train_mae, val_mae, test_mae, trained_at
    """
    import pickle

    pkl_path = os.path.join(base_dir, VOLATILITY_LGBM_PKL)
    if not os.path.exists(pkl_path):
        logger.debug(f"Volatility model not found: {pkl_path}")
        return None, None

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict) and 'model' in data:
            model = data['model']
            meta = {k: v for k, v in data.items() if k != 'model'}
            n_feat = meta.get('n_features', '?')
            mae = meta.get('test_mae', '?')
            logger.info(f"Volatility LightGBM loaded: {n_feat} features, test MAE={mae}")
            return model, meta
        else:
            # Direct model object
            logger.info("Volatility LightGBM loaded (raw model, no metadata)")
            return data, None

    except ImportError:
        logger.debug("LightGBM not installed — skipping volatility model")
        return None, None
    except Exception as e:
        logger.error(f"Failed to load volatility model: {e}")
        return None, None


def _build_volatility_features(
    price_df: 'pd.DataFrame',
    cross_vol_5: Optional[float] = None,
    oi_change_pct: Optional[float] = None,
    funding_rate_z: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Build feature vector for volatility prediction.

    Uses the standard 46 single-pair features from _compute_single_pair_features()
    plus ~13 volatility-specific features.

    Args:
        price_df: OHLCV DataFrame (needs 30+ rows for feature stability)
        cross_vol_5: 5-bar realized vol of cross-asset (BTC for alts, ETH for BTC)
        oi_change_pct: Open interest change %, or None
        funding_rate_z: Funding rate z-score, or None

    Returns:
        (1, N) numpy array ready for model.predict(), or None on failure.
    """
    import pandas as pd

    if price_df is None or len(price_df) < 20:
        return None

    try:
        # 1. Compute base 46 features
        base_feats = _compute_single_pair_features(price_df)

        # Extract last-bar values as a dict
        feat_vals = {}
        for name, series in base_feats.items():
            val = series.iloc[-1] if hasattr(series, 'iloc') else float(series)
            feat_vals[name] = float(val) if not (isinstance(val, float) and math.isnan(val)) else 0.0

        # 2. Volatility-specific features
        close = price_df['close'].astype(float)
        high = price_df['high'].astype(float) if 'high' in price_df.columns else close
        low = price_df['low'].astype(float) if 'low' in price_df.columns else close
        pct_ret = close.pct_change()

        # Vol ratios
        vol_5 = pct_ret.rolling(5).std().iloc[-1]
        vol_10 = pct_ret.rolling(10).std().iloc[-1]
        vol_20 = pct_ret.rolling(20).std().iloc[-1]
        feat_vals['vol_ratio_5_20'] = float(vol_5 / (vol_20 + 1e-10)) if not math.isnan(vol_5) else 1.0
        feat_vals['vol_ratio_10_20'] = float(vol_10 / (vol_20 + 1e-10)) if not math.isnan(vol_10) else 1.0

        # Parkinson volatility at 5, 10, 20 bars
        ln_hl = np.log(high / (low + 1e-10))
        ln_hl_sq = ln_hl ** 2 / (4 * np.log(2))
        for w in [5, 10, 20]:
            park = ln_hl_sq.rolling(w).mean().iloc[-1]
            feat_vals[f'parkinson_vol_{w}'] = float(park) if not math.isnan(park) else 0.0

        # Garman-Klass volatility (10 bars)
        _open = price_df['open'].astype(float) if 'open' in price_df.columns else close
        gk_term1 = 0.5 * np.log(high / (low + 1e-10)) ** 2
        gk_term2 = -(2 * np.log(2) - 1) * np.log(close / (_open + 1e-10)) ** 2
        gk = (gk_term1 + gk_term2).rolling(10).mean().iloc[-1]
        feat_vals['garman_klass_vol'] = float(gk) if not math.isnan(gk) else 0.0

        # Time features (sin/cos encoding)
        if 'bar_start' in price_df.columns:
            ts = pd.to_datetime(price_df['bar_start'].iloc[-1])
        elif price_df.index.dtype == 'datetime64[ns]':
            ts = price_df.index[-1]
        else:
            ts = datetime.now(timezone.utc)

        hour = ts.hour + ts.minute / 60.0
        dow = ts.weekday()
        feat_vals['hour_sin'] = float(np.sin(2 * np.pi * hour / 24))
        feat_vals['hour_cos'] = float(np.cos(2 * np.pi * hour / 24))
        feat_vals['dow_sin'] = float(np.sin(2 * np.pi * dow / 7))
        feat_vals['dow_cos'] = float(np.cos(2 * np.pi * dow / 7))

        # Cross-asset vol
        feat_vals['cross_asset_vol_5'] = float(cross_vol_5) if cross_vol_5 is not None else 0.0

        # Derivatives features
        feat_vals['oi_change_pct'] = float(oi_change_pct) if oi_change_pct is not None else 0.0
        feat_vals['funding_rate_z'] = float(funding_rate_z) if funding_rate_z is not None else 0.0

        # 3. Assemble in consistent order (sorted by name for reproducibility)
        all_names = sorted(feat_vals.keys())
        vec = np.array([feat_vals.get(n, 0.0) for n in all_names], dtype=np.float32)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec.reshape(1, -1)

    except Exception as e:
        logger.warning(f"_build_volatility_features failed: {e}")
        return None


def predict_volatility(
    model: object,
    meta: Optional[dict],
    price_df: 'pd.DataFrame',
    cross_vol_5: Optional[float] = None,
    oi_change_pct: Optional[float] = None,
    funding_rate_z: Optional[float] = None,
) -> Dict[str, Any]:
    """Predict price move magnitude and classify volatility regime.

    Args:
        model: LightGBM model from load_volatility_model()
        meta: Metadata dict with label_percentiles and feature_names
        price_df: OHLCV DataFrame (30+ rows recommended)
        cross_vol_5: Cross-asset 5-bar vol (optional)
        oi_change_pct: OI change % (optional)
        funding_rate_z: Funding rate z-score (optional)

    Returns:
        Dict with keys:
          predicted_log_magnitude: raw model output (log1p scale)
          predicted_magnitude_bps: back-transformed to bps
          vol_regime: 'dead_zone' | 'normal' | 'active' | 'explosive'
          vol_multiplier: float for Kelly sizing (0.0 to 2.0)
    """
    result = {
        'predicted_log_magnitude': 0.0,
        'predicted_magnitude_bps': 100.0,  # default fallback
        'vol_regime': 'normal',
        'vol_multiplier': 1.0,
    }

    if model is None:
        return result

    try:
        features = _build_volatility_features(
            price_df, cross_vol_5, oi_change_pct, funding_rate_z,
        )
        if features is None:
            return result

        # Handle feature count mismatch
        if meta and 'feature_names' in meta:
            expected_n = len(meta['feature_names'])
            n_have = features.shape[1]
            if n_have < expected_n:
                pad = np.zeros((1, expected_n - n_have), dtype=np.float32)
                features = np.concatenate([features, pad], axis=1)
            elif n_have > expected_n:
                features = features[:, :expected_n]
        else:
            # Auto-detect from model
            expected_n = None
            if hasattr(model, 'num_feature'):
                expected_n = model.num_feature()
            elif hasattr(model, 'n_features_'):
                expected_n = model.n_features_
            if expected_n is not None:
                n_have = features.shape[1]
                if n_have < expected_n:
                    pad = np.zeros((1, expected_n - n_have), dtype=np.float32)
                    features = np.concatenate([features, pad], axis=1)
                elif n_have > expected_n:
                    features = features[:, :expected_n]

        # Predict (regression output: log1p(magnitude_bps))
        pred_log = float(model.predict(features)[0])
        result['predicted_log_magnitude'] = pred_log

        # Back-transform: expm1 to get magnitude in bps
        magnitude_bps = float(np.expm1(max(pred_log, 0.0)))
        result['predicted_magnitude_bps'] = magnitude_bps

        # Classify regime using training percentiles
        percentiles = {}
        if meta and 'label_percentiles' in meta:
            percentiles = meta['label_percentiles']

        p25 = percentiles.get('p25', 2.0)
        p75 = percentiles.get('p75', 4.5)
        p90 = percentiles.get('p90', 5.5)

        min_predicted_vol_bps = 12.0
        if pred_log < p25 or magnitude_bps < min_predicted_vol_bps:
            regime = 'dead_zone'
            vol_multiplier = 0.0  # Skip trading
        elif pred_log < p75:
            regime = 'normal'
            vol_multiplier = 1.0
        elif pred_log < p90:
            regime = 'active'
            vol_multiplier = 1.5
        else:
            regime = 'explosive'
            vol_multiplier = 2.0

        result['vol_regime'] = regime
        result['vol_multiplier'] = vol_multiplier

        return result

    except Exception as e:
        logger.warning(f"predict_volatility failed: {e}")
        return result
