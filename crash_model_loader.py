"""Multi-asset crash-regime LightGBM loader.

Loads all crash models from models/crash/ directory.
Each model is keyed by (asset, horizon) — e.g., ('BTC', '2bar').
Proxy mapping routes assets without dedicated models to BTC.
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CRASH_MODEL_DIR = os.path.join('models', 'crash')

# Proxy mapping: assets without dedicated models use these instead
PROXY_MAP: Dict[str, str] = {
    'SOL': 'BTC',
    'DOGE': 'BTC',
}

# Cross-asset direction: what lead indicator each model uses
# BTC model trained with ETH as cross-asset; all alts trained with BTC as cross-asset
CROSS_ASSET_MAP: Dict[str, str] = {
    'BTC': 'ETH',
    'ETH': 'BTC',
    'SOL': 'BTC',
    'XRP': 'BTC',
    'DOGE': 'BTC',
}


@dataclass
class CrashModelEntry:
    """Registry entry for a loaded crash model."""
    asset: str
    horizon: str
    model: object  # LightGBM Booster
    meta: dict
    accuracy: float
    auc: float
    n_features: int


class CrashModelLoader:
    """Loads and manages all crash-regime LightGBM models.

    Models are stored in models/crash/ as {asset}_{horizon}.pkl with
    corresponding {asset}_{horizon}_meta.json files.

    Usage:
        loader = CrashModelLoader()
        model, meta = loader.get_model('BTC', '2bar')
        pred, conf = loader.predict(model, features)
    """

    def __init__(self, base_dir: str = '.', logger_: Optional[logging.Logger] = None):
        self.logger = logger_ or logger
        self.base_dir = base_dir
        self._models: Dict[Tuple[str, str], CrashModelEntry] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Scan models/crash/ and load all .pkl files with matching meta."""
        model_dir = os.path.join(self.base_dir, CRASH_MODEL_DIR)
        if not os.path.isdir(model_dir):
            self.logger.warning(f"Crash model directory not found: {model_dir}")
            return

        loaded = 0
        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith('.pkl'):
                continue

            # Parse filename: btc_2bar.pkl -> asset=BTC, horizon=2bar
            stem = fname[:-4]  # Remove .pkl
            parts = stem.split('_')
            if len(parts) != 2:
                self.logger.debug(f"Skipping non-standard crash model file: {fname}")
                continue

            asset = parts[0].upper()
            horizon = parts[1]

            pkl_path = os.path.join(model_dir, fname)
            meta_path = os.path.join(model_dir, f"{stem}_meta.json")

            try:
                with open(pkl_path, 'rb') as f:
                    model = pickle.load(f)

                meta = {}
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)

                n_feat = model.num_feature() if hasattr(model, 'num_feature') else 51
                accuracy = meta.get('test_acc', meta.get('test_accuracy', 0.0))
                auc = meta.get('test_auc', 0.0)

                # BTC v3 meta has nested format — extract from winner comparison
                if accuracy == 0.0 and 'comparison' in meta:
                    winner_key = meta.get('winner_key', 'B')
                    for comp_name, comp_data in meta['comparison'].items():
                        if winner_key in comp_name or comp_name == meta.get('winner', ''):
                            accuracy = comp_data.get('test_acc', 0.0)
                            auc = comp_data.get('test_auc', 0.0)
                            break
                    # Fallback: use primary horizon from horizon_comparison
                    if accuracy == 0.0 and 'horizon_comparison' in meta:
                        primary = meta.get('primary_horizon', '2-bar')
                        for h_name, h_data in meta['horizon_comparison'].items():
                            if primary.split(' ')[0] in h_name:
                                accuracy = h_data.get('test_acc', 0.0)
                                auc = h_data.get('test_auc', 0.0)
                                break

                entry = CrashModelEntry(
                    asset=asset,
                    horizon=horizon,
                    model=model,
                    meta=meta,
                    accuracy=accuracy,
                    auc=auc,
                    n_features=n_feat,
                )
                self._models[(asset, horizon)] = entry
                loaded += 1
                self.logger.info(
                    f"Crash model loaded: {asset}_{horizon} "
                    f"(acc={accuracy:.1%}, auc={auc:.3f}, {n_feat} features)"
                )

            except Exception as e:
                self.logger.error(f"Failed to load crash model {fname}: {e}")

        self.logger.info(f"CrashModelLoader: {loaded} models loaded from {model_dir}")

    def get_model(self, asset: str, horizon: str) -> Tuple[Optional[object], Optional[dict]]:
        """Get model for asset+horizon, falling back to proxy if needed.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP, DOGE)
            horizon: Prediction horizon (1bar, 2bar, 6bar)

        Returns:
            (model, meta) or (None, None) if not available.
        """
        asset_upper = asset.upper()
        key = (asset_upper, horizon)

        # Direct lookup
        entry = self._models.get(key)
        if entry:
            return entry.model, entry.meta

        # Proxy lookup
        proxy = PROXY_MAP.get(asset_upper)
        if proxy:
            proxy_key = (proxy, horizon)
            entry = self._models.get(proxy_key)
            if entry:
                self.logger.debug(f"Using proxy {proxy}_{horizon} for {asset_upper}")
                return entry.model, entry.meta

        return None, None

    def get_entry(self, asset: str, horizon: str) -> Optional[CrashModelEntry]:
        """Get full registry entry for asset+horizon."""
        asset_upper = asset.upper()
        key = (asset_upper, horizon)
        entry = self._models.get(key)
        if entry:
            return entry

        proxy = PROXY_MAP.get(asset_upper)
        if proxy:
            return self._models.get((proxy, horizon))
        return None

    def predict(self, model: object, features: Optional[np.ndarray]) -> Tuple[float, float]:
        """Run inference on a crash-regime LightGBM model.

        Args:
            model: LightGBM Booster.
            features: (1, 51) array from CrashFeatureBuilder.

        Returns:
            (prediction, confidence) where:
              prediction: float in [-1, 1] (mapped from P(UP))
              confidence: float in [0, 1]
        """
        if features is None or model is None:
            return 0.0, 0.0

        try:
            prob = float(model.predict(features)[0])  # P(UP) in [0, 1]
            prediction = float(np.clip((prob - 0.5) * 2.0, -1.0, 1.0))
            confidence = abs(prob - 0.5) * 2.0
            return prediction, confidence
        except Exception as e:
            self.logger.warning(f"Crash LightGBM inference failed: {e}")
            return 0.0, 0.0

    def predict_for_asset(
        self,
        asset: str,
        horizon: str,
        features: Optional[np.ndarray],
    ) -> Tuple[float, float, str]:
        """Convenience: get model + predict in one call.

        Returns:
            (prediction, confidence, source) where source is e.g. 'btc_2bar' or 'btc_2bar(proxy)'.
        """
        asset_upper = asset.upper()
        model, meta = self.get_model(asset_upper, horizon)
        if model is None:
            return 0.0, 0.0, 'none'

        pred, conf = self.predict(model, features)

        # Determine source label
        direct_key = (asset_upper, horizon)
        if direct_key in self._models:
            source = f"{asset_upper.lower()}_{horizon}"
        else:
            proxy = PROXY_MAP.get(asset_upper, asset_upper)
            source = f"{proxy.lower()}_{horizon}(proxy)"

        return pred, conf, source

    @property
    def available_models(self) -> List[str]:
        """List of loaded model keys like ['BTC_2bar', 'ETH_2bar', ...]."""
        return [f"{a}_{h}" for (a, h) in sorted(self._models.keys())]

    @property
    def model_count(self) -> int:
        return len(self._models)

    def get_state(self) -> dict:
        """Serializable state for dashboard/logging."""
        models = {}
        for (asset, horizon), entry in self._models.items():
            models[f"{asset}_{horizon}"] = {
                "accuracy": entry.accuracy,
                "auc": entry.auc,
                "n_features": entry.n_features,
            }
        return {
            "model_count": len(self._models),
            "models": models,
            "proxy_map": PROXY_MAP,
            "cross_asset_map": CROSS_ASSET_MAP,
        }

    @staticmethod
    def get_cross_asset(asset: str) -> str:
        """Return the cross-asset lead indicator for the given asset."""
        return CROSS_ASSET_MAP.get(asset.upper(), 'BTC')
