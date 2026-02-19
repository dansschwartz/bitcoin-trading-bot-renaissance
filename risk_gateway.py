"""
Risk Gateway Adapter
Bridges Golden Path trading with Step 9 Risk Management + VAE Anomaly Detection.
"""

import logging
import os
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from renaissance_engine_core import RiskManager as RenaissanceRiskManager
from vae_anomaly_detector import VariationalAutoEncoder

VAE_WEIGHTS_PATH = "models/trained/vae_anomaly_detector.pth"


class RiskGateway:
    """
    Risk management gate for the Renaissance Trading Bot.
    Includes VAE-based anomaly detection (only active when trained weights exist).
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = config.get("enabled", False)
        self.fail_open = config.get("fail_open", False)

        self._last_vae_loss = 0.0
        self.vae_trained = False
        self.pass_count = 0
        self.reject_count = 0

        self.risk_manager = RenaissanceRiskManager()

        # VAE Anomaly Detector — 83 features matching ml_model_loader pipeline
        self.anomaly_threshold = float(config.get("anomaly_threshold", 2.5))
        self.vae = None
        try:
            self.vae = VariationalAutoEncoder(input_dim=83, latent_dim=32)
            if os.path.exists(VAE_WEIGHTS_PATH):
                state = torch.load(VAE_WEIGHTS_PATH, map_location='cpu', weights_only=True)
                self.vae.load_state_dict(state)
                self.vae_trained = True
                self.logger.info("VAE Anomaly Detector loaded with trained weights")
            else:
                self.logger.info("VAE Anomaly Detector: no trained weights — anomaly detection disabled")
            self.vae.eval()
        except Exception as e:
            self.logger.warning(f"VAE initialization failed: {e}")
            self.vae = None

        self.logger.info(f"RiskGateway initialized (enabled={self.enabled}, vae_trained={self.vae_trained})")

    def assess_trade(self, action: str, amount: float, current_price: float,
                     portfolio_data: Dict[str, Any],
                     feature_vector: Optional[np.ndarray] = None) -> Tuple[bool, float, str]:
        """Assess if a trade is compliant with risk limits.

        Returns:
            Tuple of (is_allowed, vae_loss, reason)
        """
        if not self.enabled:
            self.pass_count += 1
            return True, self._last_vae_loss, "gateway_disabled"

        try:
            # VAE Anomaly Check (only if trained weights loaded)
            if self.vae_trained and self.vae is not None and feature_vector is not None:
                is_anomaly, score = self._check_anomaly(feature_vector)
                if is_anomaly:
                    self.logger.warning(f"ANOMALY DETECTED (score={score:.4f}). Blocking {action}.")
                    self.reject_count += 1
                    return False, score, "vae_anomaly"

            # Basic portfolio risk checks
            daily_pnl = portfolio_data.get('daily_pnl', 0.0)
            total_value = portfolio_data.get('total_value', 0.0)
            if total_value > 0 and abs(daily_pnl) / total_value > 0.15:
                self.logger.warning(f"Risk check: daily drawdown {abs(daily_pnl)/total_value:.1%} exceeds 15%")
                self.reject_count += 1
                return False, self._last_vae_loss, "daily_drawdown_limit"

            self.pass_count += 1
            return True, self._last_vae_loss, "OK"

        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            self.pass_count += 1
            return self.fail_open, self._last_vae_loss, f"error:{e}"

    def _check_anomaly(self, feature_vector: np.ndarray) -> Tuple[bool, float]:
        """VAE reconstruction error as anomaly score."""
        try:
            with torch.no_grad():
                x = torch.FloatTensor(feature_vector).reshape(1, -1)
                recon, mu, logvar = self.vae(x)
                loss = torch.mean((x - recon) ** 2).item()
                self._last_vae_loss = float(loss)
                return loss > self.anomaly_threshold, float(loss)
        except Exception as e:
            self.logger.error(f"VAE anomaly check failed: {e}")
            return False, 0.0

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Current risk metrics for dashboard."""
        if not self.enabled:
            return {}
        return {
            'vae_trained': self.vae_trained,
            'vae_loss': self._last_vae_loss,
            'var_1d': 0.0,
            'cvar_1d': 0.0,
            'volatility': 0.0,
        }
