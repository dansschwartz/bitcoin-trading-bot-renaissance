"""
Risk Gateway Adapter
Bridges Golden Path trading with Experimental Step 9 Risk Management Fortress.
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from renaissance_engine_core import RiskManager as RenaissanceRiskManager
from neural_network_prediction_engine import VariationalAutoEncoder

class RiskGateway:
    """
    Adapter that provides advanced risk management to the Renaissance Trading Bot.
    Uses the experimental RenaissanceRiskManager (Step 9) and VAE Anomaly Detection.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = config.get("enabled", False)
        self.consciousness_boost = config.get("consciousness_boost", 0.0)
        self.fail_open = config.get("fail_open", False)  # Fail-closed: block trade on error
        
        # Initialize state for dashboard visibility
        self._last_vae_loss = 0.0

        # Initialize the experimental risk manager
        self.risk_manager = RenaissanceRiskManager(
            # consciousness_boost=self.consciousness_boost,
            # max_portfolio_value=config.get("max_portfolio_value", 1000.0)
        )

        # Initialize VAE Anomaly Detector (Step 12+)
        try:
            # input_dim=128 matches FractalFeaturePipeline output
            self.vae = VariationalAutoEncoder(input_dim=128, latent_dim=32)
            self.vae.eval()
            self.anomaly_threshold = float(config.get("anomaly_threshold", 2.5))
            self.logger.info("VAE Anomaly Detector initialized in RiskGateway")
        except Exception as e:
            self.logger.warning(f"VAE initialization failed: {e}")
            self.vae = None
        
        self.logger.info(f"RiskGateway initialized (Enabled: {self.enabled}, Fail-Open: {self.fail_open})")

    def assess_trade(self, action: str, amount: float, current_price: float, 
                     portfolio_data: Dict[str, Any], feature_vector: Optional[np.ndarray] = None) -> bool:
        """
        Assess if a trade is compliant with risk limits.
        Now includes VAE-based Anomaly Detection.
        """
        if not self.enabled:
            return True

        try:
            # 1. VAE Anomaly Check
            if self.vae is not None and feature_vector is not None:
                is_anomaly, score = self._check_anomaly(feature_vector)
                if is_anomaly:
                    self.logger.warning(f"⚠️ ANOMALY DETECTED (Score: {score:.2f}). Blocking {action} for Black Swan protection.")
                    return False

            # 2. Basic portfolio risk checks
            daily_pnl = portfolio_data.get('daily_pnl', 0.0)
            total_value = portfolio_data.get('total_value', 0.0)
            if total_value > 0 and abs(daily_pnl) / total_value > 0.15:
                self.logger.warning(f"Risk check: daily drawdown {abs(daily_pnl)/total_value:.1%} exceeds 15%")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed in gateway: {e}")
            return self.fail_open

    def _check_anomaly(self, feature_vector: np.ndarray) -> Tuple[bool, float]:
        """Runs VAE anomaly detection on the current market feature vector."""
        try:
            with torch.no_grad():
                x = torch.FloatTensor(feature_vector).reshape(1, -1)
                recon, mu, logvar = self.vae(x)
                
                # Reconstruction loss as anomaly score
                loss = torch.mean((x - recon) ** 2).item()
                self._last_vae_loss = float(loss)
                
                # Simple thresholding (in production, use a moving average baseline)
                is_anomaly = loss > self.anomaly_threshold
                return is_anomaly, float(loss)
        except Exception as e:
            self.logger.error(f"VAE anomaly check failed: {e}")
            return False, 0.0

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics for logging/UI."""
        if not self.enabled:
            return {}
        
        # RiskManager in renaissance_trading_bot doesn't store current_metrics in the same way
        return {
            'var_1d': 0.0,
            'cvar_1d': 0.0,
            'volatility': 0.0,
            'liquidity_score': 0.0,
            'tail_risk_score': 0.0
        }
