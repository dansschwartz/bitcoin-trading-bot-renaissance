"""
Intelligence Layer
==================
Higher-order analysis modules for the Renaissance trading bot.

B1  - RegimeDetector           : 3-state HMM market regime classification
B2  - SignalValidator          : 6-gate signal validation pipeline
D3  - InsurancePremiumScanner  : Systematic de-risk premium harvesting
D9  - FastMeanReversionScanner : Sub-bar dislocation detection (1s eval)
D11 - MicrostructurePredictor  : Ultra-short-term 1s-30s price prediction
D11 - StatisticalPredictor     : Medium-term 30s-5m statistical prediction
D11 - RegimePredictor          : Long-term 5m-60m regime-based prediction
D11 - MultiHorizonEstimator    : MHPE orchestrator (probability cones)
D11 - ConeCalibrator           : Daily MHPE accuracy tracking & correction
"""

from intelligence.regime_detector import RegimeDetector
from intelligence.signal_validator import (
    SignalValidator,
    SignalValidationReport,
    ValidationResult,
)
from intelligence.insurance_scanner import InsurancePremiumScanner
from intelligence.fast_mean_reversion import FastMeanReversionScanner
from intelligence.microstructure_predictor import MicrostructurePredictor
from intelligence.statistical_predictor import StatisticalPredictor
from intelligence.regime_predictor import RegimePredictor
from intelligence.multi_horizon_estimator import MultiHorizonEstimator
from intelligence.cone_calibrator import ConeCalibrator

__all__ = [
    "RegimeDetector",
    "SignalValidator",
    "SignalValidationReport",
    "ValidationResult",
    "InsurancePremiumScanner",
    "FastMeanReversionScanner",
    "MicrostructurePredictor",
    "StatisticalPredictor",
    "RegimePredictor",
    "MultiHorizonEstimator",
    "ConeCalibrator",
]
