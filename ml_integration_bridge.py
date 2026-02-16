
"""
ML Integration Bridge for Renaissance Trading Bot
Connects trained PyTorch models with the existing Renaissance bot system.

Uses ml_model_loader to load models with exact architecture matching (no random weights).
"""

import numpy as np
import pandas as pd
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
import traceback
import warnings
warnings.filterwarnings('ignore')

# Import trained model loader (exact-match architectures)
try:
    from ml_model_loader import load_trained_models, predict_with_models, build_feature_sequence
    HAS_TRAINED_MODELS = True
except ImportError as e:
    print(f"Warning: ml_model_loader not available: {e}")
    HAS_TRAINED_MODELS = False

# Import Renaissance types
from renaissance_types import TradingDecision as TradingSignal, SignalType, MLSignalPackage

@dataclass
class MLModelStatus:
    """Status of individual ML models"""
    model_name: str
    is_loaded: bool
    last_prediction_time: Optional[datetime]
    error_count: int
    performance_score: float
    health_status: str  # 'healthy', 'degraded', 'failed'

class MLModelManager:
    """Manages individual ML models with health monitoring"""

    def __init__(self):
        self.models = {}
        self.model_status = {}
        self.fallback_enabled = True
        self.logger = logging.getLogger(__name__)

    def initialize_models(self) -> bool:
        """Initialize trained PyTorch models with exact architecture matching."""
        if not HAS_TRAINED_MODELS:
            self.logger.warning("ml_model_loader not available — ML disabled")
            return False

        try:
            loaded = load_trained_models()
            for name, model in loaded.items():
                self.models[name] = model
                self.model_status[name] = MLModelStatus(
                    model_name=name, is_loaded=True,
                    last_prediction_time=None, error_count=0,
                    performance_score=0.8, health_status='healthy',
                )

            if loaded:
                self.logger.info(
                    f"ML models initialized: {list(loaded.keys())} "
                    f"({len(loaded)} models with trained weights)"
                )
                return True
            else:
                self.logger.warning("No trained models loaded — ML disabled")
                return False

        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            return False

    def get_model_health(self) -> Dict[str, MLModelStatus]:
        """Get health status of all models"""
        return self.model_status.copy()

    def is_model_healthy(self, model_name: str) -> bool:
        """Check if a specific model is healthy"""
        status = self.model_status.get(model_name)
        return status is not None and status.health_status == 'healthy'

class ConsciousnessEngine:
    """Meta-confidence scoring using consciousness-like processing"""

    def __init__(self):
        self.attention_weights = {
            'prediction_consistency': 0.3,
            'cross_model_agreement': 0.25,
            'temporal_stability': 0.2,
            'market_regime_fit': 0.15,
            'uncertainty_quantification': 0.1
        }
        self.prediction_history = []

    def calculate_meta_confidence(self, predictions: List[Any], 
                                 market_context: Dict[str, Any]) -> float:
        """Calculate meta-confidence score using consciousness-like processing"""
        if not predictions:
            return 0.0

        confidence_components = {}

        try:
            # Prediction consistency
            if len(predictions) > 1:
                pred_values = [p.get('strength', 0) if isinstance(p, dict) else 0 for p in predictions]
                consistency = 1.0 - np.std(pred_values) if len(pred_values) > 1 else 0.8
                confidence_components['prediction_consistency'] = min(consistency, 1.0)
            else:
                confidence_components['prediction_consistency'] = 0.6

            # Cross-model agreement
            agreement_score = self._calculate_cross_model_agreement(predictions)
            confidence_components['cross_model_agreement'] = agreement_score

            # Temporal stability
            stability_score = self._calculate_temporal_stability(predictions)
            confidence_components['temporal_stability'] = stability_score

            # Market regime fit
            regime_fit = self._assess_market_regime_fit(market_context)
            confidence_components['market_regime_fit'] = regime_fit

            # Uncertainty quantification
            uncertainty = self._quantify_uncertainty(predictions)
            confidence_components['uncertainty_quantification'] = 1.0 - uncertainty

            # Weighted combination
            meta_confidence = sum(
                score * self.attention_weights.get(component, 0.1)
                for component, score in confidence_components.items()
            )

            return np.clip(meta_confidence, 0.0, 1.0)

        except Exception as e:
            logging.error(f"Error in meta-confidence calculation: {e}")
            return 0.5  # Default moderate confidence

    def _calculate_cross_model_agreement(self, predictions: List[Any]) -> float:
        """Calculate how well models agree with each other"""
        if len(predictions) < 2:
            return 0.7

        # Extract prediction strengths
        strengths = []
        for pred in predictions:
            if hasattr(pred, 'strength'):
                strengths.append(pred.strength)
            elif isinstance(pred, dict) and 'strength' in pred:
                strengths.append(pred['strength'])
            else:
                strengths.append(0.0)

        if not strengths:
            return 0.5

        # Calculate agreement as inverse of standard deviation
        agreement = 1.0 - min(np.std(strengths), 1.0)
        return max(agreement, 0.1)

    def _calculate_temporal_stability(self, predictions: List[Any]) -> float:
        """Calculate temporal stability of predictions"""
        # Add current predictions to history
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'predictions': predictions
        })

        # Keep only recent history (last 10 predictions)
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)

        if len(self.prediction_history) < 3:
            return 0.6

        # Calculate stability over time
        recent_strengths = []
        for hist_entry in self.prediction_history[-5:]:
            for pred in hist_entry['predictions']:
                if hasattr(pred, 'strength'):
                    recent_strengths.append(pred.strength)
                elif isinstance(pred, dict) and 'strength' in pred:
                    recent_strengths.append(pred['strength'])

        if len(recent_strengths) < 2:
            return 0.5

        stability = 1.0 - min(np.std(recent_strengths), 1.0)
        return max(stability, 0.1)

    def _assess_market_regime_fit(self, market_context: Dict[str, Any]) -> float:
        """Assess how well current predictions fit the market regime"""
        # Simple heuristic based on volatility and trend
        volatility = market_context.get('volatility', 0.02)
        trend_strength = abs(market_context.get('trend', 0.0))

        # Models work better in certain regimes
        if volatility < 0.01:  # Low volatility
            return 0.8
        elif volatility > 0.05:  # High volatility
            return 0.6
        else:  # Medium volatility
            return 0.7 + trend_strength * 0.2

    def _quantify_uncertainty(self, predictions: List[Any]) -> float:
        """Quantify prediction uncertainty"""
        if not predictions:
            return 1.0

        # Calculate uncertainty based on prediction spread and confidence
        confidences = []
        for pred in predictions:
            if hasattr(pred, 'confidence'):
                confidences.append(pred.confidence)
            elif isinstance(pred, dict) and 'confidence' in pred:
                confidences.append(pred['confidence'])

        if not confidences:
            return 0.8

        # Uncertainty is inverse of average confidence
        avg_confidence = np.mean(confidences)
        uncertainty = 1.0 - avg_confidence

        return np.clip(uncertainty, 0.0, 1.0)

class FractalAnalyzer:
    """Fractal pattern analysis for enhanced signal processing"""

    def __init__(self):
        self.fractal_history = []

    def analyze_fractal_patterns(self, market_data: pd.DataFrame, 
                                predictions: List[Any]) -> Dict[str, Any]:
        """Analyze fractal patterns in market data and predictions"""
        insights = {
            'fractal_dimension': 0.0,
            'self_similarity': 0.0,
            'pattern_strength': 0.0,
            'scale_invariance': 0.0,
            'regime_detection': 'normal'
        }

        try:
            if len(market_data) < 50:
                return insights

            # Calculate fractal dimension using box-counting method
            price_series = market_data['close'].values
            fractal_dim = self._calculate_fractal_dimension(price_series)
            insights['fractal_dimension'] = fractal_dim

            # Analyze self-similarity across scales
            self_sim = self._analyze_self_similarity(price_series)
            insights['self_similarity'] = self_sim

            # Pattern strength based on fractal analysis
            pattern_strength = self._calculate_pattern_strength(price_series, predictions)
            insights['pattern_strength'] = pattern_strength

            # Detect market regime based on fractal properties
            regime = self._detect_market_regime(fractal_dim, self_sim)
            insights['regime_detection'] = regime

        except Exception as e:
            logging.error(f"Error in fractal analysis: {e}")

        return insights

    def _calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using simplified box-counting"""
        if len(data) < 10:
            return 1.5

        # Normalize data
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)

        # Simple fractal dimension estimate
        differences = np.abs(np.diff(data_norm))
        avg_diff = np.mean(differences)

        # Heuristic mapping to fractal dimension
        fractal_dim = 1.0 + min(avg_diff * 10, 1.0)

        return fractal_dim

    def _analyze_self_similarity(self, data: np.ndarray) -> float:
        """Analyze self-similarity across different scales"""
        if len(data) < 20:
            return 0.5

        # Compare patterns at different scales
        scales = [5, 10, 20]
        similarities = []

        for scale in scales:
            if len(data) >= scale * 3:
                # Downsample to different scales
                downsampled = data[::scale]
                if len(downsampled) >= 3:
                    # Simple autocorrelation as similarity measure
                    autocorr = np.corrcoef(downsampled[:-1], downsampled[1:])[0, 1]
                    if not np.isnan(autocorr):
                        similarities.append(abs(autocorr))

        return np.mean(similarities) if similarities else 0.5

    def _calculate_pattern_strength(self, data: np.ndarray, predictions: List[Any]) -> float:
        """Calculate pattern strength based on fractal properties"""
        if len(data) < 10:
            return 0.5

        # Measure pattern regularity
        returns = np.diff(np.log(data + 1e-6))
        volatility = np.std(returns)

        # Higher volatility = lower pattern strength
        pattern_strength = max(0.1, 1.0 - min(volatility * 100, 0.9))

        return pattern_strength

    def _detect_market_regime(self, fractal_dim: float, self_similarity: float) -> str:
        """Detect market regime based on fractal properties"""
        if fractal_dim > 1.8 and self_similarity < 0.3:
            return 'chaotic'
        elif fractal_dim < 1.3 and self_similarity > 0.7:
            return 'trending'
        elif self_similarity > 0.6:
            return 'mean_reverting'
        else:
            return 'normal'

class MLIntegrationBridge:
    """Main integration bridge connecting ML components with Renaissance bot"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Initialize components
        self.model_manager = MLModelManager()
        self.consciousness_engine = ConsciousnessEngine()
        self.fractal_analyzer = FractalAnalyzer()

        # Integration settings
        self.ml_enabled = True
        self.fallback_mode = False
        self.integration_weights = {
            'traditional_signals': 0.6,
            'ml_predictions': 0.4
        }

        # Performance tracking
        self.performance_metrics = {
            'ml_prediction_count': 0,
            'successful_integrations': 0,
            'fallback_activations': 0,
            'avg_processing_time': 0.0
        }

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> bool:
        """Initialize the ML integration bridge"""
        self.logger.info("Initializing ML Integration Bridge...")

        # Initialize ML models
        ml_success = self.model_manager.initialize_models()

        if not ml_success:
            self.logger.warning("ML models failed to initialize - enabling fallback mode")
            self.fallback_mode = True
            self.ml_enabled = False

        self.logger.info(f"ML Integration Bridge initialized (ML enabled: {self.ml_enabled})")
        return True

    async def generate_ml_signals(self, market_data: pd.DataFrame,
                                 traditional_signals: List[TradingSignal]) -> MLSignalPackage:
        """Generate ML-enhanced signals using trained PyTorch models."""
        start_time = datetime.now()

        if not self.ml_enabled or self.fallback_mode:
            return MLSignalPackage(
                primary_signals=[],
                ml_predictions=[],
                ensemble_score=0.0,
                confidence_score=0.5,
                fractal_insights={},
                timestamp=datetime.now(),
                processing_time_ms=0.0
            )

        ml_signals = []
        ml_predictions = []

        try:
            models = self.model_manager.models
            if not models:
                return MLSignalPackage(
                    primary_signals=[], ml_predictions=[],
                    ensemble_score=0.0, confidence_score=0.5,
                    fractal_insights={}, timestamp=datetime.now(),
                    processing_time_ms=0.0,
                )

            # Build feature sequence from market data
            features = build_feature_sequence(market_data, seq_len=30)
            if features is None:
                return MLSignalPackage(
                    primary_signals=[], ml_predictions=[],
                    ensemble_score=0.0, confidence_score=0.5,
                    fractal_insights={}, timestamp=datetime.now(),
                    processing_time_ms=0.0,
                )

            # Run inference on all loaded models
            raw_preds = predict_with_models(models, features)

            # Convert raw predictions to ML prediction dicts
            for model_name, pred_value in raw_preds.items():
                if not self.model_manager.is_model_healthy(model_name):
                    continue
                try:
                    pred_dict = {
                        'model': model_name,
                        'prediction': float(pred_value),
                        'strength': float(np.clip(pred_value, -1.0, 1.0)),
                        'confidence': min(abs(float(pred_value)) + 0.5, 0.95),
                        'horizon': 5,
                        'metadata': {'source': 'trained_pytorch'}
                    }
                    ml_predictions.append(pred_dict)
                    signal = self._convert_ml_to_trading_signal(pred_dict, model_name)
                    if signal:
                        ml_signals.append(signal)
                    self._update_model_health(model_name, True)
                except Exception as e:
                    self.logger.warning(f"Model {model_name} prediction processing failed: {e}")
                    self._update_model_health(model_name, False)

            # Generate ensemble score
            ensemble_score = self._calculate_ensemble_score(ml_predictions)

            # Calculate meta-confidence
            market_context = self._extract_market_context(market_data)
            confidence_score = self.consciousness_engine.calculate_meta_confidence(
                ml_predictions, market_context
            )

            # Fractal analysis
            fractal_insights = self.fractal_analyzer.analyze_fractal_patterns(
                market_data, ml_predictions
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            self.performance_metrics['ml_prediction_count'] += 1
            self.performance_metrics['successful_integrations'] += 1
            self.performance_metrics['avg_processing_time'] = (
                (self.performance_metrics['avg_processing_time'] *
                 (self.performance_metrics['ml_prediction_count'] - 1) + processing_time) /
                self.performance_metrics['ml_prediction_count']
            )

            return MLSignalPackage(
                primary_signals=ml_signals,
                ml_predictions=ml_predictions,
                ensemble_score=ensemble_score,
                confidence_score=confidence_score,
                fractal_insights=fractal_insights,
                timestamp=datetime.now(),
                processing_time_ms=processing_time
            )

        except Exception as e:
            self.logger.error(f"Error in ML signal generation: {e}")
            self.performance_metrics['fallback_activations'] += 1

            return MLSignalPackage(
                primary_signals=[],
                ml_predictions=[],
                ensemble_score=0.0,
                confidence_score=0.3,
                fractal_insights={},
                timestamp=datetime.now(),
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    def _convert_ml_to_trading_signal(self, ml_prediction: Dict[str, Any],
                                     model_name: str) -> Optional[TradingSignal]:
        """Convert ML prediction to TradingSignal (TradingDecision) format"""
        try:
            strength = float(ml_prediction.get('strength', 0))
            action = 'BUY' if strength > 0 else ('SELL' if strength < 0 else 'HOLD')
            return TradingSignal(
                action=action,
                confidence=float(ml_prediction.get('confidence', 0.5)),
                position_size=0.0,
                reasoning={
                    'ml_model': model_name,
                    'strength': strength,
                    'source': 'ml_integration_bridge',
                },
                timestamp=datetime.now(),
            )
        except Exception as e:
            self.logger.error(f"Error converting ML prediction to TradingSignal: {e}")
            return None

    def _calculate_ensemble_score(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate ensemble score from multiple ML predictions"""
        if not predictions:
            return 0.0

        # Weight predictions by confidence and combine
        total_weight = 0.0
        weighted_sum = 0.0

        for pred in predictions:
            strength = pred.get('strength', 0.0)
            confidence = pred.get('confidence', 0.5)

            weighted_sum += strength * confidence
            total_weight += confidence

        if total_weight > 0:
            ensemble_score = weighted_sum / total_weight
        else:
            ensemble_score = 0.0

        return np.clip(ensemble_score, -1.0, 1.0)

    def _extract_market_context(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract market context for meta-confidence calculation"""
        context = {
            'volatility': 0.02,
            'trend': 0.0,
            'data_quality': 1.0
        }

        try:
            if len(market_data) >= 20:
                returns = market_data['close'].pct_change().fillna(0)
                context['volatility'] = returns.rolling(20).std().iloc[-1]
                context['trend'] = returns.rolling(10).mean().iloc[-1]
                context['data_quality'] = 1.0 if not market_data.isnull().any().any() else 0.8
        except Exception as e:
            self.logger.error(f"Error extracting market context: {e}")

        return context

    def _update_model_health(self, model_name: str, is_healthy: bool):
        """Update model health status"""
        if model_name in self.model_manager.model_status:
            status = self.model_manager.model_status[model_name]
            if not is_healthy:
                status.error_count += 1
                if status.error_count > 5:
                    status.health_status = 'failed'
                elif status.error_count > 2:
                    status.health_status = 'degraded'
            else:
                status.health_status = 'healthy'
                status.last_prediction_time = datetime.now()

    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        return {
            'ml_enabled': self.ml_enabled,
            'fallback_mode': self.fallback_mode,
            'model_health': self.model_manager.get_model_health(),
            'performance_metrics': self.performance_metrics.copy(),
            'integration_weights': self.integration_weights.copy()
        }

    def update_integration_weights(self, traditional_weight: float, ml_weight: float):
        """Update integration weights based on performance"""
        total = traditional_weight + ml_weight
        if total > 0:
            self.integration_weights['traditional_signals'] = traditional_weight / total
            self.integration_weights['ml_predictions'] = ml_weight / total
