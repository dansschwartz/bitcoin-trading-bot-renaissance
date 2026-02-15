
"""
ML Integration Bridge for Renaissance Trading Bot
Connects all 7 ML components with the existing Renaissance bot system
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

# Import ML components
try:
    from ml_config import MLPatternConfig as MLConfig
    from ml_pattern_engine import MLPatternEngine, PatternSignal, MLPrediction
    from cnn_lstm_model import CNNLSTMModel
    from nbeats_forecaster import NBEATSForecaster
    HAS_ML_COMPONENTS = True
except ImportError as e:
    print(f"Warning: Some ML components not available: {e}")
    HAS_ML_COMPONENTS = False

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
        """Initialize ML models — tries TF path first, falls back to PyTorch trained models."""
        # Try TensorFlow path first
        if HAS_ML_COMPONENTS:
            try:
                self.models['cnn_lstm'] = CNNLSTMModel()
                self.model_status['cnn_lstm'] = MLModelStatus(
                    model_name='cnn_lstm', is_loaded=True,
                    last_prediction_time=None, error_count=0,
                    performance_score=0.8, health_status='healthy',
                )
                self.models['nbeats'] = NBEATSForecaster()
                self.model_status['nbeats'] = MLModelStatus(
                    model_name='nbeats', is_loaded=True,
                    last_prediction_time=None, error_count=0,
                    performance_score=0.75, health_status='healthy',
                )
                self.models['pattern_engine'] = MLPatternEngine()
                self.model_status['pattern_engine'] = MLModelStatus(
                    model_name='pattern_engine', is_loaded=True,
                    last_prediction_time=None, error_count=0,
                    performance_score=0.85, health_status='healthy',
                )
                self.logger.info("All ML models initialized (TensorFlow path)")
                return True
            except Exception as e:
                self.logger.info(f"TF models unavailable ({e}), trying PyTorch fallback...")

        # Fallback: load trained PyTorch models
        return self._initialize_pytorch_fallback()

    def _initialize_pytorch_fallback(self) -> bool:
        """Load trained PyTorch models as fallback when TF components unavailable."""
        try:
            import torch
            import os
            from neural_network_prediction_engine import LegendaryNeuralPredictionEngine, PredictionConfig

            nn_config = PredictionConfig()
            nn_engine = LegendaryNeuralPredictionEngine(nn_config)

            # Map model names to their .pth files and creation functions
            # Models were trained with input_dim=83
            trained_dim = 83
            model_defs = {
                'quantum_transformer': ('models/trained/best_quantum_transformer_model.pth',
                                        lambda: nn_engine._create_quantum_transformer(input_dim=trained_dim)),
                'bidirectional_lstm': ('models/trained/best_bidirectional_lstm_model.pth',
                                       lambda: nn_engine._create_bidirectional_lstm(input_dim=trained_dim)),
                'cnn': ('models/trained/best_cnn_model.pth',
                        lambda: nn_engine._create_dilated_cnn(input_dim=trained_dim)),
            }

            loaded_count = 0
            for model_name, (model_path, create_fn) in model_defs.items():
                if not os.path.exists(model_path):
                    continue
                try:
                    saved_data = torch.load(model_path, map_location='cpu', weights_only=False)
                    input_features = saved_data.get('input_features', 115)
                    model = create_fn()
                    state_dict = saved_data.get('model_state_dict', saved_data)
                    # Load with strict=False to handle architecture evolution
                    # (trained core weights load, newer layers keep random init)
                    result = model.load_state_dict(state_dict, strict=False)
                    loaded_params = len(state_dict) - len(result.unexpected_keys)
                    model.eval()
                    self.models[model_name] = model
                    self.model_status[model_name] = MLModelStatus(
                        model_name=model_name, is_loaded=True,
                        last_prediction_time=None, error_count=0,
                        performance_score=0.8, health_status='healthy',
                    )
                    loaded_count += 1
                    self.logger.info(
                        f"PyTorch model loaded: {model_name} "
                        f"({loaded_params} trained params, {len(result.missing_keys)} new params)"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to load PyTorch model {model_name}: {e}")

            if loaded_count > 0:
                self.logger.info(f"PyTorch fallback: {loaded_count} trained models loaded")
                return True
            else:
                self.logger.warning("No PyTorch models could be loaded — ML in fallback mode")
                return False

        except ImportError:
            self.logger.warning("PyTorch not available — ML in fallback mode")
            return False
        except Exception as e:
            self.logger.warning(f"PyTorch fallback init failed: {e}")
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
        """Generate ML-enhanced signals"""
        start_time = datetime.now()

        if not self.ml_enabled or self.fallback_mode:
            # Return empty ML package in fallback mode
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
            # Generate predictions from each ML model
            models = self.model_manager.models

            # CNN-LSTM predictions
            if 'cnn_lstm' in models and self.model_manager.is_model_healthy('cnn_lstm'):
                try:
                    cnn_pred = await self._get_cnn_lstm_prediction(market_data)
                    if cnn_pred:
                        ml_predictions.append(cnn_pred)
                        # Convert to TradingSignal
                        signal = self._convert_ml_to_trading_signal(cnn_pred, 'cnn_lstm')
                        if signal:
                            ml_signals.append(signal)
                except Exception as e:
                    self.logger.error(f"CNN-LSTM prediction failed: {e}")
                    self._update_model_health('cnn_lstm', False)

            # N-BEATS predictions
            if 'nbeats' in models and self.model_manager.is_model_healthy('nbeats'):
                try:
                    nbeats_pred = await self._get_nbeats_prediction(market_data)
                    if nbeats_pred:
                        ml_predictions.append(nbeats_pred)
                        signal = self._convert_ml_to_trading_signal(nbeats_pred, 'nbeats')
                        if signal:
                            ml_signals.append(signal)
                except Exception as e:
                    self.logger.error(f"N-BEATS prediction failed: {e}")
                    self._update_model_health('nbeats', False)

            # Pattern Engine predictions (includes ensemble)
            if 'pattern_engine' in models and self.model_manager.is_model_healthy('pattern_engine'):
                try:
                    pattern_preds = await self._get_pattern_engine_predictions(market_data)
                    ml_predictions.extend(pattern_preds)
                    for pred in pattern_preds:
                        signal = self._convert_ml_to_trading_signal(pred, 'pattern_engine')
                        if signal:
                            ml_signals.append(signal)
                except Exception as e:
                    self.logger.error(f"Pattern Engine prediction failed: {e}")
                    self._update_model_health('pattern_engine', False)

            # Generate ensemble score
            ensemble_score = self._calculate_ensemble_score(ml_predictions)

            # Calculate meta-confidence using consciousness engine
            market_context = self._extract_market_context(market_data)
            confidence_score = self.consciousness_engine.calculate_meta_confidence(
                ml_predictions, market_context
            )

            # Perform fractal analysis
            fractal_insights = self.fractal_analyzer.analyze_fractal_patterns(
                market_data, ml_predictions
            )

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Update performance metrics
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

            # Return empty package on error
            return MLSignalPackage(
                primary_signals=[],
                ml_predictions=[],
                ensemble_score=0.0,
                confidence_score=0.3,
                fractal_insights={},
                timestamp=datetime.now(),
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    async def _get_cnn_lstm_prediction(self, market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get prediction from CNN-LSTM model"""
        # Simulate CNN-LSTM prediction
        if len(market_data) < 50:
            return None

        # Mock prediction - in real implementation, this would call the actual model
        returns = market_data['close'].pct_change().fillna(0)
        volatility = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else 0.02

        prediction = {
            'model': 'cnn_lstm',
            'prediction': np.random.normal(0, volatility),
            'strength': np.clip(np.random.normal(0, 0.3), -1, 1),
            'confidence': np.random.uniform(0.4, 0.9),
            'horizon': 5,  # 5 periods ahead
            'metadata': {'volatility': float(volatility)}
        }

        return prediction

    async def _get_nbeats_prediction(self, market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get prediction from N-BEATS model"""
        if len(market_data) < 30:
            return None

        # Mock N-BEATS prediction
        trend = market_data['close'].pct_change(10).iloc[-1] if len(market_data) >= 10 else 0

        prediction = {
            'model': 'nbeats',
            'prediction': trend * np.random.uniform(0.8, 1.2),
            'strength': np.clip(trend * 2, -1, 1),
            'confidence': np.random.uniform(0.5, 0.85),
            'horizon': 10,  # 10 periods ahead
            'metadata': {'trend': float(trend)}
        }

        return prediction

    async def _get_pattern_engine_predictions(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get predictions from Pattern Engine (multiple patterns)"""
        predictions = []

        if len(market_data) < 20:
            return predictions

        # Mock multiple pattern predictions
        patterns = ['head_shoulders', 'double_top', 'triangle', 'flag']

        for pattern in patterns[:2]:  # Limit to 2 patterns
            pred = {
                'model': 'pattern_engine',
                'pattern': pattern,
                'strength': np.random.uniform(-0.8, 0.8),
                'confidence': np.random.uniform(0.3, 0.8),
                'horizon': np.random.randint(3, 15),
                'metadata': {'pattern_type': pattern}
            }
            predictions.append(pred)

        return predictions

    def _convert_ml_to_trading_signal(self, ml_prediction: Dict[str, Any], 
                                     model_name: str) -> Optional[TradingSignal]:
        """Convert ML prediction to TradingSignal format"""
        try:
            return TradingSignal(
                signal_type=SignalType.TECHNICAL,  # Map ML signals to technical for now
                strength=float(ml_prediction.get('strength', 0)),
                confidence=float(ml_prediction.get('confidence', 0.5)),
                timeframe="ml_prediction",
                timestamp=datetime.now(),
                metadata={
                    'ml_model': model_name,
                    'original_prediction': ml_prediction,
                    'source': 'ml_integration_bridge'
                }
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
