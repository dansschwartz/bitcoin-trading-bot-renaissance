"""
Renaissance Signal Fusion (LEGACY — not used in golden path)

This module is imported but never instantiated by renaissance_trading_bot.py.
The golden path uses renaissance_engine_core.SignalFusion instead.
Kept for reference; imports fixed to use class names instead of non-existent singletons.
"""

import logging
from typing import Dict, Any, Tuple
from datetime import datetime

from analysis.microstructure_engine import MicrostructureEngine
from analysis.enhanced_technical_indicators import EnhancedTechnicalIndicators
from enhanced_config_manager import EnhancedConfigManager

logger = logging.getLogger(__name__)

class RenaissanceSignalFusion:
    """
    Renaissance Technologies Signal Fusion Engine (LEGACY)
    Combines all signal sources with research-optimized weights.
    Not used in the golden path — see renaissance_engine_core.SignalFusion.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Instantiate engines (legacy — golden path creates these per-asset in the main bot)
        self._microstructure = MicrostructureEngine()
        self._technical = EnhancedTechnicalIndicators()
        self._config_manager = EnhancedConfigManager("config")
        
        # Renaissance Technologies research-optimized weights
        self.signal_weights = {
            'microstructure': 0.70,    # 70% - Order flow, depth, volume spikes
            'technical': 0.25,         # 25% - Enhanced technical indicators
            'alternative_data': 0.05   # 5% - Remaining alternative data
        }
        
        self.logger.info("✅ Renaissance Signal Fusion initialized")
    
    def get_combined_renaissance_signal(self) -> Dict[str, Any]:
        """
        Get complete Renaissance Technologies signal analysis
        
        Returns:
            Combined signal with all Renaissance Technologies components
        """
        try:
            # Get microstructure signals (70% weight)
            microstructure_data = self._microstructure.get_signal_summary()
            microstructure_signal = microstructure_data.get('overall_signal', 0.0) if microstructure_data.get('status') != 'no_data' else 0.0
            microstructure_confidence = microstructure_data.get('confidence', 0.0) if microstructure_data.get('status') != 'no_data' else 0.0
            
            # Get enhanced technical signals (25% weight)
            technical_data = self._technical.get_signals_summary()
            technical_signal = technical_data.get('combined_signal', 0.0) if technical_data.get('status') != 'no_data' else 0.0
            technical_confidence = technical_data.get('confidence', 0.0) if technical_data.get('status') != 'no_data' else 0.0
            
            # Get configuration data
            config_data = self._config_manager.get_config_summary()
            
            # Alternative data signal (placeholder - 5% weight)
            alternative_signal = 0.0  # Will be expanded in Step 5
            alternative_confidence = 0.0
            
            # Combine signals with Renaissance Technologies weights
            combined_signal = (
                microstructure_signal * self.signal_weights['microstructure'] +
                technical_signal * self.signal_weights['technical'] +
                alternative_signal * self.signal_weights['alternative_data']
            )
            
            # Weighted confidence
            total_weight = sum(self.signal_weights.values())
            combined_confidence = (
                microstructure_confidence * self.signal_weights['microstructure'] +
                technical_confidence * self.signal_weights['technical'] +
                alternative_confidence * self.signal_weights['alternative_data']
            ) / total_weight
            
            # Generate trading decision
            decision = self._generate_trading_decision(combined_signal, combined_confidence)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'renaissance_signal': combined_signal,
                'confidence': combined_confidence,
                'decision': decision,
                'regime': config_data.get('current_regime', 'unknown'),
                'components': {
                    'microstructure': {
                        'signal': microstructure_signal,
                        'confidence': microstructure_confidence,
                        'weight': self.signal_weights['microstructure'],
                        'details': microstructure_data.get('components', {})
                    },
                    'technical': {
                        'signal': technical_signal,
                        'confidence': technical_confidence,
                        'weight': self.signal_weights['technical'],
                        'details': technical_data.get('indicators', {})
                    },
                    'alternative_data': {
                        'signal': alternative_signal,
                        'confidence': alternative_confidence,
                        'weight': self.signal_weights['alternative_data']
                    }
                },
                'signal_weights': self.signal_weights,
                'status': 'active'
            }
            
        except Exception as e:
            self.logger.error(f"Error in Renaissance signal fusion: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'renaissance_signal': 0.0,
                'confidence': 0.0,
                'decision': 'HOLD',
                'status': 'error',
                'error': str(e)
            }
    
    def _generate_trading_decision(self, signal: float, confidence: float) -> str:
        """Generate trading decision based on signal and confidence"""
        try:
            # Get current configuration thresholds
            config = self._config_manager.get_current_config()
            
            buy_threshold = config.trading_thresholds.buy_threshold
            sell_threshold = config.trading_thresholds.sell_threshold
            confidence_threshold = config.trading_thresholds.confidence_threshold
            
            # Decision logic
            if confidence < confidence_threshold:
                return 'HOLD'  # Low confidence
            elif signal > buy_threshold:
                return 'BUY'
            elif signal < sell_threshold:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            self.logger.error(f"Error generating trading decision: {e}")
            return 'HOLD'

# Global Renaissance signal fusion instance
renaissance_signal_fusion = RenaissanceSignalFusion()
