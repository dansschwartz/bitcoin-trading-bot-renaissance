"""
Renaissance Technologies Integration
Connect your existing bot to the enhanced data pipeline
"""

import sys
import os
import asyncio
from pathlib import Path
import json
import logging


class RenaissanceIntegration:
    """Integrates Renaissance-style signals with your existing bot"""

    def __init__(self):
        self.alternative_weight = 0.20  # Start with 20%, increase to 40% gradually
        self.logger = logging.getLogger(__name__)

        # We'll collect data directly from API clients instead of using the full pipeline
        self._setup_clients()

    def _setup_clients(self):
        """Setup individual API clients"""
        try:
            # Add data_pipeline to path
            data_pipeline_path = str(Path(__file__).parent.parent / "data_pipeline")
            if data_pipeline_path not in sys.path:
                sys.path.append(data_pipeline_path)

            # Load config
            config_path = Path(__file__).parent.parent / "data_pipeline" / "config" / "data_pipeline_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Import and setup only the clients we need (avoiding the full pipeline)
            from clients.fear_greed_client import FearGreedClient
            from clients.twitter_client import TwitterClient

            # Initialize working clients
            self.fear_greed_client = FearGreedClient(config.get('fear_greed', {}))

            # Twitter client (handle rate limits gracefully)
            try:
                self.twitter_client = TwitterClient(config.get('twitter', {}))
            except Exception as e:
                self.logger.warning(f"Twitter client unavailable: {e}")
                self.twitter_client = None

        except Exception as e:
            self.logger.error(f"Error setting up clients: {e}")
            # Fallback - no external data
            self.fear_greed_client = None
            self.twitter_client = None

    async def get_renaissance_signals(self):
        """Get Renaissance-style alternative data signals"""
        try:
            signals = {
                'social_sentiment': 0,
                'fear_greed': 0,
                'market_momentum': 0,
                'confidence': 0.5,
                'timestamp': asyncio.get_event_loop().time()
            }

            # Get Fear & Greed Index (most reliable source)
            if self.fear_greed_client:
                try:
                    fng_data = self.fear_greed_client.get_fear_greed_index()
                    if fng_data and not fng_data.get('error'):
                        fear_greed_value = fng_data.get('fear_greed_value', 50)
                        # Normalize to -1 to 1 range (50 = neutral = 0)
                        signals['fear_greed'] = (fear_greed_value - 50) / 50
                        signals['confidence'] = 0.8
                except Exception as e:
                    self.logger.warning(f"Fear & Greed data unavailable: {e}")

            # Get Twitter sentiment (if available and not rate limited)
            if self.twitter_client:
                try:
                    twitter_data = await self.twitter_client.get_bitcoin_sentiment()
                    if twitter_data and not twitter_data.get('error'):
                        signals['social_sentiment'] = twitter_data.get('sentiment_score', 0)
                        signals['confidence'] = min(signals['confidence'] + 0.2, 1.0)
                except Exception as e:
                    self.logger.warning(f"Twitter sentiment unavailable: {e}")

            return signals

        except Exception as e:
            self.logger.error(f"Error getting Renaissance signals: {e}")
            return self._get_default_signals()

    def _get_default_signals(self):
        """Return neutral signals if data collection fails"""
        return {
            'social_sentiment': 0,
            'fear_greed': 0,
            'market_momentum': 0,
            'confidence': 0.3,  # Low confidence when using defaults
            'timestamp': asyncio.get_event_loop().time()
        }

    def calculate_enhanced_signal(self, existing_signals, alternative_signals=None):
        """
        Calculate final signal combining your existing signals with Renaissance data

        Args:
            existing_signals: Your current signals dict with keys: rsi, macd, bollinger, order_flow, volume
            alternative_signals: Renaissance signals (optional - will fetch if None)

        Returns:
            Enhanced signal score
        """

        if alternative_signals is None:
            # Get alternative signals, but don't fail if it doesn't work
            try:
                alternative_signals = asyncio.run(self.get_renaissance_signals())
            except:
                alternative_signals = self._get_default_signals()

        # Adjust weight based on confidence
        confidence = alternative_signals.get('confidence', 0.5)
        effective_alternative_weight = self.alternative_weight * confidence
        technical_weight = 1.0 - effective_alternative_weight

        # Calculate technical analysis score (your existing logic)
        technical_score = (
                                  existing_signals.get('rsi', 0) * 0.25 +
                                  existing_signals.get('macd', 0) * 0.30 +
                                  existing_signals.get('bollinger', 0) * 0.20 +
                                  existing_signals.get('order_flow', 0) * 0.15 +
                                  existing_signals.get('volume', 0) * 0.10
                          ) * technical_weight

        # Calculate Renaissance alternative data score
        alternative_score = (
                                    alternative_signals.get('social_sentiment', 0) * 0.40 +
                                    alternative_signals.get('fear_greed', 0) * 0.40 +
                                    alternative_signals.get('market_momentum', 0) * 0.20
                            ) * effective_alternative_weight

        final_score = technical_score + alternative_score

        # Return both the score and metadata for debugging
        return {
            'signal': final_score,
            'technical_component': technical_score,
            'alternative_component': alternative_score,
            'alternative_weight': effective_alternative_weight,
            'confidence': confidence,
            'alternative_signals': alternative_signals
        }

    def set_alternative_weight(self, weight):
        """Set the weight for alternative data (start with 0.1, gradually increase to 0.4)"""
        self.alternative_weight = max(0.0, min(0.4, weight))

    def get_simple_signal(self, existing_signals):
        """Simple version that just returns the enhanced signal number"""
        result = self.calculate_enhanced_signal(existing_signals)
        return result['signal']


# Simple test function
if __name__ == "__main__":
    async def test():
        print("Testing Renaissance Integration...")
        renaissance = RenaissanceIntegration()

        # Test with sample signals (replace with your actual signal values)
        test_signals = {
            'rsi': 0.6,  # Your RSI signal
            'macd': 0.7,  # Your MACD signal
            'bollinger': 0.5,  # Your Bollinger signal
            'order_flow': 0.8,  # Your order flow signal
            'volume': 0.4  # Your volume signal
        }

        print("Getting Renaissance signals...")
        alt_signals = await renaissance.get_renaissance_signals()
        print(f"Alternative signals: {alt_signals}")

        print("\nCalculating enhanced signal...")
        result = renaissance.calculate_enhanced_signal(test_signals, alt_signals)

        print(f"\nResults:")
        print(f"Final signal: {result['signal']:.3f}")
        print(f"Technical component: {result['technical_component']:.3f}")
        print(f"Alternative component: {result['alternative_component']:.3f}")
        print(f"Alternative weight used: {result['alternative_weight']:.1%}")
        print(f"Confidence: {result['confidence']:.1%}")


    asyncio.run(test())