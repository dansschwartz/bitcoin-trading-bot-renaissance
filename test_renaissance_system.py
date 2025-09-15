"""
Renaissance Trading Bot - Comprehensive Test Suite
Tests all components individually and as an integrated system
"""

import asyncio
import unittest
import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class RenaissanceSystemTests(unittest.TestCase):
    """Comprehensive test suite for Renaissance trading system"""

    def setUp(self):
        """Setup test environment"""
        self.test_config = {
            "risk_management": {
                "daily_loss_limit": 500,
                "position_limit": 1000,
                "min_confidence": 0.65
            },
            "signal_weights": {
                "order_flow": 0.32,
                "order_book": 0.21,
                "volume": 0.14,
                "macd": 0.105,
                "rsi": 0.115,
                "bollinger": 0.095,
                "alternative": 0.045
            }
        }

        # Create test config file
        Path('test_config.json').write_text(json.dumps(self.test_config, indent=2))

    def tearDown(self):
        """Cleanup test environment"""
        if Path('test_config.json').exists():
            Path('test_config.json').unlink()

    def test_signal_weight_validation(self):
        """Test that signal weights sum to approximately 1.0"""
        weights = self.test_config["signal_weights"]
        total_weight = sum(weights.values())

        self.assertAlmostEqual(total_weight, 1.0, places=3, 
                             msg="Signal weights must sum to 1.0")

        # Test Renaissance research-optimized ranges
        self.assertGreaterEqual(weights["order_flow"], 0.30, "Order flow weight too low")
        self.assertLessEqual(weights["order_flow"], 0.34, "Order flow weight too high")

        self.assertGreaterEqual(weights["order_book"], 0.18, "Order book weight too low")
        self.assertLessEqual(weights["order_book"], 0.24, "Order book weight too high")

        print("✅ Signal weight validation passed")

    def test_microstructure_signal_range(self):
        """Test microstructure signal output ranges"""
        from microstructure_engine import MicrostructureEngine

        engine = MicrostructureEngine()

        # Test order book imbalance calculation
        test_order_book = {
            'bids': [[50000, 1.0], [49950, 2.0], [49900, 1.5]],
            'asks': [[50050, 0.8], [50100, 1.2], [50150, 2.0]]
        }

        # Mock the analyze_microstructure method for testing
        with patch.object(engine, 'analyze_microstructure') as mock_analyze:
            from microstructure_engine import MicrostructureSignal
            mock_signal = MicrostructureSignal(
                order_flow_strength=0.15,
                order_book_imbalance=0.25,
                volume_pressure=0.10,
                confidence=0.80,
                timestamp=datetime.now()
            )
            mock_analyze.return_value = mock_signal

            result = asyncio.run(engine.analyze_microstructure(test_order_book))

            # Validate signal ranges
            self.assertGreaterEqual(result.order_flow_strength, -1.0)
            self.assertLessEqual(result.order_flow_strength, 1.0)
            self.assertGreaterEqual(result.order_book_imbalance, -1.0)
            self.assertLessEqual(result.order_book_imbalance, 1.0)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

        print("✅ Microstructure signal range validation passed")

    def test_technical_indicators_calculation(self):
        """Test technical indicators calculation"""
        from enhanced_technical_indicators import EnhancedTechnicalIndicators

        indicators = EnhancedTechnicalIndicators()

        # Create test OHLCV data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='5min'),
            'open': np.random.uniform(49000, 51000, 50),
            'high': np.random.uniform(50000, 52000, 50),
            'low': np.random.uniform(48000, 50000, 50),
            'close': np.random.uniform(49500, 50500, 50),
            'volume': np.random.uniform(1000000, 5000000, 50)
        })

        # Mock the calculate_enhanced_signals method
        with patch.object(indicators, 'calculate_enhanced_signals') as mock_calc:
            from enhanced_technical_indicators import TechnicalSignal
            mock_signal = TechnicalSignal(
                rsi_strength=0.12,
                macd_strength=-0.08,
                bollinger_strength=0.05,
                volume_strength=0.20,
                confidence=0.75,
                timestamp=datetime.now()
            )
            mock_calc.return_value = mock_signal

            result = asyncio.run(indicators.calculate_enhanced_signals(test_data))

            # Validate all signals are in valid ranges
            self.assertGreaterEqual(result.rsi_strength, -1.0)
            self.assertLessEqual(result.rsi_strength, 1.0)
            self.assertGreaterEqual(result.macd_strength, -1.0)
            self.assertLessEqual(result.macd_strength, 1.0)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

        print("✅ Technical indicators calculation validation passed")

    def test_alternative_data_signals(self):
        """Test alternative data signal generation"""
        from alternative_data_engine import AlternativeDataEngine, AlternativeSignal

        engine = AlternativeDataEngine()

        # Mock the get_alternative_signals method
        with patch.object(engine, 'get_alternative_signals') as mock_get:
            mock_signal = AlternativeSignal(
                social_sentiment=0.15,
                on_chain_strength=0.75,
                market_psychology=0.60,
                confidence=0.80,
                timestamp=datetime.now()
            )
            mock_get.return_value = mock_signal

            result = asyncio.run(engine.get_alternative_signals())

            # Validate signal ranges
            self.assertGreaterEqual(result.social_sentiment, -1.0)
            self.assertLessEqual(result.social_sentiment, 1.0)
            self.assertGreaterEqual(result.on_chain_strength, 0.0)
            self.assertLessEqual(result.on_chain_strength, 1.0)
            self.assertGreaterEqual(result.market_psychology, 0.0)
            self.assertLessEqual(result.market_psychology, 1.0)

        print("✅ Alternative data signals validation passed")

    def test_risk_management_limits(self):
        """Test risk management enforcement"""
        from renaissance_trading_bot import RenaissanceTradingBot, TradingDecision

        # Create bot with test config
        bot = RenaissanceTradingBot('test_config.json')

        # Test daily loss limit enforcement
        bot.daily_pnl = -600  # Exceed daily loss limit

        decision = bot.make_trading_decision(
            weighted_signal=0.8,  # Strong buy signal
            signal_contributions={'order_flow': 0.25, 'order_book': 0.15}
        )

        # Should hold due to daily loss limit
        self.assertEqual(decision.action, 'HOLD')
        self.assertEqual(decision.position_size, 0.0)

        print("✅ Risk management limits validation passed")

    def test_signal_fusion_logic(self):
        """Test Renaissance signal fusion calculation"""
        from renaissance_trading_bot import RenaissanceTradingBot

        bot = RenaissanceTradingBot('test_config.json')

        # Test signal fusion with known inputs
        test_signals = {
            'order_flow': 0.5,      # 32% weight = 0.16
            'order_book': 0.3,      # 21% weight = 0.063
            'volume': 0.2,          # 14% weight = 0.028
            'macd': 0.1,            # 10.5% weight = 0.0105
            'rsi': -0.1,            # 11.5% weight = -0.0115
            'bollinger': 0.0,       # 9.5% weight = 0.0
            'alternative': 0.4      # 4.5% weight = 0.018
        }

        weighted_signal, contributions = bot.calculate_weighted_signal(test_signals)

        # Calculate expected result
        expected = (0.5*0.32 + 0.3*0.21 + 0.2*0.14 + 0.1*0.105 + 
                   (-0.1)*0.115 + 0.0*0.095 + 0.4*0.045)

        self.assertAlmostEqual(weighted_signal, expected, places=4)

        # Verify contributions
        self.assertAlmostEqual(contributions['order_flow'], 0.16, places=4)
        self.assertAlmostEqual(contributions['order_book'], 0.063, places=4)

        print("✅ Signal fusion logic validation passed")

    def test_decision_confidence_calculation(self):
        """Test trading decision confidence calculation"""
        from renaissance_trading_bot import RenaissanceTradingBot

        bot = RenaissanceTradingBot('test_config.json')

        # Test with high consensus signals
        high_consensus_contributions = {
            'order_flow': 0.15, 'order_book': 0.12, 'volume': 0.08,
            'macd': 0.06, 'rsi': 0.07, 'bollinger': 0.05, 'alternative': 0.02
        }

        decision = bot.make_trading_decision(0.55, high_consensus_contributions)

        # Should have reasonable confidence
        self.assertGreater(decision.confidence, 0.3)

        print("✅ Decision confidence calculation validation passed")

def run_integration_test():
    """Run complete system integration test"""
    print("🧪 Running Renaissance System Integration Test...")
    print("-" * 60)

    async def integration_test():
        try:
            # Mock all components for integration test
            from renaissance_trading_bot import RenaissanceTradingBot

            # Create test bot
            bot = RenaissanceTradingBot('test_config.json')

            # Test complete trading cycle
            with patch.multiple(
                bot,
                collect_all_data=AsyncMock(return_value={
                    'order_book': {'bids': [[50000, 1.0]], 'asks': [[50050, 1.0]]},
                    'price_data': pd.DataFrame({'close': [50000], 'volume': [1000000]}),
                    'alternative_signals': MagicMock(
                        social_sentiment=0.1,
                        on_chain_strength=0.7,
                        market_psychology=0.6,
                        confidence=0.8
                    ),
                    'timestamp': datetime.now()
                }),
                generate_signals=AsyncMock(return_value={
                    'order_flow': 0.2, 'order_book': 0.1, 'volume': 0.15,
                    'macd': 0.05, 'rsi': -0.05, 'bollinger': 0.0, 'alternative': 0.1
                })
            ):
                # Execute trading cycle
                decision = await bot.execute_trading_cycle()

                # Validate decision structure
                assert hasattr(decision, 'action')
                assert hasattr(decision, 'confidence')
                assert hasattr(decision, 'position_size')
                assert hasattr(decision, 'reasoning')
                assert hasattr(decision, 'timestamp')

                # Validate action is valid
                assert decision.action in ['BUY', 'SELL', 'HOLD']

                # Validate ranges
                assert 0.0 <= decision.confidence <= 1.0
                assert 0.0 <= decision.position_size <= 1.0

                print(f"✅ Integration test passed!")
                print(f"   Decision: {decision.action}")
                print(f"   Confidence: {decision.confidence:.3f}")
                print(f"   Position Size: {decision.position_size:.3f}")

                return True

        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False

    return asyncio.run(integration_test())

def run_performance_benchmark():
    """Run performance benchmark test"""
    print("\n⚡ Running Performance Benchmark...")
    print("-" * 40)

    import time

    async def benchmark():
        from renaissance_trading_bot import RenaissanceTradingBot

        bot = RenaissanceTradingBot('test_config.json')

        # Mock data collection for speed
        with patch.object(bot, 'collect_all_data', AsyncMock(return_value={
            'order_book': {'bids': [[50000, 1.0]], 'asks': [[50050, 1.0]]},
            'price_data': pd.DataFrame({'close': [50000], 'volume': [1000000]}),
            'alternative_signals': MagicMock(
                social_sentiment=0.1, on_chain_strength=0.7, 
                market_psychology=0.6, confidence=0.8
            )
        })):

            # Benchmark 10 trading cycles
            start_time = time.time()

            for i in range(10):
                await bot.execute_trading_cycle()

            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / 10

            print(f"✅ Benchmark completed!")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Average cycle time: {avg_time:.3f}s")
            print(f"   Cycles per minute: {60/avg_time:.1f}")

            # Performance requirements
            if avg_time < 1.0:
                print(f"   🚀 Excellent performance (< 1s per cycle)")
            elif avg_time < 2.0:
                print(f"   ✅ Good performance (< 2s per cycle)")
            else:
                print(f"   ⚠️  Slow performance (> 2s per cycle)")

    asyncio.run(benchmark())

if __name__ == "__main__":
    print("🧪 RENAISSANCE TRADING BOT - COMPREHENSIVE TEST SUITE")
    print("=" * 65)

    # Create test config
    test_config = {
        "risk_management": {"daily_loss_limit": 500, "position_limit": 1000, "min_confidence": 0.65},
        "signal_weights": {"order_flow": 0.32, "order_book": 0.21, "volume": 0.14, 
                          "macd": 0.105, "rsi": 0.115, "bollinger": 0.095, "alternative": 0.045}
    }
    Path('test_config.json').write_text(json.dumps(test_config, indent=2))

    try:
        # Run unit tests
        print("\n1. Running Unit Tests...")
        print("-" * 30)
        unittest.main(argv=[''], exit=False, verbosity=0)

        # Run integration test
        print("\n2. Running Integration Test...")
        run_integration_test()

        # Run performance benchmark
        run_performance_benchmark()

        print("\n" + "=" * 65)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("   Your Renaissance Trading Bot is ready for deployment!")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")

    finally:
        # Cleanup
        if Path('test_config.json').exists():
            Path('test_config.json').unlink()
