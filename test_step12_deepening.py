import unittest
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime
from renaissance_trading_bot import RenaissanceTradingBot

class TestStep12Deepening(unittest.TestCase):
    def setUp(self):
        self.config = {
            "real_time_pipeline": {
                "enabled": True,
                "apply_to_confidence": True
            },
            "risk_management": {
                "min_confidence": 0.5
            }
        }
        with patch('renaissance_trading_bot.RenaissanceTradingBot._load_config', return_value=self.config):
            self.bot = RenaissanceTradingBot()

    def test_confidence_overlay_boost(self):
        # Weighted signal is positive (Buy)
        weighted_signal = 0.5
        signal_contributions = {'order_flow': 0.5}
        
        # Models also positive (agree)
        real_time_result = {
            "predictions": {
                "Model1": 0.8,
                "Model2": 0.9
            }
        }
        
        # Base confidence (without overlay)
        # abs(0.5) + (1.0 - std([0.5])) = 0.5 + 1.0 = 1.5 -> / 2 = 0.75
        
        decision = self.bot.make_trading_decision(weighted_signal, signal_contributions, real_time_result=real_time_result)
        
        # Should be boosted (max +0.05)
        self.assertGreater(decision.confidence, 0.75)
        print(f"Boosted confidence: {decision.confidence:.4f}")

    def test_confidence_overlay_reduction(self):
        # Weighted signal is positive (Buy)
        weighted_signal = 0.5
        signal_contributions = {'order_flow': 0.5}
        
        # Models negative (disagree)
        real_time_result = {
            "predictions": {
                "Model1": -0.8,
                "Model2": -0.9
            }
        }
        
        decision = self.bot.make_trading_decision(weighted_signal, signal_contributions, real_time_result=real_time_result)
        
        # Should be reduced
        self.assertLess(decision.confidence, 0.75)
        print(f"Reduced confidence: {decision.confidence:.4f}")

    def test_no_overlay_when_disabled(self):
        self.bot.config["real_time_pipeline"]["apply_to_confidence"] = False
        
        weighted_signal = 0.5
        signal_contributions = {'order_flow': 0.5}
        real_time_result = {"predictions": {"Model1": 0.8}}
        
        decision = self.bot.make_trading_decision(weighted_signal, signal_contributions, real_time_result=real_time_result)
        self.assertEqual(decision.confidence, 0.75)

if __name__ == '__main__':
    unittest.main()
