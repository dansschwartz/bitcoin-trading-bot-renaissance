"""
Unit tests for RegimeOverlay and RiskGateway adapters.
"""

import unittest
import pandas as pd
import numpy as np
from regime_overlay import RegimeOverlay
from risk_gateway import RiskGateway

class TestAdapters(unittest.TestCase):
    def setUp(self):
        self.config = {
            "regime_overlay": {"enabled": True, "consciousness_boost": 0.142},
            "risk_gateway": {"enabled": True, "fail_open": True, "max_portfolio_value": 1000.0}
        }

    def test_regime_overlay_weight_adjustment(self):
        overlay = RegimeOverlay(self.config["regime_overlay"])
        
        # Mock a regime
        overlay.current_regime = {
            'regime_weights': {
                'volatility_weight': 1.5,
                'trend_weight': 1.2,
                'liquidity_weight': 0.8
            },
            'confidence_score': 0.8
        }
        
        base_weights = {
            'macd': 0.1,
            'rsi': 0.1,
            'bollinger': 0.1,
            'order_flow': 0.4,
            'order_book': 0.3
        }
        
        adjusted = overlay.get_adjusted_weights(base_weights)
        
        # Verify normalization
        self.assertAlmostEqual(sum(adjusted.values()), 1.0)
        
        # MACD should be boosted by trend_weight (1.2)
        # Bollinger should be boosted by vol_weight (1.5)
        # Order flow should be reduced by liq_weight (0.8)
        self.assertGreater(adjusted['bollinger'], adjusted['macd'])
        self.assertLess(adjusted['order_flow'] / 0.4, adjusted['bollinger'] / 0.1)

    def test_risk_gateway_assess_trade(self):
        gateway = RiskGateway(self.config["risk_gateway"])
        
        portfolio_data = {
            'total_value': 1000.0,
            'daily_pnl': 0.0,
            'positions': {'BTC': 0.0},
            'current_price': 50000.0,
            'historical_returns': [0.001] * 20, # Very stable returns
            'market_data': {
                'volume': 100000000,
                'spread': 0.0001,
                'market_depth': 100000000
            }
        }
        
        # Should allow a normal trade if metrics are okay
        is_allowed = gateway.assess_trade("BUY", 0.1, 50000.0, portfolio_data)
        self.assertTrue(is_allowed)

    def test_risk_gateway_fail_open(self):
        from unittest.mock import patch
        gateway = RiskGateway(self.config["risk_gateway"])
        
        with patch.object(gateway.risk_manager, 'calculate_portfolio_risk_metrics', side_effect=Exception("Risk Manager Error")):
            gateway.fail_open = True
            is_allowed = gateway.assess_trade("BUY", 0.1, 50000.0, {})
            self.assertTrue(is_allowed) # Fails open

            gateway.fail_open = False
            is_allowed = gateway.assess_trade("BUY", 0.1, 50000.0, {})
            self.assertFalse(is_allowed) # Fails closed

if __name__ == '__main__':
    unittest.main()
