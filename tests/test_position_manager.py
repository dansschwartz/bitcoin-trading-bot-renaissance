"""Unit tests for EnhancedPositionManager core logic."""

import unittest
from datetime import datetime
from typing import Dict, Any, Optional

from position_manager import (
    EnhancedPositionManager,
    RiskLimits,
    Position,
    PositionSide,
    PositionStatus,
)


class MockClient:
    """Minimal mock Coinbase client for unit tests."""

    def __init__(self):
        self.orders = []

    def create_market_order(self, product_id: str, side: str,
                            size: Optional[float] = None,
                            funds: Optional[float] = None,
                            client_order_id: Optional[str] = None) -> Dict[str, Any]:
        self.orders.append({"type": "market", "product_id": product_id, "side": side, "size": size})
        return {"order": {"order_id": f"mock_{len(self.orders)}", "status": "FILLED"}}

    def create_limit_order(self, product_id: str, side: str, size: float,
                           price: float, post_only: bool = False,
                           client_order_id: Optional[str] = None) -> Dict[str, Any]:
        self.orders.append({"type": "limit", "product_id": product_id, "side": side, "size": size, "price": price})
        return {"order": {"order_id": f"mock_limit_{len(self.orders)}", "status": "OPEN"}}

    def cancel_order(self, order_id: str = "") -> Dict[str, Any]:
        return {"success": True}

    def get_product(self, product_id: str = "BTC-USD") -> Dict[str, Any]:
        return {"price": "50000.00"}


class TestOpenPosition(unittest.TestCase):
    """Tests for opening positions and risk limit enforcement."""

    def setUp(self):
        self.client = MockClient()
        self.limits = RiskLimits(
            max_position_size_usd=1000.0,
            max_daily_loss_usd=500.0,
            max_total_exposure_usd=2000.0,
            max_positions_per_product=3,
            max_total_positions=5,
        )
        self.pm = EnhancedPositionManager(self.client, self.limits)

    def test_open_position_success(self):
        success, msg, pos = self.pm.open_position("BTC-USD", "LONG", 0.01, entry_price=50000.0)
        self.assertTrue(success)
        self.assertIsNotNone(pos)
        self.assertEqual(pos.side, PositionSide.LONG)
        self.assertEqual(pos.status, PositionStatus.OPEN)
        self.assertIn(pos.position_id, self.pm.positions)
        self.assertEqual(self.pm.stats["positions_opened"], 1)

    def test_open_position_risk_limit_rejected(self):
        # 0.03 BTC @ 50000 = $1500 > $1000 limit
        success, msg, pos = self.pm.open_position("BTC-USD", "LONG", 0.03, entry_price=50000.0)
        self.assertFalse(success)
        self.assertIsNone(pos)
        self.assertIn("exceeds limit", msg)
        self.assertEqual(self.pm.stats["risk_limit_violations"], 1)

    def test_daily_loss_limit_blocks_new_trades(self):
        self.pm.daily_pnl = -600.0  # Already past -$500 limit
        success, msg, pos = self.pm.open_position("BTC-USD", "LONG", 0.001, entry_price=50000.0)
        self.assertFalse(success)
        self.assertIn("Daily loss", msg)

    def test_stop_loss_price_calculation_long(self):
        success, _, pos = self.pm.open_position("BTC-USD", "LONG", 0.01, entry_price=50000.0)
        self.assertTrue(success)
        expected_sl = 50000.0 * (1 - self.limits.stop_loss_percentage / 100)
        self.assertAlmostEqual(pos.stop_loss_price, expected_sl, places=2)

    def test_stop_loss_price_calculation_short(self):
        success, _, pos = self.pm.open_position("BTC-USD", "SHORT", 0.01, entry_price=50000.0)
        self.assertTrue(success)
        expected_sl = 50000.0 * (1 + self.limits.stop_loss_percentage / 100)
        self.assertAlmostEqual(pos.stop_loss_price, expected_sl, places=2)


class TestEmergencyStop(unittest.TestCase):
    """Tests for emergency stop and position closure."""

    def setUp(self):
        self.client = MockClient()
        self.pm = EnhancedPositionManager(self.client, RiskLimits(
            max_position_size_usd=5000.0,
            max_total_exposure_usd=10000.0,
        ))

    def test_emergency_stop_closes_all_positions(self):
        self.pm.open_position("BTC-USD", "LONG", 0.01, entry_price=50000.0)
        self.pm.open_position("ETH-USD", "SHORT", 0.1, entry_price=3000.0)
        self.assertEqual(len(self.pm.positions), 2)

        self.pm.set_emergency_stop(True, "Test shutdown")

        self.assertTrue(self.pm.emergency_stop)
        self.assertEqual(len(self.pm.positions), 0)
        self.assertEqual(len(self.pm.closed_positions), 2)
        self.assertEqual(self.pm.stats["emergency_stops"], 1)

    def test_emergency_stop_blocks_new_trades(self):
        self.pm.set_emergency_stop(True, "Blocked")
        success, msg, _ = self.pm.open_position("BTC-USD", "LONG", 0.001, entry_price=50000.0)
        self.assertFalse(success)
        self.assertIn("Emergency stop", msg)

    def test_close_position_updates_pnl(self):
        _, _, pos = self.pm.open_position("BTC-USD", "LONG", 0.01, entry_price=50000.0)
        # Simulate price going up
        pos.current_price = 51000.0
        success, _ = self.pm.close_position(pos.position_id, "Take profit")
        self.assertTrue(success)
        # PnL = (51000-50000) * 0.01 = $10
        self.assertAlmostEqual(self.pm.daily_pnl, 10.0, places=2)


if __name__ == "__main__":
    unittest.main()
