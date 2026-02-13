"""Unit tests for position reconciliation with exchange."""

import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from position_manager import (
    EnhancedPositionManager,
    RiskLimits,
    Position,
    PositionSide,
    PositionStatus,
)


def _make_manager(positions=None):
    """Create an EnhancedPositionManager with a mocked client."""
    client = MagicMock()
    logger = MagicMock()
    risk_limits = RiskLimits()
    pm = EnhancedPositionManager(
        coinbase_client=client,
        risk_limits=risk_limits,
        logger=logger,
    )
    if positions:
        pm.positions = positions
    return pm


class TestReconciliation(unittest.TestCase):
    """Test reconcile_with_exchange()."""

    def test_no_discrepancies(self):
        """When exchange balances match, status should be OK."""
        pm = _make_manager()
        # Mock exchange responses
        pm.client.get_accounts.return_value = {
            "accounts": [
                {"currency": "BTC", "available_balance": {"value": "0.5"}},
                {"currency": "USD", "available_balance": {"value": "10000"}},
            ]
        }
        pm.client.list_orders.return_value = {"orders": []}

        report = pm.reconcile_with_exchange()

        self.assertEqual(report["status"], "OK")
        self.assertEqual(len(report["discrepancies"]), 0)
        self.assertAlmostEqual(report["exchange_balances"]["BTC"], 0.5)

    def test_detects_balance_mismatch(self):
        """When tracked position > exchange balance, detect mismatch."""
        pos = Position(
            position_id="pos_1",
            product_id="BTC-USD",
            side=PositionSide.LONG,
            size=1.0,
            entry_price=50000.0,
            status=PositionStatus.OPEN,
        )
        pm = _make_manager(positions={"pos_1": pos})

        # Exchange only has 0.1 BTC but we track 1.0
        pm.client.get_accounts.return_value = {
            "accounts": [
                {"currency": "BTC", "available_balance": {"value": "0.1"}},
            ]
        }
        pm.client.list_orders.return_value = {"orders": []}

        report = pm.reconcile_with_exchange()

        self.assertEqual(report["status"], "MISMATCH")
        self.assertEqual(len(report["discrepancies"]), 1)
        self.assertEqual(report["discrepancies"][0]["type"], "BALANCE_MISMATCH")
        self.assertAlmostEqual(report["discrepancies"][0]["expected"], 1.0)
        self.assertAlmostEqual(report["discrepancies"][0]["actual"], 0.1)
        pm.logger.critical.assert_called()

    def test_no_mismatch_within_tolerance(self):
        """Exchange balance within 95% of tracked should be OK."""
        pos = Position(
            position_id="pos_1",
            product_id="BTC-USD",
            side=PositionSide.LONG,
            size=1.0,
            entry_price=50000.0,
            status=PositionStatus.OPEN,
        )
        pm = _make_manager(positions={"pos_1": pos})

        # 0.96 BTC >= 1.0 * 0.95, so within tolerance
        pm.client.get_accounts.return_value = {
            "accounts": [
                {"currency": "BTC", "available_balance": {"value": "0.96"}},
            ]
        }
        pm.client.list_orders.return_value = {"orders": []}

        report = pm.reconcile_with_exchange()

        self.assertEqual(report["status"], "OK")
        self.assertEqual(len(report["discrepancies"]), 0)

    def test_exchange_error_returns_error_status(self):
        """If exchange API call fails, status should be ERROR."""
        pm = _make_manager()
        pm.client.get_accounts.side_effect = Exception("API timeout")

        report = pm.reconcile_with_exchange()

        self.assertEqual(report["status"], "ERROR")
        self.assertIn("API timeout", report["error"])

    def test_no_positions_ok(self):
        """With no tracked positions, reconciliation should be OK."""
        pm = _make_manager()
        pm.client.get_accounts.return_value = {"accounts": []}
        pm.client.list_orders.return_value = {"orders": []}

        report = pm.reconcile_with_exchange()

        self.assertEqual(report["status"], "OK")


if __name__ == "__main__":
    unittest.main()
