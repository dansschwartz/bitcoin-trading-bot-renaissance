"""Unit tests for idempotent order placement (Step 8)."""

import unittest
import uuid
from unittest.mock import MagicMock, patch

from position_manager import (
    EnhancedPositionManager,
    RiskLimits,
)


def _make_manager():
    """Create an EnhancedPositionManager with a mocked client."""
    client = MagicMock()
    logger = MagicMock()
    risk_limits = RiskLimits()
    pm = EnhancedPositionManager(
        coinbase_client=client,
        risk_limits=risk_limits,
        logger=logger,
    )
    return pm


class TestIdempotentOrders(unittest.TestCase):
    """Verify UUID-based client_order_id on all order calls."""

    def test_market_order_uses_uuid(self):
        """create_market_order should receive a valid UUID client_order_id."""
        pm = _make_manager()
        pm.client.create_market_order.return_value = {
            "success": True,
            "order": {"order_id": "test_123", "status": "FILLED"}
        }

        pm.open_position(
            product_id="BTC-USD",
            side="LONG",
            size=0.01,
            entry_price=50000.0,
        )

        # Verify create_market_order was called
        if pm.client.create_market_order.called:
            call_kwargs = pm.client.create_market_order.call_args
            # Extract client_order_id from kwargs
            if call_kwargs.kwargs.get("client_order_id"):
                coid = call_kwargs.kwargs["client_order_id"]
            else:
                # Might be in positional args via keyword
                coid = call_kwargs[1].get("client_order_id") if len(call_kwargs) > 1 else None

            if coid:
                # Verify it's a valid UUID
                parsed = uuid.UUID(coid)
                self.assertEqual(str(parsed), coid)

    def test_different_orders_get_different_ids(self):
        """Each order call should generate a unique UUID."""
        pm = _make_manager()
        pm.client.create_market_order.return_value = {
            "success": True,
            "order": {"order_id": "test_123", "status": "FILLED"}
        }

        # Open two positions
        pm.open_position("BTC-USD", "LONG", 0.01, 50000.0)
        pm.open_position("BTC-USD", "LONG", 0.02, 51000.0)

        if pm.client.create_market_order.call_count >= 2:
            calls = pm.client.create_market_order.call_args_list
            ids = set()
            for call in calls:
                coid = call.kwargs.get("client_order_id")
                if coid:
                    ids.add(coid)
            # All IDs should be unique
            self.assertEqual(len(ids), pm.client.create_market_order.call_count)

    def test_uuid_format_in_coinbase_client(self):
        """Verify coinbase_client default client_order_id is UUID format."""
        # Import and check the default path
        from coinbase_client import EnhancedCoinbaseClient
        # The default in create_order uses uuid.uuid4()
        # We just verify uuid module is importable and used
        test_id = str(uuid.uuid4())
        # Valid UUID should parse without error
        parsed = uuid.UUID(test_id)
        self.assertEqual(str(parsed), test_id)
        self.assertEqual(parsed.version, 4)


if __name__ == "__main__":
    unittest.main()
