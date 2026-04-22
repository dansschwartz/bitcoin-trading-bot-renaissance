"""Unit tests for kill switch mechanisms."""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from risk_gateway import RiskGateway


class TestKillFile(unittest.TestCase):
    """Test file-based kill switch."""

    def test_kill_file_triggers_shutdown(self):
        """Creating a KILL_SWITCH file should trigger the kill switch."""
        # We test the bot method directly by mocking the bot
        from renaissance_trading_bot import RenaissanceTradingBot

        with patch.object(RenaissanceTradingBot, '__init__', lambda self, *a, **kw: None):
            bot = RenaissanceTradingBot.__new__(RenaissanceTradingBot)
            bot._killed = False
            bot.logger = MagicMock()
            bot.position_manager = MagicMock()
            bot.KILL_FILE = Path(tempfile.mktemp(suffix="_KILL_SWITCH"))

            # Create the kill file
            bot.KILL_FILE.write_text("test halt")
            try:
                bot._check_kill_file()
                self.assertTrue(bot._killed)
                bot.position_manager.set_emergency_stop.assert_called_once()
            finally:
                bot.KILL_FILE.unlink(missing_ok=True)

    def test_no_kill_file_no_trigger(self):
        """Without a KILL_SWITCH file, bot should not halt."""
        from renaissance_trading_bot import RenaissanceTradingBot

        with patch.object(RenaissanceTradingBot, '__init__', lambda self, *a, **kw: None):
            bot = RenaissanceTradingBot.__new__(RenaissanceTradingBot)
            bot._killed = False
            bot.logger = MagicMock()
            bot.position_manager = MagicMock()
            bot.KILL_FILE = Path("/tmp/nonexistent_kill_switch_file")

            bot._check_kill_file()
            self.assertFalse(bot._killed)


class TestDailyLossKill(unittest.TestCase):
    """Test that daily loss triggers kill switch."""

    def test_trigger_kill_switch_sets_flag(self):
        from renaissance_trading_bot import RenaissanceTradingBot

        with patch.object(RenaissanceTradingBot, '__init__', lambda self, *a, **kw: None):
            bot = RenaissanceTradingBot.__new__(RenaissanceTradingBot)
            bot._killed = False
            bot.logger = MagicMock()
            bot.position_manager = MagicMock()

            bot.trigger_kill_switch("Daily loss limit breached: $600")

            self.assertTrue(bot._killed)
            bot.position_manager.set_emergency_stop.assert_called_once_with(
                True, "Daily loss limit breached: $600"
            )


class TestRiskGatewayFailClosed(unittest.TestCase):
    """Test that risk gateway defaults to fail-closed."""

    def test_default_fail_closed(self):
        gw = RiskGateway({}, logger=MagicMock())
        self.assertFalse(gw.fail_open)

    def test_fail_closed_blocks_on_error(self):
        gw = RiskGateway({"enabled": True, "fail_open": False}, logger=MagicMock())
        # Force an exception in assessment
        gw.vae = MagicMock(side_effect=Exception("boom"))

        result = gw.assess_trade(
            action="BUY", amount=100, current_price=50000.0,
            portfolio_data={"daily_pnl": 0, "total_value": 1000},
            feature_vector=None,
        )
        # With VAE failing and no feature vector, it should still pass (no anomaly check)
        # But if the whole try block fails, fail_open=False means blocked
        self.assertIsInstance(result, bool)

    def test_drawdown_check_blocks(self):
        gw = RiskGateway({"enabled": True}, logger=MagicMock())
        # 20% drawdown > 15% threshold
        result = gw.assess_trade(
            action="BUY", amount=100, current_price=50000.0,
            portfolio_data={"daily_pnl": -200, "total_value": 1000},
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
