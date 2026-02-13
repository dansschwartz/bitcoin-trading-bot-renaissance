"""Unit tests for AlertManager."""

import asyncio
import logging
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

from alert_manager import AlertManager


class TestAlertManager(unittest.TestCase):
    """Test alert dispatching and formatting."""

    def setUp(self):
        self.logger = MagicMock(spec=logging.Logger)
        self.config = {"enabled": True, "slack_webhook_env": "SLACK_WEBHOOK_URL"}
        self.am = AlertManager(self.config, self.logger)

    def test_alert_logged(self):
        """Alerts should always be logged regardless of Slack config."""
        asyncio.get_event_loop().run_until_complete(
            self.am.send_alert("WARNING", "Test Alert", "Something happened")
        )
        self.logger.log.assert_called_once()
        args = self.logger.log.call_args
        self.assertIn("Test Alert", str(args))

    def test_critical_alert_uses_critical_level(self):
        asyncio.get_event_loop().run_until_complete(
            self.am.send_alert("CRITICAL", "Kill Switch", "Emergency halt")
        )
        args = self.logger.log.call_args
        self.assertEqual(args[0][0], logging.CRITICAL)

    def test_warning_alert_uses_warning_level(self):
        asyncio.get_event_loop().run_until_complete(
            self.am.send_alert("WARNING", "Drift", "Position drift detected")
        )
        args = self.logger.log.call_args
        self.assertEqual(args[0][0], logging.WARNING)

    def test_alert_history_recorded(self):
        asyncio.get_event_loop().run_until_complete(
            self.am.send_alert("WARNING", "Test", "msg1")
        )
        asyncio.get_event_loop().run_until_complete(
            self.am.send_alert("CRITICAL", "Test2", "msg2")
        )
        history = self.am.get_recent_alerts()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["title"], "Test")
        self.assertEqual(history[1]["level"], "CRITICAL")

    def test_disabled_alerts_not_sent(self):
        config = {"enabled": False}
        am = AlertManager(config, self.logger)
        asyncio.get_event_loop().run_until_complete(
            am.send_alert("CRITICAL", "Title", "msg")
        )
        # When disabled, logger.log should NOT be called for the alert
        self.logger.log.assert_not_called()

    def test_no_slack_webhook_graceful(self):
        """Without a webhook URL, alert should still work (log only)."""
        self.assertEqual(self.am.slack_webhook_url, "")
        asyncio.get_event_loop().run_until_complete(
            self.am.send_alert("WARNING", "Test", "no slack")
        )
        # Should not raise, and should log
        self.logger.log.assert_called_once()

    def test_get_recent_alerts_limit(self):
        for i in range(30):
            asyncio.get_event_loop().run_until_complete(
                self.am.send_alert("WARNING", f"Alert {i}", f"msg {i}")
            )
        recent = self.am.get_recent_alerts(limit=10)
        self.assertEqual(len(recent), 10)
        # Should be the last 10
        self.assertEqual(recent[0]["title"], "Alert 20")


if __name__ == "__main__":
    unittest.main()
