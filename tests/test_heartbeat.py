"""Unit tests for heartbeat / health check (Step 15)."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestHeartbeat(unittest.TestCase):
    """Verify _write_heartbeat writes expected fields."""

    def _make_bot_with_mocks(self, heartbeat_path):
        """Create a bot instance with heavy deps mocked out."""
        with patch("renaissance_trading_bot.RenaissanceTradingBot.__init__", return_value=None):
            from renaissance_trading_bot import RenaissanceTradingBot
            bot = RenaissanceTradingBot.__new__(RenaissanceTradingBot)
            bot._background_tasks = []
            bot._weights_lock = asyncio.Lock()
            bot._killed = False
            bot.decision_history = [None] * 42  # 42 decisions
            bot.logger = MagicMock()
            bot.coinbase_client = MagicMock()
            bot.coinbase_client.paper_trading = True
            # Override HEARTBEAT_FILE with a Path object (matches production code)
            bot.HEARTBEAT_FILE = Path(heartbeat_path)
            # Bind the real method
            bot._write_heartbeat = RenaissanceTradingBot._write_heartbeat.__get__(bot)
            return bot

    def test_heartbeat_file_written(self):
        """Heartbeat file should be created with expected fields."""
        tmpdir = tempfile.mkdtemp()
        hb_path = os.path.join(tmpdir, "heartbeat.json")

        bot = self._make_bot_with_mocks(hb_path)
        bot._write_heartbeat()

        self.assertTrue(os.path.exists(hb_path))
        with open(hb_path) as f:
            data = json.load(f)

        self.assertTrue(data["alive"])
        self.assertIn("timestamp", data)
        self.assertEqual(data["cycle_count"], 42)
        self.assertFalse(data["killed"])
        self.assertTrue(data["paper_mode"])

    def test_heartbeat_updates_on_kill(self):
        """After kill switch, heartbeat should reflect killed=True."""
        tmpdir = tempfile.mkdtemp()
        hb_path = os.path.join(tmpdir, "heartbeat.json")

        bot = self._make_bot_with_mocks(hb_path)
        bot._killed = True
        bot._write_heartbeat()

        with open(hb_path) as f:
            data = json.load(f)

        self.assertTrue(data["killed"])

    def test_heartbeat_directory_created(self):
        """Heartbeat should create parent directory if it doesn't exist."""
        tmpdir = tempfile.mkdtemp()
        hb_path = os.path.join(tmpdir, "subdir", "heartbeat.json")

        bot = self._make_bot_with_mocks(hb_path)
        bot._write_heartbeat()

        self.assertTrue(os.path.exists(hb_path))


if __name__ == "__main__":
    unittest.main()
