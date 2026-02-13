"""Unit tests for graceful shutdown (Step 11)."""

import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock


class TestGracefulShutdown(unittest.TestCase):
    """Verify task tracking and shutdown behavior."""

    def _make_bot_with_mocks(self):
        """Create a bot instance with heavy deps mocked out."""
        with patch("renaissance_trading_bot.RenaissanceTradingBot.__init__", return_value=None):
            from renaissance_trading_bot import RenaissanceTradingBot
            bot = RenaissanceTradingBot.__new__(RenaissanceTradingBot)
            bot._background_tasks = []
            bot._weights_lock = asyncio.Lock()
            bot._killed = False
            bot.logger = MagicMock()
            # Bind the real methods
            bot._track_task = RenaissanceTradingBot._track_task.__get__(bot)
            bot._shutdown = RenaissanceTradingBot._shutdown.__get__(bot)
            return bot

    def test_track_task_adds_to_list(self):
        """_track_task should add a task to _background_tasks."""
        bot = self._make_bot_with_mocks()

        async def _run():
            async def dummy():
                await asyncio.sleep(10)

            task = bot._track_task(dummy())
            self.assertEqual(len(bot._background_tasks), 1)
            self.assertIs(bot._background_tasks[0], task)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.get_event_loop().run_until_complete(_run())

    def test_shutdown_cancels_tasks(self):
        """_shutdown should cancel all tracked tasks."""
        bot = self._make_bot_with_mocks()

        async def _run():
            async def long_task():
                await asyncio.sleep(3600)

            bot._track_task(long_task())
            bot._track_task(long_task())
            self.assertEqual(len(bot._background_tasks), 2)

            await bot._shutdown()
            # All tasks should be done (cancelled)
            for t in bot._background_tasks:
                self.assertTrue(t.done())

        asyncio.get_event_loop().run_until_complete(_run())

    def test_shutdown_clears_task_list(self):
        """After shutdown, _background_tasks should be empty."""
        bot = self._make_bot_with_mocks()

        async def _run():
            async def short_task():
                return 42

            bot._track_task(short_task())
            await asyncio.sleep(0.1)  # Let it complete
            await bot._shutdown()
            self.assertEqual(len(bot._background_tasks), 0)

        asyncio.get_event_loop().run_until_complete(_run())


if __name__ == "__main__":
    unittest.main()
