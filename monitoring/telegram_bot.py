"""
Telegram Alert System for the Renaissance Trading Bot.

Provides asynchronous Telegram message delivery with rate limiting,
deduplication, and a console fallback when no bot token is configured.

Usage:
    alerter = TelegramAlerter(bot_token="123:ABC", chat_id="456")
    await alerter.send_critical("Exchange API down!")

When *bot_token* is ``None`` or empty every alert is printed to the
console logger instead of being sent over the network.  This keeps the
rest of the system working identically during local development.

All public methods catch **every** exception internally so that alerting
failures can never crash the trading system.
"""

import asyncio
import hashlib
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MESSAGES_PER_MINUTE: int = 10
DUPLICATE_WINDOW_SECONDS: float = 60.0
TELEGRAM_API_BASE: str = "https://api.telegram.org"


class TelegramAlerter:
    """Sends formatted alerts to a Telegram chat or falls back to console.

    Args:
        bot_token: Telegram Bot API token.  Falls back to the
                   ``TELEGRAM_BOT_TOKEN`` environment variable.  If still
                   empty, console-only mode is activated.
        chat_id:   Telegram chat / group / channel ID.  Falls back to the
                   ``TELEGRAM_CHAT_ID`` environment variable.
        config:    Optional dict with keys ``bot_token`` and ``chat_id``
                   (used when constructing from the bot config file).
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Resolve token: explicit arg > config dict > env var
        if bot_token:
            self._bot_token: Optional[str] = bot_token
        elif config and config.get("bot_token"):
            self._bot_token = config["bot_token"]
        else:
            self._bot_token = os.environ.get("TELEGRAM_BOT_TOKEN") or None

        if chat_id:
            self._chat_id: Optional[str] = chat_id
        elif config and config.get("chat_id"):
            self._chat_id = config["chat_id"]
        else:
            self._chat_id = os.environ.get("TELEGRAM_CHAT_ID") or None

        self._console_only: bool = not (self._bot_token and self._chat_id)

        # Rate-limiting state: timestamps of recently sent messages
        self._send_timestamps: Deque[float] = deque()

        # Deduplication state: content_hash -> last_sent_timestamp
        self._recent_hashes: Dict[str, float] = {}

        if self._console_only:
            logger.info(
                "TelegramAlerter running in CONSOLE-ONLY mode "
                "(no bot_token/chat_id configured)"
            )
        else:
            logger.info("TelegramAlerter configured for Telegram delivery")

    # ------------------------------------------------------------------
    # Public convenience methods
    # ------------------------------------------------------------------

    async def send_trade_executed(self, trade_result: dict) -> None:
        """Send a notification after every executed trade.

        Args:
            trade_result: Dict with keys such as ``symbol``, ``side``,
                          ``quantity``, ``price``, ``pnl``, ``exchange``, etc.
        """
        try:
            symbol = trade_result.get("symbol", trade_result.get("product_id", "N/A"))
            side = trade_result.get("side", "N/A")
            qty = trade_result.get("quantity", trade_result.get("size", "N/A"))
            price = trade_result.get("price", "N/A")
            pnl = trade_result.get("pnl", "")
            exchange = trade_result.get("exchange", "")
            status = trade_result.get("status", "")

            pnl_line = f"\nP&L: <b>{pnl}</b>" if pnl else ""
            exchange_line = f"\nExchange: {exchange}" if exchange else ""
            status_line = f"\nStatus: {status}" if status else ""

            text = (
                f"<b>Trade Executed</b>\n"
                f"Symbol: {symbol}\n"
                f"Side: <b>{side}</b>\n"
                f"Qty: {qty}\n"
                f"Price: {price}"
                f"{pnl_line}"
                f"{exchange_line}"
                f"{status_line}"
            )
            await self._send("INFO", text)
        except Exception:
            logger.exception("Failed to send trade_executed alert")

    async def send_daily_summary(self, daily: dict) -> None:
        """Send a daily P&L summary.

        Args:
            daily: Dict with keys such as ``date``, ``net_profit_usd``,
                   ``total_trades``, ``winning_trades``, ``losing_trades``,
                   ``max_drawdown_usd``, ``total_equity_usd``, etc.
        """
        try:
            date = daily.get("date", "N/A")
            net = daily.get("net_profit_usd", 0.0)
            gross = daily.get("gross_profit_usd", 0.0)
            fees = daily.get("total_fees_usd", 0.0)
            total = daily.get("total_trades", 0)
            wins = daily.get("winning_trades", 0)
            losses = daily.get("losing_trades", 0)
            failed = daily.get("failed_trades", 0)
            dd = daily.get("max_drawdown_usd", 0.0)
            equity = daily.get("total_equity_usd", "N/A")

            emoji_prefix = "+" if net >= 0 else ""

            text = (
                f"<b>Daily Summary — {date}</b>\n"
                f"Net P&L: <b>{emoji_prefix}{net:.2f} USD</b>\n"
                f"Gross: {gross:.2f} | Fees: {fees:.2f}\n"
                f"Trades: {total} (W:{wins} L:{losses} F:{failed})\n"
                f"Max Drawdown: {dd:.2f} USD\n"
                f"Equity: {equity}"
            )
            await self._send("INFO", text)
        except Exception:
            logger.exception("Failed to send daily_summary alert")

    async def send_warning(self, message: str) -> None:
        """Send a WARNING-level alert.

        Args:
            message: Free-text warning description.
        """
        try:
            text = f"<b>WARNING</b>\n{message}"
            await self._send("WARNING", text)
        except Exception:
            logger.exception("Failed to send warning alert")

    async def send_critical(self, message: str) -> None:
        """Send a CRITICAL-level alert.  Bypasses rate limits.

        Args:
            message: Free-text description of the critical event.
        """
        try:
            text = f"<b>CRITICAL</b>\n{message}"
            await self._send("CRITICAL", text)
        except Exception:
            logger.exception("Failed to send critical alert")

    async def send_hourly_heartbeat(self, stats: dict) -> None:
        """Send an hourly heartbeat / status message.

        Args:
            stats: Dict with runtime statistics (e.g. ``uptime``,
                   ``trades_1h``, ``pnl_1h``, ``open_positions``).
        """
        try:
            uptime = stats.get("uptime", "N/A")
            trades = stats.get("trades_1h", 0)
            pnl = stats.get("pnl_1h", 0.0)
            positions = stats.get("open_positions", 0)
            exchanges = stats.get("exchanges_healthy", "N/A")

            text = (
                f"<b>Heartbeat</b>\n"
                f"Uptime: {uptime}\n"
                f"Trades (1h): {trades}\n"
                f"P&L (1h): {pnl:.2f} USD\n"
                f"Open Positions: {positions}\n"
                f"Exchanges Healthy: {exchanges}"
            )
            await self._send("INFO", text)
        except Exception:
            logger.exception("Failed to send hourly_heartbeat alert")

    async def send_system_event(self, event: str, details: str) -> None:
        """Send a system state-change notification.

        Args:
            event:   Short event label (e.g. ``STARTUP``, ``SHUTDOWN``).
            details: Longer human-readable description.
        """
        try:
            text = (
                f"<b>System Event — {event}</b>\n"
                f"{details}"
            )
            await self._send("INFO", text)
        except Exception:
            logger.exception("Failed to send system_event alert")

    # ------------------------------------------------------------------
    # Internal delivery logic
    # ------------------------------------------------------------------

    async def _send(self, level: str, html_text: str) -> None:
        """Route a message to Telegram or the console fallback.

        Critical alerts bypass both rate limiting and deduplication.

        Args:
            level:     One of ``INFO``, ``WARNING``, ``CRITICAL``.
            html_text: HTML-formatted message body.
        """
        is_critical = level == "CRITICAL"

        # --- deduplication (skip for critical) ---
        if not is_critical:
            content_hash = self._hash(html_text)
            now = time.monotonic()
            last_sent = self._recent_hashes.get(content_hash)
            if last_sent is not None and (now - last_sent) < DUPLICATE_WINDOW_SECONDS:
                logger.debug("Duplicate alert suppressed (hash=%s)", content_hash[:8])
                return
            self._recent_hashes[content_hash] = now
            # Evict old entries to avoid unbounded growth
            self._evict_old_hashes(now)

        # --- rate limiting (skip for critical) ---
        if not is_critical and not self._check_rate_limit():
            logger.warning("Alert rate-limited (level=%s)", level)
            return

        # --- delivery ---
        if self._console_only:
            self._log_to_console(level, html_text)
        else:
            await self._send_telegram(html_text)

    def _check_rate_limit(self) -> bool:
        """Return True if the message is within the rate limit window."""
        now = time.monotonic()
        # Remove timestamps older than 60 seconds
        while self._send_timestamps and (now - self._send_timestamps[0]) > 60.0:
            self._send_timestamps.popleft()
        if len(self._send_timestamps) >= MAX_MESSAGES_PER_MINUTE:
            return False
        self._send_timestamps.append(now)
        return True

    @staticmethod
    def _hash(text: str) -> str:
        """Return a short SHA-256 hex digest for deduplication."""
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def _evict_old_hashes(self, now: float) -> None:
        """Remove deduplication entries older than the window."""
        expired = [
            h for h, ts in self._recent_hashes.items()
            if (now - ts) > DUPLICATE_WINDOW_SECONDS
        ]
        for h in expired:
            del self._recent_hashes[h]

    # ------------------------------------------------------------------
    # Console fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _log_to_console(level: str, html_text: str) -> None:
        """Pretty-print the alert to the console logger."""
        # Strip HTML tags for readability
        import re
        plain = re.sub(r"<[^>]+>", "", html_text)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        separator = "-" * 60
        log_fn = logger.warning if level in ("WARNING", "CRITICAL") else logger.info
        log_fn(
            "\n%s\n[TELEGRAM %s] %s\n%s\n%s",
            separator, level, ts, plain, separator,
        )

    # ------------------------------------------------------------------
    # Telegram HTTP delivery
    # ------------------------------------------------------------------

    async def _send_telegram(self, html_text: str) -> None:
        """POST the message to the Telegram Bot API using aiohttp.

        All exceptions are caught and logged — never propagated.
        """
        url = f"{TELEGRAM_API_BASE}/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": html_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning(
                            "Telegram API returned %s: %s", resp.status, body[:200]
                        )
                    else:
                        logger.debug("Telegram message sent successfully")
        except ImportError:
            logger.warning(
                "aiohttp is not installed — falling back to console output"
            )
            self._log_to_console("INFO", html_text)
        except asyncio.CancelledError:
            raise  # Let cancellation propagate normally
        except Exception:
            logger.exception("Failed to deliver Telegram message")
