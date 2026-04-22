"""
Alert Manager — Slack webhook and log-based alerting for production monitoring.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AlertManager:
    """Sends alerts via Slack webhook and logs."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = config.get("enabled", True)
        webhook_env = config.get("slack_webhook_env", "SLACK_WEBHOOK_URL")
        self.slack_webhook_url = os.environ.get(webhook_env, "")
        self.alert_history: List[Dict[str, Any]] = []
        self.logger.info(
            f"AlertManager initialized (enabled={self.enabled}, "
            f"slack={'configured' if self.slack_webhook_url else 'not configured'})"
        )

    async def send_alert(self, level: str, title: str, message: str):
        """Send an alert via all configured channels."""
        if not self.enabled:
            return

        alert = {
            "level": level,
            "title": title,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.alert_history.append(alert)

        # Always log
        log_level = logging.CRITICAL if level == "CRITICAL" else logging.WARNING
        self.logger.log(log_level, f"ALERT [{level}] {title}: {message}")

        # Slack
        if self.slack_webhook_url:
            await self._send_slack(alert)

    async def _send_slack(self, alert: Dict[str, Any]):
        """Post alert to a Slack incoming webhook."""
        try:
            import aiohttp
            color = "#FF0000" if alert["level"] == "CRITICAL" else "#FFA500"
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"[{alert['level']}] {alert['title']}",
                    "text": alert["message"],
                    "ts": alert["timestamp"],
                }]
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        self.logger.warning(f"Slack webhook returned {resp.status}")
        except ImportError:
            # aiohttp not installed — try requests as fallback
            try:
                import requests
                payload = {"text": f"[{alert['level']}] {alert['title']}: {alert['message']}"}
                requests.post(self.slack_webhook_url, json=payload, timeout=10)
            except Exception as e:
                self.logger.warning(f"Slack alert failed (requests): {e}")
        except Exception as e:
            self.logger.warning(f"Slack alert failed: {e}")

    def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent alerts."""
        return self.alert_history[-limit:]
