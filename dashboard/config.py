"""Dashboard configuration loader and feature-flag reader."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_DASHBOARD_CONFIG: Dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 8080,
    "refresh_interval_ms": 2000,
    "dark_mode": True,
    "alerts": {
        "pnl_threshold": -200,
        "drawdown_threshold": 0.05,
        "consecutive_loss_threshold": 5,
    },
    "display": {
        "equity_curve_points": 500,
        "activity_feed_limit": 100,
        "decision_table_limit": 200,
    },
}


class DashboardConfig:
    """Reads the bot's config.json, extracts feature flags + dashboard settings."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is None:
            config_path = str(
                Path(__file__).resolve().parent.parent / "config" / "config.json"
            )
        self._path = Path(config_path)
        self._raw: Dict[str, Any] = {}
        self.dashboard: Dict[str, Any] = dict(DEFAULT_DASHBOARD_CONFIG)
        self.flags: Dict[str, bool] = {}
        self.reload()

    def reload(self) -> None:
        try:
            with self._path.open("r", encoding="utf-8") as f:
                self._raw = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {self._path}: {e}")
            self._raw = {}

        # Merge dashboard_config section
        dc = self._raw.get("dashboard_config", {})
        self.dashboard = {**DEFAULT_DASHBOARD_CONFIG, **dc}

        # Build feature flags from top-level config
        self.flags = {
            "regime_overlay": self._raw.get("regime_overlay", {}).get("enabled", False),
            "risk_gateway": self._raw.get("risk_gateway", {}).get("enabled", False),
            "real_time_pipeline": self._raw.get("real_time_pipeline", {}).get("enabled", False),
            "ghost_runner": self._raw.get("ghost_runner", {}).get("enabled", False),
            "market_making": self._raw.get("market_making", {}).get("enabled", False),
            "basis_trading": bool(self._raw.get("basis_trading", {}).get("enabled", False)),
            "alternative_data": bool(self._raw.get("alternative_data", {}).get("enabled", False)),
            "breakout_scanner": self._raw.get("breakout_scanner", {}).get("enabled", False),
            "correlation_network": self._raw.get("correlation_network", {}).get("enabled", False),
            "garch_volatility": self._raw.get("garch_volatility", {}).get("enabled", False),
            "mean_reversion": bool(self._raw.get("mean_reversion", {})),
            "ml_integration": self._raw.get("ml_integration", {}).get("enabled", True),
            "historical_data_cache": self._raw.get("historical_data_cache", {}).get("enabled", False),
            "alerting": self._raw.get("alerting", {}).get("enabled", False),
            "database": self._raw.get("database", {}).get("enabled", True),
        }

    @property
    def db_path(self) -> str:
        return self._raw.get("database", {}).get("path", "data/renaissance_bot.db")

    @property
    def signal_weights(self) -> Dict[str, float]:
        return self._raw.get("signal_weights", {})

    @property
    def product_ids(self) -> list:
        return self._raw.get("trading", {}).get("product_ids", ["BTC-USD"])

    @property
    def paper_trading(self) -> bool:
        return self._raw.get("trading", {}).get("paper_trading", True)

    @property
    def raw(self) -> Dict[str, Any]:
        return self._raw
