"""Unit tests for config validation at startup (Step 9)."""

import unittest
from unittest.mock import MagicMock, patch
import copy


# Minimal valid config for the bot
VALID_CONFIG = {
    "risk_management": {
        "daily_loss_limit": 500,
        "position_limit": 50000,
    },
    "trading": {
        "min_confidence": 0.6,
        "cycle_interval_seconds": 60,
        "product_ids": ["BTC-USD"],
    },
    "signal_weights": {
        "technical": 0.4,
        "sentiment": 0.3,
        "ml": 0.3,
    },
}


class TestConfigValidation(unittest.TestCase):
    """Test _validate_config() logic."""

    def _get_bot_class(self):
        """Import and return the bot class (lazy to avoid heavy imports)."""
        from renaissance_trading_bot import RenaissanceTradingBot
        return RenaissanceTradingBot

    def _make_validator(self):
        """Create a bare validator function by extracting from the class."""
        BotClass = self._get_bot_class()
        # Create a mock instance to call the method on
        instance = MagicMock()
        instance.logger = MagicMock()
        return lambda cfg: BotClass._validate_config(instance, cfg)

    def test_valid_config_passes(self):
        """A well-formed config should not raise."""
        validate = self._make_validator()
        validate(copy.deepcopy(VALID_CONFIG))  # Should not raise

    def test_invalid_daily_loss_limit(self):
        """daily_loss_limit outside (0, 100000] should raise."""
        validate = self._make_validator()
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["risk_management"]["daily_loss_limit"] = -100
        with self.assertRaises(ValueError) as ctx:
            validate(cfg)
        self.assertIn("daily_loss_limit", str(ctx.exception))

    def test_invalid_position_limit(self):
        """position_limit outside (0, 1000000] should raise."""
        validate = self._make_validator()
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["risk_management"]["position_limit"] = 0
        with self.assertRaises(ValueError) as ctx:
            validate(cfg)
        self.assertIn("position_limit", str(ctx.exception))

    def test_invalid_min_confidence(self):
        """min_confidence outside (0, 1.0] should raise."""
        validate = self._make_validator()
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["risk_management"]["min_confidence"] = 1.5
        with self.assertRaises(ValueError) as ctx:
            validate(cfg)
        self.assertIn("min_confidence", str(ctx.exception))

    def test_invalid_cycle_interval(self):
        """cycle_interval_seconds outside [10, 3600] should raise."""
        validate = self._make_validator()
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["trading"]["cycle_interval_seconds"] = 5
        with self.assertRaises(ValueError) as ctx:
            validate(cfg)
        self.assertIn("cycle_interval", str(ctx.exception))

    def test_signal_weights_normalized(self):
        """Weights that don't sum to 1.0 should be auto-normalized."""
        validate = self._make_validator()
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["signal_weights"] = {"technical": 2.0, "sentiment": 2.0, "ml": 1.0}
        validate(cfg)  # Should normalize, not raise
        total = sum(cfg["signal_weights"].values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_signal_weights_already_correct(self):
        """Weights summing to 1.0 should not be modified."""
        validate = self._make_validator()
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["signal_weights"] = {"a": 0.5, "b": 0.5}
        validate(cfg)
        self.assertAlmostEqual(cfg["signal_weights"]["a"], 0.5)


if __name__ == "__main__":
    unittest.main()
