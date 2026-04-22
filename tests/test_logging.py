"""Unit tests for secret masking and log rotation."""

import logging
import logging.handlers
import unittest

from logger import SecretMaskingFilter, RenaissanceAuditLogger


class TestSecretMaskingFilter(unittest.TestCase):
    """Test that secrets are redacted from log records."""

    def setUp(self):
        self.f = SecretMaskingFilter()

    def _make_record(self, msg):
        return logging.LogRecord(
            "test", logging.INFO, "", 0, msg, (), None
        )

    def test_masks_api_key(self):
        record = self._make_record('api_key=sk_live_abc123def456ghi789')
        self.f.filter(record)
        self.assertNotIn("abc123", record.msg)
        self.assertIn("REDACTED", record.msg)

    def test_masks_api_secret(self):
        record = self._make_record('api_secret: "SuperSecretValue12345678"')
        self.f.filter(record)
        self.assertNotIn("SuperSecretValue", record.msg)
        self.assertIn("REDACTED", record.msg)

    def test_masks_bearer_token(self):
        record = self._make_record('Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature')
        self.f.filter(record)
        self.assertNotIn("eyJhbGci", record.msg)
        self.assertIn("REDACTED", record.msg)

    def test_masks_cb_access_key(self):
        record = self._make_record('CB-ACCESS-KEY: abcdef123456789')
        self.f.filter(record)
        self.assertNotIn("abcdef123456789", record.msg)
        self.assertIn("REDACTED", record.msg)

    def test_masks_cb_access_sign(self):
        record = self._make_record('CB-ACCESS-SIGN: somesignaturevalue123')
        self.f.filter(record)
        self.assertNotIn("somesignaturevalue123", record.msg)
        self.assertIn("REDACTED", record.msg)

    def test_masks_password(self):
        record = self._make_record('password="MyPassword12345678"')
        self.f.filter(record)
        self.assertNotIn("MyPassword", record.msg)
        self.assertIn("REDACTED", record.msg)

    def test_preserves_normal_messages(self):
        record = self._make_record("Trading cycle 42 completed successfully")
        self.f.filter(record)
        self.assertEqual(record.msg, "Trading cycle 42 completed successfully")

    def test_masks_in_args(self):
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "Config: %s", ("api_key=DEADBEEF12345678",), None
        )
        self.f.filter(record)
        self.assertNotIn("DEADBEEF", str(record.args[0]))

    def test_filter_always_returns_true(self):
        """Filter should never suppress records, only mask them."""
        record = self._make_record("api_key=secret123456789")
        result = self.f.filter(record)
        self.assertTrue(result)


class TestLogRotation(unittest.TestCase):
    """Verify that RotatingFileHandler is used instead of plain FileHandler."""

    def test_audit_logger_uses_rotating_handler(self):
        logger = RenaissanceAuditLogger()
        handlers = logger.logger.handlers
        rotating = [h for h in handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        self.assertGreater(len(rotating), 0, "Expected at least one RotatingFileHandler")

    def test_rotating_handler_config(self):
        logger = RenaissanceAuditLogger()
        for h in logger.logger.handlers:
            if isinstance(h, logging.handlers.RotatingFileHandler):
                self.assertEqual(h.maxBytes, 50 * 1024 * 1024)
                self.assertEqual(h.backupCount, 5)
                break
        else:
            self.fail("No RotatingFileHandler found")

    def test_secret_masking_applied_to_handlers(self):
        """All handlers should have the SecretMaskingFilter attached."""
        logger = RenaissanceAuditLogger()
        for handler in logger.logger.handlers:
            filter_types = [type(f) for f in handler.filters]
            self.assertIn(SecretMaskingFilter, filter_types,
                          f"Handler {handler} missing SecretMaskingFilter")


if __name__ == "__main__":
    unittest.main()
