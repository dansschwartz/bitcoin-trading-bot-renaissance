"""
Tests for dashboard API routers.

Tests router handler functions directly (bypassing TestClient) to avoid
Python 3.14 urllib3 import deadlock issues with httpx.
"""

import asyncio
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_db() -> str:
    """Create a temp SQLite DB with full schema via DatabaseManager."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from database_manager import DatabaseManager
    import asyncio
    db = DatabaseManager({"path": db_path})
    asyncio.run(db.init_database())
    # Also create five_minute_bars if missing (used by brain router)
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS five_minute_bars (
        id INTEGER PRIMARY KEY, pair TEXT, close REAL)""")
    conn.commit()
    conn.close()
    return db_path


def _make_mock_request(db_path: str) -> MagicMock:
    """Create a mock FastAPI Request with app.state."""
    request = MagicMock()
    cfg = MagicMock()
    cfg.db_path = db_path
    cfg.paper_trading = True
    cfg.product_ids = ["BTC-USD", "ETH-USD"]
    cfg.signal_weights = {"order_flow": 0.5, "ml_ensemble": 0.5}
    cfg.flags = {"regime_overlay": True}
    cfg.dashboard = {"alerts": {"pnl_threshold": -200, "drawdown_threshold": 0.05}}

    emitter = MagicMock()
    emitter.get_cached.return_value = None

    ws_manager = MagicMock()
    ws_manager.active_count = 0

    request.app.state.dashboard_config = cfg
    request.app.state.start_time = datetime.now(timezone.utc)
    request.app.state.emitter = emitter
    request.app.state.ws_manager = ws_manager
    request.app.state.active_alerts = []

    return request


@pytest.fixture
def test_db():
    db_path = _create_test_db()
    yield db_path
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def mock_request(test_db):
    return _make_mock_request(test_db)


# ---------------------------------------------------------------------------
# Tests: System Router
# ---------------------------------------------------------------------------

class TestSystemRouter:
    def test_system_status(self, mock_request):
        from dashboard.routers.system import system_status
        result = asyncio.run(system_status(mock_request))

        assert result["status"] == "OPERATIONAL"
        assert "uptime_seconds" in result
        assert "cycle_count" in result
        assert "trade_count" in result
        assert result["paper_trading"] is True
        assert "timestamp" in result

    def test_system_config(self, mock_request):
        from dashboard.routers.system import system_config
        result = asyncio.run(system_config(mock_request))

        assert "flags" in result
        assert "signal_weights" in result
        assert "product_ids" in result
        assert result["paper_trading"] is True

    def test_success_criteria(self, mock_request):
        from dashboard.routers.system import success_criteria
        result = asyncio.run(success_criteria(mock_request))

        assert isinstance(result, dict)

    def test_activity_feed(self, mock_request):
        from dashboard.routers.system import activity_feed
        result = asyncio.run(activity_feed(mock_request))
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: Brain Router
# ---------------------------------------------------------------------------

class TestBrainRouter:
    def test_regime_status_empty_db(self, mock_request):
        from dashboard.routers.brain import regime_status
        result = asyncio.run(regime_status(mock_request))

        assert "current" in result
        assert "history" in result
        assert isinstance(result["history"], list)

    def test_ensemble_status_empty_db(self, mock_request):
        from dashboard.routers.brain import ensemble_status
        result = asyncio.run(ensemble_status(mock_request))

        assert "models" in result
        assert "model_count" in result
        assert result["model_count"] == 0

    def test_regime_with_data(self, test_db, mock_request):
        """Insert a decision with regime data and verify it shows up."""
        conn = sqlite3.connect(test_db)
        conn.execute(
            "INSERT INTO decisions (timestamp, product_id, action, confidence, "
            "position_size, weighted_signal, reasoning, hmm_regime) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), "BTC-USD", "HOLD",
             0.5, 0.0, 0.0, '{}', "trending"),
        )
        conn.commit()
        conn.close()

        from dashboard.routers.brain import regime_status
        result = asyncio.run(regime_status(mock_request))

        # Should have history entries now
        assert len(result["history"]) >= 1


# ---------------------------------------------------------------------------
# Tests: Risk Router
# ---------------------------------------------------------------------------

class TestRiskRouter:
    def test_risk_metrics_empty_db(self, mock_request):
        from dashboard.routers.risk import risk_metrics
        result = asyncio.run(risk_metrics(mock_request))
        assert isinstance(result, dict)

    def test_risk_exposure_empty_db(self, mock_request):
        from dashboard.routers.risk import exposure
        result = asyncio.run(exposure(mock_request))
        assert isinstance(result, dict)

    def test_risk_alerts_empty_db(self, mock_request):
        from dashboard.routers.risk import active_alerts
        result = asyncio.run(active_alerts(mock_request))
        assert "alerts" in result
        assert isinstance(result["alerts"], list)

    def test_gateway_log_empty_db(self, mock_request):
        from dashboard.routers.risk import gateway_log
        result = asyncio.run(gateway_log(mock_request))
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: Positions Router
# ---------------------------------------------------------------------------

class TestPositionsRouter:
    def test_open_positions_empty(self, mock_request):
        from dashboard.routers.trades import open_positions
        result = asyncio.run(open_positions(mock_request))
        assert isinstance(result, list)
        assert len(result) == 0

    def test_open_positions_with_data(self, test_db, mock_request):
        conn = sqlite3.connect(test_db)
        conn.execute(
            "INSERT INTO open_positions (position_id, product_id, side, size, "
            "entry_price, opened_at, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("pos_1", "BTC-USD", "BUY", 0.001, 50000.0,
             datetime.now(timezone.utc).isoformat(), "OPEN"),
        )
        conn.commit()
        conn.close()

        from dashboard.routers.trades import open_positions
        result = asyncio.run(open_positions(mock_request))

        assert len(result) >= 1
        pos = result[0]
        assert pos["product_id"] == "BTC-USD"
        assert "unrealized_pnl" in pos
        assert "current_price" in pos

    def test_closed_positions_empty(self, mock_request):
        from dashboard.routers.trades import closed_positions
        result = asyncio.run(closed_positions(mock_request))
        assert isinstance(result, list)

    def test_position_summary_empty(self, mock_request):
        from dashboard.routers.trades import position_summary
        result = asyncio.run(position_summary(mock_request))
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Tests: db_queries module
# ---------------------------------------------------------------------------

class TestDbQueries:
    def test_get_cycle_count_empty(self, test_db):
        from dashboard.db_queries import get_cycle_count
        assert get_cycle_count(test_db) == 0

    def test_get_cycle_count_with_data(self, test_db):
        conn = sqlite3.connect(test_db)
        for i in range(5):
            conn.execute(
                "INSERT INTO decisions (timestamp, product_id, action, confidence, "
                "position_size, weighted_signal, reasoning) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), "BTC-USD", "HOLD",
                 0.5, 0.0, 0.0, '{}'),
            )
        conn.commit()
        conn.close()

        from dashboard.db_queries import get_cycle_count
        assert get_cycle_count(test_db) == 5

    def test_get_trade_count_empty(self, test_db):
        from dashboard.db_queries import get_trade_count
        assert get_trade_count(test_db) == 0

    def test_get_open_positions_empty(self, test_db):
        from dashboard.db_queries import get_open_positions
        result = get_open_positions(test_db)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_open_positions_with_data(self, test_db):
        conn = sqlite3.connect(test_db)
        conn.execute(
            "INSERT INTO open_positions (position_id, product_id, side, size, "
            "entry_price, opened_at, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("pos_1", "BTC-USD", "BUY", 0.001, 50000.0,
             datetime.now(timezone.utc).isoformat(), "OPEN"),
        )
        conn.commit()
        conn.close()

        from dashboard.db_queries import get_open_positions
        result = get_open_positions(test_db)
        assert len(result) == 1
        assert result[0]["product_id"] == "BTC-USD"

    def test_get_regime_history_empty(self, test_db):
        from dashboard.db_queries import get_regime_history
        result = get_regime_history(test_db)
        assert isinstance(result, list)

    def test_get_active_product_ids_empty(self, test_db):
        from dashboard.db_queries import get_active_product_ids
        result = get_active_product_ids(test_db)
        assert isinstance(result, list)
