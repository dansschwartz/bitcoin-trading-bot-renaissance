"""
Tests for orchestrator/bot_manager.py — BotOrchestrator.

Covers heartbeat monitoring, aggregate risk computation, bot status tracking,
pair assignment, and aggregate reporting.
"""

import json
import os
import tempfile
import time

import pytest

from orchestrator.bot_manager import BotInstance, BotOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_heartbeat(hb_dir: str, bot_id: str, **overrides) -> str:
    """Write a heartbeat JSON file and return its path."""
    payload = {
        "bot_id": bot_id,
        "timestamp": time.time(),
        "status": "running",
        "equity_usd": 10_000.0,
        "daily_pnl_usd": 50.0,
        "net_exposure_usd": 3_000.0,
        "positions": {},
        "open_orders": 0,
        "regime": "unknown",
    }
    payload.update(overrides)
    path = os.path.join(hb_dir, f"{bot_id}.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


# ---------------------------------------------------------------------------
# Tests — BotInstance dataclass
# ---------------------------------------------------------------------------

class TestBotInstance:
    def test_default_fields(self):
        bot = BotInstance(bot_id="test-1")
        assert bot.bot_id == "test-1"
        assert bot.status == "unknown"
        assert bot.capital_usd == 0.0
        assert bot.positions == {}
        assert bot.assigned_pairs == []

    def test_custom_fields(self):
        bot = BotInstance(
            bot_id="bot-2",
            status="running",
            current_equity_usd=50_000.0,
            daily_pnl_usd=200.0,
        )
        assert bot.current_equity_usd == 50_000.0
        assert bot.daily_pnl_usd == 200.0


# ---------------------------------------------------------------------------
# Tests — BotOrchestrator
# ---------------------------------------------------------------------------

class TestBotOrchestratorInit:
    def test_default_config(self):
        orch = BotOrchestrator()
        assert orch._heartbeat_dir == "data/heartbeats"
        assert orch._poll_interval == 5.0
        assert orch._max_single_asset_pct == 30.0
        assert orch._max_total_exposure_pct == 80.0
        assert orch._max_drawdown_pct == 10.0
        assert orch.bots == {}

    def test_custom_config(self):
        cfg = {
            "heartbeat_dir": "/tmp/hb",
            "poll_interval_seconds": 10,
            "aggregate_limits": {
                "max_single_asset_pct": 20,
                "max_total_exposure_pct": 60,
                "max_drawdown_pct": 5,
            },
        }
        orch = BotOrchestrator(config=cfg)
        assert orch._heartbeat_dir == "/tmp/hb"
        assert orch._poll_interval == 10.0
        assert orch._max_single_asset_pct == 20.0


class TestReadHeartbeats:
    def test_reads_heartbeat_files(self, tmp_path):
        hb_dir = str(tmp_path / "heartbeats")
        os.makedirs(hb_dir)
        _write_heartbeat(hb_dir, "bot-01", equity_usd=15_000.0, daily_pnl_usd=100.0)
        _write_heartbeat(hb_dir, "bot-02", equity_usd=25_000.0, daily_pnl_usd=-50.0)

        orch = BotOrchestrator(config={"heartbeat_dir": hb_dir})
        orch._read_heartbeats()

        assert len(orch.bots) == 2
        assert orch.bots["bot-01"].current_equity_usd == 15_000.0
        assert orch.bots["bot-02"].daily_pnl_usd == -50.0

    def test_missing_directory(self, tmp_path):
        orch = BotOrchestrator(config={"heartbeat_dir": str(tmp_path / "nonexistent")})
        orch._read_heartbeats()  # Should not raise
        assert len(orch.bots) == 0

    def test_invalid_json_skipped(self, tmp_path):
        hb_dir = str(tmp_path / "heartbeats")
        os.makedirs(hb_dir)
        # Write valid heartbeat
        _write_heartbeat(hb_dir, "good-bot")
        # Write invalid JSON
        bad_path = os.path.join(hb_dir, "bad-bot.json")
        with open(bad_path, "w") as f:
            f.write("{invalid json!!!")

        orch = BotOrchestrator(config={"heartbeat_dir": hb_dir})
        orch._read_heartbeats()  # Should not raise
        assert len(orch.bots) == 1
        assert "good-bot" in orch.bots

    def test_bot_id_from_filename_when_missing(self, tmp_path):
        hb_dir = str(tmp_path / "heartbeats")
        os.makedirs(hb_dir)
        # Write heartbeat without bot_id field
        path = os.path.join(hb_dir, "my-bot.json")
        with open(path, "w") as f:
            json.dump({"timestamp": time.time(), "equity_usd": 1000.0}, f)

        orch = BotOrchestrator(config={"heartbeat_dir": hb_dir})
        orch._read_heartbeats()
        # Should use filename stem as bot_id
        assert "my-bot" in orch.bots


class TestMonitorHealth:
    def test_running_bot_stays_running(self):
        orch = BotOrchestrator()
        orch.bots["b1"] = BotInstance(
            bot_id="b1", status="running", last_heartbeat=time.time()
        )
        orch.monitor_health()
        assert orch.bots["b1"].status == "running"

    def test_stale_heartbeat_becomes_unresponsive(self):
        orch = BotOrchestrator()
        orch.bots["b1"] = BotInstance(
            bot_id="b1", status="running", last_heartbeat=time.time() - 60
        )
        orch.monitor_health()
        assert orch.bots["b1"].status == "unresponsive"

    def test_very_stale_heartbeat_becomes_dead(self):
        orch = BotOrchestrator()
        orch.bots["b1"] = BotInstance(
            bot_id="b1", status="running", last_heartbeat=time.time() - 200
        )
        orch.monitor_health()
        assert orch.bots["b1"].status == "dead"

    def test_unknown_status_upgraded_to_running_when_fresh(self):
        orch = BotOrchestrator()
        orch.bots["b1"] = BotInstance(
            bot_id="b1", status="unknown", last_heartbeat=time.time()
        )
        orch.monitor_health()
        assert orch.bots["b1"].status == "running"


class TestCheckAggregateExposure:
    def test_no_bots_returns_zeros(self):
        orch = BotOrchestrator()
        result = orch.check_aggregate_exposure()
        assert result["total_exposure_usd"] == 0.0
        assert result["total_equity_usd"] == 0.0
        assert result["breaches"] == []

    def test_within_limits_no_breaches(self):
        orch = BotOrchestrator()
        orch.bots["b1"] = BotInstance(
            bot_id="b1",
            status="running",
            current_equity_usd=100_000.0,
            net_exposure=10_000.0,
            positions={"BTC-USD": {"net_exposure_usd": 10_000}},
        )
        result = orch.check_aggregate_exposure()
        assert result["breaches"] == []
        assert result["total_exposure_usd"] == 10_000.0
        assert result["total_equity_usd"] == 100_000.0

    def test_single_asset_breach(self):
        orch = BotOrchestrator(config={
            "aggregate_limits": {"max_single_asset_pct": 20}
        })
        orch.bots["b1"] = BotInstance(
            bot_id="b1",
            status="running",
            current_equity_usd=10_000.0,
            net_exposure=5_000.0,
            positions={"BTC-USD": {"net_exposure_usd": 5_000}},
        )
        result = orch.check_aggregate_exposure()
        # 5000/10000 = 50% > 20% limit
        assert len(result["breaches"]) >= 1
        assert "BTC-USD" in result["breaches"][0]

    def test_total_exposure_breach(self):
        orch = BotOrchestrator(config={
            "aggregate_limits": {"max_total_exposure_pct": 10}
        })
        orch.bots["b1"] = BotInstance(
            bot_id="b1",
            status="running",
            current_equity_usd=10_000.0,
            net_exposure=5_000.0,
            positions={},
        )
        result = orch.check_aggregate_exposure()
        # 5000/10000 = 50% > 10% limit
        assert any("Total exposure" in b for b in result["breaches"])

    def test_dead_bots_excluded(self):
        orch = BotOrchestrator()
        orch.bots["dead"] = BotInstance(
            bot_id="dead",
            status="dead",
            current_equity_usd=100_000.0,
            net_exposure=99_000.0,
        )
        result = orch.check_aggregate_exposure()
        assert result["total_equity_usd"] == 0.0
        assert result["total_exposure_usd"] == 0.0


class TestAssignPairs:
    def test_round_robin_assignment(self):
        orch = BotOrchestrator()
        orch.bots["b1"] = BotInstance(bot_id="b1", status="running")
        orch.bots["b2"] = BotInstance(bot_id="b2", status="running")

        pairs = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"]
        assignment = orch.assign_pairs(pairs)

        assert set(assignment.keys()) == {"b1", "b2"}
        # 4 pairs across 2 bots = 2 each
        assert len(assignment["b1"]) == 2
        assert len(assignment["b2"]) == 2

    def test_dead_bots_excluded_from_assignment(self):
        orch = BotOrchestrator()
        orch.bots["live"] = BotInstance(bot_id="live", status="running")
        orch.bots["dead"] = BotInstance(bot_id="dead", status="dead")

        assignment = orch.assign_pairs(["BTC-USD", "ETH-USD"])
        assert "dead" not in assignment
        assert len(assignment["live"]) == 2

    def test_no_active_bots_returns_empty(self):
        orch = BotOrchestrator()
        orch.bots["dead"] = BotInstance(bot_id="dead", status="dead")
        assert orch.assign_pairs(["BTC-USD"]) == {}


class TestGetAggregateReport:
    def test_report_structure(self):
        orch = BotOrchestrator()
        orch.bots["b1"] = BotInstance(
            bot_id="b1",
            status="running",
            current_equity_usd=10_000.0,
            daily_pnl_usd=100.0,
            net_exposure=2_000.0,
            positions={"BTC": {"net_exposure_usd": 2_000}},
        )

        report = orch.get_aggregate_report()

        assert "timestamp" in report
        assert report["total_equity_usd"] == 10_000.0
        assert report["total_daily_pnl_usd"] == 100.0
        assert report["active_bots"] == 1
        assert report["total_bots"] == 1
        assert len(report["bots"]) == 1

    def test_report_excludes_dead_from_totals(self):
        orch = BotOrchestrator()
        orch.bots["live"] = BotInstance(
            bot_id="live", status="running",
            current_equity_usd=10_000.0, daily_pnl_usd=100.0,
        )
        orch.bots["dead"] = BotInstance(
            bot_id="dead", status="dead",
            current_equity_usd=50_000.0, daily_pnl_usd=500.0,
        )

        report = orch.get_aggregate_report()
        assert report["total_equity_usd"] == 10_000.0
        assert report["total_daily_pnl_usd"] == 100.0
        assert report["active_bots"] == 1
        assert report["total_bots"] == 2


class TestStopMethod:
    def test_stop_sets_running_false(self):
        orch = BotOrchestrator()
        orch._running = True
        orch.stop()
        assert orch._running is False
