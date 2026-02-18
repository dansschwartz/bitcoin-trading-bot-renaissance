"""ObservationCollector — compiles weekly observation reports from DB + agent data.

``compile_weekly_report()`` queries existing tables and returns a JSON-serializable
dict covering portfolio, signals, regimes, ML models, execution, risk events,
data quality, and config snapshot.

Can be run standalone:
    python -m agents.observation_collector
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.coordinator import AgentCoordinator

logger = logging.getLogger(__name__)


class ObservationCollector:
    """Compiles a comprehensive weekly observation report."""

    def __init__(
        self,
        db_path: str,
        config: Dict[str, Any],
        coordinator: Optional[AgentCoordinator] = None,
    ) -> None:
        self.db_path = db_path
        self.config = config
        self.coordinator = coordinator

    def compile_weekly_report(self, window_hours: int = 168) -> Dict[str, Any]:
        """Build the full weekly observation report.

        Returns a JSON-serializable dict with sections for every dimension
        the quant researcher needs.
        """
        now = datetime.now(timezone.utc)
        week_start = (now - timedelta(hours=window_hours)).isoformat()
        week_end = now.isoformat()

        report: Dict[str, Any] = {
            "meta": {
                "generated_at": now.isoformat(),
                "week_start": week_start,
                "week_end": week_end,
                "window_hours": window_hours,
            },
        }

        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro", uri=True, timeout=10.0,
        )
        conn.row_factory = sqlite3.Row

        report["portfolio"] = self._portfolio_section(conn, window_hours)
        report["signals"] = self._signals_section(conn, window_hours)
        report["regimes"] = self._regimes_section(conn, window_hours)
        report["ml_models"] = self._ml_section(conn, window_hours)
        report["execution"] = self._execution_section(conn, window_hours)
        report["risk_events"] = self._risk_events_section(conn, window_hours)
        report["data_quality"] = self._data_quality_section(conn, window_hours)
        report["config_snapshot"] = self._config_snapshot()

        # Collect agent observations if coordinator is available
        if self.coordinator:
            report["agent_observations"] = self.coordinator.get_all_observations(window_hours)

        conn.close()

        # Compute summary metrics
        report["summary"] = self._compute_summary(report)

        return report

    def save_report(self, report: Dict[str, Any]) -> str:
        """Save report to weekly_reports table and to disk as JSON.

        Returns the file path of the saved JSON.
        """
        # Save to DB
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            summary = report.get("summary", {})
            conn.execute(
                """INSERT INTO weekly_reports
                   (week_start, week_end, report_json, sharpe_7d, total_pnl, total_trades, win_rate)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    report["meta"]["week_start"],
                    report["meta"]["week_end"],
                    json.dumps(report),
                    summary.get("sharpe_7d"),
                    summary.get("total_pnl"),
                    summary.get("total_trades"),
                    summary.get("win_rate"),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("Failed to save report to DB: %s", exc)

        # Save to disk
        reports_dir = Path(self.db_path).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = reports_dir / f"weekly_report_{date_str}.json"
        try:
            filepath.write_text(json.dumps(report, indent=2, default=str))
            logger.info("Weekly report saved to %s", filepath)
        except Exception as exc:
            logger.warning("Failed to write report file: %s", exc)

        return str(filepath)

    # ── Section builders ──

    def _portfolio_section(self, conn: sqlite3.Connection, hours: int) -> Dict[str, Any]:
        section: Dict[str, Any] = {}
        try:
            # Trades summary
            row = conn.execute(
                """SELECT COUNT(*) as cnt,
                          SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                          SUM(pnl) as total_pnl,
                          AVG(pnl) as avg_pnl,
                          MAX(pnl) as best_trade,
                          MIN(pnl) as worst_trade
                   FROM trades
                   WHERE close_time >= datetime('now', ? || ' hours')""",
                (f"-{hours}",),
            ).fetchone()
            if row:
                section["total_trades"] = row["cnt"] or 0
                section["wins"] = row["wins"] or 0
                section["total_pnl"] = round(row["total_pnl"] or 0.0, 2)
                section["avg_pnl"] = round(row["avg_pnl"] or 0.0, 4)
                section["best_trade"] = round(row["best_trade"] or 0.0, 4)
                section["worst_trade"] = round(row["worst_trade"] or 0.0, 4)
                trades = section["total_trades"]
                section["win_rate"] = round(section["wins"] / trades, 4) if trades > 0 else 0.0

            # Open positions
            row = conn.execute(
                "SELECT COUNT(*) FROM open_positions WHERE status='open'"
            ).fetchone()
            section["open_positions"] = row[0] if row else 0

            # Daily performance
            rows = conn.execute(
                """SELECT date, total_pnl, sharpe_ratio
                   FROM daily_performance
                   WHERE date >= date('now', ? || ' days')
                   ORDER BY date""",
                (f"-{hours // 24}",),
            ).fetchall()
            section["daily_pnl"] = [
                {"date": r["date"], "pnl": r["total_pnl"], "sharpe": r["sharpe_ratio"]}
                for r in rows
            ]
        except Exception as exc:
            section["error"] = str(exc)
        return section

    def _signals_section(self, conn: sqlite3.Connection, hours: int) -> Dict[str, Any]:
        section: Dict[str, Any] = {}
        try:
            rows = conn.execute(
                """SELECT signal_type, SUM(daily_pnl) as pnl, COUNT(*) as days
                   FROM signal_daily_pnl
                   WHERE date >= date('now', ? || ' days')
                   GROUP BY signal_type
                   ORDER BY pnl DESC""",
                (f"-{hours // 24}",),
            ).fetchall()
            section["per_signal_pnl"] = [
                {"signal": r["signal_type"], "pnl": round(r["pnl"], 2), "days": r["days"]}
                for r in rows
            ]
            # Decision confidence distribution
            rows = conn.execute(
                """SELECT
                    CASE
                        WHEN confidence < 0.3 THEN 'low'
                        WHEN confidence < 0.6 THEN 'medium'
                        ELSE 'high'
                    END as bucket,
                    COUNT(*) as cnt
                   FROM decisions
                   WHERE timestamp >= datetime('now', ? || ' hours')
                   GROUP BY bucket""",
                (f"-{hours}",),
            ).fetchall()
            section["confidence_distribution"] = {r["bucket"]: r["cnt"] for r in rows}
        except Exception as exc:
            section["error"] = str(exc)
        return section

    def _regimes_section(self, conn: sqlite3.Connection, hours: int) -> Dict[str, Any]:
        section: Dict[str, Any] = {}
        try:
            rows = conn.execute(
                """SELECT hmm_regime, COUNT(*) as cnt,
                          AVG(confidence) as avg_conf
                   FROM decisions
                   WHERE timestamp >= datetime('now', ? || ' hours')
                   GROUP BY hmm_regime
                   ORDER BY cnt DESC""",
                (f"-{hours}",),
            ).fetchall()
            section["regime_distribution"] = [
                {"regime": r["hmm_regime"], "decisions": r["cnt"],
                 "avg_confidence": round(r["avg_conf"], 4)}
                for r in rows
            ]
        except Exception as exc:
            section["error"] = str(exc)
        return section

    def _ml_section(self, conn: sqlite3.Connection, hours: int) -> Dict[str, Any]:
        section: Dict[str, Any] = {}
        try:
            rows = conn.execute(
                """SELECT model_name, COUNT(*) as cnt, AVG(confidence) as avg_conf
                   FROM ml_predictions
                   WHERE timestamp >= datetime('now', ? || ' hours')
                   GROUP BY model_name""",
                (f"-{hours}",),
            ).fetchall()
            section["predictions_per_model"] = [
                {"model": r["model_name"], "count": r["cnt"],
                 "avg_confidence": round(r["avg_conf"], 4) if r["avg_conf"] else None}
                for r in rows
            ]
        except sqlite3.OperationalError:
            section["predictions_per_model"] = []
        except Exception as exc:
            section["error"] = str(exc)
        # Model ledger
        try:
            rows = conn.execute(
                "SELECT * FROM model_ledger ORDER BY timestamp DESC LIMIT 10"
            ).fetchall()
            section["model_ledger"] = [dict(r) for r in rows]
        except sqlite3.OperationalError:
            section["model_ledger"] = []
        return section

    def _execution_section(self, conn: sqlite3.Connection, hours: int) -> Dict[str, Any]:
        section: Dict[str, Any] = {}
        try:
            rows = conn.execute(
                """SELECT signal_type,
                          COUNT(*) as fills,
                          AVG(slippage_bps) as avg_slip,
                          AVG(latency_signal_to_fill_ms) as avg_latency,
                          AVG(devil) as avg_devil
                   FROM devil_tracker
                   WHERE signal_timestamp >= datetime('now', ? || ' hours')
                   GROUP BY signal_type""",
                (f"-{hours}",),
            ).fetchall()
            section["per_signal_execution"] = [
                {
                    "signal_type": r["signal_type"],
                    "fills": r["fills"],
                    "avg_slippage_bps": round(r["avg_slip"], 2) if r["avg_slip"] else 0.0,
                    "avg_latency_ms": round(r["avg_latency"], 1) if r["avg_latency"] else 0.0,
                    "avg_devil": round(r["avg_devil"], 4) if r["avg_devil"] else 0.0,
                }
                for r in rows
            ]
        except Exception as exc:
            section["error"] = str(exc)
        return section

    def _risk_events_section(self, conn: sqlite3.Connection, hours: int) -> Dict[str, Any]:
        section: Dict[str, Any] = {}
        try:
            rows = conn.execute(
                """SELECT event_type, severity, COUNT(*) as cnt
                   FROM agent_events
                   WHERE timestamp >= datetime('now', ? || ' hours')
                   GROUP BY event_type, severity
                   ORDER BY cnt DESC""",
                (f"-{hours}",),
            ).fetchall()
            section["events_by_type"] = [
                {"type": r["event_type"], "severity": r["severity"], "count": r["cnt"]}
                for r in rows
            ]
        except sqlite3.OperationalError:
            section["events_by_type"] = []
        except Exception as exc:
            section["error"] = str(exc)
        return section

    def _data_quality_section(self, conn: sqlite3.Connection, hours: int) -> Dict[str, Any]:
        section: Dict[str, Any] = {}
        try:
            rows = conn.execute(
                """SELECT pair, COUNT(*) as bars,
                          MIN(bar_start) as first_bar,
                          MAX(bar_end) as last_bar
                   FROM five_minute_bars
                   GROUP BY pair
                   ORDER BY bars DESC"""
            ).fetchall()
            section["bar_completeness"] = [
                {"pair": r["pair"], "total_bars": r["bars"],
                 "first": r["first_bar"], "last": r["last_bar"]}
                for r in rows
            ]
            # Expected bars in window
            expected = hours * 12  # 12 five-minute bars per hour
            for entry in section["bar_completeness"]:
                entry["completeness_pct"] = round(
                    min(entry["total_bars"] / expected * 100, 100.0), 1,
                )
        except Exception as exc:
            section["error"] = str(exc)
        return section

    def _config_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        snapshot["signal_weights"] = self.config.get("signal_weights", {})
        snapshot["risk_management"] = self.config.get("risk_management", {})
        snapshot["regime_overlay"] = self.config.get("regime_overlay", {})
        snapshot["risk_gateway"] = self.config.get("risk_gateway", {})
        return snapshot

    def _compute_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        portfolio = report.get("portfolio", {})
        summary: Dict[str, Any] = {
            "total_pnl": portfolio.get("total_pnl", 0.0),
            "total_trades": portfolio.get("total_trades", 0),
            "win_rate": portfolio.get("win_rate", 0.0),
            "open_positions": portfolio.get("open_positions", 0),
        }
        # Compute 7-day Sharpe from daily P&L
        daily_pnl = portfolio.get("daily_pnl", [])
        if len(daily_pnl) >= 2:
            pnls = [d.get("pnl", 0.0) for d in daily_pnl if d.get("pnl") is not None]
            if pnls:
                import statistics
                mean_pnl = statistics.mean(pnls)
                std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 1.0
                summary["sharpe_7d"] = round(mean_pnl / std_pnl, 4) if std_pnl > 0 else 0.0
            else:
                summary["sharpe_7d"] = None
        else:
            summary["sharpe_7d"] = None

        # Identify worst and best signals
        signals = report.get("signals", {}).get("per_signal_pnl", [])
        if signals:
            summary["best_signal"] = signals[0].get("signal") if signals else None
            summary["worst_signal"] = signals[-1].get("signal") if signals else None

        return summary


# ── Standalone entry point ──

def _main() -> None:
    """Run observation collector standalone for testing."""
    import sys

    db_path = os.environ.get("BOT_DB_PATH", "data/renaissance_bot.db")
    config_path = os.environ.get("BOT_CONFIG_PATH", "config/config.json")

    if not os.path.exists(db_path):
        print(f"DB not found at {db_path}")
        sys.exit(1)

    config: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    collector = ObservationCollector(db_path=db_path, config=config)
    report = collector.compile_weekly_report()
    filepath = collector.save_report(report)
    print(json.dumps(report.get("summary", {}), indent=2))
    print(f"\nFull report saved to: {filepath}")


if __name__ == "__main__":
    _main()
