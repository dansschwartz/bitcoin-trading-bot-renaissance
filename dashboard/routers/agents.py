"""Agent coordination dashboard endpoints (Doc 15)."""

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])


def _conn(db_path: str):
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_count(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    """Run a COUNT query, returning 0 if the table doesn't exist."""
    try:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


@router.get("/status")
async def agent_statuses(request: Request):
    """Return status for all 7 agents (if coordinator is available)."""
    db = request.app.state.dashboard_config.db_path
    try:
        # Try to get live agent statuses from coordinator if available
        coordinator = getattr(request.app.state, "agent_coordinator", None)
        if coordinator:
            return coordinator.get_all_statuses()

        # Fallback: query real data sources per agent
        conn = _conn(db)

        # Each agent gets its event count from the most relevant table
        agent_queries: dict[str, str] = {
            "signal": "SELECT COUNT(*) FROM agent_events WHERE agent_name = 'signal'",
            "data": "SELECT COUNT(*) FROM five_minute_bars WHERE timestamp > datetime('now','-24 hours')",
            "risk": "SELECT COUNT(*) FROM risk_gateway_log",
            "execution": None,  # special: sum of polymarket_bets + breakout_bets
            "portfolio": "SELECT COUNT(*) FROM positions WHERE status='closed'",
            "monitoring": "SELECT COUNT(DISTINCT scan_time) FROM breakout_scans",
            "meta": None,  # always 0
        }

        statuses = []
        for name in ["data", "signal", "risk", "execution", "portfolio", "monitoring", "meta"]:
            q = agent_queries.get(name)
            if name == "execution":
                count = (_safe_count(conn, "SELECT COUNT(*) FROM polymarket_bets")
                         + _safe_count(conn, "SELECT COUNT(*) FROM breakout_bets"))
            elif name == "meta":
                count = 0
            elif q:
                count = _safe_count(conn, q)
            else:
                count = 0

            # Last event from agent_events (best-effort)
            last_event = None
            try:
                row = conn.execute(
                    "SELECT MAX(timestamp) FROM agent_events WHERE agent_name = ?",
                    (name,),
                ).fetchone()
                if row:
                    last_event = row[0]
            except sqlite3.OperationalError:
                pass

            statuses.append({
                "name": name,
                "state": "active" if count > 0 else "idle",
                "event_count": count,
                "last_event": last_event,
            })
        conn.close()
        return statuses
    except Exception as e:
        return [{"name": "error", "state": "error", "error": str(e)}]


@router.get("/events")
async def agent_events(
    request: Request,
    limit: int = Query(100, le=500),
    agent: Optional[str] = None,
    severity: Optional[str] = None,
):
    """Recent agent events from agent_events table."""
    db = request.app.state.dashboard_config.db_path
    try:
        conn = _conn(db)
        query = "SELECT * FROM agent_events WHERE 1=1"
        params: list = []
        if agent:
            query += " AND agent_name = ?"
            params.append(agent)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        conn.close()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("payload"):
                try:
                    d["payload"] = json.loads(d["payload"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(d)
        return result
    except sqlite3.OperationalError:
        return []
    except Exception as e:
        return {"error": str(e)}


@router.get("/proposals")
async def proposals(
    request: Request,
    status: Optional[str] = None,
    limit: int = Query(50, le=200),
):
    """Proposals with safety gate results."""
    db = request.app.state.dashboard_config.db_path
    try:
        conn = _conn(db)
        query = "SELECT * FROM proposals WHERE 1=1"
        params: list = []
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        conn.close()
        result = []
        for r in rows:
            d = dict(r)
            for json_field in ("config_changes", "safety_gate_results"):
                if d.get(json_field):
                    try:
                        d[json_field] = json.loads(d[json_field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            result.append(d)
        return result
    except sqlite3.OperationalError:
        return []
    except Exception as e:
        return {"error": str(e)}


@router.get("/improvements")
async def improvements(request: Request, limit: int = Query(50, le=200)):
    """Improvement log timeline."""
    db = request.app.state.dashboard_config.db_path
    try:
        conn = _conn(db)
        rows = conn.execute(
            "SELECT * FROM improvement_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            # Seed with actual changes — use existing schema columns
            # Schema: id, timestamp, proposal_id, change_type, description,
            #         metric_before, metric_after, config_snapshot, reverted
            seed = [
                ("2026-02-27", None, "risk", "Stop loss (40%) + never-add-to-losers gate for Polymarket Strategy A",
                 "no stop loss", "40% stop loss active", None, 0),
                ("2026-02-27", None, "model", "Disabled BiLSTM, DilatedCNN, GRU — kept QT, CNN, LightGBM, Meta",
                 "7 models (noisy)", "4 models (clean)", None, 0),
                ("2026-02-26", None, "strategy", "Strategy A v3 — confidence-gated entry with active management",
                 "no Polymarket trading", "live Polymarket trades", None, 0),
                ("2026-02-26", None, "strategy", "$2K breakout scanner with separate wallet and dashboard",
                 "no breakout strategy", "breakout strategy live", None, 0),
                ("2026-02-22", None, "infra", "Expanded universe v2 — dynamic Binance pairs, 70-90 assets, 4-tier scan",
                 "3 pairs", "70-90 pairs", None, 0),
                ("2026-02-25", None, "infra", "Cross-exchange realistic fills — withdrawal fees, taker penalty, adverse moves",
                 "phantom P&L", "realistic P&L", None, 0),
            ]
            for ts, pid, ct, desc, mb, ma, cs, rev in seed:
                conn.execute(
                    "INSERT INTO improvement_log (timestamp, proposal_id, change_type, description, metric_before, metric_after, config_snapshot, reverted) VALUES (?,?,?,?,?,?,?,?)",
                    (ts, pid, ct, desc, mb, ma, cs, rev),
                )
            conn.commit()
            rows = conn.execute(
                "SELECT * FROM improvement_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"improvements error: {e}")
        return []


@router.get("/reports/latest")
async def latest_report(request: Request):
    """Most recent weekly observation report."""
    db = request.app.state.dashboard_config.db_path
    try:
        conn = _conn(db)
        row = conn.execute(
            "SELECT * FROM weekly_reports ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            d = dict(row)
            if d.get("report_json"):
                try:
                    d["report_json"] = json.loads(d["report_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
            return d
        return {"message": "No weekly reports available yet"}
    except sqlite3.OperationalError:
        return {"message": "Weekly reports table not yet created"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/models")
async def model_ledger(request: Request, limit: int = Query(20, le=100)):
    """Model ledger — all versions, accuracy, status."""
    db = request.app.state.dashboard_config.db_path
    try:
        conn = _conn(db)
        # Schema: id, timestamp, model_name, model_version, accuracy, sharpe,
        #         max_drawdown, file_path, file_hash, status, replaced_by
        rows = conn.execute(
            "SELECT * FROM model_ledger ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            seed = [
                ("quantum_transformer", "v7-retrain", None, None, None,
                 "models/trained/quantum_transformer.pth", None, "active", None),
                ("cnn", "v7-retrain", None, None, None,
                 "models/trained/cnn_model.pth", None, "active", None),
                ("lightgbm", "v7-retrain", None, None, None,
                 "models/trained/lightgbm_model.pkl", None, "active", None),
                ("meta_ensemble", "v7-retrain", None, None, None,
                 "models/trained/meta_ensemble.pth", None, "active", None),
                ("bilstm", "v7-retrain", None, None, None,
                 "models/trained/bilstm_model.pth", None, "disabled", None),
                ("dilated_cnn", "v7-retrain", None, None, None,
                 "models/trained/dilated_cnn_model.pth", None, "disabled", None),
                ("gru", "v7-retrain", None, None, None,
                 "models/trained/gru_model.pth", None, "disabled", None),
            ]
            for mn, mv, acc, sh, md, fp, fh, st, rb in seed:
                conn.execute(
                    "INSERT INTO model_ledger (model_name, model_version, accuracy, sharpe, max_drawdown, file_path, file_hash, status, replaced_by) VALUES (?,?,?,?,?,?,?,?,?)",
                    (mn, mv, acc, sh, md, fp, fh, st, rb),
                )
            conn.commit()
            rows = conn.execute(
                "SELECT * FROM model_ledger ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"model_ledger error: {e}")
        return []
