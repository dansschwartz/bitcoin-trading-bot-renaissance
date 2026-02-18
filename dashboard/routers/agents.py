"""Agent coordination dashboard endpoints (Doc 15)."""

import json
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, Query

router = APIRouter(prefix="/api/agents", tags=["agents"])


def _conn(db_path: str):
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/status")
async def agent_statuses(request: Request):
    """Return status for all 7 agents (if coordinator is available)."""
    db = request.app.state.dashboard_config.db_path
    try:
        # Try to get live agent statuses from coordinator if available
        coordinator = getattr(request.app.state, "agent_coordinator", None)
        if coordinator:
            return coordinator.get_all_statuses()

        # Fallback: return basic status from DB
        conn = _conn(db)
        agent_names = ["data", "signal", "risk", "execution", "portfolio", "monitoring", "meta"]
        statuses = []
        for name in agent_names:
            row = conn.execute(
                """SELECT COUNT(*) as events,
                          MAX(timestamp) as last_event
                   FROM agent_events WHERE agent_name = ?""",
                (name,),
            ).fetchone()
            statuses.append({
                "name": name,
                "state": "active" if row and row["events"] > 0 else "idle",
                "event_count": row["events"] if row else 0,
                "last_event": row["last_event"] if row else None,
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
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    except Exception as e:
        return {"error": str(e)}


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
        rows = conn.execute(
            "SELECT * FROM model_ledger ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    except Exception as e:
        return {"error": str(e)}
