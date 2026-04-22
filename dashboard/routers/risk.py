"""Risk monitoring endpoints — exposure, metrics, gateway log, alerts, leverage."""

import sqlite3
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Request

from dashboard import db_queries

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("/exposure")
async def exposure(request: Request):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_exposure(db)


@router.get("/metrics")
async def risk_metrics(request: Request):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_risk_metrics(db)


@router.get("/gateway/log")
async def gateway_log(request: Request, limit: int = 100):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_risk_gateway_log(db, limit=min(limit, 500))


@router.get("/alerts")
async def active_alerts(request: Request):
    """Active alerts — generated from risk metrics + any live WebSocket alerts."""
    db = request.app.state.dashboard_config.db_path
    cfg = request.app.state.dashboard_config
    thresholds = cfg.dashboard.get("alerts", {})

    # Compute alerts from current risk state
    computed = db_queries.evaluate_risk_alerts(db, thresholds if thresholds else None)

    # Merge with any live WS-pushed alerts
    ws_alerts = getattr(request.app.state, "active_alerts", [])

    return {"alerts": computed + ws_alerts}


@router.get("/leverage")
async def leverage_info(request: Request):
    """Current leverage status from LeverageManager data."""
    db = request.app.state.dashboard_config.db_path
    try:
        conn = sqlite3.connect(db, timeout=5.0)
        conn.row_factory = sqlite3.Row

        # Compute current leverage from open positions vs equity
        positions = conn.execute(
            "SELECT product_id, side, size, entry_price FROM open_positions WHERE status='OPEN'"
        ).fetchall()

        gross_exposure = 0.0
        for p in positions:
            gross_exposure += abs(float(p["size"]) * float(p["entry_price"]))

        # Get equity estimate from recent market data
        initial_capital = 10000.0  # paper trading default
        current_leverage = gross_exposure / initial_capital if initial_capital > 0 else 0.0

        # Get recent trade performance for consistency metrics
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        trades = conn.execute(
            "SELECT COUNT(*) as cnt FROM trades WHERE timestamp >= ?",
            (cutoff,),
        ).fetchone()

        conn.close()

        return {
            "current_leverage": round(current_leverage, 4),
            "gross_exposure_usd": round(gross_exposure, 2),
            "initial_capital_usd": initial_capital,
            "open_positions": len(positions),
            "trades_30d": trades["cnt"] if trades else 0,
        }
    except Exception as e:
        return {"error": str(e), "current_leverage": 0.0}


@router.get("/alerts/recent")
async def recent_alerts(request: Request, limit: int = 50):
    """Historical alerts from system_state_log table."""
    db = request.app.state.dashboard_config.db_path
    try:
        conn = sqlite3.connect(db, timeout=5.0)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM system_state_log ORDER BY rowid DESC LIMIT ?",
            (min(limit, 200),),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e), "alerts": []}
