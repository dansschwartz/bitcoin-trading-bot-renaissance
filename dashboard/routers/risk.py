"""Risk monitoring endpoints — exposure, metrics, gateway log, alerts."""

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
    """Active alerts — populated from emitter state."""
    alerts = getattr(request.app.state, "active_alerts", [])
    return {"alerts": alerts}
