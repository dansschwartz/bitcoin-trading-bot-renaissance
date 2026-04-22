"""Decision history and signal weight endpoints."""

from fastapi import APIRouter, Request

from dashboard import db_queries

router = APIRouter(prefix="/api", tags=["decisions"])


@router.get("/decisions/recent")
async def recent_decisions(request: Request, limit: int = 100):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_recent_decisions(db, limit=min(limit, 500))


@router.get("/decisions/{decision_id}")
async def decision_detail(request: Request, decision_id: int):
    db = request.app.state.dashboard_config.db_path
    result = db_queries.get_decision_by_id(db, decision_id)
    if not result:
        return {"error": "Decision not found"}
    return result


@router.get("/signals/weights")
async def signal_weights(request: Request):
    cfg = request.app.state.dashboard_config
    return {"weights": cfg.signal_weights}
