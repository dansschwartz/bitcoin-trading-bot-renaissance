"""Decision Audit Log dashboard endpoints — SQL-powered signal attribution & gate analysis."""

import logging
from fastapi import APIRouter, Request, Query
from dashboard import db_queries

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("/summary")
async def audit_summary(request: Request, hours: int = Query(default=24, ge=1, le=168)):
    """Aggregated audit summary: gate blocks, signal attribution, regime distribution."""
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_audit_summary(db, hours=hours)


@router.get("/recent")
async def audit_recent(request: Request, limit: int = Query(default=50, ge=1, le=500)):
    """Most recent audit rows (flat pipeline traces)."""
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_audit_recent(db, limit=limit)
