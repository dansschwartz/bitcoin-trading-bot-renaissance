"""Brain endpoints — ML ensemble, regime, VAE, confluence."""

from fastapi import APIRouter, Request

from dashboard import db_queries

router = APIRouter(prefix="/api/brain", tags=["brain"])


@router.get("/ensemble")
async def ensemble_status(request: Request):
    """Current ML ensemble state (from bot emitter if live, else last DB snapshot)."""
    db = request.app.state.dashboard_config.db_path
    preds = db_queries.get_ml_predictions_history(db, hours=1)

    # Group by model name, take latest
    latest_by_model = {}
    for p in preds:
        model = p.get("model_name", "unknown")
        if model not in latest_by_model:
            latest_by_model[model] = p

    return {
        "models": latest_by_model,
        "model_count": len(latest_by_model),
    }


@router.get("/predictions/history")
async def prediction_history(request: Request, hours: int = 24):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_ml_predictions_history(db, hours=min(hours, 168))


@router.get("/regime")
async def regime_status(request: Request):
    db = request.app.state.dashboard_config.db_path
    history = db_queries.get_regime_history(db, limit=50)
    current = history[0] if history else None

    # Overlay live regime from emitter cache (has classifier + bar_count)
    emitter = getattr(request.app.state, "emitter", None)
    live = emitter.get_cached("regime") if emitter else None
    if live:
        current = {
            **(current or {}),
            "hmm_regime": live.get("hmm_regime", "unknown"),
            "confidence": live.get("confidence", 0.0),
            "classifier": live.get("classifier", "none"),
            "bar_count": live.get("bar_count", 0),
            "details": live.get("details", ""),
        }

    return {
        "current": current,
        "history": history,
    }


@router.get("/confluence")
async def confluence_status(request: Request):
    """Confluence data — populated via emitter cache from live bot."""
    # Primary: read from emitter's channel cache (works even with 0 WS clients)
    emitter = getattr(request.app.state, "emitter", None)
    if emitter:
        cached = emitter.get_cached("confluence")
        if cached:
            return cached
    # Fallback: legacy WS relay cache
    legacy = getattr(request.app.state, "last_confluence", None)
    return legacy or {"status": "no_live_data", "message": "Waiting for bot cycle"}


@router.get("/vae")
async def vae_history(request: Request):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_vae_history(db, limit=200)
