"""Brain endpoints — ML ensemble, regime, VAE, confluence."""

import json
import logging
import sqlite3

from fastapi import APIRouter, Request

from dashboard import db_queries

logger = logging.getLogger(__name__)
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
    elif current:
        # DB fallback: derive classifier + bar_count when emitter is unavailable
        # (bot and dashboard run as separate processes)
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            bar_count = conn.execute(
                "SELECT COUNT(*) FROM five_minute_bars"
            ).fetchone()[0]
            conn.close()
            classifier = "hmm" if bar_count >= 200 else "bootstrap"
            current["classifier"] = classifier
            current["bar_count"] = bar_count
        except Exception as e:
            logger.debug(f"Bar count query failed: {e}")
            current.setdefault("classifier", "hmm")
            current.setdefault("bar_count", 0)

    return {
        "current": current,
        "history": history,
    }


@router.get("/confluence")
async def confluence_status(request: Request):
    """Confluence data — from emitter cache or reconstructed from DB."""
    # Primary: read from emitter's channel cache (works even with 0 WS clients)
    emitter = getattr(request.app.state, "emitter", None)
    if emitter:
        cached = emitter.get_cached("confluence")
        if cached:
            return cached
    # Fallback: legacy WS relay cache
    legacy = getattr(request.app.state, "last_confluence", None)
    if legacy:
        return legacy

    # DB fallback: reconstruct confluence from latest decision's signals
    db = request.app.state.dashboard_config.db_path
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        row = conn.execute(
            "SELECT reasoning, timestamp FROM decisions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row and row[0]:
            reasoning = json.loads(row[0])
            signals = reasoning.get("signal_contributions", {})
            if signals:
                # Run confluence engine against latest signals
                from confluence_engine import ConfluenceEngine
                ce = ConfluenceEngine()
                result = ce.calculate_confluence_boost(signals)
                result["source"] = "db_fallback"
                result["decision_timestamp"] = row[1]
                return result
    except Exception as e:
        logger.debug(f"Confluence DB fallback failed: {e}")

    return {"status": "no_live_data", "message": "Waiting for bot cycle"}


@router.get("/regime/hierarchy")
async def regime_hierarchy(request: Request):
    """Hierarchical regime state — macro, crypto, micro, and model route."""
    emitter = getattr(request.app.state, "emitter", None)
    if emitter:
        cached = emitter.get_cached("regime_hierarchy")
        if cached:
            return {"status": "live", **cached}
    return {
        "status": "waiting",
        "macro": {"regime": "UNKNOWN", "confidence": 0.0},
        "crypto": {"regime": "UNKNOWN", "confidence": 0.0},
        "router": None,
    }


@router.get("/vae")
async def vae_history(request: Request):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_vae_history(db, limit=200)
