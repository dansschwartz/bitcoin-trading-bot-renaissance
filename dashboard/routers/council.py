"""Research Council dashboard endpoints — reads session data from disk."""

import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/council", tags=["council"])

SESSIONS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "research_sessions"
RESEARCHERS = ["mathematician", "cryptographer", "physicist", "linguist", "systems_engineer"]

# Simple TTL cache for session list
_cache: dict[str, Any] = {"sessions": None, "ts": 0.0}
CACHE_TTL = 60  # seconds


def _scan_sessions() -> list[dict[str, Any]]:
    """Scan research_sessions directory and return metadata for each session."""
    now = time.time()
    if _cache["sessions"] is not None and (now - _cache["ts"]) < CACHE_TTL:
        return _cache["sessions"]

    sessions: list[dict[str, Any]] = []
    if not SESSIONS_DIR.is_dir():
        return sessions

    for d in sorted(SESSIONS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        session_id = d.name
        ranked_path = d / "ranked_proposals.json"

        proposal_count = 0
        review_count = 0
        consensus_count = 0

        if ranked_path.exists():
            try:
                ranked = json.loads(ranked_path.read_text())
                proposal_count = len(ranked)
                consensus_count = sum(1 for p in ranked if p.get("passes_consensus"))
                review_count = sum(p.get("review_count", 0) for p in ranked)
            except Exception:
                pass

        # Parse timestamp from session_id (format: YYYYMMDD_HHMMSS)
        timestamp = None
        try:
            from datetime import datetime
            timestamp = datetime.strptime(session_id, "%Y%m%d_%H%M%S").isoformat()
        except ValueError:
            timestamp = session_id

        sessions.append({
            "session_id": session_id,
            "timestamp": timestamp,
            "proposal_count": proposal_count,
            "review_count": review_count,
            "consensus_count": consensus_count,
        })

    _cache["sessions"] = sessions
    _cache["ts"] = now
    return sessions


def _load_ranked(session_id: str) -> list[dict[str, Any]]:
    """Load ranked_proposals.json for a session."""
    path = SESSIONS_DIR / session_id / "ranked_proposals.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


@router.get("/sessions")
async def list_sessions():
    """List all council sessions with summary stats."""
    return _scan_sessions()


@router.get("/sessions/{session_id}")
async def session_detail(session_id: str):
    """Full detail for a single council session."""
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.is_dir():
        raise HTTPException(404, f"Session {session_id} not found")

    proposals = _load_ranked(session_id)

    # Parse timestamp
    timestamp = session_id
    try:
        from datetime import datetime
        timestamp = datetime.strptime(session_id, "%Y%m%d_%H%M%S").isoformat()
    except ValueError:
        pass

    scores = [p.get("consensus_score", 0) for p in proposals if p.get("consensus_score")]
    return {
        "session_id": session_id,
        "timestamp": timestamp,
        "proposals": proposals,
        "stats": {
            "total": len(proposals),
            "consensus_passed": sum(1 for p in proposals if p.get("passes_consensus")),
            "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
        },
    }


@router.get("/sessions/{session_id}/researchers/{name}")
async def researcher_detail(session_id: str, name: str):
    """Proposals and reviews from a specific researcher in a session."""
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.is_dir():
        raise HTTPException(404, f"Session {session_id} not found")
    if name not in RESEARCHERS:
        raise HTTPException(400, f"Unknown researcher: {name}")

    proposals: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []

    prop_path = session_dir / "proposals" / name / "proposals.json"
    if prop_path.exists():
        try:
            proposals = json.loads(prop_path.read_text())
        except Exception:
            pass

    rev_path = session_dir / "reviews" / name / "reviews.json"
    if rev_path.exists():
        try:
            reviews = json.loads(rev_path.read_text())
        except Exception:
            pass

    return {
        "name": name,
        "proposals": proposals,
        "reviews": reviews,
        "proposal_count": len(proposals),
        "review_count": len(reviews),
    }


@router.get("/latest")
async def latest_session():
    """Convenience: return the most recent session's data."""
    sessions = _scan_sessions()
    if not sessions:
        return {"session_id": None, "proposals": [], "stats": {"total": 0, "consensus_passed": 0, "avg_score": 0}}

    latest_id = sessions[0]["session_id"]
    proposals = _load_ranked(latest_id)
    scores = [p.get("consensus_score", 0) for p in proposals if p.get("consensus_score")]

    return {
        **sessions[0],
        "proposals": proposals,
        "stats": {
            "total": len(proposals),
            "consensus_passed": sum(1 for p in proposals if p.get("passes_consensus")),
            "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
        },
    }
