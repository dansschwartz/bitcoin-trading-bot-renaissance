"""
Update the council's shared outcome ledger from the proposals and improvement_log tables.
Run after each council session or deployment outcome.

Usage: .venv/bin/python3 scripts/update_council_memory.py
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent
LEDGER_PATH = PROJ_DIR / "data" / "council_memory" / "outcome_ledger.json"


def find_db():
    for name in ["trading.db", "renaissance_bot.db"]:
        p = PROJ_DIR / "data" / name
        if p.exists():
            return p
    return None


def update() -> None:
    db = find_db()
    if not db:
        print("No database found")
        return

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row

    # Deployed proposals
    deployed = []
    try:
        for row in conn.execute(
            "SELECT id, title, source, created_at, deployed_at, notes "
            "FROM proposals WHERE status = 'deployed' ORDER BY deployed_at DESC"
        ):
            deployed.append({
                "id": row["id"],
                "title": row["title"],
                "proposer": row["source"],
                "deployed_date": row["deployed_at"],
                "notes": row["notes"],
            })
    except Exception:
        pass

    # Rolled back
    rolled_back = []
    try:
        for row in conn.execute(
            "SELECT id, title, source, notes FROM proposals "
            "WHERE status = 'rolled_back' ORDER BY updated_at DESC"
        ):
            rolled_back.append({
                "id": row["id"],
                "title": row["title"],
                "proposer": row["source"],
                "reason": row["notes"],
            })
    except Exception:
        pass

    # Active sandbox
    sandbox = []
    try:
        for row in conn.execute(
            "SELECT id, title, source, sandbox_end FROM proposals "
            "WHERE status = 'sandboxing'"
        ):
            sandbox.append({
                "id": row["id"],
                "title": row["title"],
                "proposer": row["source"],
                "sandbox_ends": row["sandbox_end"],
            })
    except Exception:
        pass

    # Counts
    def safe_count(query: str) -> int:
        try:
            return conn.execute(query).fetchone()[0]
        except Exception:
            return 0

    total_generated = safe_count("SELECT COUNT(*) FROM proposals")
    total_deployed = safe_count("SELECT COUNT(*) FROM proposals WHERE status='deployed'")
    total_rejected = safe_count(
        "SELECT COUNT(*) FROM proposals WHERE status IN ('safety_failed','rejected')"
    )
    total_rolled = safe_count("SELECT COUNT(*) FROM proposals WHERE status='rolled_back'")

    # Cumulative improvement
    cumulative_bps = 0.0
    try:
        row = conn.execute(
            "SELECT SUM(metric_after - metric_before) FROM improvement_log WHERE reverted = 0"
        ).fetchone()
        if row and row[0]:
            cumulative_bps = round(row[0], 2)
    except Exception:
        pass

    conn.close()

    ledger = {
        "deployed_improvements": deployed,
        "rolled_back": rolled_back,
        "active_sandbox": sandbox,
        "cumulative_improvement_bps": cumulative_bps,
        "proposals_generated_total": total_generated,
        "proposals_deployed_total": total_deployed,
        "proposals_rejected_total": total_rejected,
        "proposals_rolled_back_total": total_rolled,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, default=str))
    print(f"Outcome ledger updated: {LEDGER_PATH}")
    print(json.dumps(ledger, indent=2, default=str))


if __name__ == "__main__":
    update()
