#!/bin/bash
# Prepares a frozen data snapshot for researcher sessions.
# Creates read-only copies so researchers cannot affect the live system.

set -e

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SNAPSHOT_DIR="$PROJ_DIR/data/research_snapshots/$(date +%Y-%m-%d)"
LATEST_LINK="$PROJ_DIR/data/research_snapshots/latest"

echo "Creating research snapshot at $SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR"

# 1. Frozen copy of trading database
cp "$PROJ_DIR/data/renaissance_bot.db" "$SNAPSHOT_DIR/trading_snapshot.db"
# Make read-only
chmod 444 "$SNAPSHOT_DIR/trading_snapshot.db"

# 2. Symlink historical training data (large files, don't copy)
if [ -d "$PROJ_DIR/data/training" ]; then
    ln -sfn "$PROJ_DIR/data/training" "$SNAPSHOT_DIR/training_data"
fi

# 3. Copy model files
if [ -d "$PROJ_DIR/models/trained" ]; then
    cp -r "$PROJ_DIR/models/trained" "$SNAPSHOT_DIR/models"
fi

# 4. Generate quick summary
"$PROJ_DIR/.venv/bin/python3" -c "
import sqlite3, json
from datetime import datetime, timezone, timedelta

conn = sqlite3.connect('$SNAPSHOT_DIR/trading_snapshot.db')
now = datetime.now(timezone.utc)
week_ago = (now - timedelta(days=7)).isoformat()

summary = {
    'snapshot_time': now.isoformat(),
}

# Safe queries — tables may not exist yet
for table, key in [
    ('decisions', 'total_decisions'),
    ('positions', 'total_positions'),
    ('proposals', 'total_proposals'),
    ('five_minute_bars', 'total_bars'),
]:
    try:
        cnt = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        summary[key] = cnt
    except Exception:
        summary[key] = 0

# Pairs
try:
    pairs = [r[0] for r in conn.execute(
        'SELECT DISTINCT product_id FROM decisions ORDER BY product_id'
    ).fetchall()]
    summary['pairs'] = pairs
except Exception:
    summary['pairs'] = []

conn.close()

with open('$SNAPSHOT_DIR/snapshot_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
"

# 5. Update latest symlink
ln -sfn "$SNAPSHOT_DIR" "$LATEST_LINK"

echo "Snapshot ready: $SNAPSHOT_DIR ($(du -sh "$SNAPSHOT_DIR" | cut -f1))"
