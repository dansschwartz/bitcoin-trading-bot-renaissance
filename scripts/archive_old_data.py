#!/usr/bin/env python3
"""Archive old data from renaissance_bot.db.

Moves rows older than a cutoff date into an archive DB,
then deletes them from the main DB and VACUUMs.
"""

import sqlite3
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Configuration
MAIN_DB = Path(__file__).resolve().parent.parent / "data" / "renaissance_bot.db"
ARB_DB = Path(__file__).resolve().parent.parent / "data" / "arbitrage.db"
CUTOFF = "2026-03-06T00:00:00"  # Keep data from March 6 onward

# Tables with timestamp columns to archive
# Format: (table_name, timestamp_column)
MAIN_TABLES = [
    ("decisions", "timestamp"),
    ("five_minute_bars", "bar_start"),
    ("ml_predictions", "timestamp"),
    ("market_data", "timestamp"),
    ("token_spray_log", "opened_at"),
    ("breakout_scans", "timestamp"),
    ("polymarket_scanner", "scanned_at"),
    ("portfolio_snapshots", "timestamp"),
    ("straddle_log", "opened_at"),
    ("straddle_legs", "opened_at"),
    ("polymarket_skip_log", "timestamp"),
    ("polymarket_lifecycle", "created_at"),
    ("polymarket_bankroll_log", "timestamp"),
    ("system_state_log", "timestamp"),
    ("sub_bar_events", "timestamp"),
    ("random_baseline", "opened_at"),
]

ARB_TABLES = [
    ("arb_trades", "timestamp"),
    ("arb_signals", "timestamp"),
]


def archive_db(db_path: Path, tables: list, cutoff: str, label: str):
    """Archive old rows from a database."""
    if not db_path.exists():
        print(f"  {label}: DB not found at {db_path}, skipping")
        return

    archive_path = db_path.parent / f"{db_path.stem}_archive_{datetime.now().strftime('%Y%m%d')}.db"

    # Create backup first
    backup_path = db_path.parent / f"{db_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    print(f"  Backing up {db_path.name} -> {backup_path.name}")
    shutil.copy2(db_path, backup_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")

    # Attach archive DB
    conn.execute(f"ATTACH DATABASE '{archive_path}' AS archive")

    total_archived = 0
    total_deleted = 0

    for table_name, ts_col in tables:
        try:
            # Check if table exists
            exists = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            ).fetchone()[0]
            if not exists:
                continue

            # Count rows to archive
            old_count = conn.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE {ts_col} < ?",
                (cutoff,)
            ).fetchone()[0]

            if old_count == 0:
                continue

            total_before = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            # Create table in archive DB (copy schema)
            schema = conn.execute(
                f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            ).fetchone()[0]
            archive_schema = schema.replace(
                f"CREATE TABLE {table_name}",
                f"CREATE TABLE IF NOT EXISTS archive.{table_name}",
                1
            )
            # Handle quoted table names
            archive_schema = archive_schema.replace(
                f'CREATE TABLE "{table_name}"',
                f'CREATE TABLE IF NOT EXISTS archive."{table_name}"',
                1
            )
            try:
                conn.execute(archive_schema)
            except sqlite3.OperationalError:
                # Table might already exist in archive
                pass

            # Copy old rows to archive
            conn.execute(
                f"INSERT OR IGNORE INTO archive.{table_name} SELECT * FROM {table_name} WHERE {ts_col} < ?",
                (cutoff,)
            )

            # Delete old rows from main
            conn.execute(
                f"DELETE FROM {table_name} WHERE {ts_col} < ?",
                (cutoff,)
            )

            remaining = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  {table_name}: {total_before} -> {remaining} rows (archived {old_count})")
            total_archived += old_count
            total_deleted += old_count

        except Exception as e:
            print(f"  {table_name}: ERROR - {e}")

    conn.execute("DETACH DATABASE archive")
    conn.commit()

    if total_deleted > 0:
        print(f"  VACUUMing {db_path.name}...")
        conn.execute("VACUUM")

    conn.close()

    print(f"  {label} total: archived {total_archived} rows, archive at {archive_path.name}")

    # Clean up backup if nothing changed
    if total_archived == 0:
        backup_path.unlink(missing_ok=True)
        print(f"  No changes, removed backup")


def main():
    cutoff = CUTOFF
    if len(sys.argv) > 1:
        cutoff = sys.argv[1]

    print(f"Archiving data older than {cutoff}")
    print(f"Current time: {datetime.now(timezone.utc).isoformat()}")
    print()

    print("=== Main DB (renaissance_bot.db) ===")
    archive_db(MAIN_DB, MAIN_TABLES, cutoff, "Main DB")
    print()

    print("=== Arb DB (arbitrage.db) ===")
    archive_db(ARB_DB, ARB_TABLES, cutoff, "Arb DB")
    print()

    # Show final row counts
    print("=== Final Row Counts ===")
    for db_path, label in [(MAIN_DB, "Main"), (ARB_DB, "Arb")]:
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        print(f"\n{label} DB:")
        for (t,) in tables:
            if t.startswith("sqlite_"):
                continue
            cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            if cnt > 0:
                print(f"  {t}: {cnt}")
        conn.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
