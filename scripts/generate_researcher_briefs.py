"""
Generate dynamic knowledge briefs from live code + DB before each council session.
Usage: .venv/bin/python3 scripts/generate_researcher_briefs.py
"""
import json, sqlite3, re, ast
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent
BRIEF_DIR = PROJ_DIR / "data" / "council_memory" / "briefs"

def find_db():
    for name in ["trading.db", "renaissance_bot.db"]:
        p = PROJ_DIR / "data" / name
        if p.exists(): return p
    return None

def safe_query(conn, sql, params=(), default=None):
    try: return conn.execute(sql, params).fetchall() or default
    except: return default

def extract_signal_weights():
    bot = PROJ_DIR / "renaissance_trading_bot.py"
    if not bot.exists(): return {}
    try:
        match = re.search(r"signal_weights\s*=\s*\{([^}]+)\}", bot.read_text())
        if match:
            d = "{" + match.group(1) + "}"
            d = re.sub(r"[a-zA-Z_]\w*(?!\s*['\"])", "0", d)
            return ast.literal_eval(d)
    except: pass
    return {}

def get_metrics(conn, days=7):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    m = {}
    rows = safe_query(conn, "SELECT COUNT(*), SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END), SUM(pnl), AVG(pnl) FROM decisions WHERE timestamp>=?", (cutoff,), [(0,0,0,0)])
    if rows:
        total = rows[0][0] or 0
        m.update({"trades": total, "wins": rows[0][1] or 0, "win_rate": round((rows[0][1] or 0)/max(total,1),4),
                   "total_pnl": round(rows[0][2] or 0,2), "avg_pnl": round(rows[0][3] or 0,4)})
    # Per pair
    rows = safe_query(conn, "SELECT product_id, COUNT(*), SUM(pnl) FROM decisions WHERE timestamp>=? GROUP BY product_id ORDER BY SUM(pnl) DESC", (cutoff,), [])
    m["per_pair"] = {r[0]: {"trades": r[1], "pnl": round(r[2] or 0,2)} for r in (rows or [])}
    # Model accuracy
    rows = safe_query(conn, "SELECT model_name, COUNT(*), SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) FROM model_predictions WHERE timestamp>=? GROUP BY model_name", (cutoff,), [])
    m["model_accuracy"] = {r[0]: round(r[2]/max(r[1],1),4) for r in (rows or []) if r[1]>0}
    # Regime
    rows = safe_query(conn, "SELECT regime, COUNT(*) FROM regime_history WHERE timestamp>=? GROUP BY regime", (cutoff,), [])
    total_r = sum(r[1] for r in (rows or []))
    m["regimes"] = {r[0]: round(r[1]/max(total_r,1),3) for r in (rows or [])}
    # Devil
    rows = safe_query(conn, "SELECT COUNT(*), SUM(devil_cost), AVG(slippage_bps) FROM devil_tracker WHERE timestamp>=?", (cutoff,), [(0,0,0)])
    m["devil"] = {"entries": rows[0][0] or 0, "total_cost": round(rows[0][1] or 0,2), "avg_slip_bps": round(rows[0][2] or 0,4)}
    # DB sizes
    tables = safe_query(conn, "SELECT name FROM sqlite_master WHERE type='table'", default=[])
    m["db_sizes"] = {}
    for (t,) in (tables or []):
        cnt = safe_query(conn, f"SELECT COUNT(*) FROM [{t}]", default=[(0,)])
        m["db_sizes"][t] = cnt[0][0] if cnt else 0
    return m

def get_models():
    model_dir = PROJ_DIR / "models" / "trained"
    if not model_dir.exists(): return []
    return [{"name": f.stem, "type": f.suffix, "mb": round(f.stat().st_size/1024/1024, 2)}
            for f in sorted(model_dir.iterdir()) if f.suffix in (".pth",".pkl",".joblib",".txt")]

def generate_brief(name, m, models, weights):
    ts = datetime.now(timezone.utc).isoformat()
    common = f"""## Performance (7d)
Trades: {m.get('trades','?')} | Win rate: {m.get('win_rate','?')} | P&L: ${m.get('total_pnl','?')} | Avg: ${m.get('avg_pnl','?')}

## Per-Pair
{json.dumps(m.get('per_pair',{}), indent=2)}

## Regimes
{json.dumps(m.get('regimes',{}), indent=2)}
"""
    specifics = {
        "mathematician": f"## Signal Weights\n{json.dumps(weights, indent=2)}\nTotal: {sum(weights.values()):.3f}\n\n## Models\n{json.dumps(models, indent=2)}",
        "cryptographer": f"## Model Accuracy\n{json.dumps(m.get('model_accuracy',{}), indent=2)}",
        "physicist": f"## DB Health\n{json.dumps(m.get('db_sizes',{}), indent=2)}",
        "linguist": f"## Model Accuracy\n{json.dumps(m.get('model_accuracy',{}), indent=2)}\n\n## Models\n{json.dumps(models, indent=2)}",
        "systems_engineer": f"## Devil Tracker\n{json.dumps(m.get('devil',{}), indent=2)}\n\n## DB Health\n{json.dumps(m.get('db_sizes',{}), indent=2)}",
    }
    return f"# {name.upper()} — Dynamic Brief\n# Generated: {ts}\n\n{common}\n{specifics.get(name, '')}\n"

def main():
    BRIEF_DIR.mkdir(parents=True, exist_ok=True)
    db = find_db()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True) if db else sqlite3.connect(":memory:")
    m = get_metrics(conn)
    models = get_models()
    weights = extract_signal_weights()
    for name in ["mathematician", "cryptographer", "physicist", "linguist", "systems_engineer"]:
        brief = generate_brief(name, m, models, weights)
        out = BRIEF_DIR / f"{name}_brief.md"
        out.write_text(brief)
        print(f"Generated: {out} ({len(brief)} chars)")
    conn.close()

if __name__ == "__main__":
    main()
