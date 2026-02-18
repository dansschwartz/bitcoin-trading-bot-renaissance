# Renaissance Trading Bot — Claude Code Operating Manual

> **Read this entire file before starting any work.**
> **This is your operating context for autonomous multi-hour sessions.**

---

## IDENTITY & MISSION

You are a senior quantitative systems engineer working on a Renaissance Technologies-inspired
cryptocurrency trading bot. The system paper trades on MEXC and Binance exchanges via REST
and WebSocket APIs, runs a real-time dashboard at localhost:8080, and targets consistent
daily profit through statistical edge extraction across multiple strategies.

Your job: make this system work correctly, profitably, and reliably. You operate
autonomously. You do not stop to ask permission for routine operations. You verify
your own work with evidence before moving on.

---

## PROJECT STRUCTURE

```
bitcoin-trading-bot-renaissance/
├── main.py                          # Main bot entry point — the core loop
├── config.yaml                      # All configuration (strategies, thresholds, pairs, etc.)
├── dashboard/                       # Web dashboard (localhost:8080)
├── models/                          # Trained ML models (.pkl files)
│   └── hmm_regime.pkl               # Trained 3-state GaussianHMM (exists but not wired correctly)
├── data/
│   └── trading.db                   # SQLite database (positions, decisions, bars, P&L)
├── .venv/                           # Python virtual environment
├── logs/                            # Application logs
└── [spec documents]                 # 12 specification documents (see below)
```

**Tech stack:** Python 3.11, SQLite, WebSockets, REST APIs, hmmlearn, scikit-learn, asyncio
**Exchanges:** MEXC (primary), Binance (arbitrage/reference)
**Dashboard:** HTML/JS frontend served by Python backend at port 8080

---

## SPECIFICATION DOCUMENTS — YOUR REFERENCE LIBRARY

These 12 documents contain ~14,700 lines of detailed specifications. They are your
source of truth. When fixing bugs or building features, READ THE RELEVANT SPEC FIRST.

| Doc | File | Lines | Purpose |
|-----|------|-------|---------|
| 1 | RENAISSANCE_OF_ONE_CLAUDE_CODE_BLUEPRINT.md | 1,038 | System philosophy, architecture, strategies |
| 2 | ARBITRAGE_MODULE_SPEC_FOR_CLAUDE_CODE.md | 2,052 | Cross-exchange arb, funding rate, triangular |
| 3 | OPERATIONS_AND_INTELLIGENCE_SPEC.md | 1,916 | Ops, monitoring, Telegram, health checks |
| 4 | MEDALLION_INTELLIGENCE_EXTRACTION.md | 541 | Book findings — Renaissance techniques |
| 5 | MEDALLION_INTELLIGENCE_ADDENDUM.md | 285 | Web research — additional Renaissance intel |
| 6 | MEDALLION_CLAUDE_CODE_SPECS.md | 1,672 | 15 new modules (Devil Tracker, HMM, Kelly, etc.) |
| 7 | SYSTEM_INTEGRATION_MANIFEST.md | 916 | How all modules wire together |
| 8 | SYSTEM_ERRATA_AND_FIXES.md | 785 | 14 cross-document bugs with fixes |
| 9 | CODEBASE_UPDATE_RECOMMENDATIONS.md | 844 | WebSocket, speed tiers, multi-bot orchestrator |
| 10 | CONTINUOUS_POSITION_REEVALUATION_ENGINE.md | 1,776 | Continuous position re-evaluation every cycle |
| 11 | MULTI_HORIZON_PROBABILITY_ESTIMATOR.md | 2,126 | Probability cones across time horizons |
| 12 | DASHBOARD_FIXES.md | 920 | Dashboard bugs and fixes for all 6 tabs |

**When in doubt, consult the spec.** The specs were written by an architect who has
studied the entire system. They contain root cause analysis, code examples, and
exact fix instructions.

---

## CURRENT PRIORITIES (updated 2026-02-15)

### 🔴 CRITICAL — Fix immediately
1. **Regime detector not classifying** — HMM needs OHLCV bars from five_minute_bars table,
   not ticker snapshots. Seed historical bars via exchange API (200 five-minute candles
   per pair). Implement bootstrap regime rules for immediate classification.
   See: DASHBOARD_FIXES.md Bug B-1

2. **Position netting missing** — System opens opposing positions (long AND short) on same
   asset simultaneously, bleeding money through spread costs.
   See: DASHBOARD_FIXES.md Bug P-1

3. **Low_volatility regime blocks all trading** — The regime label "Low_volatility" is
   zeroing out signal confidence. Low vol should BOOST mean reversion, not kill all signals.
   Find the code path: regime label → signal confidence multiplier → 0.0%.

4. **Exposure calculation broken** — Shows $0 despite 41 open positions.
   See: DASHBOARD_FIXES.md Bug X-1

5. **Equity tracking broken** — Max drawdown 0%, peak equity $0, Sharpe blank.
   See: DASHBOARD_FIXES.md Bug X-2

### 🟡 HIGH — Fix soon
6. Realized vs unrealized P&L split (Bug X-3)
7. VAE loss not persisted to database (add to decision persist dict)
8. Confluence WebSocket relay stale (clear on bot startup)
9. Risk alerts not firing (Bug R-3)
10. Risk gateway log empty (Bug R-4)

### 🟢 MEDIUM — Enhance when critical items done
11. Activity feed filtering (Bug CC-1)
12. Asset summary replacing position list on Command Center (Bug CC-4)
13. System health bar (Bug CC-5)
14. Doc 8 errata — 14 cross-document bugs

### 🔵 FUTURE — After system is stable and profitable
15. Doc 9 — WebSocket streams, speed tiers, multi-bot
16. Doc 10 — Continuous position re-evaluation engine
17. Doc 11 — Multi-horizon probability estimator

---

## AUTONOMOUS OPERATION RULES

### What you SHOULD do without asking:
- Read any file in the project
- Edit any Python, JavaScript, HTML, CSS, YAML, or config file
- Run Python scripts, tests, grep, find, curl localhost, git commands
- Query the SQLite database to diagnose issues
- Install Python packages via pip (use .venv/bin/pip)
- Restart the bot process to test changes
- Hit dashboard API endpoints (curl localhost:8080/api/*)
- Create new files (modules, tests, configs)
- Run diagnostic commands to verify fixes
- Commit changes to git with descriptive messages
- Move to the next priority item when current one is verified fixed

### What you should NEVER do:
- Delete the database without backing it up first
- Push to remote repositories
- Modify .env files or credentials
- Run commands that require sudo
- Make network calls to external services other than MEXC/Binance APIs
- Deploy to production or switch from paper trading to live trading
- Ignore a failing test — fix it or explain why it's expected

### The Verification Rule (MOST IMPORTANT)
**Never claim something is "fixed" without evidence.**

After every fix:
1. Show the code change (what changed and why)
2. Show runtime evidence that the fix works:
   - Query the database: `.venv/bin/python3 -c "import sqlite3; ..."`
   - Hit the API: `curl -s localhost:8080/api/brain/regime | python3 -m json.tool`
   - Check logs: `grep -i "regime" logs/*.log | tail -20`
   - Check the dashboard data directly
3. If you can't verify (e.g., need to wait for data), say so explicitly:
   "Fix deployed. Will need 5 minutes of runtime to verify bar accumulation."

**Example of GOOD verification:**
```
Fixed: RegimeDetector now reads from five_minute_bars table.
Evidence: 
  $ .venv/bin/python3 -c "..."
  > Current regime: 'mean_reverting', confidence: 0.72, bars_used: 187
  Dashboard API confirms:
  $ curl -s localhost:8080/api/brain/regime
  > {"regime": "mean_reverting", "confidence": 0.72, "source": "hmm"}
```

**Example of BAD verification:**
```
Fixed: Updated the regime detector to use the correct data source.
The regime should now show correctly on the dashboard.
```
(No evidence. No proof. This tells us nothing.)

---

## KNOWN SYSTEM STATE

### Database: data/trading.db
Key tables:
- `five_minute_bars` — OHLCV bars (pair, exchange, bar_start, bar_end, open, high, low, close, volume, ...)
- `decisions` — Every cycle's decision (product_id, action, confidence, signal, hmm_regime, ...)
- `positions` — Open and closed positions
- `devil_tracker` — Cost tracking (may be empty)
- `signal_daily_pnl` — Daily P&L by signal

### Known data issues:
- five_minute_bars: Coinbase pairs have very few bars (BTC: 9, ETH: 11, SOL: 3)
  Binance arb pairs have 60 each. Need to seed historical data.
- decisions.hmm_regime: ALL rows are "unknown" (2,783+ decisions)
- decisions.vae_loss: ALL rows are NULL (not being persisted)
- devil_tracker: 0 rows (not wired up)

### HMM Model:
- File: models/hmm_regime.pkl
- Contains: Trained 3-state GaussianHMM (dict with keys: model, feature_means,
  feature_stds, state_to_regime, n_states, last_train_time)
- Problem: Used by MedallionRegimeDetector (observation loop), NOT by the main
  decision path which uses AdvancedRegimeDetector
- The AdvancedRegimeDetector needs min_samples=200, gets fed ticker snapshots
  instead of OHLCV bars

### Architecture Pain Points:
- RegimeOverlay.update() feeds ticker snapshots to AdvancedRegimeDetector
  which expects OHLCV bars — this is the regime bug
- BarAggregator IS working (bars accumulate correctly in DB)
- The main decision loop runs ~every 10 seconds
- Paper trading: slippage is simulated at 0%, which masks execution issues

---

## CODING STANDARDS

### Python
- Use type hints on all function signatures
- Docstrings on all classes and public methods
- Log important state changes: `logger.info(f"Regime changed: {old} → {new}")`
- Handle exceptions explicitly — no bare `except:` clauses
- Use Decimal for all financial calculations (prices, sizes, P&L)
- Async where the existing codebase uses async (main loop is asyncio-based)

### Database
- Always use parameterized queries (no f-string SQL)
- Add indexes for columns used in WHERE clauses on large tables
- Back up before schema changes: `cp data/trading.db data/trading.db.bak`

### Configuration
- All magic numbers go in config.yaml, not hardcoded
- New config sections should have sensible defaults
- Document new config keys with inline comments

### Git
- Commit after each logical unit of work (not after each file change)
- Commit messages: `fix(regime): wire AdvancedRegimeDetector to five_minute_bars table`
- Format: `type(scope): description` where type is fix/feat/refactor/docs/test

---

## DIAGNOSTIC COMMANDS (use these liberally)

```bash
# Check regime status
curl -s localhost:8080/api/brain/regime 2>&1 | python3 -m json.tool

# Check confluence
curl -s localhost:8080/api/brain/confluence 2>&1 | python3 -m json.tool

# Check VAE
curl -s localhost:8080/api/brain/vae 2>&1 | python3 -m json.tool

# Check bar counts per pair
.venv/bin/python3 -c "
import sqlite3
conn = sqlite3.connect('data/trading.db')
rows = conn.execute('SELECT pair, COUNT(*) as cnt FROM five_minute_bars GROUP BY pair ORDER BY cnt DESC').fetchall()
for r in rows: print(f'{r[0]}: {r[1]} bars')
"

# Check decision regime distribution
.venv/bin/python3 -c "
import sqlite3
conn = sqlite3.connect('data/trading.db')
rows = conn.execute('SELECT hmm_regime, COUNT(*) FROM decisions GROUP BY hmm_regime ORDER BY COUNT(*) DESC').fetchall()
for r in rows: print(f'{r[0]}: {r[1]} decisions')
"

# Check open positions
.venv/bin/python3 -c "
import sqlite3
conn = sqlite3.connect('data/trading.db')
rows = conn.execute('SELECT * FROM positions WHERE status=\"open\" LIMIT 10').fetchall()
print(f'Open positions: {len(rows)}')
for r in rows: print(r)
"

# Check all tables and row counts
.venv/bin/python3 -c "
import sqlite3
conn = sqlite3.connect('data/trading.db')
tables = conn.execute('SELECT name FROM sqlite_master WHERE type=\"table\"').fetchall()
for t in tables:
    cnt = conn.execute(f'SELECT COUNT(*) FROM {t[0]}').fetchone()[0]
    print(f'{t[0]}: {cnt} rows')
"

# Check recent logs for errors
grep -i "error\|exception\|traceback" logs/*.log | tail -30

# Check bot process
ps aux | grep -i "python.*main" | grep -v grep
```

---

## WORKFLOW FOR MULTI-HOUR AUTONOMOUS SESSIONS

When starting a long session, follow this protocol:

### Phase 1: Assess (5 minutes)
1. Read this CLAUDE.md (you're doing it now)
2. Run diagnostics above to understand current system state
3. Check which priority items from the list above are already fixed
4. Identify the highest-priority unfixed item

### Phase 2: Fix (iterate until done)
For each priority item:
1. Read the relevant spec document section
2. Find the code that needs to change (use grep, find, read)
3. Understand the current code before modifying it
4. Make the fix
5. Verify with evidence (database query, API call, log check)
6. Git commit with descriptive message
7. Move to the next priority item

### Phase 3: Validate (every 30 minutes)
Periodically:
1. Check the dashboard is still running (curl localhost:8080)
2. Check the bot is still running (ps aux | grep python)
3. Run the diagnostic commands to confirm fixes are holding
4. If something regressed, fix it before continuing forward

### Phase 4: Report (when session ends or hits a blocker)
Summarize:
- What was fixed (with evidence)
- What's still broken (with diagnosis)
- What to do next
- Any blockers that need human input

---

## SUBAGENT INSTRUCTIONS

If you're running as a subagent or spawning subagents, these rules apply:

### For subagents you spawn:
- Give each subagent ONE clear task, not multiple
- Include the relevant spec document reference
- Include the diagnostic commands they'll need
- Tell them to verify their fix with evidence
- Tell them to commit their changes

### If YOU are a subagent:
- Focus only on your assigned task
- Don't modify files outside your scope
- Verify your fix before reporting back
- Report: what you changed, evidence it works, any issues found

### Scope boundaries for parallel work:
- **Regime work**: RegimeOverlay, AdvancedRegimeDetector, MedallionRegimeDetector,
  RegimePredictor, HMM model loading, bootstrap rules, bar seeding
- **Position work**: PortfolioEngine, position opening/closing logic, netting,
  exposure calculation, equity tracking
- **Dashboard work**: dashboard/ directory, API endpoints, WebSocket relay,
  frontend rendering
- **Risk work**: RiskAlertEngine, risk gateway, VAE anomaly detection,
  drawdown calculation
- **ML work**: ML ensemble models, signal weights, confluence engine,
  prediction history

These scopes are designed to minimize file conflicts between parallel agents.

---

## EXCHANGE API REFERENCE (for seeding historical data)

### MEXC REST API — Historical Klines
```
GET https://api.mexc.com/api/v3/klines
Parameters:
  symbol: BTCUSDT (no slash)
  interval: 5m
  limit: 200 (max 1000)
  
Example:
  curl "https://api.mexc.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=200"
  
Response: [[open_time, open, high, low, close, volume, close_time, ...], ...]
```

### Binance REST API — Historical Klines
```
GET https://api.binance.com/api/v3/klines
Parameters:
  symbol: BTCUSDT
  interval: 5m
  limit: 200 (max 1000)

Example:
  curl "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=200"

Response: [[open_time, open, high, low, close, volume, close_time, ...], ...]
```

Use these to seed the five_minute_bars table with 200 historical candles per pair
so the HMM can activate immediately instead of waiting 17 hours.

---

## COMMON PITFALLS

1. **Don't trust "I already fixed this"** — Verify with evidence every time.
2. **The bot may need restarting** after code changes to main.py or core modules.
   Use: `pkill -f "python.*main" && sleep 2 && .venv/bin/python3 main.py &`
3. **SQLite locks** — If the bot is running, DB writes from diagnostics may fail.
   Use read-only connections for diagnostics: `sqlite3.connect('file:data/trading.db?mode=ro', uri=True)`
4. **Paper vs live** — We are PAPER TRADING. Do not change this. The `PAPER TRADING`
   badge should always be visible on the dashboard.
5. **Config reloads** — Some config changes require bot restart, others are hot-reloaded.
   When in doubt, restart.
6. **The five_minute_bars table uses 'pair' not 'product_id'** — Column names matter.
   Always check schema before writing queries.

---

## SUCCESS CRITERIA

The system is working correctly when:
- [ ] Regime detector classifies a non-"Unknown" regime with >50% confidence
- [ ] No opposing positions on the same asset
- [ ] Exposure calculation shows actual dollar amounts, not $0
- [ ] Equity curve tracks peak equity and drawdown correctly
- [ ] Sharpe ratio computes (not blank)
- [ ] P&L is split into realized and unrealized everywhere
- [ ] Risk alerts fire when thresholds are breached
- [ ] Risk gateway log shows entries (both PASS and REJECT)
- [ ] Win rate includes context (closed trades vs open positions)
- [ ] Signal confidence is not zeroed out by Low_volatility regime
- [ ] The system generates at least some BUY/SELL signals per hour (not just HOLD)
- [ ] The bot has been running for 1+ hours without crashing

When ALL checkboxes are met, move to the MEDIUM and FUTURE priority items.

---

*Built to run autonomously. Verify everything. Trust nothing without evidence.*
