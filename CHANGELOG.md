# Changelog

All notable changes to the Renaissance Trading Bot, documented across 46 commits.

## Metrics

| Metric | Before | After |
|--------|--------|-------|
| `renaissance_trading_bot.py` | 7,681 lines (god class) | 2,149 lines + 9 modules in `bot/` |
| `bot/` package | did not exist | 6,513 lines across 9 modules |
| Dead code removed | — | ~10,460 lines (consciousness + 12 files) |
| Test count | 0 | 87 |
| ML model accuracy | 47-49% (worse than random) | 6 root causes fixed |
| Crash rate | ~13 crashes/day | 0 |
| Infrastructure | none | Docker, CI, Makefile, ruff |

---

## Structural Refactoring

### God Class Decomposition (8 commits)
The monolithic `RenaissanceTradingBot` class was broken into focused subsystems:

- `93ec52c` — Extract `__init__` body to `bot/builder.py` (BotBuilder)
- `394ac37` — Extract signals and decision logic from god class
- `d8c319b` — Extract data collection methods to `bot/data_collection.py`
- `9f5b0e5` — Extract position management methods to `bot/position_ops.py`
- `b1576a6` — Extract lifecycle + adaptive methods from god class
- `ebbdb49` — Extract helpers + delete unused stubs (3,545 to 3,254 lines)
- `4b3f928` — Extract cycle_ops + helpers + MacroDataCache (3,254 to 2,149 lines)

### Dead Code Removal
- `a98b1f3` — Remove dead consciousness_boost code (~2,200 lines)
- `eafac1a` — Remove 12 dead code files (8,260 lines)
- `f565cb7` — Untrack binaries, notebooks, and reports from git

### Code Quality
- `b1ccd16` — Add missing return type hints to core files
- `6cd8276` — Replace silent `except/pass` with logged warnings
- `073850e` — Document canonical regime detection hierarchy
- `130f7a6` — Clarify canonical vs legacy alert manager

---

## Bug Fixes

### Critical Dashboard & Trading Bugs
- `02c691b` — Resolve low_volatility regime blocking and dashboard data gaps
  - Regime detector now classifies correctly (bootstrap + HMM)
  - Low volatility boosts mean reversion instead of zeroing signals
  - Exposure calculation fixed (was showing $0)
  - Equity tracking fixed (peak equity, drawdown, Sharpe)
  - Position netting added (no more opposing positions on same asset)

### High Priority Fixes
- `a0f4f34` — P&L split, VAE persistence, confluence cache, alerts, gateway log
  - Realized vs unrealized P&L separated everywhere
  - VAE loss now persisted to decisions table
  - Confluence WebSocket relay cleared on startup
  - Risk alerts now fire when thresholds breached
  - Risk gateway log now shows PASS/REJECT entries

### Import & Module Fixes
- `05c4e3c` — Resolve circular import in `ml_enhanced_signal_fusion.py`
- `d84b4c5` — Rename `market_mictostructure_analyzer.py` to fix typo
- `49b7022` — Fix module-level singleton references in `renaissance_signal_fusion.py`

### Trading Logic Fixes
- `a2286dd` — Prevent zero-confidence trades from bypassing min_confidence gate
- `bacf079` — Add staleness check to discard ML predictions older than 15 minutes
- `89ed4f3` — Normalize signal weights to sum to exactly 1.0
- `d2c2898` — Resolve market price from Binance before computing realized P&L on forced closes
- `e18679d` — Correct misleading max_trade_pct comment (20% to 40%)

### Stability
- `b872a64` — Resolve 4 crash loop causes (13 crashes/day reduced to 0)

---

## ML Fixes

Six root causes of sub-50% directional accuracy identified and fixed (see `docs/ML_ACCURACY_INVESTIGATION.md`):

- `602fc81` — **Finding 1**: LightGBM training/inference feature mismatch (104 vs 294 features)
- `5740ae7` — **Finding 2**: Disable prediction debiaser/normalizer that inverts predictions
- `0d72efe` — **Finding 3**: Add std floor to prevent noise amplification in quiet markets
- `f884d77` — **Finding 4**: Add feature zero-rate logging for prediction reliability
- `4ad29f1` — **Finding 5**: Use v7 DirectionalLoss defaults in all training scripts
- `0a86ebc` — **Finding 6**: Add feature staleness logging to detect stale predictions

### Investigation
- `1297afa` — Document sub-50% directional accuracy across all 7 ML models

---

## Polymarket

- `0e34bb2` — Use `window_start_price` instead of `entry_asset_price` for lifecycle result
- `6ddc9f3` — Standardize timestamps to UTC ISO 8601 with timezone
- `47a1895` — Store NULL instead of $0 for unavailable `exit_asset_price`
- `b70b0c1` — Add periodic bankroll reconciliation from first principles
- `9daa8f1` — Add README for archived strategy files

---

## Dashboard

- `9074888` — Add AssetSummaryPanel to Command Center (Bug CC-4)
- `8f901f1` — Add success criteria card, activity feed, error boundary, win-rate context

---

## Arbitrage

- `57b9457` — Add `balance_snapshots` table and SpotRebalancer module
- `0699695` — Add `_maker_fill_timeout_s` and remove duplicate velocity tracking

---

## Tests

- `39f4cfc` — Add unit tests for position_sizer, straddle, token_spray, ml_model_loader
- `9b28c2e` — Add 90 unit tests for strategy_a and spread_capture (Polymarket)
- `f80cf70` — Add 120 unit tests for core trading path (builder, decision, signals, lifecycle)

**Total: 87 test files with 210+ individual tests.**

---

## Infrastructure

- `4088f63` — Add pyproject.toml, Dockerfile, docker-compose, CI, Makefile
- `b80b3e3` — Baseline: pre-fix snapshot

---

## Documentation

- `073850e` — Document canonical regime detection hierarchy
- `130f7a6` — Clarify canonical vs legacy alert manager
- `fb44e67` — Update EXPERIMENTAL.md to reflect current codebase
- `1297afa` — ML accuracy investigation report (docs/ML_ACCURACY_INVESTIGATION.md)
- `9daa8f1` — Polymarket archive README
