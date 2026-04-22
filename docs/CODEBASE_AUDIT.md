# Renaissance Trading Bot — Codebase Audit

> Generated: 2026-02-23
> Scope: Full project inventory — files, database, data flows, configuration, processes, dependencies, module status.
> This document is factual. No opinions, no recommendations.

---

## Table of Contents

1. [File Inventory](#1-file-inventory)
2. [Database Schema](#2-database-schema)
3. [Data Flow Maps](#3-data-flow-maps)
4. [Configuration Reference](#4-configuration-reference)
5. [Process Map](#5-process-map)
6. [Dependencies](#6-dependencies)
7. [Module Status](#7-module-status)

---

## 1. File Inventory

### Summary

| Location | Files | Lines | Classes | Public Functions |
|----------|------:|------:|--------:|-----------------:|
| Root `.py` files | 113 | 43,282 | 287 | — |
| `agents/` | 18 | 3,410 | 19 | 5 |
| `arbitrage/` | 17 | 7,080 | 41 | 1 |
| `backtesting/` | 1 | 1,211 | 4 | 1 |
| `core/` | 6 | 3,345 | 15 | 0 |
| `dashboard/` | 22 | 4,761 | 5 | 148 |
| `data_module/` | 1 | 389 | 3 | 0 |
| `execution/` | 2 | 518 | 3 | 0 |
| `intelligence/` | 9 | 3,354 | 13 | 0 |
| `monitoring/` | 5 | 2,107 | 5 | 0 |
| `orchestrator/` | 3 | 434 | 3 | 1 |
| `portfolio/` | 1 | 876 | 2 | 0 |
| `recovery/` | 6 | 2,090 | 8 | 2 |
| `scripts/` | 23 | 8,074 | 11 | 111 |
| `signals/` | 4 | 1,747 | 15 | 0 |
| `tests/` | 81 | 22,786 | 302 | 90 |
| **Total** | **312** | **105,464** | **736** | **359** |

Note: 29 additional `__init__.py` files exist (341 total `.py` files).

### Top 20 Files by Line Count

| # | File | Lines |
|---|------|------:|
| 1 | `renaissance_trading_bot.py` | 5,388 |
| 2 | `ml_model_loader.py` | 1,557 |
| 3 | `stress_test_engine.py` | 1,418 |
| 4 | `spread_optimizer.py` | 1,257 |
| 5 | `position_manager.py` | 1,217 |
| 6 | `coinbase_client.py` | 1,208 |
| 7 | `pattern_confidence.py` | 1,154 |
| 8 | `tail_risk_protector.py` | 1,128 |
| 9 | `polymarket_scanner.py` | 926 |
| 10 | `enhanced_technical_indicators.py` | 810 |
| 11 | `production_trading_orchestrator.py` | 749 |
| 12 | `polymarket_executor.py` | 709 |
| 13 | `liquidity_risk_manager.py` | 701 |
| 14 | `enhanced_decision_framework.py` | 666 |
| 15 | `position_sizer.py` | 662 |
| 16 | `microstructure_engine.py` | 637 |
| 17 | `ml_enhanced_signal_fusion.py` | 628 |
| 18 | `ml_integration_bridge.py` | 627 |
| 19 | `confidence_calculator.py` | 620 |
| 20 | `threshold_manager.py` | 603 |

### Key Files by Function

| Function | File | Primary Class |
|----------|------|---------------|
| Bot entry point | `run_renaissance_bot.py` | — |
| Main bot loop | `renaissance_trading_bot.py` | `RenaissanceTradingBot` |
| ML model loading + architectures | `ml_model_loader.py` | `TrainedQuantumTransformer`, `TrainedBiLSTM`, `TrainedCNN`, `TrainedDilatedCNN`, `TrainedGRU`, `TrainedMetaEnsemble` |
| ML bridge (inference orchestration) | `ml_integration_bridge.py` | `MLIntegrationBridge` |
| Real-time ML pipeline | `real_time_pipeline.py` | `RealTimePipeline`, `MultiExchangeFeed` |
| Signal fusion | `renaissance_signal_fusion.py` | `RenaissanceSignalFusion` |
| Position management | `position_manager.py` | `EnhancedPositionManager` |
| Position sizing (Kelly) | `position_sizer.py` | `RenaissancePositionSizer` |
| Regime detection | `regime_overlay.py` | `RegimeOverlay`, `BootstrapRegimeClassifier` |
| HMM regime | `advanced_regime_detector.py` | `AdvancedRegimeDetector` |
| Risk gateway (VAE) | `risk_gateway.py` | `RiskGateway` |
| Confluence engine | `confluence_engine.py` | `ConfluenceEngine` |
| Database manager | `database_manager.py` | `DatabaseManager` |
| Binance data provider | `binance_spot_provider.py` | `BinanceSpotProvider` |
| Breakout scanner | `breakout_scanner.py` | `BreakoutScanner` |
| Technical indicators | `enhanced_technical_indicators.py` | `EnhancedTechnicalIndicators` |
| Polymarket scanner | `polymarket_scanner.py` | `PolymarketScanner` |
| Polymarket executor | `polymarket_executor.py` | `PolymarketExecutor` |
| Polymarket bridge | `polymarket_bridge.py` | `PolymarketBridge` |
| Arbitrage orchestrator | `arbitrage/orchestrator.py` | `ArbitrageOrchestrator` |
| Triangular arbitrage | `arbitrage/triangular/triangle_arb.py` | `TriangularArbitrage` |
| Cross-exchange detector | `arbitrage/detector/cross_exchange.py` | `CrossExchangeDetector` |
| Dashboard server | `dashboard/server.py` | `create_app()` |
| Agent coordinator | `agents/coordinator.py` | `AgentCoordinator` |
| Bar aggregator | `data_module/bar_aggregator.py` | `BarAggregator` |

---

## 2. Database Schema

### Database Files

| File | Size | Tables | Description |
|------|-----:|-------:|-------------|
| `data/renaissance_bot.db` | 21 MB | 34 | Main bot database |
| `data/arbitrage.db` | 1.1 MB | 6 | Arbitrage-specific data |
| `data/recovery_state.db` | 40 KB | 3 | Recovery/watchdog state |
| `data/trading.db` | 0 B | — | Legacy (unused) |
| `data/trading_bot.db` | 0 B | — | Legacy (unused) |

### renaissance_bot.db — 34 Tables

| Table | Rows | Purpose | Key Columns |
|-------|-----:|---------|-------------|
| `agent_events` | 55 | Agent coordination events | agent_name, event_type, channel, payload, severity |
| `backtest_runs` | 0 | Backtest job results | config_json, total_trades, sharpe_ratio, win_rate |
| `balance_snapshots` | 0 | Exchange balance history | exchange, currency, free, locked, total, usd_value |
| `cascade_signals` | 0 | Liquidation cascade signals | symbol, direction, risk_score, funding_rate |
| `cost_observations` | 0 | Devil tracker cost data | symbol, exchange, estimated_*_bps, realized_*_bps |
| `daily_candles` | 0 | Daily OHLCV candles | product_id, date, OHLCV |
| `daily_performance` | 0 | Daily P&L summary | date, trades, profit, fees, drawdown, equity |
| `data_refresh_log` | 0 | Historical data fetch log | product_id, last_refresh, rows_fetched |
| `decisions` | 6,276 | Every trading cycle decision | timestamp, product_id, action, confidence, weighted_signal, hmm_regime |
| `devil_tracker` | 514 | Signal-to-fill cost tracking | trade_id, pair, signal_price, fill_price, devil, slippage_bps, latency_ms |
| `exchange_health` | 0 | Exchange status monitoring | exchange, api_latency_ms, ws_connected, status |
| `five_minute_bars` | 6,208 | 5-min OHLCV bars | pair, exchange, bar_start, OHLCV, volume, vwap, log_return |
| `funding_rate_history` | 0 | Funding rate snapshots | symbol, funding_rate, exchange |
| `ghost_trades` | 0 | Shadow trading validation | param_set, product_id, action, entry/exit_price, pnl_pct |
| `improvement_log` | 0 | Agent improvement tracking | proposal_id, change_type, metric_before/after |
| `labels` | 5,267 | Decision outcome labels | decision_id, product_id, entry/exit_price, ret_pct, correct |
| `liquidation_events` | 0 | Liquidation cascade events | symbol, direction, risk_score |
| `market_data` | 6,358 | Raw market data snapshots | price, volume, bid, ask, spread, source, product_id |
| `ml_predictions` | 43,498 | ML model predictions | product_id, model_name, prediction, confidence |
| `model_ledger` | 13 | Model version registry | model_name, model_version, accuracy, file_path, status |
| `onchain_data` | 0 | On-chain metrics | active_addresses, hash_rate, network_health |
| `open_interest_history` | 0 | Futures open interest | symbol, open_interest, change_24h_pct |
| `open_positions` | 514 | Positions (open + closed) | position_id, product_id, side, size, entry_price, status, realized_pnl |
| `polymarket_bankroll_log` | — | Polymarket bankroll changes | event, amount, bankroll balance |
| `polymarket_positions` | — | Polymarket bet positions | condition_id, slug, market_type, direction, entry_price, shares, pnl |
| `polymarket_scanner` | 693 | Polymarket market scans | condition_id, question, market_type, asset, yes/no_price, edge |
| `proposals` | 0 | Agent-generated proposals | category, title, description, status, backtest metrics |
| `reeval_events` | 450 | Position re-evaluation events | position_id, event_type, reason_code, pnl_bps |
| `sentiment_data` | 0 | Social sentiment data | twitter/reddit_sentiment, fear_greed_index |
| `signal_daily_pnl` | 1 | Daily P&L by signal type | signal_type, date, pnl, num_trades, win_rate |
| `signal_throttle_log` | 0 | Signal kill/enable events | signal_name, accuracy, sample_count |
| `signals` | 0 | Arbitrage signals (recovery) | signal_type, symbol, spread_bps, confidence |
| `system_state_log` | 623 | System state transitions | state, previous_state, reason, metadata (JSON) |
| `trades` | 514 | Executed trades | product_id, side, size, price, status, algo_used, slippage |
| `weekly_reports` | 4 | Weekly observation reports | week_start/end, report_json, sharpe_7d, total_pnl |

### Indexes (32 total)

Key manually-created indexes:
- `idx_bars_lookup` on `five_minute_bars(pair, exchange, bar_start DESC)`
- `idx_devil_pair_ts` on `devil_tracker(pair, signal_timestamp)`
- `idx_scanner_edge` on `polymarket_scanner(edge)`
- `idx_scanner_type` on `polymarket_scanner(market_type)`
- `idx_state_log_ts` on `system_state_log(timestamp)`
- `idx_model_ledger_name` on `model_ledger(model_name)`
- `idx_agent_events_agent` on `agent_events(agent_name)`

No views. No triggers.

### arbitrage.db — 6 Tables

| Table | Purpose |
|-------|---------|
| `arb_trades` | All arbitrage trade attempts with status and profit |
| `arb_signals` | Detected arbitrage signals with approval status |
| `arb_daily_summary` | Daily aggregate P&L |
| `path_performance` | Per-path fill rate tracking (triangular) |
| `funding_positions` | Open/closed funding arb positions |
| `funding_rate_history` | Historical funding rates |

---

## 3. Data Flow Maps

### Flow 1: ML Directional Trading (Main Loop)

```
Entry: run_renaissance_bot.py → RenaissanceTradingBot.run_continuous_trading()
       → execute_trading_cycle() every 60-300s
```

**Step 1 — Data Fetching** (`collect_all_data()`, line 1496)
- Primary: `BinanceSpotProvider` — ticker, candles (5m×2), orderbook (depth 20)
- Legacy: Coinbase WebSocket queue → REST fallback
- Parallel: `asyncio.gather` with `Semaphore(15)`, ~5-15s for all pairs
- Pre-fetch: 200 candles per pair on first cycle for indicator warmup

**Step 2 — Feature Engineering** (14 parallel layers)
- (a) Technical indicators → RSI, MACD, Bollinger, OBV (`enhanced_technical_indicators.py`)
- (b) Microstructure → order book imbalance, large trade flow (`microstructure_engine.py`)
- (c) Alternative data → Reddit, news, sentiment, Fear & Greed (`alternative_data_engine.py`)
- (d) Volume profile → volume-at-price, POC proximity (`volume_profile_engine.py`)
- (e) Fractal intelligence → DTW pattern matching (`fractal_intelligence.py`)
- (f) Market entropy → Shannon entropy, ApEn (`market_entropy_engine.py`)
- (g) GARCH volatility → forecast vol, vol ratio (`garch_volatility_engine.py`)
- (h) Cross-asset correlation → lead-lag relationships (`cross_asset_engine.py`)
- (i) Mean reversion → Z-score pairs trading (`advanced_mean_reversion_engine.py`)
- (j) Breakout signal → 5-component score (`breakout_scanner.py`)
- (k) Liquidation cascade → funding rate + OI analysis (`signals/liquidation_detector.py`)
- (l) Fast mean reversion → 1-second tick scanner (`intelligence/fast_mean_reversion.py`)
- (m) Medallion analogs → sharp move reversion, seasonality (`medallion_signal_analogs.py`)
- (n) Multi-exchange bridge → cross-exchange momentum (`signals/multi_exchange_bridge.py`)

**Step 3 — ML Inference** (7 models, `ml_integration_bridge.py`)
- Models: QuantumTransformer, BiLSTM, DilatedCNN, CNN, GRU, LightGBM, MetaEnsemble
- Input: 98-dimensional feature vector (46 active), per-window standardized
- Output: `Dict[str, float]` — model name → prediction (~0.01 to 0.08 raw)
- Signal amplification: `ml_signal_scale = 10x` before fusion

**Step 4 — Signal Fusion** (`calculate_weighted_signal()`, line 1798)
- Delegate: `RenaissanceSignalFusion.fuse_signals_with_ml()`
- 15-25 signals weighted by `config/config.json` signal_weights
- Confluence engine boosts by up to 50% when 3+ signals agree

**Step 5 — Regime Weighting** (`regime_overlay.py`)
- HMM-based classification: 8 regime types
- Effects: weight adjustment, confidence boost, entry threshold adjustment, transition sizing

**Step 6 — Risk Gate** (`make_trading_decision()`, line 1863, 15 sequential checks)
- Cost pre-screen, confidence gate, signal threshold, ML agreement (71%)
- Anti-churn (12 cycles), signal reversal netting, anti-stacking
- Daily/weekly loss limits, VAE anomaly detection, drawdown circuit breaker
- Flash crash detection (>5% move), spread sanity (>50bps), data freshness (<60s)

**Step 7 — Kelly Sizing** (line 2163-2435)
- Primary: `RenaissancePositionSizer.calculate_size()` — Kelly formula: f* = edge / variance
- Multiplier chain: regime transition × portfolio correlation × health monitor × signal tier
- Secondary: `KellyPositionSizer.get_position_size()` — cap at minimum
- Final: base_usd = balance × 3%, floor $100, ceiling 9.9% of equity

**Step 8 — Execution** (`_execute_smart_order()`, line 2478)
- MEXC: LIMIT_MAKER orders (0% maker fee)
- Coinbase: MARKET orders (legacy pairs)
- Devil Tracker records signal price vs fill price

**Step 9 — P&L Recording** (SQLite)
- Tables written: trades, positions, decisions, market_data, ml_predictions, five_minute_bars, signal_daily_pnl
- Dashboard events emitted via `DashboardEventEmitter`

### Flow 2: Arbitrage Engine

```
Entry: ArbitrageOrchestrator (launched as background task from main bot)
       Three concurrent strategy loops via asyncio.create_task()
```

1. **Order Book Fetching** — `UnifiedBookManager` maintains dual-exchange books (MEXC + Binance) for 30 pairs
2. **Spread Detection** — Cross-exchange (`CrossExchangeDetector`), Triangular (`TriangularArbitrage`), Funding Rate (`FundingRateArbitrage`)
3. **Cost Model** — `ArbitrageCostModel.estimate_arbitrage_cost()` — MEXC maker fee: 0%
4. **Signal Generation** — `ArbitrageSignal` dataclass with 5-second TTL, min 1 bps net spread
5. **Risk Check** — `ArbitrageRiskEngine.approve_arbitrage()` — rate limit 100 trades/hr
6. **Execution** — `ArbitrageExecutor.execute_arbitrage()` — simultaneous buy/sell
7. **P&L Tracking** — `PerformanceTracker` → `data/arbitrage.db`

### Flow 3: Polymarket Scanner + Executor

```
Entry: Every 5 minutes during BTC-USD processing cycle
       (renaissance_trading_bot.py line 3757-3812)
```

1. **Market Discovery** — Gamma API paginated fetch (up to 5000 markets, 5-min cache)
2. **Classification** — Regex-based: DIRECTION, THRESHOLD, HIT_PRICE, RANGE, VOLATILITY, OTHER (~172 crypto markets from 3000+)
3. **Edge Computation** — DIRECTION: ML prediction → calibrated probability; HIT_PRICE: GBM barrier-touch model (reflection principle)
4. **Opportunity Ranking** — Sort by edge descending, persist to `polymarket_scanner` table
5. **Risk Filtering** — Time filters (HIT_PRICE: min 7 days, DIRECTION: 60-300s), max 10 positions, per-asset/total exposure limits
6. **Kelly Sizing** — Binary Kelly: f* = (p·b − q)/b, half-Kelly, max $100 per bet
7. **Position Tracking** — `polymarket_positions` + `polymarket_bankroll_log` tables
8. **Resolution Checking** — Every 12 cycles, slug-based Gamma API lookup, definitive when one outcome ≥ 0.95

### Flow 4: Breakout Scanner

```
Entry: execute_trading_cycle() Phase 0 (line 2963), every cycle
```

1. **Data Fetch** — Single Binance API call: GET `/ticker/24hr` (all 600+ pairs)
2. **Scoring** — 5 components (0-100 total): volume (0-30), price breakout (0-25), momentum (0-25), volatility (0-10), divergence (0-10)
3. **Output** — Min score 25, max 30 flagged pairs; score ≥ 40 injected as signal; score ≥ 50 triggers instant warmup

### Flow 5: Dashboard

```
Entry: FastAPI + uvicorn on port 8080
       Launched as daemon thread from main bot
```

- 75 REST endpoints across 13 routers
- 1 WebSocket endpoint (`/ws`) with heartbeat every 30s
- Static SPA served from `dashboard/frontend/dist/`
- Reads from `renaissance_bot.db` and `arbitrage.db` (read-only connections)

### Data Store Summary

| Store | Producers | Consumers |
|-------|-----------|-----------|
| `renaissance_bot.db` | | |
| — decisions | main bot cycle | dashboard, analytics, agents |
| — positions | position_manager | dashboard, anti-stacking |
| — trades | _execute_smart_order | dashboard, attribution |
| — market_data | main bot cycle | dashboard, ML feedback |
| — five_minute_bars | BarAggregator | regime detector, ML models |
| — ml_predictions | ml_bridge | dashboard brain tab |
| — breakout_scans | main bot cycle | dashboard breakout tab |
| — polymarket_scanner | PolymarketScanner | executor, dashboard |
| — polymarket_positions | PolymarketExecutor | dashboard |
| `arbitrage.db` | | |
| — arb_trades | ArbitrageExecutor | dashboard arb tab |
| — arb_signals | CrossExchangeDetector | dashboard arb tab |
| — path_performance | TriangularArbitrage | dashboard arb tab |
| `renaissance-signal.json` | PolymarketBridge | Node.js revenue-engine |
| `logs/heartbeat.json` | main bot cycle | external monitoring |

---

## 4. Configuration Reference

### config/config.json (Main Bot — 160+ keys)

#### Trading

| Key | Value | Type |
|-----|-------|------|
| `trading.product_ids` | BTC-USD, ETH-USD, SOL-USD, DOGE-USD, XRP-USD, ADA-USD, AVAX-USD, LINK-USD, DOT-USD, MATIC-USD, UNI-USD, ATOM-USD | array |
| `trading.cycle_interval_seconds` | 300 | int |
| `trading.paper_trading` | true | bool |
| `trading.buy_threshold` | 0.06 | float |
| `trading.sell_threshold` | -0.06 | float |
| `trading.execution_exchange` | "mexc" | string |

#### Universe (Expanded)

| Key | Value | Type |
|-----|-------|------|
| `universe.min_volume_usd` | 2,000,000 | int |
| `universe.max_pairs` | 150 | int |
| `universe.data_source` | "binance" | string |
| `universe.refresh_interval_hours` | 24 | int |

#### Signal Weights

| Signal | Weight |
|--------|-------:|
| `entropy` | 0.164 |
| `rsi` | 0.125 |
| `bollinger` | 0.104 |
| `lead_lag` | 0.081 |
| `volume` | 0.079 |
| `stat_arb` | 0.058 |
| `garch_vol` | 0.053 |
| `fractal` | 0.051 |
| `ml_ensemble` | 0.050 |
| `volume_profile` | 0.041 |
| `order_flow` | 0.040 |
| `multi_exchange` | 0.033 |
| `ml_cnn` | 0.030 |
| `order_book` | 0.027 |
| `macd` | 0.021 |
| `quantum` | 0.020 |
| `alternative` | 0.010 |
| `correlation_divergence` | 0.008 |

#### Risk Management

| Key | Value | Type |
|-----|-------|------|
| `risk_management.daily_loss_limit` | 500 | int |
| `risk_management.position_limit` | 5000 | int |
| `risk_management.min_confidence` | 0.25 | float |

#### Regime Overlay

| Key | Value | Type |
|-----|-------|------|
| `regime_overlay.enabled` | true | bool |
| `regime_overlay.hmm_regimes` | 5 | int |
| `regime_overlay.hmm_min_samples` | 200 | int |
| `regime_overlay.hmm_covariance_type` | "full" | string |

#### Risk Gateway

| Key | Value | Type |
|-----|-------|------|
| `risk_gateway.enabled` | true | bool |
| `risk_gateway.max_portfolio_value` | 1000.0 | float |
| `risk_gateway.anomaly_threshold` | 5.0 | float |

#### ML

| Key | Value | Type |
|-----|-------|------|
| `real_time_pipeline.enabled` | true | bool |
| `ml_signal_scale` | 10.0 | float |
| `model_retraining.schedule_day` | "saturday" | string |
| `model_retraining.epochs` | 50 | int |
| `model_retraining.rolling_days` | 180 | int |

#### Polymarket

| Key | Value | Type |
|-----|-------|------|
| `polymarket.executor_enabled` | true | bool |
| `polymarket.paper_mode` | true | bool |
| `polymarket.initial_bankroll` | 500.0 | float |
| `polymarket.max_positions` | 10 | int |

#### Dashboard

| Key | Value | Type |
|-----|-------|------|
| `dashboard_config.host` | "0.0.0.0" | string |
| `dashboard_config.port` | 8080 | int |
| `dashboard_config.refresh_interval_ms` | 2000 | int |
| `dashboard_config.alerts.pnl_threshold` | -200 | int |
| `dashboard_config.alerts.drawdown_threshold` | 0.05 | float |

#### Database

| Key | Value | Type |
|-----|-------|------|
| `database.path` | "data/renaissance_bot.db" | string |

### arbitrage/config/arbitrage.yaml (Arbitrage Module)

| Key | Value |
|-----|-------|
| `exchanges.primary` | mexc |
| `exchanges.secondary` | binance |
| `pairs.phase_1` | 10 large-cap (BTC, ETH, SOL, XRP, BNB, DOGE, ADA, AVAX, LINK, DOT) |
| `pairs.phase_2` | 20 mid-cap (VANRY, NTRN, G, DUSK, ZKP, COTI, POL, SONIC, NEAR, ALGO, ATOM, FIL, IMX, ARB, OP, APE, SAND, MANA, GALA, ENJ) |
| `cross_exchange.min_net_spread_bps` | 1.0 |
| `cross_exchange.min_trade_usd` | 10 |
| `cross_exchange.max_trade_usd` | 2000 |
| `triangular.min_net_profit_bps` | 5.0 |
| `triangular.exchange` | mexc |
| `triangular.min_trade_usd` | 50 |
| `triangular.max_trade_usd` | 3000 |
| `triangular.dynamic_sizing_enabled` | true |
| `triangular.adaptive_threshold_enabled` | true |
| `paper_trading.enabled` | true |
| `paper_trading.simulated_fill_rate` | 0.75 |
| `risk.max_trades_per_hour` | 2000 |
| `risk.max_daily_loss_usd` | 100 |

### Other Config Files

| File | Purpose |
|------|---------|
| `config/config.example.json` | Minimal template with defaults |
| `config/data_pipeline_config.json` | Data pipeline settings (Coinbase, Twitter, Fear & Greed) |
| `logs/heartbeat.json` | Runtime heartbeat (alive, cycle_count, paper_mode) |
| `data/heartbeats/bot-01.json` | Orchestrator heartbeat (bot_id, equity, regime) |

---

## 5. Process Map

### Background Tasks in Main Loop

The main bot (`renaissance_trading_bot.py`) spawns 51 background tasks via `_track_task()`:

#### Core Trading Loop
| Task | Line | Description |
|------|------|-------------|
| `execute_trading_cycle()` | 4969 | Main cycle — data fetch → ML → signal → trade |
| `_run_websocket_feed()` | 5076 | WebSocket data stream (Coinbase) |
| `_run_arbitrage_engine()` | 5081 | Arbitrage orchestrator |

#### Monitoring Loops
| Task | Line | Description |
|------|------|-------------|
| `_run_heartbeat_writer()` | 5099 | Write heartbeat.json every N seconds |
| `_run_portfolio_drift_logger()` | 5104 | Log portfolio drift events |
| `_run_insurance_scanner_loop()` | 5108 | Insurance premium scanning |
| `_run_daily_signal_review_loop()` | 5112 | Daily signal performance review |
| `_run_beta_monitor_loop()` | 5117 | Market beta monitoring |
| `_run_sharpe_monitor_loop()` | 5121 | Sharpe ratio monitoring |
| `_run_capacity_monitor_loop()` | 5125 | Capacity constraint monitoring |
| `_run_regime_detector_loop()` | 5129 | Regime detection (HMM) |
| `_run_telegram_report_loop()` | 5141 | Telegram reporting |

#### Intelligence Loops
| Task | Line | Description |
|------|------|-------------|
| `ghost_runner.start_ghost_loop()` | 5072 | Shadow trade validation |
| `_run_liquidation_detector()` | 5086 | Liquidation cascade detection |
| `_run_fast_reversion_scanner()` | 5091 | 1-second mean reversion scanner |

#### Agent Coordination
| Task | Line | Description |
|------|------|-------------|
| `agent_coordinator.run_weekly_check_loop()` | 5134 | Weekly agent research cycle |
| `agent_coordinator.run_deployment_loop()` | 5136 | Agent deployment monitoring |

#### Per-Cycle Tasks (fired each cycle)
| Task | Description |
|------|-------------|
| `db_manager.store_decision()` | Persist decision to DB |
| `db_manager.store_trade()` | Persist trade to DB |
| `db_manager.store_market_data()` | Persist market data to DB |
| `db_manager.store_ml_prediction()` | Persist ML predictions to DB |
| `dashboard_emitter.emit("cycle", ...)` | Push cycle data to WebSocket |
| `dashboard_emitter.emit("regime", ...)` | Push regime update |
| `dashboard_emitter.emit("confluence", ...)` | Push confluence data |
| `dashboard_emitter.emit("risk.alert", ...)` | Push risk alerts |
| `_run_adaptive_learning_cycle()` | Bayesian weight updating |
| `_perform_attribution_analysis()` | Signal contribution analysis |

### Dashboard (Thread)

```
threading.Thread(target=_run_dashboard, daemon=True)  (line 1029)
  → uvicorn.run(create_app(), host, port)
  → 13 FastAPI routers (75 endpoints)
  → WebSocket relay (heartbeat every 30s)
  → Static SPA serving
```

### Dashboard API Endpoints (75 total)

| Prefix | Router File | Endpoints | Description |
|--------|------------|----------:|-------------|
| `/ws` | `server.py` | 1 | WebSocket real-time updates |
| `/api/health` | `server.py` | 1 | Health check |
| `/api/system` | `system.py` | 3 | Status, config, price history |
| `/api/brain` | `brain.py` | 5 | Ensemble, predictions, regime, confluence, VAE |
| `/api` (decisions) | `decisions.py` | 3 | Recent decisions, detail, signal weights |
| `/api/analytics` | `analytics.py` | 10 | Equity, P&L, by-regime, distribution, calendar, hourly, benchmark, attribution, model accuracy |
| `/api` (trades) | `trades.py` | 6 | Open/closed positions, summary, trade detail, lifecycle |
| `/api/risk` | `risk.py` | 6 | Exposure, metrics, gateway log, alerts, leverage |
| `/api/backtest` | `backtest.py` | 6 | Runs, results, compare, start, status, download |
| `/api/medallion` | `medallion.py` | 1 | Medallion module status |
| `/api/devil` | `devil.py` | 2 | Devil tracker summary, recent |
| `/api/polymarket` | `polymarket.py` | 9 | Summary, edges, markets, signal, history, stats, positions, P&L, executor |
| `/api/agents` | `agents.py` | 6 | Status, events, proposals, improvements, reports, models |
| `/api/arbitrage` | `arbitrage.py` | 12 | Status, trades, signals, summary, wallet, funding, depth, sizing, paths, fill rate, contracts |
| `/api/breakout` | `breakout.py` | 4 | Summary, signals, history, heatmap |

---

## 6. Dependencies

### requirements.txt (~110 packages, grouped by purpose)

#### ML / Deep Learning / Scientific Computing
| Package | Purpose |
|---------|---------|
| `torch` | PyTorch deep learning framework |
| `scikit-learn` | ML algorithms, preprocessing |
| `hmmlearn>=0.3.0` | Hidden Markov Models — regime detection |
| `scipy` | Scientific computing, optimization |
| `statsmodels>=0.14.0` | Statistical models (ARIMA, GARCH helpers) |
| `arch>=6.0` | ARCH/GARCH volatility modeling |
| `cvxpy` | Convex optimization (portfolio) |
| `nltk==3.9.1` | Natural language processing |
| `textblob==0.19.0` | NLP sentiment analysis |
| `vaderSentiment==3.3.2` | Sentiment analysis |
| `joblib==1.5.1` | Model serialization |
| `networkx` | Graph algorithms (correlation networks) |
| `ollama` | Local LLM inference |

#### Data / Numerical
| Package | Purpose |
|---------|---------|
| `numpy>=1.24.0` | Numerical arrays |
| `pandas>=2.0.0` | DataFrames |
| `pyarrow>=12.0.0` | Columnar data, Parquet |

#### Web Framework / Dashboard
| Package | Purpose |
|---------|---------|
| `fastapi>=0.104.0` | ASGI web framework (dashboard API) |
| `uvicorn[standard]>=0.24.0` | ASGI server |
| `flask` | Legacy institutional dashboard |

#### Crypto Exchange APIs
| Package | Purpose |
|---------|---------|
| `ccxt==4.5.0` | Unified crypto exchange API (MEXC, Binance) |
| `python-binance==1.0.29` | Binance-specific wrapper |
| `alpha_vantage==3.0.0` | Financial market data |
| `yfinance==0.2.65` | Yahoo Finance data |

#### HTTP / Networking / WebSockets
| Package | Purpose |
|---------|---------|
| `aiohttp==3.12.15` | Async HTTP client/server |
| `requests==2.32.4` | Sync HTTP client |
| `websocket-client==1.8.0` | WebSocket client |
| `websockets==15.0.1` | WebSocket server/client |

#### Social Media / News APIs
| Package | Purpose |
|---------|---------|
| `tweepy==4.16.0` | Twitter/X API |
| `praw==7.8.1` | Reddit API |
| `newsapi-python==0.2.7` | News API client |

#### Visualization
| Package | Purpose |
|---------|---------|
| `matplotlib>=3.7.0` | Plotting |
| `seaborn>=0.12.0` | Statistical visualization |

#### Configuration / Serialization
| Package | Purpose |
|---------|---------|
| `PyYAML==6.0.2` | YAML parsing |
| `python-dotenv==1.1.1` | Environment variable loading |

#### Database
| Package | Purpose |
|---------|---------|
| `peewee==3.18.2` | Lightweight ORM |
| (sqlite3 stdlib) | Primary database |

---

## 7. Module Status

### Classification Summary

| Status | Count | Description |
|--------|------:|-------------|
| ACTIVE (main bot) | 127 | Reachable from `renaissance_trading_bot.py` import tree |
| ACTIVE (arbitrage) | 1 | Reachable only from `run_arbitrage.py` |
| ACTIVE (both) | 20 | Reachable from both entry points |
| DORMANT | 68 | `__init__.py` re-exports, unused integration modules, simulation framework |
| SCRIPT | 40 | Standalone scripts, one-off tools, training scripts |
| TEST | 85 | Unit and integration tests |
| **Total** | **341** | |

Main bot entry: `renaissance_trading_bot.py` → reachable: 147 modules
Arbitrage entry: `run_arbitrage.py` → reachable: 21 modules
Combined reachable: 148 unique ACTIVE modules

### ACTIVE Modules — Main Bot (127)

Core bot:
- `renaissance_trading_bot.py`, `run_renaissance_bot.py`
- `renaissance_engine_core.py`, `renaissance_types.py`
- `renaissance_signal_fusion.py`

ML Pipeline:
- `ml_model_loader.py`, `ml_integration_bridge.py`, `real_time_pipeline.py`
- `risk_gateway.py`, `vae_anomaly_detector.py`

Data:
- `binance_spot_provider.py`, `market_data_provider.py`, `database_manager.py`
- `data_validator.py`, `derivatives_data_provider.py`
- `data_module/bar_aggregator.py`

Signal Engines:
- `enhanced_technical_indicators.py`, `microstructure_engine.py`
- `alternative_data_engine.py`, `volume_profile_engine.py`
- `fractal_intelligence.py`, `market_entropy_engine.py`
- `garch_volatility_engine.py`, `cross_asset_engine.py`
- `advanced_mean_reversion_engine.py`, `breakout_scanner.py`
- `correlation_network_engine.py`, `medallion_signal_analogs.py`
- `confluence_engine.py`, `quantum_oscillator_engine.py`
- `signals/liquidation_detector.py`, `signals/multi_exchange_bridge.py`, `signals/signal_aggregator.py`
- `intelligence/fast_mean_reversion.py`, `intelligence/insurance_scanner.py`

Position & Risk:
- `position_manager.py`, `position_sizer.py`
- `regime_overlay.py`, `advanced_regime_detector.py`, `medallion_regime_predictor.py`
- `enhanced_config_manager.py`
- `core/devil_tracker.py`, `core/kelly_position_sizer.py`
- `core/leverage_manager.py`, `core/portfolio_engine.py`, `core/signal_throttle.py`
- `portfolio/position_reevaluator.py`, `portfolio_health_monitor.py`
- `unified_portfolio_engine.py`
- `signal_auto_throttle.py`, `signal_validation_gate.py`
- `self_reinforcing_learning.py`, `genetic_optimizer.py`
- `meta_strategy_selector.py`, `performance_attribution_engine.py`

Execution:
- `execution_algorithm_suite.py`, `execution/synchronized_executor.py`, `execution/trade_hider.py`
- `slippage_protection_system.py`, `market_mictostructure_analyzer.py`
- `coinbase_client.py`, `coinbase_advanced_client.py`
- `market_making_engine.py`

Polymarket:
- `polymarket_bridge.py`, `polymarket_scanner.py`, `polymarket_executor.py`

Monitoring:
- `monitoring/alert_manager.py`, `monitoring/telegram_bot.py`
- `monitoring/beta_monitor.py`, `monitoring/sharpe_monitor.py`, `monitoring/capacity_monitor.py`
- `alert_manager.py`, `logger.py`
- `ghost_runner.py`, `institutional_dashboard.py`

Dashboard:
- `dashboard/server.py`, `dashboard/config.py`, `dashboard/db_queries.py`
- `dashboard/event_emitter.py`, `dashboard/ws_manager.py`, `dashboard/backtest_runner.py`

Recovery:
- `recovery/database.py`, `recovery/shutdown.py`, `recovery/state_manager.py`

Intelligence:
- `intelligence/microstructure_predictor.py`, `intelligence/multi_horizon_estimator.py`
- `intelligence/regime_detector.py`, `intelligence/regime_predictor.py`
- `intelligence/statistical_predictor.py`

Agents:
- `agents/coordinator.py`, `agents/base.py`, `agents/event_bus.py`
- `agents/data_agent.py`, `agents/signal_agent.py`, `agents/risk_agent.py`
- `agents/execution_agent.py`, `agents/portfolio_agent.py`
- `agents/monitoring_agent.py`, `agents/meta_agent.py`
- `agents/model_retrainer.py`, `agents/quant_researcher.py`
- `agents/observation_collector.py`, `agents/deployment_monitor.py`
- `agents/config_deployer.py`, `agents/proposal.py`, `agents/safety_gate.py`
- `agents/db_schema.py`

Training:
- `scripts/training/retrain_weekly.py`, `scripts/training/training_utils.py`
- `scripts/training/train_quantum_transformer.py`, `scripts/training/train_bidirectional_lstm.py`
- `scripts/training/train_cnn.py`, `scripts/training/train_dilated_cnn.py`
- `scripts/training/train_gru.py`, `scripts/training/train_meta_ensemble.py`
- `scripts/training/fetch_training_data.py`, `scripts/training/fetch_historical_data.py`
- `train_vae.py`

### ACTIVE Modules — Arbitrage (20 + 1 entry)

- `run_arbitrage.py`
- `arbitrage/orchestrator.py`, `arbitrage/exchanges/base.py`
- `arbitrage/exchanges/binance_client.py`, `arbitrage/exchanges/mexc_client.py`, `arbitrage/exchanges/bybit_client.py`
- `arbitrage/detector/cross_exchange.py`
- `arbitrage/execution/engine.py`, `arbitrage/execution/triangular_executor.py`
- `arbitrage/triangular/triangle_arb.py`
- `arbitrage/funding/funding_rate_arb.py`
- `arbitrage/costs/model.py`
- `arbitrage/orderbook/unified_book.py`
- `arbitrage/risk/arb_risk.py`
- `arbitrage/safety/contract_verifier.py`
- `arbitrage/tracking/performance.py`
- `arbitrage/inventory/manager.py`
- `data_module/bar_aggregator.py`
- `execution/synchronized_executor.py`, `execution/trade_hider.py`

### DORMANT (68 modules)

Includes: all `__init__.py` files (29), simulation framework (16 `sim_*.py` files), dashboard router files (13), and unused integration modules (`enhanced_data_pipeline.py`, `fear_greed_client.py`, `glassnode_client.py`, `twitter_client.py`, `reddit_client.py`, `order_book_collector.py`, `ml_enhanced_signal_fusion.py`, `unified_ml_models.py`, `trade_logger.py`, `intelligence/cone_calibrator.py`, `intelligence/signal_validator.py`, `inventory_manager.py`, `recovery/recovery_engine.py`).

Note: Dashboard router files are classified as DORMANT because they are imported dynamically by `dashboard/server.py` via the `dashboard.routers` package, not directly by the main bot import tree. They are functionally active when the dashboard is running.

### SCRIPT (40 modules)

Standalone scripts and one-off tools:
- `backfill_bars.py`, `check_readiness.py`, `safe_startup.py`, `verify_keys_live.py`
- `replay_backtest.py`, `sim_runner.py`
- `scripts/seed_historical_bars.py`, `scripts/check_vanry.py`, `scripts/fetch_derivatives_history.py`
- `scripts/patch_notebook.py`, `scripts/populate_model_ledger.py`
- `scripts/validate_model_deployment.py`
- `scripts/backtest_ml_pipeline.py`, `scripts/test_all_architectures.py`
- `scripts/test_anticollapse.py`, `scripts/test_loss_fix.py`
- `scripts/training/train_all.py`, `scripts/training/train_lightgbm.py`
- `stress_test_engine.py`, `stress_test_scenario.py`, `spread_optimizer.py`
- `tail_risk_protector.py`, `threshold_manager.py`, `transaction_cost_minimizer.py`
- `confidence_calculator.py`, `consciousness_engine.py`, `enhanced_decision_framework.py`
- `ml_config.py`, `order_logic.py`, `pattern_confidence.py`
- `step1_ensemble_weighting.py`
- `test_adapters.py`, `test_real_time_pipeline.py`, `test_step12_deepening.py`
- Plus 6 `__main__.py` entry points

### TEST (85 modules)

- `tests/` directory: 77 test files
- `dashboard/tests/`: 4 test files
- 5 integration tests (`integration_test_STEP*.py`)
- 4 standalone test files at root level

### Import Hub

`renaissance_trading_bot.py` directly imports **87 project modules**, making it the central hub. Key dependency chains:

```
run_renaissance_bot.py
  → renaissance_trading_bot.py (87 imports)
    → agents.coordinator (15 imports)
    → arbitrage.orchestrator (14 imports)
    → core.portfolio_engine (6 imports)
    → ml_integration_bridge → ml_model_loader
    → regime_overlay → advanced_regime_detector
    → risk_gateway → vae_anomaly_detector
    → dashboard.server → 13 routers
```

---

*End of audit. Generated 2026-02-23.*
