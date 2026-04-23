# Architecture

## Overview

The Renaissance Trading Bot is a multi-strategy, multi-exchange cryptocurrency trading system. It collects market data from Binance, generates trading signals through 20+ alpha sources, filters them through regime-aware risk controls, and simulates execution on MEXC.

```
Binance (data, 70-90 pairs) ──> Signal Generation ──> Risk Gateway ──> MEXC (paper execution)
                                     │                     │
                                     ▼                     ▼
                              ML Ensemble            VAE Anomaly
                              HMM Regime             VaR / CVaR
                              Confluence             Position Limits
```

## Core Bot (`renaissance_trading_bot.py` + `bot/`)

The main bot class was refactored from a 7,681-line god class into a 2,149-line orchestrator that delegates to 9 focused modules in `bot/`:

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `builder.py` | 1,016 | Component initialization — wires all subsystems (BotBuilder) |
| `cycle_ops.py` | 1,052 | Per-cycle helpers — drawdown controls, exposure monitoring, macro cache |
| `lifecycle.py` | 1,031 | Startup, shutdown, background loops, continuous trading, kill switch |
| `decision.py` | 827 | Trading decisions — Kelly sizing, regime gates, confidence thresholds |
| `adaptive.py` | 613 | Adaptive learning — weight evolution, attribution, Kelly calibration |
| `data_collection.py` | 588 | Market data fetching, bar management, pair-tier rotation |
| `position_ops.py` | 580 | Position management — open/close, P&L computation, state restore |
| `signals.py` | 415 | Signal generation — collect data, weighted fusion, ensemble scoring |
| `helpers.py` | 391 | Logging setup, heartbeat, cycle summaries, config helpers |

**Total bot/ package:** 6,513 lines across 9 modules.

### Data Flow (Single Cycle)

```
1. data_collection.py    Fetch prices from Binance for current tier's pairs
        │
2. signals.py            Generate signals from 20+ sources (technical, ML, microstructure)
        │
3. decision.py           Apply regime gate, confidence threshold, Kelly sizing
        │
4. risk_gateway.py       Multi-stage filter: VaR, CVaR, VAE anomaly, exposure limits
        │
5. position_ops.py       Execute paper trade, update positions, compute P&L
        │
6. adaptive.py           Update signal weights based on realized outcomes
        │
7. cycle_ops.py          Log cycle summary, check drawdown, update equity curve
```

### Regime Detection

The regime system uses a 3-tier hierarchy:

1. **Bootstrap rules** (immediate) — Simple volatility + trend heuristics for instant classification
2. **AdvancedRegimeDetector** (30+ bars) — Statistical regime detection from OHLCV bars
3. **HMM (200+ bars)** — Trained 3-state GaussianHMM for probabilistic regime prediction

Regimes: `trending_up`, `trending_down`, `mean_reverting`, `high_volatility`, `low_volatility`

Low volatility boosts mean reversion confidence (fixed from the original bug where it zeroed out all signals).

## Signal Sources

### Technical Indicators (`enhanced_technical_indicators.py`)
40+ indicators including RSI, MACD, Bollinger Bands, Ichimoku, ADX, and custom derivatives.

### Microstructure Engine (`microstructure_engine.py`)
Order book imbalance, VPIN (Volume-Synchronized Probability of Informed Trading), large trade detection.

### ML Ensemble (`real_time_pipeline.py`)
7 models generating directional predictions:
- CNN-LSTM, N-BEATS, Transformer, Bi-LSTM (PyTorch)
- LightGBM (scikit-learn)
- VAE anomaly detector
- HMM regime predictor

**ML accuracy fixes applied** (see `docs/ML_ACCURACY_INVESTIGATION.md`):
1. LightGBM training/inference feature mismatch — training used 104 features, inference expected 294
2. Debiaser/normalizer corruption — Z-score normalization was inverting prediction signs
3. Per-window standardization distribution shift — std floor added for quiet markets
4. Cross-asset feature zero-padding — logging added for prediction reliability
5. DirectionalLoss hyperparameter mismatch — v7 defaults applied across all training
6. Feature staleness — staleness logging added for lagged predictions

### Other Signal Sources
- **Fractal Intelligence** — Dynamic Time Warping pattern matching
- **Quantum Oscillator** — Harmonic oscillator energy level mapping
- **Confluence Engine** — Non-linear boosting when multiple signals align
- **Genetic Optimizer** — Evolves signal weights based on realized P&L

## Risk Management

### Risk Gateway (`risk_gateway.py`)
Multi-stage filter that every trade must pass:
1. Position limit check (max 10 simultaneous)
2. Exposure cap check
3. VaR (Value at Risk) threshold
4. CVaR (Conditional VaR) threshold
5. VAE anomaly score — blocks trades in unseen market states
6. Drawdown circuit breaker

All gateway decisions (PASS/REJECT with reasons) are logged and visible on the dashboard.

### Risk Alerts (`risk_alert_engine.py`)
Fires alerts when:
- Drawdown exceeds threshold
- Exposure exceeds cap
- VAE anomaly score spikes
- Correlation breakdown detected

## Dashboard (`dashboard/`)

React + TypeScript SPA served by FastAPI at `localhost:8080`.

### Backend
- **server.py** — FastAPI + Uvicorn, serves API and static frontend
- **23 routers** in `dashboard/routers/` — system, brain, risk, trades, analytics, medallion, arbitrage, polymarket, and more
- **db_queries.py** — SQL query layer for all dashboard data
- **ws_manager.py** — WebSocket connection management for real-time updates

### Frontend
- React + TypeScript + Tailwind CSS, built with Vite
- Tabs: Overview, Brain (regime/confluence/VAE), Risk, Trades, Analytics, Command Center
- Real-time updates via WebSocket

### Key API Endpoints
```
GET /api/brain/regime        — Current market regime + confidence
GET /api/brain/confluence    — Signal confluence scores
GET /api/brain/vae           — VAE anomaly detection state
GET /api/system/health       — Bot uptime, cycle count
GET /api/risk/alerts         — Active risk alerts
GET /api/risk/gateway        — Gateway pass/reject log
GET /api/trades/decisions    — Recent trading decisions
GET /api/analytics/pnl       — P&L breakdown (realized + unrealized)
WS  /ws                      — Real-time streaming
```

## Arbitrage (`arbitrage/`)

Cross-exchange arbitrage module with 24 subdirectories:
- **Spot arbitrage** — Price discrepancies between Binance, Coinbase, MEXC, Kraken
- **Funding rate arbitrage** — Perpetual futures funding rate capture
- **Triangular arbitrage** — 3-way currency path detection
- **Market making** — Two-sided quoting with inventory management

Entry point: `python run_arbitrage.py` or `python -m arbitrage`

## Agents (`agents/`)

Multi-agent orchestration framework:
- Signal, risk, portfolio, execution, and data agents
- Event bus for publish-subscribe communication
- Safety gate with circuit breakers and kill switches
- Model retrainer for online learning

## Data Layer

### Database: `data/trading.db` (SQLite)

| Table | Purpose |
|-------|---------|
| `five_minute_bars` | OHLCV bars (pair, exchange, timestamps, OHLCV, volume) |
| `decisions` | Every trading decision with signal, confidence, regime, VAE loss |
| `positions` | Open and closed positions with entry/exit prices and P&L |
| `trades` | Executed trades with exchange, side, size, price, fee |
| `devil_tracker` | Per-position cost tracking |
| `signal_daily_pnl` | Daily P&L by signal source |

### Data Collection
- **Binance Spot API** (`binance_spot_provider.py`) — Free, no auth, 70-90 pairs
- **4-tier scanning**: Tier 1 (top 15) every cycle, Tier 2 (16-50) every 2nd, etc.
- **Parallel fetch**: `asyncio.gather` with `Semaphore(15)`, typically <15s for all pairs
- **Warmup gate**: 30 bars required before trading new pairs

## Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Web Framework | FastAPI, Uvicorn |
| Frontend | React, TypeScript, Tailwind CSS, Vite |
| Database | SQLite3 |
| Async | asyncio |
| ML/Stats | scikit-learn, hmmlearn, PyTorch, LightGBM, pandas, numpy |
| APIs | Binance, Coinbase, MEXC, Kraken (REST + WebSocket) |
| Deployment | Docker, docker-compose |
| Testing | pytest (87 tests) |
| Code Quality | ruff (linter + formatter) |

## Configuration

All configuration lives in `config/config.json`:
- **Strategy flags** — Enable/disable individual systems (regime_overlay, risk_gateway, etc.)
- **Risk limits** — Max drawdown, position size, exposure caps
- **Scanning tiers** — How many pairs per tier and scan frequency
- **Exchange settings** — Credentials loaded from `.env`
