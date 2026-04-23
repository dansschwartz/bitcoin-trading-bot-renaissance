# Renaissance Trading Bot

A Renaissance Technologies-inspired cryptocurrency trading bot that paper trades across multiple exchanges using statistical edge extraction, ML-driven regime detection, and multi-strategy signal fusion.

**Status: Paper Trading Only** — The bot simulates trades against live market data. It does not execute real orders.

## Architecture

The system is built around a core orchestrator (`renaissance_trading_bot.py`, 2,149 lines) that delegates to 9 focused subsystems in the `bot/` package. It collects data from Binance (free, no auth required), generates signals through 20+ alpha sources, filters them through a multi-stage risk gateway, and simulates execution on MEXC at 0% maker fees.

Key components:
- **bot/** — 9 extracted modules handling signals, decisions, positions, data, lifecycle, and adaptive learning
- **dashboard/** — React + FastAPI real-time dashboard at `localhost:8080` with 23 API routers
- **arbitrage/** — Cross-exchange arbitrage engine (Binance, Coinbase, MEXC, Kraken)
- **agents/** — Multi-agent orchestration framework for distributed trading logic
- **ML Models** — HMM regime detection, VAE anomaly detection, CNN-LSTM/N-BEATS/Transformer ensemble

For the full architecture breakdown, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Quickstart

### Prerequisites
- Python 3.11+
- Node.js 18+ (for dashboard frontend)

### Setup

```bash
# Clone and enter the project
cd bitcoin-trading-bot-renaissance

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and add your API keys
cp .env.example .env

# Review trading configuration
# config/config.json — trading parameters, strategy flags, risk limits
# config/config.example.json — safe template
```

### Run

```bash
# Single test cycle (verify everything works)
python run_renaissance_bot.py --test

# Paper trading (continuous)
python run_renaissance_bot.py --run

# Arbitrage module (standalone)
python run_arbitrage.py
```

### Docker

```bash
make docker-build    # Build all service images
make docker-up       # Start bot + dashboard + arbitrage (detached)
make docker-down     # Stop all services
```

The `docker-compose.yml` orchestrates three services:
1. **bot** — Main trading loop + ML inference
2. **dashboard** — FastAPI web server at port 8080
3. **arbitrage** — Cross-exchange arbitrage orchestrator

### Testing

```bash
make test            # Run all 87 tests (pytest, 60s timeout)
make lint            # Check code with ruff
make format          # Format code with ruff
```

## Project Structure

```
bitcoin-trading-bot-renaissance/
├── run_renaissance_bot.py           # Main entry point
├── renaissance_trading_bot.py       # Core orchestrator (2,149 lines)
├── bot/                             # Extracted subsystems (9 modules)
│   ├── builder.py                   # Component initialization (BotBuilder)
│   ├── signals.py                   # Signal generation & weighted fusion
│   ├── decision.py                  # Trading decisions with Kelly sizing
│   ├── data_collection.py           # Market data fetching & bar aggregation
│   ├── position_ops.py              # Position management & P&L
│   ├── lifecycle.py                 # Startup, shutdown, background loops
│   ├── cycle_ops.py                 # Per-cycle helpers, drawdown, exposure
│   ├── adaptive.py                  # Adaptive weights, attribution, Kelly
│   └── helpers.py                   # Logging setup, heartbeat, summaries
├── dashboard/                       # Real-time web dashboard
│   ├── server.py                    # FastAPI + Uvicorn backend
│   ├── routers/                     # 23 API endpoint routers
│   └── frontend/                    # React + TypeScript + Tailwind SPA
├── arbitrage/                       # Cross-exchange arbitrage package
├── agents/                          # Multi-agent orchestration framework
├── core/                            # Shared data structures & utilities
├── tests/                           # 87 unit & integration tests
├── config/                          # config.json + config.example.json
├── data/                            # SQLite database (trading.db)
├── models/                          # Trained ML models (.pkl files)
├── docs/                            # Technical documentation
├── Dockerfile                       # Python 3.11 container
├── docker-compose.yml               # 3-service stack
├── Makefile                         # test, lint, format, docker targets
└── requirements.txt                 # Python dependencies
```

## Features

### Trading Engine
- **Dynamic universe**: 70-90 Binance USDT pairs filtered by $2M+ daily volume, auto-refreshed
- **4-tier scanning**: Top pairs scanned every cycle, lower tiers on rotating schedule
- **20+ signal sources**: Technical indicators, microstructure, fractal patterns, quantum oscillator, ML ensemble
- **Regime-aware**: HMM-based regime detection (trending, mean-reverting, volatile) adjusts strategy weights
- **Risk gateway**: Multi-stage filter with VaR, CVaR, VAE anomaly detection, position limits
- **Max 10 simultaneous positions** with Kelly criterion sizing

### ML Pipeline
- 7 ML models (CNN-LSTM, N-BEATS, Transformer, Bi-LSTM, LightGBM, VAE, HMM)
- 6 root causes of sub-50% accuracy identified and fixed (see `docs/ML_ACCURACY_INVESTIGATION.md`)
- Genetic weight optimizer evolves signal weights based on realized P&L
- Adaptive learning with alpha decay detection

### Dashboard (localhost:8080)
- Real-time P&L, equity curve, and position tracking
- Market regime visualization with confidence scores
- Risk alerts and gateway log
- Signal confluence and ML model status
- Success criteria tracking card

### Infrastructure
- Docker containerization with docker-compose
- 87 automated tests across bot, arbitrage, and ML modules
- CI pipeline with GitHub Actions
- ruff for linting and formatting

## Configuration

All trading parameters live in `config/config.json`:
- Strategy feature flags (regime_overlay, risk_gateway, real_time_pipeline, etc.)
- Risk limits (max drawdown, position size, exposure caps)
- Exchange credentials (via `.env` file)
- Scanning tiers and cycle intervals

## Key Documentation

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, data flow, module responsibilities |
| [CLAUDE.md](CLAUDE.md) | Autonomous operation manual for Claude Code sessions |
| [docs/ML_ACCURACY_INVESTIGATION.md](docs/ML_ACCURACY_INVESTIGATION.md) | Forensic audit of ML model accuracy with 6 root causes |
| [CHANGELOG.md](CHANGELOG.md) | All changes across 46 commits |
| [EXPERIMENTAL.md](EXPERIMENTAL.md) | Features not in the golden path |

## Paper Trading

This bot is configured for **paper trading only**. It connects to live market data but simulates all order execution. The dashboard displays a `PAPER TRADING` badge at all times. Do not switch to live trading without thorough backtesting and risk review.
