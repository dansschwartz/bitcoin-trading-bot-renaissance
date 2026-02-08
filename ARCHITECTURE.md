# 🏛️ Renaissance Bot: Institutional Architecture

This document provides a comprehensive overview of the technical architecture and data flow of the Renaissance Technologies-inspired Bitcoin trading bot.

## 📐 System Overview
The bot is designed as a **Multi-Strategy Autonomous Predator**. It combines market microstructure, high-dimensional technical analysis, alternative data, and institutional arbitrage into a single "Golden Path" execution engine.

---

## 🏗️ Core Components

### 1. Orchestration Layer (`renaissance_trading_bot.py`)
The central nervous system of the bot. It manages the lifecycle of a trading cycle:
- **Data Collection**: Aggregates from Coinbase (Live) and other exchanges (Real-time).
- **Signal Generation**: Parallel execution of 20+ alpha sources.
- **Fusion & Confluence**: Weighted linear fusion followed by non-linear boost logic.
- **Execution**: Smart order routing (TWAP/VWAP/Sniper).

### 2. Signal Intelligence Suite
- **Microstructure Engine (`microstructure_engine.py`)**: Analyzes L2 Order Book imbalance, VPIN toxicity, and large trade flows.
- **Fractal Intelligence (`fractal_intelligence.py`)**: DTW-based pattern matching against "Golden Fractals".
- **Quantum Oscillator (`quantum_oscillator_engine.py`)**: Maps price to Quantum Harmonic Oscillator energy levels.
- **HMM Regime Predictor (`medallion_regime_predictor.py`)**: Uses Hidden Markov Models to predict market state transitions (Trending/Mean-Reverting/Chaos).

### 3. Institutional Alpha Layers
- **Basis Trading Engine (`basis_trading_engine.py`)**: Exploits Spot-Futures spreads and funding rates (Cash-and-Carry).
- **Deep NLP Bridge (`deep_nlp_bridge.py`)**: LLM-driven reasoning on news and social context.
- **Guavy Crypto API Integration**: Primary alternative data feed replacing legacy social/news modules with high-fidelity institutional data.
- **Market Making Engine (`market_making_engine.py`)**: Transitions the bot to a Liquidity Provider using inventory skewing logic.

---

## 🔄 The Self-Reinforcing Feedback Loop
The bot implements a closed-loop learning system:
1. **Persistence**: Every decision and market snapshot is saved to `data/renaissance_bot.db`.
2. **Labeling**: A background process matches decisions with realized PnL after 60 minutes.
3. **Evolution**: The `GeneticWeightOptimizer` evolves signal weights based on realized performance.
4. **Fine-tuning**: The ML Ensemble meta-learner is retrained on the most recent "correct" labels.

---

## 🛡️ Risk Management (The Fortress)
- **Risk Gateway (`risk_gateway.py`)**: A multi-stage gatekeeper using VaR (Value at Risk) and CVaR.
- **Black Swan Shield**: A Variational AutoEncoder (VAE) that blocks trades if the market feature vector is an anomaly (unseen state).
- **TCO (Transaction Cost Optimizer)**: Delays execution during high toxicity (VPIN > 0.8) or wide spreads (>1%).

---

## 📊 Monitoring & Visualization
- **Institutional Dashboard**: A Flask-based UI (Port 5000) providing real-time visibility into the "Inner Thoughts" (Consciousness) of the bot.
- **Readiness Script (`check_readiness.py`)**: Pre-flight check for API keys, latency, and DB integrity.
