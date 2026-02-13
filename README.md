# bitcoin-trading-bot-renaissance

Renaissance-inspired BTC-USD trading bot with a single, stabilized golden path. The project is currently paper-trading only and wired for live market data.

**Golden Path (Unified Architecture)**
- Entrypoint: `run_renaissance_bot.py`
- Core Engine: `renaissance_trading_bot.py` (Now includes all "Enhanced" ML and Risk capabilities by default)
- Config: `config/config.json`
- Live market data: Coinbase Advanced Trade
- Multi-Exchange: Parallel feeds from Kraken, KuCoin, Bitfinex for global volume profile.
- ML Backend: PyTorch-powered (CNN-LSTM, N-BEATS, Transformer, VAE).

**Quickstart**
1. Copy `.env.example` to `.env` and fill in your credentials.
2. Review `config/config.json` and adjust trading parameters.
3. Run a single test cycle:

```bash
python run_renaissance_bot.py --test
```

**Replay/Backtest**
Use `replay_backtest.py` with a CSV containing `timestamp, open, high, low, close, volume`:

```bash
python replay_backtest.py --csv path/to/bars.csv
```

**Configuration**
- `config/config.json` is the canonical config used by the golden path.
- `config/config.example.json` is a safe template.
- `config/data_pipeline_config.json` is reserved for pipeline experiments.

The file contains:
- `COINBASE_API_KEY`: Your Coinbase Advanced Trade API key.
- `COINBASE_API_SECRET`: Your EC private key (signed JWT authentication).
- `TWITTER_BEARER_TOKEN`: For real-time social sentiment analysis.
- `REDDIT_CLIENT_ID / SECRET`: For deep subreddit sentiment tracking.
- `NEWSAPI_KEY`: For global financial news analysis.
- `WHALE_ALERT_KEY`: For tracking large on-chain transactions.
- `GUAVY_API_KEY`: Primary source for crypto alternative data (replaces Twitter/Reddit/News).

**Advanced Features (Opt-in)**
- `regime_overlay`: Enables **Medallion-style Regime Prediction**. Uses a **Hidden Markov Model (HMM)** to predict market transitions, coupled with trend persistence and volatility acceleration scoring.
- `risk_gateway`: Enables **Advanced Risk Management Fortress**. Now includes **VAE-Based Anomaly Detection** (Black Swan Shield) to block trades in "unseen" or chaotic market states.
- `real_time_pipeline`: Enables Real-Time Pipeline (Step 12) for multi-exchange feed aggregation and parallel model processing. Powered by a **Unified PyTorch ML Backend** (CNN-LSTM, N-BEATS, Transformer, Bi-LSTM, VAE).
- `step10_execution`: Integrates **Black-Litterman Portfolio Optimization** and a **Smart Execution Suite** (TWAP/VWAP/Sniper). Now includes a **Transaction Cost Optimizer (TCO)** with **VPIN (Liquidity Toxicity)** and spread awareness.
- `multi_asset`: Support for parallel trading of **BTC-USD and ETH-USD**. Features **Cross-Asset Lead-Lag Alpha** and **Statistical Arbitrage** (Pairs Trading).
- `persistence_analytics`: **SQLite Persistence Layer** stores every decision, trade, and ML prediction. Includes **Self-Reinforcing Learning Loops** that automatically label outcomes and fine-tune models in real-time.
- `global_intelligence`: **Deep Alternative Data** (Reddit, News, Twitter, Whale Alert) and **High-Dimensional Discovery** (Fractal DTW, Market Entropy, Quantum Oscillator).
- `evolutionary_ai`: **Genetic Weight Optimizer** evolves signal weights based on **Realized PnL** and market regime feedback.
- `confluence_engine`: (New) **Non-Linear Meta-Learning**. Identifies a "Confluence of Edges" where combined signals (e.g., Order Flow + Technicals, or Fractal + Quantum) receive a non-linear boost based on institutional confluence rules. Includes divergence detection (VPIN + Bollinger) for mean reversion.
- `basis_trading`: **Arbitrage & Carry**. Exploits the price difference between Spot and Futures markets and harvests funding rates for low-risk yield.
- `deep_nlp`: **LLM Reasoning**. Connects a local Llama or GPT model to perform deep reasoning on news and social media, extracting institutional-grade context.
- `market_making`: **Liquidity Provision**. Transitions from taker to maker, providing two-sided quotes, managing inventory skew, and capturing the bid-ask spread.
- `institutional_dashboard`: **Consciousness UI**. Real-time web interface to visualize bot performance, market regimes, and the "Inner Thoughts" of the AI engine.
- `meta_strategy`: **Adaptive Execution Mode**. Dynamically switches between **Taker (Renaissance)** and **Maker (Citadel)** modes based on VPIN toxicity and market regime.
- `performance_attribution`: **Factor Analysis**. Decomposes P&L into Alpha, Beta, and specific factor exposures (Microstructure, Technical, Alternative) using realized outcome labeling.

## 🛠️ Troubleshooting & Environment

If you see "No Python Interpreter" in your IDE (PyCharm/VS Code):

### 1. PyCharm Fix
1. Go to **Settings** (Cmd + , on Mac).
2. Navigate to **Project: bitcoin-trading-bot-renaissance** > **Python Interpreter**.
3. Click **Add Interpreter** > **Add Local Interpreter**.
4. Select **System Interpreter** and point it to your Python path (usually `/Users/danielschwartz/miniconda3/bin/python` or run `which python` in terminal to find it).

### 2. VS Code Fix
1. Open the Command Palette (`Cmd + Shift + P`).
2. Type `Python: Select Interpreter`.
3. Select the recommended version (e.g., Python 3.13.x).

### 3. Verify via Terminal
You can always run the bot directly from the terminal, which bypasses IDE configuration issues:
```bash
python run_renaissance_bot.py --test
```

---

## 📚 Documentation
For deep dives into the bot's design and logic, see:
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: High-level design, data flow, and institutional layers.
- **[MODELS_GUIDE.md](MODELS_GUIDE.md)**: Mathematical foundations, ML models (HMM, VAE, CNN-LSTM), and confluence rules.

**Self-Reinforcing Loop**
The bot now implements a closed feedback loop:
1. **Collect & Persist**: Saves market snapshots and ML predictions for every cycle.
2. **Label**: Automatically calculates realized returns and labels decisions as correct/incorrect after a fixed horizon.
3. **Fine-tune**: Periodically calibrates the ML Ensemble (meta-learner) and evolves signal weights using live performance data (Realized PnL). Includes **Alpha Decay Detection** to identify signals losing their edge.

**Experimental Modules**
Created `check_readiness.py` to ensure all systems (API keys, Network, DB) are ready for live deployment.

See `EXPERIMENTAL.md` for everything not included in the golden path.
