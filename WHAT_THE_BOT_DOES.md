# What The Bot Does — Plain English

*Last updated: April 24, 2026*

---

## The One-Sentence Version

This bot watches cryptocurrency markets 24/7, combines 15+ different signals to decide when to buy and sell, and manages its positions automatically — all while paper trading (no real money at risk).

---

## How It Works (The Big Picture)

Every 5 minutes, the bot runs a **trading cycle**. Think of it as a very fast, very disciplined trader who:

1. **Looks at the market** — Pulls price data, order books, and trading volume from Binance for 11 different cryptocurrencies (BTC, ETH, SOL, DOGE, XRP, ADA, LINK, DOT, MATIC, UNI, ATOM)
2. **Analyzes everything** — Runs 15+ different analysis methods on the data
3. **Makes a decision** — Combines all those analyses into a single BUY, SELL, or HOLD verdict for each asset
4. **Executes trades** — Places orders on MEXC (chosen for its 0% maker fees)
5. **Manages positions** — Watches open positions and closes them when targets are hit or risk limits are breached
6. **Learns** — Tracks which signals were right and wrong, adjusts weights over time

---

## The 15 Signals (What It's Looking At)

The bot doesn't rely on any single indicator. It blends 15 different signals, each with a weight reflecting how much the bot trusts it:

### The Heavy Hitters (highest weights)

| Signal | Weight | What It Does |
|--------|--------|-------------|
| **ML Ensemble** | 20.3% | 7 machine learning models (neural networks + gradient boosting) that predict whether the price will go up or down in the next 5 minutes. Recently retrained with corrected parameters. |
| **Market Entropy** | 15.6% | Measures how predictable or chaotic the market is. High entropy = random/unpredictable = don't trade. Low entropy = patterns exist = trade. |
| **RSI** | 11.8% | Classic momentum indicator. Oversold (RSI < 30) = buy signal. Overbought (RSI > 70) = sell signal. |
| **Bollinger Bands** | 9.8% | Measures whether price is at the top or bottom of its recent range. Price near lower band = buy. Price near upper band = sell. |

### The Supporting Cast

| Signal | Weight | What It Does |
|--------|--------|-------------|
| **Lead-Lag** | 7.7% | Detects when BTC moves first and other coins follow. If BTC just jumped, maybe ETH is about to. |
| **Volume** | 6.6% | Big volume spikes often precede big price moves. |
| **Statistical Arbitrage** | 5.5% | Looks for pairs of coins that usually move together. When they diverge, bet on them converging. |
| **GARCH Volatility** | 4.5% | Predicts future volatility. High predicted vol = smaller positions. Low predicted vol = boost mean reversion bets. |
| **Fractal Patterns** | 4.3% | Compares current price patterns to 8 historical "golden patterns" using shape-matching math. |
| **Order Flow** | 3.4% | Analyzes the order book — are there more buyers or sellers lined up? |
| **Volume Profile** | 3.4% | Identifies price levels where the most trading happens (support/resistance). |
| **Multi-Exchange** | 2.8% | Compares prices across multiple exchanges. If Binance price diverges from Kraken, something's up. |
| **MACD** | 1.8% | Trend-following indicator that catches momentum shifts. |
| **Quantum Oscillator** | 1.7% | Physics-inspired model that maps price levels to energy states. Exotic but has a small edge. |
| **Alternative Data** | 0.8% | Social media sentiment (Twitter, Reddit), news, Fear & Greed index. Low weight because it's noisy. |

### How Signals Combine

All 15 signals are converted to a number between -1 (strong sell) and +1 (strong buy), then multiplied by their weights and added up. If the total is above +0.06, the bot buys. Below -0.06, it sells. In between, it holds.

The bot also needs **confidence above 25%** to act — this comes from how many signals agree with each other.

---

## The 7 ML Models

The "ML Ensemble" signal (20.3% weight) is itself a blend of 7 machine learning models:

| Model | Type | What It Learns |
|-------|------|---------------|
| **Quantum Transformer** | Attention-based neural net | Long-range dependencies in price patterns |
| **CNN** | Convolutional neural net | Local price patterns (like candlestick shapes) |
| **Bidirectional LSTM** | Recurrent neural net | Sequential patterns reading both forward and backward |
| **Dilated CNN** | Expanded convolutional net | Multi-scale patterns (1-min, 5-min, 30-min simultaneously) |
| **GRU** | Gated recurrent net | Similar to LSTM but faster and simpler |
| **LightGBM** | Gradient boosted trees | Non-linear feature interactions |
| **Meta Ensemble** | Neural net on top of the above | Learns which models to trust in which market conditions |

Each model sees 98 features (price data, technical indicators, cross-asset correlations, derivatives data) and outputs a prediction between -1 (price going down) and +1 (price going up).

**Note:** The GRU, BiLSTM, and Dilated CNN are currently disabled in config. Only 4 models are active: Quantum Transformer, CNN, LightGBM, and Meta Ensemble.

---

## Market Regime Detection

The bot doesn't trade the same way in all conditions. It detects the current **market regime** — essentially, what kind of market are we in right now?

| Regime | What It Means | How The Bot Reacts |
|--------|-------------|-------------------|
| **Trending Up** | Sustained upward move | Favors buying, wider stops |
| **Trending Down** | Sustained downward move | Favors selling/shorting |
| **Mean Reverting** | Choppy, oscillating | Buys dips, sells rips (the bot's sweet spot) |
| **Low Volatility** | Quiet market | Boosts mean-reversion signals, lowers entry thresholds |
| **High Volatility** | Wild swings | Reduces position sizes, tighter risk limits |
| **Chaotic** | No discernible pattern | Reduces trading, waits for clarity |

Regime detection uses a Hidden Markov Model (HMM) — a statistical model that figures out which "hidden state" the market is in based on observable price behavior.

---

## Active Strategies

### Strategy 1: Main Signal Fusion (the core)

This is the primary strategy described above — blend 15 signals, make directional bets on 11 crypto assets. Runs every 5 minutes.

### Strategy 2: BTC/ETH Straddles

Instead of picking a direction, this strategy bets on **movement itself**:
- Opens both a LONG and SHORT position simultaneously
- Each leg is $100
- If price moves sharply in either direction, one leg profits more than the other loses
- Uses tight trailing stops (1-2 basis points) to capture quick moves
- Closes positions after max 2 minutes
- Runs every 10 seconds (much faster than the main strategy)
- Active on BTC and ETH only

### Strategy 3: Oracle Trading

A separate, simpler strategy based on a research paper:
- Uses 6 small neural networks looking at 4-hour candles
- Trades 10 pairs (BTC, ETH, SOL, DOGE, AVAX, LINK, ADA, XRP, DOT, MATIC)
- Long-only with 10% stop loss
- $5,000 paper wallet
- More patient, slower trades than the main strategy

### Strategy 4: Polymarket (Prediction Markets)

The bot scans crypto prediction markets on Polymarket — questions like "Will BTC be above $X in 15 minutes?":
- Uses the **spread capture** approach: places limit orders on both sides and profits from the bid-ask spread
- Active assets: SOL, DOGE (live), others paper
- $500 initial bankroll, max 10 simultaneous positions
- Daily loss limit: $200

### Strategy 5: Cross-Exchange Arbitrage

Looks for price differences between exchanges (MEXC, Binance, KuCoin):
- **Cross-exchange:** Same coin, different price on two exchanges → buy low, sell high
- **Triangular:** USDT→BTC→ETH→USDT cycle where rounding creates a small profit
- **Funding rate:** Captures funding payments on perpetual futures
- **Basis:** Exploits spot vs futures price differences

---

## Risk Management (How It Protects Itself)

The bot has multiple layers of protection — think of it as a series of gates that a trade must pass through:

### Gate 1: Signal Confidence
A trade needs at least 25% confidence (multiple signals agreeing) before the bot will act.

### Gate 2: Regime Check  
In chaotic or high-volatility regimes, position sizes are automatically reduced.

### Gate 3: VAE Anomaly Detection
A neural network (Variational AutoEncoder) that's been trained on "normal" market conditions. If the current market looks nothing like anything it's seen before, it blocks the trade entirely. Think of it as a "this feels wrong" detector.

### Gate 4: Position Sizing (Kelly Criterion)
The bot sizes each position using the Kelly Criterion — a mathematical formula that calculates the optimal bet size based on your edge and your bankroll. It uses "half-Kelly" (betting half what the formula suggests) for extra safety.

### Gate 5: Portfolio Limits
- Maximum $5,000 per position
- Maximum 80% of capital deployed at once
- Maximum 40% in any single asset
- Correlated positions are penalized (if BTC and ETH usually move together, the bot won't go all-in on both)

### Gate 6: Daily Loss Limit
If the bot loses $500 in a day, it stops trading until midnight UTC.

### Gate 7: Anti-Churn
Can't trade the same asset again within 6 cycles (30 minutes) to prevent rapid flip-flopping.

### Gate 8: ML Staleness Check
If the ML prediction is more than 15 minutes old, it's discarded — no trading on stale predictions.

---

## Where The Actual Edge Comes From

Here's the honest truth based on forensic analysis of the bot's trading history:

**The ML models are barely better than a coin flip.** Even after retraining, the best model (LightGBM) hits ~51% accuracy. That's slightly better than random but not a strong signal by itself.

**The real edge is in the exit logic.** The bot has two exit strategies that work extremely well:

1. **EDGE_CONSUMED** — Closes a position as soon as the predicted price move has been captured. 94.3% win rate.
2. **AGED_PROFITABLE** — If a position is profitable after a set time, close it and take the money. 100% win rate.

The bot also quickly dumps losers (**STALE_LOSER** exit) to limit damage.

In other words: the entry signals are mediocre, but the bot is excellent at taking small profits quickly and cutting losses fast. It's more like a disciplined scalper than a prediction machine.

---

## What "Paper Trading" Means

The bot is NOT trading with real money. It:
- Gets real market data (live prices from Binance)
- Makes real decisions (same logic it would use with real money)
- Simulates trades (records what it would have bought/sold)
- Tracks P&L as if the trades were real

This means you can evaluate whether the strategies work without risking capital. The `PAPER_TRADING_ONLY = True` safety flag is hard-coded and cannot be changed by the bot itself.

---

## The Dashboard

The bot runs a web dashboard at `localhost:8080` with 17 pages showing:
- **Command Center** — Overview of all systems, success criteria checklist, activity feed
- **Brain** — What the ML models are thinking, regime classification, VAE anomaly score
- **Intelligence** — Signal breakdown, ensemble weights, prediction history
- **Positions** — Open and closed positions with P&L
- **Analytics** — Equity curve, daily P&L heatmap, return distribution
- **Risk** — Exposure, drawdown, risk alerts, gateway log
- Plus dedicated pages for Arbitrage, Polymarket, Straddles, Oracle, and more

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Trading cycle | Every 5 minutes |
| Assets monitored | 11 crypto pairs |
| Active ML models | 4 (of 7 total) |
| Signal sources | 15 |
| Risk gates | 8 layers |
| Execution exchange | MEXC (0% maker fee) |
| Data source | Binance (free, no auth needed) |
| Daily loss limit | $500 |
| Max position size | $5,000 |
| Paper trading | Yes (no real money) |

---

## What Changed Recently (April 2026 Overhaul)

The bot underwent a major code overhaul:
- **64 commits** fixing critical bugs, restructuring code, and improving ML
- Main bot file reduced from 7,681 lines to 2,149 lines
- All 7 ML models retrained with corrected parameters
- 6 root causes of ML underperformance identified and fixed
- 13 crashes/day reduced to 0 (SQLite locking, WebSocket reconnection, etc.)
- ~12,500 lines of dead code removed
- 690+ automated tests added
- Full Docker + CI/CD infrastructure added
- Project organized from 119 root files into 7 clean packages
