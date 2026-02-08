# 🧪 Renaissance Bot: Models & Mathematical Guide

This guide details the mathematical foundations and machine learning models driving the bot's "mostly right, 100% of the time" edge.

---

## 🔮 1. Medallion Regime Prediction (HMM)
**Module**: `medallion_regime_predictor.py` / `regime_overlay.py`

The bot uses a **Hidden Markov Model (HMM)** with Gaussian emissions to identify latent market states. 
- **States**: Typically 3 (Bullish/Trending, Bearish/Volatile, Sideways/Mean-Reverting).
- **Inputs**: Log-returns, realized volatility, and volume acceleration.
- **Goal**: Predict the *transition probability* between regimes to adjust signal weights before the shift is obvious.

---

## 🛡️ 2. Black Swan Shield (VAE Anomaly Detection)
**Module**: `risk_gateway.py` / `neural_network_prediction_engine.py`

A **Variational AutoEncoder (VAE)** is trained on historical "normal" market feature vectors (fractal dimensions, entropy, etc.).
- **Mechanism**: The VAE attempts to reconstruct the current market feature vector.
- **Signal**: If the reconstruction loss exceeds a dynamic threshold (Sigma > 2.5), the market is deemed "anomalous" (unseen state).
- **Result**: Trading is automatically halted to prevent losses in chaotic or irrational market conditions.

---

## 🧬 3. Unified ML Suite (PyTorch)
**Module**: `unified_ml_models.py` / `real_time_pipeline.py`

The bot features a high-performance PyTorch backend running multiple model architectures in parallel:
- **CNN-LSTM**: Captures both spatial (technical indicator patterns) and temporal (time-series) features.
- **N-BEATS**: A deep neural architecture for univariate time-series forecasting, optimized for trend/seasonality decomposition.
- **Quantum Transformer**: Implements multi-head attention to identify long-range dependencies in market microstructure.
- **Ensemble Meta-Learner**: A specialized layer that weights the predictions of all base models based on their recent real-world accuracy.

---

## 🏛️ 4. Confluence Engine (Non-Linear Reasoning)
**Module**: `confluence_engine.py`

Unlike standard bots that just sum up signals, the Confluence Engine identifies **Resonance**:
- **Rule 1: OrderFlow-Technical Convergence**: Boosts confidence if institutional flow (Order Flow) aligns with retail momentum (RSI/MACD).
- **Rule 2: Microstructure Exhaustion**: Identifies mean-reversion points when high toxicity (VPIN) meets Bollinger Band extremes.
- **Rule 3: Fractal-Quantum Resonance**: Boosts conviction when DTW pattern matching and Quantum Oscillator levels align.

---

## ⚖️ 5. Market Making & Execution (Avellaneda-Stoikov)
**Module**: `market_making_engine.py`

When in "Maker" mode, the bot uses a modified Avellaneda-Stoikov model:
- **Inventory Skewing**: If the bot is long BTC, it lowers its ask price and its bid price to encourage a sell and return to neutral inventory.
- **VPIN-Adaptive Spread**: The bot monitors VPIN (Volume-Synchronized Probability of Informed Trading). If toxicity is high, it automatically widens spreads to protect against informed traders.
- **TCO Spread Gating**: The Transaction Cost Optimizer (TCO) delays orders if the bid-ask spread exceeds 1% to prevent "slippage bleed".

---

## 🏛️ 6. Basis Trading (Arbitrage & Carry)
**Module**: `basis_trading_engine.py`

Exploits the relationship between Spot and Perpetual/Futures markets:
- **Cash-and-Carry**: Buy Spot and Sell Futures when the premium (Basis) and Funding Rate provide a high annualized yield (Threshold: 10%).
- **Reverse-Carry**: Sell Spot and Buy Futures when the discount is extreme.
- **Funding Harvest**: Specifically optimized to capture 8-hourly funding payments while hedging delta risk.

---

## 🧠 7. Deep NLP Reasoning (LLM)
**Module**: `deep_nlp_bridge.py`

Goes beyond simple sentiment scores by using Large Language Models (Llama/GPT):
- **Institutional Context**: Extracts the "Reasoning" behind news moves (e.g., "positive whale activity" vs. "macro-economic shift").
- **Weighted Fusion**: Deep NLP insights are weighted at 25% of the total Alternative Data signal, providing a "rational" check on social media noise.
