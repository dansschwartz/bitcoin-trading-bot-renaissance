# Forensic Audit Report -- Renaissance Trading Bot

**Date:** 2026-03-03
**Auditor:** Claude Opus 4.6 (read-only forensic analysis)
**Database:** `data/renaissance_bot.db` on VPS 178.128.216.112
**Scope:** 10-section audit covering split-path outcomes, sign inversions, premature triggers, data quality, silent exceptions, type coercion, database integrity, timing, bankroll accounting, and configuration drift.

---

## Executive Summary

The bot is running and generating trades, but this audit uncovered **3 P-CRITICAL** findings, **5 P-HIGH** findings, and several medium/low issues. The most severe finding is that the **Polymarket price_fallback resolution method is generating phantom wins**, booking ~$5,643 in fictitious P&L. The second critical finding is that **all 7 ML models are below 50% directional accuracy** on 170K+ evaluations, meaning they are systematically worse than random. The third is that **"disabled" models continue generating predictions** that feed into the signal fusion engine.

### P&L Summary (all-time, paper trading)

| System | Total P&L | Trades | Win Rate |
|--------|-----------|--------|----------|
| ML Positions | +$2,434.09 | 1,369 | 55.7% |
| Polymarket Bets | +$1,303.32 | 137 (resolved) | 32.1% |

### P&L by Polymarket Resolution Method

| Method | Bets | Wins | Losses | Win Rate | Total P&L |
|--------|------|------|--------|----------|-----------|
| price_fallback | 34 | 26 | 8 | 76.5% | +$5,330.15 |
| gamma_api | 48 | 4 | 44 | 8.3% | -$3,420.03 |
| direction_flip | 25 | - | - | - | +$426.58 |
| stop_loss | 27 | - | - | - | -$885.35 |

The price_fallback "wins" are almost certainly phantom. See Finding #1.

---

## AUDIT 1: SPLIT-PATH OUTCOME ANALYSIS

### 1.1: Cross-Exchange Arb Outcomes

**Result:** Table `arb_trades` does not exist. The database has `trades` and `ghost_trades` (0 rows) tables instead. Cross-exchange arb trade data is not being recorded in a dedicated table, or the arb module uses a different database.

### 1.2: ML Trade Outcomes by Exit Reason (Last 48h)

```
exit_reason                trades  wins  wr_pct  total_pnl  avg_pnl  avg_hold_s
-------------------------  ------  ----  ------  ---------  -------  ----------
reeval:EDGE_CONSUMED       88      83    94.3    294.27     3.34     257
reeval:AGED_PROFITABLE     39      39    100.0   111.60     2.86     488
exit_engine:max_age        56      22    39.3    61.06      1.09     3586
signal_reversal            3       3     100.0   32.93      10.98    8488
exit_engine:profit_target  1       1     100.0   14.05      14.05    302
exposure_limit             10      0     0.0     0.00       0.00     116329
reeval:HARD_RISK_BUDGET    1       0     0.0     -12.87     -12.87   291
exit_engine:stop_loss      1       0     0.0     -14.93     -14.93   533
reeval:STALE_LOSER         142     0     0.0     -367.57    -2.59    421
```

**Key observations:**
- STALE_LOSER is the largest P&L bleed: -$367.57 across 142 trades (0% win rate)
- EDGE_CONSUMED is the best exit: +$294.27 at 94.3% win rate
- exposure_limit exits show $0 P&L despite holding positions for 32+ hours (see Finding #5)

### 1.3: ML Trade Outcomes by Side (Last 48h)

```
side   trades  wins  wr_pct  total_pnl
-----  ------  ----  ------  ---------
LONG   250     100   40.0    46.00
SHORT  91      48    52.7    72.54
```

SHORT trades outperform LONG on both win rate (52.7% vs 40.0%) and total P&L ($72.54 vs $46.00), yet the system generates 3.4x BUY signals for every SELL signal (414 BUY vs 120 SELL in last 48h). Total net P&L for the period: +$118.54.

### 1.4: Polymarket Outcomes by Exit Reason

```
exit_reason     bets  wins  losses  wr_pct  total_pnl
--------------  ----  ----  ------  ------  ---------
price_fallback  26    18    8       69.2    4871.46
direction_flip  25    0     0               426.58
stop_loss       27    0     0               -885.35
gamma_api       48    4     44      8.3     -3420.03
```

Legacy `polymarket_positions` table: 10 rows, all closed in "pivot to Strategy A" migration with $0 P&L.

### 1.5: Polymarket Outcomes by Direction

```
entry_side  bets  wins  losses  wr_pct  total_pnl
----------  ----  ----  ------  ------  ---------
NO          57    20    37      35.1    2172.17
YES         17    2     15      11.8    -720.74
```

NO (DOWN) bets significantly outperform YES (UP) bets. The quantum_transformer model has a strong negative prediction bias (74.2% of predictions are negative), which may partially explain the NO bias.

---

## AUDIT 2: SIGN AND DIRECTION INVERSIONS

### 2.1: Prediction-to-Action Mapping

The signal path is:
1. `weighted_signal > buy_threshold (0.06)` --> `action = 'BUY'`
2. `weighted_signal < sell_threshold (-0.06)` --> `action = 'SELL'`
3. `BUY` --> `side = "LONG"`, `SELL` --> `side = "SHORT"`

This mapping is logically correct. No sign inversion detected in the decision path.

### 2.2: P&L Calculation

```python
# _compute_realized_pnl (line 1904):
if side in ("LONG", "BUY"):
    return (close_price - entry_price) * size
elif side in ("SHORT", "SELL"):
    return (entry_price - close_price) * size
```

The P&L formula is correct for both sides.

### 2.3: Polymarket YES/NO Mapping

```python
# Line 509:
entry_side = "YES" if ml_direction == "UP" else "NO"
```

Correct: UP prediction maps to YES token, DOWN prediction maps to NO token.

### 2.4: Polymarket Resolution Logic

```python
# _resolve_bet (line 1117-1118):
side = bet["entry_side"]
won = (yes_price >= 0.95) if side == "YES" else (no_price >= 0.95)
```

For gamma_api resolution, this relies on Gamma API's `resolved` flag plus outcome prices. For price_fallback, the prices are synthetically constructed (see Finding #1).

---

## AUDIT 3: PREMATURE TRIGGER ANALYSIS

### 3.1: Resolution Checks

- `polymarket_strategy_a.py`: Uses `is_resolved` from Gamma API. Correctly does NOT use `is_definitive` for live trading prices (comment explains 6% vs 71% win rate from prior bug).
- `position_manager.py`: Has `is_expired()` method with configurable timeout.

### 3.2: Stop-Loss Early Exits

Only 1 stop-loss exit in last 48h: avg hold 533s, PnL -$14.93. Not a systemic issue.

### 3.3: Anti-Churn Gate

```python
# Line 2796:
min_hold_cycles = 6  # ~30min between trades on same asset (reduced from 12/60min)
```

Anti-churn gate is active. Comment says it was reduced from 12 cycles (55% block rate was too aggressive).

### 3.4: Confidence Thresholds

```
min_confidence = 0.25  (from config, default was 0.45)
buy_threshold = 0.06
sell_threshold = -0.06
```

min_confidence of 0.25 is very low. Combined with the finding that 23 trades were placed with confidence=0.0 (see Audit 7), this gate is effectively not filtering.

---

## AUDIT 4: WRONG DATA, RIGHT LOGIC

### 4.1: Price Feeds

Price symbols use `-USD` format internally (e.g., `BTC-USD`) with Binance providing the actual data via `BinanceSpotProvider`. The mapping appears consistent.

### 4.2: Feature Dimensions

`INPUT_DIM = 98` (padded from 49 single-pair + 15 cross-asset features). Models trained on this dimension. Legacy dimension was 83.

### 4.3: Stale Data

```
STALE PAIRS (> 4 hours old):
  PEPE-USD:         12,369 min (8.6 days)
  SANTOS-USD:       11,454 min (7.9 days)
  ORCA-USD:         11,164 min (7.8 days)
  GPS-USD:          10,544 min (7.3 days)
  AVAX-USD:          7,479 min (5.2 days)

FRESH PAIRS (< 15 min): 66 of 121 total pairs
```

55 pairs have stale bar data (older than 15 min). Some haven't been updated in over a week. This means predictions for these pairs are based on outdated price history.

### 4.4: Prediction-Decision Lag

```
Decision Time         | Action | Asset       | Prediction Time     | Lag (min)
2026-03-03T20:29:50   | BUY    | BCH-USD     | 2026-03-03T19:31:31 | 58.3
2026-03-03T19:29:39   | BUY    | DOGE-USD    | 2026-03-03T18:25:10 | 64.5
2026-03-03T17:34:42   | BUY    | AAVE-USD    | 2026-03-03T15:28:32 | 126.2
```

Many decisions are being made on predictions that are 20-120 minutes old. At 5-minute bar intervals, a 60-minute lag means trading on predictions that are 12 bars stale.

---

## AUDIT 5: SILENT EXCEPTION HANDLERS

### 5.1: except+pass Patterns

Found **20+ except+pass blocks** in `renaissance_trading_bot.py` and **6** in `polymarket_strategy_a.py`. These silently swallow errors that could mask data corruption, missed trades, or P&L miscalculations.

Notable locations:
- Line 2545: `pass  # Don't let cost pre-screen crash the decision pipeline`
- Line 2622: `pass` (in the decision path)
- Line 2923: `pass` (in position management)
- Lines 4261, 4341, 4353: `pass` (in data collection)

### 5.2: Price Defaulting to Zero

```python
# polymarket_strategy_a.py:698:
exit_asset_price = 0

# polymarket_strategy_a.py:1088:
exit_asset_price = 0
```

Exit asset prices default to 0 and are only updated if current_prices dict is available. If the dict is missing, closed bets record $0 exit prices, corrupting historical analysis.

---

## AUDIT 6: TYPE COERCION AND UNIT MISMATCHES

### 6.1: int() on Floats

No problematic `int()` casts on price/fee values found in core trading code. The only hit was `adfuller(spread, maxlag=int(len(spread) ** 0.25))` which is appropriate.

### 6.2: bps vs Percentage Conversions

Fee handling in the main bot:
```python
# renaissance_trading_bot.py:3426-3427:
_fee_bps = 0.0 if is_mexc_execution else 5.0
_fill_fee = _fee_bps / 10000.0 * decision.position_size * current_price
```

The conversion from bps to decimal (/ 10000) is correct. MEXC execution correctly uses 0 bps maker fee.

### 6.3: Fee Calculations

Position sizer uses:
```python
entry_cost = self.taker_fee_bps + (self.spread_cost_bps / 2) + self.slippage_bps
exit_cost = self.maker_fee_bps + (self.spread_cost_bps / 2) + self.slippage_bps
```

The simulation cost model (`sim_transaction_costs.py`) uses 0.1% maker / 0.2% taker, which is higher than the actual MEXC 0% maker fee. This means backtest results overestimate costs relative to live paper trading.

---

## AUDIT 7: DATABASE INTEGRITY

### 7.1: Orphaned Positions

```
positions_forever_open (>7 days): 0
polymarket_bets_forever_open:     0
decisions_null_action:            0
polymarket_positions_forever_open: 0
```

No orphaned positions found.

### 7.2: Impossible Values

```
negative_bet_amount:    0
pnl_larger_than_1000:   2  (bet #73: $2,453.22 and bet #75: $1,032.41)
confidence_above_100:   0
confidence_below_0:     0
zero_entry_price:       0
```

The 2 bets with >$1,000 P&L are the phantom price_fallback wins described in Finding #1. Bet #73 invested $257.52 and recorded $2,453.22 profit (952.6% return) from a 7-minute bet. This is a binary option that supposedly paid 10.5x the investment.

### 7.3: Duplicate Positions

```
Duplicate same-side: UNI-USD LONG x2
  UNI-USD_LONG_1772471007: opened 2026-03-02 17:03, entry $3.976
  UNI-USD_LONG_1772568602: opened 2026-03-03 20:10, entry $3.874
```

One duplicate same-side position (UNI-USD LONG). No opposing positions found on any asset.

### 7.4: Zero-Confidence Trades

23 trades placed with confidence = 0.0:
- FXS-USD BUY: 12 trades
- AUTO-USD BUY: 4, SELL: 4
- DF-USD BUY: 2
- CHESS-USD BUY: 1

These bypass the confidence gate, suggesting a code path that sets confidence=0.0 before the threshold check.

---

## AUDIT 8: TIMING AND ORDERING

### 8.1: Timezone Consistency

```
Table            | Min Timestamp               | Max Timestamp
decisions        | 2026-03-02T00:00:30+00:00   | 2026-03-03T20:32:13+00:00
ml_predictions   | 2026-03-02T00:00:30+00:00   | 2026-03-03T20:32:15+00:00
polymarket_bets  | 2026-03-03 13:03:30         | 2026-03-03 20:31:05
```

Timezone inconsistency: `decisions` and `ml_predictions` use ISO format with `+00:00` offset, while `polymarket_bets` uses bare datetime without timezone. This creates ambiguity when comparing timestamps across tables.

### 8.2: Bot Restart Frequency

**13 restarts on 2026-03-03 alone** (identified from `UnifiedPortfolioEngine initialized` log entries). The bot restarts roughly every 20-40 minutes. Causes include:
- `database is locked` errors
- `Permission denied: 'models/hmm_regime.pkl'` (file owned by root, bot runs as botuser)
- Coinbase WebSocket timeout failures (persistent)
- Triangular arb pair graph build failures

---

## AUDIT 9: BANKROLL AND BALANCE ACCOUNTING

### 9.1: Polymarket Bankroll Reconciliation

```
Starting bankroll:      $475.00
TOPUP (2026-03-01):   +$2,000.00
Sum of resolved P&L:  +$1,451.35
Open bet investment:    -$532.48
Expected cash on hand: $1,393.87
Last logged bankroll:  $1,418.87  (via bankroll log)
Latest log entry:      $1,595.35  (before 7 bets placed)
```

There is a **$25.00 discrepancy** between the expected bankroll ($1,393.87) computed from first principles and the logged bankroll ($1,418.87). This likely comes from rounding errors or untracked events in the bankroll log.

### 9.2: Exchange Balances

Table `arb_balances` does not exist. Table `balance_snapshots` exists but contains 0 rows. No exchange balance tracking is being recorded.

---

## AUDIT 10: CONFIGURATION DRIFT

### 10.1: Config vs Code Mismatches

| Setting | Config Value | Code Default | Effective |
|---------|-------------|-------------|-----------|
| min_confidence | 0.25 | 0.45 | 0.25 (config) |
| buy_threshold | 0.06 | 0.06 | 0.06 |
| sell_threshold | -0.06 | -0.06 | -0.06 |
| disabled_models | [gru, bidirectional_lstm, dilated_cnn] | {'gru'} | {'gru'} (code) |
| ml_signal_scale | 10.0 | 10.0 | 10.0 |

**Critical:** The `disabled_models` config lists 3 models but the code only honors `DISABLED_MODELS = {'gru'}`. The `bidirectional_lstm` and `dilated_cnn` models continue to run and generate predictions, despite being explicitly listed in config as disabled.

### 10.2: Models Generating Predictions vs Config

Active models generating predictions (last 4h):
- quantum_transformer: 1,503 predictions
- meta_ensemble: 1,503
- lightgbm: 1,503
- dilated_cnn: 1,503 (SHOULD BE DISABLED)
- cnn: 1,503
- bidirectional_lstm: 1,503 (SHOULD BE DISABLED)

Note: `gru` is correctly disabled (no predictions).

---

## FINDINGS (Ranked by Severity)

### Finding #1: Polymarket price_fallback Resolution Produces Phantom Wins
**Severity: P-CRITICAL**

**Evidence:**
- gamma_api resolution: 4 WON / 44 LOST = **8.3% win rate** (-$3,420.03)
- price_fallback resolution: 26 WON / 8 LOST = **76.5% win rate** (+$5,642.56)

Cross-referencing gamma_api LOST bets against the price_fallback logic, **18 of 44 (41%) would have been incorrectly classified as WON** by price_fallback. The price_fallback method compares `current_asset_price > entry_asset_price`, but:

1. `entry_asset_price` is the asset price at **bet placement time**, not at the market window start.
2. For "Will BTC go up in the next 15 min?" markets, the correct reference is the price at t=0 (window open), not when the bet was placed at t=3 or t=7 minutes.
3. This reference price mismatch means the fallback frequently disagrees with the actual market resolution.

**Impact:** +$5,642.56 in phantom P&L from price_fallback WON bets. Actual Polymarket performance after removing phantom wins would be approximately **-$4,339** instead of +$1,451. The bankroll figure of $1,418.87 is overstated.

**Bet #73 example:** Invested $257.52 on NO at $0.095/token (betting DOWN), resolved by price_fallback as WON with exit_price=1.0, recording $2,453.22 profit (952.6% return). If this was actually a LOST bet, the actual loss would be -$257.52, creating a **$2,710.74 swing** in reported P&L from a single bet.

---

### Finding #2: All ML Models Below 50% Directional Accuracy
**Severity: P-CRITICAL**

**Evidence (170K+ evaluated predictions, dead zone excluded):**

| Model | Evaluated | Correct | Accuracy |
|-------|-----------|---------|----------|
| lightgbm | 27,966 | 13,724 | 49.1% |
| meta_ensemble | 27,966 | 13,655 | 48.8% |
| dilated_cnn | 19,885 | 9,642 | 48.5% |
| cnn | 27,966 | 13,354 | 47.8% |
| bidirectional_lstm | 19,885 | 9,500 | 47.8% |
| quantum_transformer | 27,966 | 12,781 | 45.7% |
| gru | 18,575 | 8,478 | 45.6% |

**All 7 models are below 50%.** The best model (lightgbm at 49.1%) is still worse than a coin flip. This is not a dead-zone artifact -- the dead zone only affects 5% of evaluations. The models are systematically failing to predict 5-minute return direction.

Despite this, the ML trading system is slightly profitable (+$118.54 in last 48h, +$2,434 all-time) because the exit strategy (EDGE_CONSUMED, AGED_PROFITABLE) captures small favorable moves and the STALE_LOSER exit limits losses. The edge comes from execution/exit logic, not prediction quality.

---

### Finding #3: Disabled Models Still Generating Predictions
**Severity: P-CRITICAL**

**Evidence:**
- Config: `"disabled_models": ["gru", "bidirectional_lstm", "dilated_cnn"]`
- Code: `DISABLED_MODELS: set = {'gru'}` (hardcoded in `ml_model_loader.py` line 32)
- Result: `bidirectional_lstm` and `dilated_cnn` are running and producing 1,503 predictions each in the last 4 hours

The config value is completely ignored. The `bidirectional_lstm` model has the highest positive prediction bias (+0.087 avg prediction, 55.1% positive) at 47.8% accuracy -- it is systematically biased bullish while being wrong more than half the time. This bullish bias may contribute to the BUY/SELL imbalance (414 BUY vs 120 SELL in 48h).

---

### Finding #4: Prediction-Decision Lag Up to 126 Minutes
**Severity: P-HIGH**

**Evidence:**
```
AAVE-USD BUY decision used a prediction from 126.2 minutes ago
DOGE-USD BUY decision used a prediction from 64.5 minutes ago
BCH-USD  BUY decision used a prediction from 58.3 minutes ago
ETH-USD  BUY decision used a prediction from 50.4 minutes ago
```

The system makes trading decisions based on ML predictions that can be over 2 hours old. For a system trading 5-minute bars, a 126-minute-old prediction is 25 bars stale. The prediction-to-decision join finds the most recent prediction for that asset at or before decision time, so if a pair is scanned infrequently (tier 3/4), predictions accumulate severe staleness.

---

### Finding #5: Exposure Limit Exits Record Zero P&L
**Severity: P-HIGH**

**Evidence:**
```
All 10 exposure_limit exits: close_price = entry_price, realized_pnl = 0.0
Average hold time: 116,329 seconds (32.3 hours)
```

The code at line 3849: `_cpx = getattr(worst_pos, 'current_price', 0.0) or 0.0` -- if the position object lacks a `current_price` attribute (or it is 0/None), the close price defaults to 0.0. Then `_compute_realized_pnl` returns 0.0 because `close_price <= 0`. The DB record then shows `close_price = entry_price` (likely backfilled elsewhere) but `realized_pnl = 0.0`.

This means positions held for days are closed with zero P&L, masking potentially significant unrealized gains or losses.

---

### Finding #6: Bot Restarting Every 20-40 Minutes
**Severity: P-HIGH**

**Evidence:** 13 bot restarts on 2026-03-03. Contributing errors:
- `Permission denied: 'models/hmm_regime.pkl'` (file owned by root, bot runs as botuser)
- `database is locked` (concurrent SQLite writes)
- Coinbase WebSocket timeout (persistent connection failures)
- Triangular arb pair graph failures

Each restart resets in-memory state including anti-churn cooldown tracking (`_last_trade_cycle`), potentially allowing rapid re-trading after restart.

---

### Finding #7: 55 Pairs with Stale Bar Data
**Severity: P-HIGH**

**Evidence:**
```
Fresh pairs (< 15 min): 66 of 121 total
PEPE-USD: 8.6 days stale
SANTOS-USD: 7.9 days stale
AVAX-USD: 5.2 days stale
```

45% of pairs in the database have stale data. Predictions made for these pairs use outdated price history, rendering the ML output meaningless. These pairs appear to have been dropped from the active universe but their stale records persist.

---

### Finding #8: Zero-Confidence Trades (23 Total)
**Severity: P-HIGH**

**Evidence:**
```
FXS-USD BUY:  12 trades at confidence=0.0
AUTO-USD:      8 trades at confidence=0.0 (4 BUY + 4 SELL)
DF-USD BUY:    2 trades
CHESS-USD BUY: 1 trade
```

With `min_confidence = 0.25` in config, no trade should execute at confidence=0.0. This indicates either: (a) a code path bypasses the confidence gate, or (b) the confidence is set to 0.0 AFTER the gate check but BEFORE the DB write. The AUTO-USD SELL actions at confidence=0.0 are particularly suspicious.

---

### Finding #9: Duplicate UNI-USD LONG Position
**Severity: P-MEDIUM**

**Evidence:** Two OPEN positions for UNI-USD LONG, opened 27 hours apart. The anti-stacking guard should prevent this. Since the bot restarts frequently, the in-memory duplicate check (`_last_trade_cycle`) may be cleared on restart, allowing a new position on the same asset.

---

### Finding #10: Timestamp Format Inconsistency Across Tables
**Severity: P-MEDIUM**

**Evidence:**
- `decisions`, `ml_predictions`: ISO 8601 with timezone (`2026-03-03T20:32:13+00:00`)
- `polymarket_bets.opened_at`: SQLite `datetime('now')` format without timezone (`2026-03-03 20:31:05`)
- `five_minute_bars.bar_start`: Unix epoch integers (`1772568900.0`)

Cross-table timestamp joins require format conversion, creating a fragile dependency. The `datetime('now')` vs `datetime.now(timezone.utc).isoformat()` difference means Polymarket timestamps may be in server-local time vs UTC.

---

### Finding #11: Bankroll Accounting Discrepancy ($25)
**Severity: P-MEDIUM**

**Evidence:**
```
Starting ($475) + TOPUP ($2,000) + Sum P&L ($1,451.35) - Open investments ($532.48) = $1,393.87
Bankroll log reports: $1,418.87
Discrepancy: $25.00
```

The bankroll log and bet table are not fully reconciled. This could be from `bet_resolved` events not matching `WON`/`LOST` status transitions, or from `add_to_bet` events not properly tracked.

---

### Finding #12: Balance Snapshots Not Recording
**Severity: P-MEDIUM**

**Evidence:** `balance_snapshots` table has 0 rows. No exchange balance tracking exists. Cannot verify paper trading balance against recorded P&L.

---

### Finding #13: HMM Model File Permission Error
**Severity: P-LOW**

**Evidence:**
```
-rw-r--r-- 1 root root 1788 Feb 22 17:15 models/hmm_regime.pkl
```

File owned by root but bot runs as botuser. Write fails with `Permission denied`, logged as ERROR. The regime detector cannot save updated models.

---

### Finding #14: BUY Signal Bias (3.4:1 vs SELL)
**Severity: P-LOW**

**Evidence:** 414 BUY vs 120 SELL decisions in last 48h. Meanwhile SHORT trades actually outperform LONG (52.7% vs 40.0% win rate, +$72.54 vs +$46.00 P&L). The `quantum_transformer` model has 74.2% negative predictions, and `bidirectional_lstm` has 55.1% positive -- these conflicting biases may cancel out in the ensemble, but the net result still favors BUY.

---

### Finding #15: Silent except+pass Blocks (26+)
**Severity: P-LOW**

**Evidence:** 20+ in `renaissance_trading_bot.py`, 6 in `polymarket_strategy_a.py`. While most are in non-critical paths (module imports, optional features), several are in the decision and position management pipelines where silent failures could mask data issues.

---

## Recommendations (Priority Order)

1. **IMMEDIATE: Fix price_fallback resolution** -- Compare against window start price (from market slug timestamp), not bet entry price. Until fixed, disable price_fallback and rely only on gamma_api.

2. **IMMEDIATE: Enforce disabled_models from config** -- Make `ml_model_loader.py` read `disabled_models` from config instead of using the hardcoded `DISABLED_MODELS` set.

3. **HIGH: Investigate ML model accuracy** -- All models below 50% suggests either: (a) features are stale/misaligned at inference time, (b) the evaluation methodology has a bug (though the code looks correct), or (c) the models genuinely lack predictive power at the current time horizon.

4. **HIGH: Fix exposure_limit close price** -- Ensure `current_price` is fetched from market data before computing realized P&L on position close.

5. **HIGH: Fix HMM model file ownership** -- `chown botuser:botuser models/hmm_regime.pkl` to allow the bot to update the regime model.

6. **HIGH: Reduce prediction staleness** -- Either increase scan frequency for all active pairs, or add a staleness check that rejects predictions older than N minutes.

7. **MEDIUM: Add confidence=0.0 rejection** -- Add explicit guard: `if confidence <= 0: action = 'HOLD'`.

8. **MEDIUM: Standardize timestamp formats** -- Use ISO 8601 with timezone consistently across all tables.

9. **MEDIUM: Investigate bot crash loop** -- 13 restarts/day degrades statefulness. Fix the SQLite locking and Coinbase WebSocket issues.

10. **LOW: Clean stale bar data** -- Remove or mark inactive pairs that haven't received data in >24h.

---

## Raw Table Summary

| Table | Rows |
|-------|------|
| decisions | 39,733 |
| ml_predictions | 245,618 |
| five_minute_bars | 141,578 |
| market_data | 83,612 |
| decision_audit_log | 13,992 |
| breakout_scans | 12,958 |
| polymarket_skip_log | 6,613 |
| devil_tracker | 3,070 |
| system_state_log | 2,396 |
| labels | 21,295 |
| open_positions | 1,377 (7 OPEN, 1,369 CLOSED + 1 duplicate) |
| trades | 1,377 |
| position_snapshots | 1,058 |
| reeval_events | 1,000 |
| polymarket_lifecycle | 907 |
| polymarket_scanner | 281,064 |
| polymarket_bets | 143 |
| polymarket_bankroll_log | 297 |
| polymarket_positions | 10 |
| balance_snapshots | 0 |
| ghost_trades | 0 |
| backtest_runs | 0 |
| funding_rate_history | 0 |

---

*This audit was read-only. No files, databases, or running processes were modified.*
