# Math Verification Report

**Date:** 2026-03-06
**Auditor:** Claude Code (autonomous audit)
**Truth Document:** STRATEGY_MATH_TRUTH_DOCUMENT.md
**Test File:** tests/test_math_verification.py — **63/63 PASSED**

---

## Summary

| Category | Formulas | MATCH | MISMATCH | NOT FOUND |
|----------|----------|-------|----------|-----------|
| Straddle (1.x) | 9 | 8 | 1 | 0 |
| Token Spray (2.x) | 8 | 6 | 2 | 0 |
| Polymarket (3.x) | 5 | 2 | 2 | 1 |
| Arbitrage (4.x) | 2 | 2 | 0 | 0 |
| Volatility (5.x) | 2 | 1 | 1 | 0 |
| Shared (S.x) | 3 | 3 | 0 | 0 |
| **TOTAL** | **29** | **22** | **6** | **1** |

**6 MISMATCHES + 1 GHOST (not found in code)**

---

## STRATEGY 1: BTC STRADDLE

### Formula 1.1 — Leg PnL (bps)
- **File:** `straddle_engine.py:269,271`
- **Document:** `LONG: (current - entry) / entry * 10000` / `SHORT: (entry - current) / entry * 10000`
- **Code:**
  ```python
  # Line 269
  return (current_price - leg.entry_price) / leg.entry_price * 10000
  # Line 271
  return (leg.entry_price - current_price) / leg.entry_price * 10000
  ```
- **Status:** MATCH
- **Test:** PASS (4 test cases)

### Formula 1.2 — Leg PnL (USD)
- **File:** `straddle_engine.py:341-342`
- **Document:** `pnl_usd = pnl_bps / 10000 * leg_size_usd` (NOT * 2)
- **Code:**
  ```python
  straddle.long_leg.pnl_usd = straddle.long_leg.pnl_bps / 10000 * straddle.size_usd
  straddle.short_leg.pnl_usd = straddle.short_leg.pnl_bps / 10000 * straddle.size_usd
  ```
- **Status:** MATCH
- **Test:** PASS (3 test cases)

### Formula 1.3 — Net Straddle PnL
- **File:** `straddle_engine.py:337-338`
- **Document:** `net_pnl_usd = net_pnl_bps / 10000 * leg_size_usd` (NOT * 2)
- **Code:**
  ```python
  net_pnl_bps = straddle.long_leg.pnl_bps + straddle.short_leg.pnl_bps
  net_pnl_usd = net_pnl_bps / 10000 * straddle.size_usd
  ```
- **Status:** MATCH (was `* 2` before today's fix, now correct)
- **Test:** PASS (4 test cases)

### Formula 1.4 — Peak PnL Tracking
- **File:** `straddle_engine.py:30,281-282`
- **Document:** `peak = max(peak, current_pnl_bps)`, init to 0.0
- **Code:**
  ```python
  # Line 30: peak_favorable_bps: float = 0.0
  # Lines 281-282:
  if pnl_bps > leg.peak_favorable_bps:
      leg.peak_favorable_bps = pnl_bps
  ```
- **Status:** MATCH
- **Test:** PASS (2 test cases)

### Formula 1.5 — Trailing Stop
- **File:** `straddle_engine.py:294-309`
- **Document:** triggers when `trail_active AND (peak - current) >= trail_distance_bps`. Exit at current, not peak.
- **Code:**
  ```python
  # Line 294-296: activation
  if not leg.trail_active and pnl_bps >= self.trail_activation_bps:
      leg.trail_active = True
  # Lines 303-309: trigger
  if leg.trail_active and (leg.trail_peak_bps - pnl_bps) >= self.trail_distance_bps:
      leg.pnl_bps = pnl_bps       # Exit at current
      leg.exit_price = current_price
  ```
- **Status:** MATCH
- **Test:** PASS (2 test cases)

### Formula 1.6 — Hard Stop
- **File:** `straddle_engine.py:285`
- **Document:** `pnl_bps <= -stop_loss_bps`
- **Code:**
  ```python
  if pnl_bps <= -self.stop_loss_bps:
  ```
- **Status:** MATCH
- **Test:** PASS

### Formula 1.7 — Timeout Classification
- **File:** `straddle_engine.py:311-318`
- **Document:** Classify as `timeout_flat` / `timeout_profitable` / `timeout_loss` based on peak vs min_move_bps
- **Code:**
  ```python
  if age >= self.max_hold_seconds:
      leg.exit_reason = "timeout"   # <-- Single generic label
  ```
- **Status:** MISMATCH
- **Notes:** Code uses a single `"timeout"` exit reason. Does NOT classify into `timeout_flat` / `timeout_profitable` / `timeout_loss` based on peak vs min_move_bps. The classification logic described in the truth document is completely absent.

### Formula 1.8 — Capital Deployed
- **File:** `straddle_engine.py:199`
- **Document:** `open_straddles * leg_size * 2`
- **Code:**
  ```python
  deployed = sum(s.size_usd * 2 for s in self.open_straddles)
  ```
- **Status:** MATCH
- **Test:** PASS

### Formula 1.9 — Daily Loss Circuit Breaker
- **File:** `straddle_engine.py:354-355,205-206`
- **Document:** Accumulates `net_pnl_usd`, halts when `daily_loss >= max_daily_loss`
- **Code:**
  ```python
  # Line 354-355: accumulation
  if net_pnl_usd < 0:
      self._daily_loss_usd += abs(net_pnl_usd)
  # Line 205-206: halt
  if self._daily_loss_usd >= self.MAX_DAILY_LOSS_USD:
  ```
- **Status:** MATCH
- **Test:** PASS

---

## STRATEGY 2: TOKEN SPRAY

### Formula 2.1 — Token PnL (bps)
- **File:** `token_spray_engine.py:765-766`
- **Document:** LONG = `(current - entry) / entry * 10000`, SHORT = `(entry - current) / entry * 10000`
- **Code:**
  ```python
  move = (current_price - token.entry_price) / token.entry_price * 10000
  return move if token.side == "LONG" else -move
  ```
- **Status:** MATCH
- **Test:** PASS (2 test cases)

### Formula 2.2 — Token PnL (USD)
- **File:** `token_spray_engine.py:777`
- **Document:** `pnl_usd = pnl_bps / 10000 * token_size_usd`
- **Code:**
  ```python
  pnl_usd = pnl_bps / 10000 * token.size_usd
  ```
- **Status:** MATCH
- **Test:** PASS (2 test cases)

### Formula 2.3 — Vol-Scaled Token Size
- **File:** `token_spray_engine.py:165-167,292-293`
- **Document:** `size = base * vol_scale` with scales `{low: 1.0, medium: 0.7, high: 0.4, extreme: 0.2}`
- **Code:**
  ```python
  # Line 165-167: default config
  self.vol_scaling = {"low": 1.0, "medium": 0.7, "high": 0.4}
  # Line 293: application
  vol_scale = self.vol_scaling.get(vol_regime, 1.0)
  ```
- **Status:** MISMATCH
- **Notes:** The `"extreme": 0.2` tier is missing from the default config. If `_calc_vol_regime()` returns `"extreme"`, the `.get(..., 1.0)` fallback returns **1.0** (full size) instead of **0.2** (20% size). This silently removes drawdown protection during extreme volatility.
- **Additional:** `_calc_vol_regime()` (line 848) only returns `"low"`, `"medium"`, or `"high"` — never `"extreme"`. So the `extreme` case can never fire. Either the vol regime classifier needs an extreme tier, or the truth document describes unimplemented functionality.

### Formula 2.4 — Budget Tracking
- **File:** `token_spray_engine.py:347,789,296`
- **Document:** `budget_deployed += actual size_usd` at entry, `-= stored size` at exit, gate uses actual size
- **Code:**
  ```python
  # Line 347: entry
  self.budget_deployed_usd += size_usd
  # Line 789: exit
  self.budget_deployed_usd = max(0.0, self.budget_deployed_usd - token.size_usd)
  # Line 296: gate
  if self.budget_deployed_usd + size_usd > self.max_budget_usd: return None
  ```
- **Status:** MATCH
- **Test:** PASS (2 test cases)

### Formula 2.5 — Trailing Stop
- **File:** `token_spray_engine.py:644-656`
- **Document:** Same as straddle 1.5
- **Code:**
  ```python
  # Line 644-647: activation
  if not token.trail_active and pnl_bps >= token.trail_activation_bps:
      if age >= token.min_hold_for_trail:
          token.trail_active = True
  # Line 656: trigger
  elif token.trail_active and pnl_bps <= (token.peak_pnl_bps - token.trail_distance_bps):
  ```
- **Status:** MATCH

### Formula 2.6 — Hard Stop
- **File:** `token_spray_engine.py:652`
- **Document:** `pnl_bps <= -stop_loss_bps`
- **Code:**
  ```python
  if pnl_bps <= -token.stop_loss_bps:
  ```
- **Status:** MATCH

### Formula 2.7 — Spread Gate
- **File:** `token_spray_engine.py:261-267,183`
- **Document:** `estimated_spread_bps < stop_loss_bps * max_spread_to_stop_ratio` (default 0.5)
- **Code:**
  ```python
  # Line 183: absolute threshold
  self.max_entry_spread_bps = config.get("max_entry_spread_bps", 4.0)
  # Lines 261-267: gate
  if _spread_bps > self.max_entry_spread_bps:
      return None
  ```
- **Status:** MISMATCH
- **Notes:** Code uses an **absolute threshold** (4.0 bps hard-coded default), NOT a ratio relative to `stop_loss_bps`. Truth document says the gate should be `spread < stop_loss * 0.5`, which is proportional (e.g., 7bp stop → max 3.5bp spread; 12bp stop → max 6bp spread). The code treats all pairs the same regardless of their stop_loss_bps setting.

### Formula 2.8 — Direction Rule Classification
- **File:** `token_spray_engine.py:288,330,819-846`
- **Document:** Stored from config at entry, NOT inferred after the fact
- **Code:** Direction rule determined at entry (line 288), stored in token (line 330), never recalculated.
- **Status:** MATCH

---

## STRATEGY 3: POLYMARKET

### Formula 3.1 — Kelly Criterion
- **File:** `polymarket_strategy_a.py:847,854-857`
- **Document:** `kelly = (p * b - q) / (b + eps)` where `b = (1/token_cost) - 1`. If kelly <= 0, return 0.0 IMMEDIATELY.
- **Code:**
  ```python
  b = (1.0 - token_cost) / (token_cost + 1e-10)  # algebraically = (1/cost)-1
  kelly = (p * b - q) / (b + 1e-10)
  kelly = max(0.0, kelly)
  if kelly <= 0:
      return 0.0
  ```
- **Status:** MATCH
- **Notes:** `b` formula uses equivalent algebraic form `(1-cost)/cost` instead of `(1/cost)-1`. Mathematically identical. Early return on kelly <= 0 was added today.
- **Test:** PASS (4 test cases including algebraic equivalence check)

### Formula 3.2 — Fractional Kelly Sizing
- **File:** `polymarket_strategy_a.py:860,863,866,109-110`
- **Document:** `frac_kelly = kelly * fraction`, `bet = sizing_bankroll * frac_kelly`, MAX_BET=$50, MAX_SIZING=$1000
- **Code:**
  ```python
  frac_kelly = kelly * kelly_fraction           # Line 860
  sizing_bankroll = min(self.bankroll, self.MAX_SIZING_BANKROLL)  # Line 863
  bet = sizing_bankroll * frac_kelly             # Line 866
  MAX_BET_USD = 50.0                             # Line 109
  MAX_SIZING_BANKROLL = 1000.0                   # Line 110
  ```
- **Status:** MATCH
- **Test:** PASS (3 test cases)

### Formula 3.3 — Bet Resolution
- **File:** `polymarket_strategy_a.py:1186-1216,1261-1263`
- **Document:** ONLY `gamma_api` valid. `price_fallback` PERMANENTLY DISABLED. WON = `tokens * 1.0 - invested`. LOST = `-invested`.
- **Code (WON/LOST math):**
  ```python
  exit_price = 1.0 if won else 0.0
  pnl = round((exit_price * bet["total_tokens"]) - bet["total_invested"], 2)
  ```
- **Code (price_fallback — STILL ACTIVE):**
  ```python
  # Lines 1200-1216: price-based fallback (>2 min past deadline)
  if seconds_past > 120 and current_prices:
      went_up = current_asset_price > window_start_price
      yes_price = 1.0 if went_up else 0.0
      self._resolve_bet(conn, bet, yes_price, no_price, "price_fallback", current_prices)
  ```
- **Status:** MISMATCH
- **Notes:**
  1. WON/LOST P&L formulas are **correct** (lines 1261-1263)
  2. `price_fallback` resolution path is **ACTIVE** (lines 1200-1216) but truth document says it must be **PERMANENTLY DISABLED**
  3. `force_expired` path also exists (lines 1218-1228) — refunds investment, sets pnl=0
  4. The `_resolve_bet()` method does NOT discriminate by source — all sources (gamma_api, price_fallback) trigger the same bankroll update

### Formula 3.4 — Bankroll Updates
- **File:** `polymarket_strategy_a.py:913,1281-1284`
- **Document:** WON via gamma_api: `bankroll += pnl`. WON via other: unchanged. LOST: `bankroll -= invested`.
- **Code:**
  ```python
  # At placement (line 913):
  self.bankroll -= bet_amount

  # At resolution (lines 1281-1284):
  if won:
      self.bankroll += bet["total_invested"] + pnl  # refund + profit
  if not won:
      self._last_loss_time = time.time()  # no bankroll change
  ```
- **Status:** MISMATCH (partial)
- **Notes:**
  - **Net arithmetic is correct**: placement deducts `invested`, WON adds back `invested + pnl` = net `+pnl`. LOST leaves bankroll at `initial - invested` = net `-invested`. The truth document describes the same net effect from a different accounting perspective.
  - **Source discrimination is missing**: The code applies the same WON logic for ALL resolution sources (gamma_api, price_fallback, force_expired). Truth document says only `gamma_api` wins should update bankroll.

### Formula 3.5 — Odds Filter
- **File:** NOT FOUND
- **Document:** BLOCKED if `token_cost < 0.15` or `token_cost > 0.85`
- **Code:** No `MIN_ACCEPTABLE_ODDS`, `MAX_ACCEPTABLE_ODDS`, or any token_cost bounds check exists in the entry decision path (lines 575-620).
- **Status:** NOT FOUND (GHOST CALCULATION)
- **Notes:** This filter was never implemented. The system can place bets at any token cost, including extreme prices like 0.03 or 0.97. At extreme token costs, the payout ratio is either enormous (but nearly impossible to win) or tiny (barely worth the execution risk).
- **Test:** PASS (test verifies the math; code lacks the gate)

---

## STRATEGY 4: ARBITRAGE

### Formula 4.1 — Cross-Exchange Spread
- **File:** `arbitrage/orderbook/unified_book.py:82,88,164-167`
- **Document:** `spread_bps = (bid_high - ask_low) / ask_low * 10000`. Must check BOTH directions.
- **Code:**
  ```python
  spread_1_bps = ((binance_bid - mexc_ask) / mexc_ask) * 10000
  spread_2_bps = ((mexc_bid - binance_ask) / binance_ask) * 10000
  ```
- **Status:** MATCH
- **Test:** PASS (3 test cases)

### Formula 4.2 — Arb PnL
- **File:** `arbitrage/detector/cross_exchange.py:175`, `arbitrage/execution/realistic_fill.py:51-88`
- **Document:** `gross = spread_bps/10000 * size`, `net = gross - fees - rebalancing($0.01)`
- **Code:**
  ```python
  expected_profit = notional * (net_spread / 10000)  # Line 175
  rebalance_cost_per_trade = Decimal('0.01')          # Line 51
  realistic_profit = paper_profit_usd - total_cost    # Line 88
  ```
- **Status:** MATCH
- **Notes:** Fees are deducted by exchange clients in paper fill. Realistic fill adds only costs not modeled in paper fills (rebalancing, withdrawal amortization, adverse price movement). Architecture is sound — no double-counting.
- **Test:** PASS

---

## STRATEGY 5: VOLATILITY MODEL

### Formula 5.1 — Prediction Output
- **File:** `ml_model_loader.py:2300,2304,2308-2330`
- **Document:** `predicted_bps = expm1(log_magnitude)`. Regime by percentiles from meta JSON.
- **Code:**
  ```python
  pred_log = float(model.predict(features)[0])         # Line 2300
  magnitude_bps = float(np.expm1(max(pred_log, 0.0)))  # Line 2304
  # Lines 2308-2330: regime classification
  if pred_log < p25: regime = 'dead_zone'
  elif pred_log < p75: regime = 'normal'
  elif pred_log < p90: regime = 'active'
  else: regime = 'explosive'
  ```
- **Status:** MATCH
- **Test:** PASS (3 test cases)

### Formula 5.2 — Dead Zone Gate
- **File:** `ml_model_loader.py:2317-2318`
- **Document:** Block when `predicted_bps < min_predicted_vol_bps (12.0)` OR `regime == 'dead_zone'`
- **Code:**
  ```python
  if pred_log < p25:
      regime = 'dead_zone'
      vol_multiplier = 0.0  # Blocks via zero multiplier
  ```
- **Status:** MISMATCH (threshold concern)
- **Notes:** The dead_zone gate uses percentile p25 from training data metadata. The default p25 is 2.0 (log space), which corresponds to `expm1(2.0) = 6.39 bps`. The truth document specifies `min_predicted_vol_bps = 12.0`. If the meta JSON has p25=2.0 (default), the gate triggers at ~6.4 bps instead of 12.0 bps — a significantly looser threshold that would allow straddles in low-vol conditions the truth document intends to block.
- **Test:** PASS (test verifies the conceptual math)

---

## SHARED FORMULAS

### Formula S.1 — Win Rate
- **File:** `dashboard/db_queries.py:248-300`
- **Document:** `count(pnl > 0) / count(all_resolved)`. Zero PnL is NOT a win.
- **Code:** SQL `CASE WHEN ... > 0 THEN 1 ELSE 0 END` correctly uses `> 0` (not `>= 0`).
- **Status:** MATCH
- **Test:** PASS (2 test cases)

### Formula S.2 — Average Win / Average Loss
- **File:** `dashboard/db_queries.py:263-278`
- **Document:** `mean(pnl WHERE pnl > 0)` / `mean(pnl WHERE pnl < 0)`. Excludes zeros.
- **Code:** SQL `AVG(CASE WHEN ... > 0 THEN ...)` and `AVG(CASE WHEN ... < 0 THEN ...)` correctly excludes zeros.
- **Status:** MATCH
- **Test:** PASS (2 test cases)

### Formula S.3 — Cumulative PnL
- **File:** `dashboard/db_queries.py:316-337`
- **Document:** Sum of all closed trade `pnl_usd`. Must not include open positions.
- **Code:** SQL `WHERE status = 'CLOSED'` correctly excludes open positions.
- **Status:** MATCH
- **Test:** PASS (2 test cases)

---

## MISMATCHES — DETAILED SUMMARY

### CRITICAL

| # | Formula | File:Line | Issue | Impact |
|---|---------|-----------|-------|--------|
| 1 | 3.3 | `polymarket_strategy_a.py:1200-1216` | `price_fallback` resolution still ACTIVE | Can resolve bets incorrectly (comparing crypto prices instead of Polymarket settlement). Was source of $48M phantom bankroll historically. |
| 2 | 3.5 | NOT FOUND | Odds filter `[0.15, 0.85]` never implemented | System can bet at extreme token costs (0.03, 0.97) where edge is illusory and execution risk is maximal. |

### HIGH

| # | Formula | File:Line | Issue | Impact |
|---|---------|-----------|-------|--------|
| 3 | 3.4 | `polymarket_strategy_a.py:1281-1282` | Bankroll update doesn't discriminate by resolution source | `price_fallback` wins inflate bankroll the same as verified `gamma_api` wins. Combined with #1, phantom wins can inflate sizing. |
| 4 | 2.3 | `token_spray_engine.py:165-167` | Missing `"extreme": 0.2` vol scaling tier | If vol regime ever returns "extreme", fallback gives 1.0x sizing (100%) instead of 0.2x (20%). Currently mitigated because `_calc_vol_regime()` never returns "extreme". |

### MEDIUM

| # | Formula | File:Line | Issue | Impact |
|---|---------|-----------|-------|--------|
| 5 | 1.7 | `straddle_engine.py:311-318` | Timeout uses generic `"timeout"` label, no flat/profitable/loss classification | Can't distinguish timeout outcomes in analytics. Reduces ability to tune timeout behavior. |
| 6 | 2.7 | `token_spray_engine.py:183,261-267` | Spread gate uses absolute 4.0bps threshold, not proportional to stop_loss | Pairs with tight stops (7bp) accept spreads up to 4bp (57% of stop consumed by spread). Pairs with wide stops (12bp) are over-restricted. |
| 7 | 5.2 | `ml_model_loader.py:2317` | Dead zone gate threshold may be ~6.4bps instead of 12.0bps | Depends on training data metadata p25 value. If p25=2.0 (default), straddles open in conditions the truth document intends to block. |

---

## GHOST CALCULATIONS

Formulas defined in the truth document but **completely absent from code**:

| Formula | Expected Location | Description |
|---------|-------------------|-------------|
| 3.5 Odds Filter | `polymarket_strategy_a.py` entry path | Block bets when `token_cost < 0.15` or `> 0.85` |

---

## VERIFICATION COMPLETE

- **63 unit tests** written and passing in `tests/test_math_verification.py`
- **22 of 29 formulas** match the truth document
- **6 mismatches** identified (2 critical, 2 high, 3 medium)
- **1 ghost calculation** identified (odds filter never implemented)
- **No fixes applied** — report only per spec instructions
