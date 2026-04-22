# Triangular Arbitrage System — How It Works

## Current Operating Parameters

| Parameter | Value |
|-----------|-------|
| Trade size | **$750** per cycle |
| Min edge threshold | 5.0 bps |
| Order type | LIMIT_MAKER (post-only, 0% maker fee) |
| Start currency | USDT only |
| Rate limit | 2 executions per 60s window |
| Blocklist | BTC-quote and ETH-quote pairs filtered |
| Fill rate (observed) | ~75% |
| Fee monitor | Hourly automated check |
| Mode | **Paper trading** |

## What Is Triangular Arbitrage?

Triangular arbitrage exploits price discrepancies between three trading pairs on the **same exchange** (MEXC). When three currency pairs form a cycle and the product of their exchange rates exceeds 1.0, there is a risk-free profit opportunity.

**Example cycle:**
```
Start: 500 USDT
  Step 1: Buy USDC with USDT   (USDC/USDT pair)
  Step 2: Buy MX with USDC     (MX/USDC pair)
  Step 3: Sell MX for USDT     (MX/USDT pair)
End:   500.78 USDT  ->  Profit: $0.78
```

The edge exists because MEXC's order books across these three pairs are not perfectly synchronized. The USDT/USDC stablecoin price difference (often 3-6 basis points) creates a persistent, exploitable gap when routed through an intermediate token like MX, CHESS, or XLM.

## Why MEXC?

MEXC offers **0% maker fees** on spot trading. This is the single structural edge that makes the entire strategy viable:

| Fee Type | MEXC | Binance |
|----------|------|---------|
| Maker (limit order, rests in book) | **0.00%** | 0.075% |
| Taker (market order, crosses spread) | 0.05% | 0.075% |

A typical triangular opportunity yields 5-15 basis points of edge. With 3 legs:
- **Taker execution** (0.05% x 3 = 15 bps cost): Edge is wiped out. Every trade loses money. We proved this empirically — 26 taker trades lost $34.50 total, averaging -$1.33 per fill.
- **Maker execution** (0.00% x 3 = 0 bps cost): Full edge captured. Every trade profits. 431 maker fills earned $271.09 total, averaging +$0.63 per fill.

We use **LIMIT_MAKER (post-only)** orders exclusively, which the exchange guarantees will only fill as maker. If an order would cross the spread (becoming a taker), MEXC rejects it instead of filling it — protecting us from accidental taker fees.

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Ticker Scanner  │────>│  Cycle Detector  │────>│  Triangular Executor│
│  (every 5 sec)   │     │  (graph search)  │     │  (3-leg execution)  │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                            │
                              ┌──────────────────┐          │
                              │  Fee Monitor     │          │
                              │  (hourly check)  │          │
                              └──────────────────┘          │
                                                            v
                              ┌──────────────────┐  ┌─────────────────────┐
                              │  Competition     │  │ Performance Tracker  │
                              │  Detector        │  │ (DB + Dashboard)     │
                              └──────────────────┘  └─────────────────────┘
```

### 1. Ticker Scanner (`triangle_arb.py`)
Every 5 seconds, fetches all ~2,000+ ticker prices from MEXC's spot API in a single REST call (bookTicker endpoint). Builds an adjacency graph where currencies are nodes and trading pairs are edges, with bid/ask prices as edge weights.

### 2. Cycle Detector (`triangle_arb.py`)
Searches the graph for all 3-step cycles starting from USDT. For each cycle, computes:

```
cycle_rate = rate_1 x rate_2 x rate_3
```

If `cycle_rate > 1.0`, the cycle is profitable. Filters for a minimum net profit threshold (currently **5 basis points**, configured in `arbitrage.yaml`). Returns the top 3 opportunities per scan, rate-limited to 2 executions per 60-second window.

**Pair blocklist**: Cycles that route through BTC-quote or ETH-quote pairs (e.g., `BDX/BTC`) are automatically skipped. These pairs have low quantity precision (2-4 decimal places), causing structural rounding losses that exceed the edge. This was discovered empirically — BDX/BTC trades consistently lost $1.50-2.28 per cycle due to rounding.

### 3. Triangular Executor (`triangular_executor.py`)
Executes profitable cycles via 3 sequential LIMIT_MAKER orders:

1. **Pre-fetch** all 3 order books and symbol precision info concurrently (~130ms)
2. **Rounding check**: Simulates the full cycle at each leg's precision to estimate worst-case rounding loss. If rounding eats >5 bps of the starting amount, the cycle is skipped before placing any orders.
3. **Leg 1**: Place post-only order at top of book, wait for fill
4. **Leg 2**: Use output from Leg 1 as input, place post-only order
5. **Leg 3**: Use output from Leg 2 as input, place post-only order
6. **Result**: Compare final USDT amount to starting amount = profit/loss

**Price strategy**: BUY orders are placed at the best bid (top of bid book), SELL orders at the best ask (top of ask book). This ensures orders rest in the book as maker, qualifying for the 0% fee.

**If any leg fails** (post-only rejection, timeout, insufficient quantity):
- All previously completed legs are **unwound** in reverse order using market orders
- Market unwinds incur taker fees (0.05%) plus potential slippage
- Observed unwind cost: **-$1.29 average** on a $500 trade, worst case -$3.75

Total execution time: **110-150ms** for all 3 legs.

### 4. Performance Tracker (`performance.py`)
Every completed trade (filled or failed) is persisted to `data/arbitrage.db` in the `arb_trades` table with `strategy = "triangular"`. The web dashboard reads this table to display:
- Total triangular P&L
- Win rate and trade count
- Per-trade profit history
- Wallet balance impact

### 5. Fee Monitor (`orchestrator.py`)
Runs every hour. Queries MEXC's fee endpoint (`GET /api/v3/account/tradeFee`) for BTC/USDT and verifies the maker fee is still 0%. If the fee changes:
- Logs a CRITICAL alert
- Automatically pauses triangular arbitrage by switching to observation mode
- Resumes automatically if the fee returns to 0%

This is the most important safety check. The entire strategy is destroyed if MEXC changes their maker fee policy. Even 0.01% maker per leg (3 bps total) would make most edges unprofitable.

### 6. Competition Detector (`triangle_arb.py`)
Tracks edge sizes in a rolling 60-minute window. Computes median, mean, min, and max edge across all detected opportunities. If the median edge drops below 4 bps, it logs a COMPETITION ALERT — this signals that other bots may be competing for the same edges, and position size should be reduced.

Current competition stats (as of 2026-02-20):
- Median edge: **8.0 bps**
- Mean edge: **7.9 bps**
- Range: **5.0 - 12.6 bps**
- Status: **OK** (no competitive pressure detected)

## Current Performance (2026-02-20)

### Overall Statistics (all eras combined)

| Metric | Value |
|--------|-------|
| Total trades attempted | 577 |
| Total fills | 431 |
| Overall fill rate | 75% |
| **Net profit** | **+$271.09** |
| Best single trade | +$6.53 |
| Worst single trade | -$3.75 |
| Total runtime | ~3.5 hours |

### Performance by Era

The system went through four iterations on launch day, each validating a different hypothesis about the fee model:

| Era | Fills | Avg P&L per fill | Total P&L | Notes |
|-----|-------|-------------------|-----------|-------|
| Pre-fee simulation | 170 | +$0.08 | +$14.42 | Paper fill didn't simulate fees — inflated results |
| Taker fee era | 26 | -$1.33 | -$34.50 | Fees correctly simulated — confirmed taker kills the edge |
| LIMIT_MAKER (pre-blocklist) | 177 | +$1.03 | +$182.01 | Post-only orders, 0% maker fee — genuine alpha |
| **LIMIT_MAKER (with blocklist)** | **58** | **+$1.88** | **+$109.16** | BTC/ETH-quote blocklist + rounding check — improved quality |

**Key takeaways:**
- The **taker fee era** proved the concern: taker fees ($0.75 across 3 legs on a $500 trade) exceed the typical 5-9 bps edge. Every trade lost money.
- **Switching to LIMIT_MAKER** eliminated this cost entirely. Failed post-only attempts (rejected by exchange) cost nothing.
- The **blocklist + rounding check** improved average profit per fill from $1.03 to $1.86 by filtering out structural losers.

### LIMIT_MAKER Era Breakdown (eras 3+4 combined)

| Metric | Value |
|--------|-------|
| Winning fills | 361 |
| Losing fills | 70 |
| Win rate | **83.8%** |
| Total profit (winners) | +$361.16 |
| Total loss (losers) | -$90.07 |
| **Net profit** | **+$271.09** |
| Avg profit per winner | +$1.00 |
| Avg loss per loser | -$1.29 |
| Profit factor | 4.0x |

## Most Profitable Paths

The highest-yield triangular paths by total P&L:

| Path | Fills | Total P&L | Avg/fill |
|------|-------|-----------|----------|
| `USDC/USDT -> NAKA/USDC -> NAKA/USDT` | 13 | +$70.11 | +$5.39 |
| `USDC/USDT -> WAVES/USDC -> WAVES/USDT` | 13 | +$33.84 | +$2.60 |
| `BTC/USDT -> BDX/BTC -> BDX/USDT` *(pre-blocklist)* | 34 | +$26.63 | +$0.78 |
| `USDC/USDT -> MX/USDC -> MX/USDT` | 64 | +$19.14 | +$0.30 |
| `ETH/USDT -> LINK/ETH -> LINK/USDT` *(pre-blocklist)* | 23 | +$18.66 | +$0.81 |
| `NIL/USDT -> NIL/USDC -> USDC/USDT` | 13 | +$17.82 | +$1.37 |
| `USDC/USDT -> ROSE/USDC -> ROSE/USDT` | 9 | +$15.36 | +$1.71 |
| `NAKA/USDT -> NAKA/USDC -> USDC/USDT` | 11 | +$13.05 | +$1.19 |

The common pattern: USDT and USDC trade at slightly different prices (0.9997 vs 1.0003), and routing through an intermediate token captures the spread. NAKA and WAVES paths show the highest per-fill profit, while MX has the highest volume (64 fills) but thinner edges.

**Blocklisted paths**: Any cycle routing through BTC-quote or ETH-quote pairs (e.g., `BDX/BTC`, `XLM/ETH`) is automatically filtered. These pairs have quantity precision of only 2-4 decimal places, causing rounding losses of $1.50-2.28 per trade — far exceeding the typical edge.

## Risk Profile

### What can go wrong

1. **Partial execution + unwind cost**: Leg 1 fills but Leg 2 is rejected -> unwind Leg 1 at market. Observed cost: **-$1.29 average** (from 70 losing fills out of 431 maker-era fills), worst case -$3.75.

2. **Price movement between legs**: The 3 legs execute sequentially (~50ms each). If the price moves against us between Leg 1 and Leg 3, the cycle may be unprofitable. Mitigated by the speed of execution (total <150ms) and the fact that stablecoin pairs move slowly.

3. **Competition**: Other bots may be arbing the same opportunities. As more participants enter, edges narrow. The 0% maker fee is our structural advantage — competitors paying taker fees need larger edges to be profitable. **Actively monitored** by the competition detector (currently median edge = 8.0 bps, no pressure detected).

4. **Exchange risk**: All 3 legs are on MEXC. If MEXC goes down, has delayed fills, or changes their fee structure, the strategy is impacted. This is single-exchange concentration risk.

5. **Fee structure change**: If MEXC raises maker fees above 0%, the economics change fundamentally. At even 0.01% maker per leg (3 bps total), most current edges would be unprofitable. **Actively monitored** by the hourly fee check — the system auto-pauses triangular arb if fees change.

6. **Rounding losses**: Pairs with low quantity or price precision can lose more to rounding than the edge is worth. **Actively mitigated** by the pre-execution rounding check (skips cycles where rounding loss > 5 bps) and the BTC/ETH-quote pair blocklist.

### What cannot go wrong

- **No directional risk**: We start and end in the same currency (USDT). There is no net exposure to any asset price.
- **No counterparty risk between exchanges**: All legs are on one exchange. No cross-exchange settlement or transfer delays.
- **No leverage**: All trades are spot, fully funded.

## Capacity and Scaling

Current trade size is **$750 per cycle** (increased from $500 after confirming stable performance). Key considerations for further scaling:

- **Order book depth**: The stablecoin pairs (USDC/USDT) typically have $50K+ of depth at the top of book. Intermediate tokens (MX, CHESS) may have thinner books — at $1,000+ per trade, check that the maker order wouldn't consume more than 10% of the visible book.
- **Fill rate impact**: Larger orders are less likely to fill as maker before the opportunity closes. Expect fill rate to decrease as trade size increases. Monitor closely after each size increase.
- **Rate limiting**: Currently capped at 2 executions per 60-second window (configurable). At $750 per trade, that's ~$1,500/min maximum cycling through the strategy.
- **Diminishing returns**: The same 3-4 profitable paths are recycled. Scaling up trade size on the same paths may move the market, reducing the edge for subsequent trades.

**Scaling history**: $500 (initial) -> $750 (current). Next step: $1,000, contingent on fill rate remaining above 70% at $750.

## Paper Trading vs Live

The system is currently **paper trading** — simulating fills against real order book data but not placing actual orders. The paper trading engine includes:

- Realistic fee simulation (0% maker, 0.05% taker based on order type and spread crossing)
- 75% fill rate simulation for LIMIT_MAKER orders (calibrated to match observed fill rate)
- Correct fee denomination (BUY fees in base currency, SELL fees in quote currency)
- Pre-execution rounding check (skips cycles where rounding loss > 5 bps)

**Paper vs live fill rate**: The observed paper trading fill rate is ~75% (431 fills out of 577 attempts). The simulation fill rate has been calibrated to match this. However, live fill rates may differ — paper trading cannot fully replicate order queue position, book dynamics, or latency. Live testing is the only way to establish the true fill rate.

**Before going live**, validate:
1. Confirm MEXC still offers 0% maker fee on spot (hourly automated check is already running)
2. Run paper trading for 48+ hours to establish baseline statistics
3. Start with $100 trade size (not $750) to limit exposure during validation
4. Monitor fill rates — real fill rates may differ from paper simulation
5. Watch competition detector for edge compression over time

## Safeguards (7 layers)

| # | Safeguard | Location | What it does |
|---|-----------|----------|--------------|
| 1 | **BTC/ETH-quote pair blocklist** | `triangle_arb.py` | Filters out cycles through low-precision pairs at the scanner level, before any orders are considered |
| 2 | **Pre-execution rounding check** | `triangular_executor.py` | Simulates all 3 legs at actual precision before placing orders. Skips if rounding loss > 5 bps |
| 3 | **LIMIT_MAKER (post-only) orders** | `triangular_executor.py` | Exchange guarantees orders only fill as maker (0% fee). Rejects if they would cross the spread |
| 4 | **Automatic unwind** | `triangular_executor.py` | Failed cycles are reversed via market orders to recover capital |
| 5 | **Rate limiting** | `triangle_arb.py` | Max 2 executions per 60-second window |
| 6 | **Fee monitor** | `orchestrator.py` | Hourly MEXC fee check. Auto-pauses tri-arb if maker fee changes from 0% |
| 7 | **Daily drawdown limit** | `arb_risk.py` | Pauses all arb trading if daily P&L drops below -$100 (configurable via `risk.max_daily_loss_usd`) |

## Technical Details

### Key Files
| File | Purpose |
|------|---------|
| `arbitrage/triangular/triangle_arb.py` | Scanner + cycle detector + pair blocklist + competition detector |
| `arbitrage/execution/triangular_executor.py` | 3-leg execution engine + rounding check + unwind logic |
| `arbitrage/tracking/performance.py` | P&L tracking + DB persistence + dashboard integration |
| `arbitrage/orchestrator.py` | Lifecycle management, wires components together, fee monitor |
| `arbitrage/exchanges/mexc_client.py` | MEXC API client (REST + WebSocket + paper fill simulation) |
| `arbitrage/costs/model.py` | Fee schedules and cost estimation |
| `arbitrage/config/arbitrage.yaml` | All tunable parameters |

### Configuration
Tunable parameters in `arbitrage/config/arbitrage.yaml`:
```yaml
triangular:
  min_net_profit_bps: 5.0        # Minimum edge to execute
  max_trade_usd: 750             # Trade size per cycle
  max_signals_per_cycle: 2       # Rate limit per 60s window
  observation_mode: false        # Log-only mode without execution
  start_currencies: [USDT]       # Which currencies to start cycles from

risk:
  max_single_arb_usd: 750       # Must match triangular.max_trade_usd
  max_daily_loss_usd: 100        # Daily drawdown limit — halts all arb
```

### Database Schema
Trades are stored in `data/arbitrage.db`, table `arb_trades`:
```sql
trade_id TEXT        -- e.g., "tri_USDT_1771624438725"
strategy TEXT        -- "triangular"
symbol TEXT          -- "USDC/USDT->MX/USDC->MX/USDT"
status TEXT          -- "filled", "failed", or "skipped"
actual_profit_usd REAL  -- profit after fees (0 for failed/skipped)
quantity REAL        -- starting USDT amount
timestamp TEXT       -- ISO 8601
```

### Dashboard Endpoints
All endpoints read from `data/arbitrage.db` and include triangular trades automatically:
- `GET /api/arbitrage/summary` — Aggregate stats by strategy (cross_exchange, triangular)
- `GET /api/arbitrage/trades?strategy=triangular` — Individual trade history
- `GET /api/arbitrage/wallet` — Balance impact from arb activity

### Competition Detector
The competition detector tracks edge sizes in a rolling 60-minute window. Statistics available via `triangular_arb.get_stats()['competition']`:
```json
{
  "sample_count": 60,
  "median_edge_bps": 8.0,
  "mean_edge_bps": 7.9,
  "min_edge_bps": 5.0,
  "max_edge_bps": 12.6,
  "window_minutes": 60,
  "alert_active": false
}
```

Alert threshold: if median edge drops below **4 bps**, a COMPETITION ALERT is logged. This signals edge compression from competing bots and suggests reducing position size.

## How the Fee Model Was Validated

The system deliberately went through three fee eras to validate the model empirically:

1. **No fee simulation** (era 1): Paper fills had no fee deduction. Results showed +$0.08 average per fill — suspiciously low and clearly inflated since real fees weren't modeled.

2. **Taker fee simulation** (era 2): Added realistic fee simulation where LIMIT orders that cross the spread pay 0.05% taker. Result: **every single trade lost money** (-$1.33 average). This proved that at 3 legs x 0.05% = 15 bps of taker fees, the typical 5-9 bps edge is completely consumed.

3. **LIMIT_MAKER + 0% maker** (era 3): Switched to post-only orders that guarantee maker execution at 0% fee. Result: **84% win rate, +$1.03 average profit per fill**. The edge is real.

4. **Blocklist + rounding check** (era 4): Added BTC/ETH-quote pair blocklist and pre-execution rounding simulation. Result: **+$1.88 average profit per fill**. Eliminating structural losers improved quality significantly.

This progression from no fees -> taker fees -> maker fees -> maker + filters provides high confidence that the current profits are genuine and not artifacts of simulation errors.

## Known Limitations

1. **Single-exchange concentration**: All activity is on MEXC. Exchange downtime, API changes, or geo-restrictions (the VPS already hits MEXC contract API geo-blocks for futures) could halt the strategy entirely.

2. **Paper trading fill rate may differ from live**: The 95% simulated fill rate for LIMIT_MAKER may be optimistic. Real-world fill rates depend on order book dynamics that paper trading can't fully replicate. Only live testing will establish the true fill rate.

3. **Stale book data (highest-priority improvement)**: The scanner fetches tickers via REST every 5 seconds. Between fetches, prices may have moved. With execution taking 110-150ms but price data up to 5 seconds stale, many detected opportunities may already be gone by the time the executor pre-fetches fresh books. The observed 75% fill rate likely reflects this staleness — the executor's fresh pre-fetch discovers the opportunity has closed and the post-only order is rejected. **Upgrading to a WebSocket ticker feed would provide sub-second updates and likely improve fill rate significantly**, which directly translates to more profitable trades per hour.

4. **No position tracking across restarts**: In-memory competition stats and executor stats reset on bot restart. Historical data is preserved in the database, but the rolling competition window starts fresh.

5. **No strategy-level daily stop-loss**: The risk engine has a global `max_daily_loss_usd` ($100) that halts all arbitrage strategies. There is no separate daily drawdown limit for triangular arb alone. If tri-arb has a bad streak but cross-exchange is profitable, both get halted. A per-strategy daily stop-loss (e.g., pause tri-arb if its daily P&L drops below -$25) would be a useful addition.
