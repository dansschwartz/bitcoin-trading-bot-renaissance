# Triangular Arbitrage System — How It Works

## What Is Triangular Arbitrage?

Triangular arbitrage exploits price discrepancies between three trading pairs on the **same exchange** (MEXC). When three currency pairs form a cycle and the product of their exchange rates exceeds 1.0, there is a risk-free profit opportunity.

**Example cycle:**
```
Start: 500 USDT
  Step 1: Buy USDC with USDT   (USDC/USDT pair)
  Step 2: Buy MX with USDC     (MX/USDC pair)
  Step 3: Sell MX for USDT     (MX/USDT pair)
End:   500.78 USDT  →  Profit: $0.78
```

The edge exists because MEXC's order books across these three pairs are not perfectly synchronized. The USDT/USDC stablecoin price difference (often 3-6 basis points) creates a persistent, exploitable gap when routed through an intermediate token like MX, CHESS, or XLM.

## Why MEXC?

MEXC offers **0% maker fees** on spot trading. This is the structural edge that makes the strategy viable:

| Fee Type | MEXC | Binance |
|----------|------|---------|
| Maker (limit order, rests in book) | **0.00%** | 0.075% |
| Taker (market order, crosses spread) | 0.05% | 0.075% |

A typical triangular opportunity yields 5-15 basis points of edge. With 3 legs:
- **Taker execution** (0.05% x 3 = 15 bps cost): Edge is wiped out. Every trade loses money.
- **Maker execution** (0.00% x 3 = 0 bps cost): Full edge captured. Every trade profits.

We use **LIMIT_MAKER (post-only)** orders exclusively, which the exchange guarantees will only fill as maker. If an order would cross the spread (becoming a taker), MEXC rejects it instead of filling it — protecting us from accidental taker fees.

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Ticker Scanner  │────▶│  Cycle Detector  │────▶│  Triangular Executor│
│  (every 5 sec)   │     │  (graph search)  │     │  (3-leg execution)  │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                            │
                                                            ▼
                                                 ┌─────────────────────┐
                                                 │ Performance Tracker  │
                                                 │ (DB + Dashboard)     │
                                                 └─────────────────────┘
```

### 1. Ticker Scanner
Every 5 seconds, fetches all ~2,000+ ticker prices from MEXC's spot API in a single REST call (bookTicker endpoint). Builds an adjacency graph where currencies are nodes and trading pairs are edges, with bid/ask prices as edge weights.

### 2. Cycle Detector
Searches the graph for all 3-step cycles starting from USDT. For each cycle, computes:

```
cycle_rate = rate_1 x rate_2 x rate_3
```

If `cycle_rate > 1.0`, the cycle is profitable. Filters for a minimum net profit threshold (currently **5 basis points**, configured in `arbitrage.yaml`). Returns the top 3 opportunities per scan, rate-limited to 2 executions per 60-second window.

**Pair blocklist**: Cycles that route through BTC-quote or ETH-quote pairs (e.g., `BDX/BTC`) are automatically skipped. These pairs have low quantity precision, causing structural rounding losses that exceed the edge. This was discovered empirically — BDX/BTC trades consistently lost $1.50-2.28 per cycle due to rounding.

### 3. Triangular Executor
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
- Observed unwind cost: **-$0.52 average** on a $500 trade, with worst cases up to -$2.28 (primarily on low-precision pairs, now blocklisted)

Total execution time: **110-150ms** for all 3 legs.

### 4. Performance Tracker
Every completed trade (filled or failed) is persisted to `data/arbitrage.db` in the `arb_trades` table with strategy = `"triangular"`. The web dashboard reads this table to display:
- Total triangular P&L
- Win rate and trade count
- Per-trade profit history
- Wallet balance impact

## Current Performance (2026-02-20)

These numbers are from the **LIMIT_MAKER era only** (post-only orders with 0% maker fee):

| Metric | Value |
|--------|-------|
| Trades attempted | 225 |
| Filled (successful) | 154 |
| Failed (rejected/unwound) | 71 |
| Fill rate | 68% |
| **Net profit (filled trades)** | **+$149.52** |
| Avg profit per winning fill | **+$1.19** |
| Avg loss per losing fill | -$0.52 |
| Win rate (filled trades) | 87% |
| Best single trade | +$6.53 |
| Worst single trade | -$2.28 |
| Execution speed | 110-150ms |

### Performance by Era

The system went through three iterations on launch day, validating the fee model:

| Era | Fills | Avg P&L per fill | Total P&L | Notes |
|-----|-------|-------------------|-----------|-------|
| Pre-fee simulation | 170 | +$0.08 | +$14.42 | Paper fill didn't simulate fees — inflated results |
| Taker fee era | 26 | -$1.33 | -$34.50 | Fees correctly simulated — confirmed taker kills the edge |
| **LIMIT_MAKER era** | **154** | **+$0.92** | **+$149.52** | Post-only orders, 0% maker fee — genuine alpha |

The taker fee era proved the concern was valid: taker fees ($0.75 across 3 legs on a $500 trade) exceed the typical 5-9 bps edge. Switching to LIMIT_MAKER eliminated this cost entirely. Failed post-only attempts (rejected by exchange) cost nothing — the order simply doesn't execute. The only cost comes from partial execution where Leg 1 fills but Leg 2 is rejected, requiring a market-order unwind of Leg 1.

## Most Profitable Paths

The highest-yield triangular paths consistently involve stablecoin pairs:

- `USDC/USDT -> MX/USDC -> MX/USDT` (~15 bps edge)
- `USDC/USDT -> CHESS/USDC -> CHESS/USDT` (~12 bps edge)
- `USDC/USDT -> XLM/USDC -> XLM/USDT` (~10 bps edge)

The common pattern: USDT and USDC trade at slightly different prices (0.9997 vs 1.0003), and routing through an intermediate token captures the spread.

**Blocklisted paths**: Any cycle routing through BTC-quote or ETH-quote pairs (e.g., `BDX/BTC`, `XLM/ETH`) is automatically filtered. These pairs have quantity precision of only 2-4 decimal places, causing rounding losses of $1.50-2.28 per trade — far exceeding the typical edge.

## Risk Profile

**What can go wrong:**

1. **Partial execution + unwind cost**: Leg 1 fills but Leg 2 is rejected -> unwind Leg 1 at market. Observed cost: **-$0.52 average**, worst case -$2.28 on low-precision pairs (now blocklisted). With the blocklist, expected unwind cost drops to ~$0.25.

2. **Price movement between legs**: The 3 legs execute sequentially (~50ms each). If the price moves against us between Leg 1 and Leg 3, the cycle may be unprofitable. Mitigated by the speed of execution (total <150ms) and the fact that stablecoin pairs move slowly.

3. **Competition**: Other bots may be arbing the same opportunities. As more participants enter, edges narrow. The 0% maker fee is our structural advantage — competitors paying taker fees need larger edges to be profitable.

4. **Exchange risk**: All 3 legs are on MEXC. If MEXC goes down, has delayed fills, or changes their fee structure, the strategy is impacted. This is single-exchange concentration risk.

5. **Fee structure change**: If MEXC raises maker fees above 0%, the economics change fundamentally. At even 0.01% maker per leg (3 bps total), most current edges would be unprofitable. **There is currently no automated fee monitoring** — this should be checked monthly at minimum.

6. **Rounding losses**: Pairs with low quantity or price precision can lose more to rounding than the edge is worth. Mitigated by the pre-execution rounding check (skips cycles where rounding loss > 5 bps) and the BTC/ETH-quote pair blocklist.

**What cannot go wrong:**

- **No directional risk**: We start and end in the same currency (USDT). There is no net exposure to any asset price.
- **No counterparty risk between exchanges**: All legs are on one exchange. No cross-exchange settlement or transfer delays.
- **No leverage**: All trades are spot, fully funded.

## Capacity and Scaling

Current trade size is **$500 per cycle**. Key considerations for scaling:

- **Order book depth**: The stablecoin pairs (USDC/USDT) typically have $50K+ of depth at the top of book. Intermediate tokens (MX, CHESS) may have thinner books — at $1,000+ per trade, check that the maker order wouldn't consume more than 10% of the visible book.
- **Fill rate impact**: Larger orders are less likely to fill as maker before the opportunity closes. Expect fill rate to decrease as trade size increases.
- **Rate limiting**: Currently capped at 2 executions per 60-second window (configurable). At $500 per trade with ~4 fills/min, that's ~$2,000/min cycling through the strategy.
- **Diminishing returns**: The same 3-4 profitable paths are recycled. Scaling up trade size on the same paths may move the market, reducing the edge for subsequent trades.

**Recommendation**: Before increasing trade size, collect 48+ hours of data at $500 to establish baseline fill rates and edge distribution. Then test $750, then $1,000, monitoring for fill rate degradation.

## Paper Trading vs Live

The system is currently **paper trading** — simulating fills against real order book data but not placing actual orders. The paper trading engine includes:

- Realistic fee simulation (0% maker, 0.05% taker based on order type and spread crossing)
- 85% fill rate simulation for LIMIT_MAKER orders (15% random rejection)
- Correct fee denomination (BUY fees in base currency, SELL fees in quote currency)
- Pre-execution rounding check (skips cycles where rounding loss > 5 bps)

**Before going live**, validate:
1. Confirm MEXC still offers 0% maker fee on spot (check fee schedule monthly)
2. Run paper trading for 48+ hours to establish baseline statistics
3. Start with $100 trade size (not $500) to limit exposure during validation
4. Monitor fill rates — real fill rates may differ from the 85% simulation
5. Watch for competing bots narrowing the edges over time
6. Set up automated fee verification (daily check of MEXC fee endpoint)

## Known Limitations

1. **No automated fee monitoring**: The strategy depends entirely on MEXC's 0% maker fee. There is no automated check that this policy hasn't changed. A daily cron or Middle Loop check should verify `GET /api/v3/account/tradeFee` returns 0% maker.

2. **No competition detector**: Edge sizes are not tracked over time. If median edge drops from 8 bps to 4 bps over a week, it may signal new competitors. This would be an early warning to reduce position size.

3. **Single-exchange concentration**: All activity is on MEXC. Exchange downtime, API changes, or geo-restrictions (the VPS already hits MEXC contract API geo-blocks) could halt the strategy entirely.

4. **Paper trading fill rate may differ from live**: The 85% simulated fill rate for LIMIT_MAKER may be optimistic or pessimistic vs real execution. Only live testing will establish the true fill rate.

## Technical Details

### Key Files
| File | Purpose |
|------|---------|
| `arbitrage/triangular/triangle_arb.py` | Scanner + cycle detector + pair blocklist |
| `arbitrage/execution/triangular_executor.py` | 3-leg execution engine + rounding check |
| `arbitrage/tracking/performance.py` | P&L tracking + DB persistence |
| `arbitrage/orchestrator.py` | Lifecycle management, wires components together |
| `arbitrage/exchanges/mexc_client.py` | MEXC API client (REST + WebSocket + paper fill) |
| `arbitrage/costs/model.py` | Fee schedules and cost estimation |
| `arbitrage/config/arbitrage.yaml` | All tunable parameters |

### Configuration
Tunable parameters in `arbitrage/config/arbitrage.yaml`:
- `triangular.min_net_profit_bps`: Minimum edge to execute (default: 5.0 bps)
- `triangular.max_trade_usd`: Trade size per cycle (default: $500)
- `triangular.max_signals_per_cycle`: Rate limit per 60s window (default: 2)
- `triangular.observation_mode`: Log-only mode without execution (default: false)
- `triangular.start_currencies`: Which currencies to start cycles from (default: USDT)

### Database Schema
Trades are stored in `data/arbitrage.db`, table `arb_trades`:
```sql
trade_id TEXT        -- e.g., "tri_USDT_1771624438725"
strategy TEXT        -- "triangular"
symbol TEXT          -- "USDC/USDT→MX/USDC→MX/USDT"
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

### Safeguards
1. **Pre-execution rounding check**: Simulates all 3 legs at actual precision before placing orders. Skips if rounding loss > 5 bps.
2. **BTC/ETH-quote pair blocklist**: Filters out cycles through low-precision pairs at the scanner level.
3. **Rate limiting**: Max 2 executions per 60-second window.
4. **Post-only guarantee**: LIMIT_MAKER orders cannot accidentally become taker.
5. **Automatic unwind**: Failed cycles are reversed via market orders to recover capital.
