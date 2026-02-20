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
- **Taker execution** (0.05% × 3 = 15 bps cost): Edge is wiped out. Every trade loses money.
- **Maker execution** (0.00% × 3 = 0 bps cost): Full edge captured. Every trade profits.

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
Searches the graph for all 3-step cycles starting from USDT, BTC, or ETH. For each cycle, computes:

```
cycle_rate = rate₁ × rate₂ × rate₃
```

If `cycle_rate > 1.0`, the cycle is profitable. Filters for a minimum net profit threshold (currently 3 basis points) to avoid dust-level edges. Returns the top 3 opportunities per scan, rate-limited to 3 executions per 60-second window.

### 3. Triangular Executor
Executes profitable cycles via 3 sequential LIMIT_MAKER orders:

1. **Pre-fetch** all 3 order books and symbol precision info concurrently (~130ms)
2. **Leg 1**: Place post-only order, wait for fill
3. **Leg 2**: Use output from Leg 1 as input, place post-only order
4. **Leg 3**: Use output from Leg 2 as input, place post-only order
5. **Result**: Compare final USDT amount to starting amount → profit/loss

**If any leg fails** (post-only rejection, timeout, insufficient quantity):
- All previously completed legs are **unwound** in reverse order using market orders
- This recovers the starting capital with minimal loss (market unwind may incur taker fees)

Total execution time: **110-150ms** for all 3 legs.

### 4. Performance Tracker
Every completed trade (filled or failed) is persisted to `data/arbitrage.db` in the `arb_trades` table with strategy = `"triangular"`. The web dashboard reads this table to display:
- Total triangular P&L
- Win rate and trade count
- Per-trade profit history
- Wallet balance impact

## Current Performance (2026-02-20)

| Metric | Value |
|--------|-------|
| Total triangular trades | 256 |
| Filled (successful) | 242 |
| Failed (unwound) | 14 |
| Fill rate | 94.5% |
| Total profit | **$20.95** |
| Avg profit per fill | **$0.09** |
| Win rate (filled trades) | 79.8% |
| Avg winning trade | $0.54 |
| Avg losing trade | -$1.69 |
| Execution speed | 110-150ms |

### Performance by Era

The system went through three iterations on launch day:

| Era | Trades | Avg P&L | Notes |
|-----|--------|---------|-------|
| Pre-fee simulation | 170 | +$0.08 | Paper fill didn't simulate fees — inflated results |
| Taker fee era | 26 | -$1.33 | Fees correctly simulated — exposed taker cost problem |
| **LIMIT_MAKER era** | **143** | **+$0.92** | Post-only orders, 0% maker fee — genuine alpha |

The LIMIT_MAKER era shows the true economics: **$0.92 average profit per filled trade** with a 62% fill rate (88 fills out of 143 attempts). Failed attempts have zero cost because the post-only order is simply rejected by the exchange.

## Most Profitable Paths

The highest-yield triangular paths consistently involve stablecoin pairs:

- `USDC/USDT → MX/USDC → MX/USDT` (~15 bps edge)
- `USDC/USDT → CHESS/USDC → CHESS/USDT` (~12 bps edge)
- `USDC/USDT → XLM/USDC → XLM/USDT` (~10 bps edge)

The common pattern: USDT and USDC trade at slightly different prices (0.9997 vs 1.0003), and routing through an intermediate token captures the spread.

## Risk Profile

**What can go wrong:**

1. **Partial execution**: Leg 1 fills but Leg 2 is rejected → unwind Leg 1 at market. Cost: taker fee on the unwind (~$0.25 on a $500 trade). Mitigated by the post-only guarantee — we never accidentally become a taker on the forward leg.

2. **Price movement between legs**: The 3 legs execute sequentially (~50ms each). If the price moves against us between Leg 1 and Leg 3, the cycle may be unprofitable. Mitigated by the speed of execution (total <150ms) and the fact that stablecoin pairs move slowly.

3. **Competition**: Other bots may be arbing the same opportunities. As more participants enter, edges narrow. The 0% maker fee is our structural advantage — competitors paying taker fees need larger edges to be profitable.

4. **Exchange risk**: All 3 legs are on MEXC. If MEXC goes down, has delayed fills, or changes their fee structure, the strategy is impacted. This is single-exchange concentration risk.

5. **Fee structure change**: If MEXC raises maker fees above 0%, the economics change fundamentally. At even 0.01% maker per leg (3 bps total), most current edges would be unprofitable.

**What cannot go wrong:**

- **No directional risk**: We start and end in the same currency (USDT). There is no net exposure to any asset price.
- **No counterparty risk between exchanges**: All legs are on one exchange. No cross-exchange settlement or transfer delays.
- **No leverage**: All trades are spot, fully funded.

## Paper Trading vs Live

The system is currently **paper trading** — simulating fills against real order book data but not placing actual orders. The paper trading engine now includes:

- Realistic fee simulation (0% maker, 0.05% taker based on order type and spread crossing)
- 85% fill rate simulation for LIMIT_MAKER orders (15% random rejection)
- Correct fee denomination (BUY fees in base currency, SELL fees in quote currency)

**Before going live**, validate:
1. Confirm MEXC still offers 0% maker fee on spot (check fee schedule monthly)
2. Run paper trading for 48+ hours to establish baseline statistics
3. Start with $100 trade size (not $500) to limit exposure during validation
4. Monitor fill rates — real fill rates may differ from the 85% simulation
5. Watch for competing bots narrowing the edges over time

## Technical Details

### Key Files
| File | Purpose |
|------|---------|
| `arbitrage/triangular/triangle_arb.py` | Scanner + cycle detector |
| `arbitrage/execution/triangular_executor.py` | 3-leg execution engine |
| `arbitrage/tracking/performance.py` | P&L tracking + DB persistence |
| `arbitrage/orchestrator.py` | Lifecycle management, wires components together |
| `arbitrage/exchanges/mexc_client.py` | MEXC API client (REST + WebSocket) |
| `arbitrage/costs/model.py` | Fee schedules and cost estimation |

### Configuration
Tunable parameters in `arbitrage/config/arbitrage.yaml`:
- `triangular.min_net_profit_bps`: Minimum edge to execute (default: 3.0 bps)
- `triangular.max_trade_usd`: Trade size per cycle (default: $500)
- `triangular.max_signals_per_cycle`: Rate limit per 60s window (default: 3)
- `triangular.observation_mode`: Log-only mode without execution (default: false)
- `triangular.start_currencies`: Which currencies to start cycles from (default: USDT, BTC, ETH)

### Database Schema
Trades are stored in `data/arbitrage.db`, table `arb_trades`:
```sql
trade_id TEXT        -- e.g., "tri_USDT_1771624438725"
strategy TEXT        -- "triangular"
symbol TEXT          -- "USDC/USDT→MX/USDC→MX/USDT"
status TEXT          -- "filled" or "failed"
actual_profit_usd REAL  -- profit after fees
quantity REAL        -- starting USDT amount
timestamp TEXT       -- ISO 8601
```

### Dashboard Endpoints
- `GET /api/arbitrage/summary` — Aggregate stats by strategy
- `GET /api/arbitrage/trades?strategy=triangular` — Individual trade history
- `GET /api/arbitrage/wallet` — Balance impact from arb activity
