# Archived Polymarket Strategies

These Polymarket strategy files have been **disabled** in the main bot (`renaissance_trading_bot.py`) and moved here to reduce root-directory clutter. They are preserved for reference and can be re-enabled if needed.

## Archived Files

| File | Description | Reason Archived |
|------|-------------|-----------------|
| `polymarket_strategy_a.py` | Multi-asset crash model betting (Kelly-sized, 15m/5m markets) | `STRATEGY_A_AVAILABLE = False` — replaced by spread capture |
| `polymarket_executor.py` | Legacy bet executor | Always set to `None` in `bot/builder.py`; guarded references only |
| `polymarket_live_executor.py` | Live bet placement via CLOB client | `LIVE_EXECUTOR_AVAILABLE = False` |
| `polymarket_reversal.py` | Reversal strategy (buy crowd-opposite) | `REVERSAL_STRATEGY_AVAILABLE = False` |
| `simple_up_bet.py` | $1 contrarian UP bets when crowd piles on DOWN | `SIMPLE_UP_AVAILABLE = False` |
| `polymarket_reversal_backtest.py` | Backtesting harness for reversal strategy | Standalone utility, not imported by bot |
| `polymarket_edge_simulation.py` | Edge/EV simulation across token price ranges | Standalone utility, not imported by bot |
| `polymarket_calibration.py` | Model calibration analysis | Standalone utility, not imported by bot |
| `polymarket_history.py` | Historical bet analysis | Standalone utility, not imported by bot |
| `polymarket_probability_mapper.py` | Probability mapping (only used by edge_simulation) | No active code imports |

## Active Polymarket Files (remain in root)

- `polymarket_bridge.py` — ML-to-Polymarket signal converter (always imported)
- `polymarket_scanner.py` — Market discovery and edge detection (always imported)
- `polymarket_rtds.py` — Real-time data stream via WebSocket (conditional)
- `polymarket_spread_capture.py` — 0x8dxd-style dual accumulation strategy (conditional)
- `polymarket_timing_features.py` — BTC lead-lag timing features (used by dashboard)

## Re-enabling a Strategy

1. Move the file back to the project root
2. Uncomment its import in `renaissance_trading_bot.py`
3. Set the corresponding `*_AVAILABLE` flag to `True`
4. Update initialization in `bot/builder.py`
