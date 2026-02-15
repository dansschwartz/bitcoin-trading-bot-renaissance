#!/usr/bin/env python3
"""
Renaissance of One — Arbitrage Engine Runner

Three uncorrelated revenue streams:
1. Cross-Exchange Spot Arbitrage (MEXC ↔ Binance)
2. Funding Rate Arbitrage (Perpetual Futures)
3. Triangular Arbitrage (Single Exchange)

Usage:
    python run_arbitrage.py                    # Paper trading (default)
    python run_arbitrage.py --live             # Live trading (requires API keys)
    python run_arbitrage.py --config path.yaml # Custom config

Environment Variables:
    MEXC_API_KEY, MEXC_API_SECRET       — MEXC exchange credentials
    BINANCE_API_KEY, BINANCE_API_SECRET — Binance exchange credentials
    TRADING_MODE=paper|live             — Override trading mode
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arbitrage.orchestrator import ArbitrageOrchestrator


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Renaissance Arbitrage Engine")
    parser.add_argument('--config', default='arbitrage/config/arbitrage.yaml')
    parser.add_argument('--live', action='store_true', help='Live trading mode')
    args = parser.parse_args()

    orchestrator = ArbitrageOrchestrator(config_path=args.config)

    if args.live:
        orchestrator.config.setdefault('paper_trading', {})['enabled'] = False
        print("*** LIVE TRADING MODE — Real orders will be placed ***")
    else:
        print("Paper trading mode — no real orders")

    await orchestrator.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested")
