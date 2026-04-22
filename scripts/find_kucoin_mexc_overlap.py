#!/usr/bin/env python3
"""
Find overlapping USDT trading pairs between KuCoin, MEXC, and Binance.

Uses our async exchange clients (with browser-like UA) to avoid geo-blocking.

Outputs:
  - KuCoin∩MEXC count (arb-able via hub-spoke)
  - KuCoin∩MEXC−Binance count (NEW opportunities not on Binance spoke)
  - Full overlap list saved to data/kucoin_mexc_pairs.txt
"""
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arbitrage.exchanges.kucoin_client import KuCoinClient
from arbitrage.exchanges.mexc_client import MEXCClient
from arbitrage.exchanges.binance_client import BinanceClient


async def main():
    kucoin = KuCoinClient(paper_trading=True)
    mexc = MEXCClient(paper_trading=True)
    binance = BinanceClient(paper_trading=True)

    await kucoin.connect()
    await mexc.connect()
    await binance.connect()

    try:
        print("Fetching all tickers from 3 exchanges...")
        kc_tickers, mx_tickers = await asyncio.gather(
            kucoin.get_all_tickers(),
            mexc.get_all_tickers(),
        )
        # Binance may be geo-blocked locally (451)
        try:
            bn_tickers = await binance.get_all_tickers()
        except Exception as e:
            print(f"  (Binance geo-blocked: {e} — skipping Binance comparison)")
            bn_tickers = {}

        kc_usdt = {s for s in kc_tickers if s.endswith("/USDT")}
        mx_usdt = {s for s in mx_tickers if s.endswith("/USDT")}
        bn_usdt = {s for s in bn_tickers if s.endswith("/USDT")}

        # Overlaps
        kc_mx = kc_usdt & mx_usdt
        kc_bn = kc_usdt & bn_usdt
        mx_bn = mx_usdt & bn_usdt
        all_three = kc_usdt & mx_usdt & bn_usdt
        kc_mx_not_bn = kc_mx - bn_usdt

        print(f"\n{'='*60}")
        print(f"USDT pairs per exchange:")
        print(f"  KuCoin:  {len(kc_usdt)}")
        print(f"  MEXC:    {len(mx_usdt)}")
        print(f"  Binance: {len(bn_usdt)}")
        print(f"\nOverlap analysis:")
        print(f"  KuCoin ∩ MEXC:              {len(kc_mx)} (arb-able)")
        print(f"  KuCoin ∩ Binance:           {len(kc_bn)}")
        print(f"  MEXC ∩ Binance:             {len(mx_bn)} (existing arb)")
        print(f"  All three:                  {len(all_three)}")
        print(f"  KuCoin ∩ MEXC - Binance:    {len(kc_mx_not_bn)} (NEW opportunities)")
        print(f"{'='*60}")

        # Show KuCoin-only pairs with spread data
        if kc_mx_not_bn:
            print(f"\nNEW pairs (KuCoin ∩ MEXC, not on Binance) — top 30:")
            spreads = []
            for pair in kc_mx_not_bn:
                kc_data = kc_tickers.get(pair, {})
                mx_data = mx_tickers.get(pair, {})
                kc_bid = float(kc_data.get('bid', 0) or 0)
                mx_bid = float(mx_data.get('bid', 0) or 0)
                if kc_bid > 0 and mx_bid > 0:
                    spread = abs(kc_bid - mx_bid) / min(kc_bid, mx_bid) * 10000
                    spreads.append((pair, spread))
            spreads.sort(key=lambda x: x[1], reverse=True)
            for pair, spread in spreads[:30]:
                print(f"  {pair:20s}  spread={spread:7.1f}bps")

        # Save full overlap to file
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "kucoin_mexc_pairs.txt"

        with open(output_file, "w") as f:
            f.write(f"# KuCoin ∩ MEXC overlapping USDT pairs ({len(kc_mx)} total)\n")
            f.write(f"# Also on Binance: {len(all_three)}\n")
            f.write(f"# KuCoin+MEXC only: {len(kc_mx_not_bn)}\n\n")
            for pair in sorted(kc_mx):
                tag = "" if pair in bn_usdt else " [NEW]"
                f.write(f"{pair}{tag}\n")

        print(f"\nFull list saved to {output_file}")

    finally:
        await kucoin.disconnect()
        await mexc.disconnect()
        await binance.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
