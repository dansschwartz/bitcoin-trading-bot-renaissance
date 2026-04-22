"""
Multi-Exchange Signal Bridge
=============================
Extracts directional trading signals from the arbitrage engine's live
order books across MEXC and Binance, and combines them with Coinbase
data to produce cross-exchange consensus signals.

No new exchange connections — reads from the already-running
UnifiedBookManager in the arbitrage orchestrator.

Signals produced:
    cross_exchange_momentum  — Binance/MEXC price leads Coinbase
    price_dispersion         — disagreement across venues (negative = uncertain)
    aggregated_book_imbalance — combined bid/ask pressure across all exchanges
    funding_rate_signal      — contrarian signal from perpetual funding rates
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MultiExchangeBridge:
    """Reads arbitrage book manager data and produces directional signals."""

    def __init__(self, book_manager, mexc_client=None, binance_client=None):
        self.book_manager = book_manager
        self.mexc = mexc_client
        self.binance = binance_client
        self._funding_cache: Dict[str, float] = {}
        self._funding_fetch_count = 0
        logger.info("MultiExchangeBridge initialized")

    def _to_usdt_symbol(self, product_id: str) -> str:
        """Convert Coinbase product ID to USDT pair: BTC-USD -> BTC/USDT"""
        base = product_id.split("-")[0]
        return f"{base}/USDT"

    def get_signals(
        self,
        product_id: str,
        coinbase_bid: float = 0.0,
        coinbase_ask: float = 0.0,
        coinbase_bid_vol: float = 0.0,
        coinbase_ask_vol: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute multi-exchange signals for a given product.

        Args:
            product_id: Coinbase product ID (e.g. "BTC-USD")
            coinbase_bid: Coinbase best bid price
            coinbase_ask: Coinbase best ask price
            coinbase_bid_vol: Total bid-side volume from Coinbase order book
            coinbase_ask_vol: Total ask-side volume from Coinbase order book

        Returns:
            Dict with signal names -> values in [-1, 1]
        """
        result = {
            "cross_exchange_momentum": 0.0,
            "price_dispersion": 0.0,
            "aggregated_book_imbalance": 0.0,
            "funding_rate_signal": 0.0,
        }

        try:
            symbol = self._to_usdt_symbol(product_id)
            pair_view = self.book_manager.pairs.get(symbol)

            if pair_view is None or not pair_view.is_tradeable:
                return result

            mexc_book = pair_view.mexc_book
            binance_book = pair_view.binance_book

            # Extract prices (Decimal -> float)
            mexc_mid = float(mexc_book.mid_price) if mexc_book.mid_price else 0.0
            binance_mid = float(binance_book.mid_price) if binance_book.mid_price else 0.0
            coinbase_mid = (coinbase_bid + coinbase_ask) / 2.0 if coinbase_bid > 0 and coinbase_ask > 0 else 0.0

            # --- Signal 1: Cross-exchange momentum ---
            # If Binance is ahead of Coinbase, Coinbase will likely catch up
            if coinbase_mid > 0 and binance_mid > 0:
                spread_pct = (binance_mid - coinbase_mid) / coinbase_mid
                result["cross_exchange_momentum"] = float(np.clip(spread_pct * 1000, -1.0, 1.0))

            # --- Signal 2: Price dispersion ---
            mids = [m for m in [coinbase_mid, binance_mid, mexc_mid] if m > 0]
            if len(mids) >= 2:
                mean_mid = np.mean(mids)
                if mean_mid > 0:
                    dispersion = np.std(mids) / mean_mid  # coefficient of variation
                    result["price_dispersion"] = float(-np.clip(dispersion * 100, 0.0, 1.0))

            # --- Signal 3: Aggregated book imbalance ---
            mexc_bid_vol = sum(float(lv.quantity) for lv in mexc_book.bids[:10]) if mexc_book.bids else 0.0
            mexc_ask_vol = sum(float(lv.quantity) for lv in mexc_book.asks[:10]) if mexc_book.asks else 0.0
            bn_bid_vol = sum(float(lv.quantity) for lv in binance_book.bids[:10]) if binance_book.bids else 0.0
            bn_ask_vol = sum(float(lv.quantity) for lv in binance_book.asks[:10]) if binance_book.asks else 0.0

            total_bid = coinbase_bid_vol + mexc_bid_vol + bn_bid_vol
            total_ask = coinbase_ask_vol + mexc_ask_vol + bn_ask_vol
            if total_bid + total_ask > 0:
                imbalance = (total_bid - total_ask) / (total_bid + total_ask)
                result["aggregated_book_imbalance"] = float(np.clip(imbalance, -1.0, 1.0))

            # --- Signal 4: Funding rate (contrarian) ---
            funding = self._funding_cache.get(symbol, 0.0)
            if abs(funding) > 1e-8:
                # High positive funding = crowded long = contrarian bearish
                result["funding_rate_signal"] = float(np.clip(-funding * 100, -1.0, 1.0))

        except Exception as e:
            logger.debug(f"MultiExchangeBridge.get_signals error: {e}")

        return result

    async def update_funding_rates(self):
        """Fetch latest funding rates from MEXC and Binance.
        Call this periodically (every ~5 minutes is fine since rates change every 8h)."""
        if not self.mexc and not self.binance:
            return

        for symbol in list(self.book_manager.pairs.keys()):
            try:
                rates = []
                if self.binance:
                    try:
                        fr = await self.binance.get_funding_rate(symbol)
                        rates.append(float(fr.current_rate))
                    except Exception:
                        pass
                if self.mexc:
                    try:
                        fr = await self.mexc.get_funding_rate(symbol)
                        rates.append(float(fr.current_rate))
                    except Exception:
                        pass
                if rates:
                    self._funding_cache[symbol] = sum(rates) / len(rates)
            except Exception:
                pass

        self._funding_fetch_count += 1
        if self._funding_fetch_count <= 1 or self._funding_fetch_count % 10 == 0:
            logger.info(f"Funding rates updated: {len(self._funding_cache)} symbols")
