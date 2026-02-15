"""
Live Market Data Provider
Fetches real-time market data from Coinbase and converts it into internal data structures.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import os
import time

from coinbase_client import create_client_from_config, EnhancedCoinbaseClient
from microstructure_engine import OrderBookSnapshot, OrderBookLevel
from enhanced_technical_indicators import PriceData


@dataclass
class MarketDataSnapshot:
    """Container for live market data"""
    order_book_snapshot: Optional[OrderBookSnapshot]
    price_data: Optional[PriceData]
    ticker: Dict[str, Any]


class LiveMarketDataProvider:
    """Fetches live market data using Coinbase Advanced Trade API."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config

        trading_cfg = config.get("trading", {})
        data_cfg = config.get("data", {})
        coinbase_cfg = config.get("coinbase", {})

        self.product_id = trading_cfg.get("product_id", "BTC-USD")
        self.candle_granularity = data_cfg.get("candle_granularity", "ONE_MINUTE")
        self.candle_lookback_minutes = int(data_cfg.get("candle_lookback_minutes", 120))
        self.order_book_depth = int(data_cfg.get("order_book_depth", 10))

        self._client = self._build_client(coinbase_cfg, trading_cfg)
        self._last_candle_ts: Optional[int] = None

    def _build_client(self, coinbase_cfg: Dict[str, Any], trading_cfg: Dict[str, Any]) -> Optional[EnhancedCoinbaseClient]:
        api_key_env = coinbase_cfg.get("api_key_env", "COINBASE_API_KEY")
        api_secret_env = coinbase_cfg.get("api_secret_env", "COINBASE_API_SECRET")
        api_passphrase_env = coinbase_cfg.get("api_passphrase_env", "COINBASE_API_PASSPHRASE")

        api_key = os.getenv(api_key_env, "").strip()
        api_secret = os.getenv(api_secret_env, "").strip()
        api_passphrase = os.getenv(api_passphrase_env, "").strip()
        
        # api_passphrase is optional for Coinbase Advanced Trade v3 (Cloud API Keys)
        if not (api_key and api_secret):
            self.logger.warning(
                "Coinbase credentials are missing. Set %s and %s to enable live data.",
                api_key_env,
                api_secret_env
            )
            return None

        client_config = {
            "COINBASE_API_KEY": api_key,
            "COINBASE_API_SECRET": api_secret,
            "COINBASE_API_PASSPHRASE": api_passphrase,
            "SANDBOX_MODE": bool(trading_cfg.get("sandbox", True)),
            "PAPER_TRADING": bool(trading_cfg.get("paper_trading", True)),
        }

        return create_client_from_config(client_config, logger=self.logger)

    def fetch_ticker(self, product_id: Optional[str] = None) -> Dict[str, Any]:
        if not self._client:
            return {}
        try:
            pid = product_id or self.product_id
            return self._client.get_market_trades(pid)
        except Exception as exc:
            self.logger.error("Failed to fetch ticker: %s", exc)
            return {}

    def fetch_order_book_snapshot(self, product_id: Optional[str] = None) -> Optional[OrderBookSnapshot]:
        if not self._client:
            return None

        pid = product_id or self.product_id
        ticker = self.fetch_ticker(pid)
        best_bid = _coerce_float(ticker.get("best_bid") or ticker.get("bid"))
        best_ask = _coerce_float(ticker.get("best_ask") or ticker.get("ask"))
        last_price = _coerce_float(ticker.get("price"))
        last_size = _coerce_float(ticker.get("size") or ticker.get("volume"))

        try:
            if hasattr(self._client, "get_product_book"):
                book = self._client.get_product_book(pid, limit=self.order_book_depth)
            else:
                book = {}
        except Exception as exc:
            self.logger.warning("Order book fetch failed: %s", exc)
            book = {}

        bids = _parse_order_book_side(book, "bids")
        asks = _parse_order_book_side(book, "asks")

        if not bids and best_bid:
            bids = [OrderBookLevel(price=best_bid, size=last_size or 1.0)]
        if not asks and best_ask:
            asks = [OrderBookLevel(price=best_ask, size=last_size or 1.0)]

        if not last_price:
            last_price = best_bid or best_ask or 0.0

        if not bids and not asks:
            return None

        return OrderBookSnapshot(
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks,
            last_price=last_price,
            last_size=last_size or 0.0,
        )

    def fetch_latest_candle(self, product_id: Optional[str] = None) -> Optional[PriceData]:
        if not self._client:
            return None

        pid = product_id or self.product_id
        end_ts = int(time.time())
        start_ts = end_ts - (self.candle_lookback_minutes * 60)

        response = self._client.get_product_candles(
            pid,
            start=str(start_ts),
            end=str(end_ts),
            granularity=self.candle_granularity,
        )

        candles: List[Dict[str, Any]] = response.get("candles", []) if isinstance(response, dict) else []
        if not candles:
            return None

        latest = _select_latest_candle(candles)
        if latest is None:
            return None

        ts = int(latest.get("start", 0))
        if ts and self._last_candle_ts == ts:
            return None

        self._last_candle_ts = ts or None
        return PriceData(
            timestamp=datetime.utcfromtimestamp(ts) if ts else datetime.utcnow(),
            open=_coerce_float(latest.get("open")),
            high=_coerce_float(latest.get("high")),
            low=_coerce_float(latest.get("low")),
            close=_coerce_float(latest.get("close")),
            volume=_coerce_float(latest.get("volume")),
        )

    def fetch_candle_history(self, product_id: Optional[str] = None) -> List[PriceData]:
        """
        Fetch ALL available candles (up to lookback window) sorted oldest-first.
        Used for preloading price history so technical indicators work immediately.
        """
        if not self._client:
            return []

        pid = product_id or self.product_id
        end_ts = int(time.time())
        start_ts = end_ts - (self.candle_lookback_minutes * 60)

        response = self._client.get_product_candles(
            pid,
            start=str(start_ts),
            end=str(end_ts),
            granularity=self.candle_granularity,
        )

        candles: List[Dict[str, Any]] = response.get("candles", []) if isinstance(response, dict) else []
        if not candles:
            return []

        # Sort oldest-first for chronological loading
        candles.sort(key=lambda c: int(c.get("start", 0)))

        result = []
        for c in candles:
            ts = int(c.get("start", 0))
            close_price = _coerce_float(c.get("close"))
            if close_price <= 0:
                continue
            result.append(PriceData(
                timestamp=datetime.utcfromtimestamp(ts) if ts else datetime.utcnow(),
                open=_coerce_float(c.get("open")),
                high=_coerce_float(c.get("high")),
                low=_coerce_float(c.get("low")),
                close=close_price,
                volume=_coerce_float(c.get("volume")),
            ))

        if result:
            self._last_candle_ts = int(candles[-1].get("start", 0)) or None
            self.logger.info(f"Loaded {len(result)} historical candles for {pid} "
                           f"(oldest: {result[0].timestamp}, newest: {result[-1].timestamp})")

        return result

    def fetch_snapshot(self, product_id: Optional[str] = None) -> MarketDataSnapshot:
        pid = product_id or self.product_id
        order_book_snapshot = self.fetch_order_book_snapshot(pid)
        price_data = self.fetch_latest_candle(pid)
        ticker = self.fetch_ticker(pid)
        return MarketDataSnapshot(
            order_book_snapshot=order_book_snapshot,
            price_data=price_data,
            ticker=ticker,
        )


def _coerce_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _select_latest_candle(candles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    try:
        return max(candles, key=lambda c: int(c.get("start", 0)))
    except Exception:
        return None


def _parse_order_book_side(book: Dict[str, Any], side: str) -> List[OrderBookLevel]:
    levels: List[OrderBookLevel] = []

    if not isinstance(book, dict):
        return levels

    container = book.get("pricebook") or book.get("order_book") or book
    raw_levels = container.get(side, []) if isinstance(container, dict) else []

    for entry in raw_levels:
        price = None
        size = None

        if isinstance(entry, dict):
            price = entry.get("price") or entry.get("price_level")
            size = entry.get("size") or entry.get("quantity") or entry.get("size_qty")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            price, size = entry[0], entry[1]

        price_value = _coerce_float(price)
        size_value = _coerce_float(size)

        if price_value > 0 and size_value > 0:
            levels.append(OrderBookLevel(price=price_value, size=size_value))

    reverse = side == "bids"
    levels.sort(key=lambda level: level.price, reverse=reverse)

    return levels
