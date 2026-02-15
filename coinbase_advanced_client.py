# coinbase_advanced_client.py
import numpy as np
import websockets
import json
import logging
import queue
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class MarketData:
    """Market data structure"""
    price: float
    volume: float
    bid: float
    ask: float
    spread: float
    timestamp: datetime
    source: str = "coinbase"
    symbol: str = "BTC-USD"
    order_book: Dict = field(default_factory=dict)
    recent_trades: List = field(default_factory=list)


@dataclass
class DataSource:
    """Data source structure"""
    name: str
    status: str
    last_update: datetime
    error_count: int = 0


class CoinbaseAdvancedClient:
    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config['api_key']
        self.api_secret = config['api_secret']
        self.passphrase = config['passphrase']
        self.sandbox = config.get('sandbox', True)

        self.base_url = "wss://ws-feed-public.sandbox.exchange.coinbase.com" if self.sandbox else "wss://ws-feed.exchange.coinbase.com"
        self.websocket = None
        self.logger = logging.getLogger(f"{__name__}.CoinbaseClient")

        # Order book tracking
        self.order_books = {}
        self.recent_trades = {}

    async def connect_websocket(self):
        """Connect to Coinbase WebSocket feed"""
        try:
            self.websocket = await websockets.connect(self.base_url)

            # Subscribe to channels
            subscribe_message = {
                "type": "subscribe",
                "product_ids": self.config['symbols'],
                "channels": self.config['websocket_channels']
            }

            await self.websocket.send(json.dumps(subscribe_message))
            self.logger.info("Connected to Coinbase WebSocket")

        except Exception as e:
            self.logger.error(f"Failed to connect to Coinbase WebSocket: {e}")
            raise

    async def listen_for_messages(self, data_queue: queue.Queue):
        """Listen for WebSocket messages and process them"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                processed_data = await self._process_coinbase_message(data)

                if processed_data:
                    try:
                        data_queue.put_nowait(processed_data)
                    except queue.Full:
                        # Drain stale entries and push latest — keep log quiet
                        try:
                            while not data_queue.empty():
                                data_queue.get_nowait()
                            data_queue.put_nowait(processed_data)
                        except Exception:
                            pass

        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("Coinbase WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"Error processing Coinbase message: {e}")

    async def _process_coinbase_message(self, data: Dict) -> Optional[MarketData]:
        """Process individual Coinbase messages"""
        try:
            msg_type = data.get('type')
            product_id = data.get('product_id', 'BTC-USD')

            if msg_type == 'l2update':
                # Order book update
                self._update_order_book(product_id, data)
                return self._create_market_data_snapshot(product_id)

            elif msg_type == 'match':
                # Trade execution
                self._update_recent_trades(product_id, data)
                return self._create_market_data_snapshot(product_id)

            elif msg_type == 'ticker':
                # Price ticker - fixed parameter matching
                best_bid = float(data.get('best_bid', 0))
                best_ask = float(data.get('best_ask', 0))

                return MarketData(
                    timestamp=datetime.now(),
                    source="coinbase",
                    symbol=product_id,
                    price=float(data.get('price', 0)),
                    volume=float(data.get('volume_24h', 0)),
                    bid=best_bid,
                    ask=best_ask,
                    spread=best_ask - best_bid,
                    order_book=self.order_books.get(product_id, {}),
                    recent_trades=self.recent_trades.get(product_id, [])
                )

            return None

        except Exception as e:
            self.logger.error(f"Error processing Coinbase message: {e}")
            return None

    def _update_order_book(self, product_id: str, data: Dict):
        """Update local order book with L2 data"""
        if product_id not in self.order_books:
            self.order_books[product_id] = {'bids': {}, 'asks': {}}

        changes = data.get('changes', [])
        for change in changes:
            side, price, size = change
            price = float(price)
            size = float(size)

            if side == 'buy':
                if size == 0:
                    self.order_books[product_id]['bids'].pop(price, None)
                else:
                    self.order_books[product_id]['bids'][price] = size
            else:  # sell
                if size == 0:
                    self.order_books[product_id]['asks'].pop(price, None)
                else:
                    self.order_books[product_id]['asks'][price] = size

    def _update_recent_trades(self, product_id: str, data: Dict):
        """Update recent trades list"""
        if product_id not in self.recent_trades:
            self.recent_trades[product_id] = []

        trade = {
            'price': float(data.get('price', 0)),
            'size': float(data.get('size', 0)),
            'side': data.get('side'),
            'time': data.get('time'),
            'trade_id': data.get('trade_id')
        }

        self.recent_trades[product_id].append(trade)

        # Keep only the last 100 trades
        if len(self.recent_trades[product_id]) > 100:
            self.recent_trades[product_id] = self.recent_trades[product_id][-100:]

    def _create_market_data_snapshot(self, product_id: str) -> MarketData:
        """Create market data snapshot from the current state"""
        order_book = self.order_books.get(product_id, {'bids': {}, 'asks': {}})
        recent_trades = self.recent_trades.get(product_id, [])

        # Calculate current bid/ask
        bids = sorted(order_book['bids'].keys(), reverse=True) if order_book['bids'] else []
        asks = sorted(order_book['asks'].keys()) if order_book['asks'] else []

        best_bid = bids[0] if bids else 0.0
        best_ask = asks[0] if asks else 0.0
        spread = best_ask - best_bid if best_bid and best_ask else 0.0

        # Get last trade price
        last_price = recent_trades[-1]['price'] if recent_trades else (best_bid if best_bid else 0.0)

        # Calculate volume from recent trades
        volume = sum(trade['size'] for trade in recent_trades[-10:]) if recent_trades else 0.0

        return MarketData(
            timestamp=datetime.now(),
            source="coinbase",
            symbol=product_id,
            price=last_price,
            volume=volume,
            bid=best_bid,
            ask=best_ask,
            spread=spread,
            order_book=order_book,
            recent_trades=recent_trades
        )

    # Add this method for the startup script
    async def get_market_data(self) -> Optional[Dict]:
        """Get current market data - compatibility method for startup script"""
        try:
            snapshot = self._create_market_data_snapshot('BTC-USD')
            return {
                'price': snapshot.price,
                'volume': snapshot.volume,
                'bid': snapshot.bid,
                'ask': snapshot.ask,
                'spread': snapshot.spread,
                'timestamp': snapshot.timestamp.isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None