"""
Abstract base class for exchange connectivity.
Every exchange client MUST implement this interface.
All prices/quantities use Decimal for precision.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Callable, Awaitable, List, Dict
import asyncio
import uuid


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    LIMIT_MAKER = "limit_maker"  # Rejected if would be taker — CRITICAL for zero-fee


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TimeInForce(Enum):
    GTC = "GTC"   # Good Till Cancel
    IOC = "IOC"   # Immediate Or Cancel
    FOK = "FOK"   # Fill Or Kill
    GTX = "GTX"   # Good Till Crossing (post-only / maker-only)


@dataclass
class OrderBookLevel:
    price: Decimal
    quantity: Decimal


@dataclass
class OrderBook:
    exchange: str
    symbol: str        # Normalized: "BTC/USDT"
    timestamp: datetime
    bids: List[OrderBookLevel]  # Sorted highest to lowest
    asks: List[OrderBookLevel]  # Sorted lowest to highest

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0].price if self.bids else None

    @property
    def best_bid_qty(self) -> Optional[Decimal]:
        return self.bids[0].quantity if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0].price if self.asks else None

    @property
    def best_ask_qty(self) -> Optional[Decimal]:
        return self.asks[0].quantity if self.asks else None

    @property
    def mid_price(self) -> Optional[Decimal]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread_bps(self) -> Optional[Decimal]:
        if self.best_bid and self.best_ask and self.mid_price:
            return ((self.best_ask - self.best_bid) / self.mid_price) * 10000
        return None

    def available_liquidity_at_impact(self, side: OrderSide, max_impact_bps: Decimal) -> Decimal:
        """How much size can be traded within a given price impact."""
        levels = self.asks if side == OrderSide.BUY else self.bids
        if not levels:
            return Decimal('0')

        reference_price = levels[0].price
        if reference_price == 0:
            return Decimal('0')

        if side == OrderSide.BUY:
            max_price = reference_price * (1 + max_impact_bps / 10000)
        else:
            max_price = reference_price * (1 - max_impact_bps / 10000)

        total_qty = Decimal('0')
        for level in levels:
            if side == OrderSide.BUY and level.price > max_price:
                break
            if side == OrderSide.SELL and level.price < max_price:
                break
            total_qty += level.quantity

        return total_qty


@dataclass
class OrderRequest:
    exchange: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: Optional[str] = None

    def __post_init__(self):
        if self.client_order_id is None:
            self.client_order_id = f"arb_{uuid.uuid4().hex[:12]}"


@dataclass
class OrderResult:
    exchange: str
    symbol: str
    order_id: str
    client_order_id: Optional[str]
    status: OrderStatus
    side: OrderSide
    order_type: OrderType
    requested_quantity: Decimal
    filled_quantity: Decimal
    average_fill_price: Optional[Decimal]
    fee_amount: Decimal
    fee_currency: str
    timestamp: datetime
    raw_response: dict = field(default_factory=dict)


@dataclass
class Balance:
    exchange: str
    currency: str
    free: Decimal       # Available for trading
    locked: Decimal     # In open orders
    total: Decimal      # free + locked


@dataclass
class FundingRate:
    exchange: str
    symbol: str
    current_rate: Decimal
    predicted_rate: Optional[Decimal]
    next_funding_time: datetime
    timestamp: datetime


class ExchangeClient(ABC):
    """Abstract base. Both MEXC and Binance clients implement this."""

    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    # --- Market Data ---
    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        pass

    @abstractmethod
    async def subscribe_order_book(
        self, symbol: str,
        callback: Callable[[OrderBook], Awaitable[None]],
        depth: int = 20,
    ) -> None:
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> dict:
        pass

    @abstractmethod
    async def get_all_tickers(self) -> Dict[str, dict]:
        pass

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> FundingRate:
        pass

    # --- Trading ---
    @abstractmethod
    async def place_order(self, order: OrderRequest) -> OrderResult:
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        pass

    @abstractmethod
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResult:
        pass

    # --- Account ---
    @abstractmethod
    async def get_balances(self) -> Dict[str, Balance]:
        pass

    @abstractmethod
    async def get_balance(self, currency: str) -> Balance:
        pass

    # --- Exchange Info ---
    @abstractmethod
    async def get_trading_fees(self, symbol: str) -> dict:
        pass

    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> dict:
        pass
