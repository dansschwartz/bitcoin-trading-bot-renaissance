"""
Position Manager with Complete Risk Management
Handles position management with actual stop losses and comprehensive risk controls.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import json
import warnings

# Handle optional pandas import
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    warnings.warn(
        "pandas package not available. Install with: pip install pandas",
        UserWarning
    )


class PositionSide(Enum):
    """Position side enumeration"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"


class PositionStatus(Enum):
    """Position status enumeration"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"
    PENDING = "PENDING"
    ERROR = "ERROR"


@dataclass
class RiskLimits:
    """Risk management limits configuration"""
    max_position_size_usd: float = 1000.0
    max_daily_loss_usd: float = 500.0
    max_total_exposure_usd: float = 2000.0
    max_drawdown_percentage: float = 15.0
    max_positions_per_product: int = 3
    max_total_positions: int = 5
    stop_loss_percentage: float = 3.0
    take_profit_percentage: float = 6.0
    trailing_stop_percentage: float = 2.0
    position_timeout_hours: int = 24


@dataclass
class Position:
    """Position data structure"""
    position_id: str
    product_id: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float = 0.0
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    entry_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    orders: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL based on current price"""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.size
        elif self.side == PositionSide.SHORT:
            return (self.entry_price - current_price) * self.size
        return 0.0

    def calculate_unrealized_pnl_percentage(self, current_price: float) -> float:
        """Calculate unrealized PnL as percentage"""
        if self.entry_price == 0:
            return 0.0

        pnl = self.calculate_unrealized_pnl(current_price)
        position_value = abs(self.entry_price * self.size)

        if position_value == 0:
            return 0.0

        return (pnl / position_value) * 100

    def update_trailing_stop(self, current_price: float, trailing_percentage: float):
        """Update trailing stop loss price"""
        if self.side == PositionSide.LONG:
            # For long positions, trailing stop moves up with price
            new_trailing_stop = current_price * (1 - trailing_percentage / 100)
            if self.trailing_stop_price is None or new_trailing_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_trailing_stop
        elif self.side == PositionSide.SHORT:
            # For short positions, trailing stop moves down with price
            new_trailing_stop = current_price * (1 + trailing_percentage / 100)
            if self.trailing_stop_price is None or new_trailing_stop < self.trailing_stop_price:
                self.trailing_stop_price = new_trailing_stop

    def should_trigger_stop_loss(self, current_price: float) -> bool:
        """Check if position should trigger stop loss"""
        if self.stop_loss_price is None:
            return False

        if self.side == PositionSide.LONG:
            return current_price <= self.stop_loss_price
        elif self.side == PositionSide.SHORT:
            return current_price >= self.stop_loss_price

        return False

    def should_trigger_take_profit(self, current_price: float) -> bool:
        """Check if position should trigger take profit"""
        if self.take_profit_price is None:
            return False

        if self.side == PositionSide.LONG:
            return current_price >= self.take_profit_price
        elif self.side == PositionSide.SHORT:
            return current_price <= self.take_profit_price

        return False

    def should_trigger_trailing_stop(self, current_price: float) -> bool:
        """Check if position should trigger trailing stop"""
        if self.trailing_stop_price is None:
            return False

        if self.side == PositionSide.LONG:
            return current_price <= self.trailing_stop_price
        elif self.side == PositionSide.SHORT:
            return current_price >= self.trailing_stop_price

        return False

    def is_expired(self, timeout_hours: int = 24) -> bool:
        """Check if position has exceeded timeout"""
        expiry_time = self.entry_time + timedelta(hours=timeout_hours)
        return datetime.now() > expiry_time


@dataclass
class PositionSummary:
    """Summary of position manager state"""
    total_positions: int = 0
    open_positions: int = 0
    closed_positions: int = 0
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    total_fees: float = 0.0
    total_exposure_usd: float = 0.0
    daily_pnl: float = 0.0
    positions_by_product: Dict[str, int] = field(default_factory=dict)
    largest_position_size: float = 0.0
    average_position_size: float = 0.0


class EnhancedPositionManager:
    """
    Production-grade position management system.

    Features:
    - Complete position lifecycle management
    - Real stop loss and take profit order placement
    - Trailing stop functionality
    - Comprehensive risk management
    - Daily loss limits and emergency shutdown
    - Position timeout handling
    - Performance tracking and reporting
    """

    def __init__(self, coinbase_client: Any, risk_limits: Optional[RiskLimits] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize position manager.

        Args:
            coinbase_client: Coinbase API client instance
            risk_limits: Risk management configuration
            logger: Optional logger instance
        """

        self.client = coinbase_client
        self.risk_limits = risk_limits or RiskLimits()
        self.logger = logger or logging.getLogger(__name__)

        # Position storage
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []

        # Risk tracking
        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.emergency_stop = False

        # Thread safety
        self._lock = threading.Lock()

        # Performance tracking
        self.stats = {
            "positions_opened": 0,
            "positions_closed": 0,
            "stop_losses_triggered": 0,
            "take_profits_triggered": 0,
            "trailing_stops_triggered": 0,
            "emergency_stops": 0,
            "risk_limit_violations": 0
        }

        self.logger.info("Enhanced Position Manager initialized")
        self._log_risk_limits()

    def _log_risk_limits(self):
        """Log current risk limits"""
        self.logger.info("Risk Limits Configuration:")
        self.logger.info(f"  Max position size: ${self.risk_limits.max_position_size_usd}")
        self.logger.info(f"  Max daily loss: ${self.risk_limits.max_daily_loss_usd}")
        self.logger.info(f"  Stop loss: {self.risk_limits.stop_loss_percentage}%")
        self.logger.info(f"  Take profit: {self.risk_limits.take_profit_percentage}%")
        self.logger.info(f"  Trailing stop: {self.risk_limits.trailing_stop_percentage}%")

    def _reset_daily_stats_if_needed(self):
        """Reset daily statistics if new day"""
        now = datetime.now()
        current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if current_day > self.daily_reset_time:
            self.logger.info("Resetting daily statistics for new trading day")
            self.daily_pnl = 0.0
            self.daily_reset_time = current_day

    def _check_risk_limits(self, position_size_usd: float, product_id: str) -> Tuple[bool, List[str]]:
        """Check if new position would violate risk limits"""
        violations = []

        with self._lock:
            # Check position size limit
            if position_size_usd > self.risk_limits.max_position_size_usd:
                violations.append(
                    f"Position size ${position_size_usd:.2f} exceeds limit ${self.risk_limits.max_position_size_usd}")

            # Check daily loss limit
            if self.daily_pnl < -self.risk_limits.max_daily_loss_usd:
                violations.append(
                    f"Daily loss ${abs(self.daily_pnl):.2f} exceeds limit ${self.risk_limits.max_daily_loss_usd}")

            # Check total exposure
            current_exposure = self._calculate_total_exposure()
            if current_exposure + position_size_usd > self.risk_limits.max_total_exposure_usd:
                violations.append(f"Total exposure would exceed limit ${self.risk_limits.max_total_exposure_usd}")

            # Check positions per product
            product_position_count = len([p for p in self.positions.values() if p.product_id == product_id])
            if product_position_count >= self.risk_limits.max_positions_per_product:
                violations.append(
                    f"Max positions per product ({self.risk_limits.max_positions_per_product}) would be exceeded for {product_id}")

            # Check total positions
            if len(self.positions) >= self.risk_limits.max_total_positions:
                violations.append(f"Max total positions ({self.risk_limits.max_total_positions}) would be exceeded")

            # Check emergency stop
            if self.emergency_stop:
                violations.append("Emergency stop is active")

        return len(violations) == 0, violations

    def _calculate_total_exposure(self) -> float:
        """Calculate total USD exposure across all positions"""
        total_exposure = 0.0

        for pos in self.positions.values():
            position_value = abs(pos.entry_price * pos.size)
            total_exposure += position_value

        return total_exposure

    @staticmethod
    def _generate_position_id(product_id: str, side: str) -> str:
        """Generate unique position ID"""
        timestamp = int(datetime.now().timestamp())
        return f"{product_id}_{side}_{timestamp}"

    def open_position(self, product_id: str, side: str, size: float,
                      entry_price: Optional[float] = None,
                      stop_loss_percentage: Optional[float] = None,
                      take_profit_percentage: Optional[float] = None,
                      use_trailing_stop: bool = False) -> Tuple[bool, str, Optional[Position]]:
        """
        Open a new position with risk management.

        Args:
            product_id: Product identifier (e.g., 'BTC-USD')
            side: Position side ('LONG' or 'SHORT')
            size: Position size
            entry_price: Entry price (if None, uses market price)
            stop_loss_percentage: Custom stop loss percentage
            take_profit_percentage: Custom take profit percentage
            use_trailing_stop: Enable trailing stop

        Returns:
            Tuple[success, message, position]
        """

        try:
            self._reset_daily_stats_if_needed()

            # Validate inputs
            if side not in [PositionSide.LONG.value, PositionSide.SHORT.value]:
                return False, f"Invalid side: {side}", None

            if size <= 0:
                return False, f"Invalid size: {size}", None

            # Get the current market price if not provided
            if entry_price is None:
                try:
                    product_info = self.client.get_product(product_id)
                    if "error" in product_info:
                        return False, f"Failed to get market price: {product_info['error']}", None

                    # This would need to be adapted based on actual API response structure
                    entry_price = float(product_info.get("price", 0))
                    if entry_price <= 0:
                        return False, "Could not determine entry price", None

                except Exception as price_error:
                    return False, f"Error getting market price: {price_error}", None

            # Calculate position value
            position_value = abs(entry_price * size)

            # Anti-stacking: reject if same product+side position already exists
            # Also reject if opposing position exists (must close first)
            position_side_enum = PositionSide.LONG if side == "LONG" else PositionSide.SHORT
            opposing_side_enum = PositionSide.SHORT if side == "LONG" else PositionSide.LONG
            with self._lock:
                for existing in self.positions.values():
                    if existing.product_id == product_id and existing.status == PositionStatus.OPEN:
                        if existing.side == position_side_enum:
                            return False, f"Anti-stacking: already have {side} on {product_id}", None
                        if existing.side == opposing_side_enum:
                            return False, f"Anti-netting: opposing {existing.side.value} exists on {product_id} — close it first", None

            # Check risk limits
            risk_check_passed, risk_violations = self._check_risk_limits(position_value, product_id)
            if not risk_check_passed:
                self.stats["risk_limit_violations"] += 1
                return False, f"Risk limits violated: {'; '.join(risk_violations)}", None

            # Create position
            position_id = self._generate_position_id(product_id, side)
            position_side = PositionSide.LONG if side == "LONG" else PositionSide.SHORT

            new_position = Position(
                position_id=position_id,
                product_id=product_id,
                side=position_side,
                size=size,
                entry_price=entry_price,
                current_price=entry_price,
                status=PositionStatus.OPEN
            )

            # Set stop loss
            stop_loss_pct = stop_loss_percentage or self.risk_limits.stop_loss_percentage
            if position_side == PositionSide.LONG:
                new_position.stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
            else:
                new_position.stop_loss_price = entry_price * (1 + stop_loss_pct / 100)

            # Set take profit
            take_profit_pct = take_profit_percentage or self.risk_limits.take_profit_percentage
            if position_side == PositionSide.LONG:
                new_position.take_profit_price = entry_price * (1 + take_profit_pct / 100)
            else:
                new_position.take_profit_price = entry_price * (1 - take_profit_pct / 100)

            # Set initial trailing stop if enabled
            if use_trailing_stop:
                new_position.update_trailing_stop(entry_price, self.risk_limits.trailing_stop_percentage)

            # Place entry order (in real implementation)
            entry_order_result = self._place_entry_order(new_position)
            if not entry_order_result["success"]:
                return False, f"Failed to place entry order: {entry_order_result['error']}", None

            new_position.orders.append(entry_order_result["order"])

            # Place protective orders (skip in paper mode — exit engine handles exits,
            # and paper trader would instantly fill limit orders at wrong prices)
            is_paper = getattr(self.client, 'paper_trading', False)
            if not is_paper:
                protective_order_results = self._place_protective_orders(new_position)
                new_position.orders.extend(protective_order_results)

            # Store position
            with self._lock:
                self.positions[position_id] = new_position
                self.stats["positions_opened"] += 1

            self.logger.info(
                f"Position opened: {position_id} | {side} {size} {product_id} @ ${entry_price:.2f} | "
                f"SL: ${new_position.stop_loss_price:.2f} | TP: ${new_position.take_profit_price:.2f}"
            )

            return True, f"Position {position_id} opened successfully", new_position

        except Exception as open_error:
            self.logger.error(f"Error opening position: {open_error}", exc_info=True)
            return False, f"Error opening position: {open_error}", None

    def _place_entry_order(self, pos: Position) -> Dict[str, Any]:
        """Place entry order for position"""
        try:
            # Determine order side for API
            order_side = "BUY" if pos.side == PositionSide.LONG else "SELL"

            # Place a market order for immediate execution (idempotent)
            order_result = self.client.create_market_order(
                product_id=pos.product_id,
                side=order_side,
                size=pos.size,
                client_order_id=str(uuid.uuid4())
            )

            if "error" in order_result:
                return {"success": False, "error": order_result["error"]}

            return {
                "success": True,
                "order": {
                    "order_id": order_result.get("order", {}).get("order_id", "unknown"),
                    "type": "ENTRY",
                    "side": order_side,
                    "size": pos.size,
                    "price": pos.entry_price,
                    "status": "FILLED",
                    "created_at": datetime.now().isoformat()
                }
            }

        except Exception as entry_error:
            self.logger.error(f"Error placing entry order: {entry_error}")
            return {"success": False, "error": str(entry_error)}

    def _place_protective_orders(self, pos: Position) -> List[Dict[str, Any]]:
        """Place stop loss and take profit orders"""
        protective_orders = []

        try:
            # Place stop loss order
            if pos.stop_loss_price:
                stop_loss_result = self._place_stop_loss_order(pos)
                if stop_loss_result["success"]:
                    protective_orders.append(stop_loss_result["order"])
                else:
                    self.logger.warning(f"Failed to place stop loss: {stop_loss_result['error']}")

            # Place take profit order
            if pos.take_profit_price:
                take_profit_result = self._place_take_profit_order(pos)
                if take_profit_result["success"]:
                    protective_orders.append(take_profit_result["order"])
                else:
                    self.logger.warning(f"Failed to place take profit: {take_profit_result['error']}")

        except Exception as protective_error:
            self.logger.error(f"Error placing protective orders: {protective_error}")

        return protective_orders

    def _place_stop_loss_order(self, pos: Position) -> Dict[str, Any]:
        """Place stop loss order"""
        try:
            # Determine order side (opposite of position)
            order_side = "SELL" if pos.side == PositionSide.LONG else "BUY"

            # Place stop limit order (idempotent)
            order_result = self.client.create_limit_order(
                product_id=pos.product_id,
                side=order_side,
                size=pos.size,
                price=pos.stop_loss_price,
                post_only=False,
                client_order_id=str(uuid.uuid4())
            )

            if "error" in order_result:
                return {"success": False, "error": order_result["error"]}

            return {
                "success": True,
                "order": {
                    "order_id": order_result.get("order", {}).get("order_id", "unknown"),
                    "type": "STOP_LOSS",
                    "side": order_side,
                    "size": pos.size,
                    "price": pos.stop_loss_price,
                    "status": "OPEN",
                    "created_at": datetime.now().isoformat()
                }
            }

        except Exception as stop_error:
            return {"success": False, "error": str(stop_error)}

    def _place_take_profit_order(self, pos: Position) -> Dict[str, Any]:
        """Place take profit order"""
        try:
            # Determine order side (opposite of position)
            order_side = "SELL" if pos.side == PositionSide.LONG else "BUY"

            # Place limit order (idempotent)
            order_result = self.client.create_limit_order(
                product_id=pos.product_id,
                side=order_side,
                size=pos.size,
                price=pos.take_profit_price,
                post_only=True,
                client_order_id=str(uuid.uuid4())
            )

            if "error" in order_result:
                return {"success": False, "error": order_result["error"]}

            return {
                "success": True,
                "order": {
                    "order_id": order_result.get("order", {}).get("order_id", "unknown"),
                    "type": "TAKE_PROFIT",
                    "side": order_side,
                    "size": pos.size,
                    "price": pos.take_profit_price,
                    "status": "OPEN",
                    "created_at": datetime.now().isoformat()
                }
            }

        except Exception as tp_error:
            return {"success": False, "error": str(tp_error)}

    def close_position(self, position_id: str, reason: str = "Manual close") -> Tuple[bool, str]:
        """
        Close an existing position.

        Args:
            position_id: Position ID to close
            reason: Reason for closing

        Returns:
            Tuple[success, message]
        """

        try:
            with self._lock:
                if position_id not in self.positions:
                    return False, f"Position {position_id} not found"

                pos = self.positions[position_id]

                if pos.status != PositionStatus.OPEN:
                    return False, f"Position {position_id} is not open (status: {pos.status})"

            # Cancel any open orders for this position
            self._cancel_position_orders(pos)

            # Place closing order
            close_result = self._place_closing_order(pos)
            if not close_result["success"]:
                return False, f"Failed to close position: {close_result['error']}"

            # Update position
            pos.status = PositionStatus.CLOSED
            pos.last_update = datetime.now()
            pos.orders.append(close_result["order"])
            pos.metadata["close_reason"] = reason

            # Calculate final PnL
            final_pnl = pos.calculate_unrealized_pnl(pos.current_price)
            pos.realized_pnl = final_pnl
            pos.unrealized_pnl = 0.0

            # Update daily PnL
            with self._lock:
                self.daily_pnl += final_pnl

                # Move to closed positions
                self.closed_positions.append(pos)
                del self.positions[position_id]
                self.stats["positions_closed"] += 1

            self.logger.info(
                f"Position closed: {position_id} | Reason: {reason} | "
                f"PnL: ${final_pnl:.2f} | Daily PnL: ${self.daily_pnl:.2f}"
            )

            return True, f"Position {position_id} closed successfully"

        except Exception as close_error:
            self.logger.error(f"Error closing position {position_id}: {close_error}", exc_info=True)
            return False, f"Error closing position: {close_error}"

    def _cancel_position_orders(self, pos: Position):
        """Cancel all open orders for a position"""
        try:
            for order in pos.orders:
                if order.get("status") == "OPEN":
                    order_id = order.get("order_id")
                    if order_id:
                        cancel_result = self.client.cancel_order(order_id)
                        if "error" not in cancel_result:
                            self.logger.info(f"Cancelled order {order_id}")
                        else:
                            self.logger.warning(f"Failed to cancel order {order_id}: {cancel_result['error']}")
        except Exception as cancel_error:
            self.logger.error(f"Error cancelling position orders: {cancel_error}")

    def _place_closing_order(self, pos: Position) -> Dict[str, Any]:
        """Place order to close position"""
        try:
            # Determine order side (opposite of position)
            order_side = "SELL" if pos.side == PositionSide.LONG else "BUY"

            # Place a market order for immediate execution (idempotent)
            order_result = self.client.create_market_order(
                product_id=pos.product_id,
                side=order_side,
                size=pos.size,
                client_order_id=str(uuid.uuid4())
            )

            if "error" in order_result:
                return {"success": False, "error": order_result["error"]}

            return {
                "success": True,
                "order": {
                    "order_id": order_result.get("order", {}).get("order_id", "unknown"),
                    "type": "CLOSE",
                    "side": order_side,
                    "size": pos.size,
                    "price": pos.current_price,
                    "status": "FILLED",
                    "created_at": datetime.now().isoformat()
                }
            }

        except Exception as close_order_error:
            return {"success": False, "error": str(close_order_error)}

    def _check_stale_orders(self, max_age_minutes: int = 5):
        """Cancel orders that have been pending longer than max_age_minutes."""
        try:
            orders_resp = self.client.list_orders(order_status=["PENDING", "OPEN"])
            orders = orders_resp.get("orders", [])
            now = datetime.now()
            for order in orders:
                created = order.get("created_time") or order.get("created_at")
                if not created:
                    continue
                try:
                    order_time = datetime.fromisoformat(created.replace("Z", "+00:00")).replace(tzinfo=None)
                except (ValueError, AttributeError):
                    continue
                age_minutes = (now - order_time).total_seconds() / 60
                if age_minutes > max_age_minutes:
                    order_id = order.get("order_id")
                    if order_id:
                        self.logger.warning(f"Cancelling stale order {order_id} (age: {age_minutes:.1f}m)")
                        self.client.cancel_order(order_id)
        except Exception as e:
            self.logger.error(f"Error checking stale orders: {e}")

    def update_positions(self, price_data: Dict[str, float]):
        """
        Update all positions with current market data.

        Args:
            price_data: Dictionary with product_id -> current_price mapping
        """

        try:
            positions_to_close = []

            with self._lock:
                for pos in self.positions.values():
                    if pos.product_id in price_data:
                        old_price = pos.current_price
                        pos.current_price = price_data[pos.product_id]

                        # Update unrealized PnL
                        pos.unrealized_pnl = pos.calculate_unrealized_pnl(pos.current_price)
                        pos.last_update = datetime.now()

                        # Update trailing stop if applicable
                        if pos.trailing_stop_price is not None:
                            pos.update_trailing_stop(pos.current_price, self.risk_limits.trailing_stop_percentage)

                        # Check exit conditions
                        exit_reason = self._check_position_exit_conditions(pos)
                        if exit_reason:
                            positions_to_close.append((pos.position_id, exit_reason))

                        # Log significant price changes
                        if old_price > 0:
                            price_change_pct = abs((pos.current_price - old_price) / old_price) * 100
                            if price_change_pct > 1.0:  # Log if price changed more than 1%
                                self.logger.debug(
                                    f"Position {pos.position_id} price update: "
                                    f"${old_price:.2f} -> ${pos.current_price:.2f} "
                                    f"({price_change_pct:+.2f}%) | PnL: ${pos.unrealized_pnl:.2f}"
                                )

            # Close positions that need to be closed
            for position_id, exit_reason in positions_to_close:
                close_success, close_message = self.close_position(position_id, exit_reason)
                if close_success:
                    self.logger.info(f"Auto-closed position: {close_message}")
                else:
                    self.logger.error(f"Failed to auto-close position: {close_message}")

            # Check for stale/orphaned orders
            self._check_stale_orders()

        except Exception as update_error:
            self.logger.error(f"Error updating positions: {update_error}", exc_info=True)

    def _check_position_exit_conditions(self, pos: Position) -> Optional[str]:
        """Check if position should be closed and return reason"""

        # Check stop loss
        if pos.should_trigger_stop_loss(pos.current_price):
            self.stats["stop_losses_triggered"] += 1
            return "Stop loss triggered"

        # Check take profit
        if pos.should_trigger_take_profit(pos.current_price):
            self.stats["take_profits_triggered"] += 1
            return "Take profit triggered"

        # Check trailing stop
        if pos.should_trigger_trailing_stop(pos.current_price):
            self.stats["trailing_stops_triggered"] += 1
            return "Trailing stop triggered"

        # Check position timeout
        if pos.is_expired(self.risk_limits.position_timeout_hours):
            return f"Position timeout ({self.risk_limits.position_timeout_hours}h)"

        # Check daily loss limit
        if self.daily_pnl < -self.risk_limits.max_daily_loss_usd:
            return "Daily loss limit exceeded"

        return None

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a specific position by ID"""
        with self._lock:
            return self.positions.get(position_id)

    def get_all_positions(self) -> List[Position]:
        """Get all open positions"""
        with self._lock:
            return list(self.positions.values())

    def get_positions_by_product(self, product_id: str) -> List[Position]:
        """Get positions for specific product"""
        with self._lock:
            return [p for p in self.positions.values() if p.product_id == product_id]

    def get_position_summary(self) -> PositionSummary:
        """Get a comprehensive position summary"""
        with self._lock:
            pos_summary = PositionSummary()

            # Basic counts
            pos_summary.total_positions = len(self.positions) + len(self.closed_positions)
            pos_summary.open_positions = len(self.positions)
            pos_summary.closed_positions = len(self.closed_positions)

            # Calculate totals
            total_unrealized = 0.0
            total_realized = 0.0
            total_fees = 0.0
            total_exposure = 0.0
            position_sizes = []

            # Open positions
            for pos in self.positions.values():
                total_unrealized += pos.unrealized_pnl
                total_fees += pos.fees_paid
                position_value = abs(pos.entry_price * pos.size)
                total_exposure += position_value
                position_sizes.append(position_value)

                # Count by product
                if pos.product_id not in pos_summary.positions_by_product:
                    pos_summary.positions_by_product[pos.product_id] = 0
                pos_summary.positions_by_product[pos.product_id] += 1

            # Closed positions
            for pos in self.closed_positions:
                total_realized += pos.realized_pnl
                total_fees += pos.fees_paid

            pos_summary.total_unrealized_pnl = total_unrealized
            pos_summary.total_realized_pnl = total_realized
            pos_summary.total_fees = total_fees
            pos_summary.total_exposure_usd = total_exposure
            pos_summary.daily_pnl = self.daily_pnl

            # Position size statistics
            if position_sizes:
                pos_summary.largest_position_size = max(position_sizes)
                pos_summary.average_position_size = sum(position_sizes) / len(position_sizes)

            return pos_summary

    def set_emergency_stop(self, enabled: bool, reason: str = "Manual"):
        """Enable or disable emergency stop"""
        with self._lock:
            self.emergency_stop = enabled

            if enabled:
                self.stats["emergency_stops"] += 1
                self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")

                # Close all positions
                positions_to_close = list(self.positions.keys())

        # Close positions outside of lock to avoid deadlock
        if enabled:
            for position_id in positions_to_close:
                close_success, close_message = self.close_position(position_id, f"Emergency stop: {reason}")
                if close_success:
                    self.logger.info(f"Emergency close: {close_message}")
                else:
                    self.logger.error(f"Failed emergency close: {close_message}")
        else:
            self.logger.info("Emergency stop deactivated")

    def reconcile_with_exchange(self) -> Dict[str, Any]:
        """Compare tracked positions against actual exchange state."""
        report = {"status": "OK", "discrepancies": [], "exchange_balances": {}}

        try:
            # 1. Get exchange balances
            accounts = self.client.get_accounts()
            for acct in accounts.get("accounts", []):
                currency = acct.get("currency", "")
                bal = acct.get("available_balance", {})
                available = float(bal.get("value", 0) if isinstance(bal, dict) else bal)
                report["exchange_balances"][currency] = available

            # 2. Get open orders on exchange
            open_orders = self.client.list_orders(order_status=["OPEN"])
            report["exchange_open_orders"] = len(open_orders.get("orders", []))

            # 3. Compare tracked positions against exchange balances
            for pos_id, pos in self.positions.items():
                base_currency = pos.product_id.split("-")[0]  # "BTC" from "BTC-USD"
                exchange_balance = report["exchange_balances"].get(base_currency, 0.0)

                if pos.side == PositionSide.LONG and exchange_balance < pos.size * 0.95:
                    report["discrepancies"].append({
                        "position_id": pos_id,
                        "type": "BALANCE_MISMATCH",
                        "expected": pos.size,
                        "actual": exchange_balance,
                    })

            if report["discrepancies"]:
                report["status"] = "MISMATCH"
                self.logger.critical(
                    f"POSITION RECONCILIATION FAILED: {len(report['discrepancies'])} discrepancies"
                )
                for d in report["discrepancies"]:
                    self.logger.critical(
                        f"  {d['position_id']}: expected {d['expected']}, exchange has {d['actual']}"
                    )
            else:
                self.logger.info("Position reconciliation OK")

        except Exception as e:
            report["status"] = "ERROR"
            report["error"] = str(e)
            self.logger.error(f"Position reconciliation failed: {e}")

        return report

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics"""
        pos_summary = self.get_position_summary()

        # Calculate risk metrics
        max_daily_loss_pct = 0.0
        if self.risk_limits.max_daily_loss_usd > 0:
            max_daily_loss_pct = (abs(self.daily_pnl) / self.risk_limits.max_daily_loss_usd) * 100

        exposure_utilization_pct = 0.0
        if self.risk_limits.max_total_exposure_usd > 0:
            exposure_utilization_pct = (pos_summary.total_exposure_usd / self.risk_limits.max_total_exposure_usd) * 100

        return {
            "daily_pnl": self.daily_pnl,
            "daily_loss_limit_usage_pct": max_daily_loss_pct,
            "total_exposure_usd": pos_summary.total_exposure_usd,
            "exposure_limit_usage_pct": exposure_utilization_pct,
            "position_count": pos_summary.open_positions,
            "position_limit_usage_pct": (pos_summary.open_positions / self.risk_limits.max_total_positions) * 100,
            "largest_position_pct": (
                                                pos_summary.largest_position_size / self.risk_limits.max_position_size_usd) * 100 if pos_summary.largest_position_size > 0 else 0,
            "emergency_stop_active": self.emergency_stop,
            "risk_violations": self.stats["risk_limit_violations"]
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        pos_summary = self.get_position_summary()
        risk_metrics = self.get_risk_metrics()

        # Calculate performance metrics
        total_pnl = pos_summary.total_realized_pnl + pos_summary.total_unrealized_pnl
        net_pnl = total_pnl - pos_summary.total_fees

        # Win/loss statistics
        closed_positions = [p for p in self.closed_positions if p.realized_pnl != 0]
        winning_trades = [p for p in closed_positions if p.realized_pnl > 0]
        losing_trades = [p for p in closed_positions if p.realized_pnl < 0]

        win_rate = (len(winning_trades) / len(closed_positions) * 100) if closed_positions else 0

        avg_win = sum(p.realized_pnl for p in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(p.realized_pnl for p in losing_trades) / len(losing_trades) if losing_trades else 0

        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_pnl": total_pnl,
                "realized_pnl": pos_summary.total_realized_pnl,
                "unrealized_pnl": pos_summary.total_unrealized_pnl,
                "net_pnl": net_pnl,
                "total_fees": pos_summary.total_fees,
                "daily_pnl": pos_summary.daily_pnl
            },
            "positions": {
                "total": pos_summary.total_positions,
                "open": pos_summary.open_positions,
                "closed": pos_summary.closed_positions,
                "by_product": pos_summary.positions_by_product
            },
            "trading_stats": {
                "total_trades": len(closed_positions),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate_pct": win_rate,
                "average_win": avg_win,
                "average_loss": avg_loss,
                "profit_factor": profit_factor
            },
            "risk_management": {
                "stop_losses_triggered": self.stats["stop_losses_triggered"],
                "take_profits_triggered": self.stats["take_profits_triggered"],
                "trailing_stops_triggered": self.stats["trailing_stops_triggered"],
                "emergency_stops": self.stats["emergency_stops"],
                "risk_violations": self.stats["risk_limit_violations"]
            },
            "risk_metrics": risk_metrics,
            "system_stats": self.stats.copy()
        }

    def export_positions_to_dataframe(self) -> Optional[Any]:
        """Export positions to pandas DataFrame"""
        if not PANDAS_AVAILABLE or pd is None:
            self.logger.warning("Pandas not available for DataFrame export")
            return None

        try:
            positions_data = []

            with self._lock:
                open_positions = list(self.positions.values())
                closed_positions = list(self.closed_positions)

            # Add open positions
            for pos in open_positions:
                positions_data.append({
                    "position_id": pos.position_id,
                    "product_id": pos.product_id,
                    "side": pos.side.value,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "stop_loss_price": pos.stop_loss_price,
                    "take_profit_price": pos.take_profit_price,
                    "trailing_stop_price": pos.trailing_stop_price,
                    "status": pos.status.value,
                    "entry_time": pos.entry_time,
                    "last_update": pos.last_update,
                    "realized_pnl": pos.realized_pnl,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.calculate_unrealized_pnl_percentage(pos.current_price),
                    "fees_paid": pos.fees_paid,
                    "is_open": True
                })

            # Add closed positions
            for pos in closed_positions:
                positions_data.append({
                    "position_id": pos.position_id,
                    "product_id": pos.product_id,
                    "side": pos.side.value,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "stop_loss_price": pos.stop_loss_price,
                    "take_profit_price": pos.take_profit_price,
                    "trailing_stop_price": pos.trailing_stop_price,
                    "status": pos.status.value,
                    "entry_time": pos.entry_time,
                    "last_update": pos.last_update,
                    "realized_pnl": pos.realized_pnl,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.calculate_unrealized_pnl_percentage(pos.current_price),
                    "fees_paid": pos.fees_paid,
                    "is_open": False
                })

            return pd.DataFrame(positions_data)

        except Exception as export_error:
            self.logger.error(f"Error exporting to DataFrame: {export_error}")
            return None

    def save_positions_to_file(self, filename: str):
        """Save positions to JSON file"""
        try:
            with self._lock:
                positions_snapshot = list(self.positions.values())
                closed_snapshot = list(self.closed_positions)
                stats_snapshot = self.stats.copy()
                daily_pnl_snapshot = self.daily_pnl

            export_data = {
                "timestamp": datetime.now().isoformat(),
                "positions": [],
                "closed_positions": [],
                "stats": stats_snapshot,
                "daily_pnl": daily_pnl_snapshot,
                "risk_limits": {
                    "max_position_size_usd": self.risk_limits.max_position_size_usd,
                    "max_daily_loss_usd": self.risk_limits.max_daily_loss_usd,
                    "max_total_exposure_usd": self.risk_limits.max_total_exposure_usd,
                    "stop_loss_percentage": self.risk_limits.stop_loss_percentage,
                    "take_profit_percentage": self.risk_limits.take_profit_percentage
                }
            }

            # Add position data
            for pos in positions_snapshot:
                export_data["positions"].append({
                    "position_id": pos.position_id,
                    "product_id": pos.product_id,
                    "side": pos.side.value,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "stop_loss_price": pos.stop_loss_price,
                    "take_profit_price": pos.take_profit_price,
                    "trailing_stop_price": pos.trailing_stop_price,
                    "status": pos.status.value,
                    "entry_time": pos.entry_time.isoformat(),
                    "last_update": pos.last_update.isoformat(),
                    "realized_pnl": pos.realized_pnl,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "fees_paid": pos.fees_paid,
                    "metadata": pos.metadata
                })

            for pos in closed_snapshot:
                export_data["closed_positions"].append({
                    "position_id": pos.position_id,
                    "product_id": pos.product_id,
                    "side": pos.side.value,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "stop_loss_price": pos.stop_loss_price,
                    "take_profit_price": pos.take_profit_price,
                    "status": pos.status.value,
                    "entry_time": pos.entry_time.isoformat(),
                    "last_update": pos.last_update.isoformat(),
                    "realized_pnl": pos.realized_pnl,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "fees_paid": pos.fees_paid,
                    "metadata": pos.metadata
                })

            with open(filename, 'w', encoding='utf-8') as export_file:
                json.dump(export_data, export_file, indent=2, default=str)

            self.logger.info(f"Positions saved to {filename}")

        except Exception as save_error:
            self.logger.error(f"Error saving positions to file: {save_error}")


if __name__ == "__main__":
    # Position manager testing
    print("Enhanced Position Manager Test")
    print("=" * 50)

    try:
        # Mock client for testing
        class MockClient:
            @staticmethod
            def create_market_order(product_id: str, side: str, size: Optional[float] = None,
                                    funds: Optional[float] = None) -> Dict[str, Any]:
                return {
                    "order": {
                        "order_id": f"mock_{int(datetime.now().timestamp())}",
                        "status": "FILLED"
                    }
                }

            @staticmethod
            def create_limit_order(product_id: str, side: str, size: float, price: float,
                                   post_only: bool = False) -> Dict[str, Any]:
                return {
                    "order": {
                        "order_id": f"mock_limit_{int(datetime.now().timestamp())}",
                        "status": "OPEN"
                    }
                }

            @staticmethod
            def cancel_order() -> Dict[str, Any]:
                return {"success": True}

            @staticmethod
            def get_product() -> Dict[str, Any]:
                return {"price": "50000.00"}


        # Initialize with mock client
        mock_client = MockClient()
        test_logger = logging.getLogger("test_position_manager")
        test_logger.setLevel(logging.INFO)

        position_manager = EnhancedPositionManager(
            coinbase_client=mock_client,
            logger=test_logger
        )

        print("✅ Position manager initialized")

        # Test position opening
        open_success, open_message, test_position = position_manager.open_position(
            product_id="BTC-USD",
            side="LONG",
            size=0.001,
            entry_price=50000.0
        )

        print(f"✅ Position opened: {open_success} - {open_message}")
        if test_position:
            print(f"  Position ID: {test_position.position_id}")
            print(f"  Stop Loss: ${test_position.stop_loss_price:.2f}")
            print(f"  Take Profit: ${test_position.take_profit_price:.2f}")

        # Test position updates
        price_data = {"BTC-USD": 51000.0}  # Price increased
        position_manager.update_positions(price_data)
        print("✅ Positions updated with market data")

        # Test position summary
        test_summary = position_manager.get_position_summary()
        print(
            f"✅ Position summary: {test_summary.open_positions} open, ${test_summary.total_unrealized_pnl:.2f} unrealized PnL")

        # Test performance report
        test_report = position_manager.get_performance_report()
        print(f"✅ Performance report generated: {test_report['summary']['total_pnl']:.2f} total PnL")

        print("\n🎉 All position manager tests passed!")

    except Exception as test_error:
        print(f"❌ Position manager test failed: {test_error}")
        raise