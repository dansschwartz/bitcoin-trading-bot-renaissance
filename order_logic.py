"""
Order Logic Module

Defines data structures and logic for microstructure analysis (OrderFlowData,
OrderBookDepthData) as well as any combined signals or risk management.

Provides sophisticated order flow and depth analysis for trading decisions.
"""

import time
import logging
from typing import Optional

# Self-contained logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('order_logic')


class OrderFlowData:
    """
    Tracks short-term buy vs sell volume (market takers).

    This class provides sophisticated order flow analysis by tracking
    the imbalance between buy and sell volumes over time periods.
    Reset periodically (e.g. every minute) if desired.

    Attributes:
        buy_volume (float): Total buy volume tracked
        sell_volume (float): Total sell volume tracked
        last_reset_ts (float): Timestamp of last reset
    """

    def __init__(self) -> None:
        """Initialize OrderFlowData with zero volumes."""
        self.buy_volume: float = 0.0
        self.sell_volume: float = 0.0
        self.last_reset_ts: float = time.time()

        logger.info("OrderFlowData initialized")

    def add_buy(self, amount: float) -> None:
        """
        Add buy volume to tracking.

        Args:
            amount: Volume amount to add to buy side
        """
        if amount < 0:
            logger.warning(f"Negative buy amount provided: {amount}")
            return

        self.buy_volume += amount
        logger.debug(
            f"Added buy volume: {amount}, total_buy: {self.buy_volume}"
        )

    def add_sell(self, amount: float) -> None:
        """
        Add sell volume to tracking.

        Args:
            amount: Volume amount to add to sell side
        """
        if amount < 0:
            logger.warning(f"Negative sell amount provided: {amount}")
            return

        self.sell_volume += amount
        logger.debug(
            f"Added sell volume: {amount}, total_sell: {self.sell_volume}"
        )

    def reset(self) -> None:
        """Reset volume counters and update timestamp."""
        time_since_reset = time.time() - self.last_reset_ts

        logger.info(
            f"Resetting OrderFlowData - old_buy: {self.buy_volume}, "
            f"old_sell: {self.sell_volume}, "
            f"time_since_reset: {time_since_reset:.2f}s"
        )

        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.last_reset_ts = time.time()

    def net_flow(self) -> float:
        """
        Calculate the raw net flow difference.

        Returns:
            float: buy_volume - sell_volume
        """
        net = self.buy_volume - self.sell_volume
        logger.debug(f"Net flow calculated: {net}")
        return net

    def flow_imbalance(self) -> float:
        """
        Calculate flow imbalance ratio for microstructure signals.

        A ratio often used for microstructure analysis:
        (buy_volume - sell_volume) / (buy_volume + sell_volume)

        Returns:
            float: Imbalance ratio between -1 and 1
                  Positive = more buying pressure
                  Negative = more selling pressure
                  0 = balanced or no volume
        """
        total_volume = self.buy_volume + self.sell_volume

        if total_volume == 0.0:
            logger.debug("No volume for imbalance calculation")
            return 0.0

        imbalance = (self.buy_volume - self.sell_volume) / total_volume
        logger.debug(
            f"Flow imbalance: {imbalance:.4f} "
            f"(buy: {self.buy_volume}, sell: {self.sell_volume})"
        )

        return imbalance

    def get_statistics(self) -> dict:
        """
        Get comprehensive flow statistics.

        Returns:
            dict: Complete flow analysis data
        """
        total_volume = self.buy_volume + self.sell_volume
        time_active = time.time() - self.last_reset_ts

        stats = {
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'total_volume': total_volume,
            'net_flow': self.net_flow(),
            'flow_imbalance': self.flow_imbalance(),
            'time_active_seconds': time_active,
            'buy_percentage': (self.buy_volume / total_volume * 100) if total_volume > 0 else 0,
            'sell_percentage': (self.sell_volume / total_volume * 100) if total_volume > 0 else 0
        }

        return stats


class OrderBookDepthData:
    """
    Stores and analyzes order book depth information.

    This class tracks bid/ask volumes and prices to detect
    institutional activity and market microstructure changes.

    Attributes:
        bid_volume (float): Total volume on bid side
        ask_volume (float): Total volume on ask side
        bid_price (Optional[float]): Best bid price
        ask_price (Optional[float]): Best ask price
    """

    def __init__(self) -> None:
        """Initialize OrderBookDepthData with empty values."""
        self.bid_volume: float = 0.0
        self.ask_volume: float = 0.0
        self.bid_price: Optional[float] = None
        self.ask_price: Optional[float] = None
        self.last_update_ts: float = time.time()

        logger.info("OrderBookDepthData initialized")

    def update(
            self,
            bid_vol: float,
            ask_vol: float,
            bid_price: float,
            ask_price: float
    ) -> None:
        """
        Update order book depth data.

        Args:
            bid_vol: Total bid side volume
            ask_vol: Total ask side volume
            bid_price: Best bid price
            ask_price: Best ask price
        """
        # Validation
        if bid_vol < 0 or ask_vol < 0:
            logger.warning(f"Negative volumes provided: bid={bid_vol}, ask={ask_vol}")
            return

        if bid_price <= 0 or ask_price <= 0:
            logger.warning(f"Invalid prices provided: bid={bid_price}, ask={ask_price}")
            return

        if bid_price >= ask_price:
            logger.warning(f"Crossed book detected: bid={bid_price} >= ask={ask_price}")

        self.bid_volume = bid_vol
        self.ask_volume = ask_vol
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.last_update_ts = time.time()

        logger.debug(
            f"OrderBook updated - bid_vol: {bid_vol}, ask_vol: {ask_vol}, "
            f"bid_price: {bid_price}, ask_price: {ask_price}"
        )

    def depth_imbalance(self) -> float:
        """
        Calculate order book depth imbalance ratio.

        Similar ratio to flow imbalance but for order book:
        (bid_volume - ask_volume) / (bid_volume + ask_volume)

        Returns:
            float: Depth imbalance ratio between -1 and 1
                  Positive = more bid support (bullish pressure)
                  Negative = more ask resistance (bearish pressure)
                  0 = balanced or no volume
        """
        total_volume = self.bid_volume + self.ask_volume

        if total_volume == 0.0:
            logger.debug("No volume for depth imbalance calculation")
            return 0.0

        imbalance = (self.bid_volume - self.ask_volume) / total_volume
        logger.debug(
            f"Depth imbalance: {imbalance:.4f} "
            f"(bid_vol: {self.bid_volume}, ask_vol: {self.ask_volume})"
        )

        return imbalance

    def get_spread(self) -> Optional[float]:
        """
        Calculate bid-ask spread.

        Returns:
            Optional[float]: Spread in price units, None if prices not set
        """
        if self.bid_price is None or self.ask_price is None:
            return None

        spread = self.ask_price - self.bid_price
        logger.debug(f"Spread calculated: {spread}")
        return spread

    def get_mid_price(self) -> Optional[float]:
        """
        Calculate mid-price.

        Returns:
            Optional[float]: Mid-price, None if prices not set
        """
        if self.bid_price is None or self.ask_price is None:
            return None

        mid_price = (self.bid_price + self.ask_price) / 2.0
        logger.debug(f"Mid price calculated: {mid_price}")
        return mid_price

    def get_statistics(self) -> dict:
        """
        Get comprehensive depth statistics.

        Returns:
            dict: Complete depth analysis data
        """
        total_volume = self.bid_volume + self.ask_volume
        time_since_update = time.time() - self.last_update_ts

        stats = {
            'bid_volume': self.bid_volume,
            'ask_volume': self.ask_volume,
            'total_volume': total_volume,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'depth_imbalance': self.depth_imbalance(),
            'spread': self.get_spread(),
            'mid_price': self.get_mid_price(),
            'bid_percentage': (self.bid_volume / total_volume * 100) if total_volume > 0 else 0,
            'ask_percentage': (self.ask_volume / total_volume * 100) if total_volume > 0 else 0,
            'seconds_since_update': time_since_update
        }

        return stats

    def is_data_fresh(self, max_age_seconds: float = 60.0) -> bool:
        """
        Check if order book data is fresh.

        Args:
            max_age_seconds: Maximum age in seconds to consider fresh

        Returns:
            bool: True if data is fresh, False otherwise
        """
        age = time.time() - self.last_update_ts
        is_fresh = age <= max_age_seconds

        if not is_fresh:
            logger.warning(f"Order book data is stale: {age:.1f}s old")

        return is_fresh


class MicrostructureAnalyzer:
    """
    Combined analyzer for order flow and depth data.

    This class provides high-level microstructure analysis by combining
    order flow and order book depth information for trading signals.
    """

    def __init__(self) -> None:
        """Initialize microstructure analyzer."""
        self.order_flow = OrderFlowData()
        self.order_book = OrderBookDepthData()
        logger.info("MicrostructureAnalyzer initialized")

    def get_combined_signal(self) -> dict:
        """
        Generate combined microstructure signal.

        Returns:
            dict: Combined analysis with signal strength
        """
        flow_imbalance = self.order_flow.flow_imbalance()
        depth_imbalance = self.order_book.depth_imbalance()

        # Simple combined signal - can be enhanced
        combined_score = (flow_imbalance + depth_imbalance) / 2.0

        signal_strength = "neutral"
        if abs(combined_score) > 0.6:
            signal_strength = "strong"
        elif abs(combined_score) > 0.3:
            signal_strength = "moderate"

        direction = "bullish" if combined_score > 0 else "bearish"

        signal = {
            'flow_imbalance': flow_imbalance,
            'depth_imbalance': depth_imbalance,
            'combined_score': combined_score,
            'direction': direction,
            'strength': signal_strength,
            'timestamp': time.time()
        }

        logger.info(
            f"Combined microstructure signal: {direction} ({signal_strength}) "
            f"score: {combined_score:.3f}"
        )

        return signal

    def reset_flow_data(self) -> None:
        """Reset order flow data for new period."""
        self.order_flow.reset()

    def get_full_analysis(self) -> dict:
        """
        Get complete microstructure analysis.

        Returns:
            dict: Full analysis including all statistics
        """
        analysis = {
            'order_flow_stats': self.order_flow.get_statistics(),
            'order_book_stats': self.order_book.get_statistics(),
            'combined_signal': self.get_combined_signal(),
            'analysis_timestamp': time.time()
        }

        return analysis


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Enhanced Order Logic Module")

    # Test OrderFlowData
    flow = OrderFlowData()
    flow.add_buy(100.5)
    flow.add_sell(85.2)
    print(f"Flow imbalance: {flow.flow_imbalance():.4f}")
    print(f"Flow statistics: {flow.get_statistics()}")

    # Test OrderBookDepthData
    book = OrderBookDepthData()
    book.update(1000.0, 800.0, 50000.0, 50005.0)
    print(f"Depth imbalance: {book.depth_imbalance():.4f}")
    print(f"Spread: {book.get_spread()}")
    print(f"Book statistics: {book.get_statistics()}")

    # Test MicrostructureAnalyzer
    analyzer = MicrostructureAnalyzer()
    analyzer.order_flow.add_buy(200.0)
    analyzer.order_flow.add_sell(150.0)
    analyzer.order_book.update(1200.0, 900.0, 50000.0, 50005.0)

    analysis = analyzer.get_full_analysis()
    print(f"Combined analysis: {analysis['combined_signal']}")

