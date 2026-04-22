# data_validator.py
"""
Data Validator — Quality gate for market data entering the signal pipeline.
Validates price sanity, bid/ask consistency, and detects anomalous spikes.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional


class DataValidator:
    """Validates market data dicts before they enter the signal pipeline."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(f"{__name__}.DataValidator")
        self.price_history: Dict[str, list] = {}  # product_id -> recent prices
        self._validation_count = 0
        self._rejection_count = 0

    def validate_market_data(self, market_data: Dict[str, Any], product_id: str = "BTC-USD") -> bool:
        """
        Validate a market_data dict from collect_all_data().
        Returns True if data is safe to use, False if it should be skipped.
        """
        self._validation_count += 1
        try:
            ticker = market_data.get('ticker', {})
            if not ticker:
                self.logger.warning(f"[DataValidator] {product_id}: No ticker data")
                self._rejection_count += 1
                return False

            price = float(ticker.get('price', 0.0) or 0.0)
            if price <= 0:
                price = float(ticker.get('last', 0.0) or 0.0)
            if price <= 0:
                self.logger.warning(f"[DataValidator] {product_id}: Invalid price ({price})")
                self._rejection_count += 1
                return False

            # Bid/ask sanity check (if available)
            bid = float(ticker.get('bid', 0.0) or 0.0)
            ask = float(ticker.get('ask', 0.0) or 0.0)
            if bid > 0 and ask > 0 and bid >= ask:
                self.logger.warning(f"[DataValidator] {product_id}: Crossed market bid={bid} >= ask={ask}")
                self._rejection_count += 1
                return False

            # Spread sanity — spread > 5% of price is suspicious
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / price
                if spread_pct > 0.05:
                    self.logger.warning(f"[DataValidator] {product_id}: Extreme spread {spread_pct:.2%}")
                    self._rejection_count += 1
                    return False

            # Price spike detection against recent history
            hist = self.price_history.setdefault(product_id, [])
            if len(hist) >= 3:
                last_price = hist[-1]
                if last_price > 0:
                    price_change = abs(price - last_price) / last_price
                    if price_change > 0.15:  # 15% single-cycle move
                        self.logger.warning(
                            f"[DataValidator] {product_id}: Extreme price move "
                            f"{price_change:.2%} ({last_price:.2f} -> {price:.2f})"
                        )
                        self._rejection_count += 1
                        return False

            # Update price history
            hist.append(price)
            if len(hist) > 100:
                hist.pop(0)

            return True

        except Exception as e:
            self.logger.error(f"[DataValidator] {product_id}: Validation error: {e}")
            return True  # Fail open on unexpected errors — don't block trading

    def get_stats(self) -> Dict[str, Any]:
        """Return validation statistics."""
        return {
            'total_validations': self._validation_count,
            'rejections': self._rejection_count,
            'rejection_rate': self._rejection_count / max(self._validation_count, 1),
        }


class DataPipelineMonitor:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Monitor")
        self.metrics = {
            'data_points_processed': 0,
            'errors_count': 0,
            'last_update': {},
            'queue_depths': {},
            'processing_times': []
        }

    def record_data_point(self, data_type: str):
        self.metrics['data_points_processed'] += 1
        self.metrics['last_update'][data_type] = datetime.now()

    def record_error(self, error_type: str, error_msg: str):
        self.metrics['errors_count'] += 1
        self.logger.error(f"{error_type}: {error_msg}")

    def record_queue_depth(self, queue_name: str, depth: int):
        self.metrics['queue_depths'][queue_name] = depth

    def get_status_report(self) -> Dict:
        now = datetime.now()
        status = {
            'timestamp': now,
            'data_points_processed': self.metrics['data_points_processed'],
            'error_rate': self.metrics['errors_count'] / max(self.metrics['data_points_processed'], 1),
            'queue_status': self.metrics['queue_depths'],
            'last_updates': {}
        }
        for data_type, last_update in self.metrics['last_update'].items():
            age_seconds = (now - last_update).total_seconds()
            status['last_updates'][data_type] = {
                'last_update': last_update,
                'age_seconds': age_seconds,
                'status': 'healthy' if age_seconds < 300 else 'stale'
            }
        return status
