import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time


class GlassnodeClient:
    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config['api_key']
        self.base_url = "https://api.glassnode.com/v1/metrics"
        self.logger = logging.getLogger(__name__)

        # Test API connection
        if not self._test_connection():
            self.logger.error("Failed to connect to Glassnode API")

    def _test_connection(self) -> bool:
        """Test API connection"""
        try:
            response = requests.get(
                f"{self.base_url}/market/price_usd_close",
                params={
                    'a': 'BTC',
                    'api_key': self.api_key,
                    'f': 'JSON',
                    's': int((datetime.now() - timedelta(days=1)).timestamp()),
                    'u': int(datetime.now().timestamp())
                },
                timeout=10
            )
            return response.status_code == 200
        except:
            return False

    def get_onchain_metrics(self) -> Dict:
        """Get comprehensive on-chain metrics"""
        try:
            metrics = {}

            # Define metrics to collect
            metric_endpoints = {
                'active_addresses': 'addresses/active_count',
                'transaction_count': 'transactions/count',
                'hash_rate': 'mining/hash_rate_mean',
                'difficulty': 'mining/difficulty_latest',
                'exchange_inflow': 'transactions/transfers_volume_exchanges_net',
                'hodl_waves': 'supply/hodl_waves',
                'realized_cap': 'market/marketcap_realized_usd',
                'mvrv_ratio': 'market/mvrv',
                'nvt_ratio': 'indicators/nvt',
                'sopr': 'indicators/sopr'
            }

            current_timestamp = int(datetime.now().timestamp())
            yesterday_timestamp = int((datetime.now() - timedelta(days=1)).timestamp())

            for metric_name, endpoint in metric_endpoints.items():
                try:
                    value = self._fetch_metric(endpoint, yesterday_timestamp, current_timestamp)
                    metrics[metric_name] = value
                    time.sleep(0.1)  # Rate limiting
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {metric_name}: {e}")
                    metrics[metric_name] = None

            # Calculate composite scores
            metrics['onchain_health_score'] = self._calculate_health_score(metrics)
            metrics['network_momentum'] = self._calculate_momentum(metrics)
            metrics['timestamp'] = datetime.utcnow().isoformat()

            return metrics

        except Exception as e:
            self.logger.error(f"Error fetching on-chain metrics: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}

    def _fetch_metric(self, endpoint: str, start_time: int, end_time: int) -> Optional[float]:
        """Fetch individual metric from Glassnode"""
        try:
            response = requests.get(
                f"{self.base_url}/{endpoint}",
                params={
                    'a': 'BTC',
                    'api_key': self.api_key,
                    'f': 'JSON',
                    's': start_time,
                    'u': end_time
                },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return float(data[-1]['v'])  # Get latest value

            return None

        except Exception as e:
            self.logger.error(f"Error fetching metric {endpoint}: {e}")
            return None

    def _calculate_health_score(self, metrics: Dict) -> float:
        """Calculate overall network health score"""
        try:
            score = 0
            weight_sum = 0

            # Active addresses (weight: 0.2)
            if metrics.get('active_addresses'):
                # Normalize based on typical range (200k-800k)
                normalized = min(max((metrics['active_addresses'] - 200000) / 600000, 0), 1)
                score += normalized * 0.2
                weight_sum += 0.2

            # Hash rate trend (weight: 0.15)
            if metrics.get('hash_rate'):
                # Assuming higher hash rate is better (simplified)
                normalized = min(metrics['hash_rate'] / 200e18, 1)  # 200 EH/s as reference
                score += normalized * 0.15
                weight_sum += 0.15

            # MVRV ratio (weight: 0.2)
            if metrics.get('mvrv_ratio'):
                # MVRV around 1-3 is healthy
                mvrv = metrics['mvrv_ratio']
                if 1 <= mvrv <= 3:
                    normalized = 1.0
                elif mvrv < 1:
                    normalized = mvrv  # Below 1 is concerning
                else:
                    normalized = max(1 - (mvrv - 3) / 7, 0)  # Above 3 starts to be risky
                score += normalized * 0.2
                weight_sum += 0.2

            # NVT ratio (weight: 0.15)
            if metrics.get('nvt_ratio'):
                # Lower NVT is generally better (more utility)
                nvt = metrics['nvt_ratio']
                normalized = max(1 - nvt / 100, 0)  # Simplified normalization
                score += normalized * 0.15
                weight_sum += 0.15

            # SOPR (weight: 0.3)
            if metrics.get('sopr'):
                # SOPR around 1.0-1.1 is healthy
                sopr = metrics['sopr']
                if 1.0 <= sopr <= 1.1:
                    normalized = 1.0
                else:
                    normalized = max(1 - abs(sopr - 1.05) / 0.5, 0)
                score += normalized * 0.3
                weight_sum += 0.3

            return score / weight_sum if weight_sum > 0 else 0.5

        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")
            return 0.5

    def _calculate_momentum(self, metrics: Dict) -> float:
        """Calculate network momentum score"""
        try:
            # This is a simplified momentum calculation
            # In practice, you'd want historical data for proper momentum

            momentum_factors = []

            # Transaction count momentum
            if metrics.get('transaction_count'):
                # Simplified: assume higher tx count = higher momentum
                normalized_tx = min(metrics['transaction_count'] / 300000, 1)
                momentum_factors.append(normalized_tx)

            # Active addresses momentum
            if metrics.get('active_addresses'):
                normalized_addr = min(metrics['active_addresses'] / 500000, 1)
                momentum_factors.append(normalized_addr)

            # Exchange flow (negative inflow = positive momentum)
            if metrics.get('exchange_inflow'):
                # Negative inflow is bullish
                flow_score = max(1 + metrics['exchange_inflow'] / 10000, 0)
                momentum_factors.append(min(flow_score, 1))

            return sum(momentum_factors) / len(momentum_factors) if momentum_factors else 0.5

        except Exception as e:
            self.logger.error(f"Error calculating momentum: {e}")
            return 0.5

    def get_whale_activity(self) -> Dict:
        """Get whale activity metrics"""
        try:
            # Fetch large transaction metrics
            whale_metrics = {}

            current_time = int(datetime.now().timestamp())
            day_ago = int((datetime.now() - timedelta(days=1)).timestamp())

            # Large transactions
            large_tx_value = self._fetch_metric('transactions/transfers_volume_sum_1k_inf', day_ago, current_time)
            whale_metrics['large_tx_volume'] = large_tx_value

            # Exchange flows
            exchange_inflow = self._fetch_metric('transactions/transfers_volume_exchanges_net', day_ago, current_time)
            whale_metrics['exchange_net_flow'] = exchange_inflow

            # Calculate whale activity score
            whale_score = 0.5  # Default neutral

            if large_tx_value and exchange_inflow is not None:
                # High large tx volume with negative exchange flow = bullish whale activity
                if large_tx_value > 1000000 and exchange_inflow < 0:  # 1M+ BTC moved, net outflow
                    whale_score = 0.8
                elif large_tx_value > 500000 and exchange_inflow < 0:
                    whale_score = 0.7
                elif exchange_inflow > 1000000:  # Large exchange inflow = bearish
                    whale_score = 0.2

            whale_metrics['whale_activity_score'] = whale_score
            whale_metrics['timestamp'] = datetime.utcnow().isoformat()

            return whale_metrics

        except Exception as e:
            self.logger.error(f"Error fetching whale activity: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}