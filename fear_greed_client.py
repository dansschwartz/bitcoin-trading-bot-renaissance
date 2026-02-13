import requests
import logging
from datetime import datetime
from typing import Dict, Optional


class FearGreedClient:
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = "https://api.alternative.me/fng/"
        self.logger = logging.getLogger(__name__)

    def get_fear_greed_index(self) -> Dict:
        """Get current Fear & Greed Index"""
        try:
            response = requests.get(
                self.base_url,
                params={'limit': 1, 'format': 'json'},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                if 'data' in data and len(data['data']) > 0:
                    fng_data = data['data'][0]

                    return {
                        'fear_greed_value': int(fng_data['value']),
                        'fear_greed_classification': fng_data['value_classification'],
                        'timestamp': fng_data['timestamp'],
                        'time_until_update': fng_data.get('time_until_update'),
                        'normalized_score': self._normalize_fng_score(int(fng_data['value'])),
                        'fetch_timestamp': datetime.utcnow().isoformat()
                    }

            return {'error': f'API returned status {response.status_code}'}

        except Exception as e:
            self.logger.error(f"Error fetching Fear & Greed Index: {e}")
            return {'error': str(e), 'fetch_timestamp': datetime.utcnow().isoformat()}

    def get_historical_fng(self, days: int = 7) -> Dict:
        """Get historical Fear & Greed Index data"""
        try:
            response = requests.get(
                self.base_url,
                params={'limit': days, 'format': 'json'},
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()

                if 'data' in data:
                    historical_data = []
                    values = []

                    for entry in data['data']:
                        historical_data.append({
                            'value': int(entry['value']),
                            'classification': entry['value_classification'],
                            'timestamp': entry['timestamp']
                        })
                        values.append(int(entry['value']))

                    # Calculate trend and volatility
                    if len(values) >= 2:
                        trend = values[0] - values[-1]  # Current vs oldest
                        volatility = self._calculate_volatility(values)
                    else:
                        trend = 0
                        volatility = 0

                    return {
                        'historical_data': historical_data,
                        'average_value': sum(values) / len(values),
                        'trend': trend,
                        'volatility': volatility,
                        'current_vs_average': values[0] - (sum(values) / len(values)),
                        'fetch_timestamp': datetime.utcnow().isoformat()
                    }

            return {'error': f'API returned status {response.status_code}'}

        except Exception as e:
            self.logger.error(f"Error fetching historical Fear & Greed data: {e}")
            return {'error': str(e), 'fetch_timestamp': datetime.utcnow().isoformat()}

    def _normalize_fng_score(self, value: int) -> float:
        """Normalize Fear & Greed score to 0-1 range"""
        # 0-24: Extreme Fear (0.0-0.24)
        # 25-49: Fear (0.25-0.49)
        # 50-74: Greed (0.50-0.74)
        # 75-100: Extreme Greed (0.75-1.0)
        return value / 100.0

    def _calculate_volatility(self, values: list) -> float:
        """Calculate volatility of Fear & Greed values"""
        if len(values) < 2:
            return 0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def get_sentiment_signal(self) -> Dict:
        """Get actionable sentiment signal based on Fear & Greed Index"""
        try:
            current_data = self.get_fear_greed_index()
            historical_data = self.get_historical_fng(days=14)

            if 'error' in current_data or 'error' in historical_data:
                return {'signal': 'neutral', 'confidence': 0, 'error': 'Data unavailable'}

            current_value = current_data['fear_greed_value']
            avg_value = historical_data['average_value']
            trend = historical_data['trend']

            # Generate trading signal
            signal = 'neutral'
            confidence = 0.5

            # Extreme conditions often signal reversals
            if current_value <= 20:  # Extreme Fear
                if trend < -10:  # Getting more fearful
                    signal = 'buy'  # Contrarian signal
                    confidence = 0.8
                else:
                    signal = 'neutral'
                    confidence = 0.6

            elif current_value >= 80:  # Extreme Greed
                if trend > 10:  # Getting more greedy
                    signal = 'sell'  # Contrarian signal
                    confidence = 0.8
                else:
                    signal = 'neutral'
                    confidence = 0.6

            elif 25 <= current_value <= 75:  # Normal range
                # Follow trend in normal conditions
                if current_value > avg_value + 10:
                    signal = 'buy'
                    confidence = 0.6
                elif current_value < avg_value - 10:
                    signal = 'sell'
                    confidence = 0.6

            return {
                'signal': signal,
                'confidence': confidence,
                'current_value': current_value,
                'average_value': avg_value,
                'trend': trend,
                'reasoning': self._get_signal_reasoning(signal, current_value, trend),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error generating sentiment signal: {e}")
            return {'signal': 'neutral', 'confidence': 0, 'error': str(e)}

    def _get_signal_reasoning(self, signal: str, current_value: int, trend: float) -> str:
        """Get human-readable reasoning for the signal"""
        if signal == 'buy':
            if current_value <= 20:
                return f"Extreme fear (FNG: {current_value}) often signals buying opportunity"
            else:
                return f"Fear & Greed rising above average, potential bullish momentum"

        elif signal == 'sell':
            if current_value >= 80:
                return f"Extreme greed (FNG: {current_value}) suggests market top risk"
            else:
                return f"Fear & Greed falling below average, potential bearish momentum"

        else:
            return f"Neutral market sentiment (FNG: {current_value}), no clear signal"