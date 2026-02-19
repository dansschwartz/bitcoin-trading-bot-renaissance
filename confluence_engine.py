"""
Confluence Engine: Non-linear Signal Interaction & Meta-Learning
Identifies when a combination of signals creates a "Confluence of Edges" 
that is more predictive than the sum of its parts.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

class ConfluenceEngine:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        # Confluence rules — signals must match those actually generated in the pipeline
        self.rules = [
            {
                'name': 'OrderFlow_Technical_Convergence',
                'signals': ['order_flow', 'macd', 'rsi'],
                'logic': 'all_same_sign',
                'boost': 0.15
            },
            {
                'name': 'Bollinger_Volume_MeanReversion',
                'signals': ['bollinger', 'volume'],
                'logic': 'divergence',
                'boost': 0.12
            },
            {
                'name': 'StatArb_OrderFlow_Confirmation',
                'signals': ['stat_arb', 'order_flow'],
                'logic': 'high_correlation',
                'boost': 0.18
            },
            {
                'name': 'ML_Technical_Agreement',
                'signals': ['ml_ensemble', 'macd', 'bollinger'],
                'logic': 'all_same_sign',
                'boost': 0.15
            },
            {
                'name': 'Multi_Signal_Alignment',
                'signals': ['order_flow', 'macd', 'rsi', 'bollinger'],
                'logic': 'all_same_sign',
                'boost': 0.20
            },
        ]
        self.logger.info("🏛️ Confluence Engine initialized: Non-linear Meta-Learning ready.")

    def calculate_confluence_boost(self, signals: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyzes signal interactions and returns a non-linear boost factor.
        """
        active_rules = []
        total_boost = 0.0

        # Create a hardened copy of signals to ensure all values are standard floats
        # This is critical to prevent sequence-multiplication errors in fusion
        hardened_signals = {}
        for k, v in signals.items():
            try:
                temp_v = v
                if temp_v is None:
                    hardened_signals[k] = 0.0
                    continue
                # Standard numpy/sequence unpacking
                while hasattr(temp_v, '__iter__') and not isinstance(temp_v, (str, bytes, dict)):
                    if hasattr(temp_v, '__len__') and len(temp_v) > 0:
                        temp_v = temp_v[0]
                    else:
                        temp_v = 0.0
                        break
                if hasattr(temp_v, 'item'): 
                    temp_v = temp_v.item()
                hardened_signals[k] = float(temp_v)
            except:
                hardened_signals[k] = 0.0

        evaluated_rules = []
        for rule in self.rules:
            try:
                boost_val = self._evaluate_rule(rule, hardened_signals)
                fired = boost_val > 0
                evaluated_rules.append({
                    'name': rule['name'],
                    'signals': rule['signals'],
                    'boost': float(rule['boost']),
                    'fired': fired,
                })
                if fired:
                    active_rules.append({'name': rule['name'], 'boost': float(boost_val)})
                    total_boost += float(boost_val)
            except Exception as e:
                self.logger.warning(f"Error evaluating rule {rule.get('name')}: {e}")

        # Cap total confluence boost at 30% to prevent overfitting
        total_boost = min(float(total_boost), 0.30)

        return {
            'total_confluence_boost': float(total_boost),
            'active_rules': active_rules,
            'evaluated_rules': evaluated_rules,
            'signal_count': len(hardened_signals),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _evaluate_rule(self, rule: Dict[str, Any], signals: Dict[str, float]) -> float:
        """Evaluates a specific confluence rule against current signals."""
        try:
            rule_signals = [signals.get(s, 0.0) for s in rule['signals']]
            
            # 1. All Same Sign (Trend Confirmation)
            if rule['logic'] == 'all_same_sign':
                if len(rule_signals) >= 2 and all(s > 0.1 for s in rule_signals):
                    return rule['boost']
                if len(rule_signals) >= 2 and all(s < -0.1 for s in rule_signals):
                    return rule['boost']

            # 2. Divergence (Mean Reversion)
            elif rule['logic'] == 'divergence':
                # Bollinger band extreme + volume confirmation
                boll = signals.get('bollinger', 0.0)
                vol = signals.get('volume', 0.0)
                if abs(boll) > 0.6 and np.sign(boll) != np.sign(vol):
                    return rule['boost']

            # 3. High Correlation (Confirmation)
            elif rule['logic'] == 'high_correlation':
                s1 = signals.get(rule['signals'][0], 0.0)
                s2 = signals.get(rule['signals'][1], 0.0)
                if np.sign(s1) == np.sign(s2) and abs(s1) > 0.3 and abs(s2) > 0.3:
                    return rule['boost']

            # 4. Both Strong (Alpha Multiplier)
            elif rule['logic'] == 'both_strong':
                if all(abs(s) > 0.5 for s in rule_signals):
                    return rule['boost']

            return 0.0
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule['name']}: {e}")
            return 0.0
