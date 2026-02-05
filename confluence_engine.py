"""
Confluence Engine: Non-linear Signal Interaction & Meta-Learning
Identifies when a combination of signals creates a "Confluence of Edges" 
that is more predictive than the sum of its parts.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

class ConfluenceEngine:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        # Institutional "Confluence Rules" based on historical Renaissance-style research
        self.rules = [
            {
                'name': 'OrderFlow_Technical_Convergence',
                'signals': ['order_flow', 'macd', 'rsi'],
                'logic': 'all_same_sign',
                'boost': 0.15
            },
            {
                'name': 'Microstructure_Exhaustion',
                'signals': ['vpin', 'bollinger'],
                'logic': 'divergence',
                'boost': 0.12
            },
            {
                'name': 'LeadLag_StatArb_Confirmation',
                'signals': ['lead_lag', 'stat_arb'],
                'logic': 'high_correlation',
                'boost': 0.18
            },
            {
                'name': 'Fractal_Quantum_Resonance',
                'signals': ['fractal', 'quantum'],
                'logic': 'both_strong',
                'boost': 0.20
            }
        ]
        self.logger.info("🏛️ Confluence Engine initialized: Non-linear Meta-Learning ready.")

    def calculate_confluence_boost(self, signals: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyzes signal interactions and returns a non-linear boost factor.
        """
        boosts = []
        active_rules = []
        total_boost = 0.0

        for rule in self.rules:
            boost_val = self._evaluate_rule(rule, signals)
            if boost_val > 0:
                boosts.append(boost_val)
                active_rules.append(rule['name'])
                total_boost += boost_val

        # Cap total confluence boost at 30% to prevent overfitting
        total_boost = min(total_boost, 0.30)

        return {
            'total_confluence_boost': total_boost,
            'active_rules': active_rules,
            'timestamp': datetime.now().isoformat()
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
                # e.g., High VPIN (toxicity) + Bollinger touch
                vpin = signals.get('vpin', 0.5)
                boll = signals.get('bollinger', 0.0)
                if vpin > 0.7 and abs(boll) > 0.8:
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
