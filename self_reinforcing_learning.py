"""
Self-Reinforcing Learning Engine (Step 19)
Automates feedback loops by labeling historical decisions and fine-tuning ML models.
"""

import sqlite3
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple

class SelfReinforcingLearningEngine:
    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self.lookforward_minutes = 60 # How far to look for realized PnL
        self._latest_alpha_corrs = {}
        
    async def run_learning_cycle(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes one complete self-reinforcing learning cycle.
        1. Label recent decisions with actual market outcomes.
        2. Retrain/Fine-tune models using labeled data.
        3. Discover feature importance shifts.
        """
        self.logger.info("🧠 Initiating Self-Reinforcing Learning Cycle...")
        
        # 1. Label history
        labeled_data = self._label_historical_decisions()
        if not labeled_data:
            self.logger.info("Insufficient labeled data for learning cycle.")
            return {"status": "skipped", "reason": "insufficient_data"}
            
        # 2. Online model calibration (Fine-tuning)
        calibration_results = self._fine_tune_models(models, labeled_data)
        
        # 3. Alpha decay detection
        decay_metrics = self._analyze_alpha_decay(labeled_data)
        
        self.logger.info(f"✅ Self-Reinforcing Cycle Complete. Labeled {len(labeled_data)} events.")
        return {
            "status": "success",
            "events_processed": len(labeled_data),
            "calibration": calibration_results,
            "alpha_decay": decay_metrics
        }

    def _label_historical_decisions(self) -> List[Dict[str, Any]]:
        """
        Joins decisions with subsequent market data to calculate 'Actual Outcome'.
        Label = price change percentage after N minutes.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get decisions that haven't been labeled yet
            # We look for decisions older than lookforward_minutes
            cutoff_time = (datetime.now(timezone.utc) - timedelta(minutes=self.lookforward_minutes)).isoformat()
            
            query = """
                SELECT d.*, m.price as entry_price
                FROM decisions d
                JOIN market_data m ON d.product_id = m.product_id 
                    AND datetime(d.timestamp) BETWEEN datetime(m.timestamp, '-30 seconds') AND datetime(m.timestamp, '+30 seconds')
                LEFT JOIN labels l ON d.id = l.decision_id
                WHERE d.timestamp < ? AND l.id IS NULL
                ORDER BY d.timestamp DESC
                LIMIT 100
            """
            cursor.execute(query, (cutoff_time,))
            decisions = [dict(row) for row in cursor.fetchall()]
            
            labeled_data = []
            for d in decisions:
                # Find 'future' price (lookforward_minutes later)
                future_time_dt = datetime.fromisoformat(d['timestamp']) + timedelta(minutes=self.lookforward_minutes)
                future_start = (future_time_dt - timedelta(seconds=60)).isoformat()
                future_end = (future_time_dt + timedelta(seconds=60)).isoformat()
                
                cursor.execute("""
                    SELECT price, timestamp FROM market_data 
                    WHERE product_id = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY ABS(strftime('%s', timestamp) - strftime('%s', ?)) ASC
                    LIMIT 1
                """, (d['product_id'], future_start, future_end, future_time_dt.isoformat()))
                
                row = cursor.fetchone()
                if row:
                    exit_price = row['price']
                    entry_price = d['entry_price']
                    price_change_pct = (exit_price - entry_price) / entry_price
                    
                    # Store original reasoning
                    reasoning = json.loads(d['reasoning'])
                    
                    correct = 0
                    if d['action'] == 'BUY' and price_change_pct > 0.002: # 0.2% threshold for 'correct'
                        correct = 1
                    elif d['action'] == 'SELL' and price_change_pct < -0.002:
                        correct = 1
                    elif d['action'] == 'HOLD' and abs(price_change_pct) < 0.002:
                        correct = 1
                    
                    label_entry = {
                        'decision_id': d['id'],
                        'product_id': d['product_id'],
                        't_entry': d['timestamp'],
                        'entry_price': entry_price,
                        't_exit': row['timestamp'],
                        'exit_price': exit_price,
                        'horizon_min': self.lookforward_minutes,
                        'ret_pct': price_change_pct,
                        'correct': correct
                    }
                    
                    # Persist label
                    cursor.execute("""
                        INSERT OR IGNORE INTO labels 
                        (decision_id, product_id, t_entry, entry_price, t_exit, exit_price, horizon_min, ret_pct, correct)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        label_entry['decision_id'], label_entry['product_id'], label_entry['t_entry'],
                        label_entry['entry_price'], label_entry['t_exit'], label_entry['exit_price'],
                        label_entry['horizon_min'], label_entry['ret_pct'], label_entry['correct']
                    ))
                    
                    label_entry['features'] = reasoning.get('signal_contributions', {})
                    label_entry['ml_predictions'] = reasoning.get('real_time_predictions', {}).get('predictions', {})
                    labeled_data.append(label_entry)
                    
            conn.commit()
            conn.close()
            return labeled_data
            
        except Exception as e:
            self.logger.error(f"Labeling failed: {e}")
            return []

    def _fine_tune_models(self, models: Dict[str, Any], labeled_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fine-tunes model weights using recent labeled experience.
        Especially updates the 'Ensemble' meta-learner.
        """
        results = {}
        if "Ensemble" in models:
            try:
                ensemble = models["Ensemble"]
                # Convert labeled data to X (predictions) and y (outcome)
                X = []
                y = []
                for item in labeled_data:
                    preds = item['ml_predictions']
                    if preds:
                        # Ensure we have a consistent feature vector from model predictions
                        # Use keys in deterministic order
                        model_keys = sorted(preds.keys())
                        val_vec = [preds[k] for k in model_keys]
                        if len(val_vec) > 0:
                            X.append(val_vec)
                            # Target: price change direction scaled to [-1, 1]
                            y.append(np.clip(item['ret_pct'] * 100, -1, 1))
                
                if X:
                    # PyTorch Ensemble meta-learner online update
                    ensemble.train_meta_learner(
                        X=np.array(X), 
                        y=np.array(y),
                        epochs=5
                    )
                    results["Ensemble"] = "fine_tuned"
                    self.logger.info(f"🧠 Meta-Learner self-correction: fine-tuned on {len(X)} recent labeled samples.")
            except Exception as e:
                self.logger.warning(f"Ensemble fine-tuning failed: {e}")
                
        return results

    def _analyze_alpha_decay(self, labeled_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Detects which signals are losing their predictive power.
        Calculates correlation between each signal and actual outcome.
        """
        correlations = {}
        try:
            if not labeled_data:
                return {}
                
            df = pd.DataFrame(labeled_data)
            outcomes = df['ret_pct'].values
            
            # Extract signal contributions
            feature_keys = set()
            for feat_map in df['features']:
                feature_keys.update(feat_map.keys())
                
            for key in feature_keys:
                signals = [f.get(key, 0.0) for f in df['features']]
                if len(signals) > 1:
                    corr = np.corrcoef(signals, outcomes)[0, 1]
                    correlations[key] = float(corr) if not np.isnan(corr) else 0.0
                else:
                    correlations[key] = 0.0
                
            # Log significant decay
            for key, corr in correlations.items():
                if abs(corr) < 0.05:
                    self.logger.info(f"⚠️ Low correlation detected for {key}: {corr:.4f} (Potential Alpha Decay)")
            
            # Save latest correlations for dashboard
            self._latest_alpha_corrs = correlations
                    
        except Exception as e:
            self.logger.error(f"Alpha decay analysis failed: {e}")
            
        return correlations
