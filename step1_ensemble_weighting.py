"""
Step 1: Ensemble Weighting Optimization
Performance-weighted ensemble for neural network prediction engine
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

class EnsembleWeightOptimizer:
    """
    Performance-weighted ensemble optimizer for neural network predictions
    Replaces simple averaging with intelligent model weighting based on performance metrics
    """

    def __init__(self, lookback_window: int = 100, rebalance_frequency: int = 50):
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        self.model_weights = {}
        self.performance_history = []
        self.prediction_count = 0
        self.last_rebalance = 0

        # Performance tracking
        self.model_performance = {}
        self.validation_window = 20

        # Safety settings
        self.min_weight = 0.1  # Minimum weight for any model
        self.max_weight = 0.6  # Maximum weight for any model

        logging.info("ðŸ”§ EnsembleWeightOptimizer initialized")

    def update_performance_history(self, predictions: Dict[str, torch.Tensor],
                                 actual: torch.Tensor, model_names: List[str]):
        """Update performance tracking for each model"""

        performance_entry = {
            'timestamp': datetime.now(),
            'predictions': {},
            'actual': actual.clone() if isinstance(actual, torch.Tensor) else torch.tensor(actual),
            'model_performance': {}
        }

        for model_name in model_names:
            if model_name in predictions and predictions[model_name] is not None:
                pred = predictions[model_name]
                if isinstance(pred, torch.Tensor):
                    pred = pred.clone()
                else:
                    pred = torch.tensor(pred)

                performance_entry['predictions'][model_name] = pred

                # Calculate individual model performance
                try:
                    # Directional accuracy
                    pred_direction = torch.sign(pred)
                    actual_direction = torch.sign(actual)
                    directional_accuracy = (pred_direction == actual_direction).float().mean().item()

                    # MSE
                    mse = torch.mean((pred - actual) ** 2).item()

                    # RÂ² approximation
                    ss_res = torch.sum((actual - pred) ** 2).item()
                    ss_tot = torch.sum((actual - actual.mean()) ** 2).item()
                    r2 = 1 - (ss_res / (ss_tot + 1e-8))

                    performance_entry['model_performance'][model_name] = {
                        'directional_accuracy': directional_accuracy,
                        'mse': mse,
                        'r2': r2,
                        'rmse': np.sqrt(mse)
                    }

                except Exception as e:
                    logging.warning(f"âš ï¸ Performance calculation failed for {model_name}: {e}")
                    performance_entry['model_performance'][model_name] = {
                        'directional_accuracy': 0.5,
                        'mse': 1.0,
                        'r2': 0.0,
                        'rmse': 1.0
                    }

        self.performance_history.append(performance_entry)

        # Keep only recent history
        if len(self.performance_history) > self.lookback_window:
            self.performance_history = self.performance_history[-self.lookback_window:]

    def calculate_model_weights(self, model_names: List[str]) -> Dict[str, float]:
        """Calculate performance-weighted model weights"""

        if len(self.performance_history) < 5:  # Need minimum history
            # Equal weights for insufficient history
            equal_weight = 1.0 / len(model_names)
            return {name: equal_weight for name in model_names}

        # Calculate recent performance for each model
        model_scores = {}

        for model_name in model_names:
            recent_performance = []

            # Get recent performance entries
            recent_entries = self.performance_history[-self.validation_window:]

            for entry in recent_entries:
                if model_name in entry['model_performance']:
                    perf = entry['model_performance'][model_name]

                    # Weighted composite score
                    composite_score = (
                        0.5 * perf['directional_accuracy'] +  # 50% weight on directional accuracy
                        0.25 * max(0, perf['r2']) +           # 25% weight on RÂ²
                        0.15 * (1.0 / (1.0 + perf['rmse'])) + # 15% weight on RMSE (inverted)
                        0.1 * (1.0 / (1.0 + perf['mse']))    # 10% weight on MSE (inverted)
                    )

                    recent_performance.append(composite_score)

            if recent_performance:
                # Average recent performance with some stability
                avg_performance = np.mean(recent_performance)
                stability = 1.0 - np.std(recent_performance)  # Penalize inconsistency
                model_scores[model_name] = avg_performance * (0.8 + 0.2 * stability)
            else:
                model_scores[model_name] = 0.5  # Default neutral score

        # Convert scores to weights using softmax for stability
        scores = np.array([model_scores.get(name, 0.5) for name in model_names])

        # Apply softmax with temperature for controlled distribution
        temperature = 2.0  # Higher temperature = more even distribution
        exp_scores = np.exp(scores / temperature)
        weights = exp_scores / np.sum(exp_scores)

        # Apply min/max constraints
        weights = np.clip(weights, self.min_weight, self.max_weight)

        # Renormalize after clipping
        weights = weights / np.sum(weights)

        weight_dict = {name: float(weight) for name, weight in zip(model_names, weights)}

        return weight_dict

    def get_weighted_prediction(self, predictions: Dict[str, torch.Tensor],
                              model_names: Optional[List[str]] = None) -> torch.Tensor:
        """
        Generate performance-weighted ensemble prediction

        Args:
            predictions: Dictionary of model predictions
            model_names: List of model names to include

        Returns:
            Weighted ensemble prediction tensor
        """

        self.prediction_count += 1

        # Filter valid predictions
        valid_predictions = {}
        if model_names is None:
            model_names = list(predictions.keys())

        for name in model_names:
            if name in predictions and predictions[name] is not None:
                pred = predictions[name]
                if isinstance(pred, torch.Tensor) and not torch.isnan(pred).any():
                    valid_predictions[name] = pred
                elif isinstance(pred, (int, float)) and not np.isnan(pred):
                    valid_predictions[name] = torch.tensor(pred, dtype=torch.float32)

        if not valid_predictions:
            logging.warning("âš ï¸ No valid predictions available for ensemble")
            return torch.tensor(0.0)

        valid_model_names = list(valid_predictions.keys())

        # Rebalance weights if needed
        if (self.prediction_count - self.last_rebalance >= self.rebalance_frequency or
            not self.model_weights):

            self.model_weights = self.calculate_model_weights(valid_model_names)
            self.last_rebalance = self.prediction_count

            logging.info(f"ðŸ”„ Model weights updated: {self.model_weights}")

        # Apply weights to predictions
        weighted_sum = torch.tensor(0.0)
        total_weight = 0.0

        for name in valid_model_names:
            weight = self.model_weights.get(name, 1.0 / len(valid_model_names))
            weighted_sum += weight * valid_predictions[name]
            total_weight += weight

        if total_weight > 0:
            ensemble_prediction = weighted_sum / total_weight
        else:
            # Fallback to simple average
            ensemble_prediction = torch.stack(list(valid_predictions.values())).mean()

        return ensemble_prediction

    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights"""
        return self.model_weights.copy()

    def get_performance_summary(self) -> Dict:
        """Get a performance summary for monitoring"""
        if not self.performance_history:
            return {"status": "insufficient_data"}

        recent_entries = self.performance_history[-10:]
        summary = {
            "total_predictions": self.prediction_count,
            "history_length": len(self.performance_history),
            "current_weights": self.model_weights.copy(),
            "recent_performance": {}
        }

        # Calculate recent average performance per model
        for entry in recent_entries:
            for model_name, perf in entry['model_performance'].items():
                if model_name not in summary["recent_performance"]:
                    summary["recent_performance"][model_name] = []
                summary["recent_performance"][model_name].append(perf['directional_accuracy'])

        # Average recent performance
        for model_name, accuracies in summary["recent_performance"].items():
            summary["recent_performance"][model_name] = {
                "avg_directional_accuracy": np.mean(accuracies),
                "consistency": 1.0 - np.std(accuracies)
            }

        return summary


class Step1TestingFramework:
    """Testing and validation framework for Step 1 ensemble optimization"""

    def __init__(self):
        self.baseline_metrics = {}
        self.enhanced_metrics = {}
        self.rollback_triggers = {
            'r2_threshold': 0.025,
            'accuracy_threshold': 0.60,
            'rmse_increase_threshold': 0.5
        }

    def validate_step1_success(self, enhanced_metrics: Dict, baseline_metrics: Dict) -> bool:
        """Validate if Step 1 implementation is successful"""

        success_criteria = {
            'directional_accuracy_improved': enhanced_metrics.get('directional_accuracy', 0) > baseline_metrics.get('directional_accuracy', 0),
            'r2_stable': enhanced_metrics.get('r2', 0) >= self.rollback_triggers['r2_threshold'],
            'rmse_acceptable': enhanced_metrics.get('rmse', float('inf')) <= baseline_metrics.get('rmse', 0) * (1 + self.rollback_triggers['rmse_increase_threshold']),
            'accuracy_above_threshold': enhanced_metrics.get('directional_accuracy', 0) >= self.rollback_triggers['accuracy_threshold']
        }

        all_criteria_met = all(success_criteria.values())

        logging.info(f"ðŸ“Š Step 1 Validation Results:")
        for criterion, met in success_criteria.items():
            status = "âœ…" if met else "âŒ"
            logging.info(f"  {status} {criterion}: {met}")

        return all_criteria_met

    def should_rollback(self, current_metrics: Dict, baseline_metrics: Dict) -> Tuple[bool, List[str]]:
        """Check if the system should roll back to baseline"""

        rollback_reasons = []

        # Check RÂ² degradation
        if current_metrics.get('r2', 0) < self.rollback_triggers['r2_threshold']:
            rollback_reasons.append(f"RÂ² below threshold: {current_metrics.get('r2', 0):.4f} < {self.rollback_triggers['r2_threshold']}")

        # Check directional accuracy
        if current_metrics.get('directional_accuracy', 0) < self.rollback_triggers['accuracy_threshold']:
            rollback_reasons.append(f"Directional accuracy below threshold: {current_metrics.get('directional_accuracy', 0):.4f} < {self.rollback_triggers['accuracy_threshold']}")

        # Check RMSE increase
        baseline_rmse = baseline_metrics.get('rmse', 0)
        current_rmse = current_metrics.get('rmse', 0)
        if current_rmse > baseline_rmse * (1 + self.rollback_triggers['rmse_increase_threshold']):
            rollback_reasons.append(f"RMSE increased too much: {current_rmse:.6f} vs baseline {baseline_rmse:.6f}")

        should_rollback = len(rollback_reasons) > 0

        return should_rollback, rollback_reasons


def test_ensemble_optimizer():
    """Test function for EnsembleWeightOptimizer"""
    print("ðŸ§ª Testing EnsembleWeightOptimizer...")

    # Create test instance
    optimizer = EnsembleWeightOptimizer()

    # Test with sample predictions
    test_predictions = {
        'quantum_transformer': torch.tensor([0.1, 0.2, 0.3]),
        'bidirectional_lstm': torch.tensor([0.15, 0.25, 0.35]),
        'dilated_cnn': torch.tensor([0.05, 0.15, 0.25])
    }

    # Get weighted prediction
    weighted_pred = optimizer.get_weighted_prediction(test_predictions)
    print(f"âœ… Weighted prediction: {weighted_pred}")

    # Test performance update
    actual = torch.tensor([0.12, 0.22, 0.32])
    optimizer.update_performance_history(test_predictions, actual, list(test_predictions.keys()))

    # Get performance summary
    summary = optimizer.get_performance_summary()
    print(f"âœ… Performance summary: {summary}")

    print("ðŸŽ‰ EnsembleWeightOptimizer test completed successfully!")


if __name__ == "__main__":
    test_ensemble_optimizer()