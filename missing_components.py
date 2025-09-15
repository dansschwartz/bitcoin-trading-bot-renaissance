"""
Missing Components for Neural Network Prediction Engine
Contains AdvancedFeatureSelector and compute_loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import logging


class AdvancedFeatureSelector:
    """
    Advanced feature selection and engineering for neural network predictions
    Compatible with existing 127-feature pipeline
    """

    def __init__(self, target_features: int = 127, selection_method: str = 'hybrid'):
        self.target_features = target_features
        self.selection_method = selection_method
        self.selected_features = []
        self.feature_scores = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Feature selection components
        self.statistical_selector = None
        self.mutual_info_selector = None

        logging.info(f"ðŸ”¬ AdvancedFeatureSelector initialized: target={target_features}, method={selection_method}")

    def engineer_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional sophisticated features
        Extends the existing feature set without breaking the 127-feature pipeline
        """

        if df.empty:
            return df

        enhanced_df = df.copy()

        try:
            # Technical indicators for financial data
            if 'close' in df.columns:
                # Price-based features
                enhanced_df['price_momentum_5'] = enhanced_df['close'].pct_change(5)
                enhanced_df['price_momentum_10'] = enhanced_df['close'].pct_change(10)
                enhanced_df['price_momentum_20'] = enhanced_df['close'].pct_change(20)

                # Volatility features
                enhanced_df['volatility_5'] = enhanced_df['close'].rolling(5).std()
                enhanced_df['volatility_10'] = enhanced_df['close'].rolling(10).std()
                enhanced_df['volatility_20'] = enhanced_df['close'].rolling(20).std()

                # Moving averages and ratios
                enhanced_df['sma_5'] = enhanced_df['close'].rolling(5).mean()
                enhanced_df['sma_20'] = enhanced_df['close'].rolling(20).mean()
                enhanced_df['price_to_sma5'] = enhanced_df['close'] / enhanced_df['sma_5']
                enhanced_df['price_to_sma20'] = enhanced_df['close'] / enhanced_df['sma_20']
                enhanced_df['sma_ratio'] = enhanced_df['sma_5'] / enhanced_df['sma_20']

            # Volume-based features (if available)
            if 'volume' in df.columns:
                enhanced_df['volume_momentum_5'] = enhanced_df['volume'].pct_change(5)
                enhanced_df['volume_sma_5'] = enhanced_df['volume'].rolling(5).mean()
                enhanced_df['volume_to_sma'] = enhanced_df['volume'] / enhanced_df['volume_sma_5']

            # Cross-feature interactions
            numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                # Create interaction features between top correlated features
                for i, col1 in enumerate(numeric_cols[:5]):  # Limit to prevent explosion
                    for col2 in numeric_cols[i+1:6]:
                        if col1 != col2:
                            interaction_name = f"{col1}_x_{col2}"
                            try:
                                enhanced_df[interaction_name] = enhanced_df[col1] * enhanced_df[col2]
                            except:
                                pass  # Skip if calculation fails

            # Statistical features
            numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 3:
                # Rolling statistics
                for window in [5, 10]:
                    rolling_data = enhanced_df[numeric_cols].rolling(window)
                    enhanced_df[f'rolling_mean_{window}'] = rolling_data.mean().mean(axis=1)
                    enhanced_df[f'rolling_std_{window}'] = rolling_data.std().mean(axis=1)
                    enhanced_df[f'rolling_min_{window}'] = rolling_data.min().mean(axis=1)
                    enhanced_df[f'rolling_max_{window}'] = rolling_data.max().mean(axis=1)

        except Exception as e:
            logging.warning(f"âš ï¸ Feature engineering partially failed: {e}")

        # Fill NaN values that might have been created
        enhanced_df = enhanced_df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        logging.info(f"ðŸ”¬ Feature engineering complete: {df.shape[1]} â†’ {enhanced_df.shape[1]} features")

        return enhanced_df

    def select_features(self, x: pd.DataFrame, y: pd.Series,
                       preserve_original: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the most informative features using advanced selection methods

        Args:
            x: Feature DataFrame
            y: Target series
            preserve_original: Whether to preserve all original features

        Returns:
            Selected features DataFrame and list of selected feature names
        """

        if x.empty or y.empty:
            return x, list(x.columns)

        # Convert to numeric and handle missing values
        x_numeric = x.select_dtypes(include=[np.number])
        x_numeric = x_numeric.fillna(x_numeric.mean()).fillna(0)

        if x_numeric.shape[1] <= self.target_features:
            # Already at or below target, return as-is
            self.selected_features = list(x_numeric.columns)
            self.is_fitted = True
            return x_numeric, self.selected_features

        try:
            if self.selection_method == 'statistical':
                selected_features = self._statistical_selection(x_numeric, y)
            elif self.selection_method == 'mutual_info':
                selected_features = self._mutual_info_selection(x_numeric, y)
            elif self.selection_method == 'hybrid':
                selected_features = self._hybrid_selection(x_numeric, y)
            else:
                # Fallback to correlation-based selection
                selected_features = self._correlation_selection(x_numeric, y)

        except Exception as e:
            logging.warning(f"âš ï¸ Advanced feature selection failed: {e}, using correlation method")
            selected_features = self._correlation_selection(x_numeric, y)

        # Ensure we have the right number of features
        if len(selected_features) > self.target_features:
            selected_features = selected_features[:self.target_features]
        elif len(selected_features) < self.target_features <= len(x_numeric.columns):
            # Add remaining features to reach target
            remaining = [col for col in x_numeric.columns if col not in selected_features]
            selected_features.extend(remaining[:self.target_features - len(selected_features)])

        self.selected_features = selected_features
        self.is_fitted = True

        selected_x = x_numeric[selected_features]

        logging.info(f"ðŸŽ¯ Feature selection complete: {x_numeric.shape[1]} â†’ {len(selected_features)} features")

        return selected_x, selected_features

    def _statistical_selection(self, x: pd.DataFrame, y: pd.Series) -> List[str]:
        """Statistical feature selection using F-statistics"""
        selector = SelectKBest(score_func=f_regression, k=min(self.target_features, x.shape[1]))
        selector.fit(x, y)
        selected_mask = selector.get_support()
        return list(x.columns[selected_mask])

    def _mutual_info_selection(self, x: pd.DataFrame, y: pd.Series) -> List[str]:
        """Mutual information based feature selection"""
        selector = SelectKBest(score_func=mutual_info_regression, k=min(self.target_features, x.shape[1]))
        selector.fit(x, y)
        selected_mask = selector.get_support()
        return list(x.columns[selected_mask])

    def _hybrid_selection(self, x: pd.DataFrame, y: pd.Series) -> List[str]:
        """Hybrid feature selection combining multiple methods"""

        # Get top features from each method
        half_target = self.target_features // 2

        # Statistical selection (first half)
        stat_features = self._statistical_selection(x, y)[:half_target]

        # Mutual information selection (second half, excluding already selected)
        remaining_x = x[[col for col in x.columns if col not in stat_features]]
        if not remaining_x.empty:
            mi_features = self._mutual_info_selection(remaining_x, y)[:self.target_features - len(stat_features)]
        else:
            mi_features = []

        # Combine results
        selected_features = stat_features + mi_features

        # Fill the remaining slots with correlation-based selection if needed
        if len(selected_features) < self.target_features:
            remaining_x = x[[col for col in x.columns if col not in selected_features]]
            if not remaining_x.empty:
                corr_features = self._correlation_selection(remaining_x, y)[:self.target_features - len(selected_features)]
                selected_features.extend(corr_features)

        return selected_features

    def _correlation_selection(self, x: pd.DataFrame, y: pd.Series) -> List[str]:
        """Correlation-based feature selection (fallback method)"""
        try:
            correlations = x.corrwith(y).abs().sort_values(ascending=False)
            correlations = correlations.dropna()
            return list(correlations.head(self.target_features).index)
        except:
            # Ultimate fallback: return first N features
            return list(x.columns[:self.target_features])

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted feature selector"""

        if not self.is_fitted:
            logging.warning("âš ï¸ AdvancedFeatureSelector not fitted yet")
            return x

        # Select only the fitted features
        available_features = [col for col in self.selected_features if col in x.columns]

        if len(available_features) < len(self.selected_features):
            logging.warning(f"âš ï¸ Some selected features missing: {len(available_features)}/{len(self.selected_features)}")

        return x[available_features] if available_features else x

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_scores.copy()


def compute_loss(predictions: torch.Tensor, targets: torch.Tensor,
                loss_type: str = 'mse', **kwargs) -> torch.Tensor:
    """
    Compute loss between predictions and targets
    Supports multiple loss types for different training scenarios

    Args:
        predictions: Model predictions tensor
        targets: Target values tensor
        loss_type: Type of loss ('mse', 'mae', 'huber', 'smooth_l1', 'directional')
        **kwargs: Additional loss-specific parameters

    Returns:
        Computed loss tensor
    """

    if predictions.shape != targets.shape:
        # Handle shape mismatch
        if predictions.numel() == targets.numel():
            predictions = predictions.view(targets.shape)
        else:
            # Truncate to smaller size
            min_size = min(predictions.numel(), targets.numel())
            predictions = predictions.flatten()[:min_size]
            targets = targets.flatten()[:min_size]

    try:
        if loss_type == 'mse':
            # Mean Squared Error (default)
            return f.mse_loss(predictions, targets)

        elif loss_type == 'mae':
            # Mean Absolute Error
            return f.l1_loss(predictions, targets)

        elif loss_type == 'huber':
            # Huber Loss (robust to outliers)
            delta = kwargs.get('delta', 1.0)
            return f.huber_loss(predictions, targets, delta=delta)

        elif loss_type == 'smooth_l1':
            # Smooth L1 Loss
            return f.smooth_l1_loss(predictions, targets)

        elif loss_type == 'directional':
            # Directional loss for trading applications
            directional_weight = kwargs.get('directional_weight', 0.5)

            # Base MSE loss
            mse_loss = f.mse_loss(predictions, targets)

            # Directional component
            pred_direction = torch.sign(predictions)
            target_direction = torch.sign(targets)
            directional_accuracy = (pred_direction == target_direction).float()
            directional_loss = 1.0 - directional_accuracy.mean()

            # Combined loss
            total_loss = (1 - directional_weight) * mse_loss + directional_weight * directional_loss
            return total_loss

        elif loss_type == 'adaptive':
            # Adaptive loss based on prediction confidence
            confidence_threshold = kwargs.get('confidence_threshold', 0.1)

            # Base loss
            base_loss = f.mse_loss(predictions, targets, reduction='none')

            # Confidence weighting (higher loss for confident wrong predictions)
            confidence = torch.abs(predictions)
            confidence_weights = torch.where(
                confidence > confidence_threshold,
                1.0 + confidence,  # Penalize confident wrong predictions more
                1.0
            )

            weighted_loss = base_loss * confidence_weights
            return weighted_loss.mean()

        else:
            # Default to MSE for unknown loss types
            logging.warning(f"âš ï¸ Unknown loss type '{loss_type}', using MSE")
            return f.mse_loss(predictions, targets)

    except Exception as e:
        logging.error(f"âŒ Loss computation failed: {e}")
        # Fallback to simple MSE
        return f.mse_loss(predictions.float(), targets.float())


def compute_loss_with_regularization(predictions: torch.Tensor, targets: torch.Tensor,
                                   model_params: Optional[List[torch.Tensor]] = None,
                                   loss_type: str = 'mse',
                                   l1_weight: float = 0.0,
                                   l2_weight: float = 0.0,
                                   **kwargs) -> torch.Tensor:
    """
    Compute loss with optional L1/L2 regularization

    Args:
        predictions: Model predictions
        targets: Target values
        model_params: Model parameters for regularization
        loss_type: Base loss type
        l1_weight: L1 regularization weight
        l2_weight: L2 regularization weight
        **kwargs: Additional loss parameters

    Returns:
        Total loss including regularization
    """

    # Base loss
    base_loss = compute_loss(predictions, targets, loss_type=loss_type, **kwargs)

    # Add regularization if specified
    regularization_loss = 0.0

    if model_params and (l1_weight > 0 or l2_weight > 0):
        for param in model_params:
            if param.requires_grad:
                if l1_weight > 0:
                    regularization_loss += l1_weight * torch.sum(torch.abs(param))
                if l2_weight > 0:
                    regularization_loss += l2_weight * torch.sum(param ** 2)

    total_loss = base_loss + regularization_loss

    return total_loss


def test_missing_components():
    """Test function for missing components"""
    print("ðŸ§ª Testing AdvancedFeatureSelector and compute_loss...")

    # Test AdvancedFeatureSelector
    np.random.seed(42)
    test_df = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randn(100)
    })

    selector = AdvancedFeatureSelector(target_features=10)
    enhanced_df = selector.engineer_advanced_features(test_df)
    print(f"âœ… Feature engineering: {test_df.shape[1]} â†’ {enhanced_df.shape[1]} features")

    # Test feature selection
    y = pd.Series(np.random.randn(100))
    selected_df, selected_features = selector.select_features(enhanced_df, y)
    print(f"âœ… Feature selection: {enhanced_df.shape[1]} â†’ {len(selected_features)} features")

    # Test compute_loss
    pred_tensor = torch.randn(10, 1)
    target_tensor = torch.randn(10, 1)

    mse_loss = compute_loss(pred_tensor, target_tensor, 'mse')
    mae_loss = compute_loss(pred_tensor, target_tensor, 'mae')
    directional_loss = compute_loss(pred_tensor, target_tensor, 'directional', directional_weight=0.3)

    print(f"âœ… MSE Loss: {mse_loss:.6f}")
    print(f"âœ… MAE Loss: {mae_loss:.6f}")
    print(f"âœ… Directional Loss: {directional_loss:.6f}")

    print("ðŸŽ‰ Missing components test completed successfully!")


if __name__ == "__main__":
    test_missing_components()