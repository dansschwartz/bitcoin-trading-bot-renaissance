"""
N-BEATS Forecaster for Renaissance Trading Bot
Neural Basis Expansion Analysis for Time Series - Advanced interpretable forecasting
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
import json
import pickle
from pathlib import Path

# Import ML libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    HAS_ML_LIBS = True
except ImportError as e:
    print(f"Warning: ML libraries not available: {e}")
    HAS_ML_LIBS = False

# Import configuration
from ml_config import NBEATSConfig, TimeFrame

@dataclass
class NBEATSMetrics:
    """N-BEATS model performance metrics"""
    mse: float
    mae: float
    mape: float  # Mean Absolute Percentage Error
    rmse: float
    r2_score: float
    val_mse: float
    val_mae: float
    val_mape: float
    val_rmse: float
    val_r2_score: float
    training_time: float
    epochs_trained: int
    trend_component_strength: float
    seasonality_component_strength: float

@dataclass
class NBEATSPrediction:
    """N-BEATS prediction result with interpretable components"""
    forecast: np.ndarray
    trend_component: np.ndarray
    seasonality_component: np.ndarray
    generic_component: np.ndarray
    backcast: np.ndarray
    confidence_interval: Tuple[np.ndarray, np.ndarray]
    trend_direction: str
    seasonality_pattern: Dict[str, float]
    prediction_horizon: int
    timestamp: datetime
    metadata: Dict[str, Any]

class NBEATSBlock(layers.Layer):
    """
    Individual N-BEATS block implementation
    Each block produces a backcast and forecast
    """

    def __init__(self, 
                 units: int,
                 thetas_dim: int,
                 backcast_length: int,
                 forecast_length: int,
                 share_thetas: bool = False,
                 block_type: str = 'generic',
                 **kwargs):
        super(NBEATSBlock, self).__init__(**kwargs)
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.block_type = block_type

        # Fully connected layers
        self.hidden_layers = [
            layers.Dense(units, activation='relu', name=f'{self.name}_fc_{i}')
            for i in range(4)
        ]

        # Theta layers (coefficients for basis functions)
        if share_thetas:
            self.theta_b_fc = layers.Dense(thetas_dim, activation='linear', use_bias=False, name=f'{self.name}_theta_b')
            self.theta_f_fc = self.theta_b_fc
        else:
            self.theta_b_fc = layers.Dense(thetas_dim, activation='linear', use_bias=False, name=f'{self.name}_theta_b')
            self.theta_f_fc = layers.Dense(thetas_dim, activation='linear', use_bias=False, name=f'{self.name}_theta_f')

    def call(self, inputs):
        # Forward pass through fully connected layers
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)

        # Generate theta coefficients
        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        # Generate basis functions and compute backcast/forecast
        if self.block_type == 'trend':
            backcast, forecast = self._trend_model(theta_b, theta_f)
        elif self.block_type == 'seasonality':
            backcast, forecast = self._seasonality_model(theta_b, theta_f)
        else:  # generic
            backcast, forecast = self._generic_model(theta_b, theta_f)

        return backcast, forecast

    def _generic_model(self, theta_b, theta_f):
        """Generic basis functions - learned representations"""
        # Create basis matrices (simplified version)
        T_b = tf.eye(self.backcast_length, dtype=tf.float32)
        T_f = tf.eye(self.forecast_length, dtype=tf.float32)

        # Pad or truncate to match thetas_dim
        if self.thetas_dim < self.backcast_length:
            T_b = T_b[:, :self.thetas_dim]
        elif self.thetas_dim > self.backcast_length:
            padding = self.thetas_dim - self.backcast_length
            T_b = tf.pad(T_b, [[0, 0], [0, padding]])

        if self.thetas_dim < self.forecast_length:
            T_f = T_f[:, :self.thetas_dim]
        elif self.thetas_dim > self.forecast_length:
            padding = self.thetas_dim - self.forecast_length
            T_f = tf.pad(T_f, [[0, 0], [0, padding]])

        backcast = tf.linalg.matvec(T_b, theta_b)
        forecast = tf.linalg.matvec(T_f, theta_f)

        return backcast, forecast

    def _trend_model(self, theta_b, theta_f):
        """Trend basis functions - polynomial trends"""
        # Create polynomial basis
        t_b = tf.range(self.backcast_length, dtype=tf.float32) / self.backcast_length
        t_f = tf.range(self.forecast_length, dtype=tf.float32) / self.forecast_length + 1.0

        # Polynomial basis matrices
        T_b = tf.stack([t_b ** i for i in range(self.thetas_dim)], axis=1)
        T_f = tf.stack([t_f ** i for i in range(self.thetas_dim)], axis=1)

        backcast = tf.linalg.matvec(T_b, theta_b)
        forecast = tf.linalg.matvec(T_f, theta_f)

        return backcast, forecast

    def _seasonality_model(self, theta_b, theta_f):
        """Seasonality basis functions - Fourier series"""
        # Create Fourier basis
        t_b = tf.range(self.backcast_length, dtype=tf.float32)
        t_f = tf.range(self.forecast_length, dtype=tf.float32) + self.backcast_length

        # Fourier basis matrices
        freqs = tf.range(1, self.thetas_dim // 2 + 1, dtype=tf.float32)

        T_b_cos = tf.cos(2 * np.pi * freqs[None, :] * t_b[:, None] / self.backcast_length)
        T_b_sin = tf.sin(2 * np.pi * freqs[None, :] * t_b[:, None] / self.backcast_length)
        T_b = tf.concat([T_b_cos, T_b_sin], axis=1)

        T_f_cos = tf.cos(2 * np.pi * freqs[None, :] * t_f[:, None] / self.backcast_length)
        T_f_sin = tf.sin(2 * np.pi * freqs[None, :] * t_f[:, None] / self.backcast_length)
        T_f = tf.concat([T_f_cos, T_f_sin], axis=1)

        # Adjust dimensions
        if T_b.shape[1] > self.thetas_dim:
            T_b = T_b[:, :self.thetas_dim]
            T_f = T_f[:, :self.thetas_dim]

        backcast = tf.linalg.matvec(T_b, theta_b)
        forecast = tf.linalg.matvec(T_f, theta_f)

        return backcast, forecast

class NBEATSStack(layers.Layer):
    """
    N-BEATS stack containing multiple blocks of the same type
    """

    def __init__(self, 
                 stack_type: str,
                 nb_blocks: int,
                 units: int,
                 thetas_dim: int,
                 backcast_length: int,
                 forecast_length: int,
                 share_weights: bool = False,
                 **kwargs):
        super(NBEATSStack, self).__init__(**kwargs)
        self.stack_type = stack_type
        self.nb_blocks = nb_blocks
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        # Create blocks
        self.blocks = []
        for i in range(nb_blocks):
            block = NBEATSBlock(
                units=units,
                thetas_dim=thetas_dim,
                backcast_length=backcast_length,
                forecast_length=forecast_length,
                share_thetas=share_weights,
                block_type=stack_type,
                name=f'{stack_type}_block_{i}'
            )
            self.blocks.append(block)

    def call(self, inputs):
        x = inputs
        forecast_sum = tf.zeros((tf.shape(inputs)[0], self.forecast_length))

        for block in self.blocks:
            backcast, forecast = block(x)
            # Residual connection
            x = x - backcast
            # Accumulate forecasts
            forecast_sum = forecast_sum + forecast

        return x, forecast_sum

class NBEATSForecaster:
    """
    N-BEATS (Neural Basis Expansion Analysis) Forecaster

    Interpretable neural network for time series forecasting with:
    - Trend decomposition
    - Seasonality analysis  
    - Generic patterns
    """

    def __init__(self, config: Optional[NBEATSConfig] = None):
        """Initialize N-BEATS forecaster"""
        self.logger = logging.getLogger(__name__)
        self.config = config or NBEATSConfig()
        self.model = None
        self.scaler = None
        self.training_history = {}
        self.metrics = None
        self.is_trained = False
        self.trend_stack = None
        self.seasonality_stack = None
        self.generic_stack = None

        # Initialize scaler
        self._initialize_scaler()

        # Build model architecture
        if HAS_ML_LIBS:
            self._build_model()
        else:
            self.logger.warning("ML libraries not available, model will run in simulation mode")

        self.logger.info("N-BEATS forecaster initialized")

    def _initialize_scaler(self):
        """Initialize data scaler"""
        self.scaler = StandardScaler()

    def _build_model(self):
        """Build N-BEATS model architecture"""
        try:
            # Input layer
            inputs = keras.Input(
                shape=(self.config.backcast_length,),
                name='time_series_input'
            )

            # Normalize input
            x = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1))(inputs)

            # Initialize stacks
            stacks = []
            forecast_outputs = []

            # Create stacks based on configuration
            for stack_type in self.config.stack_types:
                if stack_type == 'trend':
                    thetas_dim = self.config.polynomial_degree + 1
                elif stack_type == 'seasonality':
                    thetas_dim = 2 * self.config.harmonics
                else:  # generic
                    thetas_dim = max(self.config.backcast_length, self.config.forecast_length)

                stack = NBEATSStack(
                    stack_type=stack_type,
                    nb_blocks=self.config.nb_blocks_per_stack,
                    units=self.config.hidden_layer_units,
                    thetas_dim=thetas_dim,
                    backcast_length=self.config.backcast_length,
                    forecast_length=self.config.forecast_length,
                    share_weights=self.config.share_weights_in_stack,
                    name=f'{stack_type}_stack'
                )
                stacks.append(stack)

            # Forward pass through stacks
            residual = x
            total_forecast = tf.zeros((tf.shape(inputs)[0], self.config.forecast_length))

            for stack in stacks:
                residual, stack_forecast = stack(residual)
                total_forecast = total_forecast + stack_forecast
                forecast_outputs.append(stack_forecast)

            # Create model
            self.model = keras.Model(
                inputs=inputs,
                outputs={
                    'forecast': total_forecast,
                    'residual': residual,
                    **{f'{stack.stack_type}_forecast': forecast_outputs[i] 
                       for i, stack in enumerate(stacks)}
                },
                name='NBEATS_Forecaster'
            )

            # Compile model
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss={
                    'forecast': 'mse',
                    'residual': 'mse',
                    **{f'{stack.stack_type}_forecast': 'mse' 
                       for stack in stacks}
                },
                loss_weights={
                    'forecast': 1.0,
                    'residual': 0.1,
                    **{f'{stack.stack_type}_forecast': 0.3 
                       for stack in stacks}
                },
                metrics={
                    'forecast': ['mae', 'mse'],
                    **{f'{stack.stack_type}_forecast': ['mae'] 
                       for stack in stacks}
                }
            )

            self.logger.info(f"N-BEATS model built successfully")
            self.logger.info(f"Model parameters: {self.model.count_params():,}")

        except Exception as e:
            self.logger.error(f"Error building N-BEATS model: {e}")
            self.model = None

    def prepare_data(self, data: np.ndarray, targets: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for N-BEATS training/prediction

        Args:
            data: Time series data
            targets: Target values for supervised learning

        Returns:
            X: Input sequences
            y: Target sequences (if targets provided)
        """
        try:
            if len(data) < self.config.backcast_length + self.config.forecast_length:
                raise ValueError(f"Data length {len(data)} is insufficient for backcast + forecast")

            # Create sliding windows
            n_samples = len(data) - self.config.backcast_length - self.config.forecast_length + 1

            X = np.zeros((n_samples, self.config.backcast_length))
            y = None

            if targets is not None:
                y = np.zeros((n_samples, self.config.forecast_length))

            for i in range(n_samples):
                X[i] = data[i:i + self.config.backcast_length]
                if targets is not None:
                    y[i] = targets[i + self.config.backcast_length:i + self.config.backcast_length + self.config.forecast_length]

            return X, y

        except Exception as e:
            self.logger.error(f"Error preparing N-BEATS data: {e}")
            return np.array([]), np.array([])

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> bool:
        """
        Train the N-BEATS model

        Args:
            X_train: Training input sequences
            y_train: Training target sequences
            X_val: Validation input sequences
            y_val: Validation target sequences

        Returns:
            bool: Training success status
        """
        try:
            if not HAS_ML_LIBS or self.model is None:
                self.logger.warning("Training in simulation mode")
                self.is_trained = True
                return True

            self.logger.info("Starting N-BEATS model training...")
            start_time = datetime.now()

            # Scale the data
            X_train_scaled = self.scaler.fit_transform(X_train)

            # Prepare multi-output targets
            y_train_multi = self._prepare_multi_targets(y_train)

            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                y_val_multi = self._prepare_multi_targets(y_val)
                validation_data = (X_val_scaled, y_val_multi)
            else:
                validation_data = None

            # Training callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.config.patience // 2,
                    min_lr=1e-7,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath='models/nbeats_best.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]

            # Train the model
            history = self.model.fit(
                X_train_scaled,
                y_train_multi,
                validation_data=validation_data,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )

            # Store training history
            self.training_history = history.history

            # Calculate metrics
            if validation_data is not None:
                self._calculate_metrics(X_val_scaled, y_val)

            # Mark as trained
            self.is_trained = True

            training_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"N-BEATS training completed in {training_time:.2f} seconds")

            return True

        except Exception as e:
            self.logger.error(f"Error training N-BEATS model: {e}")
            return False

    def _prepare_multi_targets(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Prepare multiple target outputs for N-BEATS"""
        targets = {
            'forecast': y,
            'residual': np.zeros((y.shape[0], self.config.backcast_length))
        }

        # Add individual stack targets (simplified)
        for stack_type in self.config.stack_types:
            if stack_type == 'trend':
                # Generate trend component
                trend_targets = np.array([
                    np.polyval(np.polyfit(range(len(seq)), seq, 2), 
                              range(len(seq))) for seq in y
                ])
                targets[f'{stack_type}_forecast'] = trend_targets
            elif stack_type == 'seasonality':
                # Generate seasonality component (simplified)
                seasonality_targets = np.sin(2 * np.pi * np.arange(y.shape[1]) / 12) * np.std(y, axis=1, keepdims=True)
                targets[f'{stack_type}_forecast'] = seasonality_targets
            else:  # generic
                # Generic component is remainder
                targets[f'{stack_type}_forecast'] = y * 0.3

        return targets

    def _calculate_metrics(self, X_val: np.ndarray, y_val: np.ndarray):
        """Calculate N-BEATS performance metrics"""
        try:
            # Make predictions
            predictions = self.model.predict(X_val, verbose=0)
            y_pred = predictions['forecast']

            # Calculate standard metrics
            mse = mean_squared_error(y_val.flatten(), y_pred.flatten())
            mae = mean_absolute_error(y_val.flatten(), y_pred.flatten())
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val.flatten(), y_pred.flatten())

            # Calculate MAPE
            mape = np.mean(np.abs((y_val.flatten() - y_pred.flatten()) / (y_val.flatten() + 1e-8))) * 100

            # Validation metrics from history
            val_mse = min(self.training_history.get('val_forecast_mse', [mse]))
            val_mae = min(self.training_history.get('val_forecast_mae', [mae]))
            val_rmse = np.sqrt(val_mse)
            val_r2 = max(self.training_history.get('val_forecast_r2_score', [r2])) if 'val_forecast_r2_score' in self.training_history else r2
            val_mape = mape  # Simplified

            # Component strength analysis
            trend_strength = 0.0
            seasonality_strength = 0.0

            if 'trend_forecast' in predictions:
                trend_component = predictions['trend_forecast']
                trend_strength = np.std(trend_component) / (np.std(y_pred) + 1e-8)

            if 'seasonality_forecast' in predictions:
                seasonality_component = predictions['seasonality_forecast']
                seasonality_strength = np.std(seasonality_component) / (np.std(y_pred) + 1e-8)

            self.metrics = NBEATSMetrics(
                mse=mse,
                mae=mae,
                mape=mape,
                rmse=rmse,
                r2_score=r2,
                val_mse=val_mse,
                val_mae=val_mae,
                val_mape=val_mape,
                val_rmse=val_rmse,
                val_r2_score=val_r2,
                training_time=len(self.training_history.get('loss', [])),
                epochs_trained=len(self.training_history.get('loss', [])),
                trend_component_strength=trend_strength,
                seasonality_component_strength=seasonality_strength
            )

            self.logger.info(f"N-BEATS metrics calculated: MSE={mse:.6f}, MAE={mae:.6f}, MAPE={mape:.2f}%")

        except Exception as e:
            self.logger.error(f"Error calculating N-BEATS metrics: {e}")

    def predict(self, X: np.ndarray, return_components: bool = True) -> Union[np.ndarray, NBEATSPrediction]:
        """
        Generate forecasts using N-BEATS

        Args:
            X: Input sequences of shape (batch_size, backcast_length)
            return_components: Whether to return component decomposition

        Returns:
            Forecast array or detailed NBEATSPrediction
        """
        try:
            if not self.is_trained:
                self.logger.warning("Model not trained, returning zero predictions")
                if return_components:
                    return NBEATSPrediction(
                        forecast=np.zeros((X.shape[0], self.config.forecast_length)),
                        trend_component=np.zeros((X.shape[0], self.config.forecast_length)),
                        seasonality_component=np.zeros((X.shape[0], self.config.forecast_length)),
                        generic_component=np.zeros((X.shape[0], self.config.forecast_length)),
                        backcast=np.zeros((X.shape[0], self.config.backcast_length)),
                        confidence_interval=(np.zeros((X.shape[0], self.config.forecast_length)),
                                           np.zeros((X.shape[0], self.config.forecast_length))),
                        trend_direction='sideways',
                        seasonality_pattern={},
                        prediction_horizon=self.config.forecast_length,
                        timestamp=datetime.now(),
                        metadata={'model_trained': False}
                    )
                return np.zeros((X.shape[0], self.config.forecast_length))

            if not HAS_ML_LIBS or self.model is None:
                # Simulation mode
                forecasts = np.random.randn(X.shape[0], self.config.forecast_length) * 0.1
                if return_components:
                    return NBEATSPrediction(
                        forecast=forecasts,
                        trend_component=forecasts * 0.6,
                        seasonality_component=forecasts * 0.3,
                        generic_component=forecasts * 0.1,
                        backcast=X,
                        confidence_interval=(forecasts - 0.1, forecasts + 0.1),
                        trend_direction='sideways',
                        seasonality_pattern={'amplitude': 0.1, 'period': 12},
                        prediction_horizon=self.config.forecast_length,
                        timestamp=datetime.now(),
                        metadata={'simulation_mode': True}
                    )
                return forecasts

            # Scale input data
            X_scaled = self.scaler.transform(X)

            # Make predictions
            predictions = self.model.predict(X_scaled, verbose=0)

            main_forecast = predictions['forecast']

            if return_components:
                # Extract components
                trend_component = predictions.get('trend_forecast', 
                                                np.zeros_like(main_forecast))
                seasonality_component = predictions.get('seasonality_forecast',
                                                      np.zeros_like(main_forecast))
                generic_component = predictions.get('generic_forecast',
                                                  np.zeros_like(main_forecast))

                # Calculate confidence intervals (simplified)
                forecast_std = np.std(main_forecast, axis=0)
                confidence_lower = main_forecast - 1.96 * forecast_std
                confidence_upper = main_forecast + 1.96 * forecast_std

                # Determine trend direction
                trend_slope = np.mean(np.diff(trend_component, axis=1))
                if trend_slope > 0.01:
                    trend_direction = 'upward'
                elif trend_slope < -0.01:
                    trend_direction = 'downward'
                else:
                    trend_direction = 'sideways'

                # Seasonality pattern analysis
                seasonality_pattern = {
                    'amplitude': float(np.std(seasonality_component)),
                    'period': 12,  # Simplified
                    'phase': 0.0   # Simplified
                }

                return NBEATSPrediction(
                    forecast=main_forecast,
                    trend_component=trend_component,
                    seasonality_component=seasonality_component,
                    generic_component=generic_component,
                    backcast=X,
                    confidence_interval=(confidence_lower, confidence_upper),
                    trend_direction=trend_direction,
                    seasonality_pattern=seasonality_pattern,
                    prediction_horizon=self.config.forecast_length,
                    timestamp=datetime.now(),
                    metadata={
                        'model_trained': True,
                        'stack_types': self.config.stack_types,
                        'prediction_std': float(np.std(main_forecast))
                    }
                )

            return main_forecast

        except Exception as e:
            self.logger.error(f"Error making N-BEATS predictions: {e}")
            if return_components:
                return NBEATSPrediction(
                    forecast=np.zeros((X.shape[0], self.config.forecast_length)),
                    trend_component=np.zeros((X.shape[0], self.config.forecast_length)),
                    seasonality_component=np.zeros((X.shape[0], self.config.forecast_length)),
                    generic_component=np.zeros((X.shape[0], self.config.forecast_length)),
                    backcast=X,
                    confidence_interval=(np.zeros((X.shape[0], self.config.forecast_length)),
                                       np.zeros((X.shape[0], self.config.forecast_length))),
                    trend_direction='sideways',
                    seasonality_pattern={},
                    prediction_horizon=self.config.forecast_length,
                    timestamp=datetime.now(),
                    metadata={'error': str(e)}
                )
            return np.zeros((X.shape[0], self.config.forecast_length))

    def predict_single(self, sequence: np.ndarray) -> NBEATSPrediction:
        """Predict for a single sequence with full component decomposition"""
        if len(sequence.shape) == 1:
            sequence = sequence.reshape(1, -1)
        return self.predict(sequence, return_components=True)

    def save_model(self, filepath: str):
        """Save the trained N-BEATS model"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if HAS_ML_LIBS and self.model is not None:
                # Save Keras model
                self.model.save(f"{filepath}_nbeats_model.h5")

                # Save scaler
                with open(f"{filepath}_scaler.pkl", 'wb') as f:
                    pickle.dump(self.scaler, f)

                # Save training history and metadata
                metadata = {
                    'config': asdict(self.config),
                    'training_history': self.training_history,
                    'metrics': asdict(self.metrics) if self.metrics else None,
                    'is_trained': self.is_trained
                }

                with open(f"{filepath}_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

            self.logger.info(f"N-BEATS model saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving N-BEATS model: {e}")

    def load_model(self, filepath: str) -> bool:
        """Load a trained N-BEATS model"""
        try:
            filepath = Path(filepath)

            if HAS_ML_LIBS:
                # Load Keras model
                model_file = f"{filepath}_nbeats_model.h5"
                if Path(model_file).exists():
                    self.model = keras.models.load_model(model_file, custom_objects={
                        'NBEATSBlock': NBEATSBlock,
                        'NBEATSStack': NBEATSStack
                    })

                # Load scaler
                scaler_file = f"{filepath}_scaler.pkl"
                if Path(scaler_file).exists():
                    with open(scaler_file, 'rb') as f:
                        self.scaler = pickle.load(f)

                # Load metadata
                metadata_file = f"{filepath}_metadata.json"
                if Path(metadata_file).exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        self.training_history = metadata.get('training_history', {})
                        self.is_trained = metadata.get('is_trained', False)
                        if metadata.get('metrics'):
                            self.metrics = NBEATSMetrics(**metadata['metrics'])

            self.logger.info(f"N-BEATS model loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading N-BEATS model: {e}")
            return False

    def get_model_summary(self) -> Dict[str, Any]:
        """Get N-BEATS model summary and status"""
        summary = {
            'model_type': 'N-BEATS',
            'is_trained': self.is_trained,
            'config': asdict(self.config),
            'has_ml_libs': HAS_ML_LIBS
        }

        if self.model is not None:
            summary['parameters'] = self.model.count_params()
            summary['layers'] = len(self.model.layers)

        if self.metrics:
            summary['metrics'] = asdict(self.metrics)

        if self.training_history:
            summary['training_epochs'] = len(self.training_history.get('loss', []))
            summary['best_val_loss'] = min(self.training_history.get('val_loss', [float('inf')]))

        return summary

# Example usage and testing
if __name__ == "__main__":
    # Create N-BEATS forecaster
    forecaster = NBEATSForecaster()

    # Generate sample time series data
    n_samples = 500
    t = np.arange(n_samples)

    # Create synthetic time series with trend + seasonality + noise
    trend = 0.02 * t
    seasonality = 5 * np.sin(2 * np.pi * t / 50) + 2 * np.cos(2 * np.pi * t / 20)
    noise = np.random.randn(n_samples) * 0.5
    time_series = trend + seasonality + noise

    # Prepare data
    X, y = forecaster.prepare_data(time_series, time_series)

    print(f"Prepared N-BEATS data: X shape={X.shape}, y shape={y.shape}")

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Train model
    success = forecaster.train(X_train, y_train, X_val, y_val)
    print(f"N-BEATS training success: {success}")

    # Test prediction
    test_sequence = X_val[:1]
    prediction = forecaster.predict_single(test_sequence)
    print(f"N-BEATS prediction: forecast shape={prediction.forecast.shape}")
    print(f"Trend direction: {prediction.trend_direction}")
    print(f"Seasonality pattern: {prediction.seasonality_pattern}")

    # Get model summary
    summary = forecaster.get_model_summary()
    print(f"N-BEATS model summary: {summary}")
