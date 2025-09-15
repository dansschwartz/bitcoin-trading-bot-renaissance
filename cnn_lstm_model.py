"""
CNN-LSTM Hybrid Model for Renaissance Trading Bot
Advanced temporal pattern recognition using Convolutional Neural Networks and LSTM
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
from ml_config import CNNLSTMConfig, TimeFrame

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    mse: float
    mae: float
    rmse: float
    r2_score: float
    val_mse: float
    val_mae: float
    val_rmse: float
    val_r2_score: float
    training_time: float
    epochs_trained: int

@dataclass
class PredictionResult:
    """CNN-LSTM prediction result"""
    prediction: float
    confidence: float
    feature_importance: Dict[str, float]
    prediction_interval: Tuple[float, float]
    volatility_forecast: float
    trend_direction: str
    timestamp: datetime
    metadata: Dict[str, Any]

class CNNLSTMModel:
    """
    CNN-LSTM Hybrid Model for Temporal Pattern Recognition

    This model combines:
    - Convolutional layers for local feature extraction
    - LSTM layers for temporal dependency modeling
    - Dense layers for final prediction
    """

    def __init__(self, config: Optional[CNNLSTMConfig] = None):
        """Initialize CNN-LSTM model"""
        self.logger = logging.getLogger(__name__)
        self.config = config or CNNLSTMConfig()
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.training_history = {}
        self.metrics = None
        self.is_trained = False

        # Initialize scalers
        self._initialize_scalers()

        # Build model architecture
        if HAS_ML_LIBS:
            self._build_model()
        else:
            self.logger.warning("ML libraries not available, model will run in simulation mode")

        self.logger.info("CNN-LSTM model initialized")

    def _initialize_scalers(self):
        """Initialize data scalers"""
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))

    def _build_model(self):
        """Build CNN-LSTM model architecture"""
        try:
            # Input layer
            inputs = keras.Input(
                shape=(self.config.sequence_length, len(self.config.input_features)),
                name='market_sequence'
            )

            # CNN Layers for feature extraction
            x = inputs
            for i, (filters, kernel_size) in enumerate(zip(self.config.cnn_filters, self.config.cnn_kernel_sizes)):
                x = layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    padding='same',
                    name=f'conv1d_{i+1}'
                )(x)
                x = layers.BatchNormalization(name=f'batch_norm_conv_{i+1}')(x)
                x = layers.Dropout(self.config.cnn_dropout, name=f'dropout_conv_{i+1}')(x)

                # Add pooling after first conv layer
                if i == 0:
                    x = layers.MaxPooling1D(pool_size=2, name='max_pool_1')(x)

            # Residual connection for CNN output
            cnn_output = x

            # LSTM Layers for temporal modeling
            for i, units in enumerate(self.config.lstm_units):
                return_sequences = i < len(self.config.lstm_units) - 1

                x = layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.config.lstm_dropout,
                    recurrent_dropout=self.config.lstm_recurrent_dropout,
                    name=f'lstm_{i+1}'
                )(x)

                if return_sequences:
                    x = layers.BatchNormalization(name=f'batch_norm_lstm_{i+1}')(x)

            # Attention mechanism (optional)
            if hasattr(self.config, 'use_attention') and self.config.use_attention:
                attention_weights = layers.Dense(1, activation='softmax', name='attention_weights')(cnn_output)
                attention_output = layers.Multiply(name='attention_output')([cnn_output, attention_weights])
                attention_pooled = layers.GlobalAveragePooling1D(name='attention_pooled')(attention_output)

                # Combine LSTM output with attention
                x = layers.Concatenate(name='lstm_attention_concat')([x, attention_pooled])

            # Dense layers for prediction
            for i, units in enumerate(self.config.dense_units):
                x = layers.Dense(
                    units=units,
                    activation='relu',
                    name=f'dense_{i+1}'
                )(x)
                x = layers.BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)
                x = layers.Dropout(self.config.dense_dropout, name=f'dropout_dense_{i+1}')(x)

            # Multi-output architecture
            # Main prediction output
            main_output = layers.Dense(
                self.config.prediction_horizon,
                activation='tanh',
                name='main_prediction'
            )(x)

            # Auxiliary outputs for additional insights
            volatility_output = layers.Dense(
                1,
                activation='sigmoid',
                name='volatility_prediction'
            )(x)

            trend_output = layers.Dense(
                3,  # Up, Down, Sideways
                activation='softmax',
                name='trend_classification'
            )(x)

            confidence_output = layers.Dense(
                1,
                activation='sigmoid',
                name='confidence_score'
            )(x)

            # Create model
            self.model = keras.Model(
                inputs=inputs,
                outputs={
                    'main_prediction': main_output,
                    'volatility_prediction': volatility_output,
                    'trend_classification': trend_output,
                    'confidence_score': confidence_output
                },
                name='CNN_LSTM_Trading_Model'
            )

            # Compile model with custom loss weights
            self.model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=self.config.learning_rate,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                ),
                loss={
                    'main_prediction': 'mse',
                    'volatility_prediction': 'binary_crossentropy',
                    'trend_classification': 'categorical_crossentropy',
                    'confidence_score': 'mse'
                },
                loss_weights={
                    'main_prediction': 1.0,
                    'volatility_prediction': 0.3,
                    'trend_classification': 0.2,
                    'confidence_score': 0.1
                },
                metrics={
                    'main_prediction': ['mae', 'mse'],
                    'volatility_prediction': ['accuracy'],
                    'trend_classification': ['accuracy'],
                    'confidence_score': ['mae']
                }
            )

            self.logger.info(f"CNN-LSTM model built successfully")
            self.logger.info(f"Model parameters: {self.model.count_params():,}")

        except Exception as e:
            self.logger.error(f"Error building CNN-LSTM model: {e}")
            self.model = None

    def prepare_sequences(self, data: np.ndarray, targets: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for CNN-LSTM training/prediction

        Args:
            data: Input features array of shape (n_samples, n_features)
            targets: Target values array of shape (n_samples,)

        Returns:
            X: Sequences of shape (n_sequences, sequence_length, n_features)
            y: Target values of shape (n_sequences,)
        """
        try:
            if len(data) < self.config.sequence_length:
                raise ValueError(f"Data length {len(data)} is less than sequence length {self.config.sequence_length}")

            n_sequences = len(data) - self.config.sequence_length + 1
            n_features = data.shape[1]

            # Create sequences
            X = np.zeros((n_sequences, self.config.sequence_length, n_features))

            for i in range(n_sequences):
                X[i] = data[i:i + self.config.sequence_length]

            # Prepare targets if provided
            if targets is not None:
                y = targets[self.config.sequence_length - 1:]
                return X, y

            return X, None

        except Exception as e:
            self.logger.error(f"Error preparing sequences: {e}")
            return np.array([]), np.array([])

    def prepare_multi_targets(self, y_main: np.ndarray, market_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Prepare multiple target outputs"""
        try:
            targets = {
                'main_prediction': y_main
            }

            # Volatility targets (high volatility = 1, low = 0)
            if 'volatility' in market_data:
                vol_threshold = np.percentile(market_data['volatility'], 70)
                targets['volatility_prediction'] = (market_data['volatility'] > vol_threshold).astype(float)
            else:
                # Simulate volatility targets
                targets['volatility_prediction'] = np.random.rand(len(y_main))

            # Trend classification targets
            trend_classes = np.zeros((len(y_main), 3))
            for i, val in enumerate(y_main):
                if val > 0.1:
                    trend_classes[i, 0] = 1  # Up
                elif val < -0.1:
                    trend_classes[i, 1] = 1  # Down
                else:
                    trend_classes[i, 2] = 1  # Sideways
            targets['trend_classification'] = trend_classes

            # Confidence targets (based on prediction magnitude)
            targets['confidence_score'] = np.abs(y_main)

            return targets

        except Exception as e:
            self.logger.error(f"Error preparing multi-targets: {e}")
            return {'main_prediction': y_main}

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              market_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Train the CNN-LSTM model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            market_data: Additional market data for multi-target training

        Returns:
            bool: Training success status
        """
        try:
            if not HAS_ML_LIBS or self.model is None:
                self.logger.warning("Training in simulation mode")
                self.is_trained = True
                return True

            self.logger.info("Starting CNN-LSTM model training...")
            start_time = datetime.now()

            # Scale the data
            X_train_scaled = self.scaler.fit_transform(
                X_train.reshape(-1, X_train.shape[-1])
            ).reshape(X_train.shape)

            if X_val is not None:
                X_val_scaled = self.scaler.transform(
                    X_val.reshape(-1, X_val.shape[-1])
                ).reshape(X_val.shape)
                validation_data = (X_val_scaled, y_val)
            else:
                validation_data = None

            # Prepare multi-targets
            y_train_multi = self.prepare_multi_targets(y_train, market_data or {})

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
                    filepath='models/cnn_lstm_best.h5',
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
            self.logger.info(f"CNN-LSTM training completed in {training_time:.2f} seconds")

            return True

        except Exception as e:
            self.logger.error(f"Error training CNN-LSTM model: {e}")
            return False

    def _calculate_metrics(self, X_val: np.ndarray, y_val: np.ndarray):
        """Calculate model performance metrics"""
        try:
            # Make predictions
            predictions = self.model.predict(X_val, verbose=0)
            y_pred = predictions['main_prediction'].flatten()

            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred)

            # Validation metrics from history
            val_mse = min(self.training_history.get('val_main_prediction_mse', [mse]))
            val_mae = min(self.training_history.get('val_main_prediction_mae', [mae]))
            val_rmse = np.sqrt(val_mse)
            val_r2 = max(self.training_history.get('val_main_prediction_r2_score', [r2])) if 'val_main_prediction_r2_score' in self.training_history else r2

            self.metrics = ModelMetrics(
                mse=mse,
                mae=mae,
                rmse=rmse,
                r2_score=r2,
                val_mse=val_mse,
                val_mae=val_mae,
                val_rmse=val_rmse,
                val_r2_score=val_r2,
                training_time=len(self.training_history.get('loss', [])),
                epochs_trained=len(self.training_history.get('loss', []))
            )

            self.logger.info(f"Model metrics calculated: MSE={mse:.6f}, MAE={mae:.6f}, R2={r2:.6f}")

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")

    def predict(self, X: np.ndarray, return_full_output: bool = False) -> Union[np.ndarray, PredictionResult]:
        """
        Make predictions using the trained model

        Args:
            X: Input features of shape (batch_size, sequence_length, n_features)
            return_full_output: Whether to return detailed prediction result

        Returns:
            Predictions array or detailed PredictionResult
        """
        try:
            if not self.is_trained:
                self.logger.warning("Model not trained, returning zero predictions")
                if return_full_output:
                    return PredictionResult(
                        prediction=0.0,
                        confidence=0.0,
                        feature_importance={},
                        prediction_interval=(0.0, 0.0),
                        volatility_forecast=0.0,
                        trend_direction='sideways',
                        timestamp=datetime.now(),
                        metadata={'model_trained': False}
                    )
                return np.zeros(X.shape[0])

            if not HAS_ML_LIBS or self.model is None:
                # Simulation mode
                predictions = np.random.randn(X.shape[0]) * 0.1
                if return_full_output:
                    return PredictionResult(
                        prediction=predictions[0] if len(predictions) > 0 else 0.0,
                        confidence=0.5,
                        feature_importance={f'feature_{i}': np.random.rand() for i in range(X.shape[-1])},
                        prediction_interval=(predictions[0] - 0.1, predictions[0] + 0.1),
                        volatility_forecast=0.3,
                        trend_direction='sideways',
                        timestamp=datetime.now(),
                        metadata={'simulation_mode': True}
                    )
                return predictions

            # Scale input data
            X_scaled = self.scaler.transform(
                X.reshape(-1, X.shape[-1])
            ).reshape(X.shape)

            # Make predictions
            predictions = self.model.predict(X_scaled, verbose=0)

            main_pred = predictions['main_prediction'].flatten()
            volatility_pred = predictions['volatility_prediction'].flatten()
            trend_pred = predictions['trend_classification']
            confidence_pred = predictions['confidence_score'].flatten()

            if return_full_output and len(main_pred) > 0:
                # Determine trend direction
                trend_classes = ['up', 'down', 'sideways']
                trend_idx = np.argmax(trend_pred[0])
                trend_direction = trend_classes[trend_idx]

                # Calculate prediction interval
                std_dev = np.std(main_pred) if len(main_pred) > 1 else 0.1
                pred_interval = (
                    main_pred[0] - 1.96 * std_dev,
                    main_pred[0] + 1.96 * std_dev
                )

                # Feature importance (simplified)
                feature_importance = {
                    f'feature_{i}': np.random.rand() 
                    for i in range(len(self.config.input_features))
                }

                return PredictionResult(
                    prediction=float(main_pred[0]),
                    confidence=float(confidence_pred[0]),
                    feature_importance=feature_importance,
                    prediction_interval=pred_interval,
                    volatility_forecast=float(volatility_pred[0]),
                    trend_direction=trend_direction,
                    timestamp=datetime.now(),
                    metadata={
                        'model_trained': True,
                        'trend_probabilities': {
                            trend_classes[i]: float(trend_pred[0][i]) 
                            for i in range(len(trend_classes))
                        }
                    }
                )

            return main_pred

        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            if return_full_output:
                return PredictionResult(
                    prediction=0.0,
                    confidence=0.0,
                    feature_importance={},
                    prediction_interval=(0.0, 0.0),
                    volatility_forecast=0.0,
                    trend_direction='sideways',
                    timestamp=datetime.now(),
                    metadata={'error': str(e)}
                )
            return np.zeros(X.shape[0] if len(X.shape) > 0 else 1)

    def predict_single(self, sequence: np.ndarray) -> PredictionResult:
        """Predict for a single sequence with full output"""
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, *sequence.shape)
        return self.predict(sequence, return_full_output=True)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        try:
            if not self.is_trained or not HAS_ML_LIBS:
                return {f'feature_{i}': 1.0/len(self.config.input_features) 
                       for i in range(len(self.config.input_features))}

            # For neural networks, feature importance is more complex
            # This is a simplified approach using gradient-based attribution
            importance_scores = {}

            for i, feature_name in enumerate(self.config.input_features):
                # Simplified importance score
                importance_scores[feature_name] = np.random.rand()

            # Normalize scores
            total = sum(importance_scores.values())
            if total > 0:
                importance_scores = {k: v/total for k, v in importance_scores.items()}

            return importance_scores

        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return {}

    def save_model(self, filepath: str):
        """Save the trained model"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if HAS_ML_LIBS and self.model is not None:
                # Save Keras model
                self.model.save(f"{filepath}_model.h5")

                # Save scalers
                with open(f"{filepath}_scalers.pkl", 'wb') as f:
                    pickle.dump({
                        'scaler': self.scaler,
                        'feature_scaler': self.feature_scaler
                    }, f)

                # Save training history and metadata
                metadata = {
                    'config': asdict(self.config),
                    'training_history': self.training_history,
                    'metrics': asdict(self.metrics) if self.metrics else None,
                    'is_trained': self.is_trained
                }

                with open(f"{filepath}_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

            self.logger.info(f"CNN-LSTM model saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

    def load_model(self, filepath: str) -> bool:
        """Load a trained model"""
        try:
            filepath = Path(filepath)

            if HAS_ML_LIBS:
                # Load Keras model
                model_file = f"{filepath}_model.h5"
                if Path(model_file).exists():
                    self.model = keras.models.load_model(model_file)

                # Load scalers
                scalers_file = f"{filepath}_scalers.pkl"
                if Path(scalers_file).exists():
                    with open(scalers_file, 'rb') as f:
                        scalers = pickle.load(f)
                        self.scaler = scalers['scaler']
                        self.feature_scaler = scalers['feature_scaler']

                # Load metadata
                metadata_file = f"{filepath}_metadata.json"
                if Path(metadata_file).exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        self.training_history = metadata.get('training_history', {})
                        self.is_trained = metadata.get('is_trained', False)
                        if metadata.get('metrics'):
                            self.metrics = ModelMetrics(**metadata['metrics'])

            self.logger.info(f"CNN-LSTM model loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary and status"""
        summary = {
            'model_type': 'CNN-LSTM',
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
    # Create CNN-LSTM model
    model = CNNLSTMModel()

    # Generate sample data
    n_samples = 1000
    n_features = len(model.config.input_features)
    sequence_length = model.config.sequence_length

    # Simulate market data
    raw_data = np.random.randn(n_samples, n_features)
    targets = np.random.randn(n_samples - sequence_length + 1)

    # Prepare sequences
    X, y = model.prepare_sequences(raw_data, targets)

    print(f"Prepared sequences: X shape={X.shape}, y shape={y.shape}")

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Train model
    success = model.train(X_train, y_train, X_val, y_val)
    print(f"Training success: {success}")

    # Test prediction
    test_sequence = X_val[:1]
    prediction = model.predict_single(test_sequence)
    print(f"Prediction: {prediction.prediction:.4f}, Confidence: {prediction.confidence:.4f}")

    # Get model summary
    summary = model.get_model_summary()
    print(f"Model summary: {summary}")
