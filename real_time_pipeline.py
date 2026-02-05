"""
Step 12: Real-Time Pipeline Rollout (Stubs)
Aggregate live feeds from multiple exchanges and process features across multiple models.
"""

import logging
import asyncio
import ccxt
import numpy as np
import torch
import torch.nn as nn
import importlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from ensemble_predictor import QuantumEnsemblePredictor, EnsembleConfig
from unified_ml_models import PyTorchCNNLSTM, PyTorchNBeats

class MultiExchangeFeed:
    """
    Multiplexer for live market data from multiple exchanges.
    Aggregates Binance, Coinbase, Kraken, KuCoin, and Bitfinex.
    """
    
    def __init__(self, exchanges: List[str], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        # Map exchange names to ccxt IDs
        self.name_map = {
            "Coinbase": "coinbase",
            "Binance": "binance",
            "Kraken": "kraken",
            "KuCoin": "kucoin",
            "Bitfinex": "bitfinex"
        }
        self.exchanges = [self.name_map.get(ex, ex.lower()) for ex in exchanges]
        self.clients: Dict[str, Any] = {}
        self.active = False
        self.logger.info(f"MultiExchangeFeed initialized for: {self.exchanges}")

    async def start(self):
        """Start all exchange feed connections."""
        self.active = True
        for ex_id in self.exchanges:
            try:
                client = getattr(ccxt, ex_id)({'enableRateLimit': True})
                self.clients[ex_id] = client
                self.logger.info(f"Initialized ccxt client for {ex_id}")
            except Exception as e:
                self.logger.error(f"Failed to initialize {ex_id}: {e}")
        
    async def stop(self):
        """Stop all exchange feed connections."""
        self.active = False
        self.clients = {}
        self.logger.info("Stopping real-time multi-exchange feeds...")

    async def get_aggregated_snapshot(self) -> Dict[str, Any]:
        """
        Aggregate the latest data from all active exchanges.
        Returns a unified market state.
        """
        if not self.active or not self.clients:
            return {}
            
        tasks = []
        for ex_id, client in self.clients.items():
            symbol = 'BTC/USDT' if ex_id != 'coinbase' else 'BTC/USD'
            tasks.append(self._fetch_ticker(ex_id, client, symbol))
            
        tickers = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_tickers = [t for t in tickers if isinstance(t, dict) and t.get('last') is not None]
        if not valid_tickers:
            return {}
            
        avg_price = sum(float(t['last']) for t in valid_tickers) / len(valid_tickers)
        
        return {
            'timestamp': datetime.now(timezone.utc),
            'avg_price': avg_price,
            'source_count': len(valid_tickers),
            'tickers': {t['exchange']: t for t in valid_tickers},
            'global_liquidity': sum(float(t.get('quoteVolume') or 0) for t in valid_tickers)
        }

    async def _fetch_ticker(self, ex_id: str, client: Any, symbol: str) -> Dict[str, Any]:
        try:
            ticker = await asyncio.to_thread(client.fetch_ticker, symbol)
            return {
                'exchange': ex_id,
                'last': ticker.get('last'),
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'quoteVolume': ticker.get('quoteVolume'),
                'timestamp': ticker.get('timestamp')
            }
        except Exception as e:
            self.logger.warning(f"Failed to fetch ticker from {ex_id}: {e}")
            return {'exchange': ex_id, 'error': str(e)}

class FeatureFanOutProcessor:
    """
    Processes features across multiple models in parallel.
    Target: 94% directional accuracy.
    """
    
    def __init__(self, models: List[str], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.model_names = models # Quantum Transformer, Bi-LSTM, CNN, VAE, Ensemble
        self.models: Dict[str, Any] = {}
        self._initialize_models()
        self.logger.info(f"FeatureFanOutProcessor initialized with models: {self.model_names}")

    def _initialize_models(self):
        """Initialize real ML models or stubs if weights unavailable."""
        try:
            # 1. Ensemble Predictor (PyTorch)
            if "Ensemble" in self.model_names:
                config = EnsembleConfig(n_base_models=5)
                predictor = QuantumEnsemblePredictor(config)
                # Add some base models
                for i in range(5):
                    model = nn.Sequential(
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.Tanh()
                    )
                    predictor.add_base_model(model, f"base_model_{i}")
                self.models["Ensemble"] = predictor
                
            # 2. CNN-LSTM (PyTorch Unified)
            if "CNN" in self.model_names:
                try:
                    model = PyTorchCNNLSTM(sequence_length=30, feature_count=10)
                    self.models["CNN"] = model
                    self.logger.info("CNN-LSTM (PyTorch) model initialized successfully")
                except Exception as e:
                    self.logger.warning(f"CNN-LSTM (PyTorch) initialization failed: {e}")

            # 3. Quantum Transformer (PyTorch)
            if "QuantumTransformer" in self.model_names:
                try:
                    from neural_network_prediction_engine import LegendaryNeuralPredictionEngine, PredictionConfig
                    nn_config = PredictionConfig()
                    nn_engine = LegendaryNeuralPredictionEngine(nn_config)
                    # Use input_dim=128 to match ensemble's feature vector for now
                    transformer = nn_engine._create_quantum_transformer(input_dim=128)
                    self.models["QuantumTransformer"] = transformer
                    self.logger.info("QuantumTransformer model initialized")
                except Exception as e:
                    self.logger.warning(f"QuantumTransformer initialization failed: {e}")

            # 4. Bi-LSTM (PyTorch)
            if "Bi-LSTM" in self.model_names:
                try:
                    from neural_network_prediction_engine import LegendaryNeuralPredictionEngine, PredictionConfig
                    nn_config = PredictionConfig()
                    nn_engine = LegendaryNeuralPredictionEngine(nn_config)
                    bilstm = nn_engine._create_bidirectional_lstm(input_dim=128)
                    self.models["Bi-LSTM"] = bilstm
                    self.logger.info("Bi-LSTM model initialized")
                except Exception as e:
                    self.logger.warning(f"Bi-LSTM initialization failed: {e}")

            # 5. VAE (PyTorch)
            if "VAE" in self.model_names:
                try:
                    from neural_network_prediction_engine import VariationalAutoEncoder
                    vae = VariationalAutoEncoder(input_dim=128, latent_dim=32)
                    self.models["VAE"] = vae
                    self.logger.info("VAE model initialized")
                except Exception as e:
                    self.logger.warning(f"VAE initialization failed: {e}")

            # 6. N-BEATS (PyTorch Unified)
            if "NBEATS" in self.model_names:
                try:
                    model = PyTorchNBeats(backcast_length=30, forecast_length=1)
                    self.models["NBEATS"] = model
                    self.logger.info("N-BEATS (PyTorch) model initialized")
                except Exception as e:
                    self.logger.warning(f"N-BEATS (PyTorch) initialization failed: {e}")
            
            self.logger.info("ML models initialized (PyTorch unified suite)")
            
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {e}")

    async def process_all_models(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Process features through all models and collect their predictions.
        """
        start_time = datetime.now(timezone.utc)
        predictions = {model: 0.0 for model in self.model_names}
        
        try:
            # Step 12/16 Bridge: Use real feature vector if provided, else fallback to random noise
            real_vector = features.get('feature_vector')
            if real_vector is not None:
                # Ensure it's the right shape (1, dim)
                if len(real_vector.shape) == 1:
                    feature_vector = real_vector.reshape(1, -1).astype(np.float32)
                else:
                    feature_vector = real_vector.astype(np.float32)
            else:
                # 1. Ensemble (PyTorch) input fallback
                feature_vector = np.random.randn(1, 128).astype(np.float32)
            
            # CNN (TensorFlow/PyTorch) input - expects (batch, seq, features)
            # Try to extract from price_df if provided
            price_df = features.get('price_df')
            if price_df is not None and not price_df.empty:
                # Extract last 30 bars, 10 technical features (simplified)
                # In production, we'd use a dedicated scaler here
                last_30 = price_df.tail(30)
                if len(last_30) == 30:
                    # Select numeric columns and normalize
                    numeric_df = last_30.select_dtypes(include=[np.number])
                    # Use up to 10 features
                    cols = list(numeric_df.columns)[:10]
                    cnn_data = numeric_df[cols].values.reshape(1, 30, len(cols)).astype(np.float32)
                else:
                    cnn_data = np.random.randn(1, 30, 10).astype(np.float32)
            else:
                # 2. CNN fallback
                cnn_data = np.random.randn(1, 30, 10).astype(np.float32)
            
            tasks = []
            for name in self.model_names:
                if name == "Ensemble" and "Ensemble" in self.models:
                    tasks.append(self._run_ensemble_inference(feature_vector))
                elif name == "CNN" and "CNN" in self.models:
                    tasks.append(self._run_cnn_inference(cnn_data))
                elif name == "QuantumTransformer" and "QuantumTransformer" in self.models:
                    tasks.append(self._run_transformer_inference(feature_vector))
                elif name == "Bi-LSTM" and "Bi-LSTM" in self.models:
                    tasks.append(self._run_bilstm_inference(feature_vector))
                elif name == "VAE" and "VAE" in self.models:
                    tasks.append(self._run_vae_inference(feature_vector))
                elif name == "NBEATS" and "NBEATS" in self.models:
                    tasks.append(self._run_nbeats_inference(feature_vector))
                else:
                    # Stubs for others
                    tasks.append(self._run_stub_inference(name))
            
            results = await asyncio.gather(*tasks)
            for name, pred in results:
                predictions[name] = float(pred)
                
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.logger.info(f"Processed {len(self.model_names)} models in {processing_time:.4f}s")
        
        return predictions

    async def _run_ensemble_inference(self, X: np.ndarray) -> Tuple[str, float]:
        try:
            result = await self.models["Ensemble"].predict(X)
            # Predictions is an array, take first element
            return "Ensemble", float(result['predictions'][0])
        except Exception as e:
            self.logger.warning(f"Ensemble inference failed: {e}")
            return "Ensemble", 0.0

    async def _run_cnn_inference(self, X: np.ndarray) -> Tuple[str, float]:
        try:
            model = self.models["CNN"]
            model.eval()
            with torch.no_grad():
                # X is already (batch, seq, features) = (1, 30, 10)
                X_tensor = torch.FloatTensor(X)
                output = model(X_tensor)
                # price_direction is [UP, DOWN, SIDEWAYS] probabilities
                direction_probs = output['price_direction'][0]
                # Direction score = prob(UP) - prob(DOWN)
                score = float(direction_probs[0] - direction_probs[1])
                return "CNN", score
        except Exception as e:
            self.logger.warning(f"CNN inference failed: {e}")
            return "CNN", 0.0

    async def _run_transformer_inference(self, X: np.ndarray) -> Tuple[str, float]:
        try:
            model = self.models["QuantumTransformer"]
            # model is LegendaryNeuralPredictionEngine's internal transformer or stub
            if hasattr(model, 'eval'):
                model.eval()
            
            with torch.no_grad():
                # X is (1, 128)
                X_tensor = torch.FloatTensor(X)
                if len(X_tensor.shape) == 2:
                    X_tensor = X_tensor.unsqueeze(1).expand(-1, 30, -1) # (1, 30, 128)
                
                # Check if model is a torch module or the wrapper
                if isinstance(model, nn.Module):
                    output = model(X_tensor)
                else:
                    # It might be the engine itself
                    return "QuantumTransformer", 0.0
                
                # output is (predictions_dict, uncertainty)
                if isinstance(output, tuple):
                    predictions_dict = output[0]
                    # Use the first available horizon
                    first_horizon = list(predictions_dict.keys())[0]
                    score = float(predictions_dict[first_horizon][0][0])
                elif isinstance(output, dict):
                    # Fallback for dict
                    first_horizon = list(output.keys())[0]
                    score = float(output[first_horizon][0][0])
                else:
                    # Fallback for tensor output
                    score = float(output[0][0])
                return "QuantumTransformer", score
        except Exception as e:
            self.logger.warning(f"QuantumTransformer inference failed: {e}")
            return "QuantumTransformer", 0.0

    async def _run_bilstm_inference(self, X: np.ndarray) -> Tuple[str, float]:
        try:
            model = self.models["Bi-LSTM"]
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).unsqueeze(1).expand(-1, 30, -1)
                output = model(X_tensor)
                if isinstance(output, dict):
                    # output is predictions_dict with horizon keys
                    first_horizon = list(output.keys())[0]
                    score = float(output[first_horizon][0][0])
                else:
                    score = float(output[0][0])
                return "Bi-LSTM", score
        except Exception as e:
            self.logger.warning(f"Bi-LSTM inference failed: {e}")
            return "Bi-LSTM", 0.0

    async def _run_vae_inference(self, X: np.ndarray) -> Tuple[str, float]:
        try:
            model = self.models["VAE"]
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                # VAE returns reconstruction, mu, logvar
                recon, mu, logvar = model(X_tensor)
                # Latent space mu as a proxy for 'uniqueness' or 'signal'
                score = float(torch.tanh(mu.mean()))
                return "VAE", score
        except Exception as e:
            self.logger.warning(f"VAE inference failed: {e}")
            return "VAE", 0.0

    async def _run_nbeats_inference(self, X: np.ndarray) -> Tuple[str, float]:
        try:
            model = self.models["NBEATS"]
            model.eval()
            with torch.no_grad():
                # N-BEATS expects (batch, backcast_length)
                X_tensor = torch.FloatTensor(X[:, 0, 0]).unsqueeze(0) # Simple stub for shape
                if X_tensor.shape[1] != 30:
                    X_tensor = torch.randn(1, 30)
                
                output = model(X_tensor)
                forecast = output['forecast'][0] # [horizon=1]
                # Dummy direction from mean forecast
                score = float(torch.tanh(forecast.mean()))
                return "NBEATS", score
        except Exception as e:
            self.logger.warning(f"N-BEATS inference failed: {e}")
            return "NBEATS", 0.0

    async def _run_stub_inference(self, name: str) -> Tuple[str, float]:
        await asyncio.sleep(0.005)
        # Return a deterministic-ish but varying value for stubs based on name
        # to see some action in the logs
        val = (sum(ord(c) for c in name) % 100) / 50.0 - 1.0 # -1 to 1
        return name, val

class RealTimePipeline:
    """
    Orchestrates the Step 12 Real-Time Pipeline.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = config.get("enabled", False)
        
        exchanges = config.get("exchanges", ["Coinbase", "Binance", "Kraken", "KuCoin", "Bitfinex"])
        self.feed = MultiExchangeFeed(exchanges, logger=self.logger)
        
        models = config.get("models", ["QuantumTransformer", "Bi-LSTM", "CNN", "VAE", "Ensemble"])
        self.processor = FeatureFanOutProcessor(models, logger=self.logger)
        
        self.logger.info(f"RealTimePipeline initialized (Enabled: {self.enabled})")

    async def start(self):
        if self.enabled and not self.feed.active:
            await self.feed.start()

    async def run_cycle(self) -> Dict[str, Any]:
        """Execute one real-time pipeline cycle."""
        if not self.enabled:
            return {}
            
        snapshot = await self.feed.get_aggregated_snapshot()
        if not snapshot:
            self.logger.warning("Real-time pipeline failed to get aggregated snapshot")
            return {}

        # Basic feature extraction from snapshot
        features = {
            'avg_price': snapshot['avg_price'],
            'global_liquidity': snapshot['global_liquidity'],
            'source_count': snapshot['source_count']
        }
        
        predictions = await self.processor.process_all_models(features)
        
        result = {
            'snapshot': snapshot,
            'features': features,
            'predictions': predictions,
            'timestamp': datetime.now(timezone.utc)
        }
        
        self.logger.info(f"Real-time cycle completed: Avg Price {snapshot['avg_price']:.2f} "
                       f"from {snapshot['source_count']} sources")
        return result
