"""
Step 12: Real-Time Pipeline Rollout
Aggregate live feeds from multiple exchanges and process features across trained models.
"""

import logging
import asyncio
import ccxt
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

try:
    from ml_model_loader import load_trained_models, predict_with_models, build_feature_sequence
    from ml_model_loader import load_crash_lgbm, build_crash_features, predict_crash_lgbm
    HAS_TRAINED_MODELS = True
except ImportError:
    HAS_TRAINED_MODELS = False

try:
    from crash_model_loader import CrashModelLoader
    from crash_feature_builder import CrashFeatureBuilder
    HAS_CRASH_MULTI = True
except ImportError:
    HAS_CRASH_MULTI = False

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
    Processes features across trained PyTorch models.
    Uses ml_model_loader for exact-match architecture loading.
    """

    def __init__(self, models: List[str], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.model_names = models
        self.models: Dict[str, Any] = {}
        self._trained_models: Dict[str, Any] = {}
        self._cross_data: Optional[Dict[str, Any]] = None
        self._current_pair: Optional[str] = None
        self._initialize_models()

    def _initialize_models(self):
        """Load trained PyTorch models with exact architecture matching."""
        if not HAS_TRAINED_MODELS:
            self.logger.warning("ml_model_loader not available — RT pipeline models disabled")
            return

        try:
            self._trained_models = load_trained_models()
            # Map trained model names to display names for DB/dashboard
            self._name_map = {
                'quantum_transformer': 'QuantumTransformer',
                'bidirectional_lstm': 'BiLSTM',
                'dilated_cnn': 'DilatedCNN',
                'cnn': 'CNN',
                'gru': 'GRU',
                'meta_ensemble': 'MetaEnsemble',
                'lightgbm': 'LightGBM',
            }
            for trained_name, model in self._trained_models.items():
                rt_name = self._name_map.get(trained_name, trained_name)
                self.models[rt_name] = model

            self.logger.info(
                f"FeatureFanOutProcessor: {len(self._trained_models)} trained models loaded "
                f"({list(self._trained_models.keys())})"
            )
        except Exception as e:
            self.logger.error(f"Error initializing trained models: {e}")

        # Load crash-regime LightGBM (legacy BTC-only, kept for backward compat)
        self._crash_lgbm = None
        self._crash_meta = None
        try:
            self._crash_lgbm, self._crash_meta = load_crash_lgbm()
            if self._crash_lgbm:
                n_feat = self._crash_meta.get('n_features', '?') if self._crash_meta else '?'
                self.logger.info(f"Crash-regime LightGBM loaded ({n_feat} features)")
        except Exception as e:
            self.logger.warning(f"Crash LightGBM load skipped: {e}")

        # Load multi-asset crash models (v2 — replaces BTC-only when available)
        self._crash_loader: Optional['CrashModelLoader'] = None
        self._crash_builder: Optional['CrashFeatureBuilder'] = None
        if HAS_CRASH_MULTI:
            try:
                self._crash_loader = CrashModelLoader(logger_=self.logger)
                self._crash_builder = CrashFeatureBuilder()
                if self._crash_loader.model_count > 0:
                    self.logger.info(
                        f"Multi-asset crash models: {self._crash_loader.available_models}"
                    )
                    # Disable legacy single-model loader when multi is available
                    self._crash_lgbm = None
                    self._crash_meta = None
            except Exception as e:
                self.logger.warning(f"Multi-asset crash loader failed: {e}")

    def set_cross_data(self, cross_data: Optional[Dict[str, Any]], pair_name: Optional[str] = None) -> None:
        """Set cross-asset data for the next inference call.

        Args:
            cross_data: Dict of pair_name → DataFrame with [close, volume] columns
            pair_name: The current pair being processed
        """
        self._cross_data = cross_data
        self._current_pair = pair_name

    async def process_all_models(self, features: Dict[str, Any],
                                price_df=None,
                                cross_data: Optional[Dict[str, Any]] = None,
                                pair_name: Optional[str] = None,
                                derivatives_data: Optional[Dict[str, Any]] = None,
                                macro_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Process features through trained models and collect predictions.

        Args:
            features: Dict with snapshot data (avg_price, etc.)
            price_df: Optional OHLCV DataFrame for build_feature_sequence().
                      If provided, overrides features.get('price_df').
            cross_data: Optional dict of pair → DataFrame for cross-asset features.
            pair_name: Current pair name for cross-asset feature computation.
            derivatives_data: Optional dict of feature_name → pd.Series for derivatives
                features (funding_rate, open_interest, etc.).
            macro_data: Optional dict with daily macro features for crash model.
        """
        start_time = datetime.now(timezone.utc)
        predictions = {}

        if not self._trained_models:
            return predictions

        # Use explicit args, fall back to instance state
        _cross = cross_data if cross_data is not None else self._cross_data
        _pair = pair_name if pair_name is not None else self._current_pair

        try:
            # Use explicit price_df arg, fallback to features dict
            if price_df is None:
                price_df = features.get('price_df')

            feat_array = None
            if price_df is not None and hasattr(price_df, 'empty') and not price_df.empty:
                feat_array = build_feature_sequence(
                    price_df, seq_len=30,
                    cross_data=_cross, pair_name=_pair,
                    derivatives_data=derivatives_data,
                )

            if feat_array is None:
                # No valid features — return empty so caller uses fallback
                _pdf_len = len(price_df) if price_df is not None and hasattr(price_df, '__len__') else 'N/A'
                self.logger.warning(
                    f"build_feature_sequence returned None (price_df rows={_pdf_len}, need≥30)"
                )
                return {}

            # Extract close prices for LightGBM momentum features
            _price_series = None
            if price_df is not None and 'close' in price_df.columns:
                _price_series = price_df['close'].values.astype(float)

            # Run inference on trained models
            raw_preds, _confidences = predict_with_models(self._trained_models, feat_array, price_series=_price_series)

            # Map to display names
            for trained_name, pred_val in raw_preds.items():
                rt_name = self._name_map.get(trained_name, trained_name)
                predictions[rt_name] = float(pred_val)

        except Exception as e:
            self.logger.error(f"Inference error: {e}", exc_info=True)

        # Crash-regime LightGBM — multi-asset (preferred) or BTC-only legacy
        _pair_str = str(_pair).upper() if _pair else ''
        _asset = None
        for _a in ('BTC', 'ETH', 'SOL', 'XRP', 'DOGE'):
            if _a in _pair_str:
                _asset = _a
                break

        if self._crash_loader and self._crash_builder and _asset:
            # Multi-asset crash models (run both horizons if available)
            for _horizon in ('2bar', '1bar'):
                model, meta = self._crash_loader.get_model(_asset, _horizon)
                if model is None:
                    continue
                try:
                    # Resolve cross-asset DataFrame from cross_data dict
                    _cross_df = None
                    _lead = self._crash_loader.get_cross_asset(_asset)
                    if _cross:
                        for _ck in _cross:
                            if _lead in str(_ck).upper():
                                _cross_df = _cross[_ck]
                                break

                    crash_feats = self._crash_builder.build(
                        asset=_asset,
                        price_df=price_df,
                        cross_price_df=_cross_df,
                        derivatives_data=derivatives_data,
                        macro_data=macro_data,
                    )
                    if crash_feats is not None:
                        crash_pred, crash_conf, crash_src = self._crash_loader.predict_for_asset(
                            _asset, _horizon, crash_feats
                        )
                        _key = f"CrashRegime_{_horizon}"
                        predictions[_key] = float(crash_pred)
                        # Also set legacy key for backward compat (2bar is primary)
                        if _horizon == '2bar':
                            predictions['CrashRegime'] = float(crash_pred)
                        self.logger.info(
                            f"Crash {_asset}_{_horizon}: pred={crash_pred:.4f} "
                            f"conf={crash_conf:.4f} src={crash_src}"
                        )
                except Exception as e:
                    self.logger.warning(f"Crash {_asset}_{_horizon} inference skipped: {e}")

        elif self._crash_lgbm is not None and 'BTC' in _pair_str:
            # Legacy BTC-only fallback
            try:
                crash_feats = build_crash_features(
                    price_df, derivatives_data=derivatives_data,
                    macro_data=macro_data, cross_data=_cross,
                )
                if crash_feats is not None:
                    crash_pred, crash_conf = predict_crash_lgbm(self._crash_lgbm, crash_feats)
                    predictions['CrashRegime'] = float(crash_pred)
                    self.logger.info(
                        f"CrashRegime pred={crash_pred:.4f} conf={crash_conf:.4f} (BTC legacy)"
                    )
            except Exception as e:
                self.logger.warning(f"Crash LightGBM inference skipped: {e}")

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.logger.info(f"Processed {len(predictions)} trained models in {processing_time:.4f}s")
        return predictions

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

    async def run_cycle(self, price_df=None) -> Dict[str, Any]:
        """Execute one real-time pipeline cycle.

        Args:
            price_df: Optional OHLCV DataFrame for ML model inference.
        """
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

        predictions = await self.processor.process_all_models(features, price_df=price_df)
        
        result = {
            'snapshot': snapshot,
            'features': features,
            'predictions': predictions,
            'timestamp': datetime.now(timezone.utc)
        }
        
        self.logger.info(f"Real-time cycle completed: Avg Price {snapshot['avg_price']:.2f} "
                       f"from {snapshot['source_count']} sources")
        return result
