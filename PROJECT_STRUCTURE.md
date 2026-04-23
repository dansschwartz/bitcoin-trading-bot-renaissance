# Project Structure Reorganization Guide

> **Purpose:** Document the proposed reorganization of ~119 root-level `.py` files into
> logical packages. This is a migration guide — no files have been moved yet.
>
> **Generated:** 2026-04-23
>
> **Strategy:** Create package directories with `__init__.py` re-exports so old imports
> continue to work while new `from package.module import Class` imports also work.
> Migrate imports file-by-file over time, then remove root stubs.

---

## Current State

- **119 `.py` files** at the project root
- **16 existing packages** with `__init__.py`: `bot/`, `core/`, `intelligence/`, `signals/`,
  `arbitrage/`, `dashboard/`, `data_module/`, `execution/`, `monitoring/`, `recovery/`,
  `orchestrator/`, `agents/`, `oracle/`, `portfolio/`, `backtesting/`, `researchers/`
- **2 central hub files** that import most modules:
  - `renaissance_trading_bot.py` — imports 30+ root modules, imported by 30+ files
  - `bot/builder.py` — imports 25+ root modules (factory/builder pattern)

---

## Proposed Package Groups

### Group A — `data/` (Data & Market Providers)

Modules that fetch, cache, or provide market data from exchanges and alternative sources.

| File | Exports | Imported By | Import Count |
|------|---------|-------------|:------------:|
| `market_data_provider.py` | `LiveMarketDataProvider` | `renaissance_trading_bot` | 1 |
| `binance_spot_provider.py` | `BinanceSpotProvider`, `to_binance_symbol`, `from_binance_symbol` | `renaissance_trading_bot` | 1 |
| `coinbase_client.py` | `EnhancedCoinbaseClient`, `CoinbaseCredentials`, `create_client_from_config` | `renaissance_trading_bot`, `market_data_provider`, `verify_keys_live`, `tests/test_order_execution` | 4 |
| `coinbase_advanced_client.py` | `CoinbaseAdvancedClient` | `renaissance_trading_bot` | 1 |
| `derivatives_data_provider.py` | `DerivativesDataProvider` | `renaissance_trading_bot`, `scripts/fetch_derivatives_history` | 2 |
| `alternative_data_engine.py` | `AlternativeDataEngine`, `AlternativeSignal` | `renaissance_trading_bot` | 1 |
| `macro_data_fetcher.py` | `MacroDataFetcher` | *(none)* | 0 |
| `fear_greed_client.py` | `FearGreedClient` | *(none)* | 0 |
| `twitter_client.py` | `TwitterClient` | *(none)* | 0 |
| `reddit_client.py` | `RedditClient` | *(none)* | 0 |
| `glassnode_client.py` | `GlassnodeClient` | *(none)* | 0 |
| `whale_activity_monitor.py` | `WhaleActivityMonitor` | `renaissance_trading_bot`, `verify_keys_live` | 2 |
| `historical_data_cache.py` | `HistoricalDataCache` | `renaissance_trading_bot`, `tests/test_historical_data_cache` | 2 |
| `order_logic.py` | *(order helpers)* | *(none)* | 0 |

**Summary:** 14 files. 7 actively imported, 7 orphaned (0 imports).
**Conflicts:** `data/` directory already contains `trading.db`. Use `data_providers/` or `market_data/` instead, or nest under existing `data_module/`.

---

### Group B — `strategies/` (Trading Strategies)

Modules implementing specific trading strategies (breakout, spray, straddle, etc.).

| File | Exports | Imported By | Import Count |
|------|---------|-------------|:------------:|
| `breakout_scanner.py` | `BreakoutScanner`, `BreakoutSignal` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `breakout_strategy.py` | `BreakoutStrategy` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `token_spray_engine.py` | `TokenSprayEngine`, `SprayToken` | `bot/builder`, `tests/test_token_spray_engine` | 2 |
| `straddle_engine.py` | `StraddleEngine`, `StraddleFleetController` | `bot/builder` (x2), `tests/test_straddle_engine` | 3 |
| `spread_optimizer.py` | `SpreadOptimizer` | *(none)* | 0 |
| `market_making_engine.py` | `MarketMakingEngine` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `basis_trading_engine.py` | `BasisTradingEngine` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `meta_strategy_selector.py` | `MetaStrategySelector` | `renaissance_trading_bot`, `bot/builder` | 2 |

**Summary:** 8 files. 7 actively imported, 1 orphaned.

---

### Group C — `ml/` (Machine Learning)

ML model loading, training, inference, and signal fusion.

| File | Exports | Imported By | Import Count |
|------|---------|-------------|:------------:|
| `ml_model_loader.py` | `build_feature_sequence`, `build_full_feature_matrix`, `INPUT_DIM`, `TrainedCNN`, `TrainedGRU`, etc. | `risk_gateway`, `train_vae`, 11x `scripts/training/*`, `scripts/backtest_ml_pipeline`, `scripts/test_*`, `tests/test_ml_model_loader` | **17** |
| `ml_integration_bridge.py` | `MLIntegrationBridge`, `MLSignalPackage` | `renaissance_trading_bot`, `ml_enhanced_signal_fusion` | 2 |
| `ml_config.py` | *(ML config constants)* | *(none)* | 0 |
| `ml_enhanced_signal_fusion.py` | *(enhanced signal fusion)* | *(none)* | 0 |
| `unified_ml_models.py` | *(unified model defs)* | *(none)* | 0 |
| `crash_model_loader.py` | *(crash model loader)* | *(none)* | 0 |
| `crash_feature_builder.py` | *(crash feature builder)* | *(none)* | 0 |
| `vae_anomaly_detector.py` | `VariationalAutoEncoder` | `risk_gateway`, `train_vae` | 2 |
| `train_vae.py` | *(training script)* | *(none — entry point)* | 0 |

**Summary:** 9 files. 3 actively imported (ml_model_loader is heavily used), 6 orphaned/scripts.
**Note:** `ml_model_loader.py` has 17 importers — highest import count of any module. Migration requires updating `scripts/training/` extensively.

---

### Group D — `analysis/` (Signal & Analysis Engines)

Technical analysis, signal generation, and market microstructure engines.

| File | Exports | Imported By | Import Count |
|------|---------|-------------|:------------:|
| `enhanced_technical_indicators.py` | `EnhancedTechnicalIndicators`, `PriceData`, `IndicatorOutput` | `renaissance_trading_bot`, `bot/builder`, `bot/cycle_ops`, `bot/data_collection` (x2), `market_data_provider`, `regime_overlay`, `renaissance_signal_fusion`, `replay_backtest`, `stress_test_scenario`, `backtesting/engine`, `tests/test_bot_signals` | **18** |
| `microstructure_engine.py` | `MicrostructureEngine`, `OrderBookSnapshot`, `OrderBookLevel`, `MicrostructureMetrics`, `TradeData` | `renaissance_trading_bot`, `bot/builder`, `bot/data_collection`, `bot/helpers`, `bot/signals`, `market_data_provider`, `renaissance_signal_fusion` | **7** |
| `confluence_engine.py` | `ConfluenceEngine` | `renaissance_trading_bot`, `bot/builder`, `dashboard/routers/brain` | 3 |
| `correlation_network_engine.py` | `CorrelationNetworkEngine` | `renaissance_trading_bot`, `bot/builder`, `tests/test_correlation_network` | 3 |
| `garch_volatility_engine.py` | `GARCHVolatilityEngine` | `renaissance_trading_bot`, `bot/builder`, `tests/test_garch_volatility` | 3 |
| `advanced_mean_reversion_engine.py` | `AdvancedMeanReversionEngine`, `PairState` | `renaissance_trading_bot`, `bot/builder`, `arbitrage/pairs/pairs_arb`, `tests/test_advanced_mean_reversion` | 4 |
| `medallion_signal_analogs.py` | `MedallionSignalAnalogs` | `renaissance_trading_bot`, `bot/builder`, `tests/test_medallion_integration` | 3 |
| `fractal_intelligence.py` | `FractalIntelligenceEngine` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `market_entropy_engine.py` | `MarketEntropyEngine` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `quantum_oscillator_engine.py` | `QuantumOscillatorEngine` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `volume_profile_engine.py` | `VolumeProfileEngine` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `cross_asset_engine.py` | `CrossAssetCorrelationEngine` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `deep_nlp_bridge.py` | `DeepNLPBridge` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `statistical_arbitrage_engine.py` | `StatisticalArbitrageEngine` | `bot/builder` | 1 |

**Summary:** 14 files. All actively imported.
**Warning:** `enhanced_technical_indicators.py` (18 imports) and `microstructure_engine.py` (7 imports) are heavily coupled — migrate these last.

---

### Group E — `risk/` (Risk Management)

Position sizing, risk gating, tail risk, slippage, and stress testing.

| File | Exports | Imported By | Import Count |
|------|---------|-------------|:------------:|
| `position_manager.py` | `EnhancedPositionManager`, `RiskLimits`, `Position`, `PositionSide`, `PositionStatus` | `renaissance_trading_bot`, `bot/builder`, `bot/position_ops` (x2), `bot/decision`, `bot/lifecycle`, `tests/test_position_manager`, `tests/test_idempotent_orders`, `tests/test_reconciliation`, `tests/test_bot_decision` | **10** |
| `risk_gateway.py` | `RiskGateway` | `renaissance_trading_bot`, `bot/builder`, `test_adapters`, `tests/test_kill_switch` | 4 |
| `position_sizer.py` | `RenaissancePositionSizer`, `SizingResult` | `renaissance_trading_bot`, `bot/builder`, `tests/test_position_sizer` | 3 |
| `slippage_protection_system.py` | `SlippageProtectionSystem` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `liquidity_risk_manager.py` | `LiquidityRiskManager` | `execution_algorithm_suite` | 1 |
| `tail_risk_protector.py` | *(tail risk)* | *(none)* | 0 |
| `stress_test_engine.py` | *(stress test)* | *(none)* | 0 |
| `transaction_cost_minimizer.py` | *(cost minimizer)* | *(none)* | 0 |

**Summary:** 8 files. 5 actively imported, 3 orphaned.
**Warning:** `position_manager.py` (10 imports) is heavily coupled — migrate carefully.

---

### Group F — `regime/` (Regime Detection)

Market regime classification (HMM, macro, crypto-specific).

| File | Exports | Imported By | Import Count |
|------|---------|-------------|:------------:|
| `regime_overlay.py` | `RegimeOverlay` | `renaissance_trading_bot`, `bot/builder`, `test_adapters` | 3 |
| `advanced_regime_detector.py` | `AdvancedRegimeDetector` | `regime_overlay`, `tests/test_advanced_regime_detector` | 2 |
| `crypto_regime_detector.py` | `CryptoRegimeDetector`, `CryptoRegime` | `renaissance_trading_bot`, `bot/builder`, `model_router` | 3 |
| `macro_regime_detector.py` | `MacroRegimeDetector`, `MacroRegime` | `renaissance_trading_bot`, `bot/builder`, `model_router` | 3 |
| `medallion_regime_predictor.py` | `MedallionRegimePredictor` | `regime_overlay` | 1 |

**Summary:** 5 files. All actively imported.
**Note:** Existing `intelligence/` package already has regime-related modules. Consider merging into `intelligence/regime/`.

---

### Group G — `infra/` (Infrastructure & Core)

Logging, database, config, types — foundational modules used everywhere.

| File | Exports | Imported By | Import Count |
|------|---------|-------------|:------------:|
| `database_manager.py` | `DatabaseManager`, `MarketData` | `renaissance_trading_bot` (x2), `bot/builder`, `stress_test_scenario`, 4x `tests/*` | **8** |
| `renaissance_types.py` | `TradingDecision`, `SignalType`, `OrderType`, `MLSignalPackage` | `renaissance_trading_bot`, `renaissance_engine_core`, `ml_integration_bridge`, `ml_enhanced_signal_fusion`, `bot/position_ops`, `bot/decision`, `tests/test_bot_decision` | **7** |
| `logger.py` | `RenaissanceAuditLogger`, `SecretMaskingFilter` | `renaissance_trading_bot`, `bot/helpers`, `production_trading_orchestrator`, `tests/test_logging` | 4 |
| `enhanced_config_manager.py` | `EnhancedConfigManager` | `renaissance_trading_bot`, `bot/builder`, `renaissance_signal_fusion` | 3 |
| `data_validator.py` | `DataValidator` | `renaissance_trading_bot`, `bot/builder`, `tests/test_medallion_integration` | 3 |

**Summary:** 5 files. All actively imported. These are foundational — migrate last (or not at all).
**Recommendation:** Keep at root or move to `core/` (which already exists).

---

### Group H — `execution/` (Execution & Trading Core)

Order execution, portfolio engine, signal fusion — the trading brain.

| File | Exports | Imported By | Import Count |
|------|---------|-------------|:------------:|
| `renaissance_trading_bot.py` | `RenaissanceTradingBot` | 30+ files (bot/*, tests/*, scripts, etc.) | **30+** |
| `renaissance_engine_core.py` | `SignalFusion`, `RiskManager` | `renaissance_trading_bot`, `bot/builder`, `risk_gateway` | 4 |
| `renaissance_signal_fusion.py` | `RenaissanceSignalFusion` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `unified_portfolio_engine.py` | `UnifiedPortfolioEngine` | `renaissance_trading_bot`, `bot/builder`, `tests/test_medallion_integration` | 3 |
| `execution_algorithm_suite.py` | `ExecutionAlgorithmSuite` | `renaissance_trading_bot`, `bot/builder`, `slippage_protection_system` | 3 |
| `production_trading_orchestrator.py` | `ProductionTradingOrchestrator`, `ProductionConfig` | `renaissance_trading_bot`, `safe_startup` | 2 |
| `signal_validation_gate.py` | `SignalValidationGate` | `renaissance_trading_bot`, `bot/builder`, `tests/test_medallion_integration` | 3 |
| `signal_auto_throttle.py` | `SignalAutoThrottle` | `renaissance_trading_bot`, `bot/builder`, `tests/test_medallion_integration` (x2) | 4 |

**Summary:** 8 files. All actively imported.
**Warning:** `renaissance_trading_bot.py` is imported by 30+ files — DO NOT MOVE. Keep at root.

---

### Group I — `monitoring/` (Monitoring & Dashboards)

Alerting, audit logging, health monitoring, performance tracking.

| File | Exports | Imported By | Import Count |
|------|---------|-------------|:------------:|
| `alert_manager.py` | `AlertManager` | `renaissance_trading_bot`, `bot/builder`, `tests/test_alerts` | 3 |
| `decision_audit_logger.py` | `DecisionAuditLogger` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `portfolio_health_monitor.py` | `PortfolioHealthMonitor` | `renaissance_trading_bot`, `bot/builder`, `tests/test_medallion_integration` | 3 |
| `institutional_dashboard.py` | `InstitutionalDashboard` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `performance_attribution_engine.py` | `PerformanceAttributionEngine` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `trade_logger.py` | *(trade logging)* | *(none)* | 0 |

**Summary:** 6 files. 5 actively imported, 1 orphaned.
**Note:** Existing `monitoring/` package already exists. Merge into it.

---

### Group J — `simulation/` (Simulation Suite)

All `sim_*` modules — self-contained simulation/backtesting system.

| File | Exports | Imported By | Import Count |
|------|---------|-------------|:------------:|
| `sim_runner.py` | *(entry point)* | *(none — top-level runner)* | 0 |
| `sim_config.py` | `SimConfig` | `sim_validation`, `sim_transaction_costs`, `sim_reporting`, tests | 5+ |
| `sim_models_base.py` | `BaseModel` | `sim_portfolio`, `sim_model_gbm`, `sim_bayesian_uncertainty`, `sim_model_ngram`, `sim_model_hmm_regime` | 5 |
| `sim_data_ingest.py` | *(data loading)* | `sim_runner`, tests | 2 |
| `sim_model_gbm.py` | *(GBM model)* | `sim_runner`, tests | 2+ |
| `sim_model_heston.py` | *(Heston model)* | `sim_runner`, tests | 2 |
| `sim_model_hmm_regime.py` | *(HMM model)* | `sim_runner`, tests | 2 |
| `sim_model_monte_carlo.py` | *(MC model)* | `sim_runner`, tests | 2 |
| `sim_model_ngram.py` | *(N-gram model)* | `sim_runner`, tests | 2 |
| `sim_portfolio.py` | *(portfolio sim)* | `sim_runner`, tests | 2 |
| `sim_reporting.py` | *(reporting)* | `sim_runner`, tests | 2 |
| `sim_statistics.py` | *(statistics)* | `sim_runner`, tests | 2 |
| `sim_strategies.py` | *(strategies)* | `sim_runner`, tests | 2 |
| `sim_stress_test.py` | *(stress tests)* | `sim_runner`, tests | 2 |
| `sim_transaction_costs.py` | *(cost model)* | `sim_runner`, `sim_strategies`, tests | 3 |
| `sim_validation.py` | *(validation)* | `sim_runner`, tests | 2 |
| `sim_bayesian_uncertainty.py` | *(Bayesian model)* | `sim_runner`, tests | 2 |

**Summary:** 17 files. All inter-dependent. Self-contained subsystem — ideal candidate for bulk move.
**Migration:** These only import each other and are imported by tests. Move all at once to `simulation/`.

---

### Group K — `polymarket/` (Polymarket Integration)

Polymarket prediction market integration modules.

| File | Exports | Imported By | Import Count |
|------|---------|-------------|:------------:|
| `polymarket_bridge.py` | *(bridge)* | `renaissance_trading_bot`, `bot/builder` | 2 |
| `polymarket_rtds.py` | *(RTDS)* | `renaissance_trading_bot`, `bot/builder` | 2 |
| `polymarket_scanner.py` | *(scanner)* | `renaissance_trading_bot`, `bot/builder` | 2 |
| `polymarket_spread_capture.py` | *(spread capture)* | `renaissance_trading_bot`, `bot/builder`, `bot/lifecycle`, tests | 4 |
| `polymarket_timing_features.py` | *(timing)* | `archive/polymarket/*`, `dashboard/routers/polymarket` | 3 |

**Summary:** 5 files. All actively imported.
**Note:** `archive/polymarket/` already exists. Consider whether these should join it or stay separate.

---

### Group L — Miscellaneous (Various)

Modules that don't fit neatly into the above groups.

| File | Exports | Imported By | Import Count |
|------|---------|-------------|:------------:|
| `model_router.py` | `ModelRouter` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `self_reinforcing_learning.py` | `SelfReinforcingLearningEngine` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `cascade_data_collector.py` | `CascadeDataCollector` | `renaissance_trading_bot` | 1 |
| `real_time_pipeline.py` | `RealTimePipeline`, `MultiExchangeFeed`, `FeatureFanOutProcessor` | `renaissance_trading_bot`, `bot/builder`, `test_real_time_pipeline` | 3 |
| `sub_bar_scanner.py` | `SubBarScanner` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `genetic_optimizer.py` | `GeneticWeightOptimizer` | `renaissance_trading_bot`, `bot/builder`, `tests/test_sql_injection` | 3 |
| `ghost_runner.py` | `GhostRunner` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `random_baseline.py` | `RandomEntryBaseline` | `renaissance_trading_bot`, `bot/builder` | 2 |
| `market_microstructure_analyzer.py` | `MarketMicrostructureAnalyzer` | `slippage_protection_system` | 1 |
| `stress_test_scenario.py` | *(scenario script)* | *(none)* | 0 |

---

### Entry Points & Scripts (DO NOT MOVE)

These are standalone entry points / scripts, not library modules.

| File | Purpose | Import Count |
|------|---------|:------------:|
| `renaissance_trading_bot.py` | Main bot class (30+ importers) | **30+** |
| `run_renaissance_bot.py` | Bot entry point | 0 |
| `run_arbitrage.py` | Arbitrage entry point | 0 |
| `safe_startup.py` | Safe startup wrapper | 0 |
| `check_readiness.py` | Pre-flight checks | 0 |
| `verify_keys_live.py` | API key verification | 0 |
| `backfill_bars.py` | Historical data backfill | 0 |
| `replay_backtest.py` | Backtest replay | 0 |
| `test_adapters.py` | Test helpers | 0 |
| `test_real_time_pipeline.py` | Pipeline tests | 0 |
| `test_step12_deepening.py` | Step 12 tests | 0 |

---

## Migration Priority Order

Based on import coupling (lower coupling = easier to move):

### Phase 1 — Easy wins (0-2 importers, self-contained)
1. **`simulation/`** — 17 `sim_*` files. Only import each other. Bulk move.
2. **Orphaned files** — 17 files with 0 imports. Can be moved freely or archived.

### Phase 2 — Low coupling (2-3 importers, mostly `bot/builder` + `renaissance_trading_bot`)
3. **`strategies/`** — 8 files. Most imported only by builder + bot.
4. **`regime/`** — 5 files. Clean dependency chain.
5. **`monitoring/` merge** — 6 files into existing `monitoring/` package.
6. **`polymarket/`** — 5 files. Cleanly separable.

### Phase 3 — Medium coupling (3-5 importers)
7. **`data_providers/`** — 14 files. Some with 0 imports, some with cross-deps.
8. **`risk/`** — 8 files. `position_manager` has 10 importers (careful).
9. **`ml/`** — 9 files. `ml_model_loader` has 17 importers (scripts/training/*).

### Phase 4 — High coupling (7+ importers, used everywhere)
10. **`analysis/`** — 14 files. `enhanced_technical_indicators` has 18 importers.

### Phase 5 — Do not move (too deeply coupled)
11. **`renaissance_trading_bot.py`** — 30+ importers. Keep at root.
12. **Infrastructure** (`database_manager`, `renaissance_types`, `logger`) — Keep at root or `core/`.

---

## Orphaned Files (0 imports across codebase)

These files are defined but never imported. Candidates for archival or deletion:

| File | Group | Notes |
|------|-------|-------|
| `macro_data_fetcher.py` | Data | Unused data fetcher |
| `fear_greed_client.py` | Data | Unused API client |
| `twitter_client.py` | Data | Unused API client |
| `reddit_client.py` | Data | Unused API client |
| `glassnode_client.py` | Data | Unused API client |
| `order_logic.py` | Data | Unused order helpers |
| `spread_optimizer.py` | Strategy | Unused strategy |
| `ml_config.py` | ML | Unused config |
| `ml_enhanced_signal_fusion.py` | ML | Unused (imports MLSignalPackage but nothing imports it) |
| `unified_ml_models.py` | ML | Unused model definitions |
| `crash_model_loader.py` | ML | Unused loader |
| `crash_feature_builder.py` | ML | Unused feature builder |
| `tail_risk_protector.py` | Risk | Unused risk module |
| `stress_test_engine.py` | Risk | Unused stress tester |
| `transaction_cost_minimizer.py` | Risk | Unused cost module |
| `trade_logger.py` | Monitoring | Unused logger |
| `stress_test_scenario.py` | Misc | Standalone script |

**Total: 17 orphaned files** — review whether these are planned-but-unwired features or dead code.

---

## Safe Migration Procedure (Per File)

For each file being moved to a package:

```bash
# 1. Find all imports
grep -rn "from module_name import\|import module_name" --include="*.py" .

# 2. Create package if needed
mkdir -p package_name
touch package_name/__init__.py

# 3. Move the file
mv module_name.py package_name/

# 4. Add re-export stub at root (backwards compat)
cat > module_name.py << 'EOF'
# Backwards compatibility stub — module moved to package_name/
# TODO: Update imports to use `from package_name.module_name import ...`
from package_name.module_name import *  # noqa: F401,F403
EOF

# 5. Update package __init__.py with re-exports
echo "from .module_name import ClassName" >> package_name/__init__.py

# 6. Update all importers to use new path (one by one)
# 7. Once all importers updated, delete root stub
```

---

## Files by Import Count (Highest Risk)

| Rank | File | Importers | Group |
|------|------|:---------:|-------|
| 1 | `renaissance_trading_bot.py` | 30+ | **DO NOT MOVE** |
| 2 | `enhanced_technical_indicators.py` | 18 | Analysis |
| 3 | `ml_model_loader.py` | 17 | ML |
| 4 | `position_manager.py` | 10 | Risk |
| 5 | `database_manager.py` | 8 | Infra |
| 6 | `renaissance_types.py` | 7 | Infra |
| 7 | `microstructure_engine.py` | 7 | Analysis |
| 8 | `sim_config.py` | 5+ | Simulation |
| 9 | `sim_models_base.py` | 5 | Simulation |
| 10 | `coinbase_client.py` | 4 | Data |

---

## Test Files Affected

Tests that import root modules directly and will need updating:

| Test File | Imports From |
|-----------|-------------|
| `tests/test_position_manager.py` | `position_manager` |
| `tests/test_idempotent_orders.py` | `position_manager` |
| `tests/test_reconciliation.py` | `position_manager` |
| `tests/test_bot_decision.py` | `position_manager`, `renaissance_types` |
| `tests/test_position_sizer.py` | `position_sizer` |
| `tests/test_kill_switch.py` | `risk_gateway` |
| `tests/test_order_execution.py` | `coinbase_client` |
| `tests/test_historical_data_cache.py` | `historical_data_cache` |
| `tests/test_ml_model_loader.py` | `ml_model_loader` |
| `tests/test_correlation_network.py` | `correlation_network_engine` |
| `tests/test_garch_volatility.py` | `garch_volatility_engine` |
| `tests/test_advanced_mean_reversion.py` | `advanced_mean_reversion_engine` |
| `tests/test_medallion_integration.py` | Multiple root modules |
| `tests/test_token_spray_engine.py` | `token_spray_engine` |
| `tests/test_straddle_engine.py` | `straddle_engine` |
| `tests/test_bot_signals.py` | `enhanced_technical_indicators` |
| `tests/test_alerts.py` | `alert_manager` |
| `tests/test_logging.py` | `logger` |
| `tests/test_sql_injection.py` | `database_manager`, `genetic_optimizer` |
| `tests/test_database_safety.py` | `database_manager` |
| `tests/test_state_recovery.py` | `database_manager` |
| `tests/test_advanced_regime_detector.py` | `advanced_regime_detector` |

---

## Summary Statistics

| Metric | Count |
|--------|------:|
| Total root `.py` files | 119 |
| Proposed packages | 12 (A-L) |
| Files to move | ~100 |
| Files to keep at root | ~19 (entry points + `renaissance_trading_bot.py`) |
| Orphaned files (0 imports) | 17 |
| Highest-risk files (10+ importers) | 4 |
| Test files needing updates | 22 |
| Existing packages to merge into | 2 (`monitoring/`, `core/`) |
