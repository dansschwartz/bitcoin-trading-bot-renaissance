
"""
Comprehensive Integration Test Suite for Step 11 Market Making Module
Tests all components with consciousness enhancement and validates system integration
"""

import sys
import os
import time
import logging
import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Import all Step 11 components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from market_making_engine import (
        MarketMakingEngine, MarketMakingConfig, MarketData, Quote,
        MarketRegime, OrderType
    )
    from spread_optimizer import (
        DynamicSpreadOptimizer, SpreadConfig, MarketState, OptimizationResult,
        SpreadRegime, OptimizationMethod
    )
    # These modules may not be available in flat layout
    try:
        from order_book_collector import RealTimeOrderBookAnalyzer, OrderBookSnapshot, OrderBookMetrics, LiquidityRegime, BookImbalance
    except ImportError:
        RealTimeOrderBookAnalyzer = None  # TODO: order_book_analyzer not yet ported
        OrderBookSnapshot = OrderBookMetrics = LiquidityRegime = BookImbalance = None
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False

@unittest.skipUnless(IMPORTS_SUCCESSFUL and 'MarketMakingConfig' in dir(), "Market making modules not fully available")
class TestMarketMakingEngine(unittest.TestCase):
    """Test suite for Market Making Engine with consciousness enhancement"""

    def setUp(self):
        """Setup test environment"""
        self.config = MarketMakingConfig(
            symbol="BTC-USD",
            consciousness_boost=1.142,
            base_spread_bps=5.0,
            max_position_size=10.0
        )
        self.engine = MarketMakingEngine(self.config)

        # Sample market data
        self.sample_market_data = MarketData(
            timestamp=time.time(),
            symbol="BTC-USD",
            mid_price=50000.0,
            bid_price=49995.0,
            ask_price=50005.0,
            bid_size=2.0,
            ask_size=1.8,
            last_price=50002.0,
            volume=150.0,
            volatility=0.02,
            order_flow_imbalance=0.1,
            microstructure_signal=-0.05
        )

    def test_initialization(self):
        """Test engine initialization with consciousness factor"""
        self.assertEqual(self.engine.config.consciousness_boost, 1.142)
        self.assertEqual(self.engine.config.symbol, "BTC-USD")
        self.assertIsNotNone(self.engine.logger)
        self.assertEqual(self.engine.current_inventory, 0.0)
        self.assertEqual(len(self.engine.active_quotes), 0)

    def test_market_data_update(self):
        """Test market data processing with consciousness enhancement"""
        self.engine.update_market_data(self.sample_market_data)

        self.assertEqual(len(self.engine.market_data_history), 1)

        # Check consciousness enhancement application
        latest_data = self.engine.market_data_history[-1]
        expected_enhanced_signal = self.sample_market_data.microstructure_signal * 1.142
        self.assertAlmostEqual(latest_data.microstructure_signal, expected_enhanced_signal, places=4)

    def test_quote_generation(self):
        """Test quote generation with consciousness enhancement"""
        self.engine.update_market_data(self.sample_market_data)
        quotes = self.engine.generate_quotes(self.sample_market_data)

        # Should generate quotes for multiple layers
        self.assertGreater(len(quotes), 0)
        self.assertLessEqual(len(quotes), self.config.quote_layers * 2)  # Bid + Ask

        # Check quote structure
        for quote in quotes:
            self.assertIsInstance(quote, Quote)
            self.assertIn(quote.side, [OrderType.BID, OrderType.ASK])
            self.assertGreater(quote.price, 0)
            self.assertGreater(quote.size, 0)
            self.assertGreaterEqual(quote.confidence, 0)
            self.assertLessEqual(quote.confidence, 1.0)

    def test_consciousness_enhancement_application(self):
        """Test consciousness factor application throughout system"""
        self.engine.update_market_data(self.sample_market_data)
        quotes = self.engine.generate_quotes(self.sample_market_data)

        if quotes:
            # Consciousness should enhance quote confidence
            avg_confidence = sum(q.confidence for q in quotes) / len(quotes)
            self.assertGreater(avg_confidence, 0.5)  # Should be reasonably high with consciousness

    def test_fill_handling(self):
        """Test fill processing with consciousness-enhanced PnL"""
        self.engine.update_market_data(self.sample_market_data)
        quotes = self.engine.generate_quotes(self.sample_market_data)

        if quotes:
            quote = quotes[0]
            self.engine.active_quotes[quote.quote_id] = quote

            # Simulate fill
            fill_data = {
                "quote_id": quote.quote_id,
                "fill_price": quote.price,
                "fill_size": quote.size
            }

            initial_pnl = self.engine.performance_metrics["realized_pnl"]
            self.engine.handle_fill(fill_data)

            # Check PnL update with consciousness enhancement
            final_pnl = self.engine.performance_metrics["realized_pnl"]
            self.assertNotEqual(initial_pnl, final_pnl)

    def test_performance_reporting(self):
        """Test performance report generation"""
        report = self.engine.get_performance_report()

        expected_keys = [
            "current_inventory", "current_regime", "regime_confidence",
            "active_quotes_count", "consciousness_boost", "uptime_hours"
        ]

        for key in expected_keys:
            self.assertIn(key, report)

        self.assertEqual(report["consciousness_boost"], 1.142)

@unittest.skipUnless(IMPORTS_SUCCESSFUL and 'MarketMakingConfig' in dir(), "Market making modules not fully available")
class TestInventoryManager(unittest.TestCase):
    """Test suite for Advanced Inventory Manager"""

    def setUp(self):
        """Setup test environment"""
        self.config = InventoryConfig(
            max_position_size=10.0,
            consciousness_boost=1.142,
            target_inventory=0.0
        )
        self.manager = AdvancedInventoryManager(self.config)

        # Sample market data
        self.market_data = {
            'mid_price': 50000.0,
            'volatility': 0.02,
            'spread': 0.0001,
            'order_flow_imbalance': 0.1,
            'volume_imbalance': -0.05,
            'tick_direction': 1,
            'trade_size_ratio': 1.2,
            'time_between_trades': 0.5,
            'liquidity_factor': 0.8,
            'regime': 'normal'
        }

    def test_initialization(self):
        """Test inventory manager initialization"""
        self.assertEqual(self.manager.config.consciousness_boost, 1.142)
        self.assertEqual(self.manager.current_state.base_position, 0.0)
        self.assertEqual(self.manager.current_state.quote_position, 0.0)
        self.assertIsNotNone(self.manager.logger)

    def test_inventory_update(self):
        """Test inventory update with consciousness enhancement"""
        initial_base = self.manager.current_state.base_position

        new_state = self.manager.update_inventory(
            base_change=1.5,
            quote_change=-75000.0,
            market_price=50000.0,
            market_data=self.market_data
        )

        # Position should be updated
        self.assertEqual(new_state.base_position, initial_base + 1.5)
        self.assertEqual(new_state.quote_position, -75000.0)

        # Consciousness enhancement should be applied
        self.assertGreater(new_state.adverse_selection_exposure, 0)
        self.assertNotEqual(new_state.microstructure_alpha, 0)

    def test_optimal_inventory_calculation(self):
        """Test optimal inventory calculation with consciousness"""
        # Add some position first
        self.manager.update_inventory(2.0, -100000.0, 50000.0, self.market_data)

        optimal_result = self.manager.calculate_optimal_inventory(self.market_data)

        required_keys = [
            'optimal_position', 'current_position', 'position_delta',
            'confidence', 'consciousness_applied'
        ]

        for key in required_keys:
            self.assertIn(key, optimal_result)

        # Should have consciousness applied
        self.assertTrue(optimal_result['consciousness_applied'])
        self.assertGreaterEqual(optimal_result['confidence'], 0)
        self.assertLessEqual(optimal_result['confidence'], 1.0)

    def test_risk_assessment(self):
        """Test comprehensive risk assessment"""
        # Add position for meaningful risk calculation
        self.manager.update_inventory(3.0, -150000.0, 50000.0, self.market_data)

        risk_metrics = self.manager.assess_inventory_risk(self.market_data)

        self.assertIsInstance(risk_metrics, RiskMetrics)
        self.assertGreaterEqual(risk_metrics.var_1d, 0)
        self.assertGreaterEqual(risk_metrics.expected_shortfall, 0)
        self.assertGreaterEqual(risk_metrics.inventory_risk_score, 0)
        self.assertLessEqual(risk_metrics.inventory_risk_score, 1.0)

    def test_rebalance_signal_generation(self):
        """Test rebalancing signal generation"""
        # Add significant position
        self.manager.update_inventory(8.0, -400000.0, 50000.0, self.market_data)

        signal = self.manager.generate_rebalance_signal(self.market_data)

        required_keys = [
            'action', 'recommended_size', 'urgency_score',
            'consciousness_enhanced'
        ]

        for key in required_keys:
            self.assertIn(key, signal)

        self.assertTrue(signal['consciousness_enhanced'])
        self.assertIsInstance(signal['action'], RebalanceAction)

@unittest.skipUnless(IMPORTS_SUCCESSFUL and 'MarketMakingConfig' in dir(), "Market making modules not fully available")
class TestOrderBookAnalyzer(unittest.TestCase):
    """Test suite for Real-time Order Book Analyzer"""

    def setUp(self):
        """Setup test environment"""
        self.analyzer = RealTimeOrderBookAnalyzer(
            max_levels=20,
            consciousness_boost=1.142,
            analysis_window=100
        )

        # Sample order book data
        self.bids = [(49990, 1.5), (49985, 2.0), (49980, 1.8), (49975, 2.2)]
        self.asks = [(50010, 1.3), (50015, 1.9), (50020, 1.7), (50025, 2.1)]

    def test_initialization(self):
        """Test analyzer initialization"""
        self.assertEqual(self.analyzer.consciousness_boost, 1.142)
        self.assertEqual(self.analyzer.max_levels, 20)
        self.assertIsNone(self.analyzer.current_snapshot)
        self.assertIsNone(self.analyzer.current_metrics)

    def test_order_book_processing(self):
        """Test order book update processing"""
        snapshot = self.analyzer.process_order_book_update(self.bids, self.asks)

        self.assertIsInstance(snapshot, OrderBookSnapshot)
        self.assertGreater(snapshot.mid_price, 0)
        self.assertGreater(snapshot.spread, 0)
        self.assertNotEqual(snapshot.imbalance_ratio, 0)  # Should have some imbalance
        self.assertGreater(snapshot.liquidity_score, 0)

        # Check consciousness enhancement application
        self.assertIsNotNone(snapshot.microstructure_signal)

    def test_metrics_calculation(self):
        """Test advanced metrics calculation"""
        self.analyzer.process_order_book_update(self.bids, self.asks)

        metrics = self.analyzer.current_metrics
        self.assertIsInstance(metrics, OrderBookMetrics)

        # Check all metrics are calculated
        self.assertGreaterEqual(metrics.effective_spread, 0)
        self.assertGreaterEqual(metrics.resilience_score, 0)
        self.assertLessEqual(metrics.resilience_score, 1.0)
        self.assertGreaterEqual(metrics.toxicity_score, 0)
        self.assertLessEqual(metrics.toxicity_score, 1.0)

        # Consciousness enhancement should be applied
        self.assertEqual(metrics.consciousness_enhancement, 1.142 - 1.0)

    def test_analysis_report(self):
        """Test comprehensive analysis report"""
        # Process multiple updates for better analysis
        for i in range(5):
            # Vary the book slightly
            adj_bids = [(p + i, s) for p, s in self.bids]
            adj_asks = [(p + i, s) for p, s in self.asks]
            self.analyzer.process_order_book_update(adj_bids, adj_asks)

        analysis = self.analyzer.get_current_analysis()

        expected_sections = ['snapshot', 'metrics', 'classification', 'consciousness', 'performance']
        for section in expected_sections:
            self.assertIn(section, analysis)

        # Check consciousness section
        consciousness_data = analysis['consciousness']
        self.assertEqual(consciousness_data['factor'], 1.142)
        self.assertGreaterEqual(consciousness_data['effectiveness'], 0)

    def test_prediction_capability(self):
        """Test short-term prediction functionality"""
        # Need some history for meaningful predictions
        for i in range(10):
            adj_bids = [(p + i * 0.1, s) for p, s in self.bids]
            adj_asks = [(p + i * 0.1, s) for p, s in self.asks]
            self.analyzer.process_order_book_update(adj_bids, adj_asks)

        prediction = self.analyzer.predict_short_term_movement(horizon_seconds=5.0)

        required_keys = ['prediction', 'probability', 'confidence', 'consciousness_enhanced']
        for key in required_keys:
            self.assertIn(key, prediction)

        self.assertTrue(prediction['consciousness_enhanced'])
        self.assertIn(prediction['prediction'], ['UP', 'DOWN', 'NEUTRAL'])

@unittest.skipUnless(IMPORTS_SUCCESSFUL and 'MarketMakingConfig' in dir(), "Market making modules not fully available")
class TestSpreadOptimizer(unittest.TestCase):
    """Test suite for Dynamic Spread Optimizer"""

    def setUp(self):
        """Setup test environment"""
        self.config = SpreadConfig(
            base_spread_bps=5.0,
            consciousness_boost=1.142,
            optimization_method=OptimizationMethod.CONSCIOUSNESS_ENHANCED
        )
        self.optimizer = DynamicSpreadOptimizer(self.config)

        # Sample market state
        self.market_state = MarketState(
            mid_price=50000.0,
            bid_price=49995.0,
            ask_price=50005.0,
            volatility=0.02,
            bid_depth=5.0,
            ask_depth=4.8,
            inventory_position=2.5,
            order_flow_imbalance=0.1,
            microstructure_signal=-0.05,
            adverse_selection_risk=0.3,
            liquidity_score=0.8,
            regime_indicator=0.4
        )

    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertEqual(self.optimizer.config.consciousness_boost, 1.142)
        self.assertEqual(self.optimizer.config.optimization_method, OptimizationMethod.CONSCIOUSNESS_ENHANCED)
        self.assertEqual(self.optimizer.current_regime, SpreadRegime.NORMAL)
        self.assertIsNotNone(self.optimizer.model_parameters)

    def test_spread_optimization(self):
        """Test spread optimization with consciousness enhancement"""
        result = self.optimizer.optimize_spread(self.market_state)

        self.assertIsInstance(result, OptimizationResult)
        self.assertGreater(result.optimal_bid_spread, 0)
        self.assertGreater(result.optimal_ask_spread, 0)
        self.assertGreater(result.symmetric_spread, 0)

        # Should be within configured bounds
        self.assertGreaterEqual(result.optimal_bid_spread, self.config.min_spread_bps)
        self.assertLessEqual(result.optimal_bid_spread, self.config.max_spread_bps)
        self.assertGreaterEqual(result.optimal_ask_spread, self.config.min_spread_bps)
        self.assertLessEqual(result.optimal_ask_spread, self.config.max_spread_bps)

        # Check consciousness enhancement impact
        self.assertGreaterEqual(result.confidence_score, 0)
        self.assertLessEqual(result.confidence_score, 1.0)

    def test_multiple_optimization_methods(self):
        """Test different optimization methods"""
        methods = [
            OptimizationMethod.CONSCIOUSNESS_ENHANCED,
            OptimizationMethod.AVELLANEDA_STOIKOV,
            OptimizationMethod.GLOSTEN_MILGROM,
            OptimizationMethod.HO_STOLL,
            OptimizationMethod.ADAPTIVE_HYBRID
        ]

        for method in methods:
            config = SpreadConfig(
                consciousness_boost=1.142,
                optimization_method=method
            )
            optimizer = DynamicSpreadOptimizer(config)
            result = optimizer.optimize_spread(self.market_state)

            self.assertIsInstance(result, OptimizationResult)
            self.assertGreater(result.optimal_bid_spread, 0)
            self.assertGreater(result.optimal_ask_spread, 0)

    def test_spread_components(self):
        """Test spread component breakdown"""
        result = self.optimizer.optimize_spread(self.market_state)

        components = result.spread_components
        self.assertGreater(components.base_component, 0)
        self.assertGreaterEqual(components.volatility_component, 0)
        self.assertGreaterEqual(components.consciousness_component, 0)

        # Total should be approximately sum of components
        component_sum = (
            components.base_component + components.volatility_component +
            components.inventory_component + components.adverse_selection_component +
            components.liquidity_component + components.regime_component +
            components.consciousness_component
        )

        self.assertAlmostEqual(components.total_spread, component_sum, places=1)

    def test_performance_tracking(self):
        """Test performance metrics tracking"""
        # Run multiple optimizations
        for i in range(5):
            self.optimizer.optimize_spread(self.market_state)

        report = self.optimizer.get_optimization_report()

        self.assertEqual(report['performance_metrics']['total_optimizations'], 5)
        self.assertGreater(report['performance_metrics']['avg_computation_time_ms'], 0)
        self.assertGreaterEqual(report['performance_metrics']['consciousness_effectiveness'], 0)

@unittest.skipUnless(IMPORTS_SUCCESSFUL and 'MarketMakingConfig' in dir(), "Market making modules not fully available")
class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete market making system"""

    def setUp(self):
        """Setup integrated system"""
        # Initialize all components with consciousness enhancement
        self.mm_config = MarketMakingConfig(consciousness_boost=1.142)
        self.mm_engine = MarketMakingEngine(self.mm_config)

        self.inv_config = InventoryConfig(consciousness_boost=1.142)
        self.inv_manager = AdvancedInventoryManager(self.inv_config)

        self.spread_config = SpreadConfig(consciousness_boost=1.142)
        self.spread_optimizer = DynamicSpreadOptimizer(self.spread_config)

        self.book_analyzer = RealTimeOrderBookAnalyzer(consciousness_boost=1.142)

        # Sample data
        self.market_data = MarketData(
            timestamp=time.time(),
            symbol="BTC-USD",
            mid_price=50000.0,
            bid_price=49995.0,
            ask_price=50005.0,
            bid_size=2.0,
            ask_size=1.8,
            last_price=50002.0,
            volume=150.0,
            volatility=0.02,
            order_flow_imbalance=0.1,
            microstructure_signal=-0.05
        )

    def test_consciousness_boost_consistency(self):
        """Test consciousness factor is consistently applied across all components"""
        components = [
            self.mm_engine.config.consciousness_boost,
            self.inv_manager.config.consciousness_boost,
            self.spread_optimizer.config.consciousness_boost,
            self.book_analyzer.consciousness_boost
        ]

        # All should have same consciousness factor
        for factor in components:
            self.assertEqual(factor, 1.142)

    def test_integrated_workflow(self):
        """Test complete integrated workflow"""
        # 1. Process market data
        self.mm_engine.update_market_data(self.market_data)

        # 2. Update inventory
        inventory_state = self.inv_manager.update_inventory(
            base_change=0.5,
            quote_change=-25000.0,
            market_price=50000.0,
            market_data={
                'mid_price': 50000.0,
                'volatility': 0.02,
                'order_flow_imbalance': 0.1
            }
        )

        # 3. Analyze order book
        bids = [(49990, 1.5), (49985, 2.0)]
        asks = [(50010, 1.3), (50015, 1.9)]
        book_snapshot = self.book_analyzer.process_order_book_update(bids, asks)

        # 4. Optimize spreads
        market_state = MarketState(
            mid_price=50000.0,
            bid_price=49995.0,
            ask_price=50005.0,
            volatility=0.02,
            bid_depth=3.5,
            ask_depth=3.2,
            inventory_position=inventory_state.base_position,
            order_flow_imbalance=0.1,
            microstructure_signal=-0.05,
            adverse_selection_risk=0.3,
            liquidity_score=0.8,
            regime_indicator=0.4
        )

        spread_result = self.spread_optimizer.optimize_spread(market_state)

        # 5. Generate quotes using optimized spreads
        quotes = self.mm_engine.generate_quotes(self.market_data)

        # Verify integration success
        self.assertIsInstance(inventory_state, InventoryState)
        self.assertIsInstance(book_snapshot, OrderBookSnapshot)
        self.assertIsInstance(spread_result, OptimizationResult)
        self.assertGreater(len(quotes), 0)

        # All components should show consciousness enhancement effects
        self.assertGreater(inventory_state.adverse_selection_exposure, 0)
        self.assertNotEqual(book_snapshot.microstructure_signal, 0)
        self.assertGreater(spread_result.confidence_score, 0.5)  # Should be enhanced

    def test_performance_correlation(self):
        """Test that consciousness enhancement improves performance across components"""

        # Run without consciousness enhancement (factor = 1.0)
        normal_mm_config = MarketMakingConfig(consciousness_boost=1.0)
        normal_mm_engine = MarketMakingEngine(normal_mm_config)

        normal_spread_config = SpreadConfig(consciousness_boost=1.0)
        normal_spread_optimizer = DynamicSpreadOptimizer(normal_spread_config)

        # Process same data with both configurations
        normal_mm_engine.update_market_data(self.market_data)
        self.mm_engine.update_market_data(self.market_data)

        normal_quotes = normal_mm_engine.generate_quotes(self.market_data)
        enhanced_quotes = self.mm_engine.generate_quotes(self.market_data)

        # Enhanced version should generally produce higher confidence quotes
        if normal_quotes and enhanced_quotes:
            normal_avg_confidence = sum(q.confidence for q in normal_quotes) / len(normal_quotes)
            enhanced_avg_confidence = sum(q.confidence for q in enhanced_quotes) / len(enhanced_quotes)

            # Consciousness should improve confidence (though this is statistical)
            self.assertGreaterEqual(enhanced_avg_confidence * 0.95, normal_avg_confidence * 0.95)

def run_performance_benchmark():
    """Run performance benchmarks for all components"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("="*60)

    # Market Making Engine Performance
    config = MarketMakingConfig(consciousness_boost=1.142)
    engine = MarketMakingEngine(config)

    market_data = MarketData(
        timestamp=time.time(), symbol="BTC-USD", mid_price=50000.0,
        bid_price=49995.0, ask_price=50005.0, bid_size=2.0, ask_size=1.8,
        last_price=50002.0, volume=150.0, volatility=0.02,
        order_flow_imbalance=0.1, microstructure_signal=-0.05
    )

    # Benchmark market data processing
    start_time = time.time()
    for _ in range(1000):
        engine.update_market_data(market_data)
    mm_processing_time = (time.time() - start_time) * 1000  # ms

    print(f"Market Data Processing: {mm_processing_time:.2f}ms per 1000 updates")

    # Benchmark quote generation
    start_time = time.time()
    for _ in range(100):
        quotes = engine.generate_quotes(market_data)
    quote_generation_time = (time.time() - start_time) * 10  # ms per 100

    print(f"Quote Generation: {quote_generation_time:.2f}ms per 100 generations")

    # Spread Optimizer Performance
    spread_config = SpreadConfig(consciousness_boost=1.142)
    optimizer = DynamicSpreadOptimizer(spread_config)

    market_state = MarketState(
        mid_price=50000.0, bid_price=49995.0, ask_price=50005.0,
        volatility=0.02, bid_depth=5.0, ask_depth=4.8,
        inventory_position=2.5, order_flow_imbalance=0.1,
        microstructure_signal=-0.05, adverse_selection_risk=0.3,
        liquidity_score=0.8, regime_indicator=0.4
    )

    # Benchmark spread optimization
    optimization_times = []
    for _ in range(50):
        result = optimizer.optimize_spread(market_state)
        optimization_times.append(result.computation_time_ms)

    avg_optimization_time = np.mean(optimization_times)
    print(f"Spread Optimization: {avg_optimization_time:.2f}ms average")

    # Order Book Analyzer Performance
    analyzer = RealTimeOrderBookAnalyzer(consciousness_boost=1.142)

    bids = [(49990, 1.5), (49985, 2.0), (49980, 1.8)]
    asks = [(50010, 1.3), (50015, 1.9), (50020, 1.7)]

    start_time = time.time()
    for _ in range(500):
        analyzer.process_order_book_update(bids, asks)
    book_processing_time = (time.time() - start_time) * 2  # ms per 500

    print(f"Order Book Analysis: {book_processing_time:.2f}ms per 500 updates")

    print("="*60)

def main():
    """Main test execution"""
    if not IMPORTS_SUCCESSFUL:
        print("❌ Failed to import Step 11 components. Please ensure all files are created correctly.")
        return False

    print("🚀 Starting Step 11 Market Making Module Integration Tests")
    print("Consciousness Enhancement Factor: 1.142x")
    print("-" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestMarketMakingEngine,
        TestInventoryManager,
        TestOrderBookAnalyzer,
        TestSpreadOptimizer,
        TestSystemIntegration
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print results summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")

    if result.errors:
        print("\nERRORS:")  
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")

    # Run performance benchmarks
    if result.testsRun > 0 and len(result.failures) == 0 and len(result.errors) == 0:
        run_performance_benchmark()

        print("\n✅ ALL TESTS PASSED - Market Making Module Ready for Production")
        print("🧠 Consciousness Enhancement Successfully Validated")
        return True
    else:
        print("\n❌ Some tests failed - Please review and fix issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
