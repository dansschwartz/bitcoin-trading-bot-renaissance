"""
Tests for arbitrage/orchestrator.py

Covers:
  - ArbitrageOrchestrator initialization and component wiring
  - Config loading (with and without config file)
  - Component creation (exchange clients, book manager, detectors, etc.)
  - Stop method coordination
  - Status logging
  - Does NOT test actual trading (all exchange calls mocked)
"""
import asyncio
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import yaml


class TestArbitrageOrchestrator:
    @pytest.fixture
    def mock_config_file(self, tmp_path):
        config = {
            'paper_trading': {'enabled': True},
            'pairs': {
                'phase_1': ['BTC/USDT', 'ETH/USDT'],
            },
            'risk': {
                'max_single_arb_usd': 500,
                'max_total_exposure_usd': 5000,
            },
            'logging': {
                'level': 'WARNING',
                'file': str(tmp_path / 'test_arb.log'),
            },
        }
        config_path = tmp_path / "test_arbitrage.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        return str(config_path)

    @patch('arbitrage.orchestrator.MEXCClient')
    @patch('arbitrage.orchestrator.BinanceClient')
    @patch('arbitrage.orchestrator.PerformanceTracker')
    def test_initialization_creates_all_components(
        self, mock_tracker_cls, mock_binance_cls, mock_mexc_cls, mock_config_file
    ):
        mock_mexc_cls.return_value = MagicMock()
        mock_binance_cls.return_value = MagicMock()
        mock_tracker_cls.return_value = MagicMock()

        from arbitrage.orchestrator import ArbitrageOrchestrator
        orch = ArbitrageOrchestrator(config_path=mock_config_file)

        assert orch.mexc is not None
        assert orch.binance is not None
        assert orch.book_manager is not None
        assert orch.cost_model is not None
        assert orch.risk_engine is not None
        assert orch.cross_exchange_detector is not None
        assert orch.executor is not None
        assert orch.funding_arb is not None
        assert orch.triangular_arb is not None
        assert orch.inventory is not None
        assert orch.tracker is not None

    @patch('arbitrage.orchestrator.MEXCClient')
    @patch('arbitrage.orchestrator.BinanceClient')
    @patch('arbitrage.orchestrator.PerformanceTracker')
    def test_config_loading_from_file(
        self, mock_tracker_cls, mock_binance_cls, mock_mexc_cls, mock_config_file
    ):
        mock_mexc_cls.return_value = MagicMock()
        mock_binance_cls.return_value = MagicMock()
        mock_tracker_cls.return_value = MagicMock()

        from arbitrage.orchestrator import ArbitrageOrchestrator
        orch = ArbitrageOrchestrator(config_path=mock_config_file)

        assert orch.config['paper_trading']['enabled'] is True
        assert 'BTC/USDT' in orch.config['pairs']['phase_1']

    @patch('arbitrage.orchestrator.MEXCClient')
    @patch('arbitrage.orchestrator.BinanceClient')
    @patch('arbitrage.orchestrator.PerformanceTracker')
    def test_config_missing_uses_defaults(
        self, mock_tracker_cls, mock_binance_cls, mock_mexc_cls
    ):
        mock_mexc_cls.return_value = MagicMock()
        mock_binance_cls.return_value = MagicMock()
        mock_tracker_cls.return_value = MagicMock()

        from arbitrage.orchestrator import ArbitrageOrchestrator
        # Use a path that definitely doesn't exist
        orch = ArbitrageOrchestrator(config_path="/tmp/nonexistent_config_12345.yaml")
        assert isinstance(orch.config, dict)

    @patch('arbitrage.orchestrator.MEXCClient')
    @patch('arbitrage.orchestrator.BinanceClient')
    @patch('arbitrage.orchestrator.PerformanceTracker')
    def test_paper_trading_mode_set(
        self, mock_tracker_cls, mock_binance_cls, mock_mexc_cls, mock_config_file
    ):
        mock_mexc_cls.return_value = MagicMock()
        mock_binance_cls.return_value = MagicMock()
        mock_tracker_cls.return_value = MagicMock()

        from arbitrage.orchestrator import ArbitrageOrchestrator
        orch = ArbitrageOrchestrator(config_path=mock_config_file)

        # MEXC should be called with paper_trading=True
        mock_mexc_cls.assert_called_once()
        call_kwargs = mock_mexc_cls.call_args
        assert call_kwargs.kwargs.get('paper_trading') is True or call_kwargs[1].get('paper_trading') is True

    @patch('arbitrage.orchestrator.MEXCClient')
    @patch('arbitrage.orchestrator.BinanceClient')
    @patch('arbitrage.orchestrator.PerformanceTracker')
    @pytest.mark.asyncio
    async def test_stop_calls_all_subsystems(
        self, mock_tracker_cls, mock_binance_cls, mock_mexc_cls, mock_config_file
    ):
        mock_mexc = AsyncMock()
        mock_binance = AsyncMock()
        mock_mexc_cls.return_value = mock_mexc
        mock_binance_cls.return_value = mock_binance
        mock_tracker_cls.return_value = MagicMock()

        from arbitrage.orchestrator import ArbitrageOrchestrator
        orch = ArbitrageOrchestrator(config_path=mock_config_file)

        # Replace detectors with mocks for stop()
        orch.cross_exchange_detector = MagicMock()
        orch.funding_arb = MagicMock()
        orch.triangular_arb = MagicMock()
        orch.book_manager = AsyncMock()

        # Mock tracker.get_summary() to return a real dict for _log_final_summary
        orch.tracker = MagicMock()
        orch.tracker.get_summary.return_value = {
            'uptime_hours': 1.5,
            'total_trades': 10,
            'total_fills': 8,
            'total_profit_usd': 2.50,
            'win_rate': 0.75,
            'avg_profit_per_fill': 0.3125,
            'by_strategy': {
                'cross_exchange': {'trades': 5, 'profit_usd': 1.50},
                'funding_rate': {'trades': 3, 'profit_usd': 0.80},
                'triangular': {'trades': 0, 'profit_usd': 0.0},
            },
        }

        await orch.stop()

        assert orch._running is False
        orch.cross_exchange_detector.stop.assert_called_once()
        orch.funding_arb.stop.assert_called_once()
        orch.triangular_arb.stop.assert_called_once()
        orch.book_manager.stop.assert_called_once()

    @patch('arbitrage.orchestrator.MEXCClient')
    @patch('arbitrage.orchestrator.BinanceClient')
    @patch('arbitrage.orchestrator.PerformanceTracker')
    def test_signal_queue_created(
        self, mock_tracker_cls, mock_binance_cls, mock_mexc_cls, mock_config_file
    ):
        mock_mexc_cls.return_value = MagicMock()
        mock_binance_cls.return_value = MagicMock()
        mock_tracker_cls.return_value = MagicMock()

        from arbitrage.orchestrator import ArbitrageOrchestrator
        orch = ArbitrageOrchestrator(config_path=mock_config_file)

        assert orch.signal_queue is not None
        assert orch.signal_queue.maxsize == 100

    @patch('arbitrage.orchestrator.MEXCClient')
    @patch('arbitrage.orchestrator.BinanceClient')
    @patch('arbitrage.orchestrator.PerformanceTracker')
    def test_risk_engine_uses_config(
        self, mock_tracker_cls, mock_binance_cls, mock_mexc_cls, mock_config_file
    ):
        mock_mexc_cls.return_value = MagicMock()
        mock_binance_cls.return_value = MagicMock()
        mock_tracker_cls.return_value = MagicMock()

        from arbitrage.orchestrator import ArbitrageOrchestrator
        orch = ArbitrageOrchestrator(config_path=mock_config_file)

        assert orch.risk_engine.max_single_arb_usd == Decimal('500')
        assert orch.risk_engine.max_total_exposure_usd == Decimal('5000')

    @patch('arbitrage.orchestrator.MEXCClient')
    @patch('arbitrage.orchestrator.BinanceClient')
    @patch('arbitrage.orchestrator.PerformanceTracker')
    def test_book_manager_uses_configured_pairs(
        self, mock_tracker_cls, mock_binance_cls, mock_mexc_cls, mock_config_file
    ):
        mock_mexc_cls.return_value = MagicMock()
        mock_binance_cls.return_value = MagicMock()
        mock_tracker_cls.return_value = MagicMock()

        from arbitrage.orchestrator import ArbitrageOrchestrator
        orch = ArbitrageOrchestrator(config_path=mock_config_file)

        assert "BTC/USDT" in orch.book_manager.monitored_pairs
        assert "ETH/USDT" in orch.book_manager.monitored_pairs

    @patch('arbitrage.orchestrator.MEXCClient')
    @patch('arbitrage.orchestrator.BinanceClient')
    @patch('arbitrage.orchestrator.PerformanceTracker')
    def test_log_status_does_not_crash(
        self, mock_tracker_cls, mock_binance_cls, mock_mexc_cls, mock_config_file
    ):
        mock_mexc_cls.return_value = MagicMock()
        mock_binance_cls.return_value = MagicMock()
        mock_tracker_cls.return_value = MagicMock()

        from arbitrage.orchestrator import ArbitrageOrchestrator
        orch = ArbitrageOrchestrator(config_path=mock_config_file)
        orch._start_time = datetime.utcnow()

        # Mock all get_stats/get_status methods
        orch.book_manager.get_status = MagicMock(return_value={
            'tradeable_pairs': 2, 'total_pairs': 2, 'total_updates': 100,
        })
        orch.cross_exchange_detector.get_stats = MagicMock(return_value={
            'scan_count': 50, 'signals_generated': 5, 'signals_approved': 3,
        })
        orch.executor.get_stats = MagicMock(return_value={
            'total_trades': 3, 'total_fills': 2, 'total_profit_usd': 0.50,
            'win_rate': 0.67, 'wins': 2, 'losses': 1,
        })
        orch.risk_engine.get_status = MagicMock(return_value={
            'halted': False, 'daily_pnl_usd': 0.50, 'total_exposure_usd': 100,
        })
        orch.funding_arb.get_stats = MagicMock(return_value={
            'open_positions': 0, 'total_funding_collected_usd': 0,
        })
        orch.triangular_arb.get_stats = MagicMock(return_value={
            'scan_count': 100, 'opportunities_found': 2,
        })

        # Should not raise
        orch._log_status()

    @patch('arbitrage.orchestrator.MEXCClient')
    @patch('arbitrage.orchestrator.BinanceClient')
    @patch('arbitrage.orchestrator.PerformanceTracker')
    def test_on_trade_with_bar_aggregator(
        self, mock_tracker_cls, mock_binance_cls, mock_mexc_cls, mock_config_file
    ):
        mock_mexc_cls.return_value = MagicMock()
        mock_binance_cls.return_value = MagicMock()
        mock_tracker_cls.return_value = MagicMock()

        from arbitrage.orchestrator import ArbitrageOrchestrator
        from arbitrage.exchanges.base import Trade, OrderSide

        orch = ArbitrageOrchestrator(config_path=mock_config_file)
        mock_bar_agg = MagicMock()
        orch.bar_aggregator = mock_bar_agg

        trade = Trade(
            exchange="mexc", symbol="BTC/USDT", trade_id="t1",
            price=Decimal('50000'), quantity=Decimal('0.01'),
            side=OrderSide.BUY, timestamp=datetime.utcnow(),
        )

        # Run the callback synchronously by calling the coroutine
        loop = asyncio.new_event_loop()
        loop.run_until_complete(orch._on_trade(trade))
        loop.close()

        mock_bar_agg.on_trade.assert_called_once()


# Need datetime import at top level for the log_status test
from datetime import datetime
from decimal import Decimal
