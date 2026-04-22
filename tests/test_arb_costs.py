"""
Tests for arbitrage/costs/model.py

Covers:
  - ArbitrageCostModel fee schedule
  - Cost estimation for different speed tiers
  - MEXC 0% maker fee verification
  - Slippage estimation (default and learned)
  - Learning from execution data
  - Estimation error tracking
  - Model accuracy reporting
"""
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

from arbitrage.costs.model import ArbitrageCostModel, CostEstimate
from arbitrage.exchanges.base import SpeedTier


class TestCostEstimate:
    def test_dataclass_creation(self):
        est = CostEstimate(
            total_cost_bps=Decimal('5.0'),
            buy_fee_bps=Decimal('0'),
            sell_fee_bps=Decimal('7.5'),
            buy_slippage_bps=Decimal('1.5'),
            sell_slippage_bps=Decimal('0.5'),
            timing_cost_bps=Decimal('0.5'),
            taker_maker_cost_bps=Decimal('8.0'),
            taker_taker_cost_bps=Decimal('12.5'),
        )
        assert est.total_cost_bps == Decimal('5.0')


class TestArbitrageCostModel:
    @pytest.fixture
    def model(self):
        return ArbitrageCostModel()

    def test_initialization(self, model):
        assert len(model._cost_history) == 0
        assert len(model._slippage_history) == 0

    def test_mexc_zero_maker_fee(self, model):
        assert model.FEES["mexc"]["spot"]["maker"] == Decimal('0.0000')

    def test_binance_bnb_discount_fee(self, model):
        assert model.FEES["binance"]["spot"]["maker"] == Decimal('0.00075')

    def test_estimate_maker_maker_cost(self, model):
        est = model.estimate_arbitrage_cost(
            symbol="BTC/USDT",
            buy_exchange="mexc",
            sell_exchange="binance",
            buy_price=Decimal('50000'),
            sell_price=Decimal('50050'),
            speed_tier=SpeedTier.MAKER_MAKER,
        )
        # Buy fee: 0.0000 * 10000 = 0 bps
        # Sell fee: 0.00075 * 10000 = 7.5 bps
        # Buy slippage: 1.5 (mexc default)
        # Sell slippage: 0.5 (binance default)
        # Timing: 0.5
        # Total = 0 + 7.5 + 1.5 + 0.5 + 0.5 = 10.0
        assert est.buy_fee_bps == Decimal('0')
        assert est.sell_fee_bps == Decimal('7.5')
        assert est.timing_cost_bps == Decimal('0.5')
        expected_total = Decimal('0') + Decimal('7.5') + Decimal('1.5') + Decimal('0.5') + Decimal('0.5')
        assert est.total_cost_bps == expected_total

    def test_estimate_taker_maker_cost(self, model):
        est = model.estimate_arbitrage_cost(
            symbol="BTC/USDT",
            buy_exchange="mexc",
            sell_exchange="binance",
            buy_price=Decimal('50000'),
            sell_price=Decimal('50050'),
            speed_tier=SpeedTier.TAKER_MAKER,
        )
        # Taker-maker: buy uses taker fee (0.0005 * 10000 = 5 bps), sell uses maker fee
        assert est.total_cost_bps == est.taker_maker_cost_bps
        assert est.taker_maker_cost_bps > est.total_cost_bps or est.taker_maker_cost_bps == est.total_cost_bps

    def test_estimate_taker_taker_cost(self, model):
        est = model.estimate_arbitrage_cost(
            symbol="BTC/USDT",
            buy_exchange="mexc",
            sell_exchange="binance",
            buy_price=Decimal('50000'),
            sell_price=Decimal('50050'),
            speed_tier=SpeedTier.TAKER_TAKER,
        )
        assert est.total_cost_bps == est.taker_taker_cost_bps
        # Taker-taker should be most expensive
        mm_est = model.estimate_arbitrage_cost(
            symbol="BTC/USDT", buy_exchange="mexc", sell_exchange="binance",
            buy_price=Decimal('50000'), sell_price=Decimal('50050'),
            speed_tier=SpeedTier.MAKER_MAKER,
        )
        assert est.total_cost_bps >= mm_est.total_cost_bps

    def test_default_slippage_mexc(self, model):
        slip = model._estimate_slippage("BTC/USDT", "mexc", None)
        assert slip == Decimal('1.5')

    def test_default_slippage_binance(self, model):
        slip = model._estimate_slippage("BTC/USDT", "binance", None)
        assert slip == Decimal('0.5')

    def test_default_slippage_unknown_exchange(self, model):
        slip = model._estimate_slippage("BTC/USDT", "kraken", None)
        assert slip == Decimal('2.0')

    def test_learned_slippage_after_enough_data(self, model):
        key = "BTC/USDT_mexc"
        from collections import deque
        model._slippage_history[key] = deque(maxlen=200)
        for _ in range(15):
            model._slippage_history[key].append(Decimal('1.0'))

        slip = model._estimate_slippage("BTC/USDT", "mexc", None)
        # avg = 1.0, with 20% safety margin = 1.2
        assert slip == Decimal('1.0') * Decimal('1.2')

    def test_update_from_execution_records_history(self, model):
        trade_result = {
            'symbol': 'BTC/USDT',
            'exchange': 'mexc',
            'quantity': Decimal('0.01'),
            'estimated_cost_bps': Decimal('5'),
            'realized_cost_bps': Decimal('6'),
        }
        model.update_from_execution(trade_result)

        assert len(model._cost_history) == 1
        assert len(model._estimation_errors) == 1
        assert model._estimation_errors[0] == Decimal('1')  # 6 - 5

    def test_update_from_execution_records_slippage(self, model):
        trade_result = {
            'symbol': 'BTC/USDT',
            'exchange': 'mexc',
            'quantity': Decimal('0.01'),
            'estimated_cost_bps': Decimal('5'),
            'realized_cost_bps': Decimal('6'),
            'actual_slippage_bps': Decimal('1.2'),
        }
        model.update_from_execution(trade_result)

        assert "BTC/USDT_mexc" in model._slippage_history
        assert model._slippage_history["BTC/USDT_mexc"][-1] == Decimal('1.2')

    def test_model_accuracy_empty(self, model):
        acc = model.get_model_accuracy()
        assert acc["samples"] == 0
        assert acc["mean_error_bps"] == 0

    def test_model_accuracy_with_data(self, model):
        model._estimation_errors.append(Decimal('1.0'))
        model._estimation_errors.append(Decimal('-2.0'))
        model._estimation_errors.append(Decimal('0.5'))

        acc = model.get_model_accuracy()
        assert acc["samples"] == 3
        assert acc["mean_error_bps"] == (1.0 + 2.0 + 0.5) / 3
        assert acc["max_error_bps"] == 2.0
        assert acc["bias_bps"] == (1.0 + -2.0 + 0.5) / 3

    def test_both_mexc_to_mexc_zero_fees(self, model):
        """Triangular arb: all three legs on MEXC = 0 fees."""
        est = model.estimate_arbitrage_cost(
            symbol="BTC/USDT",
            buy_exchange="mexc",
            sell_exchange="mexc",
            buy_price=Decimal('50000'),
            sell_price=Decimal('50050'),
            speed_tier=SpeedTier.MAKER_MAKER,
        )
        assert est.buy_fee_bps == Decimal('0')
        assert est.sell_fee_bps == Decimal('0')

    def test_futures_market_type(self, model):
        est = model.estimate_arbitrage_cost(
            symbol="BTC/USDT",
            buy_exchange="mexc",
            sell_exchange="binance",
            buy_price=Decimal('50000'),
            sell_price=Decimal('50050'),
            market_type="futures",
            speed_tier=SpeedTier.MAKER_MAKER,
        )
        # Futures fees: mexc maker 0, binance futures maker 0.0002
        assert est.buy_fee_bps == Decimal('0')
        assert est.sell_fee_bps == Decimal('0.0002') * 10000
