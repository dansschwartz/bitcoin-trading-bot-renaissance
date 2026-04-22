"""
data_module -- Data ingestion, aggregation, and feature engineering.
"""

from .bar_aggregator import BarAggregator, FiveMinuteBar

__all__ = ["BarAggregator", "FiveMinuteBar"]
