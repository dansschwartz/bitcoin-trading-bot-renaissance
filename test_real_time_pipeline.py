import asyncio
import unittest
import logging
from unittest.mock import MagicMock, AsyncMock, patch
from real_time_pipeline import RealTimePipeline, MultiExchangeFeed, FeatureFanOutProcessor

class TestRealTimePipeline(unittest.IsolatedAsyncioTestCase):
    
    async def test_pipeline_initialization(self):
        config = {"enabled": True, "exchanges": ["Coinbase", "Kraken"]}
        pipeline = RealTimePipeline(config)
        self.assertTrue(pipeline.enabled)
        self.assertEqual(len(pipeline.feed.exchanges), 2)
        
    async def test_feed_aggregation(self):
        # Mock ccxt clients
        with patch('ccxt.coinbase') as mock_coinbase, \
             patch('ccxt.kraken') as mock_kraken:
            
            mock_coinbase.return_value.fetch_ticker = MagicMock(return_value={
                'last': 50000, 'bid': 49990, 'ask': 50010, 'quoteVolume': 100, 'timestamp': 123456789
            })
            mock_kraken.return_value.fetch_ticker = MagicMock(return_value={
                'last': 50010, 'bid': 50000, 'ask': 50020, 'quoteVolume': 150, 'timestamp': 123456790
            })
            
            feed = MultiExchangeFeed(["Coinbase", "Kraken"])
            await feed.start()
            
            snapshot = await feed.get_aggregated_snapshot()
            
            self.assertEqual(snapshot['source_count'], 2)
            self.assertAlmostEqual(snapshot['avg_price'], 50005.0)
            self.assertEqual(snapshot['global_liquidity'], 250)
            
    async def test_processor_fanout(self):
        models = ["ModelA", "ModelB"]
        processor = FeatureFanOutProcessor(models)
        predictions = await processor.process_all_models({"test": 1.0})
        
        self.assertEqual(len(predictions), 2)
        self.assertIn("ModelA", predictions)
        self.assertIn("ModelB", predictions)
        
    async def test_pipeline_cycle(self):
        config = {"enabled": True, "exchanges": ["Coinbase"]}
        pipeline = RealTimePipeline(config)
        
        # Mock feed
        pipeline.feed.active = True
        pipeline.feed.get_aggregated_snapshot = AsyncMock(return_value={
            'avg_price': 60000,
            'source_count': 1,
            'global_liquidity': 1000
        })
        
        result = await pipeline.run_cycle()
        
        self.assertIn('snapshot', result)
        self.assertIn('predictions', result)
        self.assertEqual(result['snapshot']['avg_price'], 60000)

if __name__ == '__main__':
    unittest.main()
