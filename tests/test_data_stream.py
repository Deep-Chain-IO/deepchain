"""
Unit tests for data streaming system.
"""

import unittest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from deepchain.core.data.stream import (
    DataSource,
    WebSocketSource,
    DataStream,
    DataPipeline,
    MarketDataStream
)
from deepchain.core.data.indicators import (
    calculate_ma,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands
)

class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        self.messages = []
    
    async def send(self, message):
        self.messages.append(message)
    
    async def close(self):
        pass

class TestDataStream(unittest.TestCase):
    """Test data streaming functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.stream = DataStream(buffer_size=1000)
        self.mock_source = Mock(spec=DataSource)
        self.stream.add_source("test", self.mock_source)
    
    async def test_source_connection(self):
        """Test data source connection."""
        self.mock_source.connect.return_value = True
        await self.stream.start()
        self.mock_source.connect.assert_called_once()
    
    def test_processor_addition(self):
        """Test adding data processors."""
        def test_processor(data):
            return data
        
        self.stream.add_processor(test_processor)
        self.assertIn(test_processor, self.stream.processors)
    
    async def test_stream_lifecycle(self):
        """Test stream start and stop."""
        await self.stream.start()
        self.assertTrue(self.stream.running)
        
        await self.stream.stop()
        self.assertFalse(self.stream.running)
    
    def test_buffer_management(self):
        """Test data buffer management."""
        self.assertEqual(self.stream.data_buffer.maxsize, 1000)
        self.assertTrue(self.stream.data_buffer.empty())

class TestWebSocketSource(unittest.TestCase):
    """Test WebSocket data source."""
    
    def setUp(self):
        """Set up test environment."""
        self.source = WebSocketSource(
            url="ws://test.com",
            api_key="test-key"
        )
        self.mock_ws = MockWebSocket()
    
    @patch('websockets.connect')
    async def test_connection(self, mock_connect):
        """Test WebSocket connection."""
        mock_connect.return_value = self.mock_ws
        connected = await self.source.connect()
        self.assertTrue(connected)
        self.assertTrue(self.source.connected)
    
    async def test_subscription(self):
        """Test market data subscription."""
        self.source.ws = self.mock_ws
        self.source.connected = True
        
        symbols = ['BTCUSDT', 'ETHUSDT']
        await self.source.subscribe(symbols)
        
        last_message = self.mock_ws.messages[-1]
        self.assertEqual(last_message['type'], 'subscribe')
        self.assertEqual(last_message['symbols'], symbols)
    
    async def test_historical_data(self):
        """Test historical data retrieval."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json.return_value = [
                {'timestamp': 1, 'close': 100},
                {'timestamp': 2, 'close': 101}
            ]
            
            data = await self.source.get_historical_data(
                'BTCUSDT',
                datetime.now() - timedelta(days=1),
                datetime.now(),
                '1h'
            )
            
            self.assertIsInstance(data, pd.DataFrame)
            self.assertEqual(len(data), 2)

class TestMarketDataStream(unittest.TestCase):
    """Test market data streaming."""
    
    def setUp(self):
        """Set up test environment."""
        self.stream = MarketDataStream(buffer_size=1000)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
    
    def test_indicator_addition(self):
        """Test adding technical indicators."""
        self.stream.add_indicator(
            "ma_20",
            calculate_ma,
            period=20
        )
        self.assertIn("ma_20", self.stream.indicators)
    
    def test_indicator_calculation(self):
        """Test technical indicator calculation."""
        # Add indicators
        self.stream.add_indicator(
            "ma_20",
            calculate_ma,
            period=20
        )
        self.stream.add_indicator(
            "rsi",
            calculate_rsi,
            period=14
        )
        
        # Process data
        data = self.sample_data.copy()
        for name, (func, params) in self.stream.indicators.items():
            result = func(data, **params)
            if isinstance(result, pd.DataFrame):
                for col in result.columns:
                    data[f'{name}_{col}'] = result[col]
            else:
                data[name] = result
        
        # Verify results
        self.assertIn("ma_20", data.columns)
        self.assertIn("rsi", data.columns)
        self.assertTrue(all(data["ma_20"].notna().tail(80)))
        self.assertTrue(all(data["rsi"].notna().tail(80)))

class TestDataPipeline(unittest.TestCase):
    """Test data processing pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.pipeline = DataPipeline()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'close': [100, 101, np.nan, 103, 104],
            'volume': [1000, np.nan, 3000, 4000, 5000]
        })
    
    def test_step_addition(self):
        """Test adding processing steps."""
        def test_step(data):
            return data
        
        self.pipeline.add_step(test_step)
        self.assertIn(test_step, self.pipeline.steps)
    
    def test_data_processing(self):
        """Test data processing through pipeline."""
        # Add processing steps
        def fill_missing(data):
            return data.fillna(method='ffill')
        
        def add_returns(data):
            data['returns'] = data['close'].pct_change()
            return data
        
        self.pipeline.add_step(fill_missing)
        self.pipeline.add_step(add_returns)
        
        # Process data
        result = self.pipeline.process(self.sample_data)
        
        # Verify results
        self.assertFalse(result.isnull().any().any())
        self.assertIn('returns', result.columns)
        self.assertEqual(len(result), len(self.sample_data))

if __name__ == '__main__':
    unittest.main() 