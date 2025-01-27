import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from deepchain.core.data.loader import DataLoader
from deepchain.core.data.processor import DataProcessor
from deepchain.core.data.stream import DataStream

class TestDataInterface(unittest.TestCase):
    """
    Integration tests for the data interface system.
    """
    
    def setUp(self):
        """Set up test environment."""
        self.loader = DataLoader()
        self.processor = DataProcessor()
        self.stream = DataStream()
        
        # Generate test data
        self.test_data = self._generate_test_data()
    
    def test_data_loading(self):
        """Test data loading from different sources."""
        # Test CSV loading
        csv_data = self.loader.load_csv("test_data.csv")
        self.assertIsInstance(csv_data, pd.DataFrame)
        
        # Test API loading
        api_data = self.loader.load_api(
            url="https://api.example.com/market_data",
            params={'symbol': 'BTC-USD'}
        )
        self.assertIsInstance(api_data, pd.DataFrame)
        
        # Test database loading
        db_data = self.loader.load_database(
            query="SELECT * FROM market_data LIMIT 1000"
        )
        self.assertIsInstance(db_data, pd.DataFrame)
    
    def test_data_preprocessing(self):
        """Test data preprocessing pipeline."""
        # Configure preprocessing steps
        self.processor.add_step('fill_missing', method='forward')
        self.processor.add_step('normalize', method='zscore')
        self.processor.add_step('add_technical_indicators', 
                              indicators=['MA', 'RSI', 'MACD'])
        
        # Process data
        processed_data = self.processor.process(self.test_data)
        
        # Verify processing results
        self.assertFalse(processed_data.isnull().any().any())
        self.assertIn('MA_5', processed_data.columns)
        self.assertIn('RSI', processed_data.columns)
        self.assertIn('MACD', processed_data.columns)
    
    def test_data_streaming(self):
        """Test real-time data streaming."""
        received_data = []
        
        def on_data(data):
            received_data.append(data)
        
        # Subscribe to data stream
        self.stream.subscribe(
            symbols=['BTC-USD', 'ETH-USD'],
            callback=on_data
        )
        
        # Wait for some data
        import time
        time.sleep(5)
        
        # Stop streaming
        self.stream.stop()
        
        # Verify received data
        self.assertGreater(len(received_data), 0)
        self.assertIsInstance(received_data[0], dict)
        self.assertIn('timestamp', received_data[0])
        self.assertIn('price', received_data[0])
    
    def test_data_pipeline(self):
        """Test complete data pipeline."""
        # 1. Load historical data
        historical_data = self.loader.load_csv("historical_data.csv")
        
        # 2. Process historical data
        processed_historical = self.processor.process(historical_data)
        
        # 3. Set up real-time pipeline
        def process_stream(data):
            # Process streaming data
            df = pd.DataFrame([data])
            processed = self.processor.process(df)
            return processed.iloc[0].to_dict()
        
        # 4. Start streaming with processing
        processed_stream = []
        self.stream.subscribe(
            symbols=['BTC-USD'],
            callback=lambda x: processed_stream.append(process_stream(x))
        )
        
        # Wait for some data
        time.sleep(5)
        self.stream.stop()
        
        # Verify pipeline results
        self.assertGreater(len(processed_stream), 0)
        self.assertIsInstance(processed_stream[0], dict)
        self.assertIn('MA_5', processed_stream[0])
    
    def test_data_validation(self):
        """Test data validation and quality checks."""
        # Define validation rules
        rules = {
            'price': {
                'type': 'float',
                'min': 0,
                'max': 1e6
            },
            'volume': {
                'type': 'float',
                'min': 0
            },
            'timestamp': {
                'type': 'datetime'
            }
        }
        
        # Validate test data
        validation_result = self.processor.validate(
            self.test_data,
            rules
        )
        
        # Verify validation results
        self.assertTrue(validation_result['is_valid'])
        self.assertEqual(len(validation_result['errors']), 0)
    
    def _generate_test_data(self):
        """Generate synthetic market data for testing."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=100),
            end=datetime.now(),
            freq='1H'
        )
        
        data = pd.DataFrame({
            'timestamp': dates,
            'price': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000, 100000, len(dates)),
            'high': np.random.randn(len(dates)).cumsum() + 102,
            'low': np.random.randn(len(dates)).cumsum() + 98
        })
        
        return data

if __name__ == '__main__':
    unittest.main() 