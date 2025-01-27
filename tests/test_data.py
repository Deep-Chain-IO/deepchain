"""
Unit tests for data processing module.
"""

import unittest
import numpy as np
import pandas as pd
from deepchain.core.data.loader import DataLoader
from deepchain.core.data.processor import DataProcessor

class TestDataProcessing(unittest.TestCase):
    """Test data processing functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.loader = DataLoader()
        self.processor = DataProcessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'price': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100),
            'missing_col': [np.nan if i % 10 == 0 else i for i in range(100)]
        })
    
    def test_data_loading(self):
        """Test data loading functionality."""
        # Test CSV loading
        self.loader.save_csv(self.sample_data, 'test_data.csv')
        loaded_data = self.loader.load_csv('test_data.csv')
        pd.testing.assert_frame_equal(loaded_data, self.sample_data)
        
        # Test API loading
        api_data = self.loader.load_api(
            'https://api.example.com/data',
            {'symbol': 'BTC/USD'}
        )
        self.assertIsInstance(api_data, pd.DataFrame)
    
    def test_missing_value_handling(self):
        """Test missing value handling."""
        # Add missing value handling step
        self.processor.add_step('fill_missing', method='forward')
        processed_data = self.processor.process(self.sample_data)
        
        # Check no missing values
        self.assertFalse(processed_data.isnull().any().any())
        
        # Check forward fill
        missing_indices = self.sample_data['missing_col'].isnull()
        self.assertTrue(
            (processed_data.loc[missing_indices, 'missing_col'] == 
             processed_data.loc[missing_indices.shift(1), 'missing_col']).all()
        )
    
    def test_normalization(self):
        """Test data normalization."""
        # Add normalization step
        self.processor.add_step('normalize', method='zscore')
        processed_data = self.processor.process(self.sample_data)
        
        # Check z-score normalization
        numeric_cols = ['price', 'volume']
        for col in numeric_cols:
            normalized = processed_data[col]
            self.assertAlmostEqual(normalized.mean(), 0, places=2)
            self.assertAlmostEqual(normalized.std(), 1, places=2)
    
    def test_technical_indicators(self):
        """Test technical indicator calculation."""
        # Add technical indicators
        self.processor.add_step(
            'add_technical_indicators',
            indicators=['MA', 'RSI', 'MACD', 'BBANDS']
        )
        processed_data = self.processor.process(self.sample_data)
        
        # Check indicators exist
        expected_columns = [
            'price_MA_5', 'price_MA_10',
            'price_RSI_14',
            'price_MACD', 'price_MACD_signal',
            'price_BB_upper', 'price_BB_lower'
        ]
        for col in expected_columns:
            self.assertIn(col, processed_data.columns)
    
    def test_data_validation(self):
        """Test data validation."""
        # Add validation rules
        validation_rules = {
            'price': {'min': 0, 'max': 1000},
            'volume': {'min': 0},
            'timestamp': {'unique': True}
        }
        self.processor.add_step('validate', rules=validation_rules)
        
        # Test valid data
        try:
            self.processor.process(self.sample_data)
        except Exception as e:
            self.fail(f"Validation failed: {e}")
        
        # Test invalid data
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'price'] = -1
        with self.assertRaises(ValueError):
            self.processor.process(invalid_data)
    
    def test_feature_engineering(self):
        """Test feature engineering."""
        # Add feature engineering steps
        self.processor.add_step(
            'add_features',
            features={
                'price_change': lambda df: df['price'].pct_change(),
                'volume_ma': lambda df: df['volume'].rolling(5).mean(),
                'price_momentum': lambda df: df['price'].diff(5)
            }
        )
        processed_data = self.processor.process(self.sample_data)
        
        # Check engineered features
        expected_features = ['price_change', 'volume_ma', 'price_momentum']
        for feature in expected_features:
            self.assertIn(feature, processed_data.columns)
    
    def test_data_pipeline(self):
        """Test complete data processing pipeline."""
        # Configure complete pipeline
        self.processor.add_step('fill_missing', method='forward')
        self.processor.add_step('normalize', method='zscore')
        self.processor.add_step(
            'add_technical_indicators',
            indicators=['MA', 'RSI']
        )
        self.processor.add_step(
            'add_features',
            features={'price_change': lambda df: df['price'].pct_change()}
        )
        
        # Process data
        processed_data = self.processor.process(self.sample_data)
        
        # Check pipeline results
        self.assertFalse(processed_data.isnull().any().any())
        self.assertIn('price_MA_5', processed_data.columns)
        self.assertIn('price_change', processed_data.columns)
    
    def tearDown(self):
        """Clean up test environment."""
        import os
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')

if __name__ == '__main__':
    unittest.main() 