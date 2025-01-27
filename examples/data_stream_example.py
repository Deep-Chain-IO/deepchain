"""
Example of using the data streaming system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
from deepchain.core.data.stream import WebSocketSource, MarketDataStream
from deepchain.core.data.indicators import (
    calculate_ma,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_vwap
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataStreamExample:
    """Example class demonstrating data streaming system usage."""
    
    def __init__(self):
        """Initialize data streaming example."""
        # Initialize data stream
        self.stream = MarketDataStream(buffer_size=1000)
        
        # Add data source
        self.source = WebSocketSource(
            url="wss://stream.binance.com:9443/ws",
            api_key="YOUR-API-KEY"  # Replace with your API key
        )
        self.stream.add_source("binance", self.source)
        
        # Add technical indicators
        self.setup_indicators()
        
        # Add data processors
        self.setup_processors()
    
    def setup_indicators(self):
        """Setup technical indicators."""
        # Add moving averages
        self.stream.add_indicator(
            "ma_20",
            calculate_ma,
            period=20
        )
        self.stream.add_indicator(
            "ma_50",
            calculate_ma,
            period=50
        )
        
        # Add RSI
        self.stream.add_indicator(
            "rsi",
            calculate_rsi,
            period=14
        )
        
        # Add MACD
        self.stream.add_indicator(
            "macd",
            calculate_macd,
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        
        # Add Bollinger Bands
        self.stream.add_indicator(
            "bbands",
            calculate_bollinger_bands,
            period=20,
            std_dev=2.0
        )
        
        # Add VWAP
        self.stream.add_indicator(
            "vwap",
            calculate_vwap
        )
    
    def setup_processors(self):
        """Setup data processors."""
        # Add missing value handler
        def handle_missing(data: pd.DataFrame) -> pd.DataFrame:
            return data.fillna(method='ffill').fillna(method='bfill')
        
        # Add outlier detector
        def detect_outliers(data: pd.DataFrame) -> pd.DataFrame:
            for col in ['close', 'volume']:
                if col in data.columns:
                    mean = data[col].mean()
                    std = data[col].std()
                    data[f'{col}_is_outlier'] = (
                        (data[col] < mean - 3 * std) |
                        (data[col] > mean + 3 * std)
                    )
            return data
        
        # Add processors to stream
        self.stream.add_processor(handle_missing)
        self.stream.add_processor(detect_outliers)
    
    async def start_streaming(self, symbols: list):
        """Start data streaming for given symbols.
        
        Args:
            symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        """
        logger.info(f"Starting data stream for symbols: {symbols}")
        
        # Start stream
        await self.stream.start()
        
        # Subscribe to symbols
        await self.source.subscribe(symbols)
        
        try:
            while True:
                # Keep the stream running
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping data stream...")
            await self.stream.stop()
    
    def process_historical_data(self, symbol: str, days: int = 30):
        """Process historical data for backtesting.
        
        Args:
            symbol: Trading pair symbol
            days: Number of days of historical data
        """
        logger.info(f"Processing historical data for {symbol}")
        
        # Get historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        data = asyncio.run(
            self.source.get_historical_data(
                symbol,
                start_time,
                end_time,
                interval='1h'
            )
        )
        
        # Calculate indicators
        for name, (func, params) in self.stream.indicators.items():
            try:
                result = func(data, **params)
                if isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        data[f'{name}_{col}'] = result[col]
                else:
                    data[name] = result
            except Exception as e:
                logger.error(f"Error calculating {name}: {e}")
        
        # Apply processors
        for processor in self.stream.processors:
            try:
                data = processor(data)
            except Exception as e:
                logger.error(f"Error in processor: {e}")
        
        return data

async def main():
    """Run data streaming example."""
    example = DataStreamExample()
    
    # Process historical data
    historical_data = example.process_historical_data('BTCUSDT', days=30)
    logger.info("\nHistorical Data Sample:")
    logger.info(historical_data.tail())
    
    # Start real-time streaming
    symbols = ['BTCUSDT', 'ETHUSDT']
    await example.start_streaming(symbols)

if __name__ == '__main__':
    asyncio.run(main()) 