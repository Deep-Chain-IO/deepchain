"""
Real-time data streaming and processing system for DeepChain.
"""

from typing import Dict, Any, List, Optional, Callable
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import asyncio
import aiohttp
import websockets
import logging
from queue import Queue
from threading import Thread
from datetime import datetime

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data source."""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data."""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str
    ) -> pd.DataFrame:
        """Get historical market data."""
        pass

class WebSocketSource(DataSource):
    """WebSocket-based data source."""
    
    def __init__(self, url: str, api_key: Optional[str] = None):
        """Initialize WebSocket source.
        
        Args:
            url: WebSocket endpoint URL
            api_key: Optional API key
        """
        self.url = url
        self.api_key = api_key
        self.ws = None
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to WebSocket."""
        try:
            self.ws = await websockets.connect(self.url)
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data."""
        if not self.connected:
            raise ConnectionError("Not connected to WebSocket")
        
        subscribe_msg = {
            'type': 'subscribe',
            'symbols': symbols
        }
        await self.ws.send(subscribe_msg)
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data."""
        if not self.connected:
            raise ConnectionError("Not connected to WebSocket")
        
        unsubscribe_msg = {
            'type': 'unsubscribe',
            'symbols': symbols
        }
        await self.ws.send(unsubscribe_msg)
    
    async def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str
    ) -> pd.DataFrame:
        """Get historical market data using REST API."""
        async with aiohttp.ClientSession() as session:
            params = {
                'symbol': symbol,
                'start_time': int(start_time.timestamp() * 1000),
                'end_time': int(end_time.timestamp() * 1000),
                'interval': interval
            }
            if self.api_key:
                params['api_key'] = self.api_key
            
            async with session.get(f"{self.url}/history", params=params) as response:
                data = await response.json()
                return pd.DataFrame(data)

class DataStream:
    """Real-time data streaming and processing."""
    
    def __init__(self, buffer_size: int = 1000):
        """Initialize data stream.
        
        Args:
            buffer_size: Size of data buffer
        """
        self.buffer_size = buffer_size
        self.sources: Dict[str, DataSource] = {}
        self.processors: List[Callable] = []
        self.data_buffer = Queue(maxsize=buffer_size)
        self.running = False
        self.processing_thread = None
    
    def add_source(self, name: str, source: DataSource) -> None:
        """Add data source.
        
        Args:
            name: Source name
            source: DataSource instance
        """
        self.sources[name] = source
    
    def add_processor(self, processor: Callable) -> None:
        """Add data processor.
        
        Args:
            processor: Processing function
        """
        self.processors.append(processor)
    
    async def start(self) -> None:
        """Start data streaming."""
        if self.running:
            return
        
        self.running = True
        self.processing_thread = Thread(target=self._process_data)
        self.processing_thread.start()
        
        # Connect to all sources
        for name, source in self.sources.items():
            connected = await source.connect()
            if not connected:
                logger.error(f"Failed to connect to source: {name}")
    
    async def stop(self) -> None:
        """Stop data streaming."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        
        # Disconnect from all sources
        for source in self.sources.values():
            if hasattr(source, 'ws') and source.ws:
                await source.ws.close()
    
    def _process_data(self) -> None:
        """Process incoming data."""
        while self.running:
            if not self.data_buffer.empty():
                data = self.data_buffer.get()
                
                # Apply all processors
                for processor in self.processors:
                    try:
                        data = processor(data)
                    except Exception as e:
                        logger.error(f"Data processing error: {e}")
                
                # Emit processed data
                self._emit_data(data)
    
    def _emit_data(self, data: Any) -> None:
        """Emit processed data."""
        # Implement your data emission logic here
        pass

class DataPipeline:
    """Data processing pipeline."""
    
    def __init__(self):
        """Initialize data pipeline."""
        self.steps: List[Callable] = []
    
    def add_step(self, step: Callable) -> None:
        """Add processing step.
        
        Args:
            step: Processing function
        """
        self.steps.append(step)
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data through pipeline.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        result = data.copy()
        for step in self.steps:
            try:
                result = step(result)
            except Exception as e:
                logger.error(f"Pipeline step failed: {e}")
                raise
        return result

class MarketDataStream(DataStream):
    """Market data streaming with technical indicators."""
    
    def __init__(self, buffer_size: int = 1000):
        """Initialize market data stream."""
        super().__init__(buffer_size)
        self.indicators = {}
    
    def add_indicator(self, name: str, func: Callable, **params) -> None:
        """Add technical indicator.
        
        Args:
            name: Indicator name
            func: Indicator calculation function
            **params: Indicator parameters
        """
        self.indicators[name] = (func, params)
    
    def _process_data(self) -> None:
        """Process market data with indicators."""
        while self.running:
            if not self.data_buffer.empty():
                data = self.data_buffer.get()
                
                # Calculate indicators
                for name, (func, params) in self.indicators.items():
                    try:
                        data[name] = func(data, **params)
                    except Exception as e:
                        logger.error(f"Indicator calculation error: {e}")
                
                # Apply processors
                for processor in self.processors:
                    try:
                        data = processor(data)
                    except Exception as e:
                        logger.error(f"Data processing error: {e}")
                
                self._emit_data(data) 