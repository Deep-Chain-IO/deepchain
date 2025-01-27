"""
Technical indicators for market data analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional

def calculate_ma(data: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
    """Calculate Moving Average.
    
    Args:
        data: Price data
        period: MA period
        column: Price column name
        
    Returns:
        MA values
    """
    return data[column].rolling(window=period).mean()

def calculate_ema(data: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
    """Calculate Exponential Moving Average.
    
    Args:
        data: Price data
        period: EMA period
        column: Price column name
        
    Returns:
        EMA values
    """
    return data[column].ewm(span=period, adjust=False).mean()

def calculate_rsi(data: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """Calculate Relative Strength Index.
    
    Args:
        data: Price data
        period: RSI period
        column: Price column name
        
    Returns:
        RSI values
    """
    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(
    data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    column: str = 'close'
) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        data: Price data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        column: Price column name
        
    Returns:
        DataFrame with MACD line, signal line and histogram
    """
    fast_ema = calculate_ema(data, fast_period, column)
    slow_ema = calculate_ema(data, slow_period, column)
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })

def calculate_bollinger_bands(
    data: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    column: str = 'close'
) -> pd.DataFrame:
    """Calculate Bollinger Bands.
    
    Args:
        data: Price data
        period: Moving average period
        std_dev: Number of standard deviations
        column: Price column name
        
    Returns:
        DataFrame with upper band, middle band and lower band
    """
    middle_band = calculate_ma(data, period, column)
    std = data[column].rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return pd.DataFrame({
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band
    })

def calculate_atr(
    data: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range.
    
    Args:
        data: Price data with high, low, close columns
        period: ATR period
        
    Returns:
        ATR values
    """
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_stochastic(
    data: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3
) -> pd.DataFrame:
    """Calculate Stochastic Oscillator.
    
    Args:
        data: Price data with high, low, close columns
        k_period: %K period
        d_period: %D period
        smooth_k: %K smoothing period
        
    Returns:
        DataFrame with %K and %D values
    """
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    
    k = 100 * ((data['close'] - low_min) / (high_max - low_min))
    k = k.rolling(window=smooth_k).mean()
    d = k.rolling(window=d_period).mean()
    
    return pd.DataFrame({
        'k': k,
        'd': d
    })

def calculate_obv(data: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume.
    
    Args:
        data: Price data with close and volume columns
        
    Returns:
        OBV values
    """
    close_diff = data['close'].diff()
    volume = data['volume']
    
    obv = pd.Series(index=data.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(data)):
        if close_diff.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close_diff.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_vwap(data: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price.
    
    Args:
        data: Price data with high, low, close and volume columns
        
    Returns:
        VWAP values
    """
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    return (typical_price * data['volume']).cumsum() / data['volume'].cumsum()

def calculate_momentum(
    data: pd.DataFrame,
    period: int = 14,
    column: str = 'close'
) -> pd.Series:
    """Calculate Momentum.
    
    Args:
        data: Price data
        period: Momentum period
        column: Price column name
        
    Returns:
        Momentum values
    """
    return data[column].diff(period)

def calculate_williams_r(
    data: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """Calculate Williams %R.
    
    Args:
        data: Price data with high, low, close columns
        period: Look-back period
        
    Returns:
        Williams %R values
    """
    highest_high = data['high'].rolling(window=period).max()
    lowest_low = data['low'].rolling(window=period).min()
    
    wr = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
    return wr 