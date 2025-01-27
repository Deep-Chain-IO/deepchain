"""
Utility functions for the DeepChain framework.
"""

import logging
import json
import hashlib
from typing import Any, Dict, List, Optional
from pathlib import Path
import yaml

# Configure logging
logger = logging.getLogger(__name__)

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    config = {
        'level': level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
    
    if log_file:
        config['filename'] = log_file
        
    logging.basicConfig(**config)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        raise

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    try:
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {str(e)}")
        raise

def compute_hash(data: Any) -> str:
    """
    Compute SHA256 hash of data.
    
    Args:
        data: Data to hash
        
    Returns:
        Hash string
    """
    try:
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute hash: {str(e)}")
        raise

def ensure_directory(path: str) -> None:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate configuration against schema.
    
    Args:
        config: Configuration to validate
        schema: Validation schema
        
    Returns:
        True if valid, False otherwise
    """
    # Simple schema validation
    try:
        for key, value_type in schema.items():
            if key not in config:
                return False
            if not isinstance(config[key], value_type):
                return False
        return True
    except Exception as e:
        logger.error(f"Config validation failed: {str(e)}")
        return False

def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f}PB"

def parse_timeframe(timeframe: str) -> int:
    """
    Parse timeframe string to seconds.
    
    Args:
        timeframe: Timeframe string (e.g., '1h', '1d')
        
    Returns:
        Seconds
    """
    units = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800
    }
    
    try:
        value = int(timeframe[:-1])
        unit = timeframe[-1].lower()
        return value * units[unit]
    except Exception as e:
        logger.error(f"Failed to parse timeframe {timeframe}: {str(e)}")
        raise

def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Retry decorator for functions.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
    """
    from functools import wraps
    import time
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    logger.warning(f"Attempt {attempts} failed: {str(e)}")
                    time.sleep(delay)
        return wrapper
    return decorator 