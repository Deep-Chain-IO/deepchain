"""
Metrics collection and aggregation for system monitoring.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
import psutil
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and aggregates system metrics."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize metrics collector.
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        
        # Initialize metric buffers
        self.latencies = deque(maxlen=window_size)
        self.memory_usages = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        
        # System metrics
        self.system_metrics = {
            'cpu_usage': deque(maxlen=window_size),
            'memory_usage': deque(maxlen=window_size),
            'disk_usage': deque(maxlen=window_size),
            'network_io': deque(maxlen=window_size)
        }
        
        # Initialize timestamps
        self.last_update = time.time()
    
    def record_latency(self, latency: float) -> None:
        """Record inference latency.
        
        Args:
            latency: Latency in milliseconds
        """
        self.latencies.append(latency)
        self._update_timestamp()
    
    def record_memory_usage(self, usage: int) -> None:
        """Record memory usage.
        
        Args:
            usage: Memory usage in bytes
        """
        self.memory_usages.append(usage)
        self._update_timestamp()
    
    def record_prediction(self, prediction: float) -> None:
        """Record model prediction.
        
        Args:
            prediction: Prediction value
        """
        self.predictions.append(prediction)
        self._update_timestamp()
    
    def record_error(self, error: float) -> None:
        """Record prediction error.
        
        Args:
            error: Error value
        """
        self.errors.append(error)
        self._update_timestamp()
    
    def record_system_metrics(self, metrics: Dict[str, float]) -> None:
        """Record system metrics.
        
        Args:
            metrics: Dictionary of system metrics
        """
        for key, value in metrics.items():
            if key in self.system_metrics:
                self.system_metrics[key].append(value)
        self._update_timestamp()
    
    def get_statistics(self) -> Dict[str, float]:
        """Get aggregated statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        stats = {}
        
        # Latency statistics
        if self.latencies:
            stats.update({
                'avg_latency': np.mean(self.latencies),
                'p95_latency': np.percentile(self.latencies, 95),
                'p99_latency': np.percentile(self.latencies, 99),
                'max_latency': np.max(self.latencies)
            })
        
        # Memory statistics
        if self.memory_usages:
            stats.update({
                'avg_memory_usage': np.mean(self.memory_usages),
                'max_memory_usage': np.max(self.memory_usages)
            })
        
        # Prediction statistics
        if self.predictions:
            stats.update({
                'prediction_count': len(self.predictions),
                'prediction_rate': len(self.predictions) / self.window_size
            })
        
        # Error statistics
        if self.errors:
            stats.update({
                'avg_error': np.mean(self.errors),
                'error_rate': len(self.errors) / self.window_size
            })
        
        # Add system metrics
        for key, values in self.system_metrics.items():
            if values:
                stats[f'avg_{key}'] = np.mean(values)
                stats[f'max_{key}'] = np.max(values)
        
        return stats
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status.
        
        Returns:
            Dictionary containing health status information
        """
        stats = self.get_statistics()
        
        # Define health thresholds
        thresholds = {
            'latency': 100,  # ms
            'error_rate': 0.01,  # 1%
            'memory_usage': 0.9,  # 90%
            'cpu_usage': 0.8  # 80%
        }
        
        # Check component status
        components = {}
        status = 'healthy'
        
        # Check latency
        if 'avg_latency' in stats:
            components['latency'] = {
                'status': 'healthy' if stats['avg_latency'] < thresholds['latency'] else 'warning',
                'value': stats['avg_latency']
            }
            if stats['avg_latency'] >= thresholds['latency']:
                status = 'warning'
        
        # Check error rate
        if 'error_rate' in stats:
            components['errors'] = {
                'status': 'healthy' if stats['error_rate'] < thresholds['error_rate'] else 'critical',
                'value': stats['error_rate']
            }
            if stats['error_rate'] >= thresholds['error_rate']:
                status = 'critical'
        
        # Check system resources
        if 'avg_memory_usage' in stats:
            components['memory'] = {
                'status': 'healthy' if stats['avg_memory_usage'] < thresholds['memory_usage'] else 'warning',
                'value': stats['avg_memory_usage']
            }
            if stats['avg_memory_usage'] >= thresholds['memory_usage']:
                status = 'warning'
        
        if 'avg_cpu_usage' in stats:
            components['cpu'] = {
                'status': 'healthy' if stats['avg_cpu_usage'] < thresholds['cpu_usage'] else 'warning',
                'value': stats['avg_cpu_usage']
            }
            if stats['avg_cpu_usage'] >= thresholds['cpu_usage']:
                status = 'warning'
        
        return {
            'status': status,
            'components': components,
            'last_update': self.last_update,
            'metrics': stats
        }
    
    def _update_timestamp(self) -> None:
        """Update last update timestamp."""
        self.last_update = time.time()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.latencies.clear()
        self.memory_usages.clear()
        self.predictions.clear()
        self.errors.clear()
        for buffer in self.system_metrics.values():
            buffer.clear()
        self._update_timestamp() 