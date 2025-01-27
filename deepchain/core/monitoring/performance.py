"""
Performance tracking and analysis system.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Tracks and analyzes system performance metrics."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize performance tracker.
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        
        # Initialize metric buffers
        self.metrics = {
            'latency': deque(maxlen=window_size),
            'throughput': deque(maxlen=window_size),
            'error_rate': deque(maxlen=window_size),
            'memory_usage': deque(maxlen=window_size),
            'cpu_usage': deque(maxlen=window_size)
        }
        
        # Performance baseline
        self.baseline = {}
        
        # Drift detection
        self.drift_thresholds = {
            'latency': 0.2,  # 20% change
            'throughput': 0.2,
            'error_rate': 0.1,
            'memory_usage': 0.3,
            'cpu_usage': 0.3
        }
        
        # Initialize timestamps
        self.last_update = time.time()
    
    def set_baseline(self, baseline: Dict[str, float]) -> None:
        """Set performance baseline.
        
        Args:
            baseline: Dictionary of baseline metrics
        """
        self.baseline = baseline.copy()
    
    def record_metrics(self, metrics: Dict[str, float]) -> None:
        """Record performance metrics.
        
        Args:
            metrics: Dictionary of current metrics
        """
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        self.last_update = time.time()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report.
        
        Returns:
            Dictionary containing performance analysis
        """
        report = {
            'current_metrics': self._get_current_metrics(),
            'trends': self._analyze_trends(),
            'model_drift': self._detect_drift(),
            'anomalies': self._detect_anomalies(),
            'statistics': self._compute_statistics(),
            'last_update': self.last_update
        }
        
        # Add performance score
        report['performance_score'] = self._compute_performance_score(report)
        
        return report
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        current = {}
        for key, values in self.metrics.items():
            if values:
                current[key] = values[-1]
        return current
    
    def _analyze_trends(self) -> Dict[str, Dict[str, float]]:
        """Analyze metric trends."""
        trends = {}
        for key, values in self.metrics.items():
            if len(values) > 1:
                values_array = np.array(values)
                x = np.arange(len(values_array))
                
                # Compute linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x,
                    values_array
                )
                
                trends[key] = {
                    'slope': slope,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'trend': 'increasing' if slope > 0 else 'decreasing'
                }
        return trends
    
    def _detect_drift(self) -> Dict[str, Dict[str, Any]]:
        """Detect model drift from baseline."""
        drift = {}
        for key, values in self.metrics.items():
            if values and key in self.baseline:
                baseline_value = self.baseline[key]
                current_value = np.mean(values)
                
                relative_change = abs(current_value - baseline_value) / baseline_value
                threshold = self.drift_thresholds.get(key, 0.2)
                
                drift[key] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'change': relative_change,
                    'threshold': threshold,
                    'significant': relative_change > threshold
                }
        return drift
    
    def _detect_anomalies(self) -> Dict[str, List[Dict[str, Any]]]:
        """Detect metric anomalies using statistical methods."""
        anomalies = {}
        for key, values in self.metrics.items():
            if len(values) > 30:  # Need enough data for statistical significance
                values_array = np.array(values)
                mean = np.mean(values_array)
                std = np.std(values_array)
                
                # Detect values outside 3 standard deviations
                z_scores = np.abs((values_array - mean) / std)
                anomaly_indices = np.where(z_scores > 3)[0]
                
                if len(anomaly_indices) > 0:
                    anomalies[key] = [
                        {
                            'index': int(i),
                            'value': float(values_array[i]),
                            'z_score': float(z_scores[i])
                        }
                        for i in anomaly_indices
                    ]
        return anomalies
    
    def _compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute detailed statistics for each metric."""
        stats = {}
        for key, values in self.metrics.items():
            if values:
                values_array = np.array(values)
                stats[key] = {
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'min': np.min(values_array),
                    'max': np.max(values_array),
                    'p25': np.percentile(values_array, 25),
                    'p50': np.percentile(values_array, 50),
                    'p75': np.percentile(values_array, 75),
                    'p95': np.percentile(values_array, 95),
                    'p99': np.percentile(values_array, 99)
                }
        return stats
    
    def _compute_performance_score(self, report: Dict[str, Any]) -> float:
        """Compute overall performance score."""
        score = 100.0
        
        # Penalize for drift
        drift = report['model_drift']
        for key, info in drift.items():
            if info['significant']:
                score -= 10 * (info['change'] / info['threshold'])
        
        # Penalize for anomalies
        anomalies = report['anomalies']
        for key, anomaly_list in anomalies.items():
            score -= len(anomaly_list) * 5
        
        # Penalize for negative trends
        trends = report['trends']
        for key, trend_info in trends.items():
            if trend_info['trend'] == 'increasing' and key in ['latency', 'error_rate']:
                score -= 5 * abs(trend_info['slope'])
            elif trend_info['trend'] == 'decreasing' and key == 'throughput':
                score -= 5 * abs(trend_info['slope'])
        
        return max(0.0, min(100.0, score))
    
    def reset(self) -> None:
        """Reset performance tracker state."""
        for buffer in self.metrics.values():
            buffer.clear()
        self.baseline.clear()
        self.last_update = time.time() 