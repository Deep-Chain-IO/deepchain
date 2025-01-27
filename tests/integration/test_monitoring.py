import unittest
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from deepchain.core.monitoring.metrics import MetricsCollector
from deepchain.core.monitoring.alerts import AlertManager
from deepchain.core.monitoring.performance import PerformanceTracker

class TestMonitoringSystem(unittest.TestCase):
    """
    Integration tests for the monitoring system.
    """
    
    def setUp(self):
        """Set up test environment."""
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        self.performance = PerformanceTracker()
        
        # Configure test thresholds
        self.thresholds = {
            'latency': 100,  # ms
            'error_rate': 0.01,  # 1%
            'memory_usage': 1024 * 1024 * 1024,  # 1GB
            'model_drift': 0.1  # 10% drift
        }
        
    def test_metrics_collection(self):
        """Test metrics collection and aggregation."""
        # Generate test metrics
        for _ in range(100):
            self.metrics.record_latency(np.random.randint(10, 200))
            self.metrics.record_memory_usage(np.random.randint(100_000, 1_000_000))
            self.metrics.record_prediction(np.random.random())
        
        # Get aggregated metrics
        stats = self.metrics.get_statistics()
        
        # Verify metrics
        self.assertIn('avg_latency', stats)
        self.assertIn('max_memory_usage', stats)
        self.assertIn('prediction_count', stats)
        self.assertEqual(stats['prediction_count'], 100)
    
    def test_alert_system(self):
        """Test alert generation and handling."""
        # Configure alert thresholds
        self.alerts.set_thresholds(self.thresholds)
        
        # Generate test alerts
        high_latency = 200  # ms
        high_memory = 2 * 1024 * 1024 * 1024  # 2GB
        
        # Record metrics that should trigger alerts
        self.metrics.record_latency(high_latency)
        self.metrics.record_memory_usage(high_memory)
        
        # Check alerts
        active_alerts = self.alerts.get_active_alerts()
        self.assertGreater(len(active_alerts), 0)
        
        # Verify alert contents
        latency_alert = next(
            (a for a in active_alerts if a['type'] == 'high_latency'),
            None
        )
        self.assertIsNotNone(latency_alert)
        self.assertEqual(latency_alert['severity'], 'critical')
    
    def test_performance_tracking(self):
        """Test performance tracking over time."""
        # Record initial performance baseline
        baseline = {
            'latency': 50,
            'throughput': 1000,
            'error_rate': 0.005
        }
        self.performance.set_baseline(baseline)
        
        # Simulate performance changes
        for _ in range(10):
            metrics = {
                'latency': 50 + np.random.randint(-10, 10),
                'throughput': 1000 + np.random.randint(-100, 100),
                'error_rate': 0.005 + np.random.random() * 0.001
            }
            self.performance.record_metrics(metrics)
        
        # Get performance report
        report = self.performance.get_performance_report()
        
        # Verify report contents
        self.assertIn('latency_trend', report)
        self.assertIn('throughput_trend', report)
        self.assertIn('error_rate_trend', report)
        self.assertIn('model_drift', report)
    
    def test_concurrent_monitoring(self):
        """Test monitoring system under concurrent load."""
        def simulate_load():
            # Simulate a busy strategy
            latency = np.random.randint(10, 100)
            memory = np.random.randint(100_000, 1_000_000)
            prediction = np.random.random()
            
            self.metrics.record_latency(latency)
            self.metrics.record_memory_usage(memory)
            self.metrics.record_prediction(prediction)
            
            return True
        
        # Run concurrent operations
        n_concurrent = 100
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(
                lambda _: simulate_load(),
                range(n_concurrent)
            ))
        
        # Verify results
        self.assertTrue(all(results))
        
        # Check metrics were recorded correctly
        stats = self.metrics.get_statistics()
        self.assertEqual(stats['prediction_count'], n_concurrent)
    
    def test_system_health_check(self):
        """Test system health monitoring."""
        # Record system metrics
        self.metrics.record_system_metrics({
            'cpu_usage': 0.5,
            'memory_usage': 0.7,
            'disk_usage': 0.3,
            'network_io': 1000
        })
        
        # Get health status
        health = self.metrics.get_system_health()
        
        # Verify health check
        self.assertIn('status', health)
        self.assertIn('components', health)
        self.assertIn('last_update', health)
        
        # Check component status
        components = health['components']
        self.assertIn('cpu', components)
        self.assertIn('memory', components)
        self.assertIn('disk', components)
        self.assertIn('network', components)

if __name__ == '__main__':
    unittest.main() 