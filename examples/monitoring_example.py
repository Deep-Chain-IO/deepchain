"""
Example of using the monitoring system.
"""

import time
import logging
from typing import Dict, Any
import numpy as np
from deepchain.core.monitoring.metrics import MetricsCollector
from deepchain.core.monitoring.alerts import AlertManager
from deepchain.core.monitoring.performance import PerformanceTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringExample:
    """Example class demonstrating monitoring system usage."""
    
    def __init__(self):
        """Initialize monitoring components."""
        # Initialize monitoring components
        self.metrics = MetricsCollector(window_size=1000)
        self.alerts = AlertManager()
        self.performance = PerformanceTracker(window_size=1000)
        
        # Configure alert handlers
        self.alerts.add_alert_handler(self._handle_alert)
        
        # Set custom thresholds
        self.alerts.set_thresholds({
            'latency': 50,  # ms
            'error_rate': 0.05,  # 5%
            'memory_usage': 0.8,  # 80%
            'cpu_usage': 0.7  # 70%
        })
        
        # Set performance baseline
        self.performance.set_baseline({
            'latency': 30,
            'throughput': 1000,
            'error_rate': 0.01,
            'memory_usage': 0.5,
            'cpu_usage': 0.4
        })
    
    def simulate_trading_system(self, n_iterations: int = 1000):
        """Simulate a trading system with monitoring.
        
        Args:
            n_iterations: Number of iterations to simulate
        """
        logger.info("Starting trading system simulation...")
        
        for i in range(n_iterations):
            # Simulate model prediction
            start_time = time.time()
            prediction = self._simulate_prediction()
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            # Record metrics
            self.metrics.record_latency(latency)
            self.metrics.record_prediction(prediction)
            
            # Simulate error calculation
            error = abs(prediction - 0.5)  # Assuming 0.5 is ground truth
            self.metrics.record_error(error)
            
            # Record system metrics
            system_metrics = self._get_system_metrics()
            self.metrics.record_system_metrics(system_metrics)
            
            # Check metrics against thresholds
            self.alerts.check_metric('latency', latency, 'model')
            self.alerts.check_metric('error_rate', error, 'model')
            
            for key, value in system_metrics.items():
                self.alerts.check_metric(key, value, 'system')
            
            # Record performance metrics
            self.performance.record_metrics({
                'latency': latency,
                'throughput': 1000 / (latency / 1000),  # Convert ms to seconds
                'error_rate': error,
                'memory_usage': system_metrics['memory_usage'],
                'cpu_usage': system_metrics['cpu_usage']
            })
            
            # Periodically log status
            if (i + 1) % 100 == 0:
                self._log_status()
            
            # Simulate some delay
            time.sleep(0.01)
    
    def _simulate_prediction(self) -> float:
        """Simulate model prediction."""
        # Add random latency
        time.sleep(np.random.exponential(0.02))
        
        # Generate prediction with some noise
        return np.random.normal(0.5, 0.1)
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        import psutil
        
        return {
            'cpu_usage': psutil.cpu_percent() / 100,
            'memory_usage': psutil.virtual_memory().percent / 100,
            'disk_usage': psutil.disk_usage('/').percent / 100,
            'network_io': sum(psutil.net_io_counters()[:2]) / 1e6  # Convert to MB
        }
    
    def _handle_alert(self, alert: Dict[str, Any]):
        """Handle generated alerts.
        
        Args:
            alert: Alert information dictionary
        """
        logger.warning(
            f"Alert: {alert['message']} "
            f"(Severity: {alert['severity']})"
        )
    
    def _log_status(self):
        """Log current system status."""
        # Get metrics
        stats = self.metrics.get_statistics()
        health = self.metrics.get_system_health()
        performance = self.performance.get_performance_report()
        
        # Log summary
        logger.info("\n=== System Status ===")
        logger.info(f"Health: {health['status']}")
        logger.info(f"Average Latency: {stats.get('avg_latency', 0):.2f}ms")
        logger.info(f"Error Rate: {stats.get('error_rate', 0):.2%}")
        logger.info(f"Performance Score: {performance['performance_score']:.1f}")
        
        # Log active alerts
        active_alerts = self.alerts.get_active_alerts()
        if active_alerts:
            logger.warning(f"Active Alerts: {len(active_alerts)}")
            for alert in active_alerts:
                logger.warning(f"- {alert['message']}")

def main():
    """Run monitoring example."""
    example = MonitoringExample()
    example.simulate_trading_system()

if __name__ == '__main__':
    main() 