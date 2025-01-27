"""
Alert management and notification system.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Alert data structure."""
    id: str
    type: str
    severity: AlertSeverity
    message: str
    timestamp: float
    value: float
    threshold: float
    component: str
    metadata: Dict[str, Any]
    active: bool = True

class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.thresholds = {}
        self.active_alerts = {}
        self.alert_history = []
        self.alert_handlers = []
        
        # Default thresholds
        self.default_thresholds = {
            'latency': 100,  # ms
            'error_rate': 0.01,  # 1%
            'memory_usage': 0.9,  # 90%
            'cpu_usage': 0.8,  # 80%
            'disk_usage': 0.9,  # 90%
            'model_drift': 0.1  # 10%
        }
    
    def set_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Set alert thresholds.
        
        Args:
            thresholds: Dictionary of metric thresholds
        """
        self.thresholds.update(thresholds)
    
    def add_alert_handler(self, handler: callable) -> None:
        """Add alert notification handler.
        
        Args:
            handler: Callback function for alert notifications
        """
        self.alert_handlers.append(handler)
    
    def check_metric(
        self,
        metric_name: str,
        value: float,
        component: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Check metric against thresholds and generate alerts.
        
        Args:
            metric_name: Name of the metric
            value: Current metric value
            component: Component name
            metadata: Additional alert metadata
        """
        threshold = self.thresholds.get(
            metric_name,
            self.default_thresholds.get(metric_name)
        )
        
        if threshold is None:
            return
        
        alert_id = f"{component}_{metric_name}"
        
        if value >= threshold:
            # Determine severity
            severity = self._get_severity(metric_name, value, threshold)
            
            # Create alert if not already active
            if alert_id not in self.active_alerts:
                alert = Alert(
                    id=alert_id,
                    type=metric_name,
                    severity=severity,
                    message=self._generate_message(
                        metric_name,
                        value,
                        threshold,
                        component
                    ),
                    timestamp=time.time(),
                    value=value,
                    threshold=threshold,
                    component=component,
                    metadata=metadata or {}
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                # Notify handlers
                self._notify_handlers(alert)
        else:
            # Clear alert if it exists
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.active = False
                del self.active_alerts[alert_id]
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts.
        
        Returns:
            List of active alert dictionaries
        """
        return [self._alert_to_dict(alert) for alert in self.active_alerts.values()]
    
    def get_alert_history(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get historical alerts within time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of historical alert dictionaries
        """
        alerts = self.alert_history
        
        if start_time is not None:
            alerts = [a for a in alerts if a.timestamp >= start_time]
        
        if end_time is not None:
            alerts = [a for a in alerts if a.timestamp <= end_time]
        
        return [self._alert_to_dict(alert) for alert in alerts]
    
    def clear_alert(self, alert_id: str) -> None:
        """Clear active alert.
        
        Args:
            alert_id: ID of alert to clear
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.active = False
            del self.active_alerts[alert_id]
    
    def _get_severity(
        self,
        metric_name: str,
        value: float,
        threshold: float
    ) -> AlertSeverity:
        """Determine alert severity based on value and threshold."""
        # Critical thresholds (2x default threshold)
        critical_thresholds = {
            'latency': 200,
            'error_rate': 0.02,
            'memory_usage': 0.95,
            'cpu_usage': 0.9,
            'disk_usage': 0.95,
            'model_drift': 0.2
        }
        
        critical = critical_thresholds.get(metric_name, threshold * 2)
        
        if value >= critical:
            return AlertSeverity.CRITICAL
        elif value >= threshold:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _generate_message(
        self,
        metric_name: str,
        value: float,
        threshold: float,
        component: str
    ) -> str:
        """Generate alert message."""
        return (
            f"{component}: {metric_name} value {value:.2f} "
            f"exceeds threshold {threshold:.2f}"
        )
    
    def _alert_to_dict(self, alert: Alert) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': alert.id,
            'type': alert.type,
            'severity': alert.severity.value,
            'message': alert.message,
            'timestamp': alert.timestamp,
            'value': alert.value,
            'threshold': alert.threshold,
            'component': alert.component,
            'metadata': alert.metadata,
            'active': alert.active
        }
    
    def _notify_handlers(self, alert: Alert) -> None:
        """Notify all registered handlers of new alert."""
        alert_dict = self._alert_to_dict(alert)
        for handler in self.alert_handlers:
            try:
                handler(alert_dict)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def reset(self) -> None:
        """Reset alert manager state."""
        self.active_alerts.clear()
        self.alert_history.clear() 