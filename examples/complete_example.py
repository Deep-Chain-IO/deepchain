"""
Complete example demonstrating the entire DeepChain workflow.
"""

import logging
import time
from typing import Dict, Any
import numpy as np
import pandas as pd
from deepchain.core.data.loader import DataLoader
from deepchain.core.data.processor import DataProcessor
from examples.advanced_strategy import AdvancedStrategy
from deepchain.core.blockchain.validator import ZKValidator
from deepchain.core.edge.deployer import EdgeOptimizer
from deepchain.core.monitoring.metrics import MetricsCollector
from deepchain.core.monitoring.alerts import AlertManager
from deepchain.core.monitoring.performance import PerformanceTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteExample:
    """Complete example of DeepChain usage."""
    
    def __init__(self):
        """Initialize components."""
        # Initialize data components
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        
        # Initialize strategy
        self.strategy = AdvancedStrategy(
            input_dim=10,
            hidden_dims=[64, 32]
        )
        
        # Initialize blockchain validator
        self.validator = ZKValidator(
            web3_provider="http://localhost:8545"
        )
        
        # Initialize edge deployer
        self.deployer = EdgeOptimizer()
        
        # Initialize monitoring
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        self.performance = PerformanceTracker()
        
        # Configure monitoring
        self._setup_monitoring()
    
    def run(self):
        """Run complete workflow."""
        try:
            # 1. Load and process data
            logger.info("Step 1: Loading and processing data...")
            data = self._prepare_data()
            
            # 2. Train strategy
            logger.info("Step 2: Training strategy...")
            self._train_strategy(data)
            
            # 3. Validate strategy
            logger.info("Step 3: Validating strategy...")
            self._validate_strategy()
            
            # 4. Deploy strategy
            logger.info("Step 4: Deploying strategy...")
            self._deploy_strategy()
            
            # 5. Run simulation
            logger.info("Step 5: Running simulation...")
            self._run_simulation(data)
            
            logger.info("Workflow completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            raise
    
    def _prepare_data(self) -> Dict[str, np.ndarray]:
        """Prepare training data."""
        # Load historical data
        raw_data = self.data_loader.load_csv("historical_data.csv")
        
        # Configure preprocessing
        self.data_processor.add_step('fill_missing', method='forward')
        self.data_processor.add_step('normalize', method='zscore')
        self.data_processor.add_step(
            'add_technical_indicators',
            indicators=['MA', 'RSI', 'MACD', 'BBANDS']
        )
        
        # Process data
        processed_data = self.data_processor.process(raw_data)
        
        # Convert to training format
        states = []
        actions = []
        rewards = []
        next_states = []
        
        for i in range(len(processed_data) - 1):
            state = processed_data.iloc[i]
            next_state = processed_data.iloc[i + 1]
            
            # Generate synthetic actions and rewards for demonstration
            action = np.random.randint(3)
            reward = self.strategy.get_reward(
                state.to_dict(),
                action
            )
            
            states.append(state.values)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state.values)
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states)
        }
    
    def _train_strategy(self, data: Dict[str, np.ndarray]):
        """Train strategy with monitoring."""
        # Set performance baseline
        self.performance.set_baseline({
            'latency': 50,  # ms
            'throughput': 100,  # predictions/second
            'error_rate': 0.01
        })
        
        # Train strategy
        self.strategy.train(data)
        
        # Save trained model
        self.strategy.save("trained_model.pt")
    
    def _validate_strategy(self):
        """Validate strategy on blockchain."""
        # Register strategy
        strategy_id = "example_strategy_001"
        metadata = {
            'name': 'Example Strategy',
            'version': '1.0.0',
            'type': 'reinforcement_learning',
            'input_dim': self.strategy.input_dim,
            'output_dim': self.strategy.output_dim
        }
        
        tx_hash = self.validator.register_strategy(
            strategy_id,
            metadata
        )
        logger.info(f"Strategy registered with tx_hash: {tx_hash}")
        
        # Generate and verify execution proof
        state = {
            'price': 100.0,
            'volume': 1000000,
            'high': 105.0,
            'low': 95.0,
            'ma_5': 101.0,
            'ma_10': 100.5,
            'rsi': 55.0,
            'macd': 0.5,
            'bollinger_upper': 110.0,
            'bollinger_lower': 90.0
        }
        
        action = self.strategy.predict(state)
        
        execution_data = {
            'timestamp': int(time.time()),
            'state': state,
            'action': action,
            'reward': self.strategy.get_reward(state, action)
        }
        
        proof = self.validator.generate_proof(
            strategy_id,
            execution_data
        )
        
        is_valid = self.validator.verify_execution(
            strategy_id,
            execution_data,
            proof
        )
        
        if not is_valid:
            raise ValueError("Strategy validation failed")
    
    def _deploy_strategy(self):
        """Deploy strategy to edge."""
        # Compress model
        compressed_model = self.deployer.compress_model(
            self.strategy.policy_net,
            target_size=1024 * 1024  # 1MB target
        )
        
        # Export model
        self.deployer.export_model(
            compressed_model,
            format="onnx",
            path="./deployed_model.onnx"
        )
        
        # Validate deployment
        test_data = {
            'inputs': np.random.randn(100, self.strategy.input_dim),
            'targets': np.random.randint(0, 3, 100)
        }
        
        metrics = self.deployer.validate_performance(
            compressed_model,
            test_data
        )
        
        logger.info(f"Deployment metrics: {metrics}")
    
    def _run_simulation(self, data: Dict[str, np.ndarray]):
        """Run trading simulation."""
        n_steps = len(data['states'])
        total_reward = 0.0
        
        for i in range(n_steps):
            # Get current state
            state = {
                f'feature_{j}': data['states'][i, j]
                for j in range(self.strategy.input_dim)
            }
            
            # Record start time
            start_time = time.time()
            
            # Get action
            action = self.strategy.predict(state)
            
            # Calculate reward
            reward = self.strategy.get_reward(state, action)
            total_reward += reward
            
            # Record metrics
            latency = (time.time() - start_time) * 1000
            self.metrics.record_latency(latency)
            self.metrics.record_prediction(action)
            
            # Check system health
            if (i + 1) % 100 == 0:
                self._check_system_health()
        
        logger.info(f"Simulation completed with total reward: {total_reward:.2f}")
    
    def _setup_monitoring(self):
        """Setup monitoring system."""
        # Set alert thresholds
        self.alerts.set_thresholds({
            'latency': 100,  # ms
            'error_rate': 0.05,  # 5%
            'memory_usage': 0.8,  # 80%
            'cpu_usage': 0.7  # 70%
        })
        
        # Add alert handler
        self.alerts.add_alert_handler(self._handle_alert)
    
    def _check_system_health(self):
        """Check system health status."""
        # Get current metrics
        stats = self.metrics.get_statistics()
        health = self.metrics.get_system_health()
        performance = self.performance.get_performance_report()
        
        # Log status
        logger.info("\n=== System Status ===")
        logger.info(f"Health: {health['status']}")
        logger.info(f"Average Latency: {stats.get('avg_latency', 0):.2f}ms")
        logger.info(f"Performance Score: {performance['performance_score']:.1f}")
    
    def _handle_alert(self, alert: Dict[str, Any]):
        """Handle system alerts."""
        logger.warning(
            f"Alert: {alert['message']} "
            f"(Severity: {alert['severity']})"
        )

def main():
    """Run complete example."""
    example = CompleteExample()
    example.run()

if __name__ == '__main__':
    main() 