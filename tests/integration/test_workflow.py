import unittest
import torch
import numpy as np
from deepchain.core.strategy.base import RLStrategy
from deepchain.core.blockchain.validator import ZKValidator
from deepchain.core.edge.deployer import EdgeOptimizer

class TestCompleteWorkflow(unittest.TestCase):
    """
    Integration tests for complete DeepChain workflow.
    """
    
    def setUp(self):
        """Set up test environment."""
        self.strategy = RLStrategy()
        self.validator = ZKValidator(web3_provider="http://localhost:8545")
        self.deployer = EdgeOptimizer()
        
        # Generate test market data
        self.market_data = self._generate_market_data()
        
    def test_complete_workflow(self):
        """Test complete workflow from training to deployment."""
        # 1. Train strategy
        print("\nStep 1: Training Strategy")
        self.strategy.train(self.market_data)
        
        # Verify training results
        state = self._get_sample_state()
        action = self.strategy.predict(state)
        self.assertIsNotNone(action)
        
        # 2. Register and validate strategy
        print("\nStep 2: Validating Strategy")
        strategy_id = "test_integration_001"
        metadata = {
            'name': 'Integration Test Strategy',
            'version': '1.0.0',
            'type': 'reinforcement_learning'
        }
        
        # Register strategy
        tx_hash = self.validator.register_strategy(strategy_id, metadata)
        self.assertIsNotNone(tx_hash)
        
        # Generate and verify execution proof
        execution_data = {
            'timestamp': 1234567890,
            'state': state,
            'action': action,
            'reward': 0.1
        }
        
        proof = self.validator.generate_proof(strategy_id, execution_data)
        is_valid = self.validator.verify_execution(
            strategy_id,
            execution_data,
            proof
        )
        self.assertTrue(is_valid)
        
        # 3. Deploy optimized model
        print("\nStep 3: Deploying Strategy")
        # Compress model
        compressed_model = self.deployer.compress_model(
            self.strategy.policy,
            target_size=1024 * 1024  # 1MB target
        )
        
        # Export model
        self.deployer.export_model(
            compressed_model,
            format="onnx",
            path="./test_model.onnx"
        )
        
        # Validate deployed model
        test_data = {
            'inputs': np.random.randn(100, 10),
            'targets': np.random.randn(100, 3)
        }
        metrics = self.deployer.validate_performance(compressed_model, test_data)
        
        self.assertIn('mse', metrics)
        self.assertIn('latency', metrics)
        self.assertLess(metrics['latency'], 0.1)  # Latency under 100ms
        
    def test_error_recovery(self):
        """Test system's ability to handle and recover from errors."""
        # 1. Test invalid model training
        with self.assertRaises(Exception):
            self.strategy.train(None)  # Invalid training data
        
        # 2. Test invalid strategy validation
        with self.assertRaises(Exception):
            self.validator.validate_strategy(
                "nonexistent_id",
                {'invalid': 'proof'}
            )
        
        # 3. Test invalid model deployment
        with self.assertRaises(Exception):
            self.deployer.export_model(
                None,
                "invalid_format",
                "invalid_path"
            )
    
    def test_concurrent_operations(self):
        """Test system's ability to handle concurrent operations."""
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        def validate_operation():
            state = self._get_sample_state()
            action = self.strategy.predict(state)
            return action is not None
        
        # Run concurrent predictions
        n_concurrent = 10
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            results = list(executor.map(
                lambda _: validate_operation(),
                range(n_concurrent)
            ))
        
        end_time = time.time()
        
        # Verify results
        self.assertTrue(all(results))
        self.assertLess(end_time - start_time, 5)  # Should complete within 5 seconds
    
    def _generate_market_data(self):
        """Generate synthetic market data for testing."""
        return {
            'states': np.random.randn(1000, 10),
            'actions': np.random.randint(0, 3, 1000),
            'rewards': np.random.randn(1000),
            'next_states': np.random.randn(1000, 10)
        }
    
    def _get_sample_state(self):
        """Get a sample market state."""
        return {
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

if __name__ == '__main__':
    unittest.main() 