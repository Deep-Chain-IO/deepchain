import unittest
import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil
import os

from examples.simple_strategy import SimpleRLStrategy
from examples.simple_validator import SimpleValidator
from examples.simple_deployer import SimpleDeployer

class PerformanceTest(unittest.TestCase):
    """
    Performance tests for the DeepChain framework.
    """
    
    def setUp(self):
        """
        Set up test environment before each test case.
        """
        self.strategy = SimpleRLStrategy()
        self.validator = SimpleValidator()
        self.deployer = SimpleDeployer()
        
        # Generate test data
        self.test_data = {
            'states': np.random.randn(1000, 10),
            'actions': np.random.randint(0, 3, 1000),
            'rewards': np.random.randn(1000)
        }
    
    def test_training_performance(self):
        """
        Test training performance.
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Train model
        self.strategy.train(self.test_data, n_epochs=100)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        training_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"\nTraining Performance:")
        print(f"Time taken: {training_time:.2f} seconds")
        print(f"Memory used: {memory_used / 1024 / 1024:.2f} MB")
        
        # Assert reasonable performance
        self.assertLess(training_time, 60)  # Should train within 60 seconds
        self.assertLess(memory_used / 1024 / 1024, 1000)  # Should use less than 1GB
    
    def test_inference_latency(self):
        """
        Test inference latency.
        """
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
        
        latencies = []
        n_iterations = 1000
        
        # Warm up
        for _ in range(10):
            _ = self.strategy.predict(state)
        
        # Measure latency
        for _ in range(n_iterations):
            start_time = time.time()
            _ = self.strategy.predict(state)
            latencies.append(time.time() - start_time)
        
        avg_latency = np.mean(latencies) * 1000  # Convert to ms
        p95_latency = np.percentile(latencies, 95) * 1000
        p99_latency = np.percentile(latencies, 99) * 1000
        
        print(f"\nInference Latency:")
        print(f"Average: {avg_latency:.2f}ms")
        print(f"P95: {p95_latency:.2f}ms")
        print(f"P99: {p99_latency:.2f}ms")
        
        # Assert reasonable latency
        self.assertLess(avg_latency, 10)  # Average latency should be under 10ms
        self.assertLess(p99_latency, 50)  # P99 latency should be under 50ms
    
    def test_concurrent_validation(self):
        """
        Test concurrent validation performance.
        """
        n_concurrent = 100
        strategy_id = "test_strategy"
        
        # Register strategy
        self.validator.register_strategy(strategy_id, {'type': 'test'})
        
        execution_data = {
            'timestamp': int(time.time()),
            'actions': ['buy', 'hold', 'sell']
        }
        
        def validate_execution():
            proof = self.validator.generate_proof(strategy_id, execution_data)
            return self.validator.verify_execution(strategy_id, execution_data, proof)
        
        start_time = time.time()
        
        # Run concurrent validations
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda _: validate_execution(), range(n_concurrent)))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        success_rate = sum(results) / len(results) * 100
        throughput = n_concurrent / total_time
        
        print(f"\nConcurrent Validation Performance:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Throughput: {throughput:.2f} validations/second")
        print(f"Success rate: {success_rate:.2f}%")
        
        # Assert reasonable performance
        self.assertGreater(throughput, 10)  # Should handle at least 10 validations/second
        self.assertEqual(success_rate, 100)  # All validations should succeed
    
    def test_model_compression_ratio(self):
        """
        Test model compression performance.
        """
        original_size = self._get_model_size(self.strategy.policy)
        
        start_time = time.time()
        compressed_model = self.deployer.compress_model(
            self.strategy.policy,
            target_size=original_size // 4  # Target 75% compression
        )
        compression_time = time.time() - start_time
        
        compressed_size = self._get_model_size(compressed_model)
        compression_ratio = original_size / compressed_size
        
        print(f"\nModel Compression Performance:")
        print(f"Original size: {original_size / 1024:.2f}KB")
        print(f"Compressed size: {compressed_size / 1024:.2f}KB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Compression time: {compression_time:.2f} seconds")
        
        # Assert reasonable compression
        self.assertGreater(compression_ratio, 2)  # Should achieve at least 2x compression
        self.assertLess(compression_time, 30)  # Should compress within 30 seconds
    
    def _get_memory_usage(self) -> int:
        """Helper function to get current memory usage."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    def _get_model_size(self, model: torch.nn.Module) -> int:
        """Helper function to get model size in bytes."""
        return sum(p.numel() * p.element_size() for p in model.parameters())

if __name__ == '__main__':
    unittest.main() 