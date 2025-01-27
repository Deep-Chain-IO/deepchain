"""
Performance benchmark tests for DeepChain components.
"""

import unittest
import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from deepchain.core.strategy.base import RLStrategy
from deepchain.core.blockchain.validator import ZKValidator
from deepchain.core.edge.deployer import EdgeOptimizer
from deepchain.core.monitoring.metrics import MetricsCollector

class PerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark test suite."""
    
    def setUp(self):
        """Set up test environment."""
        # Initialize components
        self.strategy = RLStrategy()
        self.validator = ZKValidator()
        self.deployer = EdgeOptimizer()
        self.metrics = MetricsCollector()
        
        # Create sample data
        self.sample_size = 10000
        self.input_dim = 10
        self.sample_data = {
            'states': np.random.randn(self.sample_size, self.input_dim),
            'actions': np.random.randint(0, 3, self.sample_size),
            'rewards': np.random.randn(self.sample_size),
            'next_states': np.random.randn(self.sample_size, self.input_dim)
        }
        
        # Create sample model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3)
        )
    
    def test_training_performance(self):
        """Benchmark training performance."""
        # Measure training time
        start_time = time.time()
        self.strategy.train(self.sample_data)
        training_time = time.time() - start_time
        
        # Measure memory usage
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # Log results
        print(f"\nTraining Performance:")
        print(f"Training Time: {training_time:.2f}s")
        print(f"Memory Usage: {memory_usage:.2f}MB")
        print(f"Samples/Second: {self.sample_size/training_time:.2f}")
        
        # Assert performance requirements
        self.assertLess(training_time, 60)  # Should train within 60 seconds
        self.assertLess(memory_usage, 1024)  # Should use less than 1GB RAM
    
    def test_inference_latency(self):
        """Benchmark inference latency."""
        # Prepare test data
        test_state = {
            f'feature_{i}': value 
            for i, value in enumerate(np.random.randn(self.input_dim))
        }
        
        # Warm up
        for _ in range(100):
            self.strategy.predict(test_state)
        
        # Measure latency
        n_predictions = 1000
        latencies = []
        
        for _ in range(n_predictions):
            start_time = time.time()
            self.strategy.predict(test_state)
            latencies.append((time.time() - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Log results
        print(f"\nInference Latency:")
        print(f"Average: {avg_latency:.2f}ms")
        print(f"P95: {p95_latency:.2f}ms")
        print(f"P99: {p99_latency:.2f}ms")
        
        # Assert latency requirements
        self.assertLess(avg_latency, 10)  # Average should be under 10ms
        self.assertLess(p99_latency, 50)  # P99 should be under 50ms
    
    def test_concurrent_validation(self):
        """Benchmark concurrent validation performance."""
        # Prepare validation data
        strategy_id = "test_strategy"
        execution_data = {
            'timestamp': int(time.time()),
            'state': {f'feature_{i}': 0 for i in range(self.input_dim)},
            'action': 0,
            'reward': 0
        }
        
        # Register strategy
        self.validator.register_strategy(
            strategy_id,
            {'type': 'test', 'version': '1.0.0'}
        )
        
        # Test concurrent validation
        n_concurrent = 100
        n_validations = 1000
        
        def validate_execution():
            proof = self.validator.generate_proof(strategy_id, execution_data)
            return self.validator.verify_execution(
                strategy_id,
                execution_data,
                proof
            )
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            results = list(executor.map(
                lambda _: validate_execution(),
                range(n_validations)
            ))
        total_time = time.time() - start_time
        
        # Calculate statistics
        success_rate = sum(results) / len(results)
        throughput = n_validations / total_time
        
        # Log results
        print(f"\nConcurrent Validation Performance:")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Throughput: {throughput:.2f} validations/second")
        
        # Assert performance requirements
        self.assertGreater(success_rate, 0.99)  # 99% success rate
        self.assertGreater(throughput, 100)  # At least 100 validations/second
    
    def test_model_compression(self):
        """Benchmark model compression performance."""
        # Set compression targets
        target_size = 1024 * 1024  # 1MB
        
        # Measure original model size
        original_size = self._get_model_size(self.model)
        
        # Compress model
        start_time = time.time()
        compressed_model = self.deployer.compress_model(
            self.model,
            target_size=target_size
        )
        compression_time = time.time() - start_time
        
        # Measure compressed size
        compressed_size = self._get_model_size(compressed_model)
        
        # Measure inference latency
        input_tensor = torch.randn(1, self.input_dim)
        latencies = []
        
        for _ in range(1000):
            start_time = time.time()
            with torch.no_grad():
                compressed_model(input_tensor)
            latencies.append((time.time() - start_time) * 1000)
        
        avg_latency = np.mean(latencies)
        
        # Log results
        print(f"\nModel Compression Performance:")
        print(f"Original Size: {original_size/1024:.2f}KB")
        print(f"Compressed Size: {compressed_size/1024:.2f}KB")
        print(f"Compression Ratio: {original_size/compressed_size:.2f}x")
        print(f"Compression Time: {compression_time:.2f}s")
        print(f"Inference Latency: {avg_latency:.2f}ms")
        
        # Assert compression requirements
        self.assertLess(compressed_size, target_size)
        self.assertLess(avg_latency, 5)  # Under 5ms inference
    
    def _get_model_size(self, model):
        """Get PyTorch model size in bytes."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size
    
    def test_monitoring_overhead(self):
        """Benchmark monitoring system overhead."""
        n_operations = 10000
        
        # Measure baseline latency
        start_time = time.time()
        for _ in range(n_operations):
            pass
        baseline_time = time.time() - start_time
        
        # Measure latency with monitoring
        start_time = time.time()
        for _ in range(n_operations):
            self.metrics.record_latency(1.0)
            self.metrics.record_prediction(0)
            self.metrics.record_error(0.1)
        monitored_time = time.time() - start_time
        
        # Calculate overhead
        overhead = (monitored_time - baseline_time) / n_operations * 1000000  # μs
        
        # Log results
        print(f"\nMonitoring Overhead:")
        print(f"Baseline Time: {baseline_time:.6f}s")
        print(f"Monitored Time: {monitored_time:.6f}s")
        print(f"Per-operation Overhead: {overhead:.2f}μs")
        
        # Assert overhead requirements
        self.assertLess(overhead, 10)  # Less than 10μs per operation

if __name__ == '__main__':
    unittest.main() 