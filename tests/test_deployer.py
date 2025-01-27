import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import os

from deepchain.core.edge.deployer import EdgeOptimizer
from deepchain.core.exceptions import DeploymentError

class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.input_dim = 10
    
    def forward(self, x):
        return self.network(x)

class TestEdgeDeployer(unittest.TestCase):
    """
    Unit tests for EdgeOptimizer implementation.
    """
    
    def setUp(self):
        """
        Set up test environment before each test case.
        """
        self.deployer = EdgeOptimizer()
        self.model = SimpleModel()
        self.test_input = torch.randn(1, 10)
        
        # Create temporary directory for test artifacts
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """
        Clean up after tests.
        """
        # Remove temporary files
        for file in Path(self.test_dir).glob("*"):
            os.remove(file)
        os.rmdir(self.test_dir)
    
    def test_model_compression(self):
        """
        Test model compression functionality.
        """
        original_size = self._get_model_size(self.model)
        target_size = original_size // 2
        
        compressed_model = self.deployer.compress_model(
            self.model,
            target_size=target_size
        )
        
        compressed_size = self._get_model_size(compressed_model)
        self.assertLessEqual(compressed_size, target_size)
    
    def test_model_export_onnx(self):
        """
        Test ONNX model export.
        """
        export_path = os.path.join(self.test_dir, "model.onnx")
        self.deployer.export_model(self.model, "onnx", export_path)
        
        self.assertTrue(os.path.exists(export_path))
        self.assertGreater(os.path.getsize(export_path), 0)
    
    def test_model_export_torchscript(self):
        """
        Test TorchScript model export.
        """
        export_path = os.path.join(self.test_dir, "model.pt")
        self.deployer.export_model(self.model, "torchscript", export_path)
        
        self.assertTrue(os.path.exists(export_path))
        self.assertGreater(os.path.getsize(export_path), 0)
    
    def test_invalid_export_format(self):
        """
        Test handling of invalid export format.
        """
        with self.assertRaises(ValueError):
            self.deployer.export_model(
                self.model,
                "invalid_format",
                "model.invalid"
            )
    
    def test_performance_validation(self):
        """
        Test model performance validation.
        """
        test_data = {
            'inputs': np.random.randn(100, 10),
            'targets': np.random.randn(100, 3)
        }
        
        metrics = self.deployer.validate_performance(self.model, test_data)
        
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('latency', metrics)
        self.assertIn('model_size', metrics)
    
    def test_quantization(self):
        """
        Test model quantization.
        """
        quantized_model = self.deployer.quantize_model(self.model)
        
        # Verify model still works
        with torch.no_grad():
            output = quantized_model(self.test_input)
            self.assertEqual(output.shape, (1, 3))
    
    def test_pruning(self):
        """
        Test model pruning.
        """
        pruned_model = self.deployer.prune_model(self.model)
        
        # Verify model still works
        with torch.no_grad():
            output = pruned_model(self.test_input)
            self.assertEqual(output.shape, (1, 3))
    
    def test_benchmark(self):
        """
        Test model benchmarking.
        """
        device_specs = {
            'input_shape': (1, 10),
            'required_memory': 1024 * 1024,  # 1MB
            'required_compute': 1000000
        }
        
        results = self.deployer.benchmark_model(self.model, device_specs)
        
        self.assertIn('latency', results)
        self.assertIn('memory_usage', results)
        self.assertIn('device_compatibility', results)
    
    def test_compression_with_latency_target(self):
        """
        Test model compression with latency target.
        """
        compressed_model = self.deployer.compress_model(
            self.model,
            target_latency=0.1  # 100ms
        )
        
        # Measure actual latency
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            _ = compressed_model(self.test_input)
            end_time.record()
            
            torch.cuda.synchronize()
            latency = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            
            self.assertLessEqual(latency, 0.1)
    
    def _get_model_size(self, model: nn.Module) -> int:
        """Helper function to get model size in bytes."""
        return sum(p.numel() * p.element_size() for p in model.parameters())

if __name__ == '__main__':
    unittest.main() 