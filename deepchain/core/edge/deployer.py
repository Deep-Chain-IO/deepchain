from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseDeployer(ABC):
    """
    Base class for edge deployment functionality.
    Defines interface for model compression and edge deployment.
    """

    @abstractmethod
    def compress_model(self, 
                      model: Any,
                      target_size: Optional[int] = None,
                      target_latency: Optional[float] = None) -> Any:
        """
        Compress the model for edge deployment.
        
        Args:
            model: Original model to compress
            target_size: Target model size in bytes
            target_latency: Target inference latency in seconds
            
        Returns:
            Compressed model
        """
        pass

    @abstractmethod
    def export_model(self, model: Any, format: str, path: str) -> None:
        """
        Export model to specific format for edge deployment.
        
        Args:
            model: Model to export
            format: Target format (e.g., 'onnx', 'tflite')
            path: Export path
        """
        pass

    @abstractmethod
    def validate_performance(self, 
                           model: Any,
                           test_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate performance of compressed model.
        
        Args:
            model: Compressed model to validate
            test_data: Test data for validation
            
        Returns:
            Dictionary containing performance metrics
        """
        pass

class EdgeOptimizer(BaseDeployer):
    """
    Edge deployment optimizer implementation.
    """

    @abstractmethod
    def quantize_model(self, model: Any, **kwargs) -> Any:
        """
        Quantize model for edge deployment.
        
        Args:
            model: Model to quantize
            **kwargs: Additional quantization parameters
            
        Returns:
            Quantized model
        """
        pass

    @abstractmethod
    def prune_model(self, model: Any, **kwargs) -> Any:
        """
        Prune model for edge deployment.
        
        Args:
            model: Model to prune
            **kwargs: Additional pruning parameters
            
        Returns:
            Pruned model
        """
        pass

    @abstractmethod
    def benchmark_model(self, 
                       model: Any,
                       device_specs: Dict[str, Any]) -> Dict[str, float]:
        """
        Benchmark model performance on target edge device.
        
        Args:
            model: Model to benchmark
            device_specs: Target device specifications
            
        Returns:
            Dictionary containing benchmark metrics
        """
        pass 