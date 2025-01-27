"""
Core exceptions for the DeepChain framework.
"""

class DeepChainError(Exception):
    """Base exception for all DeepChain errors."""
    pass

class ModelError(DeepChainError):
    """Errors related to model operations."""
    pass

class ValidationError(DeepChainError):
    """Errors related to validation operations."""
    pass

class BlockchainError(DeepChainError):
    """Errors related to blockchain operations."""
    pass

class APIError(DeepChainError):
    """Errors related to API operations."""
    pass

class DataError(DeepChainError):
    """Errors related to data operations."""
    pass

class ConfigurationError(DeepChainError):
    """Errors related to configuration."""
    pass

class DeploymentError(DeepChainError):
    """Errors related to model deployment."""
    pass

class SecurityError(DeepChainError):
    """Errors related to security operations."""
    pass

class ResourceError(DeepChainError):
    """Errors related to resource management."""
    pass

# Specific Exceptions
class ModelNotFoundError(ModelError):
    """Raised when a model cannot be found."""
    pass

class InvalidModelError(ModelError):
    """Raised when a model is invalid."""
    pass

class ValidationFailedError(ValidationError):
    """Raised when validation fails."""
    pass

class BlockchainConnectionError(BlockchainError):
    """Raised when blockchain connection fails."""
    pass

class APIKeyError(APIError):
    """Raised when there are issues with API keys."""
    pass

class DataSourceError(DataError):
    """Raised when there are issues with data sources."""
    pass

class InvalidConfigError(ConfigurationError):
    """Raised when configuration is invalid."""
    pass

class DeploymentFailedError(DeploymentError):
    """Raised when deployment fails."""
    pass

class SecurityBreachError(SecurityError):
    """Raised when security is compromised."""
    pass

class ResourceExhaustedError(ResourceError):
    """Raised when resources are exhausted."""
    pass 