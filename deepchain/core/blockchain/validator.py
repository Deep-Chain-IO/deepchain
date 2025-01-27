from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseValidator(ABC):
    """
    Base class for blockchain-based strategy validators.
    Defines interface for validating and verifying trading strategies on blockchain.
    """

    @abstractmethod
    def validate_strategy(self, strategy_id: str, proof: Dict[str, Any]) -> bool:
        """
        Validate a trading strategy using zero-knowledge proofs.
        
        Args:
            strategy_id: Unique identifier for the strategy
            proof: Zero-knowledge proof data
            
        Returns:
            Boolean indicating whether the strategy is valid
        """
        pass

    @abstractmethod
    def register_strategy(self, strategy_id: str, metadata: Dict[str, Any]) -> str:
        """
        Register a new strategy on the blockchain.
        
        Args:
            strategy_id: Unique identifier for the strategy
            metadata: Strategy metadata
            
        Returns:
            Transaction hash or identifier
        """
        pass

    @abstractmethod
    def verify_execution(self, 
                        strategy_id: str, 
                        execution_data: Dict[str, Any],
                        proof: Optional[Dict[str, Any]] = None) -> bool:
        """
        Verify the execution of a strategy on the blockchain.
        
        Args:
            strategy_id: Unique identifier for the strategy
            execution_data: Data about strategy execution
            proof: Optional zero-knowledge proof
            
        Returns:
            Boolean indicating whether the execution is valid
        """
        pass

class ZKValidator(BaseValidator):
    """
    Zero-knowledge proof based validator implementation.
    """

    @abstractmethod
    def generate_proof(self, strategy_id: str, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate zero-knowledge proof for strategy execution.
        
        Args:
            strategy_id: Unique identifier for the strategy
            execution_data: Data about strategy execution
            
        Returns:
            Generated proof data
        """
        pass

    @abstractmethod
    def verify_proof(self, proof: Dict[str, Any]) -> bool:
        """
        Verify a zero-knowledge proof.
        
        Args:
            proof: The proof to verify
            
        Returns:
            Boolean indicating whether the proof is valid
        """
        pass 