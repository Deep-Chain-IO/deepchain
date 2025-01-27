from typing import Dict, Any, Optional
import hashlib
import json
from web3 import Web3

from deepchain.core.blockchain.validator import ZKValidator

class SimpleValidator(ZKValidator):
    """
    Simple example implementation of a blockchain validator.
    """
    def __init__(self, web3_provider: str = "http://localhost:8545"):
        """
        Initialize the validator with Web3 connection.
        
        Args:
            web3_provider: URL of the Web3 provider
        """
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.strategies = {}  # Simple in-memory storage
    
    def validate_strategy(self, strategy_id: str, proof: Dict[str, Any]) -> bool:
        """
        Validate a trading strategy using simple hash-based proof.
        """
        if strategy_id not in self.strategies:
            return False
        
        # Simple validation logic
        stored_hash = self.strategies[strategy_id].get('proof_hash')
        current_hash = self._compute_hash(proof['data'])
        
        return stored_hash == current_hash
    
    def register_strategy(self, strategy_id: str, metadata: Dict[str, Any]) -> str:
        """
        Register a new strategy with metadata.
        """
        # Create a simple hash of the metadata
        metadata_hash = self._compute_hash(metadata)
        
        # Store strategy information
        self.strategies[strategy_id] = {
            'metadata': metadata,
            'proof_hash': metadata_hash
        }
        
        # In a real implementation, this would be a blockchain transaction hash
        return metadata_hash
    
    def verify_execution(self,
                        strategy_id: str,
                        execution_data: Dict[str, Any],
                        proof: Optional[Dict[str, Any]] = None) -> bool:
        """
        Verify the execution of a strategy.
        """
        if not proof:
            return False
        
        # Verify the proof matches the execution data
        execution_hash = self._compute_hash(execution_data)
        return proof.get('execution_hash') == execution_hash
    
    def generate_proof(self, strategy_id: str, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a simple hash-based proof for strategy execution.
        """
        execution_hash = self._compute_hash(execution_data)
        
        # Create a simple proof structure
        proof = {
            'strategy_id': strategy_id,
            'execution_hash': execution_hash,
            'timestamp': self.w3.eth.get_block('latest')['timestamp']
        }
        
        return proof
    
    def verify_proof(self, proof: Dict[str, Any]) -> bool:
        """
        Verify a proof's authenticity.
        """
        # In a real implementation, this would verify the zero-knowledge proof
        # Here we just check if the proof has valid structure
        required_fields = {'strategy_id', 'execution_hash', 'timestamp'}
        return all(field in proof for field in required_fields)
    
    def _compute_hash(self, data: Any) -> str:
        """
        Compute a hash of the input data.
        """
        # Convert data to JSON string and compute SHA256 hash
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest() 