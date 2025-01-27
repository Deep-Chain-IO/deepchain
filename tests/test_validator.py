import unittest
from unittest.mock import Mock, patch
import json
from deepchain.core.blockchain.validator import ZKValidator
from deepchain.core.exceptions import BlockchainConnectionError, ValidationFailedError

class TestZKValidator(unittest.TestCase):
    """
    Unit tests for ZKValidator implementation.
    """
    
    def setUp(self):
        """
        Set up test environment before each test case.
        """
        self.validator = ZKValidator(web3_provider="http://localhost:8545")
        self.strategy_id = "test_strategy_001"
        self.test_metadata = {
            'name': 'Test Strategy',
            'version': '0.1.0',
            'type': 'reinforcement_learning'
        }
    
    @patch('web3.Web3.HTTPProvider')
    def test_connection(self, mock_provider):
        """
        Test blockchain connection.
        """
        mock_provider.return_value = Mock()
        validator = ZKValidator(web3_provider="http://localhost:8545")
        self.assertIsNotNone(validator.w3)
    
    def test_register_strategy(self):
        """
        Test strategy registration.
        """
        tx_hash = self.validator.register_strategy(
            self.strategy_id,
            self.test_metadata
        )
        self.assertIsInstance(tx_hash, str)
        self.assertEqual(len(tx_hash), 64)  # Standard hash length
    
    def test_validate_strategy(self):
        """
        Test strategy validation.
        """
        # Register strategy first
        self.validator.register_strategy(self.strategy_id, self.test_metadata)
        
        # Test validation
        proof = {'data': self.test_metadata}
        is_valid = self.validator.validate_strategy(self.strategy_id, proof)
        self.assertTrue(is_valid)
    
    def test_validate_nonexistent_strategy(self):
        """
        Test validation of non-existent strategy.
        """
        proof = {'data': self.test_metadata}
        is_valid = self.validator.validate_strategy("nonexistent_id", proof)
        self.assertFalse(is_valid)
    
    def test_verify_execution(self):
        """
        Test execution verification.
        """
        execution_data = {
            'timestamp': 1234567890,
            'actions': ['buy', 'hold', 'sell'],
            'performance': 0.15
        }
        
        proof = self.validator.generate_proof(self.strategy_id, execution_data)
        is_valid = self.validator.verify_execution(
            self.strategy_id,
            execution_data,
            proof
        )
        self.assertTrue(is_valid)
    
    def test_verify_execution_without_proof(self):
        """
        Test execution verification without proof.
        """
        execution_data = {
            'timestamp': 1234567890,
            'actions': ['buy', 'hold', 'sell']
        }
        
        is_valid = self.validator.verify_execution(
            self.strategy_id,
            execution_data
        )
        self.assertFalse(is_valid)
    
    def test_generate_proof(self):
        """
        Test proof generation.
        """
        execution_data = {
            'timestamp': 1234567890,
            'actions': ['buy', 'hold', 'sell']
        }
        
        proof = self.validator.generate_proof(self.strategy_id, execution_data)
        
        self.assertIn('strategy_id', proof)
        self.assertIn('execution_hash', proof)
        self.assertIn('timestamp', proof)
    
    def test_verify_proof(self):
        """
        Test proof verification.
        """
        execution_data = {
            'timestamp': 1234567890,
            'actions': ['buy', 'hold', 'sell']
        }
        
        proof = self.validator.generate_proof(self.strategy_id, execution_data)
        is_valid = self.validator.verify_proof(proof)
        self.assertTrue(is_valid)
    
    def test_verify_invalid_proof(self):
        """
        Test verification of invalid proof.
        """
        invalid_proof = {
            'strategy_id': self.strategy_id
            # Missing required fields
        }
        
        is_valid = self.validator.verify_proof(invalid_proof)
        self.assertFalse(is_valid)
    
    @patch('web3.Web3.HTTPProvider')
    def test_connection_error(self, mock_provider):
        """
        Test handling of connection errors.
        """
        mock_provider.side_effect = Exception("Connection failed")
        
        with self.assertRaises(BlockchainConnectionError):
            ZKValidator(web3_provider="http://invalid:8545")
    
    def test_validation_error(self):
        """
        Test handling of validation errors.
        """
        invalid_metadata = "invalid"  # Should be a dictionary
        
        with self.assertRaises(ValidationFailedError):
            self.validator.register_strategy(self.strategy_id, invalid_metadata)

if __name__ == '__main__':
    unittest.main() 