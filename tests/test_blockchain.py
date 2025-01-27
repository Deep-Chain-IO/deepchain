"""
Unit tests for blockchain integrations.
"""

import unittest
from unittest.mock import Mock, patch
import json
from deepchain.core.blockchain.chains import (
    ChainFactory,
    SolanaChain,
    SolanaValidator
)

class TestSolanaIntegration(unittest.TestCase):
    """Test Solana blockchain integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.endpoint = "https://api.devnet.solana.com"  # Use devnet for testing
        self.chain = SolanaChain(endpoint=self.endpoint)
        self.validator = SolanaValidator(self.chain)
        
        # Mock Solana client
        self.mock_client = Mock()
        self.chain.client = self.mock_client
    
    def test_connection(self):
        """Test Solana connection."""
        # Mock successful connection
        self.mock_client.get_health.return_value = True
        self.assertTrue(self.chain.connect())
        
        # Mock failed connection
        self.mock_client.get_health.side_effect = Exception("Connection failed")
        self.assertFalse(self.chain.connect())
    
    def test_balance(self):
        """Test balance checking."""
        test_address = "test_address"
        expected_balance = 1.5  # SOL
        
        # Mock balance response
        self.mock_client.get_balance.return_value = {
            'result': {'value': int(expected_balance * 1e9)}  # Convert to lamports
        }
        
        balance = self.chain.get_balance(test_address)
        self.assertEqual(balance, expected_balance)
        self.mock_client.get_balance.assert_called_with(test_address)
    
    def test_transaction(self):
        """Test transaction handling."""
        test_tx = "test_transaction"
        test_hash = "test_hash"
        
        # Mock transaction sending
        self.mock_client.send_transaction.return_value = {
            'result': test_hash
        }
        
        # Mock transaction verification
        self.mock_client.get_confirmed_transaction.return_value = {
            'result': {'signature': test_hash}
        }
        
        # Test sending transaction
        tx_hash = self.chain.send_transaction(test_tx)
        self.assertEqual(tx_hash, test_hash)
        
        # Test verifying transaction
        self.assertTrue(self.chain.verify_transaction(test_hash))
    
    def test_strategy_validation(self):
        """Test strategy validation."""
        strategy_id = "test_strategy"
        test_proof = {'type': 'test_proof'}
        
        # Mock transaction responses
        self.mock_client.send_transaction.return_value = {'result': 'tx_hash'}
        self.mock_client.get_confirmed_transaction.return_value = {'result': {}}
        
        # Test strategy validation
        result = self.validator.validate_strategy(strategy_id, test_proof)
        self.assertTrue(result)
        
        # Test failed validation
        self.mock_client.send_transaction.side_effect = Exception("Validation failed")
        result = self.validator.validate_strategy(strategy_id, test_proof)
        self.assertFalse(result)
    
    def test_strategy_registration(self):
        """Test strategy registration."""
        strategy_id = "test_strategy"
        metadata = {'version': '1.0.0'}
        expected_tx = "tx_hash"
        
        # Mock responses
        self.mock_client.send_transaction.return_value = {'result': expected_tx}
        
        # Test registration
        tx_hash = self.validator.register_strategy(strategy_id, metadata)
        self.assertEqual(tx_hash, expected_tx)
    
    def test_execution_verification(self):
        """Test execution verification."""
        strategy_id = "test_strategy"
        execution_data = {
            'timestamp': 1234567890,
            'action': 'buy',
            'amount': 1.0
        }
        
        # Mock responses
        self.mock_client.send_transaction.return_value = {'result': 'tx_hash'}
        self.mock_client.get_confirmed_transaction.return_value = {'result': {}}
        
        # Test verification
        result = self.validator.verify_execution(strategy_id, execution_data)
        self.assertTrue(result)
    
    def test_proof_generation(self):
        """Test proof generation and verification."""
        strategy_id = "test_strategy"
        execution_data = {
            'timestamp': 1234567890,
            'action': 'buy',
            'amount': 1.0
        }
        
        # Generate proof
        proof = self.validator.generate_proof(strategy_id, execution_data)
        
        # Verify proof structure
        self.assertEqual(proof['type'], 'solana_zk_proof')
        self.assertEqual(proof['strategy_id'], strategy_id)
        self.assertEqual(proof['data'], execution_data)
        
        # Verify proof
        self.assertTrue(self.validator.verify_proof(proof))
    
    def test_chain_factory(self):
        """Test chain factory."""
        # Test Solana chain creation
        chain = ChainFactory.create_chain('solana', endpoint=self.endpoint)
        self.assertIsInstance(chain, SolanaChain)
        
        # Test validator creation
        validator = ChainFactory.create_validator(chain)
        self.assertIsInstance(validator, SolanaValidator)
        
        # Test unsupported chain
        with self.assertRaises(ValueError):
            ChainFactory.create_chain('unsupported_chain')
    
    def tearDown(self):
        """Clean up test environment."""
        pass

if __name__ == '__main__':
    unittest.main() 