"""
Unit tests for Ethereum-based chain integrations.
"""

import unittest
from unittest.mock import Mock, patch
import json
from web3 import Web3
from eth_account import Account
from deepchain.core.blockchain.ethereum import (
    EthereumChain,
    BaseChain,
    BNBChain,
    EthereumValidator
)

class TestEthereumIntegration(unittest.TestCase):
    """Test Ethereum-based chain integrations."""
    
    def setUp(self):
        """Set up test environment."""
        # Initialize chains
        self.eth_chain = EthereumChain(endpoint="http://localhost:8545")
        self.base_chain = BaseChain(endpoint="http://localhost:8545")
        self.bnb_chain = BNBChain(endpoint="http://localhost:8545")
        
        # Mock Web3 instances
        self.mock_eth_web3 = Mock()
        self.mock_base_web3 = Mock()
        self.mock_bnb_web3 = Mock()
        
        self.eth_chain.web3 = self.mock_eth_web3
        self.base_chain.web3 = self.mock_base_web3
        self.bnb_chain.web3 = self.mock_bnb_web3
        
        # Create validators
        self.eth_validator = EthereumValidator(self.eth_chain)
        self.base_validator = EthereumValidator(self.base_chain)
        self.bnb_validator = EthereumValidator(self.bnb_chain)
        
        # Test data
        self.test_address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        self.test_private_key = "0x" + "1" * 64
        self.test_contract_address = "0x" + "2" * 40
    
    def test_ethereum_connection(self):
        """Test Ethereum connection."""
        # Mock successful connection
        self.mock_eth_web3.is_connected.return_value = True
        self.assertTrue(self.eth_chain.connect())
        
        # Mock failed connection
        self.mock_eth_web3.is_connected.side_effect = Exception("Connection failed")
        self.assertFalse(self.eth_chain.connect())
    
    def test_base_connection(self):
        """Test Base connection."""
        # Mock successful connection
        self.mock_base_web3.is_connected.return_value = True
        self.assertTrue(self.base_chain.connect())
        
        # Mock failed connection
        self.mock_base_web3.is_connected.side_effect = Exception("Connection failed")
        self.assertFalse(self.base_chain.connect())
    
    def test_bnb_connection(self):
        """Test BNB Chain connection."""
        # Mock successful connection
        self.mock_bnb_web3.is_connected.return_value = True
        self.assertTrue(self.bnb_chain.connect())
        
        # Mock failed connection
        self.mock_bnb_web3.is_connected.side_effect = Exception("Connection failed")
        self.assertFalse(self.bnb_chain.connect())
    
    def test_balance_checking(self):
        """Test balance checking across chains."""
        expected_balance = 1.5  # ETH/BNB
        balance_wei = Web3.to_wei(expected_balance, 'ether')
        
        # Mock balance responses
        self.mock_eth_web3.eth.get_balance.return_value = balance_wei
        self.mock_base_web3.eth.get_balance.return_value = balance_wei
        self.mock_bnb_web3.eth.get_balance.return_value = balance_wei
        
        # Test Ethereum balance
        balance = self.eth_chain.get_balance(self.test_address)
        self.assertEqual(balance, expected_balance)
        
        # Test Base balance
        balance = self.base_chain.get_balance(self.test_address)
        self.assertEqual(balance, expected_balance)
        
        # Test BNB Chain balance
        balance = self.bnb_chain.get_balance(self.test_address)
        self.assertEqual(balance, expected_balance)
    
    def test_contract_deployment(self):
        """Test contract deployment."""
        test_abi = [{"type": "function", "name": "test"}]
        test_bytecode = "0x123456"
        
        # Mock contract deployment
        mock_contract = Mock()
        mock_contract.constructor.return_value.build_transaction.return_value = {
            'to': None,
            'data': test_bytecode
        }
        
        self.mock_eth_web3.eth.contract.return_value = mock_contract
        self.mock_eth_web3.eth.get_transaction_count.return_value = 0
        self.mock_eth_web3.eth.gas_price = 20000000000
        
        # Mock transaction sending
        self.mock_eth_web3.eth.account.sign_transaction.return_value.rawTransaction = b'raw_tx'
        self.mock_eth_web3.eth.send_raw_transaction.return_value = b'tx_hash'
        self.mock_eth_web3.eth.wait_for_transaction_receipt.return_value = {
            'contractAddress': self.test_contract_address,
            'status': 1
        }
        
        # Deploy contract
        contract_address = self.eth_chain.deploy_contract(test_abi, test_bytecode)
        self.assertEqual(contract_address, self.test_contract_address)
    
    def test_strategy_validation(self):
        """Test strategy validation across chains."""
        strategy_id = "test_strategy"
        test_proof = {'type': 'test_proof'}
        
        # Mock contract calls
        for chain in [self.eth_chain, self.base_chain, self.bnb_chain]:
            chain.contract = Mock()
            chain.contract.functions.validateStrategy.return_value.build_transaction.return_value = {
                'to': self.test_contract_address,
                'data': b'validate'
            }
        
        # Test validation on each chain
        for validator in [self.eth_validator, self.base_validator, self.bnb_validator]:
            result = validator.validate_strategy(strategy_id, test_proof)
            self.assertTrue(result)
    
    def test_strategy_registration(self):
        """Test strategy registration across chains."""
        strategy_id = "test_strategy"
        metadata = {'version': '1.0.0'}
        expected_tx = "0x" + "3" * 64
        
        # Mock contract calls
        for chain in [self.eth_chain, self.base_chain, self.bnb_chain]:
            chain.contract = Mock()
            chain.contract.functions.registerStrategy.return_value.build_transaction.return_value = {
                'to': self.test_contract_address,
                'data': b'register'
            }
            chain.web3.eth.get_transaction_count.return_value = 0
            chain.web3.eth.gas_price = 20000000000
            chain.web3.eth.account.sign_transaction.return_value.rawTransaction = b'raw_tx'
            chain.web3.eth.send_raw_transaction.return_value = expected_tx.encode()
            chain.web3.to_hex.return_value = expected_tx
        
        # Test registration on each chain
        for validator in [self.eth_validator, self.base_validator, self.bnb_validator]:
            tx_hash = validator.register_strategy(strategy_id, metadata)
            self.assertEqual(tx_hash, expected_tx)
    
    def test_execution_verification(self):
        """Test execution verification across chains."""
        strategy_id = "test_strategy"
        execution_data = {
            'timestamp': 1234567890,
            'action': 'buy',
            'amount': 1.0
        }
        test_proof = {'type': 'test_proof'}
        
        # Mock contract calls
        for chain in [self.eth_chain, self.base_chain, self.bnb_chain]:
            chain.contract = Mock()
            chain.contract.functions.verifyExecution.return_value.build_transaction.return_value = {
                'to': self.test_contract_address,
                'data': b'verify'
            }
        
        # Test verification on each chain
        for validator in [self.eth_validator, self.base_validator, self.bnb_validator]:
            result = validator.verify_execution(strategy_id, execution_data, test_proof)
            self.assertTrue(result)
    
    def test_proof_verification(self):
        """Test proof verification across chains."""
        test_proof = {
            'type': 'ethereum_zk_proof',
            'data': {'test': 'data'}
        }
        
        # Mock contract calls
        for chain in [self.eth_chain, self.base_chain, self.bnb_chain]:
            chain.contract = Mock()
            chain.contract.functions.verifyProof.return_value.call.return_value = True
        
        # Test proof verification on each chain
        for validator in [self.eth_validator, self.base_validator, self.bnb_validator]:
            result = validator.verify_proof(test_proof)
            self.assertTrue(result)
    
    def tearDown(self):
        """Clean up test environment."""
        pass

if __name__ == '__main__':
    unittest.main() 