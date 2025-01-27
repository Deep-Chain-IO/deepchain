import unittest
import torch
import numpy as np
from examples.simple_strategy import SimpleRLStrategy

class TestSimpleRLStrategy(unittest.TestCase):
    """
    Unit tests for SimpleRLStrategy implementation.
    """
    
    def setUp(self):
        """
        Set up test environment before each test case.
        """
        self.strategy = SimpleRLStrategy(input_dim=10, hidden_dim=64, output_dim=3)
        self.test_state = {
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
    
    def test_initialization(self):
        """
        Test strategy initialization.
        """
        self.assertIsNotNone(self.strategy.policy)
        self.assertEqual(self.strategy.policy.network[0].in_features, 10)
        self.assertEqual(self.strategy.policy.network[0].out_features, 64)
        self.assertEqual(self.strategy.policy.network[2].out_features, 3)
    
    def test_predict(self):
        """
        Test prediction functionality.
        """
        prediction = self.strategy.predict(self.test_state)
        
        self.assertIn('action', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn(prediction['action'], ['buy', 'hold', 'sell'])
        self.assertGreaterEqual(prediction['confidence'], 0.0)
        self.assertLessEqual(prediction['confidence'], 1.0)
    
    def test_training(self):
        """
        Test training functionality.
        """
        train_data = {
            'states': np.random.randn(100, 10),
            'actions': np.random.randint(0, 3, 100),
            'rewards': np.random.randn(100)
        }
        
        # Should not raise any exceptions
        self.strategy.train(train_data, n_epochs=2, lr=0.001)
    
    def test_save_load(self):
        """
        Test model saving and loading.
        """
        # Save model
        save_path = "test_model.pth"
        self.strategy.save(save_path)
        
        # Create new strategy and load saved model
        new_strategy = SimpleRLStrategy(input_dim=10, hidden_dim=64, output_dim=3)
        new_strategy.load(save_path)
        
        # Compare predictions
        pred1 = self.strategy.predict(self.test_state)
        pred2 = new_strategy.predict(self.test_state)
        
        self.assertEqual(pred1['action'], pred2['action'])
        self.assertEqual(pred1['confidence'], pred2['confidence'])
    
    def test_action_space(self):
        """
        Test action space definition.
        """
        action_space = self.strategy.get_action_space()
        
        self.assertEqual(action_space['type'], 'discrete')
        self.assertEqual(action_space['size'], 3)
        self.assertEqual(action_space['actions'], ['buy', 'hold', 'sell'])
    
    def test_state_space(self):
        """
        Test state space definition.
        """
        state_space = self.strategy.get_state_space()
        
        self.assertEqual(state_space['type'], 'continuous')
        self.assertEqual(state_space['shape'], (10,))
        self.assertEqual(len(state_space['features']), 10)
    
    def test_reward_calculation(self):
        """
        Test reward calculation.
        """
        state = {'price': 100, 'next_price': 110}
        
        # Test buy action
        buy_reward = self.strategy.get_reward(state, {'action': 'buy'})
        self.assertEqual(buy_reward, 10)
        
        # Test sell action
        sell_reward = self.strategy.get_reward(state, {'action': 'sell'})
        self.assertEqual(sell_reward, -10)
        
        # Test hold action
        hold_reward = self.strategy.get_reward(state, {'action': 'hold'})
        self.assertEqual(hold_reward, 0)

if __name__ == '__main__':
    unittest.main() 