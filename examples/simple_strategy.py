import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any

from deepchain.core.strategy.base import RLStrategy

class SimpleMLPPolicy(nn.Module):
    """
    Simple MLP-based policy network for the RL strategy.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class SimpleRLStrategy(RLStrategy):
    """
    Example implementation of a simple RL-based trading strategy.
    """
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 3):
        """
        Initialize the strategy with a simple MLP policy.
        
        Args:
            input_dim: Dimension of the state space
            hidden_dim: Hidden layer dimension
            output_dim: Dimension of the action space
        """
        self.policy = SimpleMLPPolicy(input_dim, hidden_dim, output_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
    
    def train(self, data: Dict[str, Any], **kwargs) -> None:
        """
        Simple training loop using policy gradient.
        """
        # Example training logic
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=kwargs.get('lr', 1e-3))
        n_epochs = kwargs.get('n_epochs', 100)
        
        for epoch in range(n_epochs):
            state = torch.FloatTensor(data['states']).to(self.device)
            action_probs = self.policy(state)
            loss = self._compute_loss(action_probs, data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading decision based on current state.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor([list(state.values())]).to(self.device)
            action_probs = self.policy(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()
            
            # Map action to trading decision
            action_map = {0: 'buy', 1: 'hold', 2: 'sell'}
            return {'action': action_map[action], 'confidence': float(action_probs.max())}
    
    def save(self, path: str) -> None:
        """
        Save the policy model.
        """
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path: str) -> None:
        """
        Load the policy model.
        """
        self.policy.load_state_dict(torch.load(path))
    
    def get_action_space(self) -> Dict[str, Any]:
        """
        Define the action space.
        """
        return {
            'type': 'discrete',
            'size': 3,  # buy, hold, sell
            'actions': ['buy', 'hold', 'sell']
        }
    
    def get_state_space(self) -> Dict[str, Any]:
        """
        Define the state space.
        """
        return {
            'type': 'continuous',
            'shape': (10,),  # 10 features
            'features': [
                'price', 'volume', 'high', 'low',
                'ma_5', 'ma_10', 'rsi', 'macd',
                'bollinger_upper', 'bollinger_lower'
            ]
        }
    
    def get_reward(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        """
        Calculate the reward for the action taken.
        """
        # Example reward calculation based on price change
        price_change = state.get('next_price', 0) - state.get('price', 0)
        
        if action['action'] == 'buy':
            return price_change
        elif action['action'] == 'sell':
            return -price_change
        else:  # hold
            return 0.0
    
    def _compute_loss(self, action_probs: torch.Tensor, data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute the policy gradient loss.
        """
        rewards = torch.FloatTensor(data['rewards']).to(self.device)
        actions = torch.LongTensor(data['actions']).to(self.device)
        
        # Simple policy gradient loss
        log_probs = torch.log(action_probs)
        selected_log_probs = log_probs[range(len(actions)), actions]
        loss = -(selected_log_probs * rewards).mean()
        
        return loss 