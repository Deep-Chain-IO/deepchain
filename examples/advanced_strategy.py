"""
Example of a complete trading strategy implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from deepchain.core.strategy.base import RLStrategy
from deepchain.core.monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)

class DQNPolicy(nn.Module):
    """Deep Q-Network policy network."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int
    ):
        """Initialize policy network.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of actions)
        """
        super().__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.network(x)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

class AdvancedStrategy(RLStrategy):
    """Advanced trading strategy using DQN."""
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: List[int] = [64, 32],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        buffer_size: int = 10000
    ):
        """Initialize strategy.
        
        Args:
            input_dim: State dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            batch_size: Training batch size
            buffer_size: Replay buffer size
        """
        super().__init__()
        
        # Network parameters
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = 3  # buy, hold, sell
        
        # Training parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Initialize networks
        self.policy_net = DQNPolicy(input_dim, hidden_dims, self.output_dim)
        self.target_net = DQNPolicy(input_dim, hidden_dims, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )
        
        # Initialize replay buffer
        self.replay_buffer = []
        self.buffer_size = buffer_size
        
        # Initialize metrics collector
        self.metrics = MetricsCollector()
    
    def train(self, data: Dict[str, np.ndarray]) -> None:
        """Train the strategy.
        
        Args:
            data: Training data dictionary
        """
        logger.info("Starting training...")
        
        states = torch.FloatTensor(data['states'])
        actions = torch.LongTensor(data['actions'])
        rewards = torch.FloatTensor(data['rewards'])
        next_states = torch.FloatTensor(data['next_states'])
        
        n_samples = len(states)
        n_epochs = n_samples // self.batch_size
        
        for epoch in range(n_epochs):
            # Sample batch
            indices = np.random.choice(n_samples, self.batch_size)
            state_batch = states[indices]
            action_batch = actions[indices]
            reward_batch = rewards[indices]
            next_state_batch = next_states[indices]
            
            # Compute Q values
            start_time = time.time()
            current_q = self.policy_net(state_batch).gather(
                1,
                action_batch.unsqueeze(1)
            )
            
            # Compute target Q values
            with torch.no_grad():
                next_q = self.target_net(next_state_batch).max(1)[0]
                target_q = reward_batch + self.gamma * next_q
            
            # Compute loss
            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record metrics
            latency = (time.time() - start_time) * 1000
            self.metrics.record_latency(latency)
            self.metrics.record_error(loss.item())
            
            # Update target network
            if epoch % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon * self.epsilon_decay
            )
            
            # Log progress
            if (epoch + 1) % 100 == 0:
                stats = self.metrics.get_statistics()
                logger.info(
                    f"Epoch {epoch + 1}/{n_epochs} - "
                    f"Loss: {loss.item():.4f}, "
                    f"Epsilon: {self.epsilon:.4f}, "
                    f"Avg Latency: {stats.get('avg_latency', 0):.2f}ms"
                )
    
    def predict(self, state: Dict[str, float]) -> int:
        """Predict action for given state.
        
        Args:
            state: Current market state
            
        Returns:
            Predicted action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor([
            list(state.values())
        ])
        
        # Record prediction start time
        start_time = time.time()
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.output_dim)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
        
        # Record metrics
        latency = (time.time() - start_time) * 1000
        self.metrics.record_latency(latency)
        self.metrics.record_prediction(action)
        
        return action
    
    def get_action_space(self) -> Dict[str, Any]:
        """Get action space definition."""
        return {
            'n_actions': self.output_dim,
            'action_meanings': ['sell', 'hold', 'buy']
        }
    
    def get_state_space(self) -> Dict[str, Any]:
        """Get state space definition."""
        return {
            'shape': (self.input_dim,),
            'features': [
                'price', 'volume', 'high', 'low',
                'ma_5', 'ma_10', 'rsi', 'macd',
                'bollinger_upper', 'bollinger_lower'
            ]
        }
    
    def get_reward(self, state: Dict[str, float], action: int) -> float:
        """Calculate reward for state-action pair."""
        # Simple reward based on price movement
        price = state['price']
        next_price = state.get('next_price', price)
        
        if action == 0:  # sell
            return price - next_price
        elif action == 2:  # buy
            return next_price - price
        else:  # hold
            return 0.0
    
    def save(self, path: str) -> None:
        """Save strategy to disk."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str) -> None:
        """Load strategy from disk."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon'] 