from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseStrategy(ABC):
    """
    Base class for all trading strategies in DeepChain framework.
    This abstract class defines the interface that all strategies must implement.
    """

    @abstractmethod
    def train(self, data: Dict[str, Any], **kwargs) -> None:
        """
        Train the strategy model with historical data.
        
        Args:
            data: Dictionary containing training data
            **kwargs: Additional training parameters
        """
        pass

    @abstractmethod
    def predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading decisions based on current market state.
        
        Args:
            state: Current market state information
            
        Returns:
            Dictionary containing trading decisions
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save strategy model to disk.
        
        Args:
            path: Path to save the model
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load strategy model from disk.
        
        Args:
            path: Path to load the model from
        """
        pass

class RLStrategy(BaseStrategy):
    """
    Base class for reinforcement learning based trading strategies.
    Extends BaseStrategy with RL-specific methods.
    """

    @abstractmethod
    def get_action_space(self) -> Dict[str, Any]:
        """
        Define the action space for the RL agent.
        
        Returns:
            Dictionary describing the action space
        """
        pass

    @abstractmethod
    def get_state_space(self) -> Dict[str, Any]:
        """
        Define the state space for the RL agent.
        
        Returns:
            Dictionary describing the state space
        """
        pass

    @abstractmethod
    def get_reward(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        """
        Calculate reward for the current state-action pair.
        
        Args:
            state: Current state information
            action: Action taken by the agent
            
        Returns:
            Reward value
        """
        pass 