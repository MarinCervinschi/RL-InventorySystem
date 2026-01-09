from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np

MaybeCallback = Union[None, list[BaseCallback], BaseCallback]

class Agent(ABC):
    """
    Abstract base class for reinforcement learning agents using Stable-Baselines3.

    All concrete agents (DQN, PPO, A2C, etc.) must implement this interface.
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        seed: Optional[int] = None,
    ):
        """
        Initialize agent.

        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            seed: Random seed for reproducibility
        """
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.seed = seed

        # Training statistics
        self.total_steps = 0
        self.total_episodes = 0

        if seed is not None:
            self._set_seed(seed)

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        except ImportError:
            pass

    @abstractmethod
    def select_action(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> int:
        """
        Select an action given an observation (Inference).

        Args:
            observation: Current observation from environment
            deterministic: If True, select best action (no exploration)
                           If False, sample from policy (exploration)

        Returns:
            Action index to execute
        """
        pass
    
    @abstractmethod
    def train(self, total_timesteps: int, callbacks: MaybeCallback = None,**kwargs) -> Any:
        """
        Train the agent.

        Args:
            total_timesteps: The total number of samples (env steps) to train on
            **kwargs: Additional arguments for the training loop
        """
        pass


    @abstractmethod
    def save(self, path: Path):
        """
        Save agent's model and parameters to disk.

        Args:
            path: Path to save directory or file
        """
        pass

    @abstractmethod
    def load(self, path: Path):
        """
        Load agent's model and parameters from disk.

        Args:
            path: Path to saved model directory or file
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current training statistics.

        Returns:
            Dictionary of statistics (total_steps, episodes, etc.)
        """
        return {
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
        }

    def __repr__(self) -> str:
        """String representation of agent."""
        return (
            f"{self.__class__.__name__}(seed={self.seed})"
        )