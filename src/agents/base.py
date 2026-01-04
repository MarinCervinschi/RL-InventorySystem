from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class Agent(ABC):
    """
    Abstract base class for reinforcement learning agents.

    All concrete agents (DQN, PPO, A2C, etc.) must implement this interface.
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Initialize agent.

        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            device: Device for training ('cpu' or 'cuda')
            seed: Random seed for reproducibility
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device
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
        Select an action given an observation.

        Args:
            observation: Current observation from environment
            deterministic: If True, select best action (no exploration)
                          If False, sample from policy (exploration)

        Returns:
            Action index to execute
        """
        pass

    @abstractmethod
    def train_step(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, float]:
        """
        Perform one training step (learn from experience).

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated (time limit)

        Returns:
            Dictionary of training metrics (loss, etc.)
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

    def reset_episode(self):
        """
        Reset internal state at the beginning of a new episode.

        Override this if your agent maintains episode-specific state
        (e.g., episodic memory, hidden states for RNNs).
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
            f"{self.__class__.__name__}("
            f"lr={self.learning_rate}, "
            f"gamma={self.gamma}, "
            f"device={self.device})"
        )
