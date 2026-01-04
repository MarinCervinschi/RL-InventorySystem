from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from src.agents.base import Agent


class OffPolicyAgent(Agent):
    """
    Abstract base class for off-policy agents (DQN, SAC, TD3, etc.).

    Off-policy agents use a replay buffer to store and reuse past experience.
    They can learn from experience collected by any policy (not just current).

    Key characteristics:
    - Learn from replay buffer (old experience)
    - More sample-efficient (reuse transitions)
    - Better for discrete action spaces (DQN)
    - Stable learning from diverse experience

    Examples: DQN, DDQN, SAC, TD3
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        seed: Optional[int] = None,
        buffer_size: int = 100000,
        batch_size: int = 64,
        learning_starts: int = 1000,
        train_freq: int = 1,
        target_update_interval: int = 1000,
    ):
        """
        Initialize off-policy agent.

        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            device: Device for training ('cpu' or 'cuda')
            seed: Random seed
            buffer_size: Maximum size of replay buffer
            batch_size: Batch size for training
            learning_starts: Start training after this many steps
            train_freq: Train every N steps
            target_update_interval: Update target network every N steps
        """
        super().__init__(
            observation_space,
            action_space,
            learning_rate,
            gamma,
            device,
            seed,
        )
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval

    @abstractmethod
    def add_experience(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ):
        """
        Add a single transition to replay buffer.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            terminated: Whether episode ended naturally (value = 0)
            truncated: Whether episode was truncated (bootstrap value)
        """
        pass

    @abstractmethod
    def sample_batch(self) -> Dict[str, np.ndarray]:
        """
        Sample a random batch from replay buffer.

        Returns:
            Dictionary containing batch data:
                - 'observations': (batch_size, obs_dim)
                - 'actions': (batch_size,)
                - 'rewards': (batch_size,)
                - 'next_observations': (batch_size, obs_dim)
                - 'terminateds': (batch_size,) - natural terminations
                - 'truncateds': (batch_size,) - time limit truncations
        """
        pass

    @abstractmethod
    def update_target_network(self):
        """
        Update target network (for Q-learning variants).

        Target networks stabilize learning by providing fixed targets
        during training. Common update strategies:
        - Hard update: Copy weights periodically
        - Soft update: Polyak averaging (τ * online + (1-τ) * target)
        """
        pass

    def can_train(self) -> bool:
        """
        Check if agent has enough experience to start training.

        Returns:
            True if:
                - Replay buffer has at least batch_size samples
                - Total steps >= learning_starts
        """
        return (
            self.total_steps >= self.learning_starts
            and self.total_steps >= self.batch_size
        )

    def should_train(self) -> bool:
        """
        Check if agent should train at current step.

        Returns:
            True if can_train() and current step is train_freq multiple
        """
        return self.can_train() and self.total_steps % self.train_freq == 0

    def should_update_target(self) -> bool:
        """
        Check if target network should be updated.

        Returns:
            True if current step is target_update_interval multiple
        """
        return self.total_steps % self.target_update_interval == 0

    def get_buffer_size(self) -> int:
        """
        Get current number of transitions in replay buffer.

        Returns:
            Number of stored transitions
        """
        raise NotImplementedError("Concrete agents must implement get_buffer_size()")
