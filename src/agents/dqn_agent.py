from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.environment import InventoryEnvironment
from src.agents.base import Agent

MaybeCallback = Union[None, list[BaseCallback], BaseCallback]

class DQNAgent(Agent):
    """
    DQN Agent for Inventory Management using Stable-Baselines3.

    Implements the Agent interface while leveraging SB3's optimized DQN.

    MDP Mapping:
    - State: 16-dim continuous (4 stacked frames × 4 features)
    - Action: Discrete(441) = (Q_max+1)² for two products
    - Reward: Negative total cost (ordering + holding + shortage)
    """

    def __init__(
        self,
        env: Monitor | InventoryEnvironment,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        exploration_fraction: float = 0.3,
        exploration_final_eps: float = 0.05,
        target_update_interval: int = 500,
        learning_starts: int = 1000,
        train_freq: int = 4,
        policy_kwargs: Optional[Dict[str, List[int]]] = None,
        device: str = "auto",
        seed: Optional[int] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
    ):
        """
        Initialize DQN Agent.

        Args:
            env: InventoryEnvironment instance
            learning_rate: Learning rate [1e-5, 1e-3]
            gamma: Discount factor [0.95, 0.999] - high for inventory planning
            buffer_size: Experience replay buffer size
            batch_size: Minibatch size for training
            exploration_fraction: Fraction of training for epsilon decay
            exploration_final_eps: Final exploration rate
            target_update_interval: Steps between target network updates
            learning_starts: Steps before training starts
            train_freq: Steps between training updates
            policy_kwargs: Network architecture
            device: 'cpu', 'cuda', or 'auto'
            seed: Random seed for reproducibility
            tensorboard_log: TensorBoard log directory
            verbose: Verbosity level
        """
        # Initialize base class
        super().__init__(
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=seed,
        )

        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device
        self.policy_kwargs = policy_kwargs

        # Store hyperparameters for logging
        self.hyperparams = {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "exploration_fraction": exploration_fraction,
            "exploration_final_eps": exploration_final_eps,
            "target_update_interval": target_update_interval,
            "learning_starts": learning_starts,
            "train_freq": train_freq,
            "policy_kwargs": policy_kwargs,
            "device": device,
            "seed": seed,
        }

        # Create SB3 DQN model
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=exploration_final_eps,
            target_update_interval=target_update_interval,
            learning_starts=learning_starts,
            train_freq=train_freq,
            gradient_steps=1,
            tau=1.0,  # Hard target update
            policy_kwargs=self.policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            device=device,
        )

        # Training history
        self.training_history = {
            "eval_rewards": [],
            "eval_steps": [],
        }

    def select_action(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> int:
        """Select action using learned policy."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def train(self, total_timesteps: int, callbacks: MaybeCallback = None, progress_bar: bool = True) -> None:
        """
        Train the agent.

        Args:
            total_timesteps: Total training steps
            progress_bar: Show training progress
            callback: Un callback singolo, una lista di callback, o None.
        """

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            callback=callbacks
        )

        self.total_steps = total_timesteps

    def save(self, path: Optional[Path] = None):
        """Save model statistics."""

        if path is None:
            path = Path("./models")

        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save(path / "dqn_model")
        print(f"DQN weights saved to {path}")

    def load(self, path: Optional[Path] = None):
        """Load model from disk."""

        if path is None:
            path = Path("./models")

        self.model = DQN.load(path / "dqn_model", env=self.env)
        print(f"DQN weights loaded from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = super().get_stats()
        stats.update(
            {
                "hyperparams": self.hyperparams,
                "exploration_rate": self.model.exploration_rate,
            }
        )
        return stats

    def __repr__(self) -> str:
        return f"DQNAgent(lr={self.learning_rate}, γ={self.gamma}, "
