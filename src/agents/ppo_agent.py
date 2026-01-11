from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from sympy import Q

from src.agents.base import Agent
from src.environment import InventoryEnvironment

MaybeCallback = Union[None, list[BaseCallback], BaseCallback]


class PPOAgent(Agent):
    """
    PPO Agent for Inventory Management using Stable-Baselines3.

    Implements the Agent interface while leveraging SB3's optimized PPO.

    MDP Mapping:
    - State: 16-dim continuous (4 stacked frames × 4 features)
    - Action: Discrete(441) = (Q_max+1)² for two products
    - Reward: Negative total cost (ordering + holding + shortage)
    """

    def __init__(
        self,
        env: InventoryEnvironment,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        policy_kwargs: Optional[Dict[str, List[int]]] = None,
        device: str = "auto",
        seed: Optional[int] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
    ):
        """
        Initialize PPO Agent.

        Args:
            env: InventoryEnvironment instance
            learning_rate: Learning rate [3e-5, 3e-3]
            gamma: Discount factor [0.95, 0.999] - high for inventory planning
            n_steps: Number of steps to run per environment per update
            batch_size: Minibatch size for training
            n_epochs: Number of epochs when optimizing the surrogate loss
            clip_range: Clipping parameter for PPO
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
            gae_lambda: GAE lambda parameter
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
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "gae_lambda": gae_lambda,
            "policy_kwargs": policy_kwargs,
            "device": device,
            "seed": seed,
        }

        # Create SB3 PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
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

    def train(
        self,
        total_timesteps: int,
        callbacks: MaybeCallback = None,
        progress_bar: bool = True,
    ) -> None:
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
            callback=callbacks,
        )

        self.total_steps = total_timesteps

    def save(self, metadata: Dict, path: Optional[Path] = None):
        """Save model and environment metadata."""
        import json

        if path is None:
            path = Path("./models")

        path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        self.model.save(path / "ppo_model")

        # Save environment metadata
        metadata_path = path / "ppo_model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"PPO model saved to {path}")
        print(f"  Metadata: k={metadata['k']}, Q_max={metadata['Q_max']}")

    def load(self, path: Optional[Path] = None):
        """Load model from disk."""

        if path is None:
            path = Path("./models")

        self.model = PPO.load(path / "ppo_model", env=self.env)
        print(f"PPO weights loaded from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = super().get_stats()
        stats.update(
            {
                "hyperparams": self.hyperparams,
            }
        )
        return stats

    def __repr__(self) -> str:
        return f"PPOAgent(lr={self.learning_rate}, γ={self.gamma}, "
