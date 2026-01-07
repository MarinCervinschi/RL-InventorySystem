import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from src.agents.base import Agent


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
        env: VecNormalize,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        exploration_fraction: float = 0.3,
        exploration_final_eps: float = 0.05,
        target_update_interval: int = 500,
        learning_starts: int = 1000,
        train_freq: int = 4,
        net_arch: Optional[List[int]] = None,
        device: str = "auto",
        seed: Optional[int] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
    ):
        """
        Initialize DQN Agent.
        
        Args:
            env: Vectorized and normalized environment
            learning_rate: Learning rate [1e-5, 1e-3] 
            gamma: Discount factor [0.95, 0.999] - high for inventory planning
            buffer_size: Experience replay buffer size
            batch_size: Minibatch size for training
            exploration_fraction: Fraction of training for epsilon decay
            exploration_final_eps: Final exploration rate
            target_update_interval: Steps between target network updates
            learning_starts: Steps before training starts
            train_freq: Steps between training updates
            net_arch: Network architecture (hidden layers)
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
        self.net_arch = net_arch or [256, 256]
        
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
            "net_arch": self.net_arch,
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
            policy_kwargs={"net_arch": self.net_arch},
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
        
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> int:
        """Select action using learned policy."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action[0]) if hasattr(action, '__len__') else int(action)
    
    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[VecNormalize] = None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 10,
        log_dir: str = "./logs",
        model_dir: str = "./models",
        progress_bar: bool = True,
    ) -> "DQNAgent":
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training steps
            eval_env: Evaluation environment (with synced normalization)
            eval_freq: Steps between evaluations
            n_eval_episodes: Episodes per evaluation
            log_dir: Directory for logs
            model_dir: Directory for model checkpoints
            progress_bar: Show training progress
            
        Returns:
            self (for method chaining)
        """
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        callbacks = []
        
        # Evaluation callback
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=model_dir,
                log_path=log_dir,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=model_dir,
            name_prefix="dqn_checkpoint",
        )
        callbacks.append(checkpoint_callback)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=progress_bar,
        )
        
        self.total_steps = total_timesteps
        return self
    
    def save(self, path: Path):
        """Save model and normalization statistics."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "dqn_model")
        self.env.save(path / "vec_normalize.pkl")
        
    def load(self, path: Path):
        """Load model from disk."""
        path = Path(path)

        stats_path = path / "vec_normalize.pkl"
        if stats_path.exists() and isinstance(self.env, VecNormalize):
            # Carica le statistiche nell'ambiente attuale dell'agente
            self.env = VecNormalize.load(str(stats_path), self.env.venv)
            # Riassegna l'env al modello (fondamentale!)
            self.model.set_env(self.env)
            print(f"Normalization stats loaded from {stats_path}")
        else:
            print("⚠️ WARNING: No normalization stats found or env not normalized.")

        self.model = DQN.load(path / "dqn_model", env=self.env)
        print(f"DQN weights loaded from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = super().get_stats()
        stats.update({
            "hyperparams": self.hyperparams,
            "exploration_rate": self.model.exploration_rate,
        })
        return stats
    
    def __repr__(self) -> str:
        return (
            f"DQNAgent(lr={self.learning_rate}, γ={self.gamma}, "
            f"arch={self.net_arch}, device={self.device})"
        )