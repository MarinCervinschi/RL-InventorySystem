"""
Environment factory utilities for creating training and evaluation environments.

Provides standardized environment creation with VecNormalize for RL agents.
"""

from typing import Optional, Tuple

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environment.gym_env import InventoryEnvironment


def make_env(
    seed: Optional[int] = None,
    episode_length: int = 100,
    k: int = 3,
    Q_max: int = 20,
):
    """
    Environment factory function for DummyVecEnv.
    
    Args:
        seed: Random seed for reproducibility
        episode_length: Steps per episode (days)
        k: Frame stacking depth (for POMDP → MDP conversion)
        Q_max: Maximum order quantity per product
        
    Returns:
        Callable that creates an InventoryEnvironment instance
    """
    def _init():
        env = InventoryEnvironment(
            k=k,
            Q_max=Q_max,
            episode_length=episode_length,
            random_seed=seed,
        )
        return env
    return _init


def create_envs(
    train_seed: int = 42,
    eval_seed: int = 123,
    episode_length: int = 100,
    k: int = 3,
    Q_max: int = 20,
    gamma: float = 0.99,
) -> Tuple[VecNormalize, VecNormalize]:
    """
    Create training and evaluation environments with normalization.
    
    Uses VecNormalize for running mean-variance normalization of observations
    and rewards. Evaluation environment has synced normalization statistics
    but doesn't update them during evaluation.
    
    Args:
        train_seed: Random seed for training environment
        eval_seed: Random seed for evaluation environment
        episode_length: Steps per episode (days)
        k: Frame stacking depth
        Q_max: Maximum order quantity per product
        gamma: Discount factor for return normalization
    
    Returns:
        (train_env, eval_env) tuple with synced normalization
    """
    # Training environment with normalization
    train_env = DummyVecEnv([
        make_env(seed=train_seed, episode_length=episode_length, k=k, Q_max=Q_max)
    ])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,       # Normalize observations (running mean/var)
        norm_reward=True,    # Normalize rewards
        clip_obs=10.0,       # Clip normalized observations
        clip_reward=10.0,    # Clip normalized rewards
        gamma=gamma,         # For return normalization
    )
    
    # Evaluation environment (separate seed!)
    eval_env = DummyVecEnv([
        make_env(seed=eval_seed, episode_length=episode_length, k=k, Q_max=Q_max)
    ])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,   # Don't normalize eval rewards (want true values!)
        clip_obs=10.0,
        training=False,      # Don't update statistics during evaluation
        gamma=gamma,
    )
    
    # Sync normalization statistics
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    
    return train_env, eval_env


def sync_eval_env(train_env: VecNormalize, eval_env: VecNormalize) -> None:
    """
    Sync normalization statistics from training to evaluation environment.
    
    Call this after training to ensure eval uses updated statistics.
    
    Args:
        train_env: Training environment with updated statistics
        eval_env: Evaluation environment to sync
    """
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    eval_env.training = False
    eval_env.norm_reward = False
