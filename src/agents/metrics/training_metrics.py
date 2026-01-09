"""Training plots for RL agents."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


class TrainingMetrics:
    """Class for generating training-related plots for RL agents."""

    def __init__(self, figsize: tuple = (12, 5)):
        """
        Initialize TrainingPlots.

        Args:
            figsize: Default figure size for plots.
        """
        self.figsize = figsize

    def plot_learning_curve(
        self,
        episode_timesteps: List[int],
        episode_rewards: List[float],
        window: int = 50,
        title: str = "Learning Curve: Agent Training Progress",
        show: bool = True,
    ) -> None:
        """
        Plot learning curve showing episode rewards over training.

        Args:
            episode_timesteps: List of timesteps at which episodes ended.
            episode_rewards: List of episode rewards.
            window: Window size for rolling average.
            title: Plot title.
            show: Whether to display the plot.
        """
        timesteps = np.array(episode_timesteps)
        rewards = np.array(episode_rewards)

        # Compute rolling average
        if len(rewards) >= window:
            rolling_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            rolling_timesteps = timesteps[window - 1 :]
        else:
            rolling_avg = rewards
            rolling_timesteps = timesteps

        # Plot
        _, ax = plt.subplots(figsize=self.figsize)
        ax.plot(
            timesteps,
            rewards,
            "b-",
            linewidth=0.5,
            alpha=0.3,
            label="Episode Reward",
        )
        ax.plot(
            rolling_timesteps,
            rolling_avg,
            "b-",
            linewidth=2,
            label=f"Rolling Average (window={window})",
        )
        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel("Episode Reward", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if show:
            plt.show()

        # Statistics Calculation
        if len(rewards) > 0:
            ep_lengths = timesteps[-1] / len(rewards)
    
            last_window_rewards = rewards[-window:] if len(rewards) >= window else rewards
            reward_per_day = rewards / ep_lengths

            print("=" * 60)
            print(f"📊 Training Statistics:")
            print("=" * 60)
            print(f"  Total Episodes: {len(rewards)}")
            print(f"  Total Timesteps: {timesteps[-1]}")
            print(f"  Episode Length: {ep_lengths:.2f} timesteps")
            
            print(f"\n  -- Performance (Last {len(last_window_rewards)} episodes) --")
            print(f"  Mean Reward:   {np.mean(last_window_rewards):.2f} ± {np.std(last_window_rewards):.2f}")
            print(f"  Min Reward:    {np.min(last_window_rewards):.2f}")
            print(f"  Max Reward:    {np.max(last_window_rewards):.2f}")
            
            print(f"\n  -- Specifics --")
            print(f"  Best Ever Reward: {np.max(rewards):.2f}")
            print(f"  Last Ep Reward:   {rewards[-1]:.2f}")
            
            print(f"\n -- Daily Rewards --")
            print(f"  Mean Reward per Day: {np.mean(reward_per_day):.2f} ± {np.std(reward_per_day):.2f}")
            print(f"  Final Mean Reward per Day (last {window} eps): {np.mean(reward_per_day[-window:]):.2f} ± {np.std(reward_per_day[-window:]):.2f}")

        
    def plot_epsilon_decay(
        self,
        total_timesteps: int,
        exploration_fraction: float,
        exploration_final_eps: float,
        exploration_initial_eps: float = 1.0,
        title: str = "Exploration vs Exploitation: Epsilon Decay",
        show: bool = True,
    ) -> None:
        """
        Plot epsilon decay schedule over training.

        Args:
            total_timesteps: Total training timesteps.
            exploration_fraction: Fraction of training for epsilon decay.
            exploration_final_eps: Final epsilon value.
            exploration_initial_eps: Initial epsilon value.
            title: Plot title.
            show: Whether to display the plot.

        Returns:
            matplotlib Figure object.
        """
        timesteps = np.linspace(0, total_timesteps, 1000)
        exploration_end = int(total_timesteps * exploration_fraction)

        epsilon_values = []
        for t in timesteps:
            if t < exploration_end:
                progress = t / exploration_end
                epsilon = exploration_initial_eps - progress * (
                    exploration_initial_eps - exploration_final_eps
                )
            else:
                epsilon = exploration_final_eps
            epsilon_values.append(epsilon)

        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(timesteps, epsilon_values, "g-", linewidth=2)
        ax.axvline(
            x=exploration_end,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Exploration ends: {exploration_end:,}",
        )
        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel("Epsilon (ε)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if show:
            plt.show()