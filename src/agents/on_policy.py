from abc import abstractmethod
from typing import Any, Dict, Optional

from src.agents.base import Agent


class OnPolicyAgent(Agent):
    """
    Abstract base class for on-policy agents (PPO, A2C, etc.).

    On-policy agents collect a batch of experience using the current policy,
    then update the policy using that batch. They don't use replay buffers.

    Key characteristics:
    - Learn from current policy only
    - Collect rollouts (trajectories) before updating
    - More sample-efficient per update
    - Better for continuous action spaces

    Examples: PPO, A2C, REINFORCE
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        seed: Optional[int] = None,
        n_steps: int = 2048,
        gae_lambda: float = 0.95,
    ):
        """
        Initialize on-policy agent.

        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            device: Device for training ('cpu' or 'cuda')
            seed: Random seed
            n_steps: Number of steps to collect per rollout
            gae_lambda: Lambda parameter for Generalized Advantage Estimation
        """
        super().__init__(
            observation_space,
            action_space,
            learning_rate,
            gamma,
            device,
            seed,
        )
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda

    @abstractmethod
    def collect_rollout(
        self,
        env: Any,
        num_steps: int,
    ) -> Dict[str, Any]:
        """
        Collect a rollout of experience using current policy.

        Args:
            env: Environment to collect experience from
            num_steps: Number of steps to collect

        Returns:
            Dictionary containing rollout data:
                - 'observations': (num_steps, obs_dim)
                - 'actions': (num_steps,)
                - 'rewards': (num_steps,)
                - 'values': (num_steps,) - value predictions
                - 'log_probs': (num_steps,) - log probabilities
                - 'terminateds': (num_steps,) - natural terminations
                - 'truncateds': (num_steps,) - time limit truncations
        """
        pass

    @abstractmethod
    def update_policy(
        self,
        rollout_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Update policy using collected rollout data.

        Args:
            rollout_data: Data collected from rollout

        Returns:
            Dictionary of training metrics:
                - 'policy_loss': Policy gradient loss
                - 'value_loss': Value function loss
                - 'entropy': Policy entropy (for exploration)
                - 'approx_kl': Approximate KL divergence (for PPO)
                - 'explained_variance': How well value function fits returns
        """
        pass

    @abstractmethod
    def compute_advantages(
        self,
        rewards: Any,
        values: Any,
        terminateds: Any,
        truncateds: Any,
    ) -> tuple[Any, Any]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        GAE balances bias-variance tradeoff:
        - λ=0: high bias, low variance (TD error)
        - λ=1: low bias, high variance (Monte Carlo)

        Args:
            rewards: Array of rewards
            values: Array of value predictions
            terminateds: Array of natural termination flags
            truncateds: Array of truncation flags

        Returns:
            (advantages, returns) tuple:
                - advantages: Advantage estimates for policy gradient
                - returns: Target values for value function
        """
        pass

    def train_step(
        self,
        observation: Any,
        action: int,
        reward: float,
        next_observation: Any,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, float]:
        """
        On-policy agents don't learn from individual transitions.

        This method is not used. Instead, use:
        1. collect_rollout() to gather experience
        2. update_policy() to learn from the batch

        Raises:
            NotImplementedError: Always raises (use collect_rollout + update_policy)
        """
        raise NotImplementedError(
            "On-policy agents don't use train_step(). "
            "Use collect_rollout() and update_policy() instead."
        )
