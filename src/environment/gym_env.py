from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.mdp.action import ActionSpace
from src.mdp.reward import RewardFunction
from src.mdp.state import (
    State,
    StateHistory,
    StateSpace,
    create_initial_history,
    update_history,
)
from src.simulation import InventorySimulation


class InventoryEnvironment(gym.Env):
    """
    Gymnasium environment for inventory management.

    State Representation: State (single timestep)
    POMDP Solution: Frame stacking via StateHistory

    Observation Space: Continuous Box((k+1)*4,) - stacked states
    Action Space: Discrete(n) where n = (Q_max + 1)²

    Reward: Negative cost (minimize cost = maximize reward)
    """

    def __init__(
        self,
        k: int = 3,
        Q_max: int = 20,
        episode_length: int = 100,
        gamma: float = 0.99,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize environment.

        Args:
            k: Number of historical frames to stack (default: 3)
            Q_max: Maximum order quantity per product (default: 20)
            episode_length: Steps per episode (default: 100)
            gamma: Discount factor (default: 0.99)
            system_params: Simulation parameters
            cost_params: Cost parameters
            random_seed: Random seed for reproducibility
        """
        self.k = k
        self.Q_max = Q_max
        self.episode_length = episode_length
        self.gamma = gamma

        # Initialize RNG
        self.np_random = np.random.default_rng(random_seed)

        # MDP components
        self.state_space_config = StateSpace(k=k)
        self.action_space_config = ActionSpace(Q_max=Q_max)
        self.reward_function = RewardFunction()

        # Simulation
        self.simulation = InventorySimulation(random_state=self.np_random)

        # Current state history (for POMDP frame stacking)
        self.state_history: Optional[StateHistory] = None
        self.current_step = 0

        # Gymnasium spaces
        # Observation space: continuous (for neural networks)
        obs_dim = (k + 1) * 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: discrete
        self.action_space = spaces.Discrete(self.action_space_config.n)

        # Episode statistics
        self.episode_costs = []
        self.episode_rewards = []

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            (observation_array, info_dict)
        """
        # Set seed if provided
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.simulation = InventorySimulation(random_state=self.np_random)

        # Create initial state history (start empty)
        self.state_history = create_initial_history(k=self.k)

        # Reset simulation
        self.simulation.reset(self.state_history.current_state)

        # Reset counters
        self.current_step = 0
        self.episode_costs = []
        self.episode_rewards = []

        # Return observation (flattened history)
        obs = self.state_history.to_array()
        info = {
            "step": self.current_step,
            "net_inventory": self.state_history.current_state.net_inventory,
            "outstanding": self.state_history.current_state.outstanding_orders,
        }

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.

        Args:
            action: Action index

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.state_history is None:
            raise RuntimeError("Must call reset() before step()")

        # Convert action index to action object
        action_obj = self.action_space_config.get_action(action)

        # Execute in simulation (returns new state)
        next_state, sim_info = self.simulation.execute_daily_decision(action_obj)

        # Calculate reward
        reward = self.reward_function(next_state, action_obj)
        cost = -reward

        # Update state history (POMDP frame stacking)
        self.state_history = update_history(self.state_history, next_state)

        # Update counters
        self.current_step += 1
        self.episode_costs.append(cost)
        self.episode_rewards.append(reward)

        # Check termination
        terminated = False  # No natural termination
        truncated = self.current_step >= self.episode_length

        # Build info
        info = {
            "step": self.current_step,
            "cost": cost,
            "reward": reward,
            "net_inventory": next_state.net_inventory,
            "outstanding": next_state.outstanding_orders,
            "num_customers": sim_info["num_customers"],
            "total_demand": sim_info["total_demand"],
        }

        if truncated:
            info["episode"] = {
                "r": sum(self.episode_rewards),
                "l": self.current_step,
                "total_cost": sum(self.episode_costs),
                "avg_cost": np.mean(self.episode_costs),
            }

        return self.state_history.to_array(), reward, terminated, truncated, info

    def get_current_state(self) -> State:
        """Get current state (state at time t)."""
        if self.state_history is None:
            raise RuntimeError("Must call reset() first")
        return self.state_history.current_state

    def get_state_history(self) -> StateHistory:
        """Get state history (for POMDP frame stacking analysis)."""
        if self.state_history is None:
            raise RuntimeError("Must call reset() first")
        return self.state_history

    def render(self):
        """Render current state (optional)."""
        if self.state_history is None:
            return

        state = self.state_history.current_state
        print(f"Step {self.current_step}:")
        print(
            f"  Product 0: net_inv={state.net_inventory[0]:4d}, outstanding={state.outstanding_orders[0]:3d}"
        )
        print(
            f"  Product 1: net_inv={state.net_inventory[1]:4d}, outstanding={state.outstanding_orders[1]:3d}"
        )
        if self.episode_costs:
            print(f"  Last cost: ${self.episode_costs[-1]:.2f}")
