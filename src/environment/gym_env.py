from typing import Any, Dict, Optional, Tuple
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from src.mdp.state import (
    InventoryState,
    StateSpace,
    create_initial_state,
    update_state_with_observation,
)
from src.mdp.action import ActionSpace
from src.mdp.reward import StandardRewardFunction, CostParameters
from src.simulation import InventorySimulation, SystemParameters


class InventoryEnvironment(gym.Env):
    """
    Gymnasium environment for inventory management with frame stacking.

    Observation Space: Continuous Box((k+1)*4,) - stacked observations
    Action Space: Discrete(n) where n = (Q_max + 1)²

    Reward: Negative cost (minimize cost = maximize reward)
    """

    def __init__(
        self,
        k: int = 3,
        Q_max: int = 20,
        episode_length: int = 100,
        gamma: float = 0.99,
        system_params: Optional[SystemParameters] = None,
        cost_params: Optional[CostParameters] = None,
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
        self.reward_function = StandardRewardFunction(cost_params or CostParameters())

        # Simulation
        self.system_params = system_params or SystemParameters.create_default()
        self.simulation = InventorySimulation(self.system_params, self.np_random)

        # Current state (with frame stacking)
        self.current_state: Optional[InventoryState] = None
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
            self.simulation = InventorySimulation(self.system_params, self.np_random)

        # Create initial state (same observation repeated k+1 times)
        initial_net_inv_0 = 50
        initial_net_inv_1 = 50
        self.current_state = create_initial_state(
            net_inventory_0=initial_net_inv_0,
            net_inventory_1=initial_net_inv_1,
            k=self.k,
        )

        # Reset simulation
        self.simulation.reset(self.current_state.current_observation)

        # Reset counters
        self.current_step = 0
        self.episode_costs = []
        self.episode_rewards = []

        # Return observation
        obs = self.current_state.to_array()
        info = {
            "step": self.current_step,
            "net_inventory": self.current_state.current_observation.net_inventory,
            "outstanding": self.current_state.current_observation.outstanding_orders,
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
        if self.current_state is None:
            raise RuntimeError("Must call reset() before step()")

        # Convert action index to action object
        action_obj = self.action_space_config.get_action(action)

        # Execute in simulation
        next_obs, sim_info = self.simulation.execute_daily_decision(action_obj)

        # Calculate reward
        reward = self.reward_function(next_obs, action_obj)
        cost = -reward

        # Update state with frame stacking
        self.current_state = update_state_with_observation(self.current_state, next_obs)

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
            "net_inventory": next_obs.net_inventory,
            "outstanding": next_obs.outstanding_orders,
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

        return self.current_state.to_array(), reward, terminated, truncated, info

    def get_current_state(self) -> InventoryState:
        """Get current state object (for analysis)."""
        if self.current_state is None:
            raise RuntimeError("Must call reset() first")
        return self.current_state

    def render(self):
        """Render current state (optional)."""
        if self.current_state is None:
            return

        obs = self.current_state.current_observation
        print(f"Step {self.current_step}:")
        print(
            f"  Product 0: net_inv={obs.net_inventory[0]:4d}, outstanding={obs.outstanding_orders[0]:3d}"
        )
        print(
            f"  Product 1: net_inv={obs.net_inventory[1]:4d}, outstanding={obs.outstanding_orders[1]:3d}"
        )
        if self.episode_costs:
            print(f"  Last cost: ${self.episode_costs[-1]:.2f}")
