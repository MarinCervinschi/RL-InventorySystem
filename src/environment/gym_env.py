from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.mdp import (
    ActionSpace,
    ActionSpaceFactory,
    CostParameters,
    InventoryState,
    RewardFunction,
    RewardFunctionFactory,
    StateSpace,
    create_initial_state,
)
from src.simulation import SimulationEngine, SystemParameters


class InventoryEnvironment(gym.Env):
    """
    Gymnasium environment for two-product inventory management.

    This environment follows the gymnasium.Env interface:
    - observation_space: Box space for continuous state
    - action_space: Discrete space for order quantities
    - reset(): Initialize episode
    - step(action): Execute action and return (obs, reward, terminated, truncated, info)

    The environment integrates discrete event simulation for realistic dynamics.
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "InventoryManagement-v0"}

    def __init__(
        self,
        system_params: Optional[SystemParameters] = None,
        cost_params: Optional[CostParameters] = None,
        action_space_config: Optional[ActionSpace] = None,
        max_steps_per_episode: int = 100,
        reward_function: Optional[RewardFunction] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the inventory management environment.

        Args:
            system_params: System parameters (demand, lead time, etc.)
            cost_params: Cost structure parameters
            action_space_config: Action space configuration
            max_steps_per_episode: Maximum decision points per episode
            reward_function: Custom reward function (optional)
            random_seed: Random seed for reproducibility
        """
        super().__init__()

        # Initialize parameters
        self.system_params = system_params or SystemParameters.create_default()
        self.cost_params = cost_params or CostParameters()
        self.max_steps = max_steps_per_episode

        # Set random seed
        if random_seed is not None:
            self._np_random = np.random.default_rng(random_seed)
        else:
            self._np_random = np.random.default_rng()

        # Initialize MDP components
        self.state_space = StateSpace()
        self.action_space_config = (
            action_space_config or ActionSpaceFactory.create_medium_action_space()
        )

        # Define gymnasium spaces
        # Observation space: continuous 6D vector (inventory, backorders, outstanding for 2 products)
        self.observation_space = spaces.Box(
            low=np.array([-100, -100, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([200, 200, 100, 100, 150, 150], dtype=np.float32),
            dtype=np.float32,
        )

        # Action space: discrete indices into action_space_config
        self.action_space = spaces.Discrete(self.action_space_config.n)

        # Initialize reward function
        self.reward_function = reward_function or RewardFunctionFactory.create_standard(
            self.cost_params
        )

        # Initialize simulation engine
        self.simulation = SimulationEngine(
            system_params=self.system_params, random_state=self._np_random
        )

        # Episode tracking
        self.current_state: Optional[InventoryState] = None
        self.steps_taken = 0
        self.episode_costs = []

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for this episode
            options: Additional options (e.g., initial inventory levels)

        Returns:
            observation: Initial state as numpy array
            info: Additional information dictionary
        """
        # Handle seed
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            self.simulation.set_random_state(self._np_random)

        # Get initial inventory from options or use default
        initial_inv_0 = 50
        initial_inv_1 = 50
        if options and "initial_inventory" in options:
            initial_inv_0, initial_inv_1 = options["initial_inventory"]

        # Reset state
        self.current_state = create_initial_state(initial_inv_0, initial_inv_1)
        self.steps_taken = 0
        self.episode_costs = []

        # Reset simulation
        self.simulation.reset(initial_state=self.current_state)

        # Prepare info
        info = {"initial_state": self.current_state, "episode": 0}

        return self.current_state.to_array(), info

    def step(
        self, action_index: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action_index: Index into the discrete action space

        Returns:
            observation: Next state as numpy array
            reward: Reward for this transition
            terminated: Whether episode is finished (natural ending)
            truncated: Whether episode is cut off (max steps reached)
            info: Additional information
        """
        if self.current_state is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        # Convert action index to action object
        action = self.action_space_config.get_action(action_index)

        # Execute action in simulation and get next state
        next_state, transition_info = self.simulation.execute_daily_decision(
            current_state=self.current_state, action=action
        )

        # Calculate reward
        reward = self.reward_function(self.current_state, action, next_state)

        # Track costs for analysis
        cost = -reward  # Convert reward back to cost
        self.episode_costs.append(cost)

        # Update state
        self.current_state = next_state
        self.steps_taken += 1

        # Check termination conditions
        terminated = False  # Could add conditions like bankruptcy
        truncated = self.steps_taken >= self.max_steps

        # Prepare info dictionary
        info = {
            "state": next_state,
            "action": action,
            "cost": cost,
            "steps": self.steps_taken,
            "cumulative_cost": sum(self.episode_costs),
            "transition_info": transition_info,
        }

        return next_state.to_array(), reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the environment state.

        Args:
            mode: Rendering mode ("human" or "ansi")

        Returns:
            String representation if mode="ansi", None otherwise
        """
        if self.current_state is None:
            return "Environment not initialized"

        output = []
        output.append("=" * 60)
        output.append(f"Inventory Management Environment (Step {self.steps_taken})")
        output.append("=" * 60)

        # State information
        output.append("\nCurrent State:")
        output.append(
            f"  Product 0: Inventory={self.current_state.inventory_levels[0]:3d}, "
            f"Backorders={self.current_state.backorders[0]:3d}, "
            f"Outstanding={self.current_state.outstanding_orders[0]:3d}"
        )
        output.append(
            f"  Product 1: Inventory={self.current_state.inventory_levels[1]:3d}, "
            f"Backorders={self.current_state.backorders[1]:3d}, "
            f"Outstanding={self.current_state.outstanding_orders[1]:3d}"
        )

        # Inventory positions
        output.append("\nInventory Positions:")
        output.append(f"  Product 0: {self.current_state.get_inventory_position(0):3d}")
        output.append(f"  Product 1: {self.current_state.get_inventory_position(1):3d}")

        # Episode statistics
        if self.episode_costs:
            output.append("\nEpisode Statistics:")
            output.append(f"  Total Cost: ${sum(self.episode_costs):.2f}")
            output.append(f"  Average Cost/Step: ${np.mean(self.episode_costs):.2f}")

        output.append("=" * 60)

        result = "\n".join(output)

        if mode == "human":
            print(result)
            return None
        elif mode == "ansi":
            return result
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        """Clean up resources."""
        pass

    # Additional utility methods

    def get_current_state(self) -> Optional[InventoryState]:
        """Get current state object."""
        return self.current_state

    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get statistics for the current episode."""
        if not self.episode_costs:
            return {}

        return {
            "total_cost": sum(self.episode_costs),
            "average_cost": np.mean(self.episode_costs),
            "std_cost": np.std(self.episode_costs),
            "min_cost": min(self.episode_costs),
            "max_cost": max(self.episode_costs),
            "steps": self.steps_taken,
        }


# Register with gymnasium (optional, but recommended)
def register_environment():
    """Register the environment with gymnasium."""
    try:
        gym.register(
            id="InventoryManagement-v0",
            entry_point="src.environment.gym_env:InventoryEnvironment",
            max_episode_steps=100,
        )
    except gym.error.Error:
        pass  # Already registered
