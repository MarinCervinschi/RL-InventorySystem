from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class State:
    """
    State of the inventory system at a single decision epoch.

    This is the fundamental MDP state - what the system IS at time t.

    Attributes:
        net_inventory: Tuple of net inventory levels (I_0, I_1, ...) where:
            - Positive: on-hand inventory
            - Negative: backorders (unfulfilled demand)
        outstanding_orders: Tuple of outstanding order quantities (O_0, O_1, ...)

    Note: Length determines number of products. Assignment uses 2, but design
    supports N products for future scalability.
    """

    net_inventory: Tuple[int, ...]
    outstanding_orders: Tuple[int, ...]

    @property
    def num_products(self) -> int:
        """Number of products in this observation."""
        return len(self.net_inventory)

    def to_array(self) -> NDArray[np.float32]:
        """Convert observation to numpy array [I_0, O_0, I_1, O_1, ...]."""
        result = []
        for i in range(self.num_products):
            result.extend([self.net_inventory[i], self.outstanding_orders[i]])
        return np.array(result, dtype=np.float32)

    def get_on_hand_inventory(self, product_id: int) -> int:
        """Get on-hand inventory for a product (max(0, net_inventory))."""
        self._validate_product_id(product_id)
        return max(0, self.net_inventory[product_id])

    def get_backorders(self, product_id: int) -> int:
        """Get backorders for a product (max(0, -net_inventory))."""
        self._validate_product_id(product_id)
        return max(0, -self.net_inventory[product_id])

    def get_inventory_position(self, product_id: int) -> int:
        """
        Get inventory position for a product.

        IP = Net_Inventory + Outstanding_Orders

        Note: Net_Inventory already accounts for backorders (I - B),
        so IP = (I - B) + O = I - B + O
        """
        self._validate_product_id(product_id)
        return self.net_inventory[product_id] + self.outstanding_orders[product_id]

    def _validate_product_id(self, product_id: int) -> None:
        """Validate product ID is within valid range."""
        if not 0 <= product_id < self.num_products:
            raise ValueError(
                f"Product ID {product_id} out of range [0, {self.num_products})"
            )


@dataclass(frozen=True)
class StateHistory:
    """
    Frame stacking tool for POMDP (history of states).

    Design Note: This is NOT the state itself, but a POMDP solution.
    - State = Single timestep (what the system IS at time t)
    - StateHistory = Tool to approximate Markov property

    By stacking k+1 recent states, the agent can infer hidden dynamics
    (e.g., lead times, in-transit orders arrival patterns).

    Attributes:
        states: Sequence of states [s_t, s_{t-1}, ..., s_{t-k}]
            Ordered from most recent (index 0) to oldest (index k)
    """

    states: Tuple[State, ...]  # Immutable sequence

    @property
    def k(self) -> int:
        """Get frame stack depth (number of historical frames)."""
        return len(self.states) - 1

    @property
    def num_products(self) -> int:
        """Number of products in this state history."""
        return self.current_state.num_products

    @property
    def current_state(self) -> State:
        """Get the most recent state (s_t)."""
        return self.states[0]

    def to_array(self) -> NDArray[np.float32]:
        """
        Convert history to flat numpy array.

        Returns:
            1D array of shape ((k+1) * 2 * num_products,) containing all stacked states
        """
        return np.concatenate([state.to_array() for state in self.states])

    @property
    def shape(self) -> Tuple[int]:
        """Get shape of history array."""
        return (len(self.states) * 2 * self.num_products,)

    def get_on_hand_inventory(self, product_id: int) -> int:
        """Get current on-hand inventory for a product."""
        return self.current_state.get_on_hand_inventory(product_id)

    def get_backorders(self, product_id: int) -> int:
        """Get current backorders for a product."""
        return self.current_state.get_backorders(product_id)

    def get_outstanding_orders(self, product_id: int) -> int:
        """Get current outstanding orders for a product."""
        return self.current_state.outstanding_orders[product_id]

    def get_inventory_position(self, product_id: int) -> int:
        """Get current inventory position for a product."""
        return self.current_state.get_inventory_position(product_id)


class StateSpace:
    """
    Configuration for the state space (Observation space).

    Design Note: State = Observation.
    The 'k' parameter is for frame stacking (POMDP tool), not part of
    the fundamental state definition.

    Defines bounds, normalization parameters, and utility functions.
    """

    def __init__(
        self,
        k: int = 3,
        net_inventory_min: int = -100,
        net_inventory_max: int = 200,
        max_outstanding: int = 150,
    ):
        """
        Initialize state space configuration.

        Args:
            k: Number of historical frames to stack (default: 3)
            net_inventory_min: Minimum net inventory (backorder limit)
            net_inventory_max: Maximum net inventory (on-hand limit)
            max_outstanding: Maximum outstanding orders
        """
        self.k = k
        self.net_inventory_min = net_inventory_min
        self.net_inventory_max = net_inventory_max
        self.max_outstanding = max_outstanding

        # State dimension: (k+1) observations × 4 features per observation
        self.dim = (k + 1) * 4

    @property
    def shape(self) -> Tuple[int]:
        """Get shape of state space."""
        return (self.dim,)

    def is_valid_state(self, state: State) -> bool:
        """Check if state is within bounds."""
        for net_inv, outstanding in zip(state.net_inventory, state.outstanding_orders):
            if net_inv < self.net_inventory_min or net_inv > self.net_inventory_max:
                return False
            if outstanding < 0 or outstanding > self.max_outstanding:
                return False
        return True

    def is_valid_history(self, history: StateHistory) -> bool:
        """Check if state history is valid."""
        if len(history.states) != self.k + 1:
            return False
        return all(self.is_valid_state(state) for state in history.states)

    def sample_state(self, random_state: Optional[np.random.Generator] = None) -> State:
        """Sample a random valid state."""
        if random_state is None:
            random_state = np.random.default_rng()

        net_inv_0 = random_state.integers(
            self.net_inventory_min, self.net_inventory_max + 1
        )
        net_inv_1 = random_state.integers(
            self.net_inventory_min, self.net_inventory_max + 1
        )
        out_0 = random_state.integers(0, self.max_outstanding + 1)
        out_1 = random_state.integers(0, self.max_outstanding + 1)

        return State(
            net_inventory=(int(net_inv_0), int(net_inv_1)),
            outstanding_orders=(int(out_0), int(out_1)),
        )

    def sample_history(
        self, random_state: Optional[np.random.Generator] = None
    ) -> StateHistory:
        """Sample a random state history (for testing/debugging)."""
        states = tuple(self.sample_state(random_state) for _ in range(self.k + 1))
        return StateHistory(states=states)


def create_state(
    net_inventory_0: int = 0,
    net_inventory_1: int = 0,
    outstanding_0: int = 0,
    outstanding_1: int = 0,
) -> State:
    """
    Convenience function to create a state.

    Args:
        net_inventory_0: Net inventory for product 0
        net_inventory_1: Net inventory for product 1
        outstanding_0: Outstanding orders for product 0
        outstanding_1: Outstanding orders for product 1

    Returns:
        State object
    """
    return State(
        net_inventory=(net_inventory_0, net_inventory_1),
        outstanding_orders=(outstanding_0, outstanding_1),
    )


def create_initial_history(
    net_inventory_0: int = 0, net_inventory_1: int = 0, k: int = 3
) -> StateHistory:
    """
    Create an initial state history with the same state repeated.

    Used at episode start before real history is built up (cold start).

    Args:
        net_inventory_0: Initial net inventory for product 0
        net_inventory_1: Initial net inventory for product 1
        k: Number of historical frames

    Returns:
        StateHistory with repeated initial state
    """
    initial_state = create_state(net_inventory_0, net_inventory_1, 0, 0)
    states = tuple(initial_state for _ in range(k + 1))
    return StateHistory(states=states)


def update_history(history: StateHistory, new_state: State) -> StateHistory:
    """
    Update state history by adding new state and dropping oldest.

    This is the frame stacking mechanism for POMDP.

    Args:
        history: Current state history
        new_state: New state to add

    Returns:
        Updated history with new state at front
    """
    # Add new state at the front, drop the last one
    new_states = (new_state,) + history.states[:-1]
    return StateHistory(states=new_states)
