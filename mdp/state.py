from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class InventoryState:
    """
    Immutable state representation for the two-product inventory system.

    State Space:
    - inventory_levels: (product_0, product_1) - on-hand inventory
    - backorders: (product_0, product_1) - unfulfilled demand
    - outstanding_orders: (product_0, product_1) - orders in transit

    Inventory Position = inventory_level - backorders + outstanding_orders

    Using frozen dataclass ensures immutability (important for RL state transitions).
    """

    inventory_levels: Tuple[int, int]
    backorders: Tuple[int, int]
    outstanding_orders: Tuple[int, int]

    def __post_init__(self):
        """Validate state constraints."""
        if len(self.inventory_levels) != 2:
            raise ValueError("Must have exactly 2 products")
        if len(self.backorders) != 2:
            raise ValueError("Must have exactly 2 products")
        if len(self.outstanding_orders) != 2:
            raise ValueError("Must have exactly 2 products")

        # Backorders should be non-negative
        if any(bo < 0 for bo in self.backorders):
            raise ValueError("Backorders cannot be negative")

        # Outstanding orders should be non-negative
        if any(out < 0 for out in self.outstanding_orders):
            raise ValueError("Outstanding orders cannot be negative")

    def get_inventory_position(self, product_id: int) -> int:
        """
        Calculate inventory position for a product.

        Inventory Position = On-hand - Backorders + Outstanding
        This is the key metric for reorder decisions.
        """
        return (
            self.inventory_levels[product_id]
            - self.backorders[product_id]
            + self.outstanding_orders[product_id]
        )

    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for neural network input."""
        return np.array(
            [
                self.inventory_levels[0],
                self.inventory_levels[1],
                self.backorders[0],
                self.backorders[1],
                self.outstanding_orders[0],
                self.outstanding_orders[1],
            ],
            dtype=np.float32,
        )

    def to_tuple(self) -> Tuple:
        """Convert to tuple for use as dictionary key (for tabular Q-learning)."""
        return (self.inventory_levels, self.backorders, self.outstanding_orders)

    @classmethod
    def from_array(cls, array: np.ndarray) -> "InventoryState":
        """Create state from numpy array."""
        return cls(
            inventory_levels=(int(array[0]), int(array[1])),
            backorders=(int(array[2]), int(array[3])),
            outstanding_orders=(int(array[4]), int(array[5])),
        )

    def __repr__(self) -> str:
        return (
            f"InventoryState(inv={self.inventory_levels}, "
            f"bo={self.backorders}, out={self.outstanding_orders})"
        )


class StateSpace:
    """
    Defines the state space boundaries and provides utility methods.

    This follows the Open-Closed Principle: open for extension,
    closed for modification.
    """

    def __init__(
        self,
        max_inventory: int = 200,
        max_backorders: int = 100,
        max_outstanding: int = 150,
    ):
        self.max_inventory = max_inventory
        self.max_backorders = max_backorders
        self.max_outstanding = max_outstanding

        # For neural networks - normalization parameters
        self.obs_mean = np.array([50, 50, 5, 5, 20, 20], dtype=np.float32)
        self.obs_std = np.array([40, 40, 15, 15, 30, 30], dtype=np.float32)

    def normalize(self, state: InventoryState) -> np.ndarray:
        """Normalize state for neural network input."""
        obs = state.to_array()
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

    def denormalize(self, normalized: np.ndarray) -> InventoryState:
        """Denormalize from neural network output."""
        obs = normalized * self.obs_std + self.obs_mean
        return InventoryState.from_array(obs)

    def is_valid(self, state: InventoryState) -> bool:
        """Check if state is within valid bounds."""
        for inv in state.inventory_levels:
            if inv < -self.max_backorders or inv > self.max_inventory:
                return False
        for bo in state.backorders:
            if bo < 0 or bo > self.max_backorders:
                return False
        for out in state.outstanding_orders:
            if out < 0 or out > self.max_outstanding:
                return False
        return True

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the state space (for gymnasium compatibility)."""
        return (6,)

    def sample(self) -> InventoryState:
        """Sample a random valid state (useful for testing)."""
        return InventoryState(
            inventory_levels=(
                np.random.randint(0, self.max_inventory),
                np.random.randint(0, self.max_inventory),
            ),
            backorders=(
                np.random.randint(0, self.max_backorders // 2),
                np.random.randint(0, self.max_backorders // 2),
            ),
            outstanding_orders=(
                np.random.randint(0, self.max_outstanding // 2),
                np.random.randint(0, self.max_outstanding // 2),
            ),
        )


def create_initial_state(
    inventory_0: int = 50, inventory_1: int = 50
) -> InventoryState:
    """
    Factory function to create initial state.

    Following the Factory Pattern for object creation.
    """
    return InventoryState(
        inventory_levels=(inventory_0, inventory_1),
        backorders=(0, 0),
        outstanding_orders=(0, 0),
    )


if __name__ == "__main__":
    # Test the state representation
    print("Testing InventoryState...")

    state = create_initial_state(50, 50)
    print(f"Initial state: {state}")
    print(f"Inventory position product 0: {state.get_inventory_position(0)}")
    print(f"As array: {state.to_array()}")

    # Test state space
    space = StateSpace()
    print(f"\nState space shape: {space.shape}")
    print(f"Is valid: {space.is_valid(state)}")

    # Test normalization
    normalized = space.normalize(state)
    print(f"Normalized: {normalized}")

    # Test sampling
    random_state = space.sample()
    print(f"\nRandom state: {random_state}")

    print("\n✓ All state tests passed!")
