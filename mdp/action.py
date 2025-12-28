from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


@dataclass(frozen=True)
class InventoryAction:
    """
    Immutable action representation for ordering decisions.

    Action Space:
    - order_quantities: (product_0_qty, product_1_qty)

    Each quantity represents units to order for that product.
    Zero means no order is placed.
    """

    order_quantities: Tuple[int, int]

    def __post_init__(self):
        """Validate action constraints."""
        if len(self.order_quantities) != 2:
            raise ValueError("Must specify order quantity for exactly 2 products")

        if any(qty < 0 for qty in self.order_quantities):
            raise ValueError("Order quantities cannot be negative")

    def to_array(self) -> np.ndarray:
        """Convert action to numpy array."""
        return np.array(self.order_quantities, dtype=np.int32)

    def to_tuple(self) -> Tuple[int, int]:
        """Convert to tuple for indexing."""
        return self.order_quantities

    @classmethod
    def from_array(cls, array: np.ndarray) -> "InventoryAction":
        """Create action from numpy array."""
        return cls(order_quantities=(int(array[0]), int(array[1])))

    def __repr__(self) -> str:
        return f"InventoryAction(order={self.order_quantities})"


class ActionSpace:
    """
    Defines the action space with discrete order quantities.

    Actions are discretized to make the problem tractable for Q-learning.
    For each product, we can order 0, increment, 2*increment, ..., max_quantity.

    This follows the Strategy Pattern - different discretization strategies
    can be implemented.
    """

    def __init__(self, max_order_quantity: int = 100, order_increment: int = 5):
        """
        Args:
            max_order_quantity: Maximum units that can be ordered at once
            order_increment: Discretization step (e.g., 5 means orders of 0, 5, 10, 15, ...)
        """
        self.max_order_quantity = max_order_quantity
        self.order_increment = order_increment

        # Generate possible order quantities for a single product
        self.possible_quantities = list(
            range(0, max_order_quantity + 1, order_increment)
        )

        # Generate all possible actions (Cartesian product)
        self.actions = [
            InventoryAction(order_quantities=(q0, q1))
            for q0 in self.possible_quantities
            for q1 in self.possible_quantities
        ]

        # Create index mapping for fast lookup
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}

    @property
    def n(self) -> int:
        """Number of discrete actions (for compatibility with gymnasium)."""
        return len(self.actions)

    def get_action(self, index: int) -> InventoryAction:
        """Get action by index."""
        if index < 0 or index >= self.n:
            raise IndexError(f"Action index {index} out of range [0, {self.n})")
        return self.actions[index]

    def get_index(self, action: InventoryAction) -> int:
        """Get index of an action."""
        return self.action_to_idx[action]

    def sample(self) -> InventoryAction:
        """Sample a random action."""
        return self.actions[np.random.randint(0, self.n)]

    def contains(self, action: InventoryAction) -> bool:
        """Check if action is in the action space."""
        return action in self.action_to_idx

    def clip(self, quantities: Tuple[int, int]) -> InventoryAction:
        """
        Clip continuous quantities to valid discrete action.
        Useful for continuous control methods.
        """
        clipped = tuple(
            min(self.possible_quantities, key=lambda x: abs(x - qty))
            for qty in quantities
        )
        return InventoryAction(order_quantities=clipped)  # type: ignore #TODO fix

    def __repr__(self) -> str:
        return (
            f"ActionSpace(n={self.n}, "
            f"max_order={self.max_order_quantity}, "
            f"increment={self.order_increment})"
        )


class ActionSpaceFactory:
    """
    Factory for creating different action space configurations.

    Follows the Factory Pattern for flexible action space creation.
    """

    @staticmethod
    def create_coarse_action_space() -> ActionSpace:
        """Coarse discretization (fewer actions, faster learning)."""
        return ActionSpace(max_order_quantity=100, order_increment=20)

    @staticmethod
    def create_medium_action_space() -> ActionSpace:
        """Medium discretization (balanced)."""
        return ActionSpace(max_order_quantity=100, order_increment=10)

    @staticmethod
    def create_fine_action_space() -> ActionSpace:
        """Fine discretization (more precision, slower learning)."""
        return ActionSpace(max_order_quantity=100, order_increment=5)

    @staticmethod
    def create_custom_action_space(max_order: int, increment: int) -> ActionSpace:
        """Custom discretization."""
        return ActionSpace(max_order_quantity=max_order, order_increment=increment)


# Action construction helpers
def no_order_action() -> InventoryAction:
    """Create action with no orders."""
    return InventoryAction(order_quantities=(0, 0))


def order_product_0(quantity: int) -> InventoryAction:
    """Order only product 0."""
    return InventoryAction(order_quantities=(quantity, 0))


def order_product_1(quantity: int) -> InventoryAction:
    """Order only product 1."""
    return InventoryAction(order_quantities=(0, quantity))


def order_both_products(quantity_0: int, quantity_1: int) -> InventoryAction:
    """Order both products."""
    return InventoryAction(order_quantities=(quantity_0, quantity_1))


if __name__ == "__main__":
    # Test action representation
    print("Testing InventoryAction...")

    action = order_both_products(20, 30)
    print(f"Action: {action}")
    print(f"As array: {action.to_array()}")

    # Test action space
    print("\nTesting ActionSpace...")
    action_space = ActionSpace(max_order_quantity=50, order_increment=10)
    print(f"Action space: {action_space}")
    print(f"Number of actions: {action_space.n}")
    print(f"Possible quantities: {action_space.possible_quantities}")

    # Test indexing
    test_action = action_space.get_action(0)
    print(f"\nAction at index 0: {test_action}")
    print(f"Index of action: {action_space.get_index(test_action)}")

    # Test sampling
    random_action = action_space.sample()
    print(f"Random action: {random_action}")

    # Test factory
    print("\nTesting ActionSpaceFactory...")
    coarse = ActionSpaceFactory.create_coarse_action_space()
    medium = ActionSpaceFactory.create_medium_action_space()
    fine = ActionSpaceFactory.create_fine_action_space()

    print(f"Coarse: {coarse.n} actions")
    print(f"Medium: {medium.n} actions")
    print(f"Fine: {fine.n} actions")

    print("\n✓ All action tests passed!")
