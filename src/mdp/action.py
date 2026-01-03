from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Action:
    """
    Represents a replenishment decision for N products.

    Design Note: This is an MDP abstraction - the agent's decision.
    The simulation components (SupplierManager) convert this to orders.

    Attributes:
        order_quantities: Tuple (q_0, q_1, ..., q_n) where q_j ∈ [0, Q_max]

    Note: Length determines number of products. Assignment uses 2, but design
    supports N products for future scalability.
    """

    order_quantities: Tuple[int, ...]

    @property
    def num_products(self) -> int:
        """Number of products in this action."""
        return len(self.order_quantities)

    def get_order_quantity(self, product_id: int) -> int:
        """Get order quantity for a specific product."""
        if not 0 <= product_id < self.num_products:
            raise ValueError(
                f"Product ID {product_id} out of range [0, {self.num_products})"
            )
        return self.order_quantities[product_id]


class ActionSpace:
    """
    Discrete action space for inventory management.

    Each product can be ordered in quantities from 0 to Q_max.
    Total action space size: (Q_max + 1)²
    """

    def __init__(self, Q_max: int = 20):
        """
        Initialize action space.

        Args:
            Q_max: Maximum order quantity per product (default: 20)
        """
        if Q_max <= 0:
            raise ValueError(f"Q_max must be positive, got {Q_max}")

        self.Q_max = Q_max
        self.possible_quantities = list(range(Q_max + 1))

        # Generate all possible actions
        self.actions: List[Action] = []
        self.action_to_index: dict = {}

        idx = 0
        for q0 in self.possible_quantities:
            for q1 in self.possible_quantities:
                action = Action(order_quantities=(q0, q1))
                self.actions.append(action)
                self.action_to_index[action] = idx
                idx += 1

        self.n = len(self.actions)

    def sample(self, random_state: Optional[np.random.Generator] = None) -> Action:
        """Sample a random action."""
        if random_state is None:
            random_state = np.random.default_rng()
        idx = random_state.integers(0, self.n)
        return self.actions[idx]

    def get_action(self, index: int) -> Action:
        """Get action by index."""
        if not 0 <= index < self.n:
            raise ValueError(f"Index {index} out of range [0, {self.n})")
        return self.actions[index]

    def get_index(self, action: Action) -> int:
        """Get index of an action."""
        if action not in self.action_to_index:
            raise ValueError(f"Action {action} not in action space")
        return self.action_to_index[action]

    def is_valid(self, action: Action) -> bool:
        """Check if action is valid."""
        return (
            0 <= action.order_quantities[0] <= self.Q_max
            and 0 <= action.order_quantities[1] <= self.Q_max
        )


def order_both_products(quantity_0: int, quantity_1: int) -> Action:
    """
    Convenience function to create an action.

    Args:
        quantity_0: Order quantity for product 0
        quantity_1: Order quantity for product 1

    Returns:
        Action object
    """
    return Action(order_quantities=(quantity_0, quantity_1))


def no_order_action() -> Action:
    """Create a no-order action (0, 0)."""
    return Action(order_quantities=(0, 0))
