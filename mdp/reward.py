from dataclasses import dataclass
from typing import Protocol, Optional
import numpy as np

from .state import InventoryState
from .action import InventoryAction


@dataclass(frozen=True)
class CostComponents:
    """
    Breakdown of costs in the inventory system.

    This provides transparency and helps with debugging/analysis.
    """

    holding_cost: float = 0.0  # Cost of holding inventory
    backorder_cost: float = 0.0  # Penalty for unfulfilled demand
    ordering_cost: float = 0.0  # Fixed cost per order
    purchase_cost: float = 0.0  # Variable cost per unit ordered

    @property
    def total_cost(self) -> float:
        """Total cost across all components."""
        return (
            self.holding_cost
            + self.backorder_cost
            + self.ordering_cost
            + self.purchase_cost
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "holding": self.holding_cost,
            "backorder": self.backorder_cost,
            "ordering": self.ordering_cost,
            "purchase": self.purchase_cost,
            "total": self.total_cost,
        }

    def __repr__(self) -> str:
        return (
            f"CostComponents(holding={self.holding_cost:.2f}, "
            f"backorder={self.backorder_cost:.2f}, "
            f"ordering={self.ordering_cost:.2f}, "
            f"purchase={self.purchase_cost:.2f}, "
            f"total={self.total_cost:.2f})"
        )


@dataclass(frozen=True)
class CostParameters:
    """
    System-wide cost parameters.

    As specified in the assignment:
    - K: Fixed ordering cost per order
    - i: Unit purchase cost per item
    - h: Holding cost per unit per day
    - π: Backorder penalty per unit per day
    """

    K: float = 10.0  # Fixed ordering cost
    i: float = 3.0  # Unit purchase cost
    h: float = 1.0  # Holding cost per unit per day
    pi: float = 7.0  # Backorder penalty per unit per day

    def validate(self) -> None:
        """Validate that all costs are non-negative."""
        if self.K < 0 or self.i < 0 or self.h < 0 or self.pi < 0:
            raise ValueError("All cost parameters must be non-negative")


class RewardFunction(Protocol):
    """
    Protocol (interface) for reward functions.

    This follows the Dependency Inversion Principle - depend on abstractions,
    not concretions. Different reward functions can be implemented.
    """

    def __call__(
        self, state: InventoryState, action: InventoryAction, next_state: InventoryState
    ) -> float:
        """
        Calculate reward for a state transition.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            Reward (negative cost in this case)
        """
        ...


class StandardRewardFunction:
    """
    Standard cost-based reward function for inventory management.

    Reward = -1 * (holding_cost + backorder_cost + ordering_cost + purchase_cost)

    The negative sign converts minimization (costs) to maximization (rewards),
    which is the standard RL formulation.
    """

    def __init__(self, cost_params: CostParameters):
        """
        Args:
            cost_params: Cost parameters for the system
        """
        cost_params.validate()
        self.params = cost_params

    def calculate_costs(
        self, state: InventoryState, action: InventoryAction
    ) -> CostComponents:
        """
        Calculate detailed cost breakdown.

        This method computes costs based on:
        1. Current inventory levels (holding cost)
        2. Current backorders (backorder penalty)
        3. Action taken (ordering and purchase costs)
        """
        # Holding cost: h * inventory_level for each product
        holding_cost = sum(
            max(0, inv) * self.params.h for inv in state.inventory_levels
        )

        # Backorder cost: π * backorder_level for each product
        backorder_cost = sum(bo * self.params.pi for bo in state.backorders)

        # Ordering cost: K per product if order placed (quantity > 0)
        num_orders = sum(1 for qty in action.order_quantities if qty > 0)
        ordering_cost = num_orders * self.params.K

        # Purchase cost: i * quantity for each product
        purchase_cost = sum(qty * self.params.i for qty in action.order_quantities)

        return CostComponents(
            holding_cost=holding_cost,
            backorder_cost=backorder_cost,
            ordering_cost=ordering_cost,
            purchase_cost=purchase_cost,
        )

    def __call__(
        self, state: InventoryState, action: InventoryAction, next_state: InventoryState
    ) -> float:
        """
        Calculate reward (negative cost).

        Note: We use the current state for costs, as costs are incurred
        based on the current inventory position and the action taken.
        """
        costs = self.calculate_costs(state, action)
        return -costs.total_cost


class ShapedRewardFunction(StandardRewardFunction):
    """
    Reward shaping variant that adds potential-based shaping.

    Shaped reward: R'(s,a,s') = R(s,a,s') + γ*Φ(s') - Φ(s)

    where Φ is a potential function. This can speed up learning without
    changing the optimal policy (Ng et al., 1999).

    This is an example of the Template Method pattern - base calculation
    from parent, with additional shaping logic.
    """

    def __init__(self, cost_params: CostParameters, gamma: float = 0.99):
        super().__init__(cost_params)
        self.gamma = gamma

    def potential(self, state: InventoryState) -> float:
        """
        Potential function based on inventory position.

        Higher potential for better inventory positions (less likely to stockout).
        """
        # Simple potential: reward being near target inventory
        target_inventory = 50
        potential = 0.0

        for product_id in range(2):
            inv_position = state.get_inventory_position(product_id)
            # Negative quadratic penalty for deviation from target
            deviation = abs(inv_position - target_inventory)
            potential -= 0.01 * (deviation**2)

        return potential

    def __call__(
        self, state: InventoryState, action: InventoryAction, next_state: InventoryState
    ) -> float:
        """Calculate shaped reward."""
        base_reward = super().__call__(state, action, next_state)
        shaping = self.gamma * self.potential(next_state) - self.potential(state)
        return base_reward + shaping


class RewardFunctionFactory:
    """
    Factory for creating different reward function configurations.
    """

    @staticmethod
    def create_standard(
        cost_params: Optional[CostParameters] = None,
    ) -> StandardRewardFunction:
        """Create standard cost-based reward function."""
        if cost_params is None:
            cost_params = CostParameters()
        return StandardRewardFunction(cost_params)

    @staticmethod
    def create_shaped(
        cost_params: Optional[CostParameters] = None, gamma: float = 0.99
    ) -> ShapedRewardFunction:
        """Create shaped reward function."""
        if cost_params is None:
            cost_params = CostParameters()
        return ShapedRewardFunction(cost_params, gamma)


if __name__ == "__main__":
    # Test cost parameters
    print("Testing CostParameters...")
    params = CostParameters()
    print(f"Cost params: K={params.K}, i={params.i}, h={params.h}, π={params.pi}")
    params.validate()
    print("✓ Parameters valid")

    # Test reward function
    print("\nTesting StandardRewardFunction...")
    from .state import create_initial_state
    from .action import order_both_products

    state = create_initial_state(40, 45)
    action = order_both_products(20, 15)
    next_state = create_initial_state(40, 45)  # Simplified for testing

    reward_fn = StandardRewardFunction(params)
    costs = reward_fn.calculate_costs(state, action)
    print(f"Cost breakdown: {costs}")

    reward = reward_fn(state, action, next_state)
    print(f"Reward (negative cost): {reward:.2f}")

    # Test shaped reward
    print("\nTesting ShapedRewardFunction...")
    shaped_fn = ShapedRewardFunction(params)
    shaped_reward = shaped_fn(state, action, next_state)
    print(f"Shaped reward: {shaped_reward:.2f}")
    print(f"Difference from standard: {shaped_reward - reward:.2f}")

    # Test factory
    print("\nTesting RewardFunctionFactory...")
    standard = RewardFunctionFactory.create_standard()
    shaped = RewardFunctionFactory.create_shaped()
    print(f"Standard reward: {standard(state, action, next_state):.2f}")
    print(f"Shaped reward: {shaped(state, action, next_state):.2f}")

    print("\n✓ All reward tests passed!")
