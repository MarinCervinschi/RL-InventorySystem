from dataclasses import dataclass
from typing import Optional

from src.mdp.action import Action
from src.mdp.state import State


@dataclass(frozen=True)
class CostParameters:
    """
    Cost parameters for the inventory system.

    From assignment specification:
    - K: Fixed ordering cost (setup cost)
    - i: Incremental cost per unit (purchase cost)
    - h: Holding cost per unit per day
    - pi: Shortage penalty per unit per day (backorder cost)
    """

    K: float = 10.0  # Fixed ordering cost
    i: float = 3.0  # Unit purchase cost
    h: float = 1.0  # Holding cost per unit per day
    pi: float = 7.0  # Shortage penalty per unit per day


@dataclass(frozen=True)
class CostComponents:
    """Breakdown of costs for analysis."""

    ordering_cost: float
    holding_cost: float
    shortage_cost: float

    @property
    def total_cost(self) -> float:
        """Get total cost."""
        return self.ordering_cost + self.holding_cost + self.shortage_cost


class RewardFunction:
    """
    Reward function as specified in the assignment.

    R(t) = -[C_order(t) + C_holding(t) + C_shortage(t)]

    Where:
    - C_order = K·𝟙{q>0} + i·q  (for each product)
    - C_holding = h·max(0, I)  (for each product)
    - C_shortage = π·max(0, -I)  (for each product)

    Design Note: Works with any number of products. The assignment specifies
    2 products, but this implementation is generic.
    """

    def __init__(self, params: Optional[CostParameters] = None):
        """
        Initialize reward function.

        Args:
            params: Cost parameters (uses defaults if None)
        """
        self.params = params or CostParameters()

    def calculate_costs(self, state: State, action: Action) -> CostComponents:
        """
        Calculate detailed cost breakdown for N products.

        Args:
            state: Current state (after action effects)
            action: Action taken

        Returns:
            Detailed cost components

        Raises:
            ValueError: If state and action have different number of products
        """
        # Validate matching product counts
        if state.num_products != action.num_products:
            raise ValueError(
                f"Product count mismatch: state has {state.num_products} "
                f"but action has {action.num_products}"
            )

        num_products = state.num_products

        # Ordering costs (K + i·q for each product ordered)
        ordering_cost = 0.0
        for j in range(num_products):
            q = action.order_quantities[j]
            if q > 0:
                ordering_cost += self.params.K + self.params.i * q

        # Holding costs (h·I⁺ for each product)
        holding_cost = 0.0
        for j in range(num_products):
            on_hand = state.get_on_hand_inventory(j)
            holding_cost += self.params.h * on_hand

        # Shortage costs (π·I⁻ for each product)
        shortage_cost = 0.0
        for j in range(num_products):
            backorders = state.get_backorders(j)
            shortage_cost += self.params.pi * backorders

        return CostComponents(
            ordering_cost=ordering_cost,
            holding_cost=holding_cost,
            shortage_cost=shortage_cost,
        )

    def __call__(self, state: State, action: Action) -> float:
        """
        Calculate reward (negative cost).

        Args:
            state: Current state
            action: Action taken

        Returns:
            Reward (negative of total cost)
        """
        costs = self.calculate_costs(state, action)
        return -costs.total_cost
