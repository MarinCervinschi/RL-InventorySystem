import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.mdp import Action, State


@dataclass
class sSPolicy:
    """
    (s,S) policy for N-product inventory system.

    Rule: If inventory_position ≤ s_min, order (s_max - IP) units (clipped to Q_max)

    Parameters:
        params: Tuple of (s_min, s_max) for each product
                e.g., ((9, 21), (8, 20)) for 2 products
        Q_max: Maximum order quantity per product (action space constraint)
    """

    params: Tuple[Tuple[int, int], ...]  # ((s_min_0, s_max_0), (s_min_1, s_max_1), ...)
    Q_max: int = 20

    def __post_init__(self):
        """Validate parameters."""
        for i, (s_min, s_max) in enumerate(self.params):
            if s_max <= s_min:
                raise ValueError(
                    f"Product {i}: s_max ({s_max}) must be > s_min ({s_min})"
                )

        if self.Q_max <= 0:
            raise ValueError("Q_max must be positive")

    @property
    def num_products(self) -> int:
        """Number of products this policy manages."""
        return len(self.params)

    def get_s_min(self, product_id: int) -> int:
        """Get reorder point for a product."""
        return self.params[product_id][0]

    def get_s_max(self, product_id: int) -> int:
        """Get order-up-to level for a product."""
        return self.params[product_id][1]

    def __call__(self, state: State) -> Action:
        """
        Make ordering decision based on current state.

        Args:
            state: Current inventory state

        Returns:
            Action with order quantities for all products
        """
        if state.num_products != self.num_products:
            raise ValueError(
                f"State has {state.num_products} products, "
                f"policy configured for {self.num_products}"
            )

        order_quantities = []
        for i in range(self.num_products):
            ip = state.get_inventory_position(i)
            s_min, s_max = self.params[i]
            q = self._get_order(ip, s_min, s_max)
            order_quantities.append(q)

        return Action(tuple(order_quantities))

    def _get_order(self, inventory_position: int, s_min: int, s_max: int) -> int:
        """Get order quantity for a single product."""
        if inventory_position <= s_min:
            return min(max(0, s_max - inventory_position), self.Q_max)
        return 0

    def as_dict(self) -> dict:
        """Return policy parameters as dictionary."""
        return {f"product_{i}": self.params[i] for i in range(self.num_products)} | {
            "Q_max": self.Q_max
        }

    def __str__(self) -> str:
        products = ", ".join(f"P{i}={self.params[i]}" for i in range(self.num_products))
        return f"(s,S) Policy: {products}, Q_max={self.Q_max}"

    def __repr__(self) -> str:
        return f"sSPolicy(params={self.params}, Q_max={self.Q_max})"


def create_sS_policy(
    *product_params: Tuple[int, int],
    Q_max: int = 20,
) -> sSPolicy:
    """
    Factory function to create (s,S) policy.

    Args:
        *product_params: (s_min, s_max) tuples for each product
        Q_max: Maximum order quantity

    Returns:
        Configured sSPolicy instance

    Example:
        policy = create_sS_policy((9, 21), (8, 20), Q_max=20)
    """
    return sSPolicy(params=product_params, Q_max=Q_max)


def calculate_policy_params(
    ddlt: Tuple[float, ...],
    safety_factor: float = 0.5,
    eoq: Optional[Tuple[float, ...]] = None,
    Q_max: int = 20,
) -> Tuple[Tuple[int, int], ...]:
    """
    Calculate (s,S) parameters from demand and lead time data.

    Args:
        ddlt: Demand during lead time for each product
        safety_factor: Safety stock as fraction of DDLT (default 0.5)
        eoq: Economic order quantities (optional, uses Q_max if None)
        Q_max: Maximum order quantity constraint

    Returns:
        Tuple of (s_min, s_max) for each product
    """
    result = []

    for i, ddlt_i in enumerate(ddlt):
        # Calculate reorder point: s_min = DDLT + safety_stock
        safety_stock = ddlt_i * safety_factor
        s_min = int(math.ceil(ddlt_i + safety_stock))

        # Calculate order-up-to level: s_max = s_min + order_size
        if eoq is not None and i < len(eoq):
            order_size = min(int(eoq[i]), Q_max)
        else:
            order_size = Q_max

        s_max = s_min + order_size
        result.append((s_min, s_max))

    return tuple(result)
