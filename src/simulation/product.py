from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Product:
    """
    Represents a product type with its specific characteristics.

    Encapsulates:
    - Demand distribution parameters
    - Lead time distribution parameters
    - Product identification

    Immutable by design (frozen=True) for safety in concurrent simulations.
    """

    product_id: int
    demand_values: Tuple[int, ...]
    demand_probabilities: Tuple[float, ...]
    lead_time_min: float
    lead_time_max: float

    def sample_demand(self, rng: np.random.Generator) -> int:
        """
        Sample a demand quantity according to the product's distribution.

        Args:
            rng: Random number generator

        Returns:
            Demand quantity (number of units)
        """
        return int(rng.choice(self.demand_values, p=self.demand_probabilities))

    def sample_lead_time(self, rng: np.random.Generator) -> float:
        """
        Sample a lead time according to the product's distribution.

        Args:
            rng: Random number generator

        Returns:
            Lead time in days (continuous)
        """
        return float(rng.uniform(self.lead_time_min, self.lead_time_max))

    @property
    def expected_demand(self) -> float:
        """Calculate expected demand value."""
        return sum(d * p for d, p in zip(self.demand_values, self.demand_probabilities))

    @property
    def expected_lead_time(self) -> float:
        """Calculate expected lead time (midpoint for uniform distribution)."""
        return (self.lead_time_min + self.lead_time_max) / 2.0

    def __repr__(self) -> str:
        return (
            f"Product(id={self.product_id}, "
            f"E[D]={self.expected_demand:.2f}, "
            f"E[L]={self.expected_lead_time:.2f})"
        )


def create_product_0() -> Product:
    """
    Create Product 0 with assignment specifications.

    Demand: {1:1/6, 2:1/3, 3:1/3, 4:1/6}
    Lead Time: Uniform(0.5, 1.0) days
    """
    return Product(
        product_id=0,
        demand_values=(1, 2, 3, 4),
        demand_probabilities=(1 / 6, 1 / 3, 1 / 3, 1 / 6),
        lead_time_min=15.0,
        lead_time_max=30.0,
    )


def create_product_1() -> Product:
    """
    Create Product 1 with assignment specifications.

    Demand: {5:1/8, 4:1/2, 3:1/4, 2:1/8}
    Lead Time: Uniform(0.2, 0.7) days
    """
    return Product(
        product_id=1,
        demand_values=(5, 4, 3, 2),
        demand_probabilities=(1 / 8, 1 / 2, 1 / 4, 1 / 8),
        lead_time_min=6.0,
        lead_time_max=21.0,
    )


def create_default_products() -> Tuple[Product, Product]:
    """Create both default products for the assignment."""
    return create_product_0(), create_product_1()
