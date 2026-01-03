from typing import Dict, List

import numpy as np

from src.simulation.product import Product


class CustomerGenerator:
    """
    Generates customer arrivals according to a stochastic process.
    """

    def __init__(
        self,
        products: List[Product],
        rng: np.random.Generator,
    ):
        """
        Initialize customer generator.

        Args:
            products: List of product types to generate demand for
            rng: Random number generator
        """
        self.products = products
        self.arrival_rate = 0.1
        self.rng = rng

    def sample_interarrival_time(self) -> float:
        """
        Sample time until next customer arrival.

        Uses exponential distribution with rate λ.

        Returns:
            Inter-arrival time in days
        """
        return float(self.rng.exponential(1.0 / self.arrival_rate))

    def generate_demands(self) -> Dict[int, int]:
        """
        Generate customer demands for all products.

        Returns:
            Dictionary mapping product_id to demand quantity
        """
        return {
            product.product_id: product.sample_demand(self.rng)
            for product in self.products
        }

    @property
    def expected_interarrival_time(self) -> float:
        """Expected time between customer arrivals."""
        return 1.0 / self.arrival_rate

    def __repr__(self) -> str:
        return (
            f"CustomerGenerator(rate={self.arrival_rate}, "
            f"E[interarrival]={self.expected_interarrival_time:.2f} days)"
        )
