from typing import Dict, List, Optional

import numpy as np
import simpy

from src.mdp.state import State, sample_initial_state
from src.simulation.engine import InventorySimulation
from src.simulation.product import Product
from src.simulation.supplier import SupplierManager
from src.simulation.warehouse import Warehouse


class ServiceTrackingWarehouse(Warehouse):
    """
    Warehouse that also counts units served immediately from stock.

    Used to compute the fill rate (fraction of demand satisfied from on-hand
    inventory). Inventory dynamics are identical to `Warehouse`.
    """

    def __init__(self, num_products: int):
        super().__init__(num_products)
        self.units_demanded: Dict[int, int] = {i: 0 for i in range(num_products)}
        self.units_served: Dict[int, int] = {i: 0 for i in range(num_products)}

    def reset_counters(self) -> None:
        """Reset demand/service counters."""
        for i in range(self.num_products):
            self.units_demanded[i] = 0
            self.units_served[i] = 0

    def fulfill_demand(self, product_id: int, quantity: int) -> None:
        """Fulfill demand and record how many units came from on-hand stock."""
        on_hand = self.get_on_hand_inventory(product_id)
        super().fulfill_demand(product_id, quantity)
        self.units_demanded[product_id] += quantity
        self.units_served[product_id] += min(quantity, on_hand)


class PerProductSupplierManager(SupplierManager):
    """
    SupplierManager that samples each product's lead times from its own stream.

    With a shared stream, an extra order of product 0 would shift the lead
    times of product 1 (and the demand sequence). Separate streams keep the
    randomness of each source aligned across policies (common random numbers).
    """

    def __init__(
        self,
        products: List[Product],
        warehouse: Warehouse,
        env: simpy.Environment,
        lead_time_rngs: Dict[int, np.random.Generator],
    ):
        first_rng = next(iter(lead_time_rngs.values()))
        super().__init__(products=products, warehouse=warehouse, env=env, rng=first_rng)
        self.lead_time_rngs = lead_time_rngs

    def place_order(self, product_id: int, quantity: int) -> None:
        """Place an order, drawing the lead time from the product's stream."""
        self.rng = self.lead_time_rngs[product_id]
        super().place_order(product_id, quantity)


class StreamedInventorySimulation(InventorySimulation):
    """
    InventorySimulation with independent random streams per source of noise.

    Streams (all derived from one seed via `np.random.SeedSequence`):
    - initial state
    - customer arrivals and demand sizes
    - lead times, one stream per product

    Given the same seed, every policy sees the same customers and the k-th
    order of each product gets the same lead time, regardless of the decisions
    taken (common random numbers). Dynamics and cost logic are unchanged.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        products: Optional[List[Product]] = None,
    ):
        super().__init__(products=products)
        self.warehouse = ServiceTrackingWarehouse(num_products=self.num_products)
        self.seed(seed)

    def seed(self, seed: Optional[int]) -> None:
        """Create fresh random streams from a seed (used by the next reset)."""
        init_ss, demand_ss, *lead_ss = np.random.SeedSequence(seed).spawn(
            2 + self.num_products
        )
        self.rng = np.random.default_rng(init_ss)
        self.demand_rng = np.random.default_rng(demand_ss)
        self.lead_time_rngs = {
            p.product_id: np.random.default_rng(ss)
            for p, ss in zip(self.products, lead_ss)
        }

    def reset(self, initial_state: Optional[State] = None) -> None:
        """
        Reset the simulation and attach the dedicated streams.

        Args:
            initial_state: Initial state (default: sampled as in
                `InventoryEnvironment`, from the initial-state stream)
        """
        if initial_state is None:
            initial_state = sample_initial_state(rng=self.rng)

        super().reset(initial_state)
        assert self.env is not None and self.customer_generator is not None

        self.customer_generator.rng = self.demand_rng
        self.supplier_manager = PerProductSupplierManager(
            products=self.products,
            warehouse=self.warehouse,
            env=self.env,
            lead_time_rngs=self.lead_time_rngs,
        )
        self.warehouse.reset_counters()
