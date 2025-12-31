from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import simpy
from src.mdp.state import Observation, create_observation
from src.mdp.action import InventoryAction


@dataclass(frozen=True)
class SystemParameters:
    """
    Stochastic system parameters from assignment specification.

    Demand distributions:
    - Product 0: {1:1/6, 2:1/3, 3:1/3, 4:1/6}
    - Product 1: {5:1/8, 4:1/2, 3:1/4, 2:1/8}

    Lead times:
    - Product 0: Uniform(0.5, 1.0) days
    - Product 1: Uniform(0.2, 0.7) days

    Customer arrivals:
    - Exponential with λ = 0.1 (mean inter-arrival = 10 days)
    """

    # Customer arrival rate
    lambda_arrival: float = 0.1

    # Demand distributions
    demand_0_values: Tuple[int, ...] = (1, 2, 3, 4)
    demand_0_probs: Tuple[float, ...] = (1 / 6, 1 / 3, 1 / 3, 1 / 6)

    demand_1_values: Tuple[int, ...] = (5, 4, 3, 2)
    demand_1_probs: Tuple[float, ...] = (1 / 8, 1 / 2, 1 / 4, 1 / 8)

    # Lead time distributions
    lead_time_0_min: float = 0.5
    lead_time_0_max: float = 1.0

    lead_time_1_min: float = 0.2
    lead_time_1_max: float = 0.7

    @staticmethod
    def create_default() -> "SystemParameters":
        """Create default system parameters."""
        return SystemParameters()


class InventorySimulation:
    """
    SimPy-based discrete event simulation for inventory management.

    Simulates:
    - Customer arrivals and demand
    - Order placement
    - Order arrivals (after lead time)
    - Inventory updates
    """

    def __init__(
        self,
        params: SystemParameters,
        random_state: Optional[np.random.Generator] = None,
    ):
        """
        Initialize simulation.

        Args:
            params: System parameters
            random_state: Random number generator
        """
        self.params = params
        self.rng = random_state or np.random.default_rng()

        # SimPy environment (will be created in reset)
        self.env: Optional[simpy.Environment] = None

        # Current physical state (not frame-stacked)
        self.net_inventory: List[int] = [0, 0]
        self.outstanding_orders: List[int] = [0, 0]

        # Statistics for current day
        self.num_customers_today = 0
        self.total_demand_today = [0, 0]
        self.num_order_arrivals_today = [0, 0]

    def reset(self, initial_observation: Observation):
        """
        Reset simulation to initial state.

        Args:
            initial_observation: Starting observation
        """
        # Create new SimPy environment
        self.env = simpy.Environment()

        # Set physical state
        self.net_inventory = list(initial_observation.net_inventory)
        self.outstanding_orders = list(initial_observation.outstanding_orders)

        # Reset statistics
        self.num_customers_today = 0
        self.total_demand_today = [0, 0]
        self.num_order_arrivals_today = [0, 0]

        # Start customer arrival process
        self.env.process(self._customer_arrival_process())

    def execute_daily_decision(
        self, action: InventoryAction
    ) -> Tuple[Observation, dict]:
        """
        Execute one decision epoch (one day).

        Steps:
        1. Place orders (if any)
        2. Run simulation for one day
        3. Process all events (customers, order arrivals)
        4. Return new observation

        Args:
            action: Ordering decision

        Returns:
            (next_observation, info_dict)
        """
        assert self.env is not None, "SimPy environment not initialized."

        # Reset daily statistics
        self.num_customers_today = 0
        self.total_demand_today = [0, 0]
        self.num_order_arrivals_today = [0, 0]

        # Step 1: Place orders
        for j in range(2):
            if action.order_quantities[j] > 0:
                self._place_order(j, action.order_quantities[j])

        # Step 2: Run simulation for 1 day
        target_time = self.env.now + 1.0
        self.env.run(until=target_time)

        # Step 3: Create observation
        next_obs = create_observation(
            net_inventory_0=self.net_inventory[0],
            net_inventory_1=self.net_inventory[1],
            outstanding_0=self.outstanding_orders[0],
            outstanding_1=self.outstanding_orders[1],
        )

        # Step 4: Return with info
        info = {
            "num_customers": self.num_customers_today,
            "total_demand": tuple(self.total_demand_today),
            "num_order_arrivals": tuple(self.num_order_arrivals_today),
            "net_inventory": tuple(self.net_inventory),
            "outstanding": tuple(self.outstanding_orders),
        }

        return next_obs, info

    def _place_order(self, product_id: int, quantity: int):
        """
        Place an order with the supplier.

        Args:
            product_id: Which product
            quantity: How many units
        """
        assert self.env is not None, "SimPy environment not initialized."

        # Increase outstanding immediately
        self.outstanding_orders[product_id] += quantity

        # Sample lead time
        if product_id == 0:
            lead_time = self.rng.uniform(
                self.params.lead_time_0_min, self.params.lead_time_0_max
            )
        else:
            lead_time = self.rng.uniform(
                self.params.lead_time_1_min, self.params.lead_time_1_max
            )

        # Schedule order arrival using SimPy process
        self.env.process(self._order_arrival_process(product_id, quantity, lead_time))

    def _order_arrival_process(self, product_id: int, quantity: int, lead_time: float):
        """
        SimPy process for order arrival.

        Args:
            product_id: Which product
            quantity: How many units
            lead_time: Delay before arrival
        """
        assert self.env is not None, "SimPy environment not initialized."
        
        # Wait for lead time
        yield self.env.timeout(lead_time)

        # Order arrives
        self.outstanding_orders[product_id] -= quantity

        # Add to net inventory
        # Note: If net_inventory < 0 (backorders exist), arriving units
        # first satisfy backorders, then add to on-hand
        self.net_inventory[product_id] += quantity

        # Track statistics
        self.num_order_arrivals_today[product_id] += 1

    def _customer_arrival_process(self):
        """
        SimPy process generating customer arrivals.

        Customers arrive according to exponential distribution.
        Each customer demands from both products.
        """
        assert self.env is not None, "SimPy environment not initialized."

        while True:
            # Wait for next customer
            inter_arrival = self.rng.exponential(1.0 / self.params.lambda_arrival)
            yield self.env.timeout(inter_arrival)

            # Customer arrives - demand both products
            demand_0 = self.rng.choice(
                self.params.demand_0_values, p=self.params.demand_0_probs
            )
            demand_1 = self.rng.choice(
                self.params.demand_1_values, p=self.params.demand_1_probs
            )

            # Fulfill demand (reduce net inventory)
            # If net_inventory goes negative, those are backorders
            self.net_inventory[0] -= demand_0
            self.net_inventory[1] -= demand_1

            # Track statistics
            self.num_customers_today += 1
            self.total_demand_today[0] += demand_0
            self.total_demand_today[1] += demand_1

    def get_current_observation(self) -> Observation:
        """Get current system observation."""
        return create_observation(
            net_inventory_0=self.net_inventory[0],
            net_inventory_1=self.net_inventory[1],
            outstanding_0=self.outstanding_orders[0],
            outstanding_1=self.outstanding_orders[1],
        )
