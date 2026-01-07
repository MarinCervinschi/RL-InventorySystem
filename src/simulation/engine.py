from typing import List, Optional, Tuple

import numpy as np
import simpy

from src.mdp.action import Action
from src.mdp.state import State, create_state
from src.simulation.customer import CustomerGenerator
from src.simulation.logger import SimulationLogger
from src.simulation.product import Product, create_default_products
from src.simulation.supplier import SupplierManager
from src.simulation.warehouse import Warehouse


class InventorySimulation:
    """
    Inventory management simulation

    Components:
    - Warehouse: Manages inventory state
    - SupplierManager: Handles order placement and delivery
    - CustomerGenerator: Generates customer arrivals and demands
    - EventLogger: Tracks statistics
    """

    def __init__(
        self,
        products: Optional[List[Product]] = None,
        random_state: Optional[np.random.Generator] = None,
    ):
        """
        Initialize simulation with modular components.

        Args:
            products: List of products (defaults to assignment products)
            random_state: Random number generator
        """
        self.rng = random_state or np.random.default_rng()

        # Initialize products (defaults to assignment specification)
        if products is None:
            self.products = list(create_default_products())
        else:
            self.products = products

        self.num_products = len(self.products)

        # Initialize modular components
        self.warehouse = Warehouse(num_products=self.num_products)
        self.logger = SimulationLogger(num_products=self.num_products)

        # SimPy environment (created in reset)
        self.env: Optional[simpy.Environment] = None
        self.supplier_manager: Optional[SupplierManager] = None
        self.customer_generator: Optional[CustomerGenerator] = None

        # Current day counter
        self.current_day = 0

    def reset(self, initial_state: Optional[State] = None) -> None:
        """
        Reset simulation to initial state.

        Args:
            initial_state: Initial state (defaults to zero inventory and orders)
        """
        # Create new SimPy environment
        self.env = simpy.Environment()

        if initial_state is None:
            initial_state = create_state()

        # Reset warehouse state
        for i in range(self.num_products):
            self.warehouse.set_inventory(i, initial_state.net_inventory[i])
            self.warehouse.set_outstanding_orders(
                i, initial_state.outstanding_orders[i]
            )

        # Initialize supplier manager (needs env)
        self.supplier_manager = SupplierManager(
            products=self.products,
            warehouse=self.warehouse,
            env=self.env,
            rng=self.rng,
        )

        # Initialize customer generator
        self.customer_generator = CustomerGenerator(
            products=self.products,
            rng=self.rng,
        )

        # Reset logger
        self.logger.reset()

        # Reset counters
        self.current_day = 0

        # Start customer arrival process
        self.env.process(self._customer_arrival_process())

    def execute_daily_decision(self, action: Action) -> Tuple[State, dict]:
        """
        Execute one decision epoch (one day).

        Steps:
        1. Place orders (if any) via SupplierManager
        2. Run simulation for one day
        3. Process all events (customers via CustomerGenerator)
        4. Return new state from Warehouse

        Args:
            action: Ordering decision

        Returns:
            (next_state, info_dict)
        """
        assert self.env is not None, "Must call reset() first"
        assert self.supplier_manager is not None, "Supplier manager not initialized"

        # Start new day in logger
        net_inv, outstanding = self.warehouse.get_state_as_dict()
        self.logger.start_new_day(self.current_day, net_inv, outstanding)

        # Step 1: Place orders via SupplierManager
        for product_id in range(self.num_products):
            quantity = action.order_quantities[product_id]
            if quantity > 0:
                self.supplier_manager.place_order(product_id, quantity)
                self.logger.log_order_placement(self.env.now, product_id, quantity)

        # Step 2: Run simulation for 1 day
        target_time = self.env.now + 1.0
        self.env.run(until=target_time)

        # Step 3: Get state from Warehouse
        next_state = self.get_current_state()

        # Step 4: Build info dict from logger
        daily_stats = self.logger.get_current_day_stats()
        info = {
            "num_customers": daily_stats["num_customers"],
            "total_demand": tuple(
                daily_stats["total_demand"].get(i, 0) for i in range(self.num_products)
            ),
            "num_order_arrivals": tuple(
                daily_stats["num_order_arrivals"].get(i, 0)
                for i in range(self.num_products)
            ),
            "net_inventory": net_inv,
            "outstanding": outstanding,
        }

        # Increment day counter
        self.current_day += 1

        return next_state, info

    def _customer_arrival_process(self):
        """
        SimPy process generating customer arrivals.

        Uses CustomerGenerator component to generate arrivals and demands.
        """
        assert self.env is not None, "SimPy environment not initialized"
        assert self.customer_generator is not None, "Customer generator not initialized"

        while True:
            # Wait for next customer (via CustomerGenerator)
            inter_arrival = self.customer_generator.sample_interarrival_time()
            yield self.env.timeout(inter_arrival)

            # Generate customer demands (via CustomerGenerator)
            demands_dict = self.customer_generator.generate_demands()

            # Fulfill demands via Warehouse
            for product_id, demand in demands_dict.items():
                self.warehouse.fulfill_demand(product_id, demand)

            # Log customer arrival
            self.logger.log_customer_arrival(self.env.now, demands_dict)

    def get_current_state(self) -> State:
        """Get current system state from Warehouse."""
        net_inv, outstanding = self.warehouse.get_state()
        return create_state(
            net_inventory_0=net_inv[0],
            net_inventory_1=net_inv[1],
            outstanding_0=outstanding[0],
            outstanding_1=outstanding[1],
        )
