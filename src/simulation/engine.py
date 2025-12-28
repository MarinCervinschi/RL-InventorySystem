import heapq
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.mdp import InventoryAction, InventoryState

from .events import Event, EventType, OutstandingOrder, SystemParameters
from .generators import (
    DemandGenerator,
    LeadTimeGenerator,
    StandardDemandGenerator,
    StandardLeadTimeGenerator,
)


class SimulationEngine:
    """
    Core discrete event simulation engine.

    This class handles:
    1. Event queue management
    2. State transitions based on events
    3. Stochastic processes (demand, lead times)

    It does NOT make policy decisions - that's the agent's job.

    Following the Single Responsibility Principle.
    """

    def __init__(
        self,
        system_params: SystemParameters,
        demand_generator: Optional[DemandGenerator] = None,
        lead_time_generator: Optional[LeadTimeGenerator] = None,
        random_state: Optional[np.random.Generator] = None,
    ):
        """
        Initialize simulation engine.

        Args:
            system_params: System parameters
            demand_generator: Custom demand generator (optional)
            lead_time_generator: Custom lead time generator (optional)
            random_state: Numpy random generator for reproducibility
        """
        self.params = system_params
        self.params.validate()

        # Initialize random state
        self.random_state = random_state or np.random.default_rng()

        # Initialize generators (Dependency Injection pattern)
        self.demand_gen = demand_generator or StandardDemandGenerator(system_params)
        self.lead_time_gen = lead_time_generator or StandardLeadTimeGenerator(
            system_params
        )

        # Simulation state
        self.current_time = 0.0
        self.event_queue = []
        self.outstanding_orders = {0: [], 1: []}

    def set_random_state(self, random_state: np.random.Generator):
        """Set random state for reproducibility."""
        self.random_state = random_state

    def reset(self, initial_state: InventoryState):
        """
        Reset simulation to initial conditions.

        Args:
            initial_state: Starting state of the system
        """
        self.current_time = 0.0
        self.event_queue = []
        self.outstanding_orders = {0: [], 1: []}

        # Initialize with outstanding orders from initial state
        # (in case we're starting from a non-empty state)
        for product_id in range(2):
            if initial_state.outstanding_orders[product_id] > 0:
                # Approximate: assume orders arrive soon
                lead_time = self.lead_time_gen.generate_lead_time(
                    product_id, self.random_state
                )
                self._place_order_internal(
                    product_id, initial_state.outstanding_orders[product_id], lead_time
                )

    def _place_order_internal(self, product_id: int, quantity: int, lead_time: float):
        """Internal method to place an order and schedule its arrival."""
        if quantity <= 0:
            return

        arrival_time = self.current_time + lead_time

        order = OutstandingOrder(
            product_id=product_id,
            quantity=quantity,
            order_time=self.current_time,
            arrival_time=arrival_time,
        )

        self.outstanding_orders[product_id].append(order)

        # Schedule arrival event
        event = Event(
            time=arrival_time,
            event_type=EventType.ORDER_ARRIVAL,
            data={"product_id": product_id, "quantity": quantity},
        )
        heapq.heappush(self.event_queue, event)

    def _process_customer_arrival(self, state: InventoryState) -> InventoryState:
        """
        Process a customer arrival and update state.

        Customer demands both products according to their distributions.
        """
        # Generate demands
        demand_0 = self.demand_gen.generate_demand(0, self.random_state)
        demand_1 = self.demand_gen.generate_demand(1, self.random_state)

        # Process demand for each product
        new_inv = list(state.inventory_levels)
        new_bo = list(state.backorders)

        for product_id, demand in enumerate([demand_0, demand_1]):
            if new_inv[product_id] >= demand:
                # Fulfill from inventory
                new_inv[product_id] -= demand
            else:
                # Partial fulfillment + backorder
                shortage = demand - new_inv[product_id]
                new_inv[product_id] = 0
                new_bo[product_id] += shortage

        return InventoryState(
            inventory_levels=tuple(new_inv),
            backorders=tuple(new_bo),
            outstanding_orders=state.outstanding_orders,
        )

    def _process_order_arrival(
        self, state: InventoryState, product_id: int, quantity: int
    ) -> InventoryState:
        """
        Process an order arrival and update state.

        Order is used to:
        1. Fill backorders first
        2. Add remainder to inventory
        """
        new_inv = list(state.inventory_levels)
        new_bo = list(state.backorders)
        new_out = list(state.outstanding_orders)

        # Reduce outstanding orders
        new_out[product_id] = max(0, new_out[product_id] - quantity)

        # Fill backorders first
        if new_bo[product_id] > 0:
            filled = min(quantity, new_bo[product_id])
            new_bo[product_id] -= filled
            quantity -= filled

        # Add remainder to inventory
        new_inv[product_id] += quantity

        return InventoryState(
            inventory_levels=tuple(new_inv),
            backorders=tuple(new_bo),
            outstanding_orders=tuple(new_out),
        )

    def execute_daily_decision(
        self,
        current_state: InventoryState,
        action: InventoryAction,
        simulation_horizon: float = 1.0,
    ) -> Tuple[InventoryState, Dict[str, Any]]:
        """
        Execute a daily ordering decision and simulate until next decision point.

        This is the key method that advances the simulation forward one time period.

        Args:
            current_state: Current system state
            action: Ordering decision (action taken by agent)
            simulation_horizon: Time to simulate (typically 1 day)

        Returns:
            next_state: State after simulation period
            info: Dictionary with transition details
        """
        # Start from current state
        state = current_state

        # Place orders based on action
        new_out = list(state.outstanding_orders)
        for product_id, quantity in enumerate(action.order_quantities):
            if quantity > 0:
                lead_time = self.lead_time_gen.generate_lead_time(
                    product_id, self.random_state
                )
                self._place_order_internal(product_id, quantity, lead_time)
                new_out[product_id] += quantity

        # Update state with new outstanding orders
        state = InventoryState(
            inventory_levels=state.inventory_levels,
            backorders=state.backorders,
            outstanding_orders=tuple(new_out),
        )

        # Schedule customer arrivals during this period
        next_arrival_time = self.demand_gen.generate_interarrival_time(
            self.random_state
        )
        while next_arrival_time < simulation_horizon:
            event = Event(
                time=self.current_time + next_arrival_time,
                event_type=EventType.CUSTOMER_ARRIVAL,
                data={},
            )
            heapq.heappush(self.event_queue, event)

            next_arrival_time += self.demand_gen.generate_interarrival_time(
                self.random_state
            )

        # Process events until end of period
        horizon_end = self.current_time + simulation_horizon

        num_customers = 0
        num_arrivals = {0: 0, 1: 0}

        while self.event_queue and self.event_queue[0].time < horizon_end:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if event.event_type == EventType.CUSTOMER_ARRIVAL:
                state = self._process_customer_arrival(state)
                num_customers += 1

            elif event.event_type == EventType.ORDER_ARRIVAL:
                product_id = event.data["product_id"]
                quantity = event.data["quantity"]
                state = self._process_order_arrival(state, product_id, quantity)
                num_arrivals[product_id] += 1

                # Remove from outstanding orders tracking
                self.outstanding_orders[product_id] = [
                    o
                    for o in self.outstanding_orders[product_id]
                    if not (
                        o.product_id == product_id
                        and abs(o.arrival_time - event.time) < 1e-6
                    )
                ]

        # Advance time to end of period
        self.current_time = horizon_end

        # Prepare transition info
        info = {
            "num_customers": num_customers,
            "num_order_arrivals": num_arrivals,
            "simulation_time": simulation_horizon,
            "current_time": self.current_time,
        }

        return state, info
