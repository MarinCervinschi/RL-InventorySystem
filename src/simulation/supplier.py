from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import simpy

from src.simulation.product import Product
from src.simulation.warehouse import Warehouse


@dataclass
class PendingOrder:
    """Simple data structure for tracking pending orders."""

    order_id: int
    product_id: int
    quantity: int
    lead_time: float


class SupplierManager:
    """
    Manages interactions with suppliers.
    """

    def __init__(
        self,
        products: List[Product],
        warehouse: Warehouse,
        env: simpy.Environment,
        rng: np.random.Generator,
    ):
        """
        Initialize supplier manager.

        Args:
            products: List of products managed
            warehouse: Warehouse to deliver to
            env: SimPy environment for scheduling
            rng: Random number generator for lead times
        """
        self.products = {p.product_id: p for p in products}
        self.warehouse = warehouse
        self.env = env
        self.rng = rng

        # Track active orders
        self._pending_orders: Dict[int, List[PendingOrder]] = {
            i: [] for i in range(len(products))
        }
        self._next_order_id = 0

        # Statistics
        self.total_orders_placed = 0
        self.total_orders_delivered = 0

    def place_order(self, product_id: int, quantity: int) -> None:
        """
        Place a replenishment order with the supplier.

        Args:
            product_id: Which product to order
            quantity: How many units to order
        """
        if product_id not in self.products:
            raise ValueError(f"Unknown product ID: {product_id}")

        if quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {quantity}")

        # Get product info
        product = self.products[product_id]

        # Sample lead time
        lead_time = product.sample_lead_time(self.rng)

        # Create order
        order = PendingOrder(
            order_id=self._next_order_id,
            product_id=product_id,
            quantity=quantity,
            lead_time=lead_time,
        )
        self._next_order_id += 1

        # Update warehouse outstanding
        self.warehouse.place_order(product_id, quantity)

        # Track order
        self._pending_orders[product_id].append(order)

        # Schedule delivery
        self.env.process(self._delivery_process(order))

        # Update statistics
        self.total_orders_placed += 1

    def _delivery_process(self, order: PendingOrder):
        """
        SimPy process for order delivery.

        Args:
            order: Order to deliver
        """
        # Wait for lead time
        yield self.env.timeout(order.lead_time)

        # Deliver to warehouse
        self.warehouse.receive_shipment(order.product_id, order.quantity)

        # Remove from pending
        self._pending_orders[order.product_id].remove(order)

        # Update statistics
        self.total_orders_delivered += 1

    def get_pending_orders(
        self, product_id: Optional[int] = None
    ) -> List[PendingOrder]:
        """
        Get list of pending orders.

        Args:
            product_id: If provided, only return orders for this product

        Returns:
            List of pending orders
        """
        if product_id is not None:
            if product_id not in self.products:
                raise ValueError(f"Unknown product ID: {product_id}")
            return list(self._pending_orders[product_id])
        else:
            # Return all pending orders
            all_orders = []
            for orders in self._pending_orders.values():
                all_orders.extend(orders)
            return all_orders

    def get_total_outstanding_quantity(self, product_id: int) -> int:
        """
        Get total quantity of outstanding orders for a product.

        Args:
            product_id: Product identifier

        Returns:
            Total outstanding quantity
        """
        return self.warehouse.get_outstanding_orders(product_id)

    def __repr__(self) -> str:
        pending_count = sum(len(orders) for orders in self._pending_orders.values())
        return (
            f"SupplierManager("
            f"placed={self.total_orders_placed}, "
            f"delivered={self.total_orders_delivered}, "
            f"pending={pending_count})"
        )
