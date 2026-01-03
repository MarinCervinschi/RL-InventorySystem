from typing import Dict, Tuple


class Warehouse:
    """
    Manages inventory levels for multiple products.
    """

    def __init__(self, num_products: int):
        """
        Initialize warehouse.

        Args:
            num_products: Number of different product types to manage
        """
        if num_products <= 0:
            raise ValueError(f"Number of products must be positive, got {num_products}")

        self.num_products = num_products
        self._net_inventory: Dict[int, int] = {i: 0 for i in range(num_products)}
        self._outstanding_orders: Dict[int, int] = {i: 0 for i in range(num_products)}

    def get_net_inventory(self, product_id: int) -> int:
        """
        Get net inventory for a product.

        Args:
            product_id: Product identifier

        Returns:
            Net inventory (positive = on-hand, negative = backorders)
        """
        self._validate_product_id(product_id)
        return self._net_inventory[product_id]

    def get_on_hand_inventory(self, product_id: int) -> int:
        """
        Get on-hand inventory (I⁺).

        Args:
            product_id: Product identifier

        Returns:
            On-hand inventory (max(0, net_inventory))
        """
        return max(0, self.get_net_inventory(product_id))

    def get_backorders(self, product_id: int) -> int:
        """
        Get backorders (I⁻).

        Args:
            product_id: Product identifier

        Returns:
            Backorders (max(0, -net_inventory))
        """
        return max(0, -self.get_net_inventory(product_id))

    def get_outstanding_orders(self, product_id: int) -> int:
        """
        Get outstanding orders.

        Args:
            product_id: Product identifier

        Returns:
            Quantity of orders in transit
        """
        self._validate_product_id(product_id)
        return self._outstanding_orders[product_id]

    def get_inventory_position(self, product_id: int) -> int:
        """
        Get inventory position (IP = Net_Inventory + Outstanding).

        Args:
            product_id: Product identifier

        Returns:
            Inventory position
        """
        return self.get_net_inventory(product_id) + self.get_outstanding_orders(
            product_id
        )

    def set_inventory(self, product_id: int, net_inventory: int) -> None:
        """
        Set net inventory level (for initialization).

        Args:
            product_id: Product identifier
            net_inventory: Net inventory level to set
        """
        self._validate_product_id(product_id)
        self._net_inventory[product_id] = net_inventory

    def receive_shipment(self, product_id: int, quantity: int) -> None:
        """
        Receive an order shipment.

        Increases net inventory and decreases outstanding orders.
        Arriving units automatically satisfy backorders first.

        Args:
            product_id: Product identifier
            quantity: Quantity received
        """
        self._validate_product_id(product_id)
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        # Decrease outstanding
        if self._outstanding_orders[product_id] < quantity:
            raise ValueError(
                f"Cannot receive {quantity} units when only "
                f"{self._outstanding_orders[product_id]} are outstanding"
            )
        self._outstanding_orders[product_id] -= quantity

        # Increase net inventory (automatically handles backorder satisfaction)
        self._net_inventory[product_id] += quantity

    def fulfill_demand(self, product_id: int, quantity: int) -> None:
        """
        Fulfill customer demand.

        Decreases net inventory. If inventory goes negative, creates backorders.

        Args:
            product_id: Product identifier
            quantity: Demand quantity
        """
        self._validate_product_id(product_id)
        if quantity < 0:
            raise ValueError(f"Demand quantity cannot be negative, got {quantity}")

        self._net_inventory[product_id] -= quantity

    def place_order(self, product_id: int, quantity: int) -> None:
        """
        Record that an order has been placed.

        Increases outstanding orders count.

        Args:
            product_id: Product identifier
            quantity: Order quantity
        """
        self._validate_product_id(product_id)
        if quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {quantity}")

        self._outstanding_orders[product_id] += quantity

    def get_state(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Get complete warehouse state.

        Returns:
            (net_inventory_tuple, outstanding_orders_tuple)
        """
        net_inv = tuple(self._net_inventory[i] for i in range(self.num_products))
        outstanding = tuple(
            self._outstanding_orders[i] for i in range(self.num_products)
        )
        return net_inv, outstanding

    def get_state_as_dict(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Get complete warehouse state as dictionaries.
        Returns:
            (net_inventory_dict, outstanding_orders_dict)
        """

        return self._net_inventory.copy(), self._outstanding_orders.copy()

    def _validate_product_id(self, product_id: int) -> None:
        """Validate product ID is within range."""
        if not 0 <= product_id < self.num_products:
            raise ValueError(
                f"Product ID {product_id} out of range [0, {self.num_products})"
            )

    def __repr__(self) -> str:
        state_str = ", ".join(
            f"P{i}(I={self._net_inventory[i]:+3d}, O={self._outstanding_orders[i]:2d})"
            for i in range(self.num_products)
        )
        return f"Warehouse({state_str})"
