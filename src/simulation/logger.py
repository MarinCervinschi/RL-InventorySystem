from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DailyStatistics:
    """Statistics for a single simulation day."""

    day: int
    num_customers: int
    total_demand_per_product: Dict[int, int]
    num_order_arrivals_per_product: Dict[int, int]
    net_inventory_per_product: Dict[int, int]
    outstanding_per_product: Dict[int, int]


class SimulationLogger:
    """
    Event logger for simulation statistics.

    Collects statistics for daily reporting and analysis.
    """

    def __init__(self, num_products: int):
        """
        Initialize event logger.

        Args:
            num_products: Number of products to track
        """
        self.num_products = num_products
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self._current_day = 0
        self._num_customers_today = 0
        self._total_demand_today: Dict[int, int] = {
            i: 0 for i in range(self.num_products)
        }
        self._num_order_arrivals_today: Dict[int, int] = {
            i: 0 for i in range(self.num_products)
        }
        self._history: List[DailyStatistics] = []

    def log_customer_arrival(self, time: float, demands: Dict[int, int]) -> None:
        """Log a customer arrival with their demands."""
        self._num_customers_today += 1
        for product_id, quantity in demands.items():
            self._total_demand_today[product_id] += quantity

    def log_order_placement(self, time: float, product_id: int, quantity: int) -> None:
        """Log an order being placed."""
        # Can be extended to track order placements if needed
        pass

    def log_order_delivery(self, time: float, product_id: int, quantity: int) -> None:
        """Log an order delivery."""
        self._num_order_arrivals_today[product_id] += 1

    def start_new_day(
        self, day: int, net_inventory: Dict[int, int], outstanding: Dict[int, int]
    ) -> None:
        """
        Start a new day and save statistics from previous day.

        Args:
            day: Day number
            net_inventory: Current net inventory levels
            outstanding: Current outstanding orders
        """
        # Save previous day statistics if not first day
        if day > 0 or self._num_customers_today > 0:
            stats = DailyStatistics(
                day=self._current_day,
                num_customers=self._num_customers_today,
                total_demand_per_product=dict(self._total_demand_today),
                num_order_arrivals_per_product=dict(self._num_order_arrivals_today),
                net_inventory_per_product=dict(net_inventory),
                outstanding_per_product=dict(outstanding),
            )
            self._history.append(stats)

        # Reset daily counters
        self._current_day = day
        self._num_customers_today = 0
        self._total_demand_today = {i: 0 for i in range(self.num_products)}
        self._num_order_arrivals_today = {i: 0 for i in range(self.num_products)}

    def get_current_day_stats(self) -> Dict[str, Any]:
        """Get statistics for the current day (in progress)."""
        return {
            "num_customers": self._num_customers_today,
            "total_demand": dict(self._total_demand_today),
            "num_order_arrivals": dict(self._num_order_arrivals_today),
        }

    def get_daily_statistics(self, day: int) -> DailyStatistics:
        """Get statistics for a specific completed day."""
        if day < 0 or day >= len(self._history):
            raise ValueError(f"Day {day} not in history")
        return self._history[day]

    def get_all_statistics(self) -> List[DailyStatistics]:
        """Get all historical statistics."""
        return list(self._history)

    def __repr__(self) -> str:
        return (
            f"SimulationLogger(current_day={self._current_day}, "
            f"customers_today={self._num_customers_today}, "
            f"history_days={len(self._history)})"
        )
