"""Evaluation plots for RL agents."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from typing import List, Dict, Optional
from dataclasses import dataclass
from src.mdp import CostParameters


class EvaluationMetrics:
    """Class for generating evaluation-related plots for RL agents."""

    def __init__(self, figsize: tuple = (12, 5)):
        """
        Initialize EvaluationPlots.

        Args:
            figsize: Default figure size for plots.
            cost_params: Cost parameters for the inventory system.
        """
        self.figsize = figsize
        self.cost_params = CostParameters()

    def plot_welch_procedure(
        self,
        test_episodes: List[Dict],
        window_size: int = 25,
        title: str = "Warm-up Period Analysis",
        show: bool = True,
    ) -> tuple:
        """
        Plot Welch's Graphical Procedure for warm-up detection.

        Args:
            test_episodes: List of episode data dictionaries with 'total_daily_cost'.
            window: Moving average window size.
            title: Plot title.
            show: Whether to display the plot.

        Returns:
            Tuple of (figure, n_days, n_reps, Y_smoothed).
        """
        # Extract daily costs into a 2D array: days x replications
        daily_costs_matrix = np.array(
            [ep["total_daily_cost"] for ep in test_episodes]
        ).T

        n_days = daily_costs_matrix.shape[0]
        n_reps = daily_costs_matrix.shape[1]

        # Average across replications for each day
        Y_bar = np.mean(daily_costs_matrix, axis=1)

        # Apply moving average
        Y_smoothed = np.convolve(Y_bar, np.ones(window_size) / window_size, mode="valid")
        smoothed_days = np.arange(len(Y_smoothed)) + (window_size - 1) // 2

        # Steady-state mean (last 20% of smoothed data)
        ss_window_len = int(len(Y_smoothed) * 0.2) 
        if ss_window_len < 10: ss_window_len = 10 # Minimo 10 giorni
        
        ss_data = Y_smoothed[-ss_window_len:]
        ss_mean = np.mean(ss_data)
        ss_std = np.std(ss_data)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Raw average
        ax = axes[0]
        ax.plot(range(n_days), Y_bar, "b-", linewidth=1, alpha=0.7, label="Mean Daily Cost")
        ax.set_xlabel("Day (Time Step)", fontsize=11)
        ax.set_ylabel("Average Daily Cost ($)", fontsize=11)
        ax.set_title("Average Daily Cost Across Replications", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Right: Smoothed
        ax = axes[1]
        ax.plot(smoothed_days, Y_smoothed, "b-", linewidth=2, label=f"Moving Average (w={window_size})")
        ax.axhline(
            y=ss_mean,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Steady-State Mean: ${ss_mean:.2f}",
        )
        ax.set_xlabel("Day (Time Step)", fontsize=11)
        ax.set_ylabel("Smoothed Daily Cost ($)", fontsize=11)
        ax.set_title("Welch's Procedure: Identify Warm-up Period", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if show:
            plt.show()

        # Print summary
        print("=" * 60)
        print(f"📊 Welch's Procedure Summary:")
        print("=" * 60)
        print(f"   Total replications: {n_reps}")
        print(f"   Episode length:     {n_days} days")
        print(f"   Moving average window:    {window_size}\n")

        
        print("=" * 60)
        print(f"📈 Results & Convergence:")
        print("=" * 60)
        print(f"   Steady-State Mean: ${ss_mean:.2f}")
        print(f"   Stability (Std Dev):    {ss_std:.2f} (Lower is better/flatter)")
   
        print(f"\n⚠️  Visually inspect the right plot to identify where the curve flattens.")

        return n_days, n_reps

    def plot_cost_breakdown_by_product(
        self,
        test_episodes: List[Dict],
        warmup_length: int,
        n_days: int,
        title: str = "Cost Component Breakdown by Product",
        show: bool = True,
    ) -> None:
        """
        Plot cost component breakdown separated by product.

        Args:
            test_episodes: List of episode data dictionaries.
            warmup_length: Number of warm-up days to exclude.
            n_days: Total number of days per episode.
            title: Plot title.
            show: Whether to display the plot.

        Returns:
            matplotlib Figure object.
        """
        costs_p1, costs_p2 = self._compute_costs_by_product(test_episodes, warmup_length)

        # Calculate means
        p1_ord = np.mean(costs_p1["ordering"])
        p1_hold = np.mean(costs_p1["holding"])
        p1_short = np.mean(costs_p1["shortage"])
        p2_ord = np.mean(costs_p2["ordering"])
        p2_hold = np.mean(costs_p2["holding"])
        p2_short = np.mean(costs_p2["shortage"])

        # Plot
        x = np.arange(3)
        width = 0.35
        labels = [
            "Ordering\n(K=$10 + i=$3)",
            "Holding\n(h=$1/unit/day)",
            "Shortage\n(π=$7/unit/day)",
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(
            x - width / 2,
            [p1_ord, p1_hold, p1_short],
            width,
            label="Product 1",
            color="steelblue",
            edgecolor="black",
        )
        bars2 = ax.bar(
            x + width / 2,
            [p2_ord, p2_hold, p2_short],
            width,
            label="Product 2",
            color="forestgreen",
            edgecolor="black",
        )

        ax.set_ylabel("Average Cost per Episode ($)", fontsize=12)
        ax.set_title(
            f"{title} (Steady-State, days {warmup_length+1}-{n_days})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(
                f"${height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(
                f"${height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()

        if show:
            plt.show()

        # Print totals
        total_p1 = p1_ord + p1_hold + p1_short
        total_p2 = p2_ord + p2_hold + p2_short
        total_all = total_p1 + total_p2
        
        print("=" * 60)
        print(f"📊 Steady-State Average Cost per Episode (days {warmup_length+1}-{n_days}):")
        print("=" * 60)
        print(f"\n  Product 1: ${total_p1:.2f}")
        print(f"    Ordering: ${p1_ord:.2f} ({p1_ord/total_p1*100:.1f}%)")
        print(f"    Holding:  ${p1_hold:.2f} ({p1_hold/total_p1*100:.1f}%)")
        print(f"    Shortage: ${p1_short:.2f} ({p1_short/total_p1*100:.1f}%)")
        print(f"\n  Product 2: ${total_p2:.2f}")
        print(f"    Ordering: ${p2_ord:.2f} ({p2_ord/total_p2*100:.1f}%)")
        print(f"    Holding:  ${p2_hold:.2f} ({p2_hold/total_p2*100:.1f}%)")
        print(f"    Shortage: ${p2_short:.2f} ({p2_short/total_p2*100:.1f}%)\n")
        print("-" * 60)
        print(f"\n  TOTAL: ${total_all:.2f}")

    def _compute_costs_by_product(
        self, episodes: List[Dict], warmup: int
    ) -> tuple:
        """Compute cost components separately for each product."""
        costs_p1 = {"ordering": [], "holding": [], "shortage": []}
        costs_p2 = {"ordering": [], "holding": [], "shortage": []}
        params = self.cost_params

        for ep in episodes:
            q0_ss = ep["q0"][warmup:]
            q1_ss = ep["q1"][warmup:]
            inv0_ss = ep["net_inv_0"][warmup:]
            inv1_ss = ep["net_inv_1"][warmup:]

            # Product 1 costs
            ord_p1 = sum(params.K + params.i * q for q in q0_ss if q > 0)
            hold_p1 = sum(params.h * max(0, inv) for inv in inv0_ss)
            short_p1 = sum(params.pi * max(0, -inv) for inv in inv0_ss)

            # Product 2 costs
            ord_p2 = sum(params.K + params.i * q for q in q1_ss if q > 0)
            hold_p2 = sum(params.h * max(0, inv) for inv in inv1_ss)
            short_p2 = sum(params.pi * max(0, -inv) for inv in inv1_ss)

            costs_p1["ordering"].append(ord_p1)
            costs_p1["holding"].append(hold_p1)
            costs_p1["shortage"].append(short_p1)
            costs_p2["ordering"].append(ord_p2)
            costs_p2["holding"].append(hold_p2)
            costs_p2["shortage"].append(short_p2)

        return costs_p1, costs_p2

    def plot_operational_timeseries(
        self,
        episode_data: Dict,
        title: str = "Operational Time Series",
        show: bool = True,
    ) -> None:
        """
        Plot operational time series showing inventory, orders, and demand.

        Args:
            episode_data: Episode data dictionary with inventory, order, and demand data.
            title: Plot title.
            show: Whether to display the plot.

        Returns:
            matplotlib Figure object.
        """
        days = np.arange(len(episode_data["net_inv_0"]))
        has_demand = "demand_0" in episode_data and "demand_1" in episode_data

        n_rows = 3 if has_demand else 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), sharex=True)

        # Product 1 - Net Inventory
        ax = axes[0, 0]
        ax.plot(days, episode_data["net_inv_0"], "b-", linewidth=1.5)
        ax.axhline(y=0, color="k", linestyle="-", linewidth=2, label="Stockout threshold")
        ax.fill_between(
            days,
            episode_data["net_inv_0"],
            0,
            where=[x < 0 for x in episode_data["net_inv_0"]],
            color="red",
            alpha=0.3,
            label="Backlog",
        )
        ax.set_ylabel("Net Inventory (P1)", fontsize=11)
        ax.set_title("Product 1: Inventory Level", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Product 1 - Replenishment
        ax = axes[1, 0]
        ax.bar(days, episode_data["q0"], color="steelblue", alpha=0.8, width=1.0)
        ax.set_ylabel("Order Quantity (P1)", fontsize=11)
        ax.set_title("Product 1: Replenishment Actions", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if not has_demand:
            ax.set_xlabel("Day", fontsize=11)

        # Product 2 - Net Inventory
        ax = axes[0, 1]
        ax.plot(days, episode_data["net_inv_1"], "g-", linewidth=1.5)
        ax.axhline(y=0, color="k", linestyle="-", linewidth=2, label="Stockout threshold")
        ax.fill_between(
            days,
            episode_data["net_inv_1"],
            0,
            where=[x < 0 for x in episode_data["net_inv_1"]],
            color="red",
            alpha=0.3,
            label="Backlog",
        )
        ax.set_ylabel("Net Inventory (P2)", fontsize=11)
        ax.set_title("Product 2: Inventory Level", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Product 2 - Replenishment
        ax = axes[1, 1]
        ax.bar(days, episode_data["q1"], color="forestgreen", alpha=0.8, width=1.0)
        ax.set_ylabel("Order Quantity (P2)", fontsize=11)
        ax.set_title("Product 2: Replenishment Actions", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if not has_demand:
            ax.set_xlabel("Day", fontsize=11)

        # Demand plots (if available)
        if has_demand:
            # Product 1 - Demand
            ax = axes[2, 0]
            ax.bar(days, episode_data["demand_0"], color="coral", alpha=0.8, width=1.0)
            ax.axhline(y=np.mean(episode_data["demand_0"]), color="darkred", linestyle="--", 
                      linewidth=1.5, label=f"Mean: {np.mean(episode_data['demand_0']):.1f}")
            ax.set_xlabel("Day", fontsize=11)
            ax.set_ylabel("Daily Demand (P1)", fontsize=11)
            ax.set_title("Product 1: Customer Demand", fontsize=12, fontweight="bold")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

            # Product 2 - Demand
            ax = axes[2, 1]
            ax.bar(days, episode_data["demand_1"], color="darkorange", alpha=0.8, width=1.0)
            ax.axhline(y=np.mean(episode_data["demand_1"]), color="darkred", linestyle="--", 
                      linewidth=1.5, label=f"Mean: {np.mean(episode_data['demand_1']):.1f}")
            ax.set_xlabel("Day", fontsize=11)
            ax.set_ylabel("Daily Demand (P2)", fontsize=11)
            ax.set_title("Product 2: Customer Demand", fontsize=12, fontweight="bold")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            f"{title} ({len(days)} Days)", fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        if show:
            plt.show()


    def plot_inventory_histogram(
        self,
        test_episodes: List[Dict],
        title: str = "Inventory Distribution - Risk Profile",
        show: bool = True,
    ) -> None:
        """
        Plot inventory distribution histogram.

        Args:
            test_episodes: List of episode data dictionaries.
            title: Plot title.
            show: Whether to display the plot.

        Returns:
            matplotlib Figure object.
        """
        # Collect all inventory levels
        all_inv_p1 = []
        all_inv_p2 = []
        for ep in test_episodes:
            all_inv_p1.extend(ep["net_inv_0"])
            all_inv_p2.extend(ep["net_inv_1"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Product 1 Histogram
        ax = axes[0]
        bins = np.linspace(min(all_inv_p1) - 5, max(all_inv_p1) + 5, 40)
        n, bin_edges, patches = ax.hist(all_inv_p1, bins=bins, edgecolor="black", alpha=0.7)
        for i, patch in enumerate(patches):
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            patch.set_facecolor("red" if bin_center < 0 else "green")
        ax.axvline(x=0, color="k", linestyle="-", linewidth=2)
        ax.set_xlabel("Net Inventory Level", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Product 1: Inventory Distribution", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        legend_elements = [
            Patch(facecolor="red", label="Backlog (I < 0)"),
            Patch(facecolor="green", label="On-Hand (I > 0)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        # Product 2 Histogram
        ax = axes[1]
        bins = np.linspace(min(all_inv_p2) - 5, max(all_inv_p2) + 5, 40)
        n, bin_edges, patches = ax.hist(all_inv_p2, bins=bins, edgecolor="black", alpha=0.7)
        for i, patch in enumerate(patches):
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            patch.set_facecolor("red" if bin_center < 0 else "green")
        ax.axvline(x=0, color="k", linestyle="-", linewidth=2)
        ax.set_xlabel("Net Inventory Level", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Product 2: Inventory Distribution", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if show:
            plt.show()

        # Print service level stats
        stockout_p1 = sum(1 for x in all_inv_p1 if x < 0) / len(all_inv_p1)
        stockout_p2 = sum(1 for x in all_inv_p2 if x < 0) / len(all_inv_p2)
        
        print("=" * 60)
        print(f"⚙️Service Level (% days without stockout):")
        print("=" * 60)
        print(f"  Product 1: {(1-stockout_p1)*100:.1f}%")
        print(f"  Product 2: {(1-stockout_p2)*100:.1f}%")