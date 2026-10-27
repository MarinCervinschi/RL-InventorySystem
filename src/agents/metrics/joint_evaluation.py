from __future__ import annotations

from statistics import NormalDist
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.environment.product_control import ProductPolicy, product_costs
from src.mdp import Action
from src.simulation.streams import StreamedInventorySimulation

DEFAULT_WARMUP = 200  # days discarded before steady-state statistics

JointPolicy = Tuple[ProductPolicy, ProductPolicy]


def run_joint_episode(
    policies: JointPolicy,
    seed: int,
    episode_length: int = 1000,
) -> Dict[str, np.ndarray]:
    """
    Simulate the full two-product system with one policy per product.

    Uses `StreamedInventorySimulation`: with the same seed every policy faces
    the same customers and the same lead-time sequence (common random numbers).

    Args:
        policies: (policy for product 0, policy for product 1)
        seed: Episode seed
        episode_length: Number of days

    Returns:
        Per-day arrays. Keys `q0`, `q1`, `net_inv_0`, `net_inv_1`,
        `total_daily_cost` are compatible with `EvaluationMetrics` plots.
    """
    if [p.product_id for p in policies] != [0, 1]:
        raise ValueError("policies must be ordered as (product 0, product 1)")

    sim = StreamedInventorySimulation(seed=seed)
    sim.reset()
    warehouse = sim.warehouse
    for policy in policies:
        policy.reset()

    keys = [
        "q",
        "net_inv",
        "outstanding",
        "ordering_cost",
        "holding_cost",
        "shortage_cost",
        "demand",
        "served",
    ]
    data: Dict[str, List[List[float]]] = {k: [[], []] for k in keys}
    state = sim.get_current_state()

    for _ in range(episode_length):
        quantities = tuple(policy.act(state) for policy in policies)
        demanded_before = dict(warehouse.units_demanded)
        served_before = dict(warehouse.units_served)

        state, _ = sim.execute_daily_decision(Action(quantities))

        for j in range(2):
            costs = product_costs(state, j, quantities[j])
            data["q"][j].append(quantities[j])
            data["net_inv"][j].append(state.net_inventory[j])
            data["outstanding"][j].append(state.outstanding_orders[j])
            data["ordering_cost"][j].append(costs.ordering_cost)
            data["holding_cost"][j].append(costs.holding_cost)
            data["shortage_cost"][j].append(costs.shortage_cost)
            data["demand"][j].append(warehouse.units_demanded[j] - demanded_before[j])
            data["served"][j].append(warehouse.units_served[j] - served_before[j])

    episode: Dict[str, np.ndarray] = {}
    for k in keys:
        for j in range(2):
            episode[f"{k}_{j}"] = np.asarray(data[k][j], dtype=float)
    episode["q0"], episode["q1"] = episode["q_0"], episode["q_1"]
    for j in range(2):
        episode[f"cost_{j}"] = (
            episode[f"ordering_cost_{j}"]
            + episode[f"holding_cost_{j}"]
            + episode[f"shortage_cost_{j}"]
        )
    episode["total_daily_cost"] = episode["cost_0"] + episode["cost_1"]
    return episode


def _t_quantile(p: float, df: int) -> float:
    """
    Student-t quantile via Cornish-Fisher expansion (no scipy dependency).

    At p = 0.975 the error is < 1e-3 for df >= 9 (df=19: 2.0930) and < 0.01 for df >= 4.
    """
    z = NormalDist().inv_cdf(p)
    return (
        z
        + (z**3 + z) / (4 * df)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
    )


def _mean_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Mean and half-width of a Student-t confidence interval across replications."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, float("nan")
    half = (
        _t_quantile(0.5 + confidence / 2, n - 1) * np.std(values, ddof=1) / np.sqrt(n)
    )
    return mean, float(half)


def summarize_joint_episodes(
    episodes: Sequence[Dict[str, np.ndarray]],
    warmup: int = DEFAULT_WARMUP,
) -> Dict[str, float]:
    """
    Steady-state statistics over independent replications.

    Each replication contributes one observation (its mean daily cost after
    `warmup` days); CIs are Student-t over replications.

    Service levels per product (steady state):
        fill_rate   = units served from stock / units demanded
        no_stockout = fraction of days ending without backorders
    """
    summary: Dict[str, float] = {"n_reps": len(episodes), "warmup": warmup}

    per_rep = np.array([np.mean(ep["total_daily_cost"][warmup:]) for ep in episodes])
    summary["cost_mean"], summary["cost_ci"] = _mean_ci(per_rep)

    for j in range(2):
        rep_cost = np.array([np.mean(ep[f"cost_{j}"][warmup:]) for ep in episodes])
        summary[f"p{j}_cost_mean"], summary[f"p{j}_cost_ci"] = _mean_ci(rep_cost)
        for comp in ("ordering", "holding", "shortage"):
            summary[f"p{j}_{comp}"] = float(
                np.mean([np.mean(ep[f"{comp}_cost_{j}"][warmup:]) for ep in episodes])
            )
        demand = sum(ep[f"demand_{j}"][warmup:].sum() for ep in episodes)
        served = sum(ep[f"served_{j}"][warmup:].sum() for ep in episodes)
        summary[f"p{j}_fill_rate"] = (
            float(served / demand) if demand > 0 else float("nan")
        )
        summary[f"p{j}_no_stockout"] = float(
            np.mean([np.mean(ep[f"net_inv_{j}"][warmup:] >= 0) for ep in episodes])
        )
        summary[f"p{j}_orders_per_100d"] = float(
            100 * np.mean([np.mean(ep[f"q_{j}"][warmup:] > 0) for ep in episodes])
        )
    return summary


def compare_joint_policies(
    configurations: Mapping[str, JointPolicy],
    seeds: Sequence[int],
    episode_length: int = 1000,
    warmup: int = DEFAULT_WARMUP,
    reference: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, np.ndarray]]]]:
    """
    Evaluate several joint policies on the same seeds (common random numbers).

    Args:
        configurations: name -> (policy product 0, policy product 1)
        seeds: Episode seeds, shared by all configurations
        episode_length: Days per episode
        warmup: Days discarded at the start of each episode
        reference: Configuration used for paired differences (default: first)
        verbose: Print progress

    Returns:
        (summary table, raw episodes per configuration). The table includes
        the paired difference in mean daily cost vs `reference`, whose CI is
        much tighter than the unpaired CIs thanks to common random numbers.
    """
    episodes: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for name, policies in configurations.items():
        if verbose:
            print(f"Evaluating {name} on {len(seeds)} seeds...")
        episodes[name] = [run_joint_episode(policies, s, episode_length) for s in seeds]

    return summarize_comparison(episodes, warmup, reference), episodes


def summarize_comparison(
    episodes: Mapping[str, Sequence[Dict[str, np.ndarray]]],
    warmup: int = DEFAULT_WARMUP,
    reference: Optional[str] = None,
) -> pd.DataFrame:
    """
    Summary table of episodes produced on the same seeds by several policies.

    Can be recomputed with a different warmup without re-simulating.

    Args:
        episodes: name -> episodes (same seeds, same order for every policy)
        warmup: Days discarded at the start of each episode
        reference: Configuration used for paired differences (default: first)
    """
    reference = reference or next(iter(episodes))
    ref_rep = np.array(
        [np.mean(ep["total_daily_cost"][warmup:]) for ep in episodes[reference]]
    )

    rows = []
    for name, eps in episodes.items():
        row = {"policy": name, **summarize_joint_episodes(eps, warmup)}
        rep = np.array([np.mean(ep["total_daily_cost"][warmup:]) for ep in eps])
        row["diff_vs_ref_mean"], row["diff_vs_ref_ci"] = _mean_ci(rep - ref_rep)
        rows.append(row)

    return pd.DataFrame(rows).set_index("policy")


def format_comparison(table: pd.DataFrame) -> str:
    """Render the comparison table as Markdown."""
    lines = [
        "| Policy | $/day ± CI95 | Δ vs ref ± CI95 | P0 $/day (ord/hold/short) "
        "| P1 $/day (ord/hold/short) | Fill rate P0/P1 | No-stockout days P0/P1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in table.iterrows():
        prods = [
            f"{r[f'p{j}_cost_mean']:.2f} ({r[f'p{j}_ordering']:.1f}/"
            f"{r[f'p{j}_holding']:.1f}/{r[f'p{j}_shortage']:.1f})"
            for j in range(2)
        ]
        lines.append(
            f"| {name} | {r['cost_mean']:.2f} ± {r['cost_ci']:.2f} "
            f"| {r['diff_vs_ref_mean']:+.2f} ± {r['diff_vs_ref_ci']:.2f} "
            f"| {prods[0]} | {prods[1]} "
            f"| {r['p0_fill_rate']:.1%} / {r['p1_fill_rate']:.1%} "
            f"| {r['p0_no_stockout']:.1%} / {r['p1_no_stockout']:.1%} |"
        )
    return "\n".join(lines)
