"""
Grid search of the (s,S) parameters per product with common random numbers.

Products are independent (separate suppliers, separate K, separate random
streams), so one joint episode with the same (s,S) on both products gives
the cost of that (s,S) for each product.

Two stages, on seeds disjoint from the test seeds (0..299) and from the
DQN validation seeds (900+):
    1. coarse: every (s,S) in the grid on --coarse-seeds seeds
    2. refine: the --top best per product on --refine-seeds new seeds

Usage (from the repository root):
    uv run python -m scripts.tune_ss_policy --workers 4
"""

import argparse
import json
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.agents.metrics.joint_evaluation import DEFAULT_WARMUP, run_joint_episode
from src.environment import SSProductPolicy
from src.seeds import generate_seeds

COARSE_SEED_START = 300
REFINE_SEED_START = 400

SS = Tuple[int, int]


def evaluate_ss(
    args: Tuple[SS, Sequence[int], int, int],
) -> Tuple[SS, List[float], List[float]]:
    """Per-seed steady-state daily cost of (s,S) for product 0 and product 1."""
    (s, S), seeds, episode_length, warmup = args
    pair = (SSProductPolicy(0, s, S), SSProductPolicy(1, s, S))
    costs: Tuple[List[float], List[float]] = ([], [])
    for seed in seeds:
        ep = run_joint_episode(pair, seed, episode_length)
        for j in range(2):
            costs[j].append(float(np.mean(ep[f"cost_{j}"][warmup:])))
    return (s, S), costs[0], costs[1]


def run_grid(
    grid: Sequence[SS],
    seeds: Sequence[int],
    episode_length: int,
    warmup: int,
    workers: int,
) -> Dict[SS, Tuple[List[float], List[float]]]:
    tasks = [(ss, list(seeds), episode_length, warmup) for ss in grid]
    with Pool(workers) as pool:
        results = pool.map(evaluate_ss, tasks, chunksize=4)
    return {ss: (c0, c1) for ss, c0, c1 in results}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="(s,S) grid search with CRN")
    p.add_argument("--s-min", type=int, default=5)
    p.add_argument("--s-max", type=int, default=35)
    p.add_argument("--gap-min", type=int, default=5, help="minimum S - s")
    p.add_argument("--gap-max", type=int, default=40, help="maximum S - s")
    p.add_argument("--coarse-seeds", type=int, default=20)
    p.add_argument("--refine-seeds", type=int, default=100)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    grid = [
        (s, s + gap)
        for s in range(args.s_min, args.s_max + 1)
        for gap in range(args.gap_min, args.gap_max + 1)
    ]
    coarse_seeds = generate_seeds(args.coarse_seeds, start_index=COARSE_SEED_START)
    refine_seeds = generate_seeds(args.refine_seeds, start_index=REFINE_SEED_START)

    print(f"Coarse stage: {len(grid)} (s,S) × {len(coarse_seeds)} seeds")
    coarse = run_grid(
        grid, coarse_seeds, args.episode_length, args.warmup, args.workers
    )

    best: Dict[int, Dict] = {}
    for j in range(2):
        ranked = sorted(grid, key=lambda ss: np.mean(coarse[ss][j]))[: args.top]
        print(f"\nRefine stage P{j}: top {len(ranked)} × {len(refine_seeds)} seeds")
        refine = run_grid(
            ranked, refine_seeds, args.episode_length, args.warmup, args.workers
        )
        rows = sorted(
            (
                {
                    "s": ss[0],
                    "S": ss[1],
                    "mean": float(np.mean(refine[ss][j])),
                    "ci": float(
                        1.96
                        * np.std(refine[ss][j], ddof=1)
                        / np.sqrt(len(refine_seeds))
                    ),
                    "coarse_mean": float(np.mean(coarse[ss][j])),
                }
                for ss in ranked
            ),
            key=lambda r: r["mean"],
        )
        for r in rows[:5]:
            print(f"  (s,S)=({r['s']},{r['S']}): ${r['mean']:.2f} ± {r['ci']:.2f}/day")
        best[j] = {"best": rows[0], "ranking": rows}

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"ss_grid_search_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(
        json.dumps(
            {
                "config": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "coarse_seed_start": COARSE_SEED_START,
                "refine_seed_start": REFINE_SEED_START,
                "products": {f"P{j}": best[j] for j in range(2)},
                "coarse": {
                    f"{s},{S}": [float(np.mean(c0)), float(np.mean(c1))]
                    for (s, S), (c0, c1) in coarse.items()
                },
            },
            indent=2,
        )
    )
    print(
        f"\nBest: P0 {best[0]['best']['s'], best[0]['best']['S']}, P1 {best[1]['best']['s'], best[1]['best']['S']}"
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
