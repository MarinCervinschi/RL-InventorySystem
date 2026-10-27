"""
Joint evaluation of per-product policies against the (s,S) baseline.

Usage (from the repository root):
    uv run python -m scripts.evaluate_joint \
        --p0 models/dqn_p0_xxx/best --p1 models/dqn_p1_xxx/best --n-seeds 20

All configurations run on the same seeds (common random numbers):
    (s,S) | DQN-P0 + (s,S)-P1 | (s,S)-P0 + DQN-P1 | DQN + DQN
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.agents.metrics import EvaluationMetrics  # noqa: E402
from src.agents.metrics.joint_evaluation import (  # noqa: E402
    DEFAULT_WARMUP,
    JointPolicy,
    compare_joint_policies,
    format_comparison,
)
from src.environment import ModelProductPolicy, SSProductPolicy  # noqa: E402
from src.seeds import generate_seeds  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint evaluation vs (s,S)")
    p.add_argument("--p0", type=Path, default=None, help="DQN model dir for product 0")
    p.add_argument("--p1", type=Path, default=None, help="DQN model dir for product 1")
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--out", type=Path, default=Path("results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ss = (SSProductPolicy.baseline(0), SSProductPolicy.baseline(1))
    dqn = {
        pid: ModelProductPolicy.load(path, label=f"DQN-P{pid}")
        for pid, path in ((0, args.p0), (1, args.p1))
        if path is not None
    }

    configs: Dict[str, JointPolicy] = {"(s,S)": ss}
    if 0 in dqn:
        configs["DQN-P0 + (s,S)-P1"] = (dqn[0], ss[1])
    if 1 in dqn:
        configs["(s,S)-P0 + DQN-P1"] = (ss[0], dqn[1])
    if 0 in dqn and 1 in dqn:
        configs["DQN + DQN"] = (dqn[0], dqn[1])

    seeds = generate_seeds(args.n_seeds, start_index=args.seed_start)
    table, episodes = compare_joint_policies(
        configs,
        seeds,
        episode_length=args.episode_length,
        warmup=args.warmup,
        reference="(s,S)",
    )

    # Welch's procedure (existing implementation) as a sanity check on the warmup
    welch = {}
    for name, eps in episodes.items():
        _, _, welch[name] = EvaluationMetrics().plot_welch_procedure(eps, show=False)
        plt.close("all")
    table["welch_warmup"] = table.index.map(welch)

    print()
    print(
        f"Seeds: {len(seeds)} (start_index={args.seed_start}), "
        f"episode={args.episode_length} days, fixed warmup={args.warmup} days"
    )
    print(format_comparison(table))
    print("\nWelch-detected warmup (days):", welch)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"joint_eval_{datetime.now():%Y%m%d_%H%M%S}"
    table.to_csv(f"{stem}.csv")
    with open(f"{stem}.md", "w") as f:
        f.write(format_comparison(table) + "\n")
    print(f"\nSaved {stem}.csv / .md")


if __name__ == "__main__":
    main()
