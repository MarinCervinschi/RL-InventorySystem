"""
Ablation: DQN on the ORIGINAL joint `InventoryEnvironment` with a scaled reward.

Same hyperparameters as `scripts/train_product_dqn.py`; the only differences
with the per-product agents are the problem structure (joint action
Discrete((Q_max+1)^2), 31 stacked raw frames) and the reward scale, which is
applied by a wrapper (the environment itself is unchanged).

Usage (from the repository root):
    uv run python -m scripts.train_joint_dqn --reward-scale 0.01 --timesteps 1000000
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from gymnasium.wrappers import TransformReward
from stable_baselines3.common.logger import configure

from src.agents import DQNAgent
from src.agents.callbacks.joint_eval_callback import JointEvalCallback
from src.environment import FrameStackJointPolicy, InventoryEnvironment
from src.seeds import TRAINING_SEED, generate_seeds

VALIDATION_SEED_START = 900


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint-env DQN ablation")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--reward-scale", type=float, default=0.01)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--q-max", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=TRAINING_SEED)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--eval-seeds", type=int, default=10)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    out = args.out or Path("models") / f"dqn_joint_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    scale = args.reward_scale
    env = TransformReward(
        InventoryEnvironment(k=args.k, Q_max=args.q_max, random_seed=args.seed),
        lambda r: r * scale,
    )

    agent = DQNAgent(
        env=env,  # type: ignore[arg-type]
        learning_rate=args.lr,
        gamma=0.99,
        buffer_size=200_000,
        batch_size=128,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        target_update_interval=2_000,
        learning_starts=5_000,
        train_freq=4,
        policy_kwargs={"net_arch": [128, 128]},
        device="cpu",
        seed=args.seed,
        verbose=0,
    )
    agent.model.set_logger(configure(str(out), ["csv"]))

    metadata = {
        "env": "InventoryEnvironment",
        "k": args.k,
        "Q_max": args.q_max,
        "reward_scale": scale,
        "hyperparams": agent.hyperparams,
        "total_timesteps": args.timesteps,
    }
    eval_callback = JointEvalCallback(
        make_policies=lambda model: FrameStackJointPolicy(
            model, k=args.k, Q_max=args.q_max
        ).as_product_policies(),
        cost_key="total_daily_cost",
        seeds=generate_seeds(args.eval_seeds, start_index=VALIDATION_SEED_START),
        eval_freq=args.eval_freq,
        best_model_dir=out / "best",
        metadata=metadata,
    )

    start = time.time()
    agent.train(args.timesteps, callbacks=eval_callback, progress_bar=False)
    metadata["training_time_s"] = time.time() - start
    agent.save(metadata, out / "final")
    with open(out / "eval_history.json", "w") as f:
        json.dump(eval_callback.history, f, indent=2)
    print(f"Done in {metadata['training_time_s'] / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
