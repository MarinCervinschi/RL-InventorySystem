"""
Train one DQN per product on `ProductInventoryEnvironment`.

Usage (from the repository root):
    uv run python -m scripts.train_product_dqn --product 0 --timesteps 1000000
    uv run python -m scripts.train_product_dqn --product 1 --timesteps 1000000

Outputs (models/ is gitignored):
    <out>/final/dqn_model.zip + dqn_model_metadata.json   model at the end of training
    <out>/best/ ...                                       best model on validation seeds
    <out>/progress.csv                                    SB3 log (rollout + eval/*)
    <out>/eval_history.json                               validation curve
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3.common.logger import configure

from src.agents import DQNAgent
from src.agents.callbacks.joint_eval_callback import JointEvalCallback
from src.environment import (
    ProductControlSpec,
    ProductInventoryEnvironment,
    SSProductPolicy,
)
from src.seeds import TRAINING_SEED, generate_seeds

VALIDATION_SEED_START = 900  # validation seeds are disjoint from test seeds (0..)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--product", type=int, required=True, choices=[0, 1])
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--target-update", type=int, default=2_000)
    p.add_argument("--exploration-fraction", type=float, default=0.2)
    p.add_argument("--final-eps", type=float, default=0.02)
    p.add_argument("--net-arch", type=int, nargs="+", default=[128, 128])
    p.add_argument("--reward-scale", type=float, default=1 / 20)
    p.add_argument("--obs-scale", type=float, default=50.0)
    p.add_argument("--max-level", type=int, default=50)
    p.add_argument("--seed", type=int, default=TRAINING_SEED)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--eval-seeds", type=int, default=5)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)

    out = args.out or Path("models") / (
        f"dqn_p{args.product}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out.mkdir(parents=True, exist_ok=True)

    spec = ProductControlSpec.for_product(
        args.product, levels=tuple(range(args.max_level + 1)), obs_scale=args.obs_scale
    )
    partner = SSProductPolicy.baseline(1 - args.product)
    env = ProductInventoryEnvironment(
        product_id=args.product,
        partner=partner,
        spec=spec,
        reward_scale=args.reward_scale,
        random_seed=args.seed,
    )
    print(env)

    agent = DQNAgent(
        env=env,
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.final_eps,
        target_update_interval=args.target_update,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        policy_kwargs={"net_arch": args.net_arch},
        device="cpu",
        seed=args.seed,
        verbose=0,
    )
    agent.model.set_logger(configure(str(out), ["csv", "stdout"]))

    metadata = {
        "env": "ProductInventoryEnvironment",
        "product_id": args.product,
        "spec": spec.to_dict(),
        "reward_scale": args.reward_scale,
        "episode_length": env.episode_length,
        "partner": partner.name,
        "hyperparams": {**agent.hyperparams, "net_arch": args.net_arch},
        "total_timesteps": args.timesteps,
    }

    eval_callback = JointEvalCallback(
        spec=spec,
        partner=partner,
        seeds=generate_seeds(args.eval_seeds, start_index=VALIDATION_SEED_START),
        eval_freq=args.eval_freq,
        best_model_dir=out / "best",
        metadata=metadata,
    )

    start = time.time()
    agent.train(args.timesteps, callbacks=eval_callback, progress_bar=False)
    elapsed = time.time() - start

    metadata["training_time_s"] = elapsed
    metadata["timesteps"] = args.timesteps
    metadata["best_eval"] = (
        min(eval_callback.history, key=lambda r: r["product_cost"])
        if eval_callback.history
        else None
    )
    agent.save(metadata, out / "final")
    with open(out / "eval_history.json", "w") as f:
        json.dump(eval_callback.history, f, indent=2)
    print(f"Done in {elapsed / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
