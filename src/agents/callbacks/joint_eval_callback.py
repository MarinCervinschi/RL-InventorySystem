import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.agents.metrics.joint_evaluation import DEFAULT_WARMUP, run_joint_episode
from src.environment.product_control import (
    ModelProductPolicy,
    ProductControlSpec,
    ProductPolicy,
)


class JointEvalCallback(BaseCallback):
    """
    Periodically evaluates the policy being trained inside the joint system.

    Every `eval_freq` steps the greedy policy runs next to `partner` on fixed
    validation seeds; the steady-state daily cost of the controlled product is
    logged (`eval/product_cost`, `eval/total_cost`) and the best model so far
    is saved to `best_model_dir` together with `metadata`.
    """

    def __init__(
        self,
        spec: ProductControlSpec,
        partner: ProductPolicy,
        seeds: Sequence[int],
        eval_freq: int = 50_000,
        episode_length: int = 1000,
        warmup: int = DEFAULT_WARMUP,
        best_model_dir: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.spec = spec
        self.partner = partner
        self.seeds = list(seeds)
        self.eval_freq = eval_freq
        self.episode_length = episode_length
        self.warmup = warmup
        self.best_model_dir = Path(best_model_dir) if best_model_dir else None
        self.metadata = metadata or {}
        self.best_cost = float("inf")
        self.history: List[Dict[str, float]] = []

    def _evaluate(self) -> Dict[str, float]:
        policy = ModelProductPolicy(model=self.model, spec=self.spec)
        pid = self.spec.product_id
        pair = (policy, self.partner) if pid == 0 else (self.partner, policy)
        own, total = [], []
        for seed in self.seeds:
            ep = run_joint_episode(pair, seed, self.episode_length)
            own.append(np.mean(ep[f"cost_{pid}"][self.warmup :]))
            total.append(np.mean(ep["total_daily_cost"][self.warmup :]))
        return {
            "timesteps": self.num_timesteps,
            "product_cost": float(np.mean(own)),
            "product_cost_std": float(np.std(own)),
            "total_cost": float(np.mean(total)),
        }

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        result = self._evaluate()
        self.history.append(result)
        self.logger.record("eval/product_cost", result["product_cost"])
        self.logger.record("eval/total_cost", result["total_cost"])

        improved = result["product_cost"] < self.best_cost
        if improved:
            self.best_cost = result["product_cost"]
            if self.best_model_dir is not None:
                self.best_model_dir.mkdir(parents=True, exist_ok=True)
                self.model.save(self.best_model_dir / "dqn_model")
                meta = {
                    **self.metadata,
                    "timesteps": self.num_timesteps,
                    "eval": result,
                }
                with open(self.best_model_dir / "dqn_model_metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)

        if self.verbose:
            print(
                f"[eval] step={self.num_timesteps:>8d}  "
                f"P{self.spec.product_id} cost=${result['product_cost']:.2f}/day  "
                f"total=${result['total_cost']:.2f}/day"
                + ("  *best*" if improved else ""),
                flush=True,
            )
        return True
