"""
Backfill MLflow with the experiments of docs/experiments/README.md.

Logs to the `rl-inventory-system` experiment:
    - one run per training folder in models/ (params, curves, checkpoints); the best
      checkpoint of each per-product run is registered as rl-inventory-dqn-p{0,1}
    - one run for the (s,S) grid search (data/05, data/06)
    - one run for the final Welch evaluation (notebook v3, data/09)
and gives the `champion` alias to the DQN checkpoints selected in notebook v3
(lowest cost on validation seeds 900-909, then evaluated on test seeds 0-299).

Each training run is tagged with the first commit containing the code that
reproduces it (they were trained before being committed).

Usage (from the repository root):
    export MLFLOW_TRACKING_URI=http://100.89.8.106:5000
    uv run python -m scripts.log_runs_to_mlflow            # everything
    uv run python -m scripts.log_runs_to_mlflow --dry-run  # only print the plan
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import mlflow

from src.tracking import EXPERIMENT_NAME, git_commit, log_training_run, require_tracking_uri

MODELS = Path("models")
DATA = Path("docs/experiments/data")

# First commits containing the code of each phase
COMMIT_PHASE_A = "f2c6dc5"  # per-product DQN, order-up-to actions, joint evaluation
COMMIT_PHASE_C = "1e2ebfc"  # re-tuned (s,S), --no-purchase-cost, joint-env ablation
COMMIT_PHASE_D = "493d4da"  # residual policy

# Checkpoints selected on the validation seeds in notebook v3 (section 2)
CHAMPIONS = {0: "dqn_p0_residual_1M", 1: "dqn_p1_residual_1M"}


@dataclass
class TrainingRun:
    """A training folder and how it maps to the experiment log."""

    run_dir: Path
    phase: str
    experiment_row: str
    description: str
    commit: str
    product: Optional[int]

    @property
    def registered_model_name(self) -> Optional[str]:
        return None if self.product is None else f"rl-inventory-dqn-p{self.product}"

    @property
    def log_file(self) -> Path:
        return MODELS / f"{self.run_dir.name.replace('dqn_', 'train_', 1)}.log"


def classify(run_dir: Path) -> Optional[TrainingRun]:
    """Phase / experiment row of a training folder (None if unknown)."""
    name = run_dir.name
    m = re.match(r"dqn_p(\d)_", name)
    product = int(m.group(1)) if m else None
    rules = [
        (r"dqn_joint_", "ablation", "8", "original joint env, same hyperparameters", COMMIT_PHASE_C),
        (r"_residual", "D", "9", "residual (δs, δS) on the re-tuned (s,S)", COMMIT_PHASE_D),
        (r"_rs100_noiq", "C", "6", "reward x1/100 without i·q", COMMIT_PHASE_C),
        (r"_rs100_s1234", "A", "7", "reward x1/100, second training seed", COMMIT_PHASE_A),
        (r"_rs100_1M", "A", "3", "reward x1/100", COMMIT_PHASE_A),
        (r"_lr3e-5", "A", "2", "reward x1/20, lr 3e-5", COMMIT_PHASE_A),
        (r"_ab_", "A", "1", "reward x1/20", COMMIT_PHASE_A),
    ]
    for pattern, phase, row, desc, commit in rules:
        if re.search(pattern, name):
            return TrainingRun(run_dir, phase, row, desc, commit, product)
    return None


def full_commit(short: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", short], capture_output=True, text=True, check=True
    ).stdout.strip()


def log_training_runs(runs: List[TrainingRun]) -> Dict[str, mlflow.entities.Run]:
    logged = {}
    for r in runs:
        tags = {
            "phase": r.phase,
            "experiment_row": r.experiment_row,
            "mlflow.note.content": r.description,
            "backfilled": "true",
        }
        if r.product is not None:
            tags["product"] = f"P{r.product}"
        run = log_training_run(
            r.run_dir,
            tags=tags,
            registered_model_name=r.registered_model_name,
            commit=full_commit(r.commit),
            log_file=r.log_file,
        )
        logged[r.run_dir.name] = run
        print(f"  {r.run_dir.name:<28} -> run {run.info.run_id}", flush=True)
    return logged


def log_grid_search() -> str:
    with mlflow.start_run(run_name="ss-grid-search") as run:
        mlflow.set_tags(
            {
                "mlflow.source.git.commit": full_commit(COMMIT_PHASE_C),
                "phase": "ss-tuning",
                "experiment_row": "5",
                "run_type": "evaluation",
                "backfilled": "true",
            }
        )
        data = json.loads((DATA / "05_ss_grid_search.json").read_text())
        mlflow.log_params({"algo": "(s,S) grid search", **data["config"]})
        mlflow.log_params(
            {"coarse_seed_start": data["coarse_seed_start"], "refine_seed_start": data["refine_seed_start"]}
        )
        for product, res in data["products"].items():
            best = res["best"]
            mlflow.log_params({f"{product}_s": best["s"], f"{product}_S": best["S"]})
            mlflow.log_metrics({f"{product}/cost": best["mean"], f"{product}/cost_ci": best["ci"]})
        for path in sorted(DATA.glob("0[56]_ss_grid_search*")):
            mlflow.log_artifact(str(path), "results")
        return run.info.run_id


def log_welch_v3() -> str:
    data = json.loads((DATA / "09_welch_v3_results.json").read_text())
    cfg = data["configuration"]
    with mlflow.start_run(run_name="welch-v3-test300") as run:
        mlflow.set_tags(
            {
                "mlflow.source.git.commit": full_commit(COMMIT_PHASE_D),
                "phase": "evaluation",
                "experiment_row": "9",
                "run_type": "evaluation",
                "reference": cfg["reference"],
                "backfilled": "true",
            }
        )
        mlflow.log_params(
            {
                "algo": "Welch + CRN joint evaluation",
                "sim_days": cfg["sim_days"],
                "num_replications": cfg["num_replications"],
                "window_size": cfg["window_size"],
                "warmup_days": cfg["warmup_days"],
                "seed_indices": "0-299",
                "validation_seed_indices": "900-909",
                "p0_checkpoint": CHAMPIONS[0],
                "p1_checkpoint": CHAMPIONS[1],
            }
        )
        slug = {
            "(s,S) baseline": "ss_baseline",
            "(s,S) tuned": "ss_tuned",
            "DQN (i·q in reward)": "dqn_iq",
            "DQN (no i·q)": "dqn_noiq",
            "DQN residual (s,S)": "dqn_residual",
        }
        for policy, res in data["results"].items():
            key = slug.get(policy, re.sub(r"\W+", "_", policy).strip("_").lower())
            mlflow.log_metrics(
                {f"{key}/{k}": float(v) for k, v in res.items() if isinstance(v, (int, float))}
            )
        mlflow.log_artifact(str(DATA / "09_welch_v3_results.json"), "results")
        mlflow.log_artifact("notebooks/welch_procedure_product_dqn_v3.ipynb", "notebook")
        return run.info.run_id


def set_champions(logged: Dict[str, mlflow.entities.Run], eval_run_id: str) -> None:
    client = mlflow.MlflowClient()
    for product, run_dir in CHAMPIONS.items():
        name = f"rl-inventory-dqn-p{product}"
        run_id = logged[run_dir].info.run_id
        (version,) = client.search_model_versions(f"name='{name}' and run_id='{run_id}'")
        client.set_registered_model_alias(name, "champion", version.version)
        client.set_model_version_tag(name, version.version, "evaluation_run", eval_run_id)
        client.update_registered_model(
            name,
            description=(
                f"Per-product DQN for product {product} (ProductInventoryEnvironment). "
                "Input: normalized observation; output: discrete action, see the spec in "
                "the checkpoint metadata. @champion = checkpoint selected on validation "
                "seeds 900-909 and evaluated on test seeds 0-299 (notebook v3)."
            ),
        )
        print(f"  {name} v{version.version} ({run_dir}) -> @champion")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="*", default=None, help="training folder names to log")
    p.add_argument("--skip-evaluations", action="store_true")
    args = p.parse_args()

    runs = [r for d in sorted(MODELS.glob("dqn_*")) if (r := classify(d)) is not None]
    if args.only:
        runs = [r for r in runs if r.run_dir.name in args.only]
    for r in runs:
        print(
            f"{r.run_dir.name:<28} phase {r.phase:<8} row {r.experiment_row} "
            f"commit {r.commit} register {r.registered_model_name} log {r.log_file.exists()}"
        )
    if args.dry_run:
        return

    print(f"tracking {require_tracking_uri()} / {EXPERIMENT_NAME} (code {git_commit()})")
    mlflow.set_experiment(EXPERIMENT_NAME)
    logged = log_training_runs(runs)
    if args.skip_evaluations:
        return
    print(f"  ss-grid-search -> run {log_grid_search()}")
    eval_run_id = log_welch_v3()
    print(f"  welch-v3-test300 -> run {eval_run_id}")
    if all(d in logged for d in CHAMPIONS.values()):
        set_champions(logged, eval_run_id)


if __name__ == "__main__":
    main()
