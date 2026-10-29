"""
Log a DQN training run (the output folder of the training scripts) to MLflow.

A run folder contains:
    best/dqn_model.zip + dqn_model_metadata.json    best checkpoint on the validation seeds
    final/ ...                                      checkpoint at the end of training (optional)
    progress.csv                                    SB3 log (rollout/*, train/*, eval/*)
    eval_history.json                               validation curve (optional)

Each folder becomes one MLflow run: parameters from the metadata, the curves of
progress.csv / eval_history.json as step metrics, the best checkpoint as a pyfunc
model (optionally registered), the rest as artifacts.

The tracking server is read from MLFLOW_TRACKING_URI (no local fallback).
"""

import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
from mlflow.entities import Metric
from mlflow.models.model import ModelInfo

EXPERIMENT_NAME = "rl-inventory-system"
PYFUNC_MODEL_CODE = Path(__file__).with_name("dqn_pyfunc_model.py")

_MAX_BATCH = 1000  # MLflow limit on metrics per log_batch call


def require_tracking_uri() -> str:
    """Tracking URI from the environment; refuse to silently log to ./mlruns."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set (e.g. export MLFLOW_TRACKING_URI=http://100.89.8.106:5000)"
        )
    return uri


def git_commit(repo: Path = Path(".")) -> str:
    """HEAD commit, with a "-dirty" suffix if tracked files have uncommitted changes."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{commit}-dirty" if status else commit


def run_params(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Flat MLflow parameters from a checkpoint metadata dict."""
    hp = dict(metadata.get("hyperparams", {}))
    hp.pop("policy_kwargs", None)
    params: Dict[str, Any] = {
        "algo": "DQN",
        "env": metadata.get("env"),
        "total_timesteps": metadata.get("total_timesteps"),
        "reward_scale": metadata.get("reward_scale"),
        **hp,
    }
    for key in ("product_id", "partner", "episode_length", "eval_seeds", "k", "Q_max"):
        if metadata.get(key) is not None:
            params[key] = metadata[key]
    if metadata.get("env") == "ProductInventoryEnvironment":
        # older runs did not store the flag: the purchase cost was always in the reward
        params["purchase_cost_in_reward"] = metadata.get("purchase_cost_in_reward", True)
    spec = metadata.get("spec")
    if spec:
        residual = spec.get("residual_base") is not None
        params["action_mode"] = "residual" if residual else "order_up_to"
        params["history_length"] = spec["history_length"]
        params["obs_scale"] = spec["obs_scale"]
        if residual:
            params["residual_base"] = spec["residual_base"]
            params["max_delta"] = spec["max_delta"]
        else:
            params["max_level"] = max(spec["levels"])
    return {k: str(v) for k, v in params.items() if v is not None}


def _read_progress(path: Path) -> List[Dict[str, float]]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items() if v not in ("", None)})
    return rows


def _log_metrics(client: mlflow.MlflowClient, run_id: str, metrics: List[Metric]) -> None:
    for i in range(0, len(metrics), _MAX_BATCH):
        client.log_batch(run_id, metrics=metrics[i : i + _MAX_BATCH])


def log_dqn_model(
    checkpoint_dir: Path, registered_model_name: Optional[str] = None
) -> ModelInfo:
    """Log a DQN checkpoint folder as a pyfunc model in the active run."""
    from stable_baselines3 import DQN

    obs_shape = DQN.load(checkpoint_dir / "dqn_model.zip", device="cpu").observation_space.shape
    return mlflow.pyfunc.log_model(
        name="model",
        python_model=str(PYFUNC_MODEL_CODE),
        artifacts={"checkpoint": str(checkpoint_dir)},
        input_example=np.zeros((1, *obs_shape), dtype=np.float32),
        registered_model_name=registered_model_name,
    )


def log_training_run(
    run_dir: Path,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    registered_model_name: Optional[str] = None,
    commit: Optional[str] = None,
    log_file: Optional[Path] = None,
    experiment_name: str = EXPERIMENT_NAME,
) -> mlflow.entities.Run:
    """
    Log one training output folder as an MLflow run.

    Args:
        run_dir: training output folder (see module docstring)
        run_name: defaults to the folder name
        tags: extra run tags (phase, product, ...)
        registered_model_name: if given, the best checkpoint is registered under this name
        commit: code commit for `mlflow.source.git.commit` (default: current HEAD, "-dirty" if modified)
        log_file: training stdout to attach (optional)
        experiment_name: MLflow experiment

    Returns:
        The finished MLflow run.
    """
    require_tracking_uri()
    run_dir = Path(run_dir)
    best = run_dir / "best"
    final = run_dir / "final"
    progress_path = run_dir / "progress.csv"
    history_path = run_dir / "eval_history.json"

    metadata = json.loads((best / "dqn_model_metadata.json").read_text())
    final_meta = (
        json.loads((final / "dqn_model_metadata.json").read_text())
        if (final / "dqn_model_metadata.json").exists()
        else {}
    )
    progress = _read_progress(progress_path) if progress_path.exists() else []
    history = json.loads(history_path.read_text()) if history_path.exists() else []

    # Wall-clock times of the original training: end = last write of progress.csv
    end = progress_path.stat().st_mtime if progress_path.exists() else time.time()
    elapsed = final_meta.get("training_time_s") or (
        progress[-1].get("time/time_elapsed", 0.0) if progress else 0.0
    )
    start = end - elapsed

    experiment = mlflow.set_experiment(experiment_name)
    client = mlflow.MlflowClient()
    all_tags = {
        "mlflow.source.git.commit": commit or git_commit(),
        "mlflow.runName": run_name or run_dir.name,
        "run_dir": run_dir.name,
        **(tags or {}),
    }
    run = client.create_run(
        experiment.experiment_id, start_time=int(start * 1000), tags=all_tags
    )
    run_id = run.info.run_id

    client.log_batch(
        run_id,
        params=[
            mlflow.entities.Param(k, v)
            for k, v in run_params({**metadata, **final_meta}).items()
        ],
    )

    metrics: List[Metric] = []
    for row in progress:
        step = int(row.get("time/total_timesteps", 0))
        ts = int((start + row.get("time/time_elapsed", 0.0)) * 1000)
        for key, value in row.items():
            if key.startswith("time/") or not math.isfinite(value):
                continue
            metrics.append(Metric(key, value, ts, step))
    for rec in history:
        ts = int(end * 1000)
        step = int(rec["timesteps"])
        metrics.append(Metric("validation/cost", rec["product_cost"], ts, step))
        metrics.append(Metric("validation/cost_std", rec["product_cost_std"], ts, step))
        metrics.append(Metric("validation/total_cost", rec["total_cost"], ts, step))
    best_eval = metadata.get("eval") or {}
    summary = {
        "best/cost": best_eval.get("product_cost"),
        "best/cost_std": best_eval.get("product_cost_std"),
        "best/total_cost": best_eval.get("total_cost"),
        "best/timesteps": metadata.get("timesteps"),
        "training_time_s": final_meta.get("training_time_s"),
    }
    metrics += [
        Metric(k, float(v), int(end * 1000), 0) for k, v in summary.items() if v is not None
    ]
    _log_metrics(client, run_id, metrics)

    with mlflow.start_run(run_id=run_id):
        for path in (progress_path, history_path):
            if path.exists():
                mlflow.log_artifact(str(path), "training")
        if log_file is not None and log_file.exists():
            mlflow.log_artifact(str(log_file), "training")
        if final.exists():
            mlflow.log_artifacts(str(final), "checkpoints/final")
        info = log_dqn_model(best, registered_model_name)

    client.set_terminated(run_id, "FINISHED", end_time=int(end * 1000))
    if registered_model_name and info.registered_model_version is not None:
        version = str(info.registered_model_version)
        for key in ("run_dir", "phase", "experiment_row"):
            if key in all_tags:
                client.set_model_version_tag(registered_model_name, version, key, all_tags[key])
    return client.get_run(run_id)
