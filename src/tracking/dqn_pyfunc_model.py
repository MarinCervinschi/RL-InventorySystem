"""
MLflow pyfunc wrapper of a Stable-Baselines3 DQN checkpoint (models-from-code).

Logged by `src.tracking.mlflow_logging.log_dqn_model`; loading the model only needs
stable-baselines3, not this repository. The checkpoint folder (dqn_model.zip +
dqn_model_metadata.json) is stored in the model under artifacts/best/, so the
original policy can also be rebuilt from a download:

    model = mlflow.pyfunc.load_model("models:/rl-inventory-dqn-p0@champion")
    actions = model.predict(observations)  # (n, obs_dim) float32 -> (n,) int

    path = mlflow.artifacts.download_artifacts("models:/rl-inventory-dqn-p0@champion")
    policy = ModelProductPolicy.load(Path(path) / "artifacts" / "best")
"""

from pathlib import Path

import mlflow
import numpy as np


class DQNPolicyModel(mlflow.pyfunc.PythonModel):
    """Greedy action of a DQN for a batch of (already normalized) observations."""

    def load_context(self, context) -> None:
        from stable_baselines3 import DQN

        checkpoint = Path(context.artifacts["checkpoint"])
        self.model = DQN.load(checkpoint / "dqn_model.zip", device="cpu")

    def predict(self, context, model_input, params=None) -> np.ndarray:
        obs = np.asarray(model_input, dtype=np.float32)
        actions, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(actions, dtype=np.int64)


mlflow.models.set_model(DQNPolicyModel())
