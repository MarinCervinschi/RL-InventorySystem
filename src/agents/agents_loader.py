from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from stable_baselines3 import DQN, PPO

from src.agents import DQNAgent, PPOAgent
from src.environment import InventoryEnvironment


class AgentsLoader:
    """
    Load and manage trained RL agents.

    Supports multiple agent types and provides a unified interface
    for loading pre-trained models.
    """

    def __init__(self, env: InventoryEnvironment, models_dir: Optional[Path] = None):
        """
        Initialize agent loader.

        Args:
            env: Environment instance for agent compatibility
            models_dir: Directory containing saved models (default: ./models)
        """
        self.env = env
        self.models_dir = Path(models_dir) if models_dir else Path("./models")

        # Registry of available agent types
        self.agent_types = {
            "dqn": self._load_dqn,
            "ppo": self._load_ppo,
            # Add more agent types here as you implement them:
            # 'a2c': self._load_a2c,
        }

        # Cache loaded agents
        self._loaded_agents: Dict[str, DQNAgent] = {}

    def load_agent(self, model_id: str, agent_type: str = "dqn") -> DQNAgent:
        """
        Load a trained agent.

        Args:
            model_id: Model identifier (e.g., 'dqn_baseline', 'dqn_20260108_104709')
            agent_type: Type of agent ('dqn', 'a2c', 'ppo', etc.)

        Returns:
            Loaded agent instance

        Raises:
            ValueError: If agent_type not supported
            FileNotFoundError: If model file doesn't exist
        """
        # Check cache
        cache_key = f"{agent_type}_{model_id}"
        if cache_key in self._loaded_agents:
            print(f"✅ Using cached agent: {cache_key}")
            return self._loaded_agents[cache_key]

        # Load agent
        if agent_type not in self.agent_types:
            raise ValueError(
                f"Unknown agent type '{agent_type}'. "
                f"Available: {list(self.agent_types.keys())}"
            )

        print(f"📥 Loading {agent_type.upper()} agent: {model_id}...")
        agent = self.agent_types[agent_type](model_id)

        # Cache and return
        self._loaded_agents[cache_key] = agent
        print(f"✅ Agent loaded successfully")
        return agent

    def _load_dqn(self, model_id: str) -> DQNAgent:
        """Load DQN agent from disk."""
        model_path = self.models_dir / f"{model_id}"

        # Check if file exists (with or without .zip extension)
        if not model_path.exists() and not Path(str(model_path) + ".zip").exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Available models in {self.models_dir}:\n"
                + "\n".join(f"  - {f.stem}" for f in self.models_dir.glob("*.zip"))
            )

        # Create agent wrapper with environment
        agent = DQNAgent(
            env=self.env,
            verbose=0,
        )

        # Load trained weights
        agent.model = DQN.load(str(model_path), env=self.env)

        return agent

    def _load_ppo(self, model_id: str) -> PPOAgent:
        """Load PPO agent from disk."""
        model_path = self.models_dir / f"{model_id}"

        # Check if file exists (with or without .zip extension)
        if not model_path.exists() and not Path(str(model_path) + ".zip").exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Available models in {self.models_dir}:\n"
                + "\n".join(f"  - {f.stem}" for f in self.models_dir.glob("*.zip"))
            )

        # Create agent wrapper with environment
        agent = PPOAgent(
            env=self.env,
            verbose=0,
        )

        # Load trained weights
        agent.model = PPO.load(str(model_path), env=self.env)

        return agent

    def list_available_models(self, agent_type: Optional[str] = None) -> list[str]:
        """
        List available trained models.

        Args:
            agent_type: Filter by agent type (default: all types)

        Returns:
            List of available model IDs
        """
        if not self.models_dir.exists():
            print(f"⚠️  Models directory not found: {self.models_dir}")
            return []

        models = []

        # Search for .zip files (SB3 format)
        for model_file in self.models_dir.glob("*.zip"):
            model_id = model_file.stem

            # Filter by agent type if specified
            if agent_type is None or model_id.startswith(agent_type):
                models.append(model_id)

        return sorted(models)

    def __repr__(self) -> str:
        return (
            f"AgentsLoader(models_dir={self.models_dir}, "
            f"supported_types={list(self.agent_types.keys())})"
        )
