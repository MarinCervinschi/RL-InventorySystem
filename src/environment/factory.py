from src.environment.gym_env import InventoryEnvironment


class InventoryEnvironmentFactory:
    """Factory for creating environment configurations."""

    @staticmethod
    def create_default() -> InventoryEnvironment:
        """Create environment with default parameters."""
        # TODO adjust Q_max for default?
        return InventoryEnvironment(k=3, Q_max=200, episode_length=100, gamma=0.99)

    @staticmethod
    def create_short_horizon(gamma: float = 0.95) -> InventoryEnvironment:
        """Create environment with short horizon (γ=0.95)."""
        return InventoryEnvironment(k=3, Q_max=20, episode_length=100, gamma=gamma)

    @staticmethod
    def create_long_horizon(gamma: float = 0.999) -> InventoryEnvironment:
        """Create environment with long horizon (γ=0.999)."""
        return InventoryEnvironment(k=3, Q_max=20, episode_length=100, gamma=gamma)

    @staticmethod
    def create_small_action_space(Q_max: int = 10) -> InventoryEnvironment:
        """Create environment with smaller action space."""
        return InventoryEnvironment(k=3, Q_max=Q_max, episode_length=100, gamma=0.99)

    @staticmethod
    def create_for_training(
        k: int = 3, Q_max: int = 20, gamma: float = 0.99, episode_length: int = 100
    ) -> InventoryEnvironment:
        """Create environment with custom hyperparameters."""
        return InventoryEnvironment(
            k=k, Q_max=Q_max, episode_length=episode_length, gamma=gamma
        )
