from stable_baselines3.common.callbacks import BaseCallback


class LearningCurveCallback(BaseCallback):
    """Callback to record episode rewards during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_timesteps = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        # Accumulate reward
        self.current_episode_reward += self.locals['rewards'][0]
        
        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_timesteps.append(self.num_timesteps)
            self.current_episode_reward = 0
            
        return True
