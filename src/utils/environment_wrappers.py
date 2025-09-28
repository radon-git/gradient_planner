import gymnasium as gym

class SparseRewardWrapper(gym.Wrapper):
    """
    A wrapper for the environment that accumulates rewards during an episode
    and returns only the total reward at the end.
    """
    def __init__(self, env):
        super().__init__(env)
        self.cumulative_reward = 0.0

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.cumulative_reward += reward
        
        done = terminated or truncated
        if done:
            # Return the accumulated reward at the end of the episode
            final_reward = self.cumulative_reward
            self.cumulative_reward = 0.0
            return next_state, final_reward, terminated, truncated, info
        else:
            # Return 0 reward during the episode
            return next_state, 0.0, terminated, truncated, info

    def reset(self, **kwargs):
        self.cumulative_reward = 0.0
        return self.env.reset(**kwargs)