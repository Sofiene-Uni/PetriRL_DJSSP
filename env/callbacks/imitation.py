from stable_baselines3.common.callbacks import BaseCallback

class TopKBufferCallback(BaseCallback):
    def __init__(self, k=10, verbose=0):
        super().__init__(verbose)
        self.k = k
        self.top_k_buffer = []  # List of (reward, [(obs, action, mask), ...])
        self.episode_data = []
        self.episode_reward = 0

    def _on_step(self) -> bool:
        obs = self.locals["new_obs"]
        actions = self.locals["actions"]
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        mask = self.training_env.get_attr("action_masks")[0]() 
        

        for i in range(len(dones)):
            self.episode_data.append((obs[i], actions[i], mask))
            self.episode_reward += rewards[i]

            if dones[i]:
                self._maybe_store_episode()
                self.episode_data = []
                self.episode_reward = 0

        return True

    def _maybe_store_episode(self):
        if len(self.top_k_buffer) < self.k:
            self.top_k_buffer.append((self.episode_reward, list(self.episode_data)))
        else:
            min_r = min(self.top_k_buffer, key=lambda x: x[0])[0]
            if self.episode_reward > min_r:
                idx = self.top_k_buffer.index(min(self.top_k_buffer, key=lambda x: x[0]))
                self.top_k_buffer[idx] = (self.episode_reward, list(self.episode_data))

    def get_top_k(self):
        return self.top_k_buffer

