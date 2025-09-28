import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, state_shape, action_shape, device):
        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        
        self._p = 0
        self.size = 0 # 現在のバッファサイズを追跡する属性を追加
        self.buffer_size = buffer_size

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        # log_piがNoneの場合も考慮し、0.0にフォールバック
        self.log_pis[self._p] = float(log_pi if log_pi is not None else 0.0)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        
        self._p = (self._p + 1) % self.buffer_size
        # sizeを更新（最大値はbuffer_size）
        self.size = min(self.size + 1, self.buffer_size)

    def __len__(self):
        """len(buffer)で現在のサイズを返せるようにする"""
        return self.size

    def sample(self, batch_size):
        """バッファからランダムにバッチサイズ分のデータをサンプリングする"""
        assert batch_size <= self.size, "Batch size is larger than the current buffer size."
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def get(self):
        assert self._p == 0, 'Buffer needs to be full.'
        return self.states, self.actions, self.rewards, self.dones, self.log_pis, self.next_states

class TrajectoryBuffer:
    def __init__(self):
        self.trajectories = []

    def add_trajectory(self, states, actions, rewards, next_states):
        self.trajectories.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states
        })

    def get_trajectories(self):
        return self.trajectories

    def clear(self):
        self.trajectories = []