import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from src.models.sac_models import SACActor, SACCritic
from src.utils.replay_buffers import ReplayBuffer

class SACAgent:
    def __init__(self, state_shape, action_shape, action_high, device, seed=0, lr=3e-4, gamma=0.99, tau=0.005, buffer_size=1000000, batch_size=256, alpha=0.2):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha

        np.random.seed(seed)
        torch.manual_seed(seed)

        state_dim = state_shape[0]
        action_dim = action_shape[0]

        self.actor = SACActor(state_dim, action_dim, action_high).to(device)
        self.critic = SACCritic(state_dim, action_dim).to(device)
        self.critic_target = SACCritic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size, state_shape, action_shape, device)

    def explore(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(state_tensor, deterministic=False)
        return action.cpu().numpy()[0]

    def exploit(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(state_tensor, deterministic=True)
        return action.cpu().numpy()[0]

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Criticの更新
        with torch.no_grad():
            next_actions, next_log_pi = self.actor(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actorの更新
        new_actions, log_pi = self.actor(states)
        q1_new, q2_new = self.critic(states, new_actions)
        actor_loss = (self.alpha * log_pi - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ターゲットネットワークのソフトアップデート
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)