import torch
import torch.optim as optim
import numpy as np
from src.models.ppo_models import PPOActor, PPOCritic


class RolloutBuffer:
    def __init__(self, buffer_size, state_shape, action_shape, device):
        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self._p, self.buffer_size = 0, buffer_size

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(action if isinstance(action, torch.Tensor) else torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self._p = (self._p + 1)

    def is_full(self):
        return self._p == self.buffer_size

    def get(self):
        assert self.is_full(), 'Buffer needs to be full.'
        self._p = 0
        return self.states, self.actions, self.rewards, self.dones, self.log_pis, self.next_states


def calculate_advantage(values, rewards, dones, next_values, gamma=0.995, lambd=0.997):
    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = torch.empty_like(rewards)
    advantages[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        advantages[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * advantages[t + 1]
    targets = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return targets, advantages


class PPOAgent:
    def __init__(self, state_shape, action_shape, action_high, device, seed=0, lr=3e-4, gamma=0.99, clip_eps=0.2, num_updates=10, batch_size=64, rollout_length=2048):
        self.device = device
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.batch_size = batch_size
        self.num_updates = num_updates
        self.rollout_length = rollout_length

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.actor = PPOActor(state_shape, action_shape, action_high, device).to(device)
        self.critic = PPOCritic(state_shape).to(device)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer = RolloutBuffer(rollout_length, state_shape, action_shape, device)

    def explore(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state_tensor)
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor)
        return action.cpu().numpy()[0]

    def update(self):
        states, actions, rewards, dones, log_pis_old, next_states = self.buffer.get()
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
        targets, advantages = calculate_advantage(values, rewards, dones, next_values, self.gamma)

        for _ in range(self.num_updates):
            indices = np.random.permutation(self.rollout_length)
            for start in range(0, self.rollout_length, self.batch_size):
                idxes = indices[start:start + self.batch_size]
                self.update_critic(states[idxes], targets[idxes])
                self.update_actor(states[idxes], actions[idxes], log_pis_old[idxes], advantages[idxes])

    def update_critic(self, states, targets):
        loss_critic = (self.critic(states) - targets).pow(2).mean()
        self.optim_critic.zero_grad()
        loss_critic.backward()
        self.optim_critic.step()

    def update_actor(self, states, actions, log_pis_old, advantages):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        ratios = (log_pis - log_pis_old).exp()
        loss_actor1 = -ratios * advantages
        loss_actor2 = -torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()
        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

    def train(self, env, num_steps):
        state, _ = env.reset()
        # rollout_lengthステップごとに更新するため、ステップカウンターを追加
        t = 0
        for step in range(num_steps):
            # exploreを呼び出してactionとlog_piを取得
            action, log_pi = self.explore(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # バッファに追加
            # PPOではエピソードの途中でも固定長で更新するため、マスクされたdoneを渡す
            if t == self.rollout_length - 1:
                done_masked = False
            else:
                done_masked = done
            self.buffer.append(state, action, reward, done_masked, log_pi, next_state)
            
            t += 1
            if done:
                state, _ = env.reset()
                t = 0
            else:
                state = next_state

            # バッファが満杯になったら更新
            if self.buffer.is_full():
                self.update()
                # get()メソッド内でポインタがリセットされるため、clear()は不要

    def evaluate(self, env, num_episodes):
        total_reward = 0.0
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
        return total_reward / num_episodes