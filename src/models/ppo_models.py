import torch
import torch.nn as nn
import math

# --- 方策計算のためのヘルパー関数 ---
def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

def calculate_log_pi(log_stds, noises, actions_tanh):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)
    return gaussian_log_probs - torch.log(1 - actions_tanh.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

def reparameterize(means, log_stds):
    stds = log_stds.exp()
    noises = torch.randn_like(means)
    us = means + noises * stds
    actions_tanh = torch.tanh(us)
    log_pis_tanh = calculate_log_pi(log_stds, noises, actions_tanh)
    return actions_tanh, log_pis_tanh

def evaluate_lop_pi(means, log_stds, actions_tanh):
    noises = (atanh(actions_tanh) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions_tanh)


class PPOActor(nn.Module):
    def __init__(self, state_shape, action_shape, action_high, device):
        super().__init__()
        self.action_high = torch.tensor(action_high, dtype=torch.float, device=device)
        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_shape[0]),
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states)) * self.action_high

    def sample(self, states):
        means = self.net(states)
        actions_tanh, log_pis_tanh = reparameterize(means, self.log_stds)
        actions = actions_tanh * self.action_high
        log_pis = log_pis_tanh - torch.log(self.action_high).sum()
        return actions, log_pis

    def evaluate_log_pi(self, states, actions):
        actions_tanh = actions / self.action_high
        actions_tanh = torch.clamp(actions_tanh, -1 + 1e-6, 1 - 1e-6)
        log_pis_tanh = evaluate_lop_pi(self.net(states), self.log_stds, actions_tanh)
        log_pis = log_pis_tanh - torch.log(self.action_high).sum()
        return log_pis

class PPOCritic(nn.Module):
    def __init__(self, state_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, states):
        return self.net(states)