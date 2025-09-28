import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SACActor(nn.Module):
    """
    SAC Actor (Policy) Network.
    Outputs a stochastic action and its log probability.
    """
    def __init__(self, state_dim, action_dim, action_high, device=None):
        super().__init__()
        self.action_high = torch.tensor(action_high, dtype=torch.float)
        if device:
            self.action_high = self.action_high.to(device)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(self, state, deterministic=False, with_logprob=True):
        net_out = self.net(state)
        mean = self.mean_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = Normal(mean, std)

        if deterministic:
            u = mean
        else:
            u = dist.rsample()

        action = torch.tanh(u) * self.action_high.to(u.device)

        if with_logprob:
            # Enforce Action Bounds
            log_prob = dist.log_prob(u) - torch.log(self.action_high.to(u.device) * (1 - torch.tanh(u).pow(2)) + 1e-6)
            log_prob = log_prob.sum(axis=-1, keepdim=True)
        else:
            log_prob = None

        return action, log_prob


class SACCritic(nn.Module):
    """
    SAC Critic (Q-Value) Network.
    Uses two Q-networks to mitigate overestimation bias.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1