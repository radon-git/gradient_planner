import torch
import torch.nn as nn


class DynamicsModel(nn.Module):
    """ s_t, a_t -> s_{t+1} を予測するモデル """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.gru = nn.GRU(state_dim + action_dim, hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        gru_out, _ = self.gru(x)
        delta_s_pred = self.mlp(gru_out)
        return states + delta_s_pred


class TerminalRewardModel(nn.Module):
    """ (s, a)の系列 -> 最終報酬R_T を予測するモデル """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.gru = nn.GRU(state_dim + action_dim, hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        return self.mlp(last_output)