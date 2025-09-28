import torch
import torch.optim as optim

class SparseRewardPlannerAgent:
    def __init__(self, dynamics_model, reward_model, action_dim, action_high):
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.action_dim = action_dim
        # モデルのパラメータからdevice情報を取得する
        model_device = next(self.dynamics_model.parameters()).device
        self.action_high = torch.tensor(action_high, dtype=torch.float, device=model_device)
        self.device = model_device
        self.dynamics_model.eval()
        self.reward_model.eval()

    def plan(self, initial_state, horizon=15, iterations=100, lr=0.1):
        # モデルをtrainモードに設定して勾配を計算可能にする
        self.dynamics_model.train()
        self.reward_model.train()
        
        # 最適化対象の行動系列を初期化
        raw_actions = torch.randn(horizon, self.action_dim, requires_grad=True, device=self.device)
        optimizer = optim.Adam([raw_actions], lr=lr)
        initial_state_tensor = torch.tensor(initial_state, dtype=torch.float, device=self.device)

        for _ in range(iterations):
            optimizer.zero_grad()
            
            # [-1, 1]の範囲に正規化し、スケーリング
            actions = torch.tanh(raw_actions) * self.action_high
            
            # 1. DynamicsModelで未来の状態系列を予測
            states_list = []
            current_s = initial_state_tensor
            for t in range(horizon):
                states_list.append(current_s)
                # モデルの入力形式 (batch, seq_len, dim) に合わせる
                s_in = current_s.unsqueeze(0).unsqueeze(0)
                a_in = actions[t].unsqueeze(0).unsqueeze(0)
                
                # DynamicsModelは1ステップずつ予測
                next_s_pred = self.dynamics_model(s_in, a_in)
                current_s = next_s_pred.squeeze(0).squeeze(0)
            
            # 2. 予測した状態・行動系列をTerminalRewardModelに入力
            predicted_states = torch.stack(states_list).unsqueeze(0) # (1, H, state_dim)
            predicted_actions = actions.unsqueeze(0) # (1, H, action_dim)
            predicted_terminal_reward = self.reward_model(predicted_states, predicted_actions)
            
            # 3. 最終報酬を最大化（損失は負の報酬）
            loss = -predicted_terminal_reward
            loss.backward()
            optimizer.step()
        
        # 評価モードに戻す
        self.dynamics_model.eval()
        self.reward_model.eval()
        
        # 最適化された行動系列の最初の行動を返す
        final_actions = torch.tanh(raw_actions.detach()) * self.action_high
        return final_actions[0].cpu().numpy()