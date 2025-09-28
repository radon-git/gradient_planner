import torch
import numpy as np
import time

class GradientPlannerAgent:
    """
    学習済みの世界モデルと報酬モデルを使い、勾配法によって行動計画を最適化するエージェント。
    エピソード全体の行動系列を一度に最適化するモードを追加。
    """
    def __init__(self, dynamics_model, reward_model, action_dim, action_high,
                 plan_horizon=15, num_iterations=100, learning_rate=1e-2,
                 use_full_sequence=False):
        """
        Args:
            dynamics_model (nn.Module): 状態遷移を予測するモデル s_t, a_t -> s_{t+1}
            reward_model (nn.Module): 行動系列の最終報酬を予測するモデル
            action_dim (int): 行動の次元数
            action_high (np.ndarray): 行動の最大値
            plan_horizon (int): 計画する未来のステップ数
            num_iterations (int): 行動計画を最適化する際の反復回数
            learning_rate (float): 行動計画の最適化に使う学習率
            use_full_sequence (bool): 行動系列全体を保存して使用するモード。Trueなら一度の最適化で
                                     エピソード終了まで使う行動系列を計算し、再計画しない。
        """
        model_device = next(dynamics_model.parameters()).device
        self.device = model_device
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.action_dim = action_dim
        self.action_high = torch.from_numpy(action_high).float().to(model_device)
        self.plan_horizon = plan_horizon
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.use_full_sequence = use_full_sequence
        
        # 行動系列を保存するための変数
        self.current_action_sequence = None
        self.current_action_index = 0
        

    def plan(self, state, reset=False):
        """
        現在の状態から、未来の報酬を最大化する行動計画を勾配法で探索し、
        use_full_sequence=Falseの場合は最初の行動を返す。
        use_full_sequence=Trueの場合は、行動系列全体を保存し、
        呼び出しごとに順番に返す（エピソード全体で一度だけ計画）。
        
        Args:
            state: 現在の環境の状態
            reset (bool): 新しいエピソードを開始する場合はTrue
        """
        
        # フルシーケンスモードでresetフラグが立っているか、行動系列が空の場合は新しく計画
        if reset or self.current_action_sequence is None or not self.use_full_sequence:
            # モデルを訓練モードに設定 (cudnnでの逆伝播を可能にするため)
            self.dynamics_model.train()
            self.reward_model.train()
            
            # 最適化対象の行動系列を初期化
            # requires_grad=Trueにすることで、このテンソルに対する勾配を計算できるようになる
            action_sequence = torch.zeros(1, self.plan_horizon, self.action_dim,
                                        requires_grad=True, device=self.device)
            
            # 現在の状態をTensorに変換
            current_state = torch.from_numpy(state).float().to(self.device).unsqueeze(0).unsqueeze(1)  # [1, 1, state_dim]
            
            # オプティマイザを作成
            optimizer = torch.optim.Adam([action_sequence], lr=self.learning_rate)
            
            # 勾配ベースの最適化ループ
            for _ in range(self.num_iterations):
                # 勾配をリセット
                optimizer.zero_grad()
                
                # --- 順伝播 ---
                # 行動系列を使って未来の状態系列を予測
                states_seq = [current_state]
                temp_state = current_state.clone()
                
                for i in range(self.plan_horizon):
                    action = action_sequence[:, i:i+1, :]  # [1, 1, action_dim]
                    next_state = self.dynamics_model(temp_state, action)
                    states_seq.append(next_state)
                    temp_state = next_state
                
                # 最初の状態を除いて結合（最初の状態は現在の状態で予測ではない）
                predicted_states_seq = torch.cat(states_seq[1:], dim=1)  # [1, plan_horizon, state_dim]
                
                # 予測した状態・行動系列から最終報酬を予測
                predicted_reward = self.reward_model(predicted_states_seq, action_sequence)
                
                # --- 誤差逆伝播 ---
                # 報酬を最大化したいので、損失は-1を掛けたものとする
                loss = -predicted_reward
                
                # 勾配を計算
                loss.backward()
                
                # オプティマイザを使って行動系列を更新
                optimizer.step()
                
                # 行動の範囲内にクリップ
                with torch.no_grad():
                    action_sequence.data.clamp_(-self.action_high, self.action_high)
            
            # フルシーケンスモードの場合は、行動系列全体を保存
            if self.use_full_sequence:
                self.current_action_sequence = action_sequence.detach().cpu().numpy()[0]
                self.current_action_index = 0
            
            # モデルを元の状態に戻す
            self.dynamics_model.eval()
            self.reward_model.eval()
        
        # 返す行動の決定
        if self.use_full_sequence:
            # 現在のインデックスの行動を取得
            if self.current_action_index < self.plan_horizon:
                action = self.current_action_sequence[self.current_action_index]
                self.current_action_index += 1
            else:
                # インデックスがホライゾンを超えた場合、最後の行動を返す
                action = self.current_action_sequence[-1]
            return action
        else:
            # 従来通り最初の行動だけを返す
            return action_sequence.detach().cpu().numpy()[0, 0]

    def reset(self):
        """
        新しいエピソードのために状態をリセットする
        """
        self.current_action_sequence = None
        self.current_action_index = 0
        
    def explore(self, state):
        """
        状態を観測し、探索的な行動を返す
        新しいエピソードの開始時には reset=True を渡す
        """
        return self.plan(state)

    def exploit(self, state):
        """
        状態を観測し、活用的な行動を返す
        新しいエピソードの開始時には reset=True を渡す
        """
        return self.plan(state)