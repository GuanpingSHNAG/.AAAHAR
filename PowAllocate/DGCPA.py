import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= 新增：密集图卷积层 =================
class DenseGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 自身节点特征提取
        self.self_linear = nn.Linear(in_features, out_features)
        # 邻居节点消息提取
        self.msg_linear = nn.Linear(in_features, out_features)
        # 层归一化加速收敛
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x, adj):
        """
        x: 节点特征 [B, K, in_features]
        adj: 归一化后的边权重 (协方差矩阵) [B, K, K]
        """
        # 1. 变换自身特征
        self_feat = self.self_linear(x)
        
        # 2. 变换邻居特征并基于边权重(adj)进行聚合
        msg_feat = self.msg_linear(x)
        # 使用批量矩阵乘法完成 Message Passing: [B, K, K] x [B, K, F] -> [B, K, F]
        agg_feat = torch.bmm(adj, msg_feat) 
        
        # 3. 残差连接 + 激活 + 归一化
        out = self_feat + agg_feat
        return F.gelu(self.norm(out))


class Allocator(nn.Module):
    def __init__(self, N=1024, num_dynamic_pilots=300): 
        super().__init__()
        self.N = N
        self.K = num_dynamic_pilots
        
        # 【修改点 1】: 删除了原来依赖 K 维度的 MLP 和 triu_indices
        # 建立基于 GNN 的感知功率分配网络 (完全解耦了 K 的维度)
        hidden_dim = 48
        self.gnn_layers = nn.ModuleList([
            DenseGNNLayer(in_features=1, out_features=hidden_dim),
            DenseGNNLayer(in_features=hidden_dim, out_features=hidden_dim),
            #DenseGNNLayer(in_features=hidden_dim, out_features=hidden_dim)
        ])
        
        # 输出层：将每个节点的 hidden_dim 映射回 1 维标量
        self.power_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.power_head.weight, -0.5)
        nn.init.zeros_(self.power_head.bias)

    def forward(self, c_sem_batch, snr_budget_batch):
        """
        c_sem_batch:[B, 1024, 1024]
        """
        B = c_sem_batch.shape[0]
        device = c_sem_batch.device
        
        # =======================================================
        # Step 1: 确定性与随机性结合的 Top-K 选择 来简化模拟diffusion的随机性
        # =======================================================
        c_sem_diag = torch.diagonal(c_sem_batch, dim1=1, dim2=2).clone() # [B, 1024]
        c_sem_diag[:, :96] = float('-inf')
        
        U = torch.rand_like(c_sem_diag)
        gumbel_noise = -torch.log(-torch.log(U + 1e-8) + 1e-8)
        temperature = 0.5
        noisy_diag = c_sem_diag + gumbel_noise * temperature
        
        _, topk_indices = torch.topk(noisy_diag, self.K, dim=1)
        c_sem_batch = torch.nan_to_num(c_sem_batch, nan=0.0, posinf=5.0, neginf=-5.0)

        # =======================================================
        # Step 2: 提取这 K 个点的联合相关性子矩阵 C'_sem (KxK)
        # =======================================================
        batch_idx = torch.arange(B).view(B, 1, 1).to(device)
        idx_row = topk_indices.view(B, self.K, 1)
        idx_col = topk_indices.view(B, 1, self.K)
        
        c_sem_sub = c_sem_batch[batch_idx, idx_row, idx_col] # [B, K, K]
        
        # ================= [修改点 2: 构造 GNN 输入] =================
        # 1. 构建节点特征 X：取协方差子图的对角线，并增加特征维度 -> [B, K, 1]
        x = torch.diagonal(c_sem_sub, dim1=1, dim2=2).unsqueeze(-1)
        
        # 节点特征归一化 (防崩溃)
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-6
        x = (x - x_mean) / x_std
        
        # 2. 构建邻接矩阵 A (边权重)
        # 利用 Softmax 沿着行维度归一化，相当于将物理相关性转为 GNN 聚合时的注意力权重
        # 除以 32.0 (温度系数) 是为了防止协方差数值过大导致 softmax 极化为 One-hot
        adj = F.softmax(c_sem_sub / 32.0, dim=-1)

        # =======================================================
        # Step 3: GNN 消息传递与功率输出
        # =======================================================
        # 通过 GNN 提取结构化特征
        for layer in self.gnn_layers:
            x = layer(x, adj)
            
        # 打分网络映射：[B, K, hidden_dim] -> [B, K, 1] -> 移除最后一维 -> [B, K]
        power_logits = self.power_head(x).squeeze(-1) 
        
        # 后续功率分配逻辑保持不变
        raw_power = F.softmax(power_logits, dim=-1)
        power_alloc = raw_power * snr_budget_batch * self.K # [B, K]
        
        return topk_indices, power_alloc


    def simulate_sensing_link(self, clean_csi, topk_indices, power_alloc, fixed_mask):
        # 此函数完全无需修改，保持你原有的逻辑即可
        B = clean_csi.shape[0]
        device = clean_csi.device
        clean_csi_1d = clean_csi.view(B, self.N)
        fixed_mask_1d = fixed_mask.view(B, self.N)
        
        dynamic_mask_1d = torch.zeros(B, self.N, device=device)
        dynamic_mask_1d.scatter_(1, topk_indices, 1.0)
        
        clean_selected = clean_csi_1d.gather(1, topk_indices)
        
        signal_power = torch.mean(clean_csi_1d ** 2, dim=1, keepdim=True) + 1e-8
        valid_power = torch.clamp(power_alloc, min=1e-9)
        noise_std = torch.sqrt(signal_power / valid_power)
        
        noise = torch.randn_like(clean_selected)
        noise_term = noise * noise_std
        
        y_dynamic_selected = clean_selected + noise_term
        
        y_dynamic = torch.zeros_like(clean_csi_1d)
        y_dynamic.scatter_(1, topk_indices, y_dynamic_selected)
        
        y_fixed = clean_csi_1d * fixed_mask_1d
        
        y_received_1d = y_dynamic + y_fixed
        y_received = y_received_1d.view(B, 1, 32, 32)

        total_mask_1d = (dynamic_mask_1d + fixed_mask_1d).clamp(max=1.0)
        total_mask_2d = total_mask_1d.view(B, 1, 32, 32)
        
        return y_received, total_mask_2d