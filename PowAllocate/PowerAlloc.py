import torch
import torch.nn as nn
import torch.nn.functional as F

class Allocator(nn.Module):
    def __init__(self, N=1024, num_dynamic_pilots=300): # 注意：根据您之前的代码把N和K的默认值对齐了
        super().__init__()
        self.N = N
        self.K = num_dynamic_pilots
        
        # 【修改点 1】: 计算对称矩阵的不重复元素个数
        # 公式: K*(K-1)/2 (非对角线的一半) + K (对角线) = K*(K+1)/2
        input_dim = self.K * (self.K + 1) // 2
        
        # 建立感知功率分配 MLP
        self.power_mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, self.K)
        )

        # 【修改点 2】: 预先生成上三角的行列索引，并注册为 buffer
        # 这样在模型被 .to(device) 时，这两个索引也会自动转移到对应的 GPU 上，不占用额外的计算时间
        triu_row, triu_col = torch.triu_indices(self.K, self.K)
        self.register_buffer('triu_row', triu_row)
        self.register_buffer('triu_col', triu_col)

    def forward(self, c_sem_batch, snr_budget_batch):
            """
            c_sem_batch:[B, 1024, 1024]
            """
            B = c_sem_batch.shape[0]
            device = c_sem_batch.device
            
            # =======================================================
            # Step 1: 确定性与随机性结合的 Top-K 选择
            # =======================================================
            c_sem_diag = torch.diagonal(c_sem_batch, dim1=1, dim2=2).clone() # [B, 1024]
            
            # 排除前3个时间步 (前 96 个元素)，强制设为负无穷，保证绝对不会被选到
            c_sem_diag[:, :96] = float('-inf')
            # 加入 Gumbel 噪声进行随机采样来抵消diffusion的噪声随机性，增强模型鲁棒性
            # temperature 控制随机性大小：T 接近 0 -> 退化为严格的 Top-K (确定性) T 越大 -> 噪声主导，越接近完全随机采样
            U = torch.rand_like(c_sem_diag)
            gumbel_noise = -torch.log(-torch.log(U + 1e-8) + 1e-8) # 计算 Gumbel 噪声: -log(-log(U))
            
            # 将噪声按温度缩放后加到原本的对角线能量上
            temperature = 0.1
            noisy_diag = c_sem_diag + gumbel_noise * temperature
            
            # 取加噪后的 Top-K，原本能量高的点，加上噪声后大概率依然排在前面；原本在边缘排徊的点，由于噪声的存在，有机会被选中
            _, topk_indices = torch.topk(noisy_diag, self.K, dim=1)
            c_sem_batch = torch.nan_to_num(c_sem_batch, nan=0.0, posinf=5.0, neginf=-5.0)
            #=======================================================
            # Step 2: 提取这 K 个点的联合相关性子矩阵 C'_sem (KxK)
            # =======================================================
            batch_idx = torch.arange(B).view(B, 1, 1).to(device)
            idx_row = topk_indices.view(B, self.K, 1)
            idx_col = topk_indices.view(B, 1, self.K)
            
            c_sem_sub = c_sem_batch[batch_idx, idx_row, idx_col] # [B, K, K]
            
            # 提取上三角元素
            c_sem_sub_triu = c_sem_sub[:, self.triu_row, self.triu_col] # [B, K*(K+1)/2]
            
            # ================= [防崩溃极简补丁：特征归一化] =================
            # 强制将 4 万多维的输入拉回安全范围 (均值 0，方差 1)
            # 避免第一层 Linear 发生天文数字相加
            c_mean = c_sem_sub_triu.mean(dim=1, keepdim=True)
            c_std = c_sem_sub_triu.std(dim=1, keepdim=True) + 1e-6
            c_sem_norm = (c_sem_sub_triu - c_mean) / c_std
            # ==============================================================

            # =======================================================
            # Step 3: MLP 输出功率分配
            # =======================================================
            # 注意这里喂进去的是归一化后的 c_sem_norm！
            power_logits = self.power_mlp(c_sem_norm) 
            
            raw_power = torch.sigmoid(power_logits) 
            raw_power = raw_power / (raw_power.sum(dim=-1, keepdim=True) + 1e-6) 
            power_alloc = raw_power * snr_budget_batch *self.K # [B, K]
            
            return topk_indices, power_alloc





    def simulate_sensing_link(self, clean_csi, topk_indices, power_alloc, fixed_mask):
            B = clean_csi.shape[0]
            device = clean_csi.device
            clean_csi_1d = clean_csi.view(B, self.N)
            fixed_mask_1d = fixed_mask.view(B, self.N)
            
            dynamic_mask_1d = torch.zeros(B, self.N, device=device)
            dynamic_mask_1d.scatter_(1, topk_indices, 1.0)
            
            # 1. 提取被选中的干净信号 [B, K]
            clean_selected = clean_csi_1d.gather(1, topk_indices)
            
            # ================= [真正的核心修正：基于信号真实功率计算噪声] =================
            # 计算当前每个样本 CSI 的平均真实物理功率 
            # dim=1 求整个 1024 维度的均方值，keepdim=True 保证形状为 [B, 1] 方便后续广播
            # 加上 1e-8 防止遇到全 0 背景导致 log(0) 或除零崩溃
            signal_power = torch.mean(clean_csi_1d ** 2, dim=1, keepdim=True) + 1e-8
            
            # 对分配的功率设置下限，防止网络给极小值导致除以 0
            valid_power = torch.clamp(power_alloc, min=1e-3)
            
            # 根据推导公式：噪声标准差 = sqrt(信号真实功率 / 分配的目标SNR)
            noise_std = torch.sqrt(signal_power / valid_power)
            
            # 生成标准高斯噪声，并乘上计算好的标准差！
            noise = torch.randn_like(clean_selected)
            noise_term = noise * noise_std
            # ============================================================================
            
            # 局部加噪 (现在的 noise_term 是与信号量级完美匹配的)
            y_dynamic_selected = clean_selected + noise_term
            
            # 2. 将加完噪的信号塞回 1024 维的全图阵列中
            y_dynamic = torch.zeros_like(clean_csi_1d)
            y_dynamic.scatter_(1, topk_indices, y_dynamic_selected)
            
            # 固定反馈导频信号
            y_fixed = clean_csi_1d * fixed_mask_1d
            
            # 合并！
            y_received_1d = y_dynamic + y_fixed
            y_received = y_received_1d.view(B, 1, 32, 32)

            total_mask_1d = (dynamic_mask_1d + fixed_mask_1d).clamp(max=1.0)
            total_mask_2d = total_mask_1d.view(B, 1, 32, 32)
            
            return y_received, total_mask_2d