import torch
import torch.nn as nn
import torch.nn.functional as F

class Allocator(nn.Module):
    def __init__(self, N=1024, num_dynamic_pilots=300): # 注意：根据您之前的代码把N和K的默认值对齐了
        super().__init__()
        self.N = N
        self.K = num_dynamic_pilots
        
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


            
            power_alloc = torch.ones(B, self.K, device=device) * snr_budget_batch # [B, K]
            
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