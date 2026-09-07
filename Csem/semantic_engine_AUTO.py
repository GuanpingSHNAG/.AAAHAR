import os
import sys
import torch
import torch.nn as nn

# 添加你的项目路径
sys.path.append("/home/gshang/.AAAHAR/rawdata_train")
sys.path.append("/home/gshang/.AAAHAR/Diffusion")

TIMESTEP=20

from model_linearT import ConditionalUNet, CSIDiffusion
# 导入你训练好的真实 MLP 分类头
from model.supervised.models import MLPClassifier

class SemanticEngine(nn.Module):
    """
    端到端在线语义提取引擎 (基于精确 Autograd Jacobian 版)
    用于提供绝对精确的 Ground Truth C_sem 对比，耗时较长。
    """
    def __init__(self, device="cuda", num_classes=5):
        super().__init__()
        self.device = device
        
        # ================= 1. 路径与核心参数配置 =================
        self.diffusion_path = "/home/gshang/.AAAHAR/Diffusion/best_diffusion_LinearT.pth"
        self.mlp_path = "/home/gshang/.AAAHAR/rawdata_train/best_baseline_model.pth"
        self.target_t_list =list(range(5, 15, 5))  # 融合的时间步
        
        # ================= 2. 加载并彻底冻结 Diffusion =================
        print("[*] SemanticEngine (Autograd): 正在加载并冻结 Diffusion 模型...")
        unet = ConditionalUNet(in_channels=1, out_channels=1)
        unet.load_state_dict(torch.load(self.diffusion_path, map_location=device))
        self.diffusion = CSIDiffusion(unet, timesteps=TIMESTEP).to(device)
        self.diffusion.eval()
        for param in self.diffusion.parameters():
            param.requires_grad = False
            
        # ================= 3. 加载并彻底冻结 MLP 分类头 =================
        print("[*] SemanticEngine (Autograd): 正在加载并冻结 MLP 分类器...")
        self.mlp = MLPClassifier(win_len=50, feature_size=56, num_classes=num_classes).to(device)
        self.mlp.load_state_dict(torch.load(self.mlp_path, map_location=device))
        self.mlp.eval()
        for param in self.mlp.parameters():
            param.requires_grad = False
            
        print("[*] SemanticEngine (Autograd): 初始化完成！准备提供精确版 C_sem。")

    def forward(self, cond_csi, mask):
        """
        在线提取批量的 C_sem 
        Args:
            cond_csi: 输入的已知导频条件 (已经乘以了 mask)[B, 1, 50, 56]
            mask: 导频掩码[B, 1, 50, 56]
        Returns:
            C_sem_batch: 融合后的精确语义协方差矩阵 [B, 2800, 2800]
            gen_csi_batch: Diffusion 补全的 CSI[B, 1, 50, 56]
        """
        B = cond_csi.shape[0]
        
        C_sem_list = []
        gen_csi_list =[]
        
        # 必须逐样本提取，因为 Autograd 算 Jacobian 一次算多个会内存爆炸和混淆
        for i in range(B):
            cond_i = cond_csi[i:i+1] #[1, 1, 50, 56]
            mask_i = mask[i:i+1]     #[1, 1, 50, 56]
            
            # ---------------------------------------------------------
            # Step 1: Supply 侧 - 调用精确的 Autograd 提取均值 Covariance
            # ---------------------------------------------------------
            gen_csi_i, mean_cov_i = self.diffusion.sample_with_autograd_mean_covariance(
                mask=mask_i,
                cond=cond_i,
                target_cov_t_list=self.target_t_list
            )
            
            # ---------------------------------------------------------
            # Step 2: Demand 侧 - MLP 盲提取梯度 (Classifier Guidance)
            # ---------------------------------------------------------
            gen_csi_req_grad = gen_csi_i.clone().detach().requires_grad_(True)
            
            # 【修复关键点】：强制开启计算图引擎，防止外部套了 torch.no_grad()
            with torch.enable_grad():
                logits = self.mlp(gen_csi_req_grad)
                # 无 Label 提取！直接信任模型最大预测概率的类别
                pseudo_loss = logits.max(dim=1)[0].sum()
                # 求解敏感度 (梯度)
                grad_x = torch.autograd.grad(outputs=pseudo_loss, inputs=gen_csi_req_grad)[0]
            
            # 提取重要性 \gamma_i
            gamma_i = grad_x.abs() ** 2             #[1, 1, 50, 56]
            gamma_i = gamma_i.view(-1)              # 展平为 [2800]
            
            # Min-Max 归一化
            gamma_i = gamma_i / (gamma_i.max() + 1e-8)
            
            # ---------------------------------------------------------
            # Step 3: 合同变换融合 - 生成半正定 C_sem
            # ---------------------------------------------------------
            sqrt_gamma = torch.sqrt(gamma_i)
            
            row_scale = sqrt_gamma.unsqueeze(1)     # [2800, 1]
            col_scale = sqrt_gamma.unsqueeze(0)     # [1, 2800]
            
            # 精确 C_sem = \Gamma * Cov * \Gamma
            C_sem_i = mean_cov_i * (row_scale * col_scale)
            C_sem_i = C_sem_i / torch.norm(C_sem_i) 
            C_sem_list.append(C_sem_i.detach())
            gen_csi_list.append(gen_csi_i.detach())
            
        # 拼接返回
        C_sem_batch = torch.stack(C_sem_list, dim=0)   #[B, 2800, 2800]
        gen_csi_batch = torch.cat(gen_csi_list, dim=0) #[B, 1, 50, 56]
        
        return C_sem_batch, gen_csi_batch