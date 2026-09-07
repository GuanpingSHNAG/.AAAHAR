import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加你的项目路径
sys.path.append("/home/gshang/.AAAHAR/rawdata_train")
sys.path.append("/home/gshang/.AAAHAR/Diffusion")

TIMESTEP=200

from model_linearT import ConditionalUNet, CSIDiffusion
# 导入你训练好的真实 MLP 分类头
from model.supervised.models import MLPClassifier

class SemanticEngine(nn.Module):
    """
    端到端在线语义提取引擎 (无监督 Classifier Guidance 版)
    无论在训练还是推理阶段，均无需真实标签。
    """
    def __init__(self, device="cuda", num_classes=5):
        super().__init__()
        self.device = device
        
        # ================= 1. 路径与核心参数配置 =================
        self.diffusion_path = "/home/gshang/.AAAHAR/Diffusion/best_diffusion_LinearT.pth"
        self.mlp_path = "/home/gshang/.AAAHAR/Csem/稀疏随机MLP/pre.pth"
        self.target_t_list = list(range(50, TIMESTEP-50, 10))  # 融合的“语义甜点区”时间步
        
        # ================= 2. 加载并彻底冻结 Diffusion =================
        print("[*] SemanticEngine: 正在加载并冻结 Diffusion 模型...")
        unet = ConditionalUNet(in_channels=1, out_channels=1)
        unet.load_state_dict(torch.load(self.diffusion_path, map_location=device))
        self.diffusion = CSIDiffusion(unet, timesteps=TIMESTEP).to(device)
        self.diffusion.eval()
        for param in self.diffusion.parameters():
            param.requires_grad = False
            
        # ================= 3. 加载并彻底冻结 MLP 分类头 =================
        print("[*] SemanticEngine: 正在加载并冻结 MLP 分类器...")
        # 按照 rawdata_train.py 的配置初始化
        self.mlp = MLPClassifier(win_len=32, feature_size=32, num_classes=num_classes).to(device)
        self.mlp.load_state_dict(torch.load(self.mlp_path, map_location=device))
        self.mlp.eval()
        for param in self.mlp.parameters():
            param.requires_grad = False
            
        # ================= 4. Free Hunch 初始化矩阵 =================
        # 根据你 32x32 的 CSI 数据维度
        self.N = 32 * 32
        self.D_init = torch.ones(self.N, device=self.device)
        self.Gamma_mat = torch.eye(self.N, device=self.device)
        
        print("[*] SemanticEngine: 初始化完成！准备提供无监督 C_sem。")

    def forward(self, cond_csi, mask):
        """
        在线提取批量的 C_sem (可直接作为其他 DNN 的网络层调用)
        Args:
            cond_csi: 输入的已知导频条件 (已经乘以了 mask)[B, 1, 32, 32]
            mask: 导频掩码[B, 1, 32, 32]
        Returns:
            C_sem_batch: 融合后的语义协方差矩阵 [B, 1024, 1024]
            gen_csi_batch: Diffusion 补全的 CSI[B, 1, 32, 32]
        """
        B = cond_csi.shape[0]
        
        C_sem_list = []
        gen_csi_list =[]
        
        # 【极其重要】FreeHunch 的 EVD (特征值分解) 是单样本状态机，必须逐样本提取
        for i in range(B):
            cond_i = cond_csi[i:i+1] #[1, 1, 32, 32]
            mask_i = mask[i:i+1]     #[1, 1, 32, 32]
            
            # ---------------------------------------------------------
            # Step 1: Supply 侧 - 调用 FH 极速提取均值 Covariance
            # ---------------------------------------------------------
            gen_csi_i, mean_cov_i = self.diffusion.sample_with_FH_mean_covariance(
                mask=mask_i,
                cond=cond_i,
                target_cov_t_list=self.target_t_list,
                D_init=self.D_init,
                Gamma=self.Gamma_mat
            )
            
# ---------------------------------------------------------
            # Step 2: Demand 侧 - MLP 盲提取梯度 (Classifier Guidance)
            # ---------------------------------------------------------
            # 开启梯度追踪，但这本身还不够，必须配合 enable_grad 抵消外部的 no_grad
            gen_csi_req_grad = gen_csi_i.clone().detach().requires_grad_(True)
            
            # 【修复关键点】：强制开启计算图引擎！
            with torch.enable_grad():
                # 前向传播 (MLP 看完补全图后给出各个动作的置信度)
                logits = self.mlp(gen_csi_req_grad)
                
                # 无 Label 提取！直接信任模型最大预测概率的类别
                pseudo_loss = logits.max(dim=1)[0].sum()
                
                # 求出该最强置信度对输入 CSI 像素的敏感度 (梯度)
                # 这时因为开启了 enable_grad，pseudo_loss 就有了完整的计算图
                grad_x = torch.autograd.grad(outputs=pseudo_loss, inputs=gen_csi_req_grad)[0]
            
            # 提取重要性 \gamma_i (一阶梯度绝对值的平方，等价经验 Fisher)
            gamma_i = grad_x.abs() ** 2             # [1, 1, 32, 32]
            gamma_i = gamma_i.view(-1)              # 展平为 [1024]
            
            # Min-Max 归一化，限定在 [0, 1] 之间，防止数值爆炸
            gamma_i = gamma_i / (gamma_i.max() + 1e-8)
            
            # ---------------------------------------------------------
            # Step 3: 合同变换融合 - 生成半正定 C_sem
            # ---------------------------------------------------------
            sqrt_gamma = torch.sqrt(gamma_i)        # 对角元素 [1024]
            
            # 极其优雅的 O(N^2) 矩阵广播相乘，取代 O(N^3)
            row_scale = sqrt_gamma.unsqueeze(1)     # [1024, 1]
            col_scale = sqrt_gamma.unsqueeze(0)     #[1, 1024]
            
            # C_sem = \Gamma * Cov * \Gamma (严格 PSD)
            C_sem_i = mean_cov_i * (row_scale * col_scale)
            
            # ========== 添加的归一化代码 ==========
            C_sem_i = C_sem_i / torch.norm(C_sem_i)  


            C_sem_list.append(C_sem_i.detach())
            gen_csi_list.append(gen_csi_i.detach())
            
        # 拼接返回
        C_sem_batch = torch.stack(C_sem_list, dim=0)   #[B, 1024, 1024]
        gen_csi_batch = torch.cat(gen_csi_list, dim=0) # [B, 1, 32, 32]
        
        return C_sem_batch, gen_csi_batch