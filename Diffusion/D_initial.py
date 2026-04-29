import torch
import math
from tqdm import tqdm

def get_1d_dct_matrix(N, device='cpu'):
    """生成 1D DCT-II 转换矩阵"""
    n = torch.arange(N, device=device)
    k = torch.arange(N, device=device).unsqueeze(1)
    matrix = torch.cos(math.pi / N * k * (n + 0.5))
    matrix[0] *= 1 / math.sqrt(2)
    matrix *= math.sqrt(2 / N)
    return matrix

def get_2d_dct_matrix(H, W, device='cpu'):
    """生成 2D 展平数据的 DCT 转换矩阵 Gamma"""
    C_H = get_1d_dct_matrix(H, device)
    C_W = get_1d_dct_matrix(W, device)
    # 使用克罗内克积 (Kronecker product) 组合 2D 转换
    Gamma = torch.kron(C_H, C_W)
    return Gamma

def compute_DCT_prior_variance(train_loader, H=50, W=56, device='cpu'):
    """
    遍历训练集，计算 DCT 域下的对角线方差 D_init
    """
    Gamma = get_2d_dct_matrix(H, W, device)
    N = H * W
    
    sum_x = torch.zeros(N, device=device)
    sum_sq_x = torch.zeros(N, device=device)
    total_samples = 0
    
    print("[*] 开始计算训练数据的 DCT 先验方差 D_init...")
    with torch.no_grad():
        for inputs, _ in tqdm(train_loader):
            # 将 [B, 1, 50, 56] 展平为 [B, 2800]
            x_flat = inputs.to(device).view(inputs.shape[0], -1) 
            
            # 批量转换到 DCT 空间: Y = X @ Gamma.T
            dct_x = x_flat @ Gamma.t()
            
            sum_x += dct_x.sum(dim=0)
            sum_sq_x += (dct_x ** 2).sum(dim=0)
            total_samples += x_flat.shape[0]
            
    # 计算方差：Var(X) = E[X^2] - (E[X])^2
    mean = sum_x / total_samples
    variance = (sum_sq_x / total_samples) - (mean ** 2)
    
    # 限制最小值防止数值不稳定
    D_init = torch.clamp(variance, min=1e-6)
    
    print(f"[*] 计算完成！D_init 形状: {D_init.shape}")
    return D_init, Gamma