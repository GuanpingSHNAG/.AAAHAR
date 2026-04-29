# /home/gshang/.AAAHAR/Diffusion/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 核心组件 1：Valid-Only Normalization (有效物理场范围约束)
# =====================================================================
def valid_only_normalization(cond, mask, eps=1e-6):
    """
    仅对 Mask=1 的导频区域进行均值和方差统计，防止 98.7% 的 0 元素导致严重均值漂移。
    输入输出形状皆为 [Batch, 1, H, W]
    """
    # 统计有效导频数量 (避免除零错误)
    sum_mask = mask.sum(dim=(2, 3), keepdim=True)
    sum_mask = torch.clamp(sum_mask, min=1.0)
    
    # 仅计算 Mask 内的均值
    mean = (cond * mask).sum(dim=(2, 3), keepdim=True) / sum_mask
    
    # 仅计算 Mask 内的方差
    var = (((cond - mean) * mask) ** 2).sum(dim=(2, 3), keepdim=True) / sum_mask
    std = torch.sqrt(var + eps)
    
    # 标准化并用 Mask 重新过滤，确保空白区域绝对为 0
    cond_norm = ((cond - mean) / std) * mask
    return cond_norm












# =====================================================================
# 基础组件：局部卷积层 (Partial Convolution)
# =====================================================================
class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # 1. 处理真实信号的卷积 (不要 bias，因为我们在后面手动加)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # 2. 处理 Mask 的卷积，用于计算感受野内有效像素的数量
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        
        # 将 mask_conv 的权重固定为 1，并且不需要计算梯度
        nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x, mask):
        # 如果 mask 是 1 个通道，但特征图 x 变成了多通道，把 mask 扩充到对应的通道数
        if mask.shape[1] == 1 and x.shape[1] > 1:
            mask = mask.expand(-1, x.shape[1], -1, -1)

        # --------- 核心逻辑 ---------
        # 1. 强制让 mask 外的元素变为 0 (只处理已知区域)
        x_masked = x * mask
        out = self.conv(x_masked)

        # 2. 计算当前滑窗内有多少个有效像素 (mask_sum)
        with torch.no_grad():
            mask_sum = self.mask_conv(mask)
            mask_sum = torch.clamp(mask_sum, min=1e-8) # 防除零

        # 3. 信号校正放大：有效像素越少，说明被0稀释得越严重，需要放大的倍数就越大
        win_size = self.conv.kernel_size[0] * self.conv.kernel_size[1] * x.shape[1]
        out = out * (win_size / mask_sum) + self.bias.view(1, -1, 1, 1)

        # 4. 更新下一层的 Mask (只要滑窗内有1个有效点，输出的这个位置就算有效)
        new_mask = torch.clamp(mask_sum, 0, 1)
        
        # 最终用新 mask 再过滤一次，保证完全无效的区域严格为 0
        return out * new_mask, new_mask

# =====================================================================
# 改造后的条件编码器 (P-Conv Condition Encoder)
# =====================================================================
class ConditionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 【严禁使用 BatchNorm/GroupNorm】 
        # 注意：现在的输入 in_channels 变成了 1，因为不再 cat mask
        
        # --- 编码器块 1 (输出 64x64) ---
        self.enc1_1 = PartialConv2d(1, 8, kernel_size=3, padding=1)
        self.enc1_2 = PartialConv2d(8, 8, kernel_size=3, padding=1)
        
        # --- 下采样块 1 (输出 32x32) ---
        self.down1_1 = PartialConv2d(8, 16, kernel_size=4, stride=2, padding=1)
        self.down1_2 = PartialConv2d(16, 16, kernel_size=3, padding=1)
        
        # --- 下采样块 2 (输出 16x16) ---
        self.down2_1 = PartialConv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.down2_2 = PartialConv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, mask, cond_norm):
        # 初始状态下：x 仅为归一化后的有效数值，m 为 0/1 掩码
        x, m = cond_norm, mask 

        # --- 第一层级计算 ---
        x, m = self.enc1_1(x, m)
        x = F.silu(x)
        x, m = self.enc1_2(x, m)
        f1 = F.silu(x)

        # --- 第二层级计算 (下采样) ---
        x, m = self.down1_1(f1, m)
        x = F.silu(x)
        x, m = self.down1_2(x, m)
        f2 = F.silu(x)

        # --- 第三层级计算 (下采样) ---
        x, m = self.down2_1(f2, m)
        x = F.silu(x)
        x, m = self.down2_2(x, m)
        f3 = F.silu(x)

        return f1, f2, f3
    













    
# =====================================================================
# 核心组件 3：空间自适应调制层 (Spatial FiLM Injection)
# =====================================================================
class SpatialModulation(nn.Module):
    def __init__(self, in_channels, cond_channels):
        super().__init__()
        # 主干道特征执行 GroupNorm (关闭自带的仿射参数)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels, affine=False)
        # 用 1x1 卷积将导频特征映射为 Scale (gamma) 和 Shift (beta)
        self.proj = nn.Conv2d(cond_channels, in_channels * 2, kernel_size=1)
        
    def forward(self, x, cond_feat):
        x_norm = self.norm(x)
        params = self.proj(cond_feat)
        gamma, beta = params.chunk(2, dim=1) # 切割成两半
        # 仿射变换: x' = (1 + gamma) * Norm(x) + beta
        return x_norm * (1 + gamma) + beta

# =====================================================================
# 主 U-Net 基础构件 (带调制的残差块)
# =====================================================================
class ModulatedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_ch):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        
        self.mod1 = SpatialModulation(in_ch, cond_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        self.mod2 = SpatialModulation(out_ch, cond_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, cond_feat):
        # 1. 调制 -> 激活 -> 卷积
        h = self.mod1(x, cond_feat)
        #h = F.silu(h)
        h = self.conv1(h)
        
        # 2. 注入时间嵌入
        h = h + self.time_mlp(t_emb)[..., None, None]
        
        # 3. 再次调制 -> 激活 -> 卷积
        #h = self.mod2(h, cond_feat)#
        h = self.norm2(h) # 先归一化再激活，避免调制参数过大导致数值爆炸
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.res_conv(x)




    
# 时间嵌入基类
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings












class SelfAttention2d(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        
        # 归一化层
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        
        # Q, K, V 投影 (使用 1x1 卷积代替 Linear，更适合图像特征图)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        
        # 输出投影
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 归一化
        h = self.norm(x)
        
        # 2. 计算 Q, K, V
        # 形状变换: [B, C*3, H, W] -> [B, 3, heads, head_dim, H*W]
        qkv = self.qkv(h).view(B, 3, self.heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2] # 取出 q, k, v，每个形状为 [B, heads, head_dim, N]
        
        # 3. 计算注意力分数 (Attention Scores)
        # q.transpose: [B, heads, N, head_dim]
        # k: [B, heads, head_dim, N]
        # attn: [B, heads, N, N]
        attn = (q.transpose(-2, -1) @ k) * (1.0 / math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        
        # 4. 注意力加权与输出
        # v: [B, heads, head_dim, N]
        # v @ attn.transpose: [B, heads, head_dim, N]
        out = (v @ attn.transpose(-2, -1)).view(B, C, H, W)
        
        # 5. 投影并加上残差连接
        return x + self.proj(out)
    




# =====================================================================
# 最终整合：Modulated Conditional U-Net
# =====================================================================
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=128):
        """
        注意：现在主 U-Net 的 in_channels 变为 1 (仅处理加噪的 X_t)
        """
        super().__init__()
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # 条件编码器
        self.cond_encoder = ConditionEncoder()
        
        # 主干输入层
        self.init_conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # --- 下采样网络 (Down) ---
        self.down1 = ModulatedResBlock(in_ch=8, out_ch=8, time_emb_dim=time_emb_dim, cond_ch=8)
        self.down2 = ModulatedResBlock(in_ch=8, out_ch=16, time_emb_dim=time_emb_dim, cond_ch=16)
        self.down3 = ModulatedResBlock(in_ch=16, out_ch=32, time_emb_dim=time_emb_dim, cond_ch=32)


        # --- 【新增】瓶颈层 (Bottleneck) ---
        # 经过三次下采样，分辨率变为 16x16，通道数为 256
        #self.mid_attn = SelfAttention2d(channels=64, heads=4)
                                        


        # --- 上采样网络 (Up) ---
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 注意：拼接了 skip connection 后，输入通道是 in_ch + skip_ch
        self.up1 = ModulatedResBlock(in_ch=32 + 16, out_ch=16, time_emb_dim=time_emb_dim, cond_ch=16)
        self.up2 = ModulatedResBlock(in_ch=16 + 8,  out_ch=8,  time_emb_dim=time_emb_dim, cond_ch=8)
        
        self.final_conv = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, x_t, t, mask, cond):
        # 1. 物理层空间维度补齐: 50x56 -> 64x64
        x_t = F.pad(x_t, (4, 4, 7, 7), mode='constant', value=0)
        mask = F.pad(mask, (4, 4, 7, 7), mode='constant', value=0)
        cond = F.pad(cond, (4, 4, 7, 7), mode='constant', value=0)
        
        # 2. 有效物理量标准化
        cond_norm = valid_only_normalization(cond, mask)
        
        # 3. 提取空间尺度对齐的条件特征图 (无 Norm 避免塌陷)
        cf1, cf2, cf3 = self.cond_encoder(mask, cond_norm)
        
        # 4. 主干网络计算
        t_emb = self.time_mlp(t)
        
        # [Down 阶段]
        h = self.init_conv(x_t)                # 64x64
        d1 = self.down1(h, t_emb, cf1)         # 64x64 (接受 cf1 调制)
        
        h = self.pool(d1)                      # 32x32
        d2 = self.down2(h, t_emb, cf2)         # 32x32 (接受 cf2 调制)
        
        h = self.pool(d2)                      # 16x16
        d3 = self.down3(h, t_emb, cf3)         # 16x16 (接受 cf3 调制)
        



        #d3 = self.mid_attn(d3) 不用attention 反而效果更好！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！






        # [Up 阶段]
        h = self.upsample(d3)                  # 32x32
        h = torch.cat([h, d2], dim=1)          # 拼接 Skip Connection
        u1 = self.up1(h, t_emb, cf2)           # 32x32 (接受 cf2 调制)
        
        h = self.upsample(u1)                  # 64x64
        h = torch.cat([h, d1], dim=1)          # 拼接 Skip Connection
        u2 = self.up2(h, t_emb, cf1)           # 64x64 (接受 cf1 调制)
        
        out = self.final_conv(u2)
        
        # 5. 裁剪还原回原本的物理维度: 64x64 -> 50x56
        out = out[:, :, 7:-7, 4:-4]
        return out


# =====================================================================
# 扩散过程包装器 (Diffusion Forward Wrapper)
# =====================================================================
class CSIDiffusion(nn.Module):
    def __init__(self, model, timesteps=100):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        # 线性 Noise Schedule
        beta = torch.linspace(1e-4, 0.2, timesteps)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        
        self.register_buffer('beta', beta)
        self.register_buffer('alpha_bar', alpha_bar)

    def forward(self, x_0, mask, cond):
        """训练阶段: 加噪并计算重构损失"""
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        
        # 对干净的 X_0 进行加噪，得到 X_t
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        x_t = x_t*(1-mask) + cond*mask # 强制保持导频位置不被加噪干扰（直接用条件值覆盖）

        
        # 模型基于加噪的 X_t 与 完美的导频 Cond，预测全图施加的 Noise
        pred_noise = self.model(x_t, t, mask, cond)
        
        # 使用 MSE Loss
        loss = F.mse_loss(pred_noise, noise)
        return loss
    


    @torch.no_grad()
    def sample(self, mask, cond, return_all=False):
        """
        从纯噪声开始，逐步去噪生成完整 CSI。
        Args:
            mask: 导频掩码 [B,1,H,W]
            cond: 条件（导频位置的值）[B,1,H,W]
            return_all: 若为 True，返回所有中间步的结果（用于动画）
        Returns:
            x0: 生成结果 [B,1,H,W]
            (可选) intermediates: list of [B,1,H,W]
        """
        B = cond.shape[0]
        H, W = cond.shape[-2:]
        device = cond.device
        
        # 初始纯噪声
        noise_T = torch.randn(B, 1, H, W, device=device)
        x_t=cond+noise_T*(1-mask) # 在导频位置保持原值，其他位置从噪声开始
        alpha = 1 - self.beta
        alpha_bar = self.alpha_bar
        
        if return_all:
            intermediates = [x_t.cpu()]
        
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = self.model(x_t, t_tensor, mask, cond)
            
            alpha_t = alpha[t]
            alpha_bar_t = alpha_bar[t]
            beta_t = self.beta[t]
            
            # DDPM 更新公式
            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            mean = coeff1 * (x_t - coeff2 * pred_noise)
            
            if t > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(beta_t)
                x_t = mean + sigma_t * noise
            else:
                x_t = mean
            
            x_t = x_t*(1-mask) + cond # 强制保持导频位置不被更新（直接用条件值覆盖）


            if return_all:
                intermediates.append(x_t.cpu())


        if return_all:
            return x_t, intermediates
        return x_t



    @torch.enable_grad()
    def get_covariance_matrix(self, x_t, t_val, mask, cond):
        """
        获取真实去噪器协方差矩阵。利用 Autograd 计算 Jacobian。
        注意: 为避免批量维度的交叉梯度，建议 batch_size = 1
        """
        assert x_t.shape[0] == 1, "计算精确协方差矩阵时，必须设置 batch_size=1"
        B, C, H, W = x_t.shape
        N = C * H * W
        device = x_t.device
        
        alpha_bar_t = self.alpha_bar[t_val]
        
        # 1. 包装一个纯函数：输入当前的带噪 x_t，输出预测的干净 x_0
        def compute_x0_hat(x_in):
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
            # 模型预测噪声
            pred_noise = self.model(x_in, t_tensor, mask, cond)
            # 标准 DDPM 公式还原 x0
            x0_hat = (x_in - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            return x0_hat

        # 2. 调用 PyTorch 原生 API 计算 Jacobian 
        # Jacobian 形状将会是: [B, C, H, W, B, C, H, W] 即 [1, 1, 50, 56, 1, 1, 50, 56]
        jac = torch.autograd.functional.jacobian(compute_x0_hat, x_t)
        
        # 3. 将高维张量展平为二维矩阵 [N, N] (即 2800 x 2800)
        jac = jac.view(N, N)
        
        # 4. 根据 Tweedie 定理，将 Jacobian 缩放为真实的 Covariance
        scale_factor = (1 - alpha_bar_t) / torch.sqrt(alpha_bar_t)
        cov_matrix = scale_factor * jac
        
        return cov_matrix


    @torch.no_grad()
    def sample_with_covariance(self, mask, cond, target_cov_t=None, return_all=False):
        """
        从纯噪声开始去噪，并在特定的 t 时刻返回真实的协方差矩阵。
        Args:
            mask, cond: 输入条件
            target_cov_t: 指定在哪个时间步 (例如 t=20) 提取并返回协方差矩阵
            return_all: 返回所有中间步骤
        """
        B = cond.shape[0]
        H, W = cond.shape[-2:]
        device = cond.device
        
        # 初始纯噪声
        noise_T = torch.randn(B, 1, H, W, device=device)
        x_t = cond + noise_T * (1 - mask) 
        alpha = 1 - self.beta
        alpha_bar = self.alpha_bar
        
        if return_all:
            intermediates = [x_t.cpu()]
            
        extracted_cov = None
        
        for t in reversed(range(self.timesteps)):
            
            # =============== 【核心拦截点】 ===============
            # 如果到达了指定的 t 并且 batch_size 为 1，则触发求导机制计算协方差
            if target_cov_t is not None and t == target_cov_t and B == 1:
                # 克隆一份独立的 x_t 并开启梯度，避免破坏原有的 no_grad 上下文
                x_t_req_grad = x_t.clone().detach().requires_grad_(True)
                extracted_cov = self.get_covariance_matrix(x_t_req_grad, t, mask, cond)
            # ==============================================
            
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = self.model(x_t, t_tensor, mask, cond)
            
            alpha_t = alpha[t]
            alpha_bar_t = alpha_bar[t]
            beta_t = self.beta[t]
            
            # DDPM 更新公式
            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            mean = coeff1 * (x_t - coeff2 * pred_noise)
            
            if t > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(beta_t)
                x_t = mean + sigma_t * noise
            else:
                x_t = mean
            
            x_t = x_t * (1 - mask) + cond

            if return_all:
                intermediates.append(x_t.cpu())

        # 返回生成结果，如果提取了协方差则一并返回
        result = (x_t, extracted_cov) if target_cov_t is not None else x_t
        if return_all:
            return result, intermediates
        return result
    


    @torch.no_grad() # Free Hunch 纯前向，完全不需要开启梯度！
    def sample_with_FHcovariance(self, mask, cond, target_cov_t_list=None, D_init=None, Gamma=None, return_all=False):
        """
        使用 Free Hunch 方法在线迭代计算并提取多个特定 t 时刻的协方差矩阵。
        Args:
            mask, cond: 输入条件
            target_cov_t_list: 想要提取协方差矩阵的时间步数组，例如 [80, 50, 20]
            D_init: 初始化对角阵 (如不提供默认用全1)
            return_all: 返回所有中间步骤
        Returns:
            x_t: 最终生成结果
            extracted_covs: 字典格式，键为 t，值为 2800x2800 的真实协方差矩阵
        """
        if target_cov_t_list is None:
            target_cov_t_list = []
            
        B = cond.shape[0]
        H, W = cond.shape[-2:]
        device = cond.device
        N = H * W  # 对于你的是 50x56 = 2800
        
        # 1. 初始化 Free Hunch 管理器
        if D_init is None:
            # 如果没有提供数据先验方差，假设归一化后方差为 1
            D_init = torch.ones(N, device=device) 
            
        fh_manager = FreeHunchManager(D_init,Gamma, device=device, max_rank=20)
        
        # 初始纯噪声
        noise_T = torch.randn(B, 1, H, W, device=device)
        x_t = cond + noise_T * (1 - mask) 
        alpha = 1 - self.beta
        alpha_bar = self.alpha_bar
        
        if return_all:
            intermediates = [x_t.cpu()]
            
        extracted_covs = {}  # 存放提取的矩阵结果
        
        # 用于保存上一步的状态，以计算轨迹的差分 (dx 和 dmu)
        x_prev = None
        mu_prev = None
        sigma_prev = None
        
        for t in reversed(range(self.timesteps)):
            
            # --- 1. 网络前向预测 ---
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = self.model(x_t, t_tensor, mask, cond)
            
            alpha_t = alpha[t]
            alpha_bar_t = alpha_bar[t]
            beta_t = self.beta[t]
            
            # [核心转换]：将 DDPM 的离散排期映射到连续的等效噪声水平 sigma
            # 物理意义：当前 x_t 相对于干净图像的噪声标准差
            sigma_t = torch.sqrt((1 - alpha_bar_t) / alpha_bar_t).item()
            
            # 根据 DDPM 公式计算当前步预测的“纯净信号” mu_t (即 x0_hat)
            mu_t = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            

# ==========================================================
            # --- 2. Free Hunch 核心更新逻辑 (包含解析分数传递) ---
            # ==========================================================
# ==========================================================
            # --- 2. Free Hunch 核心更新逻辑 (包含解析分数传递) ---
            # ==========================================================
            if sigma_prev is not None:
                # 【修改这里】：针对 T=100 放宽 sigma 限制。
                # 0.05 对应大概 t=3 左右，避免最后几步极低噪声导致的数值除零爆炸
                is_active_window = 0.05 <= sigma_t <= 5.0
                is_active_window =1
                # (a) 拦截点：使用解析分数传递同步 mu_prev
                if is_active_window:
                    mu_prev_aligned = fh_manager.analytical_score_transport(
                        x_spatial=x_prev, 
                        mu_curr_spatial=mu_prev, 
                        sigma_curr=sigma_prev, 
                        sigma_next=sigma_t
                    )
                else:
                    mu_prev_aligned = mu_prev 

                # (b) 时间更新 (Time Update)
                fh_manager.time_update(sigma_curr=sigma_prev, sigma_next=sigma_t)
                
                # (c) 空间更新 (Space Update / BFGS)
                if is_active_window:
                    dx = x_t - x_prev           
                    dmu = mu_t - mu_prev_aligned 
                    fh_manager.space_update(dx, dmu, sigma_t)
                    
            # 记录当前状态给下一步用
            x_prev = x_t.clone()
            mu_prev = mu_t.clone()
            sigma_prev = sigma_t
            
            # ==========================================================
            # --- 3. 拦截点：提取多时刻的协方差矩阵 ---
            # ==========================================================
            if t in target_cov_t_list:
                # 极速获取当前的 2800x2800 矩阵，耗时不到 1 毫秒
                extracted_covs[t] = fh_manager.get_full_matrix().clone()
            
            
            # --- 4. 标准 DDPM 采样步进 (走向下一个 t) ---
            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            mean = coeff1 * (x_t - coeff2 * pred_noise)
            
            if t > 0:
                noise = torch.randn_like(x_t)
                sigma_ddpm = torch.sqrt(beta_t)
                x_t = mean + sigma_ddpm * noise
            else:
                x_t = mean
            
            # 强制导频约束
            x_t = x_t * (1 - mask) + cond

            if return_all:
                intermediates.append(x_t.cpu())

        result = (x_t, extracted_covs) if len(target_cov_t_list) > 0 else x_t
        if return_all:
            return result, intermediates
        return result


class FreeHunchManager:
    def __init__(self, D_init, Gamma, device, max_rank=1):
        self.N = D_init.shape[0]
        self.device = device
        self.max_rank = max_rank
        self.Gamma = Gamma.to(device) 
        
        self.D = D_init.clone().to(device)
        self.U = torch.zeros((self.N, 0), device=device)
        self.V = torch.zeros((self.N, 0), device=device)
        

    def _invert_indefinite(self, d, U, V):
        """
        纯实数域的不定矩阵 Woodbury 求逆，核心 EVD 同样由 CPU 护航。
        """
        # 对角线安全求逆
        d_safe = torch.where(d.abs() < 1e-8, torch.sign(d) * 1e-8 + (d == 0) * 1e-8, d)
        d_inv = 1.0 / d_safe
        
        if U.shape[1] == 0 and V.shape[1] == 0:
            return d_inv, U, V
            
        W = torch.cat([U, V], dim=1)
        k_u, k_v = U.shape[1], V.shape[1]
        
        d_inv_W = d_inv.unsqueeze(1) * W
        Wt_d_inv_W = W.t() @ d_inv_W
        
        # 构建签名矩阵 S
        S = torch.zeros((k_u + k_v, k_u + k_v), device=self.device)
        if k_u > 0: S[:k_u, :k_u] = torch.eye(k_u, device=self.device)
        if k_v > 0: S[k_u:, k_u:] = -torch.eye(k_v, device=self.device)
            
        K = S + Wt_d_inv_W
        
        # ==========================================================
        # --- 核心安全区：将极小的不定矩阵交由 CPU 处理 ---
        K_cpu = K.detach().cpu()
        K_cpu = (K_cpu + K_cpu.t()) / 2.0 # 强制实对称
        
        # 熔断机制：若包含 NaN/Inf，丢弃历史秩信息，相当于一次软重启
        if torch.isnan(K_cpu).any() or torch.isinf(K_cpu).any():
            return d_inv, torch.zeros((self.N, 0), device=self.device), torch.zeros((self.N, 0), device=self.device)
            
        L, Q = torch.linalg.eigh(K_cpu)
        L = L.to(self.device)
        Q = Q.to(self.device)
        # ==========================================================
        
        L_safe = torch.where(L.abs() < 1e-8, torch.sign(L) * 1e-8 + (L == 0) * 1e-8, L)
        L_inv = 1.0 / L_safe
        
        P = d_inv_W @ Q
        neg_L_inv = -L_inv # Woodbury 恒等式外部有一个负号
        
        pos_mask = neg_L_inv > 1e-8
        neg_mask = neg_L_inv < -1e-8
        
        U_inv_new = P[:, pos_mask] * torch.sqrt(neg_L_inv[pos_mask]).unsqueeze(0)
        V_inv_new = P[:, neg_mask] * torch.sqrt(-neg_L_inv[neg_mask]).unsqueeze(0)
        
        return d_inv, U_inv_new, V_inv_new
    

    def analytical_score_transport(self, x_spatial, mu_curr_spatial, sigma_curr, sigma_next):
        """
        解析分数传递：利用 Hessian 动态将 mu 从 sigma_curr 映射到 sigma_next
        """
        # 1. 转换到 DCT 空间并计算当前 score
        x_dct = self.Gamma @ x_spatial.flatten()
        mu_curr_dct = self.Gamma @ mu_curr_spatial.flatten()
        s_curr = (mu_curr_dct - x_dct) / (sigma_curr**2)
        
        # 2. 从当前协方差 C 提取 Hessian 矩阵: H = (C - sigma^2 I) / sigma^4
        H_curr_diag = (self.D - sigma_curr**2) / (sigma_curr**4)
        H_curr_U = self.U / (sigma_curr**2)
        H_curr_V = self.V / (sigma_curr**2)
        
        # 3. 获取当前 H 的逆
        H_inv_curr_diag, H_inv_curr_U, H_inv_curr_V = self._invert_indefinite(H_curr_diag, H_curr_U, H_curr_V)
        
        # 4. 时间更新：H_inv_next = H_inv_curr + (sigma_curr^2 - sigma_next^2) I
        H_inv_next_diag = H_inv_curr_diag + (sigma_curr**2 - sigma_next**2)
        H_inv_next_U, H_inv_next_V = H_inv_curr_U, H_inv_curr_V
        
        # 5. 获取新的 H
        H_next_diag, H_next_U, H_next_V = self._invert_indefinite(H_inv_next_diag, H_inv_next_U, H_inv_next_V)
        
        # 6. 执行传递: s_next = H_next @ (H_inv_curr @ s_curr)
        # 第一步: v = H_inv_curr @ s_curr
        v = H_inv_curr_diag * s_curr
        if H_inv_curr_U.shape[1] > 0:
            v += H_inv_curr_U @ (H_inv_curr_U.t() @ s_curr)
        if H_inv_curr_V.shape[1] > 0:
            v -= H_inv_curr_V @ (H_inv_curr_V.t() @ s_curr)
            
        # 第二步: s_next = H_next @ v
        s_next = H_next_diag * v
        if H_next_U.shape[1] > 0:
            s_next += H_next_U @ (H_next_U.t() @ v)
        if H_next_V.shape[1] > 0:
            s_next -= H_next_V @ (H_next_V.t() @ v)
            
        # 7. 映射回物理空间
        mu_next_dct = x_dct + (sigma_next**2) * s_next
        mu_next_spatial = self.Gamma.t() @ mu_next_dct
        return mu_next_spatial.view_as(x_spatial)
    
    def _safe_inverse_sqrt(self, A):
        """
        计算 A^{-1/2}，将微小矩阵转移到 CPU 计算，彻底规避 GPU cuSOLVER 的内存越界 Bug。
        """
        # 1. 转移到 CPU，并强制实对称
        A_cpu = A.detach().cpu()
        A_cpu = (A_cpu + A_cpu.t()) / 2.0
        
        # 2. 熔断机制：如果由于极端轨迹导致数值爆炸，直接返回单位阵保命
        if torch.isnan(A_cpu).any() or torch.isinf(A_cpu).any():
            return torch.eye(A.shape[0], device=A.device)
            
        # 3. 在极其稳定的 CPU LAPACK 上执行特征值分解
        L, Q = torch.linalg.eigh(A_cpu)
        
        # 4. 限制最小特征值
        L_safe = torch.clamp(L, min=1e-8)
        
        # 5. 构造 W = Q * L^{-1/2} 并无缝送回 GPU
        W_cpu = Q * (1.0 / torch.sqrt(L_safe)).unsqueeze(0)

        return W_cpu.to(A.device)
    

    def _invert_low_rank(self, D, U, V):
        # 增加极小的下限，防止出现 Inf
        D_inv = 1.0 / torch.clamp(D, min=1e-8)
        
        if U.shape[1] > 0:
            D_inv_U = D_inv.unsqueeze(1) * U
            K = torch.eye(U.shape[1], device=self.device) + U.t() @ D_inv_U
            
            # 【修复点 1】：使用绝对安全的 EVD 替代 Cholesky
            L_mat = self._safe_inverse_sqrt(K)
            V_prime = D_inv_U @ L_mat
        else:
            V_prime = torch.zeros((self.N, 0), device=self.device)

        if V.shape[1] > 0:
            term1 = V.t() @ (D_inv.unsqueeze(1) * V)
            
            V_prime_t_V = V_prime.t() @ V               
            term2 = (V.t() @ V_prime) @ V_prime_t_V     
            
            M = torch.eye(V.shape[1], device=self.device) - (term1 - term2)
            
            # 【修复点 2】：使用绝对安全的 EVD 替代 Cholesky
            W = self._safe_inverse_sqrt(M)
            
            U_prime = (D_inv.unsqueeze(1) * V - V_prime @ V_prime_t_V) @ W
        else:
            U_prime = torch.zeros((self.N, 0), device=self.device)
            
        return D_inv, U_prime, V_prime

    def time_update(self, sigma_curr, sigma_next):
        # 防止除 0 导致数值溢出
        sigma_curr = max(sigma_curr, 1e-5)
        sigma_next = max(sigma_next, 1e-5)
        
        D_inv, U_inv, V_inv = self._invert_low_rank(self.D, self.U, self.V)
        delta_inv_sigma2 = sigma_next**-2 - sigma_curr**-2
        
        D_inv_new = D_inv + delta_inv_sigma2
        self.D, self.U, self.V = self._invert_low_rank(D_inv_new, U_inv, V_inv)
   
   
    def space_update(self, dx_spatial, dmu_spatial, sigma_curr):
        dx_dct = self.Gamma @ dx_spatial.flatten()
        dmu_dct = self.Gamma @ dmu_spatial.flatten()
        
        de = (sigma_curr**2) * dmu_dct
        
        # 1. 计算 C_{old} * dx
        sig_dx = self.D * dx_dct
        if self.U.shape[1] > 0:
            sig_dx += self.U @ (self.U.t() @ dx_dct)
        if self.V.shape[1] > 0:
            sig_dx -= self.V @ (self.V.t() @ dx_dct)
            
        # 2. 处理 de de^T / (dx^T de)
        denom_u = torch.dot(de, dx_dct)
        if denom_u > 1e-8:
            u_new = de / torch.sqrt(denom_u)
            self.U = torch.cat([self.U, u_new.view(-1, 1)], dim=1)
        elif denom_u < -1e-8:
            # 【核心修复 1】：当曲率为负，这是一个负定更新，必须追加到 V 矩阵！
            v_new_de = de / torch.sqrt(-denom_u)
            self.V = torch.cat([self.V, v_new_de.view(-1, 1)], dim=1)
            
        # 3. 处理 - (C_{old} dx)(C_{old} dx)^T / (dx^T C_{old} dx)
        denom_v = torch.dot(dx_dct, sig_dx)
        if denom_v > 1e-8:
            v_new_sig = sig_dx / torch.sqrt(denom_v)
            self.V = torch.cat([self.V, v_new_sig.view(-1, 1)], dim=1)
        elif denom_v < -1e-8:
            # 【核心修复 2】：如果 C_old 本身导出了负方差，翻转为正更新追加到 U
            u_new_sig = sig_dx / torch.sqrt(-denom_v)
            self.U = torch.cat([self.U, u_new_sig.view(-1, 1)], dim=1)

        # 4. 严格维持内存与计算效率
        if self.U.shape[1] > self.max_rank:
            self.U = self.U[:, -self.max_rank:]
        if self.V.shape[1] > self.max_rank:
            self.V = self.V[:, -self.max_rank:]

    def get_full_matrix(self):
        """
        【极致优化提取】：消除 O(N^3) 运算
        """
        # 原版 Gamma.t() @ diag(D) @ Gamma 需要两次 N^3 运算
        # 新版 (Gamma.t() * D) @ Gamma 利用广播，将计算量直接压缩至原本的 1/2 以下！
        base_cov = (self.Gamma.t() * self.D) @ self.Gamma
        
        # 将 U 和 V 从 DCT 映射回物理空间 (仅 O(N^2 * r)，极快)
        U_spatial = self.Gamma.t() @ self.U
        V_spatial = self.Gamma.t() @ self.V
        
        cov_spatial = base_cov
        if self.U.shape[1] > 0:
            cov_spatial += U_spatial @ U_spatial.t()
        if self.V.shape[1] > 0:
            cov_spatial -= V_spatial @ V_spatial.t()
            
        return cov_spatial