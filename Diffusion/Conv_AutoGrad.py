# visualize_covariance.py
import time
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from D_initial import compute_DCT_prior_variance
# 添加你的项目路径
sys.path.append("/home/gshang/.AAAHAR/rawdata_train")
from load.supervised.benchmark_loader import load_benchmark_supervised
from model_linearT import ConditionalUNet, CSIDiffusion  # 确保 model.py 已包含 sample_with_covariance

TARGET=50

def plot_covariance_analysis(real_csi, gen_csi, cov_matrix, target_t, save_path):
    """
    绘制真实CSI、生成CSI、2800x2800协方差矩阵以及50x56方差热力图
    """
    # 1. 数据预处理
    real = real_csi.squeeze() # [50, 56]
    gen = gen_csi.squeeze()   # [50, 56]
    
    # 提取完整协方差矩阵 (转为 numpy)
    cov_np = cov_matrix.cpu().numpy()# [2800, 2800]
    
    # 提取对角线元素 (即每个像素的方差)，并重塑为物理场维度
    variance_1d = torch.diag(cov_matrix)
    variance_2d = variance_1d.view(50, 56).cpu().numpy() # [50, 56]
    
    # 2. 创建画布 (2x2 布局)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f"Diffusion Uncertainty Analysis at Timestep t={target_t}", fontsize=18, y=0.95)
    
    # --- 图 1: 真实 CSI ---
    im1 = axes[0, 0].imshow(real, cmap='viridis', aspect='auto')
    axes[0, 0].set_title("Ground Truth CSI", fontsize=14)
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # --- 图 2: 最终生成的 CSI ---
    im2 = axes[0, 1].imshow(gen, cmap='viridis', aspect='auto')
    axes[0, 1].set_title("Final Generated CSI", fontsize=14)
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # --- 图 3: 2800x2800 完整协方差矩阵 ---
    # 使用 RdBu_r 发散型色图，0 为白色，红色为正相关，蓝色为负相关
    # 为了防止极值导致颜色失真，取 1% 和 99% 分位数作为 vmin 和 vmax
    vmax_cov = np.percentile(np.abs(cov_np), 99) 
    im3 = axes[1, 0].imshow(cov_np, cmap='RdBu_r', aspect='auto', vmin=-vmax_cov, vmax=vmax_cov)
    axes[1, 0].set_title(f"Full Covariance Matrix (2800 x 2800)\nCaptures Element Correlations", fontsize=14)
    axes[1, 0].set_xlabel("Flattened CSI Index")
    axes[1, 0].set_ylabel("Flattened CSI Index")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # --- 图 4: 50x56 像素级方差(不确定度)热力图 ---
    # 方差严格为正，使用 hot 或 magma 色图
    im4 = axes[1, 1].imshow(variance_2d, cmap='magma', aspect='auto')
    axes[1, 1].set_title(f"Pixel-wise Variance (Uncertainty) at t={target_t}\nDiagonal of Covariance", fontsize=14)
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[*] 协方差分析可视化已保存: {save_path}")


def main():
    # ---------- 配置 ----------
    DATASET_ROOT = "/home/gshang/.AAAHAR/HumanActivityRecognition"
    TASK_NAME = "HumanActivityRecognition"
    BATCH_SIZE = 1  # 计算 Jacobian 必须严格等于 1
    MODEL_PATH = "/home/gshang/.AAAHAR/Diffusion/best_diffusion_LinearT.pth"
    SAVE_DIR = "/home/gshang/.AAAHAR/Diffusion/AutogradCov_results"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 设定你想观察的特定时间步 (例如: t=50 表示处于中等噪声水平时的不确定度)

    
    print(f"[*] 运行设备: {DEVICE}")
    print(f"[*] 目标观测时间步: t={TARGET}")





    # ---------- 加载测试数据 ----------
    print("[*] 加载数据集...")
    data_info = load_benchmark_supervised(
        dataset_root=DATASET_ROOT,
        task_name=TASK_NAME,
        batch_size=BATCH_SIZE,
        train_split="train_id",
        val_split="val_id",
        test_splits=["test_id"],
        num_workers=2,
        use_root_as_task_dir=False
    )
    train_loader = data_info['loaders']['train']
    test_loader = data_info['loaders']['test']



    print("[*] 加载训练集以初始化D...")

    # ---------- 构建掩码 ----------
    mask = torch.zeros((1, 1, 50, 56), dtype=torch.float32).to(DEVICE)
    mask[0, 0, 0, 0:55:10] = 1.0
    mask[0, 0, 1, 5:56:10] = 1.0

    # ---------- 加载模型 ----------
    print("[*] 加载预训练扩散模型...")
    unet = ConditionalUNet(in_channels=1, out_channels=1).to(DEVICE)
    unet.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    diffusion = CSIDiffusion(unet, timesteps=500).to(DEVICE)
    diffusion.eval()




    # ---------- 采样并提取协方差 ----------
    num_samples = [20,41,99] # 只测试前几个样本，因为求 Jacobian 比较耗时
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            if idx not in num_samples:
                continue
            
            print(f"\n[->] 正在处理样本 {idx+1}/{num_samples} (Label: {labels.item()})")
            print(f"     当执行到 t={TARGET} 时可能会卡顿数秒以计算 2800x2800 Jacobian，请稍候...")
            
            clean_csi = inputs.to(DEVICE).float()
            cond = clean_csi * mask





            with torch.no_grad():
                _ = diffusion.sample(mask, cond)
            torch.cuda.synchronize()  # 阻塞 CPU，直到 GPU 把预热的活干完
            # 直接调用 diffusion 的 sample 方法生成完整 CSI
            T1 = time.time()
            with torch.no_grad():
                (gen_csi, cov_matrix) = diffusion.sample_with_covariance(
                mask=mask, 
                cond=cond, 
                target_cov_t=TARGET,
                return_all=False
                )
            #torch.cuda.synchronize()
            T2 = time.time()
            print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
            #input("按回车键继续...")


            # 检查是否成功提取
            if cov_matrix is None:
                print("❌ 提取协方差失败！请检查 target_cov_t 是否超出了 timesteps 范围。")
                continue

            # 转为 numpy 并画图
            real_np = clean_csi.cpu().numpy()
            gen_np = gen_csi.cpu().numpy()
            
            save_path = os.path.join(SAVE_DIR, f"cov_analysis_sample_{idx+1}_t{TARGET}.png")
            
            # 调用画图函数
            plot_covariance_analysis(
                real_csi=real_np, 
                gen_csi=gen_np, 
                cov_matrix=cov_matrix, 
                target_t=TARGET, 
                save_path=save_path
            )
    
    print(f"\n🎉 协方差分析完成！结果保存在 {SAVE_DIR}")

if __name__ == "__main__":
    main()