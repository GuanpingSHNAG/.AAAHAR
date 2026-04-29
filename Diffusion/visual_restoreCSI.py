# test_visualize.py
import time
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("/home/gshang/.AAAHAR/rawdata_train")
from load.supervised.benchmark_loader import load_benchmark_supervised
from model import ConditionalUNet, CSIDiffusion   # 注意 CSIDiffusion 现在有 sample 方法
def visualize_reconstruction(real_csi, gen_csi, mask, cond, save_path, vmin=None, vmax=None):
    """
    绘制热力图对比真实与生成的 CSI。
    
    Args:
        real_csi: 真实完整 CSI [1, H, W] (numpy)
        gen_csi:  生成完整 CSI [1, H, W] (numpy)
        mask:     导频掩码 [1, H, W] (numpy)
        cond:     条件输入 (导频值) [1, H, W] (numpy)
        save_path: 保存路径
        vmin, vmax: 热力图颜色范围
    """
    H, W = real_csi.shape[-2:]
    
    # 去掉 channel 维度
    real = real_csi.squeeze()
    gen = gen_csi.squeeze()
    msk = mask.squeeze()
    cnd = cond.squeeze()
    
    # 误差图
    error = np.abs(real - gen)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 真实 CSI
    im1 = axes[0,0].imshow(real, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0,0].set_title("Ground Truth CSI")
    axes[0,0].axis('off')
    plt.colorbar(im1, ax=axes[0,0], fraction=0.046)
    
    # 生成 CSI
    im2 = axes[0,1].imshow(gen, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0,1].set_title("Generated CSI (Diffusion)")
    axes[0,1].axis('off')
    plt.colorbar(im2, ax=axes[0,1], fraction=0.046)
    
    # 误差热力图
    im3 = axes[0,2].imshow(error, cmap='hot', aspect='auto')
    axes[0,2].set_title("Absolute Error")
    axes[0,2].axis('off')
    plt.colorbar(im3, ax=axes[0,2], fraction=0.046)
    
    # 导频掩码 (白色为导频位置)
    axes[1,0].imshow(msk, cmap='gray', aspect='auto')
    axes[1,0].set_title("Pilot Mask (White = Pilot)")
    axes[1,0].axis('off')
    
    # 条件输入 (导频值)
    im4 = axes[1,1].imshow(cnd, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1,1].set_title("Condition (Pilot Values)")
    axes[1,1].axis('off')
    plt.colorbar(im4, ax=axes[1,1], fraction=0.046)
    
    # 散点图：真实 vs 生成 (在导频位置)
    pilot_indices = msk = 0
    real_pilot = real[pilot_indices]
    gen_pilot = gen[pilot_indices]
    axes[1,2].scatter(real_pilot, gen_pilot, s=10, alpha=0.6, c='blue')
    axes[1,2].plot([real_pilot.min(), real_pilot.max()], 
                   [real_pilot.min(), real_pilot.max()], 'r--', label="Ideal")
    axes[1,2].set_xlabel("Real Pilot Values")
    axes[1,2].set_ylabel("Generated Pilot Values")
    axes[1,2].set_title("Pilot Reconstruction")
    axes[1,2].legend()
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[*] 可视化结果已保存: {save_path}")


def main():
    # ---------- 配置 ----------
    DATASET_ROOT = "/home/gshang/.AAAHAR/HumanActivityRecognition"
    TASK_NAME = "HumanActivityRecognition"
    BATCH_SIZE = 1
    MODEL_PATH = "/home/gshang/.AAAHAR/Diffusion/best_diffusion_unet.pth"
    SAVE_DIR = "/home/gshang/.AAAHAR/Diffusion/test_results"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[*] 运行设备: {DEVICE}")

    # ---------- 加载测试数据 ----------
    print("[*] 加载测试集...")
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
    test_loader = data_info['loaders']['test']

    # ---------- 构建掩码 ----------
    mask = torch.zeros((1, 1, 50, 56), dtype=torch.float32).to(DEVICE)
    mask[0, 0, 0, 0:55:10] = 1.0
    mask[0, 0, 1, 5:56:10] = 1.0
    print(f"[*] 导频数量: {int(mask.sum().item())} / 2800")

    # ---------- 加载模型 ----------
    print("[*] 加载预训练扩散模型...")
    unet = ConditionalUNet(in_channels=1, out_channels=1).to(DEVICE)
    unet.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    diffusion = CSIDiffusion(unet, timesteps=20).to(DEVICE)
    diffusion.eval()

    # ---------- 采样并可视化前 N 个样本 ----------
    num_samples = [20, 41,99 ] 
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            if idx not in num_samples:
                continue

            clean_csi = inputs.to(DEVICE).float()
            cond = clean_csi *mask

            with torch.no_grad():
                _ = diffusion.sample(mask, cond)
            torch.cuda.synchronize()  # 阻塞 CPU，直到 GPU 把预热的活干完
            # 直接调用 diffusion 的 sample 方法生成完整 CSI
            T1 = time.time()
            with torch.no_grad():
                gen_csi = diffusion.sample(mask, cond)
            torch.cuda.synchronize()
            T2 = time.time()
            print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))


            #input("按回车键继续...")

            # 转 numpy 并可视化
            real_np = clean_csi.cpu().numpy()
            gen_np = gen_csi.cpu().numpy()
            mask_np = mask.cpu().numpy()
            cond_np = cond.cpu().numpy()
            
            vmin = min(real_np.min(), gen_np.min())
            vmax = max(real_np.max(), gen_np.max())
            
            save_path = os.path.join(SAVE_DIR, f"sample_{idx+1}_label_{labels.item()}.png")
            visualize_reconstruction(real_np, gen_np, mask_np, cond_np, save_path, vmin, vmax)
    
    print(f"\n🎉 完成！结果保存在 {SAVE_DIR}")

if __name__ == "__main__":
    main()