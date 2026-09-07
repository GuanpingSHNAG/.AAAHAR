import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# 添加你的项目路径
sys.path.append("/home/gshang/.AAAHAR/rawdata_train")
sys.path.append("/home/gshang/.AAAHAR/Diffusion")
sys.path.append("/home/gshang/.AAAHAR/Csem")

from load.supervised.benchmark_loader import load_benchmark_supervised
from semantic_engine_FH import SemanticEngine

def plot_csem_results(clean_csi, gen_csi, c_sem_diag, label, idx, save_dir):
    """
    绘制单样本的三联图：真实 CSI、补全 CSI、C_sem 对角线(语义重要性分布)
    """
    # 转换为 numpy 格式并剥离冗余维度 [32, 32]
    real_np = clean_csi.cpu().squeeze().numpy()
    gen_np = gen_csi.cpu().squeeze().numpy()
    csem_np = c_sem_diag.cpu().squeeze().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Semantic Engine Check - Sample Index: {idx} | GT Label: {label}", fontsize=16)
    
    # 图1：真实 CSI
    im1 = axes[0].imshow(real_np, cmap='viridis', aspect='auto')
    axes[0].set_title("1. Ground Truth CSI", fontsize=14)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # 图2：Diffusion 生成 CSI
    im2 = axes[1].imshow(gen_np, cmap='viridis', aspect='auto')
    axes[1].set_title("2. Diffusion Generated CSI ($x_0$)", fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # 图3：C_sem 对角线元素 (重要性 Map)
    # 因为 C_sem 对角线严格为正，使用热力图 (magma/hot) 展示
    im3 = axes[2].imshow(csem_np, cmap='magma', aspect='auto')
    axes[2].set_title("3. Semantic Importance Map ($C_{sem}$ Diagonal)", fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"check_Csem_sample_{idx}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[*] 已生成可视化图像: {save_path}")

def main():
    # ================= 1. 环境与参数配置 =================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET_ROOT = "/home/gshang/.AAAHAR/HumanActivityRecognition"
    TASK_NAME = "HumanActivityRecognition"
    SAVE_DIR = "/home/gshang/.AAAHAR/Csem/visual_checks"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"[*] 运行设备: {DEVICE}")
    
    # ================= 2. 加载数据集 =================
    print("[*] 正在加载测试数据集...")
    data_info = load_benchmark_supervised(
        dataset_root=DATASET_ROOT,
        task_name=TASK_NAME,
        batch_size=1, 
        train_split="train_id",
        val_split="val_id",
        test_splits=["test_id"],
        num_workers=2,
        use_root_as_task_dir=False
    )
    
    test_dataset = data_info['loaders']['test'].dataset
    dataset_size = len(test_dataset)
    print(f"[*] 测试集总数: {dataset_size}")
    
    # 随机抽取 5 个样本索引进行测试
    test_indices = random.sample(range(dataset_size), 5)
    print(f"[*] 随机选取的测试样本索引: {test_indices}")
    
    # ================= 3. 初始化 SemanticEngine =================
    # 里面自带了 Diffusion 和 MLP
    sem_engine = SemanticEngine(device=DEVICE, num_classes=5)
    
    # 构建固定导频 Mask [1, 1, 32, 32]
    mask = torch.zeros((1, 1, 32, 32), dtype=torch.float32).to(DEVICE)
    mask[0, 0, 0, 0:31:5] = 1.0
    mask[0, 0, 1, 3:32:5] = 1.0

    # ================= 4. 测试与可视化 =================
    for idx in test_indices:
        # 直接通过索引取出数据[1, 32, 32] -> [1, 1, 32, 32]
        clean_csi, label = test_dataset[idx]
        clean_csi = clean_csi.unsqueeze(0).to(DEVICE).float()
        
        # 施加导频掩码
        cond_csi = clean_csi * mask
        
        # 调用我们的语义引擎！
        print(f"\n[->] 正在处理样本 {idx} ...")
        with torch.no_grad():
            # 虽然这里开启 no_grad，但是 semantic_engine 内部对 gen_csi_req_grad 
            # 开启了局部的梯度追踪，这不冲突。
            C_sem_batch, gen_csi_batch = sem_engine(cond_csi, mask)
        
        # 取出单个样本的 C_sem 矩阵[2800, 2800]
        C_sem_matrix = C_sem_batch[0]
        
        # 提取对角线，这代表了每个像素绝对的 "语义重要性"
        c_sem_diag_1d = torch.diag(C_sem_matrix)
        
        # 展平还原回 32x32 的物理维度
        c_sem_diag_2d = c_sem_diag_1d.view(32, 32)
        
        # 绘制三联图并保存
        plot_csem_results(
            clean_csi=clean_csi,
            gen_csi=gen_csi_batch[0],
            c_sem_diag=c_sem_diag_2d,
            label=label,
            idx=idx,
            save_dir=SAVE_DIR
        )
        
    print(f"\n🎉 测试完成！去查看你的 C_sem 魔法可视化吧: {SAVE_DIR}")

if __name__ == "__main__":
    main()