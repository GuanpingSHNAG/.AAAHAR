import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ================= 1. 基础配置 (默认字体，小字号，600 DPI 超高清) =================
config = {
    "font.size": 8,              # 基础字号调小
    "axes.titlesize": 9,         # 标题字号调小
    "axes.labelsize": 8,         # 坐标轴标签字号调小
    "xtick.labelsize": 7,        # 坐标轴刻度字号调小
    "ytick.labelsize": 7,        # 坐标轴刻度字号调小
    "figure.dpi": 600,           # 绘图分辨率提升至 600 DPI
    "savefig.dpi": 600,          # 保存分辨率提升至 600 DPI
}
plt.rcParams.update(config)

# ================= 2. 路径与环境配置 =================
PROJECT_ROOT = "/home/gshang/.AAAHAR"
sys.path.append(f"{PROJECT_ROOT}/rawdata_train")
sys.path.append(f"{PROJECT_ROOT}/Diffusion")
sys.path.append(f"{PROJECT_ROOT}/Csem")

# 导入你的 Allocator
from DGCPA import Allocator

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_DYNAMIC_PILOTS = 256
    GRID_SIZE = 32
    
    # 选择测试 SNR (如 20 dB) 
    TEST_SNR_DB = 20.0
    target_snr_linear = 10.0 ** (TEST_SNR_DB / 10.0)

    # 模型和数据路径DGCPA_best_model-SNR_pilots={NUM_DYNAMIC_PILOTS}.pth
    SAVE_DIR = f"{PROJECT_ROOT}/PowAllocate/checkpoints"
    MODEL_PATH = os.path.join(SAVE_DIR, f"DGCPA_best_model-SNR_pilots={NUM_DYNAMIC_PILOTS}.pth")
    CACHE_DIR = f"{PROJECT_ROOT}/PowAllocate/cached_features2"
    VAL_CACHE_PATH = os.path.join(CACHE_DIR, "val_features_snr.pt")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到预训练模型: {MODEL_PATH}")
    if not os.path.exists(VAL_CACHE_PATH):
        raise FileNotFoundError(f"未找到验证集缓存数据: {VAL_CACHE_PATH}")

    # ================= 3. 加载数据与模型 =================
    print(f"[*] 正在加载验证集缓存数据: {VAL_CACHE_PATH}")
    val_data = torch.load(VAL_CACHE_PATH, map_location='cpu')
    c_sem_all = val_data['c_sem']  
    total_samples = c_sem_all.shape[0]

    print(f"[*] 初始化并加载 Allocator 模型...")
    allocator = Allocator(N=GRID_SIZE * GRID_SIZE, num_dynamic_pilots=NUM_DYNAMIC_PILOTS).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    allocator.load_state_dict(checkpoint['allocator_state_dict'])
    allocator.eval()

    # ================= 4. 推理并统计频次 =================
    selection_counts = torch.zeros(GRID_SIZE * GRID_SIZE, dtype=torch.float32, device=DEVICE)
    BATCH_SIZE = 256

    print(f"[*] 开始对全部验证集 {total_samples} 个样本进行推理 (测试 SNR: {TEST_SNR_DB} dB)...")
    with torch.no_grad():
        for i in range(0, total_samples, BATCH_SIZE):
            c_sem_batch = c_sem_all[i : i + BATCH_SIZE].to(DEVICE)
            B = c_sem_batch.shape[0]
            
            snr_budget_batch = torch.full((B, 1), target_snr_linear, device=DEVICE)
            
            topk_indices, _ = allocator(c_sem_batch, snr_budget_batch)
            
            flat_indices = topk_indices.view(-1)
            counts = torch.bincount(flat_indices, minlength=GRID_SIZE * GRID_SIZE)
            selection_counts += counts.float()

    # 计算概率并 Reshape 成 32x32 的二维空间 (完全保持原样，不翻转)
    probability_map = (selection_counts / total_samples).cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)

    # ================= 5. 绘制高质量热力图 =================
    print("[*] 正在生成并保存科研级热力图...")
    fig, ax = plt.subplots(figsize=(5, 4)) 

    # 莫兰迪青绿色系渐变
    custom_colors = [
        '#E8F1EB', '#C8E2D6', '#A4D3C3', '#88C2B2', 
        '#6EADA8', '#5C979C', '#4A7F89'
    ]
    custom_cmap = LinearSegmentedColormap.from_list("morandi_teal", custom_colors)

    # 绘制热力图 (原汁原味的画法，没有任何强行绑定坐标系的参数)
    cax = ax.imshow(probability_map, cmap=custom_cmap, aspect='equal', interpolation='nearest')

    # 配置 Colorbar (统一修改标签和字号)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Selection Probability', rotation=270, labelpad=12, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False) 

    ax.set_xlabel('Subcarrier indices')
    ax.set_ylabel('Symbol indices')

    # 设置刻度 (每隔 4 个点显示一个刻度)
    ticks = np.arange(0, GRID_SIZE, 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # 隐藏右侧和上方多余的边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 导出文件 (统一改为 EPS 和 PNG)
    output_eps = os.path.join(SAVE_DIR, f"Heatmap_Pilots={NUM_DYNAMIC_PILOTS}.svg")
    output_png = os.path.join(SAVE_DIR, f"Heatmap_Pilots={NUM_DYNAMIC_PILOTS}.png")
    
    plt.savefig(output_eps, format='svg', bbox_inches='tight')
    plt.savefig(output_png, format='png', bbox_inches='tight', transparent=False)
    plt.close()

    print(f"[*] 🟢 绘图完成！图表已保存至：")
    print(f"    - {output_eps}")
    print(f"    - {output_png}")

if __name__ == "__main__":
    main()