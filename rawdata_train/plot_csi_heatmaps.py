import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 如果提示找不到 load 模块，请取消下方注释并将路径指向 CSI-Bench 的根目录
# sys.path.append("/path/to/your/CSI-Bench-Real-WiFi-Sensing-Benchmark")

from load.supervised.benchmark_loader import load_benchmark_supervised


def plot_csi_heatmap(data_matrix, label_name, sample_idx, save_dir, vmin=None, vmax=None):
    """
    绘制单个 CSI 样本的热力图（时间 × 子载波）
    data_matrix: 形状为 (time_steps, subcarriers) 的 numpy 数组
    label_name: 类别名称（字符串）
    sample_idx: 该类别下的样本序号
    save_dir: 保存路径
    vmin, vmax: 颜色映射范围，若不提供则自动计算
    """
    plt.figure(figsize=(10, 6))
    im = plt.imshow(data_matrix, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='CSI Amplitude')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Time Step')
    plt.title(f'Label: {label_name} | Sample #{sample_idx}')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{label_name}_sample_{sample_idx}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[*] 已保存: {save_path}")


def main():
    # ================= 配置参数 =================
    DATASET_ROOT = "/home/gshang/.AAAHAR/HumanActivityRecognition"   # 请根据实际路径修改
    TASK_NAME = "HumanActivityRecognition"
    BATCH_SIZE = 32               # 仅用于加载器，实际采样时会逐个获取样本
    SAMPLES_PER_CLASS = 3         # 每个类别随机可视化的样本数量
    SAVE_DIR = "./csi_heatmaps"   # 热力图保存目录

    # 可选：设置颜色映射的上下限（若为 None 则自动缩放）
    VMIN = None
    VMAX = None

    # ================= 加载数据集（仅用于获取数据） =================
    print("[*] 正在加载数据集...")
    data_info = load_benchmark_supervised(
        dataset_root=DATASET_ROOT,
        task_name=TASK_NAME,
        batch_size=BATCH_SIZE,
        train_split="train_id",          # 使用训练集划分（包含所有类别）
        val_split="val_id",
        test_splits=["test_id"],
        num_workers=4,
        use_root_as_task_dir=False
    )

    loaders = data_info['loaders']
    num_classes = data_info['num_classes']
    mapper = data_info['label_mapper']
    label_names = [mapper.idx_to_label[i] for i in range(mapper.num_classes)]

    print(f"[*] 数据集加载完成，共 {num_classes} 个类别:")
    for i, name in enumerate(label_names):
        print(f"    {i}: {name}")

    # 使用训练集 loader（或验证集，通常训练集包含所有类别且样本最多）
    train_loader = loaders['train']

    # ================= 为每个类别收集样本 =================
    print("\n[*] 开始为每个类别收集样本...")

    # 存储每个类别的样本列表：dict[label_idx] -> list of (csi_matrix, sample_index)
    class_samples = {idx: [] for idx in range(num_classes)}

    # 遍历训练集，收集 CSI 矩阵及其标签
    # 注意：每个 batch 的形状通常为 (B, C, T, F)，这里我们假设 C=1 或 C=2（需要取幅度）
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc="Collecting samples")):
            # inputs: (B, C, T, F)  例如 (32, 1, 50, 56) 或 (32, 2, 50, 56)
            B = inputs.shape[0]
            for i in range(B):
                label = labels[i].item()
                sample = inputs[i]  # shape: (C, T, F)

                # 将 CSI 转换为 2D 矩阵 (T, F)
                if sample.shape[0] == 1:
                    # 单通道（幅度或实部）
                    csi_matrix = sample[0].cpu().numpy()   # (T, F)
                elif sample.shape[0] == 2:
                    # I/Q 两通道，计算幅度
                    iq = sample.cpu().numpy()              # (2, T, F)
                    csi_matrix = np.sqrt(iq[0]**2 + iq[1]**2)
                else:
                    raise ValueError(f"不支持的通道数: {sample.shape[0]}，期望 1 或 2")

                # 检查形状是否符合预期 (T, F) = (50, 56)
                if csi_matrix.shape != (50, 56):
                    print(f"警告: 样本形状为 {csi_matrix.shape}，而非 (50,56)。将继续执行，但可能需要调整可视化尺寸。")

                class_samples[label].append(csi_matrix)

                # 如果每个类别的样本已经收集足够，可提前停止（但需遍历所有类别）
                # 这里不提前 break，而是收集全部，最后随机选择

    # ================= 随机采样并可视化 =================
    print("\n[*] 开始随机采样并绘制热力图...")
    for label_idx, samples in class_samples.items():
        if len(samples) == 0:
            print(f"警告: 类别 {label_names[label_idx]} (索引 {label_idx}) 没有样本，跳过。")
            continue

        # 随机选择 SAMPLES_PER_CLASS 个样本（如果不足则取全部）
        num_to_sample = min(SAMPLES_PER_CLASS, len(samples))
        selected_indices = np.random.choice(len(samples), num_to_sample, replace=False)

        for sel_idx, sample_idx in enumerate(selected_indices):
            csi_matrix = samples[sample_idx]
            plot_csi_heatmap(
                data_matrix=csi_matrix,
                label_name=label_names[label_idx],
                sample_idx=sel_idx + 1,   # 序号从 1 开始
                save_dir=SAVE_DIR,
                vmin=VMIN,
                vmax=VMAX
            )

    print("\n[*] 所有热力图生成完毕！")


if __name__ == '__main__':
    main()