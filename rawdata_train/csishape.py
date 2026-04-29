import os
import pandas as pd
import h5py
from collections import Counter
from tqdm import tqdm

def main():
    # 1. 配置路径
    DATASET_ROOT = "/home/gshang/.AAAHAR/HumanActivityRecognition"
    METADATA_PATH = os.path.join(DATASET_ROOT, "metadata", "sample_metadata.csv")
    
    print(f"[*] 正在读取元数据表: {METADATA_PATH}")
    if not os.path.exists(METADATA_PATH):
        print("错误：找不到 sample_metadata.csv！请检查路径。")
        return
        
    df = pd.read_csv(METADATA_PATH)
    total_files = len(df)
    print(f"[*] 共发现 {total_files} 个样本记录。开始探查底层 .h5 文件的真实维度...")
    
    # 用于统计不同维度出现的次数
    shapes_counter = Counter()
    error_count = 0
    
    # 2. 遍历所有文件并读取维度
    for idx, row in tqdm(df.iterrows(), total=total_files, desc="Checking H5 files"):
        # 处理路径拼接 (去除 CSV 中可能带有的 './' 前缀)
        rel_path = str(row['file_path'])
        if rel_path.startswith('./'):
            rel_path = rel_path[2:]
        elif rel_path.startswith('E:/') or 'CSI100Hz' in rel_path:
            # 兼容官方 CSV 中可能残留的绝对路径或异常路径
            rel_path = rel_path.split('HumanActivityRecognition/')[-1]
            if rel_path.startswith('./'): rel_path = rel_path[2:]
            
        abs_path = os.path.join(DATASET_ROOT, rel_path)
        
        # 打开 h5 文件查探形状
        try:
            with h5py.File(abs_path, 'r') as f:
                # 官方代码中通常使用的是 'CSI_amps' 或 'csi'
                if 'CSI_amps' in f:
                    shape = f['CSI_amps'].shape
                    shapes_counter[shape] += 1
                elif 'csi' in f:
                    shape = f['csi'].shape
                    shapes_counter[shape] += 1
                elif 'CSI' in f:
                    shape = f['CSI'].shape
                    shapes_counter[shape] += 1
                else:
                    # 如果键名不常见，读取第一个数据集的形状
                    keys = list(f.keys())
                    if keys:
                        shape = f[keys[0]].shape
                        shapes_counter[shape] += 1
        except Exception as e:
            error_count += 1
            pass # 忽略损坏或找不到的文件，继续统计

    # 3. 打印汇总报告
    print("\n" + "="*50)
    print("📊 CSI 原始数据格式 (几乘几) 汇总报告")
    print("="*50)
    print(f"成功读取: {total_files - error_count} 个文件")
    if error_count > 0:
        print(f"读取失败: {error_count} 个文件 (可能路径错误或文件损坏)")
    
    print("\n发现的原始矩阵维度 (时间点数, 子载波特征数, 通道数):")
    # 按出现次数从高到低排序打印
    for shape, count in shapes_counter.most_common():
        percentage = (count / (total_files - error_count)) * 100
        print(f" -> 格式 {shape}: \t共 {count} 个文件 ({percentage:.2f}%)")
        
    print("\n[*] 提示：不管这里原始格式有多少种，")
    print("[*] 经过 benchmark_dataset.py 处理后，")
    print("[*] 最终输入给你模型的格式都会被统一重塑(补零/裁剪)为: [1, 500, 232]")
    print("="*50)

if __name__ == '__main__':
    main()