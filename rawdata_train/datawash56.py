import os
import pandas as pd
import h5py
from tqdm import tqdm

def main():
    DATASET_ROOT = "/home/gshang/.AAAHAR/HumanActivityRecognition"
    METADATA_PATH = os.path.join(DATASET_ROOT, "metadata", "sample_metadata.csv")
    BACKUP_PATH = os.path.join(DATASET_ROOT, "metadata", "sample_metadata_backup.csv")
    
    if not os.path.exists(METADATA_PATH):
        print("找不到 CSV 文件！")
        return

    df = pd.read_csv(METADATA_PATH)
    
    # 如果还没备份过，先备份一下最原始的表
    if not os.path.exists(BACKUP_PATH):
        df.to_csv(BACKUP_PATH, index=False)
        print(f"[*] 已备份原始 CSV 到: {BACKUP_PATH}")

    valid_indices = []
    
    print("[*] 正在扫描并过滤 232 维度的数据，只保留 56 维度...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        rel_path = str(row['file_path'])
        if rel_path.startswith('./'): rel_path = rel_path[2:]
        elif rel_path.startswith('E:/') or 'CSI100Hz' in rel_path:
            rel_path = rel_path.split('HumanActivityRecognition/')[-1]
            if rel_path.startswith('./'): rel_path = rel_path[2:]
            
        abs_path = os.path.join(DATASET_ROOT, rel_path)
        
        try:
            with h5py.File(abs_path, 'r') as f:
                # 获取真实形状
                if 'CSI_amps' in f: shape = f['CSI_amps'].shape
                elif 'csi' in f: shape = f['csi'].shape
                elif 'CSI' in f: shape = f['CSI'].shape
                else: shape = f[list(f.keys())[0]].shape
                
                # 【核心过滤逻辑】：子载波维度等于 56 才保留
                if shape[1] == 56 or shape[0] == 56: # 兼容不同的通道顺序
                    valid_indices.append(idx)
        except Exception:
            pass

    # 生成只包含 56 维度的纯净表
    pure_df = df.loc[valid_indices].reset_index(drop=True)
    pure_df.to_csv(METADATA_PATH, index=False)
    
    print("\n" + "="*40)
    print("🎯 清洗完成！")
    print(f"原始数据量: {len(df)}")
    print(f"保留数据量 (仅56维度): {len(pure_df)}")
    print(f"剔除数据量 (232维度及错误): {len(df) - len(pure_df)}")
    print("="*40)

if __name__ == '__main__':
    main()