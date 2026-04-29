# /home/gshang/.AAAHAR/Diffusion/train_diffusion.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# 将路径指向您的工作目录，以便加载官方 loader
sys.path.append("/home/gshang/.AAAHAR/rawdata_train")
from load.supervised.benchmark_loader import load_benchmark_supervised

# 导入我们在本目录下写好的 Diffusion 模型
from model_linearT import ConditionalUNet, CSIDiffusion

def plot_loss_curves(train_losses, val_losses, save_dir):
    """绘制并保存 Diffusion 模型的 MSE Loss 曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'b-', label='Train MSE Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation MSE Loss')
    plt.title('Diffusion Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'diffusion_learning_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"[*] 学习曲线已保存至: {save_path}")
    plt.close()

def main():
    # ================= 1. 配置参数 =================
    DATASET_ROOT = "/home/gshang/.AAAHAR/HumanActivityRecognition"
    TASK_NAME = "HumanActivityRecognition"
    BATCH_SIZE = 32
    EPOCHS = 50           # Diffusion 收敛较慢，建议设为 50 或更大
    LEARNING_RATE = 1e-4  # Diffusion 推荐使用 1e-4
    WEIGHT_DECAY = 1e-4
    SAVE_DIR = "/home/gshang/.AAAHAR/Diffusion"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 运行设备: {device}")

    # ================= 2. 加载数据集 =================
    print("[*] 正在调用官方懒加载机制读取数据...")
    # 严格使用您提供的 loader 配置
    data_info = load_benchmark_supervised(
        dataset_root=DATASET_ROOT,
        task_name=TASK_NAME,
        batch_size=BATCH_SIZE,
        train_split="train_id",
        val_split="val_id",
        test_splits=["test_id"],
        num_workers=4,
        use_root_as_task_dir=False
    )
    
    loaders = data_info['loaders']
    test_loader_key = 'test' if 'test' in loaders else 'test_test_id'

    # ================= 3. 构造极稀疏导频 Mask (50x56) =================
    # 根据您的设定，我们在时频图上取极少数导频点
    mask = torch.zeros((1, 1, 50, 56), dtype=torch.float32).to(device)
    mask[0, 0, 0, 0:55:10] = 1.0
    mask[0, 0, 1, 5:56:10] = 1.0
    print(f"[*] Mask 生成完毕，单样本导频数量: {int(mask.sum().item())} / 2800")

    # ================= 4. 初始化模型 =================
    unet = ConditionalUNet(in_channels=1, out_channels=1).to(device)
    diffusion = CSIDiffusion(unet, timesteps=500).to(device)
    
    optimizer = optim.AdamW(diffusion.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 记录历史数据用于画图
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf') # 保存 Val Loss 最低的模型

    # ================= 5. 训练与验证循环 =================
    print(f"[*] 开始训练 Diffusion 语义提取器，总 Epoch 数: {EPOCHS}")
    for epoch in range(EPOCHS):
        
        # ---------- 训练阶段 ----------
        diffusion.train()
        train_loss, train_total = 0.0, 0
        
        train_pbar = tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for inputs, labels in train_pbar:
            # inputs 形状为 [Batch, 1, 50, 56]
            # labels 是动作分类标签，Diffusion 重构阶段不需要它，直接丢弃
            clean_csi = inputs.to(device).float()
            batch_size = clean_csi.size(0)
            
            # 动态生成 Condition 变量
            current_mask = mask.expand(batch_size, -1, -1, -1)
            cond = clean_csi * current_mask  # 完美源数据乘以导频掩码
            
            optimizer.zero_grad()
            # 前向计算预测噪声的 MSE Loss
            loss = diffusion(clean_csi, current_mask, cond)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_size
            train_total += batch_size
            
        epoch_train_loss = train_loss / train_total
        scheduler.step()

        # ---------- 验证阶段 ----------
        diffusion.eval()
        val_loss, val_total = 0.0, 0
        
        val_pbar = tqdm(loaders['val'], desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for inputs, _ in val_pbar:
                clean_csi = inputs.to(device).float()
                batch_size = clean_csi.size(0)
                
                current_mask = mask.expand(batch_size, -1, -1, -1)
                cond = clean_csi * current_mask
                
                loss = diffusion(clean_csi, current_mask, cond)
                val_loss += loss.item() * batch_size
                val_total += batch_size
                
        epoch_val_loss = val_loss / val_total
        
        # 记录历史
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train MSE Loss: {epoch_train_loss:.5f} | Val MSE Loss: {epoch_val_loss:.5f}")
        
        # 保存最优模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(unet.state_dict(), os.path.join(SAVE_DIR, 'best_diffusion_LinearT.pth'))

    # ================= 6. 绘制并保存学习曲线 =================
    plot_loss_curves(history['train_loss'], history['val_loss'], SAVE_DIR)
    print(f"\n🚀 【训练完成】最优 Validation Loss: {best_val_loss:.5f}")
    print(f"💾 最优模型已保存为: {os.path.join(SAVE_DIR, 'best_diffusion_LinearT.pth')}")

if __name__ == '__main__':
    main()