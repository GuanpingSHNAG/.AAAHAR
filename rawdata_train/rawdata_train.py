import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 如果提示找不到 load 和 model 模块，请取消下方注释并将路径指向 CSI-Bench 的根目录
# sys.path.append("/path/to/your/CSI-Bench-Real-WiFi-Sensing-Benchmark")

from load.supervised.benchmark_loader import load_benchmark_supervised
from model.supervised.models import MLPClassifier

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    """绘制并保存 Loss 和 Accuracy 曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'learning_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"[*] 学习曲线已保存至: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_dir):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix on Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300)
    print(f"[*] 混淆矩阵已保存至: {save_path}")
    plt.close()

def main():
    # ================= 1. 配置参数 =================
    DATASET_ROOT = "/home/gshang/.AAAHAR/HumanActivityRecognition"
    TASK_NAME = "HumanActivityRecognition"
    BATCH_SIZE = 32
    EPOCHS = 30           # 根据收敛情况可以调整
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    SAVE_DIR = "/home/gshang/.AAAHAR/rawdata_train"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 运行设备: {device}")

    # ================= 2. 加载数据集 =================
    print("[*] 正在加载数据集 (同分布 ID 划分)...")
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
    num_classes = data_info['num_classes']
    # 通过字典按索引顺序提取标签名称
    mapper = data_info['label_mapper']
    label_names = [mapper.idx_to_label[i] for i in range(mapper.num_classes)]
    
    # 注意：底层 loader 代码中 test_id 的 key 默认被映射为了 'test'
    test_loader_key = 'test' if 'test' in loaders else 'test_test_id'

    # ================= 3. 初始化模型 =================
    # 输入形状为 [Batch, 1, 200, 56]
    model = MLPClassifier(win_len=1, feature_size=28, num_classes=num_classes) #in_channels=1)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 记录历史数据用于画图
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    # ================= 4. 训练与验证循环 =================
    print(f"[*] 开始训练 Baseline 模型，总 Epoch 数: {EPOCHS}")
    for epoch in range(EPOCHS):
        # ---------- 训练阶段 ----------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        train_pbar = tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100. * train_correct / train_total
        scheduler.step()

        # ---------- 验证阶段 ----------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        val_pbar = tqdm(loaders['val'], desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total
        
        # 记录历史
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
        
        # 保存最优模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_baseline_model.pth'))

    # ================= 5. 绘制并保存学习曲线 =================
    plot_learning_curves(history['train_loss'], history['val_loss'], 
                         history['train_acc'], history['val_acc'], SAVE_DIR)

    # ================= 6. 在测试集上评估 (Test Set) =================
    print("\n[*] 正在加载最优模型在 Test 集上进行最终评估...")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_baseline_model.pth')))
    model.eval()
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loaders[test_loader_key], desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
    # 计算准确率和 F1 分数
    test_acc = accuracy_score(all_targets, all_predictions) * 100
    test_f1 = f1_score(all_targets, all_predictions, average='macro') * 100
    
    print("\n" + "="*40)
    print(f"🚀 【Baseline 测试结果】")
    print(f"   Overall Accuracy : {test_acc:.2f}%")
    print(f"   Macro F1-Score   : {test_f1:.2f}%")
    print("="*40 + "\n")
    
    # 绘制并保存混淆矩阵
    plot_confusion_matrix(all_targets, all_predictions, label_names, SAVE_DIR)

if __name__ == '__main__':
    main()