import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random  # [新增] 用于随机选择导频数
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

PROJECT_ROOT = "/home/gshang/.AAAHAR"
sys.path.append(f"{PROJECT_ROOT}/rawdata_train")
sys.path.append(f"{PROJECT_ROOT}/Diffusion")
sys.path.append(f"{PROJECT_ROOT}/Csem")

from load.supervised.benchmark_loader import load_benchmark_supervised
from model.supervised.models import MLPClassifier
from semantic_engine_FH import SemanticEngine
from DGCPA import Allocator

def get_fixed_feedback_mask(B, device):
    mask = torch.zeros((B, 1, 32, 32), dtype=torch.float32, device=device)
    mask[:, 0, 0, 0:32:5] = 1.0
    mask[:, 0, 1, 3:32:5] = 1.0
    return mask

def get_cached_dataloader(dataloader, sem_engine, device, cache_path, desc="Caching", is_train=True):
    if os.path.exists(cache_path):
        print(f"[*] 🟢 找到本地缓存文件，直接加载: {cache_path}")
        cached_data = torch.load(cache_path)
        return DataLoader(TensorDataset(cached_data['inputs'], cached_data['c_sem'], cached_data['labels']), batch_size=dataloader.batch_size, shuffle=is_train)
        
    print(f"[*] 🟡 未找到缓存，开始在 GPU 上提取语义特征: {desc}")
    sem_engine.eval()
    cached_inputs, cached_c_sem, cached_labels = [], [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=desc):
            inputs = inputs.to(device).float()
            B = inputs.shape[0]
            fixed_mask = get_fixed_feedback_mask(B, device)
            cond_csi = inputs * fixed_mask
            c_sem_batch, _ = sem_engine(cond_csi, fixed_mask)
            
            cached_inputs.append(inputs.cpu())
            cached_c_sem.append(c_sem_batch.cpu())
            cached_labels.append(labels.cpu())
            
    cached_inputs = torch.cat(cached_inputs, dim=0)
    cached_c_sem = torch.cat(cached_c_sem, dim=0)
    cached_labels = torch.cat(cached_labels, dim=0)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({'inputs': cached_inputs, 'c_sem': cached_c_sem, 'labels': cached_labels}, cache_path)
    return DataLoader(TensorDataset(cached_inputs, cached_c_sem, cached_labels), batch_size=dataloader.batch_size, shuffle=is_train)


def validate_cached(allocator, sparse_mlp, val_loader_cached, target_snr, device):
    """使用缓存的语义特征进行验证集评估（此时内部接口不变，allocator.K 已在外部被动态修改）"""
    allocator.eval()
    sparse_mlp.eval()
    val_loss = 0.0
    val_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, c_sem_batch, labels in val_loader_cached:
            inputs = inputs.to(device).float()
            c_sem_batch = c_sem_batch.to(device)
            labels = labels.to(device)
            B = inputs.shape[0]
            
            # 使用传入的固定信噪比
            snr_budget_batch = torch.full((B, 1), target_snr, device=device)
            fixed_mask = get_fixed_feedback_mask(B, device)
            
            # 1. GNN 功率与导频分配 (此时会自适应外部设定的 allocator.K)
            topk_indices, power_alloc = allocator(c_sem_batch, snr_budget_batch)
            # 2. 链路仿真
            y_received_sparse, total_mask = allocator.simulate_sensing_link(inputs, topk_indices, power_alloc, fixed_mask)
            # 3. 分类
            logits = sparse_mlp(y_received_sparse)
            
            loss = criterion(logits, labels)
            val_loss += loss.item() * B
            total_samples += B
            _, predicted = logits.max(1)
            val_correct += predicted.eq(labels).sum().item()
            
    return val_loss / total_samples, 100. * val_correct / total_samples


def main():
    torch.autograd.set_detect_anomaly(True)

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except:
        local_rank = 0
    DEVICE = torch.device(f"cuda:{local_rank}")

    DATASET_ROOT = f"{PROJECT_ROOT}/HumanActivityRecognition"
    TASK_NAME = "HumanActivityRecognition"
    SAVE_DIR = f"{PROJECT_ROOT}/PowAllocate/checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # ================= [修改点 1: 泛化性实验参数配置] =================
    BATCH_SIZE = 256
    EPOCHS = 200
    FIXED_SNR_DB = 20  # 固定信噪比为 20 dB
    TARGET_SNR_LINEAR = 10.0 ** (FIXED_SNR_DB / 10.0) # 线性值 100.0
    
    # 💥 【可手动调】训练时，允许算法随机动态切换的导频数子集
    TRAIN_K_POOL = [512] 
    
    # 🔍 验证时，需要强制全面检验的导频数全集（如同变SNR一样）
    VAL_K_POOL = [32, 64, 128, 256, 512, 1024]
    
    print(f"[*] 设备: {DEVICE} | 固定信噪比: {FIXED_SNR_DB}dB")
    print(f"[*] 训练导频候选池: {TRAIN_K_POOL} | 验证泛化池: {VAL_K_POOL}")
    # =================================================================

    print("[*] 正在加载训练与验证数据...")
    data_info = load_benchmark_supervised(
        dataset_root=DATASET_ROOT, task_name=TASK_NAME,
        batch_size=BATCH_SIZE, train_split="train_id", val_split="val_id",
        test_splits=["test_id"], num_workers=12, use_root_as_task_dir=False
    )
    raw_train_loader = data_info['loaders']['train']
    raw_val_loader = data_info['loaders']['val']

    print("[*] 初始化 Semantic Engine (Frozen)...")
    sem_engine = SemanticEngine(device=DEVICE, num_classes=data_info['num_classes'])
    
    print("[*] 初始化 GNN Power Allocator (Trainable)...")
    # 初始化时传递默认值即可，后续循环中会被动态覆盖
    allocator = Allocator(N=1024, num_dynamic_pilots=256).to(DEVICE)
    
    CACHE_DIR = f"{PROJECT_ROOT}/PowAllocate/cached_features2"
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_cache_path = os.path.join(CACHE_DIR, f"train_features_snr.pt")
    val_cache_path = os.path.join(CACHE_DIR, f"val_features_snr.pt")

    needs_caching = not (os.path.exists(train_cache_path) and os.path.exists(val_cache_path))
    if needs_caching:
        sem_engine = sem_engine.to(DEVICE).eval()

    train_cached_loader = get_cached_dataloader(raw_train_loader, sem_engine, DEVICE, cache_path=train_cache_path, desc="Cache Train", is_train=True)
    val_cached_loader = get_cached_dataloader(raw_val_loader, sem_engine, DEVICE, cache_path=val_cache_path, desc="Cache Val", is_train=False)

    if needs_caching:
        del sem_engine
        torch.cuda.empty_cache()

    print("[*] 初始化 Sparse Task MLP (Trainable)...")
    sparse_mlp = MLPClassifier(win_len=32, feature_size=32, num_classes=data_info['num_classes']).to(DEVICE)
    sparse_mlp.load_state_dict(torch.load("/home/gshang/.AAAHAR/Csem/稀疏随机MLP/pre.pth", map_location=DEVICE))

    optimizer = optim.AdamW([
        {'params': allocator.parameters(), 'lr': 1e-4}, 
        {'params': sparse_mlp.parameters(), 'lr': 1e-5}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6),
        torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1e-6/optimizer.defaults['lr'])
    ], milestones=[100])

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    train_accs, val_accs = [], []

    print("\n🚀 [START] GNN K-Generalization Joint Training Initiated!")
    for epoch in range(EPOCHS):
        # ---------- 训练阶段 ----------
        allocator.train()  
        sparse_mlp.train() 
        
        train_loss, train_correct, total_samples = 0.0, 0, 0
        pbar = tqdm(train_cached_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for inputs, c_sem_batch, labels in pbar:
            inputs = inputs.to(DEVICE).float()
            c_sem_batch = c_sem_batch.to(DEVICE)
            labels = labels.to(DEVICE)
            B = inputs.shape[0]
            
            # ================= [修改点 2: 训练中动态随机选择 K，固定 SNR] =================
            snr_budget_batch = torch.full((B, 1), TARGET_SNR_LINEAR, device=DEVICE) # 固定 20dB
            fixed_mask = get_fixed_feedback_mask(B, DEVICE)
            
            # 从允许的候选池中随机抽取当前 batch 的导频数，动态注入给 GNN Allocator
            chosen_K = random.choice(TRAIN_K_POOL)
            allocator.K = chosen_K  
            # ==========================================================================
            
            optimizer.zero_grad()
            
            topk_indices, power_alloc = allocator(c_sem_batch, snr_budget_batch)
            y_received_sparse, total_mask = allocator.simulate_sensing_link(inputs, topk_indices, power_alloc, fixed_mask)
            logits = sparse_mlp(y_received_sparse)
            
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(allocator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(sparse_mlp.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * B
            total_samples += B
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'K_size': chosen_K, 'Acc': f"{100.*train_correct/total_samples:.1f}%"})
        
        epoch_train_acc = 100. * train_correct / total_samples
        epoch_train_loss = train_loss / total_samples

        # ---------- 验证阶段 ----------
        # ================= [修改点 3: 像变 SNR 一样变导频数量检验泛化性] =================
        sum_val_loss = 0.0
        sum_val_acc = 0.0
        
        print(f"\n--- Epoch {epoch+1:02d} 详细 导频数(K) 泛化性验证 ---")
        
        # 遍历全量测试集中的每一个 K 维度
        for val_K in VAL_K_POOL:
            # 在验证当前维度前，动态修改 Allocator 内部的 K
            allocator.K = val_K
            
            # 执行验证
            v_loss, v_acc = validate_cached(allocator, sparse_mlp, val_cached_loader, TARGET_SNR_LINEAR, DEVICE)
            
            print(f"  [Pilots K: {val_K:4d}] Val Acc: {v_acc:5.2f}% | Loss: {v_loss:.4f}")
            
            sum_val_loss += v_loss
            sum_val_acc += v_acc
            
        avg_val_acc = sum_val_acc / len(VAL_K_POOL)
        avg_val_loss = sum_val_loss / len(VAL_K_POOL)
        # ==============================================================================
        
        train_accs.append(epoch_train_acc)
        val_accs.append(avg_val_acc)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} 汇总 | "
              f"Train Acc: {epoch_train_acc:.2f}% | "
              f"Avg Val Acc (All K): {avg_val_acc:.2f}% Loss: {avg_val_loss:.4f}")
        
        # ---------- 保存最优模型 ----------
        # ================= [修改点 4: 修改保存模型及图片格式名称] =================
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_path = os.path.join(SAVE_DIR, "best_model-GNN_K_Generalization.pth")
            torch.save({
                'epoch': epoch + 1,
                'allocator_state_dict': allocator.state_dict(),
                'sparse_mlp_state_dict': sparse_mlp.state_dict(),
                'val_acc': avg_val_acc, 
            }, save_path)
            print(f"[*] 🚀 突破！检测到更好的平均(K泛化)验证集准确率，模型已保存至: {save_path}")

    print(f"\n[DONE] 训练完成！最佳平均验证集准确率: {best_val_acc:.2f}%")

    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS+1), val_accs, label='Avg Validation Accuracy (Across All K)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('GNN Allocator K-Generalization Learning Curve')
    plt.legend()
    plt.grid(True)
    curve_path = os.path.join(SAVE_DIR, 'GNN_training_curves_varK_Generalization.png')
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] 准确率曲线图已保存至: {curve_path}")

if __name__ == "__main__":
    main()