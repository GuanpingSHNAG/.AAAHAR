import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score # [NEW] 导入评价指标

# 配置项目根目录
PROJECT_ROOT = "/home/gshang/.AAAHAR"
sys.path.append(f"{PROJECT_ROOT}/rawdata_train")
sys.path.append(f"{PROJECT_ROOT}/Diffusion")
sys.path.append(f"{PROJECT_ROOT}/Csem")

from load.supervised.benchmark_loader import load_benchmark_supervised
from model.supervised.models import MLPClassifier, ResNet18Classifier
from semantic_engine_FH import SemanticEngine
from DGCPA import Allocator

def get_fixed_feedback_mask(B, device):
    """
    固定反馈导频掩码。位于前两个时间步。
    """
    mask = torch.zeros((B, 1, 32, 32), dtype=torch.float32, device=device)
    mask[:, 0, 0, 0:32:5] = 1.0
    mask[:, 0, 1, 3:32:5] = 1.0
    return mask

def get_fixed_feedback_mask2(B, device):
    """
    固定反馈导频掩码。位于前两个时间步。
    """
    mask = torch.zeros((B, 1, 32, 32), dtype=torch.float32, device=device)
    mask[:, 0, 0, 0:32:5] = 1.0
    mask[:, 0, 1, 3:32:5] = 1.0
    mask[:, 0, 2, 0:32:5] = 1.0
    return mask


def get_cached_dataloader(dataloader, sem_engine, device, cache_path, desc="Caching", is_train=True):
    """
    部分特征缓存函数：只缓存原始输入(inputs)、语义特征(c_sem_batch)和标签(labels)。
    """
    if os.path.exists(cache_path):
        print(f"[*] 🟢 找到本地缓存文件，直接加载: {cache_path}")
        cached_data = torch.load(cache_path)
        cached_inputs = cached_data['inputs']
        cached_c_sem = cached_data['c_sem']
        cached_labels = cached_data['labels']
    else:
        print(f"[*] 🟡 未找到缓存，开始在 GPU 上提取语义特征: {desc}")
        sem_engine.eval()
        
        cached_inputs = []
        cached_c_sem = []
        cached_labels = []
        
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
        
        print(f"[*] 💾 正在将特征保存至硬盘: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            'inputs': cached_inputs, 
            'c_sem': cached_c_sem, 
            'labels': cached_labels
        }, cache_path)
    
    dataset = TensorDataset(cached_inputs, cached_c_sem, cached_labels)
    return DataLoader(dataset, batch_size=dataloader.batch_size, shuffle=is_train)


def validate_cached(allocator, sparse_mlp, val_loader_cached, target_snr, device):
    """使用缓存的语义特征进行验证集评估，并计算 Acc 和 F1"""
    allocator.eval()
    sparse_mlp.eval()
    val_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, c_sem_batch, labels in val_loader_cached:
            inputs = inputs.to(device).float()
            c_sem_batch = c_sem_batch.to(device)
            labels = labels.to(device)
            B = inputs.shape[0]
            
            snr_budget_batch = torch.full((B, 1), target_snr, device=device)
            fixed_mask = get_fixed_feedback_mask(B, device)
            
            topk_indices, power_alloc = allocator(c_sem_batch, snr_budget_batch)
            y_received_sparse, total_mask = allocator.simulate_sensing_link(inputs, topk_indices, power_alloc, fixed_mask)
            logits = sparse_mlp(y_received_sparse)
            
            loss = criterion(logits, labels)
            val_loss += loss.item() * B
            total_samples += B
            
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = val_loss / total_samples
    
    # [NEW] 计算 Accuracy 和 Macro F1-score
    acc = accuracy_score(all_labels, all_preds) * 100.0
    f1 = f1_score(all_labels, all_preds, average='macro') * 100.0
    
    return avg_loss, acc, f1


def main():
    torch.autograd.set_detect_anomaly(True)

    # ================= 1. 环境与配置 =================
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except:
        local_rank = 0
    DEVICE = torch.device(f"cuda:{local_rank}")

    DATASET_ROOT = f"{PROJECT_ROOT}/HumanActivityRecognition"
    TASK_NAME = "HumanActivityRecognition"
    
    BATCH_SIZE = 256
    EPOCHS = 50   # 统一跑50个epoch
    NP_LIST = [32, 64, 128, 256, 512]  # 需要跑的 Np 列表
    
    print(f"[*] 设备: {DEVICE} | Epochs: {EPOCHS} | Np: {NP_LIST}")

    # ================= 2. 加载数据集 =================
    print("[*] 正在加载训练与验证数据...")
    data_info = load_benchmark_supervised(
        dataset_root=DATASET_ROOT, task_name=TASK_NAME,
        batch_size=BATCH_SIZE, train_split="train_id", val_split="val_id",
        test_splits=["test_id"], num_workers=12, use_root_as_task_dir=False
    )
    raw_train_loader = data_info['loaders']['train']
    raw_val_loader = data_info['loaders']['val']


    # ================= 3. 全局统一特征缓存 (只执行一次) =================
    CACHE_DIR = f"{PROJECT_ROOT}/PowAllocate/cached_features2"
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_cache_path = os.path.join(CACHE_DIR, f"train_features_snr.pt")
    val_cache_path = os.path.join(CACHE_DIR, f"val_features_snr.pt")

    needs_caching = not (os.path.exists(train_cache_path) and os.path.exists(val_cache_path))
    
    if needs_caching:
        print("[*] 初始化 Semantic Engine 用于提取特征 (全局只提取一次)...")
        sem_engine = SemanticEngine(device=DEVICE, num_classes=data_info['num_classes']).to(DEVICE)
        sem_engine.eval()
    else:
        print("[*] 检测到本地完整缓存，跳过 sem_engine 加载！")
        sem_engine = None

    train_cached_loader = get_cached_dataloader(
        raw_train_loader, sem_engine, DEVICE, 
        cache_path=train_cache_path, desc="Cache Train", is_train=True
    )
    val_cached_loader = get_cached_dataloader(
        raw_val_loader, sem_engine, DEVICE, 
        cache_path=val_cache_path, desc="Cache Val", is_train=False
    )

    if needs_caching:
        del sem_engine
        torch.cuda.empty_cache()


    # 用字典保存所有Np的最优结果
    final_results = {}

    # ================= 4. Np 循环训练 =================
    for NUM_DYNAMIC_PILOTS in NP_LIST:
        print(f"\n{'='*60}")
        print(f"🚀 开始端到端联合训练 | Np = {NUM_DYNAMIC_PILOTS}")
        print(f"{'='*60}")
        
        # 为当前 Np 初始化新的 Allocator
        allocator = Allocator(N=1024, num_dynamic_pilots=NUM_DYNAMIC_PILOTS).to(DEVICE)
        
        # 每次循环重新加载 Sparse MLP 的预训练权重，保证起点一致
        sparse_mlp = MLPClassifier(win_len=32, feature_size=32, num_classes=data_info['num_classes']).to(DEVICE)
        sparse_mlp.load_state_dict(torch.load("/home/gshang/.AAAHAR/Csem/稀疏随机MLP/pre.pth", map_location=DEVICE))

        optimizer = optim.AdamW([
            {'params': allocator.parameters(), 'lr': 1e-4}, 
            {'params': sparse_mlp.parameters(), 'lr': 1e-6}
        ], weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
            schedulers=[
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-20),
                torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1e-20/optimizer.defaults['lr'])
            ],
            milestones=[10]
        )

        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        best_metrics_for_this_np = {} # 记录最优 Epoch 时的所有细分 SNR 数据

        for epoch in range(EPOCHS):
            # ---------- 训练阶段 ----------
            allocator.train()
            sparse_mlp.train() 
            
            train_loss = 0.0
            train_correct = 0
            total_samples = 0
            
            pbar = tqdm(train_cached_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train Np={NUM_DYNAMIC_PILOTS}]", leave=False)
            
            for inputs, c_sem_batch, labels in pbar:
                inputs = inputs.to(DEVICE).float()
                c_sem_batch = c_sem_batch.to(DEVICE)
                labels = labels.to(DEVICE)
                B = inputs.shape[0]
                
                SNR_MIN = 1.0    
                SNR_MAX = 1000.0  
                snr_budget_batch = (SNR_MAX - SNR_MIN) * torch.rand((B, 1), device=DEVICE) + SNR_MIN
                fixed_mask = get_fixed_feedback_mask(B, DEVICE)
                
                optimizer.zero_grad()
                
                topk_indices, power_alloc = allocator(c_sem_batch, snr_budget_batch)
                y_received_sparse, total_mask = allocator.simulate_sensing_link(inputs, topk_indices, power_alloc, fixed_mask)
                logits = sparse_mlp(y_received_sparse)
                
                loss = criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(allocator.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(sparse_mlp.parameters(), max_norm=1.0)

                optimizer.step()
                
                train_loss += loss.item() * B
                total_samples += B
                _, predicted = logits.max(1)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100.*train_correct/total_samples:.1f}%"})
            
            scheduler.step()
            
            # ---------- 验证阶段 ----------
            snr_db_list = [0, 5, 10, 15, 20, 25, 30]
            sum_val_acc = 0.0
            sum_val_f1 = 0.0
            current_epoch_details = {}
            
            for snr_db in snr_db_list:
                target_snr_linear = 10.0 ** (snr_db / 10.0)
                v_loss, v_acc, v_f1 = validate_cached(allocator, sparse_mlp, val_cached_loader, target_snr_linear, DEVICE)
                
                current_epoch_details[snr_db] = {'acc': v_acc, 'f1': v_f1}
                sum_val_acc += v_acc
                sum_val_f1 += v_f1
                
            avg_val_acc = sum_val_acc / len(snr_db_list)
            avg_val_f1 = sum_val_f1 / len(snr_db_list)
            
            print(f"Np={NUM_DYNAMIC_PILOTS:3d} | Epoch {epoch+1:02d}/{EPOCHS} | Train Acc: {100.*train_correct/total_samples:.2f}% | Avg Val Acc: {avg_val_acc:.2f}% | Avg Val F1: {avg_val_f1:.2f}%")
            
            # ---------- 记录最优模型的数据 ----------
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                # 记录这一轮的详细成绩单
                best_metrics_for_this_np = {
                    'avg_acc': avg_val_acc,
                    'avg_f1': avg_val_f1,
                    'details': current_epoch_details
                }
        
        # 将当前 Np 的最优数据存入全局大字典
        final_results[NUM_DYNAMIC_PILOTS] = best_metrics_for_this_np
        print(f"✅ Np={NUM_DYNAMIC_PILOTS} 训练完毕。最优 Avg Val Acc: {best_val_acc:.2f}%")
        
        # 主动释放显存，防止 OOM
        del allocator
        del sparse_mlp
        torch.cuda.empty_cache()


    # ================= 5. 打印所有 Np 的最优验证集结果（80个结果） =================
    print("\n\n" + "="*70)
    print("🎯 所有 Np 的最优验证结果汇总 (0-30dB 各自 + 平均的 Acc & F1)")
    print("="*70)
    
    for Np in NP_LIST:
        print(f"\n[ Np = {Np} ] 最佳 Epoch 成绩:")
        metrics = final_results[Np]
        details = metrics['details']
        
        # 打印 7 个 SNR 的具体指标
        for snr in [0, 5, 10, 15, 20, 25, 30]:
            acc = details[snr]['acc']
            f1 = details[snr]['f1']
            print(f"  SNR {snr:2d} dB | Acc: {acc:6.2f}% | F1: {f1:6.2f}%")
        
        # 打印平均指标
        print(f"  {'-'*35}")
        print(f"  Average   | Acc: {metrics['avg_acc']:6.2f}% | F1: {metrics['avg_f1']:6.2f}%")

if __name__ == "__main__":
    main()