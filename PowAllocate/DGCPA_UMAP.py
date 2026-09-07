import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 配置项目根目录
PROJECT_ROOT = "/home/gshang/.AAAHAR"
sys.path.append(f"{PROJECT_ROOT}/rawdata_train")
sys.path.append(f"{PROJECT_ROOT}/Diffusion")
sys.path.append(f"{PROJECT_ROOT}/Csem")

from load.supervised.benchmark_loader import load_benchmark_supervised
from model.supervised.models import MLPClassifier, ResNet18Classifier
from semantic_engine_FH import SemanticEngine
from DGCPA import Allocator
from scipy.special import softmax

# ================= 工业级优化版：特征可视化函数 (UMAP 强力收敛顶刊版) =================
def plot_tsne_mlp_features(allocator, sparse_mlp, val_loader_cached, target_snr_db, device, save_dir, num_classes):
    """
    针对 DGCPA 模型的参数极限压榨 UMAP 可视化 (学术顶刊绘图版)
    """
    import umap.umap_ as umap
    import numpy as np

    print(f"\n[*] 开始提取 Logits 用于 UMAP 强力收敛可视化 (SNR: {target_snr_db} dB)...")
    
    mlp_output_features = []
    all_labels = []
    
    allocator.eval()
    sparse_mlp.eval()
    target_snr_linear = 10.0 ** (target_snr_db / 10.0)

    # 1. 提取 Logits
    with torch.no_grad():
        for inputs, c_sem_batch, labels in tqdm(val_loader_cached, desc="Extracting Logits"):
            inputs = inputs.to(device).float()
            c_sem_batch = c_sem_batch.to(device)
            labels = labels.to(device)
            B = inputs.shape[0]

            fixed_mask = get_fixed_feedback_mask(B, device)
            snr_budget_batch = torch.full((B, 1), target_snr_linear, device=device)
            
            # DGCPA 链路模拟与分配推理
            topk_indices, power_alloc = allocator(c_sem_batch, snr_budget_batch)
            y_received_sparse, total_mask = allocator.simulate_sensing_link(inputs, topk_indices, power_alloc, fixed_mask)
            
            # 稀疏分类器推理
            logits = sparse_mlp(y_received_sparse)
            
            mlp_output_features.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    X_logits = np.concatenate(mlp_output_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # ================= 【随机采样与打乱】 =================
    sample_size = min(4000, len(X_logits)) 
    print(f"[*] 原始样本数量: {len(X_logits)} | 随机采样数量: {sample_size}")
    
    np.random.seed(42)
    indices = np.random.choice(len(X_logits), sample_size, replace=False)
    X_sampled = X_logits[indices]
    y_sampled = y[indices]

    # ================= 【特征重构：Logits -> Probabilities】 =================
    # 限制在单纯形(Simplex)上
    l2_norms = np.linalg.norm(X_sampled, axis=1, keepdims=True)
    X_norm = X_sampled / (l2_norms + 1e-8)

    # ================= 【UMAP 顶刊级平衡参数 (保持对比公平一致)】 =================
    print("[*] 正在运行 UMAP 降维 (启用 PCA 骨架与概率余弦度量)...")
    reducer = umap.UMAP(
        n_neighbors=40,        
        min_dist=0.5,         
        metric='cosine',       
        init='pca',            
        n_components=2,
        n_epochs=1000,          
        n_jobs=-1
    )
    
    X_2d = reducer.fit_transform(X_norm) 

    # ================= 【高级视觉构图与符号映射 (顶刊规范)】 =================
    class_names = ['jumping', 'running', 'seated-breathing', 'walking', 'wavinghand']
    # 高级低饱和度科研配色 (深蓝, 砖红, 墨绿, 藕荷, 土黄)
    colors = ['#3b6291', '#943c39',  '#624c7c', '#779043','#e49f36']
    # 形状映射 (五角星已替换为倒三角 'v')
    markers = ['o', '^', 's', 'D', 'v']
    size = [55, 60, 50, 50, 60] 
    
    # 开启画布 (设置全局无加粗)
    plt.rcParams['font.weight'] = 'normal'
    plt.figure(figsize=(9.5, 7), dpi=500) 
    ax = plt.gca()
    
    # 绘制底层的 Grid 网格线 (zorder=1)
    ax.grid(True, linestyle='-', linewidth=0.6, color='#E5E5E5', zorder=1)
    
    for i in range(num_classes):
        idx = (y_sampled == i)
        plt.scatter(
            X_2d[idx, 0], X_2d[idx, 1], 
            c=colors[i], 
            marker=markers[i],
            s=size[i],               
            alpha=0.85,          
            edgecolors='white',  # 小白边
            linewidths=0.6,      # 白边宽度
            label=class_names[i],
            zorder=3             # 确保散点在网格线之上
        )
    
    # ================= 【学术风坐标轴处理】 =================
    # 黑色全包边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
        
    # 隐藏刻度值和短棍，保留内部 Grid
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    # ================= 【高级图例设计】 =================
    legend = plt.legend(
        loc='upper left',      # 放置在坐标轴内部的左上角
        fontsize=11.5,
        frameon=False    # 标题不加粗
    )
    
    # 保持图例中散点的大小，不受透明度影响
    for lh in legend.legend_handles: 
        lh.set_alpha(1)     
        lh._sizes = [70]    
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # ================= 【保存双格式：PNG + EPS】 =================
    png_path = os.path.join(save_dir, 'umap_DGCPA256.png')
    plt.savefig(png_path, dpi=600, bbox_inches='tight', transparent=False) 
    
    eps_path = os.path.join(save_dir, 'umap_DGCPA256.eps')
    plt.savefig(eps_path, format='eps', dpi=600, bbox_inches='tight') 
    plt.close()
    
    print(f"[*] 绘图完毕！顶刊风格 UMAP 图片已成功保存至:\n  - {png_path}\n  - {eps_path}")

def get_fixed_feedback_mask(B, device):
    mask = torch.zeros((B, 1, 32, 32), dtype=torch.float32, device=device)
    mask[:, 0, 0, 0:32:5] = 1.0
    mask[:, 0, 1, 3:32:5] = 1.0
    return mask



def get_cached_dataloader(dataloader, sem_engine, device, cache_path, desc="Caching", is_train=True):
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
            
            snr_budget_batch = torch.full((B, 1), target_snr, device=device)
            fixed_mask = get_fixed_feedback_mask(B, device)
            
            topk_indices, power_alloc = allocator(c_sem_batch, snr_budget_batch)
            y_received_sparse, total_mask = allocator.simulate_sensing_link(inputs, topk_indices, power_alloc, fixed_mask)
            logits = sparse_mlp(y_received_sparse)
            
            loss = criterion(logits, labels)
            
            val_loss += loss.item() * B
            total_samples += B
            _, predicted = logits.max(1)
            val_correct += predicted.eq(labels).sum().item()
            
    avg_loss = val_loss / total_samples
    avg_acc = 100. * val_correct / total_samples
    return avg_loss, avg_acc

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
    
    BATCH_SIZE = 256
    EPOCHS = 3
    NUM_DYNAMIC_PILOTS = 512
  
    print(f"[*] 设备: {DEVICE} | 动态导频数: {NUM_DYNAMIC_PILOTS} | Epochs: {EPOCHS}")

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
    
    print("[*] 初始化 图神经网络 Power Allocator (Trainable)...")
    allocator = Allocator(N=1024, num_dynamic_pilots=NUM_DYNAMIC_PILOTS).to(DEVICE)
    
    CACHE_DIR = f"{PROJECT_ROOT}/PowAllocate/cached_features2"
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_cache_path = os.path.join(CACHE_DIR, f"train_features_snr.pt")
    val_cache_path = os.path.join(CACHE_DIR, f"val_features_snr.pt")

    needs_caching = not (os.path.exists(train_cache_path) and os.path.exists(val_cache_path))
    
    if needs_caching:
        print("[*] 初始化前置模型用于提取特征...")
        sem_engine = sem_engine.to(DEVICE)
        sem_engine.eval()
    else:
        print("[*] 检测到本地完整缓存，跳过 sem_engine 加载！")

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

    print("[*] 初始化 Sparse Task MLP (Trainable)...")
    sparse_mlp = MLPClassifier(win_len=32, feature_size=32, num_classes=data_info['num_classes']).to(DEVICE)
    
    pre_path = "/home/gshang/.AAAHAR/Csem/稀疏随机MLP/pre.pth"
    if os.path.exists(pre_path):
        sparse_mlp.load_state_dict(torch.load(pre_path, map_location=DEVICE))

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
    train_accs, val_accs = [], []

    print("\n🚀 [START] Cached Joint Training Initiated!")
    for epoch in range(EPOCHS):
        allocator.train()  
        sparse_mlp.train() 
        
        train_loss = 0.0
        train_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_cached_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
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
            scheduler.step()

            train_loss += loss.item() * B
            total_samples += B
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100.*train_correct/total_samples:.1f}%"})
        
        epoch_train_acc = 100. * train_correct / total_samples
        epoch_train_loss = train_loss / total_samples

        snr_db_list = [0, 5, 10, 15, 20, 25, 30]
        sum_val_loss = 0.0
        sum_val_acc = 0.0
        
        print(f"\n--- Epoch {epoch+1:02d} 详细 SNR 验证 ---")
        for snr_db in snr_db_list:
            target_snr_linear = 10.0 ** (snr_db / 10.0)
            v_loss, v_acc = validate_cached(allocator, sparse_mlp, val_cached_loader, target_snr_linear, DEVICE)
            print(f"  [SNR: {snr_db:2d} dB] Val Acc: {v_acc:5.2f}% | Loss: {v_loss:.4f}")
            sum_val_loss += v_loss
            sum_val_acc += v_acc
            
        avg_val_acc = sum_val_acc / len(snr_db_list)
        avg_val_loss = sum_val_loss / len(snr_db_list)
        
        train_accs.append(epoch_train_acc)
        val_accs.append(avg_val_acc)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} 汇总 | "
              f"Train Acc: {epoch_train_acc:.2f}% Loss: {epoch_train_loss:.4f} | "
              f"Avg Val Acc: {avg_val_acc:.2f}% Loss: {avg_val_loss:.4f}")
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_path = os.path.join(SAVE_DIR, f"DGCPA_best_model-SNR_pilots={NUM_DYNAMIC_PILOTS}.pth")
            torch.save({
                'epoch': epoch + 1,
                'allocator_state_dict': allocator.state_dict(),
                'sparse_mlp_state_dict': sparse_mlp.state_dict(),
                'val_acc': avg_val_acc,
            }, save_path)
            print(f"[*] 🚀 突破！检测到更好的平均验证集准确率，模型已保存至: {save_path}")

    print(f"\n[DONE] 训练完成！最佳验证集准确率: {best_val_acc:.2f}%")

    import json
    history_data = {
        "method": "DGCPA Baseline",
        "num_dynamic_pilots": NUM_DYNAMIC_PILOTS,
        "epochs": list(range(1, EPOCHS + 1)),
        "train_accs": train_accs, 
        "val_accs": val_accs      
    }
    json_path = os.path.join(SAVE_DIR, f'history_DGCPA_pilots={NUM_DYNAMIC_PILOTS}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=4, ensure_ascii=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS+1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy Curves - DGCPA')
    plt.legend()
    plt.grid(True)
    curve_path = os.path.join(SAVE_DIR, f'DGCPA_training_curves_varSNR_Pilots={NUM_DYNAMIC_PILOTS}.png')
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ================= 调用 UMAP =================
    best_model_path = os.path.join(SAVE_DIR, f"DGCPA_best_model-SNR_pilots={NUM_DYNAMIC_PILOTS}.pth")
    if os.path.exists(best_model_path):
        print(f"\n[*] 加载最优模型用于 UMAP 可视化: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        allocator.load_state_dict(checkpoint['allocator_state_dict'])
        sparse_mlp.load_state_dict(checkpoint['sparse_mlp_state_dict'])
        
        TARGET_SNR_DB_FOR_TSNE = 30
        
        plot_tsne_mlp_features(
            allocator=allocator,
            sparse_mlp=sparse_mlp,
            val_loader_cached=val_cached_loader,
            target_snr_db=TARGET_SNR_DB_FOR_TSNE,
            device=DEVICE,
            save_dir=SAVE_DIR,
            num_classes=data_info['num_classes']
        )

if __name__ == "__main__":
    main()