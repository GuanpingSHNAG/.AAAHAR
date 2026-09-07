import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt   # [NEW] 导入绘图库
# 配置项目根目录 (此处按你的原代码保留)
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


# ================= 新增：预计算特征函数 =================
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
                
                # 获取 mask (注意根据你的代码，这里是生成 cond_csi 用的)
                fixed_mask = get_fixed_feedback_mask(B, device)
                cond_csi = inputs * fixed_mask
                
                # 仅做前置网络推理 (如果不需要 diffusion_out，可以用 _ 忽略)
                c_sem_batch, _ = sem_engine(cond_csi, fixed_mask)
                
                # 放入 CPU 内存，防止爆显存
                cached_inputs.append(inputs.cpu())
                cached_c_sem.append(c_sem_batch.cpu())
                cached_labels.append(labels.cpu())
                
        # 拼接张量
        cached_inputs = torch.cat(cached_inputs, dim=0)
        cached_c_sem = torch.cat(cached_c_sem, dim=0)
        cached_labels = torch.cat(cached_labels, dim=0)
        
        # 保存到硬盘
        print(f"[*] 💾 正在将特征保存至硬盘: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            'inputs': cached_inputs, 
            'c_sem': cached_c_sem, 
            'labels': cached_labels
        }, cache_path)
    
    # 封装成 TensorDataset (注意现在有 3 个数据项)
    dataset = TensorDataset(cached_inputs, cached_c_sem, cached_labels)
    return DataLoader(dataset, batch_size=dataloader.batch_size, shuffle=is_train)



# 修改点：在参数列表增加 target_snr
def validate_cached(allocator, sparse_mlp, val_loader_cached, target_snr, device):
    """使用缓存的语义特征进行验证集评估，在指定的固定 SNR 下测试"""
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
            
            # ================= [重点修改] =================
            # 删除了随机生成的代码，改为使用外部传入的固定 target_snr (线性值)
            snr_budget_batch = torch.full((B, 1), target_snr, device=device)
            # ==============================================
            
            fixed_mask = get_fixed_feedback_mask(B, device)
            
            # 1. 功率与导频分配
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
            
    avg_loss = val_loss / total_samples
    avg_acc = 100. * val_correct / total_samples
    return avg_loss, avg_acc



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
    SAVE_DIR = f"{PROJECT_ROOT}/PowAllocate/checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    BATCH_SIZE =256
    EPOCHS = 200
    NUM_DYNAMIC_PILOTS = 256
  
    
    print(f"[*] 设备: {DEVICE} | 动态导频数: {NUM_DYNAMIC_PILOTS} | Epochs: {EPOCHS}")

    # ================= 2. 加载数据集 =================
    print("[*] 正在加载训练与验证数据...")
    data_info = load_benchmark_supervised(
        dataset_root=DATASET_ROOT, task_name=TASK_NAME,
        batch_size=BATCH_SIZE, train_split="train_id", val_split="val_id",
        test_splits=["test_id"], num_workers=12, use_root_as_task_dir=False
    )
    raw_train_loader = data_info['loaders']['train']
    raw_val_loader = data_info['loaders']['val']




    # ================= 3. 初始化网络模型 =================
    print("[*] 初始化 Semantic Engine (Frozen)...")
    sem_engine = SemanticEngine(device=DEVICE, num_classes=data_info['num_classes'])
    
    # 【重点修改】Allocator 现在是可训练模型，初始化后不能再被删除了！
    print("[*] 初始化 图神经网络 Power Allocator (Trainable)...")
    allocator = Allocator(N=1024, num_dynamic_pilots=NUM_DYNAMIC_PILOTS).to(DEVICE)
    
    # 缓存路径
    CACHE_DIR = f"{PROJECT_ROOT}/PowAllocate/cached_features2"
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_cache_path = os.path.join(CACHE_DIR, f"train_features_snr.pt")
    val_cache_path = os.path.join(CACHE_DIR, f"val_features_snr.pt")

    # 判断是否需要预缓存
    needs_caching = not (os.path.exists(train_cache_path) and os.path.exists(val_cache_path))
    
    if needs_caching:
        print("[*] 初始化前置模型用于提取特征...")
        sem_engine = sem_engine.to(DEVICE)
        sem_engine.eval()
        # 注意，由于我们修改了缓存函数只提语义特征，这里缓存时其实不需要 allocator 参与了
    else:
        print("[*] 检测到本地完整缓存，跳过 sem_engine 加载！")

    # 获取缓存数据 (返回 inputs, c_sem, labels)
    train_cached_loader = get_cached_dataloader(
        raw_train_loader, sem_engine, DEVICE, 
        cache_path=train_cache_path, desc="Cache Train", is_train=True
    )
    val_cached_loader = get_cached_dataloader(
        raw_val_loader, sem_engine, DEVICE, 
        cache_path=val_cache_path, desc="Cache Val", is_train=False
    )

    # 【重点修改】现在只删除 sem_engine，保留 allocator！
    if needs_caching:
        del sem_engine
        torch.cuda.empty_cache()

    # ================= 初始化 MLP 并开始光速训练 =================
    print("[*] 初始化 Sparse Task MLP (Trainable)...")
    sparse_mlp = MLPClassifier(win_len=32, feature_size=32, num_classes=data_info['num_classes']).to(DEVICE)
    sparse_mlp.load_state_dict(torch.load("/home/gshang/.AAAHAR/Csem/稀疏随机MLP/pre.pth", map_location=DEVICE))

    # 【重点修改】将 allocator 的参数也加入优化器
    optimizer = optim.AdamW([
        {'params': allocator.parameters(), 'lr': 1e-8}, # 可以给功率分配网络单独设一个学习率
        {'params': sparse_mlp.parameters(), 'lr': 1e-15}
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

    # ================= 5. 端到端联合训练大循环 =================
    print("\n🚀 [START] Cached Joint Training Initiated!")
    for epoch in range(EPOCHS):
        # ---------- 训练阶段 ----------
        allocator.train()  # 开启 allocator 训练模式 (触发 Gumbel-Top-K 等逻辑)
        sparse_mlp.train() 
        
        train_loss = 0.0
        train_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_cached_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        # 【重点修改】解包出 3 个张量
        for inputs, c_sem_batch, labels in pbar:
            inputs = inputs.to(DEVICE).float()
            c_sem_batch = c_sem_batch.to(DEVICE)
            labels = labels.to(DEVICE)
            B = inputs.shape[0]
            
            SNR_MIN = 1.0    # 0dB 
            SNR_MAX = 1000.0  # 30dB
            snr_budget_batch = (SNR_MAX - SNR_MIN) * torch.rand((B, 1), device=DEVICE) + SNR_MIN
            fixed_mask = get_fixed_feedback_mask(B, DEVICE)
            
            optimizer.zero_grad()
            
            # ================= [显式实时推理] =================
            topk_indices, power_alloc = allocator(c_sem_batch, snr_budget_batch)
            y_received_sparse, total_mask = allocator.simulate_sensing_link(inputs, topk_indices, power_alloc, fixed_mask)
            logits = sparse_mlp(y_received_sparse)
            # ==============================================
            
            loss = criterion(logits, labels)
            
            loss.backward()


            # ================= [保命神技：梯度裁剪] =================
            # 强制把 allocator 和 mlp 中所有大于 1.0 的异常梯度按比例缩放回安全范围内
            torch.nn.utils.clip_grad_norm_(allocator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(sparse_mlp.parameters(), max_norm=1.0)
            # =======================================================


            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * B
            total_samples += B
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100.*train_correct/total_samples:.1f}%"})
        
        epoch_train_acc = 100. * train_correct / total_samples
        epoch_train_loss = train_loss / total_samples

        # ---------- 验证阶段 ----------
        # 定义你要测试的 SNR 列表 (dB)
        snr_db_list = [0, 5, 10, 15, 20, 25, 30]
        
        sum_val_loss = 0.0
        sum_val_acc = 0.0
        
        print(f"\n--- Epoch {epoch+1:02d} 详细 SNR 验证 ---")
        
        # 遍历每一个 dB 值进行测试
        for snr_db in snr_db_list:
            # 将 dB 转换为物理仿真所需的线性值 (如 0dB->1.0, 30dB->1000.0)
            target_snr_linear = 10.0 ** (snr_db / 10.0)
            
            # 调用修改后的验证函数
            v_loss, v_acc = validate_cached(allocator, sparse_mlp, val_cached_loader, target_snr_linear, DEVICE)
            
            # 打印当前 SNR 下的成绩
            print(f"  [SNR: {snr_db:2d} dB] Val Acc: {v_acc:5.2f}% | Loss: {v_loss:.4f}")
            
            sum_val_loss += v_loss
            sum_val_acc += v_acc
            
        # 计算所有测试 SNR 的平均准确率和平均 Loss
        avg_val_acc = sum_val_acc / len(snr_db_list)
        avg_val_loss = sum_val_loss / len(snr_db_list)
        
        # 记录到数组中用于最后画图
        train_accs.append(epoch_train_acc)
        val_accs.append(avg_val_acc)
        
        # 打印当前 Epoch 的总成绩单
        print(f"Epoch {epoch+1:02d}/{EPOCHS} 汇总 | "
              f"Train Acc: {epoch_train_acc:.2f}% Loss: {epoch_train_loss:.4f} | "
              f"Avg Val Acc: {avg_val_acc:.2f}% Loss: {avg_val_loss:.4f}")
        
        # ---------- 保存最优模型 ----------
        # 【重点】现在使用平均准确率 (avg_val_acc) 作为判定模型好坏的标准
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_path = os.path.join(SAVE_DIR, f"DGCPA_best_model-SNR_pilots={NUM_DYNAMIC_PILOTS}.pth")
            torch.save({
                'epoch': epoch + 1,
                'allocator_state_dict': allocator.state_dict(),
                'sparse_mlp_state_dict': sparse_mlp.state_dict(),
                'val_acc': avg_val_acc,  # 保存的也是平均准确率
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
    print(f"[*] 🟢 DGCPA 历史数据已保存至 JSON: {json_path}")



    # ================= [NEW] 绘制准确率曲线并保存到 checkpoint 文件夹 =================
    plt.figure(figsize=(10, 6))
    #plt.plot(range(1, EPOCHS+1), train_accs,label='Training Accuracy')
    plt.plot(range(1, EPOCHS+1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy Curves - DGCPA')
    plt.legend()
    plt.grid(True)
    curve_path = os.path.join(SAVE_DIR, f'DGCPA_training_curves_varSNR_Pilots={NUM_DYNAMIC_PILOTS}.png')
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] 准确率曲线图已保存至: {curve_path}")

if __name__ == "__main__":
    main()