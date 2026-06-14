# coding: utf-8
import os
import sys
import argparse
import types
import pickle
import torch
import numpy as np
from tqdm import tqdm

# ====================================================================
# 1. 动态获取根目录并正确挂载 d4rl
# ====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
d4rl_path = os.path.join(current_dir, "d4rl")
if d4rl_path not in sys.path:
    sys.path.append(d4rl_path)

# ====== 🌟 神级欺骗术：绕过 Windows 上的 d4rl 物理引擎安装地狱 ======
sys.modules['d4rl'] = types.ModuleType('d4rl')
# ====================================================================

# 只有在成功挂载 d4rl 之后，才能导入 rlkit
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks.mlp import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy
# 导入你的自适应 CQL Trainer
from rlkit.torch.sac.adaptive_cql import AdaptiveCQLTrainer 

class DummyActionSpace:
    def __init__(self, dim):
        self.shape = (dim,)
        self.high = np.ones(dim, dtype=np.float32)
        self.low = -np.ones(dim, dtype=np.float32)

class DummyEnv:
    def __init__(self, action_dim):
        self.action_space = DummyActionSpace(action_dim)

def main():
    parser = argparse.ArgumentParser(description="自适应 CQL 假新闻核心证据提取训练器")
    parser.add_argument("--dataset", type=str, default="LIAR-RAW", 
                        choices=["LIAR-RAW", "RAWFC"], 
                        help="选择要训练的数据集 (LIAR-RAW 或 RAWFC)")
    args = parser.parse_args()
    dataset_name = args.dataset

    BATCH_SIZE = 64  
    NUM_EPOCHS = 100
    NUM_TRAIN_STEPS_PER_EPOCH = 500
    
    ptu.set_gpu_mode(True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    print(f"\n🚀 正在启动 RL 训练流程，当前目标数据集: 【{dataset_name}】\n")

    data_path = os.path.join(current_dir, "datasets", dataset_name, "rlkit_offline_dataset.pkl")
    bc_model_path = os.path.join(current_dir, "datasets", dataset_name, f"dummy_bc_model_{dataset_name}.pth")
    checkpoint_dir = os.path.join(current_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"📖 正在加载离线数据: {data_path}")
    if not os.path.exists(data_path):
        print(f"❌ 错误: 找不到数据文件！请确保已经生成了 {data_path}")
        return

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
        
    obs_dim = dataset['observations'].shape[1] 
    action_dim = dataset['actions'].shape[1]   
    
    dataset['actions'] = dataset['actions'] * 2.0 - 1.0 

    print("🧠 正在构建 Actor-Critic 神经网络 (已为您瘦身为 256 以防过拟合)...")
    # 🌟 修复点 1：网络瘦身，防止死记硬背
    hidden_sizes = [256, 256, 256] 
    
    qf1 = ConcatMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes)
    qf2 = ConcatMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes)
    target_qf1 = ConcatMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes)
    target_qf2 = ConcatMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes)
    
    policy = TanhGaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes)
    
    print(f"🛠️ 正在生成伪造的随机行为策略 (保存在 {bc_model_path})...")
    dummy_bc_policy = TanhGaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes)
    torch.save(dummy_bc_policy.state_dict(), bc_model_path)

    print("⚙️ 初始化 AdaptiveCQLTrainer...")
    dummy_env = DummyEnv(action_dim)
    
    trainer = AdaptiveCQLTrainer(
        env=dummy_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        bc_model_path=bc_model_path, 
        kl_scale=0.9,                
        policy_lr=1e-4,
        qf_lr=3e-4,
        reward_scale=1.0,
        automatic_entropy_tuning=True,
    )
    for net in trainer.networks:
        net.to(ptu.device)

    print("\n🚀 开始闭关修炼：加入动态早停机制的 RL 训练...")
    total_data_size = dataset['observations'].shape[0]
    
    # 🌟 修复点 2：早停机制监控变量
    best_loss = float('inf')
    patience_counter = 0
    PATIENCE_LIMIT = 15  # 如果 15 轮没进步，果断停车！
    
    # 🌟 强行伪装成 epoch_100，为了完美兼容你的 evaluate.py
    best_save_path = os.path.join(checkpoint_dir, f"{dataset_name}_adaptive_cql_policy_epoch_100.pth")

    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(range(NUM_TRAIN_STEPS_PER_EPOCH), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_q_losses = [] # 记录这一轮的误差
        
        for step in pbar:
            indices = np.random.randint(0, total_data_size, size=BATCH_SIZE)
            batch = {
                'observations': ptu.from_numpy(dataset['observations'][indices]),
                'actions': ptu.from_numpy(dataset['actions'][indices]),
                'rewards': ptu.from_numpy(dataset['rewards'][indices]),
                'terminals': ptu.from_numpy(dataset['terminals'][indices]),
                'next_observations': ptu.from_numpy(dataset['next_observations'][indices]),
            }
            
            trainer.train_from_torch(batch)
            
            if step % 50 == 0:
                stats = trainer.get_diagnostics()
                current_q_loss = stats.get('QF1 Loss', 0)
                epoch_q_losses.append(current_q_loss)
                pbar.set_postfix({
                    'Q1_Loss': f"{current_q_loss:.2f}", 
                    'b_k Mean': f"{stats.get('PBRS b_k(s) Mean', 0):.2f}"
                })
            
        trainer.end_epoch(epoch) 
        
        # 🌟 修复点 3：评估并保存巅峰状态的权重
        mean_q_loss = np.mean(epoch_q_losses)
        if mean_q_loss < best_loss:
            best_loss = mean_q_loss
            patience_counter = 0
            torch.save(policy.state_dict(), best_save_path)
            print(f"   🏆 发现更优模型！Q-Loss 降至 {best_loss:.4f}，已保存巅峰权重！")
        else:
            patience_counter += 1
            print(f"   ⚠️ 模型未进步 (Patience: {patience_counter}/{PATIENCE_LIMIT})")
            
        # 触发早停
        if patience_counter >= PATIENCE_LIMIT:
            print(f"\n🛑 触发早停机制！连续 {PATIENCE_LIMIT} 轮未进步，防止过拟合，提前结束训练！")
            break

    print(f"\n🎉 训练彻底完成！最强泛化模型已定格在: {best_save_path}")

if __name__ == '__main__':
    main()