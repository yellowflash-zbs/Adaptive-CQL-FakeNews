# coding: utf-8
import os
import json
import numpy as np
import pickle
import argparse
from tqdm import tqdm

# 终极暴力寻路
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    parser = argparse.ArgumentParser(description="将 JSONL 转换为 RL 标准 PKL 格式")
    parser.add_argument("--dataset", type=str, default="LIAR-RAW", choices=["LIAR-RAW", "RAWFC"])
    args = parser.parse_args()
    dataset_name = args.dataset

    input_path = os.path.join(project_root, "datasets", dataset_name, "rl_offline_buffer_with_rewards.jsonl")
    output_path = os.path.join(project_root, "datasets", dataset_name, "rlkit_offline_dataset.pkl")
    
    if not os.path.exists(input_path):
        print(f"❌ 找不到输入文件: {input_path}")
        return

    observations, actions, rewards, terminals = [], [], [], []
    MAX_SENTENCES = 60
    VECTOR_DIM = 768
    
    print(f"\n📦 正在加载 【{dataset_name}】 离线经验池，准备转换为底层矩阵...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="矩阵转换进度"):
        if not line.strip(): continue
        data = json.loads(line)
        
        # 1. 动作矩阵 (Action: 60维的 Multi-hot 向量)
        action_vec = np.zeros(MAX_SENTENCES, dtype=np.float32)
        for idx in data["action_selected_indices"]:
            if idx < MAX_SENTENCES:
                action_vec[idx] = 1.0
        actions.append(action_vec)
        
        # 2. 奖励矩阵 (Reward)
        rewards.append(np.array([data["reward_total"]], dtype=np.float32))
        
        # 3. 状态矩阵 (State: 46848维)
        claim_vec = np.array(data["state_claim_vector"], dtype=np.float32).flatten()
        if len(claim_vec) > VECTOR_DIM:
            claim_vec = claim_vec[:VECTOR_DIM]
        elif len(claim_vec) < VECTOR_DIM:
            claim_vec = np.pad(claim_vec, (0, VECTOR_DIM - len(claim_vec)))
            
        cand_vecs = data["state_candidate_vectors"]
        padded_cand_vecs = np.zeros((MAX_SENTENCES, VECTOR_DIM), dtype=np.float32)
        
        valid_idx = 0
        for vec in cand_vecs:
            if valid_idx >= MAX_SENTENCES: break
            flat_vec = np.array(vec, dtype=np.float32).flatten()
            if len(flat_vec) == 0: continue 
            elif len(flat_vec) >= VECTOR_DIM:
                padded_cand_vecs[valid_idx, :] = flat_vec[:VECTOR_DIM]
            else:
                padded_cand_vecs[valid_idx, :len(flat_vec)] = flat_vec
            valid_idx += 1
            
        state_vec = np.concatenate([claim_vec, padded_cand_vecs.flatten()])
        observations.append(state_vec)
        
        # 4. 终止信号 (Terminal)
        terminals.append(np.array([True], dtype=bool))
        
    print("⏳ 正在进行底层的 Numpy 堆叠 (可能需要十几秒)...")
    dataset = {
        'observations': np.stack(observations),
        'actions': np.stack(actions),
        'rewards': np.stack(rewards),
        'terminals': np.stack(terminals),
        'next_observations': np.zeros_like(np.stack(observations)) # 单步 Bandit 任务，下一个状态补零
    }
    
    print(f"📊 数据集概览: Observations {dataset['observations'].shape} | Actions {dataset['actions'].shape}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f"🎉 转换彻底完成！PKL 已保存至: {output_path}")

if __name__ == '__main__':
    main()