# coding: utf-8
import os
import sys
import json
import torch
import numpy as np
import ijson
import argparse
from collections import Counter  # 用于统计标签比例
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from core.feature_utils import build_state_vector, build_downstream_features
from core.cql_agent import load_adaptive_cql_policy, get_rl_top5
from core.mlp_proxy import get_proxy_classifier
from core.logger import setup_logger

import warnings
warnings.filterwarnings("ignore")

# ================= 1. 核心抽取算法 =================

def get_random_top5(num_candidates):
    indices = np.arange(num_candidates)
    np.random.shuffle(indices)
    return indices[:5]

def get_cosine_top5(claim_vec, cand_vecs):
    vector_dim = 768
    scores = []
    
    c_vec = np.array(claim_vec, dtype=np.float32).flatten()[:vector_dim]
    if len(c_vec) < vector_dim:
        c_vec = np.pad(c_vec, (0, vector_dim - len(c_vec)))
    c_norm = np.linalg.norm(c_vec)
    
    for cv in cand_vecs:
        v = np.array(cv, dtype=np.float32).flatten()[:vector_dim]
        if len(v) < vector_dim:
            v = np.pad(v, (0, vector_dim - len(v)))
            
        v_norm = np.linalg.norm(v)
        if c_norm == 0 or v_norm == 0:
            scores.append(-1.0)
        else:
            cos_sim = np.dot(c_vec, v) / (c_norm * v_norm)
            scores.append(cos_sim)
            
    return np.argsort(scores)[-5:][::-1]

# ================= 2. 独立数据流解析器 =================

def extract_features_from_file(file_path, policy, desc_text, dataset_name):
    """独立的特征提取器，保证训练集和测试集物理隔离"""
    X_random, X_cosine, X_rl, y_labels = [], [], [], []
    
    # 🌟 新增：一个专门用来探底的计数器
    raw_string_labels = []
    
    def parse_label(lbl, d_name):
        lbl_str = str(lbl).strip().lower()
        if d_name == "LIAR-RAW":
            mapping = {
                'pants-fire': 0, 
                'false': 1, 
                'barely-true': 2, 
                'half-true': 3, 
                'mostly-true': 4, 
                'true': 5
            }
            return mapping.get(lbl_str, 1) 
        else:
            # 🌟 修改点：严格对齐师兄原版 reader5.py 的三分类映射
            mapping = {
                'false': 0, 
                'true': 1, 
                'half': 2
            }
            return mapping.get(lbl_str, 0) 

    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for item in tqdm(ijson.items(f, 'item'), desc=desc_text):
            cand_vecs = item["candidate_vectors"]
            num_cands = len(cand_vecs)
            
            if num_cands < 5: continue
                
            claim_vec = item["claim_vector"]
            
            # 🌟 记录最底层的真实字符串
            raw_str = str(item["ground_truth_label"]).strip().lower()
            raw_string_labels.append(raw_str)
            
            label = parse_label(item["ground_truth_label"], dataset_name)
            
            idx_random = get_random_top5(num_cands)
            feat_random = build_downstream_features(claim_vec, cand_vecs, idx_random)
            
            idx_cosine = get_cosine_top5(claim_vec, cand_vecs)
            feat_cosine = build_downstream_features(claim_vec, cand_vecs, idx_cosine)
            
            state_vec = build_state_vector(claim_vec, cand_vecs)
            idx_rl = get_rl_top5(policy, state_vec)
            feat_rl = build_downstream_features(claim_vec, cand_vecs, idx_rl)
            
            X_random.append(feat_random)
            X_cosine.append(feat_cosine)
            X_rl.append(feat_rl)
            y_labels.append(label)
            count += 1
            
    # 🌟 打印出底层的真相
    print(f"\n🕵️ [底层数据探底] {desc_text} 内部真实的字符串标签有: {Counter(raw_string_labels)}")
    return np.array(X_random), np.array(X_cosine), np.array(X_rl), np.array(y_labels)


# ================= 3. 主流程 =================

def main():
    parser = argparse.ArgumentParser(description="自适应 CQL 假新闻检测评测")
    parser.add_argument("--dataset", type=str, default="LIAR-RAW", choices=["LIAR-RAW", "RAWFC"])
    args = parser.parse_args()
    
    dataset_name = args.dataset 
    print(f"\n🚀 正在启动真实泛化评测流程，当前目标数据集: 【{dataset_name}】")

    setup_logger(log_dir="logs", prefix=f"evaluate_real_{dataset_name}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    print(f"🧠 正在加载自适应 CQL 模型权重 ({dataset_name})...")
    weight_name = f"{dataset_name}_adaptive_cql_policy_epoch_100.pth"
    weight_path = os.path.join("checkpoints", weight_name)
    
    if not os.path.exists(weight_path):
        legacy_weight_path = os.path.join("checkpoints", "adaptive_cql_policy_epoch_100.pth")
        if os.path.exists(legacy_weight_path):
            print(f"💡 自动兼容：未找到带前缀的新版权重，正在加载旧版默认权重...")
            weight_path = legacy_weight_path
        else:
            print(f"❌ 找不到权重文件！请检查 {dataset_name} 是否已经运行过训练脚本。")
            sys.exit(1)

    policy = load_adaptive_cql_policy(weight_path=weight_path)

    train_path = os.path.join("datasets", dataset_name, "rl_offline_buffer_train_features.json")
    test_path = os.path.join("datasets", dataset_name, "rl_offline_buffer_test_features.json")
    
    print("\n" + "-"*50)
    print("📖 第一阶段：提取【训练集】特征 (用于教导分类验证器)")
    Xr_train, Xc_train, Xrl_train, y_train = extract_features_from_file(train_path, policy, "处理 Train Set", dataset_name)

    print("\n" + "-"*50)
    print("📖 第二阶段：提取【测试集】特征 (用于未知的闭卷考试)")
    Xr_test, Xc_test, Xrl_test, y_test = extract_features_from_file(test_path, policy, "处理 Test Set", dataset_name)
    
    print("\n" + "="*50)
    print(f"📊 [最终映射检查] 训练集标签比例: {Counter(y_train)}")
    print(f"📊 [最终映射检查] 测试集标签比例: {Counter(y_test)}")
    print("="*50 + "\n")

    def evaluate_method(name, X_tr, y_tr, X_te, y_te):
        unique_classes = np.unique(y_tr) 
        class_indices = [np.where(y_tr == c)[0] for c in unique_classes]
        min_count = min([len(idx) for idx in class_indices])
        
        np.random.seed(42)
        balanced_indices = []
        for idx in class_indices:
            sampled_idx = np.random.choice(idx, min_count, replace=False)
            balanced_indices.extend(sampled_idx)
            
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        X_tr_balanced = X_tr[balanced_indices]
        y_tr_balanced = y_tr[balanced_indices]

        clf = get_proxy_classifier()
        clf.fit(X_tr_balanced, y_tr_balanced)
        
        y_pred = clf.predict(X_te)
        
        acc = accuracy_score(y_te, y_pred)
        p = precision_score(y_te, y_pred, average='macro', zero_division=0)
        r = recall_score(y_te, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
        
        print(f"   ➤ {name:<18} | Acc: {acc:.4f} | Prec: {p:.4f} | Recall: {r:.4f} | MacF1: {f1:.4f}")
        print(classification_report(y_te, y_pred, digits=4))
        
        return f1

    print("\n" + "="*65)
    print("🏆 最终基线对比实验结果 (真实泛化能力评估) 🏆")
    print("="*65)
    evaluate_method("Baseline: Random", Xr_train, y_train, Xr_test, y_test)
    evaluate_method("Baseline: Cosine", Xc_train, y_train, Xc_test, y_test)
    print("-" * 65)
    evaluate_method("Ours: Adaptive CQL", Xrl_train, y_train, Xrl_test, y_test)
    print("="*65)
    
    print("\n🎉 实验完成！")

if __name__ == '__main__':
    main()