# coding: utf-8
import os
import sys
import json
import time
import torch
import numpy as np
import ijson
import argparse
from tqdm import tqdm
from openai import OpenAI
from collections import Counter
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

from core.feature_utils import build_state_vector
from core.cql_agent import load_adaptive_cql_policy, get_rl_top5

import warnings
warnings.filterwarnings("ignore")

# ================= 配置区 =================
DEEPSEEK_API_KEY = "your_api_key_here" 
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
# ==========================================

# ================= 1. 核心抽取算法 (加入 Baseline) =================
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

# ================= 2. 大模型代理法官 (反偏见专业版) =================
def get_prediction_from_llm(claim_text, selected_sentences, dataset_name):
    """大模型终极法官：引入反重合偏见（Anti-Overlap Bias）的骨灰级 Prompt"""
    evidence_text = "\n".join([f"{i+1}. {sent}" for i, sent in enumerate(selected_sentences)])
    
    if dataset_name == "LIAR-RAW":
        options = "['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']"
    else:
        options = "['false', 'true', 'half']"

    # 🌟 核心突破：注入极度怀疑的专业事实核查逻辑
    prompt = f"""
    You are an elite, highly skeptical fact-checker. 
    Your task is to verify a news claim based STRICTLY on the provided evidence sentences.
    
    [NEWS CLAIM]: "{claim_text}"
    
    [RETRIEVED EVIDENCE]:
    {evidence_text}
    
    CRITICAL RULES FOR FACT-CHECKING (PREVENTING AI BIAS):
    1. DO NOT assume a claim is 'true' just because the evidence discusses the same topic, contains the same keywords, or merely quotes the claim.
    2. 'false': Look for explicit stance markers. If the evidence uses words like "debunked", "refuted", "untrue", "hoax", or explicitly contradicts the claim, output 'false'.
    3. 'half': If the evidence says the claim lacks context, is mixed, exaggerated, or only partly correct, output 'half'.
    4. 'true': ONLY output 'true' if the evidence provides absolute, explicit confirmation that the claim is 100% factual.
    
    Based on your logical deduction, classify the claim into exactly ONE of the following categories:
    {options}
    
    Output strictly in the following JSON format:
    {{
        "rationale": "Briefly explain why the evidence debunks, partially supports, or proves the claim...",
        "prediction": "the chosen category"
    }}
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a ruthless, logic-driven fact-checking JSON assistant."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # 极低温度，切断发散思维
                timeout=15 
            )
            result = json.loads(response.choices[0].message.content)
            return str(result.get("prediction", "")).strip().lower()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2) 
            else:
                print(f"\n[🚨 API 错误] {e}")
                return "unknown"

def parse_label_to_int(lbl_str, dataset_name):
    """【防塌方补丁】修改匹配优先级，防止 false 吞噬一切"""
    lbl_str = str(lbl_str).strip().lower()
    if dataset_name == "LIAR-RAW":
        # LIAR 的匹配逻辑
        if 'pants' in lbl_str or 'fire' in lbl_str: return 0
        elif 'barely' in lbl_str: return 2
        elif 'half' in lbl_str: return 3
        elif 'mostly' in lbl_str: return 4
        elif 'true' in lbl_str: return 5  # true 放在前面
        elif 'false' in lbl_str: return 1
        return 1 # default false
    else:
        # 🌟 修复点 2：RAWFC 的匹配逻辑，必须先匹配 half，再匹配 false/true
        if 'half' in lbl_str: return 2
        elif 'true' in lbl_str: return 1
        elif 'false' in lbl_str: return 0
        return 0 # default

def evaluate_results(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f" ➤ {name:<18} | Acc: {acc:.4f} | Prec: {p:.4f} | Recall: {r:.4f} | MacF1: {f1:.4f}")
    return f1

# ================= 3. 主流程 =================
def main():
    parser = argparse.ArgumentParser(description="大模型代理判决：自适应 CQL 基线对比")
    parser.add_argument("--dataset", type=str, default="RAWFC", choices=["LIAR-RAW", "RAWFC"])
    args = parser.parse_args()
    dataset_name = args.dataset 
    
    print(f"\n🚀 正在启动【大模型法官】多基线评测流水线，当前目标数据集: 【{dataset_name}】")
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    print(f"🧠 正在加载自适应 CQL 智能体权重...")
    weight_path = os.path.join("checkpoints", f"{dataset_name}_adaptive_cql_policy_epoch_100.pth")
    if not os.path.exists(weight_path):
        print(f"❌ 找不到权重文件！请确认 {weight_path} 存在。")
        sys.exit(1)
    policy = load_adaptive_cql_policy(weight_path=weight_path)

    test_path = os.path.join("datasets", dataset_name, "rl_offline_buffer_test_features.json")
    
    y_true_list = []
    y_pred_random = []
    y_pred_cosine = []
    y_pred_cql = []
    
    print("\n📖 开始大模型在线判卷 (包含三个基线，API调用量增至3倍，请耐心等待)...")
    
    with open(test_path, 'r', encoding='utf-8') as f:
        try:
            items_generator = ijson.items(f, 'item', use_float=True)
        except TypeError:
            items_generator = ijson.items(f, 'item')
            
        for item in tqdm(items_generator, desc="Evaluating Baselines on Test Set"):
            cand_vecs = item["candidate_vectors"]
            if len(cand_vecs) < 5: 
                continue
                
            claim_vec = item["claim_vector"]
            candidate_pool = item["candidate_sentences"]
            claim_text = item.get("claim_text", item.get("claim", "UNKNOWN CLAIM"))
            
            raw_truth = item["ground_truth_label"]
            y_true = parse_label_to_int(raw_truth, dataset_name)
            
            # 💡 基线 1：随机选取 5 句
            idx_random = get_random_top5(len(cand_vecs))
            sent_random = [candidate_pool[i] for i in idx_random]
            pred_random = parse_label_to_int(get_prediction_from_llm(claim_text, sent_random, dataset_name), dataset_name)
            
            # 💡 基线 2：传统余弦相似度选取 5 句
            idx_cosine = get_cosine_top5(claim_vec, cand_vecs)
            sent_cosine = [candidate_pool[i] for i in idx_cosine]
            pred_cosine = parse_label_to_int(get_prediction_from_llm(claim_text, sent_cosine, dataset_name), dataset_name)
            
            # 💡 我们的策略：自适应 CQL 选取 5 句
            state_vec = build_state_vector(claim_vec, cand_vecs)
            idx_rl = get_rl_top5(policy, state_vec)
            sent_rl = [candidate_pool[i] for i in idx_rl]
            pred_rl = parse_label_to_int(get_prediction_from_llm(claim_text, sent_rl, dataset_name), dataset_name)
            
            y_true_list.append(y_true)
            y_pred_random.append(pred_random)
            y_pred_cosine.append(pred_cosine)
            y_pred_cql.append(pred_rl)
            
    print("\n" + "="*65)
    print("🏆 最终基线对比实验结果 (大模型代理判决 LLM-as-a-Judge) 🏆")
    print("="*65)
    evaluate_results("Baseline: Random", y_true_list, y_pred_random)
    evaluate_results("Baseline: Cosine", y_true_list, y_pred_cosine)
    print("-" * 65)
    evaluate_results("Ours: Adaptive CQL", y_true_list, y_pred_cql)
    print("="*65)

if __name__ == '__main__':
    main()