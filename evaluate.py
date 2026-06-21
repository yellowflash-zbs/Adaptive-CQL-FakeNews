# coding: utf-8
import os
import sys
import json
import time
import torch
import random 
import numpy as np
import ijson
import argparse
from tqdm import tqdm
from openai import OpenAI
from collections import Counter
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

from core.feature_utils import build_state_vector
from core.cql_agent import load_adaptive_cql_policy, get_rl_top5
from core.logger import setup_logger

import warnings
warnings.filterwarnings("ignore")

# ================= 配置区 =================
# 🚨 填入你的真实 API Key！
DEEPSEEK_API_KEY = "your_api_key_here" 
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
# ==========================================

def clean_spaced_text(text):
    text = str(text).strip()
    if "   " in text:
        words = text.split("   ")
        cleaned_words = [word.replace(" ", "") for word in words]
        return " ".join(cleaned_words)
    return text

# 🌟 修复：自适应数量，句子不够 5 句就全拿
def get_random_top5(num_candidates):
    if num_candidates == 0: return []
    k = min(1, num_candidates)
    indices = np.arange(num_candidates)
    np.random.shuffle(indices)
    return indices[:k]

# 🌟 修复：自适应数量
def get_cosine_top5(claim_vec, cand_vecs):
    num_candidates = len(cand_vecs)
    if num_candidates == 0: return []
    k = min(1, num_candidates)
    
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
    return np.argsort(scores)[-k:][::-1]

# 🌟 修复：自适应数量，完美避免报错
def get_rl_top5_mmr(action_scores, cand_vecs, lambda_mmr=0.6):
    num_cands = len(cand_vecs)
    if num_cands == 0: return []
    k = min(1, num_cands)
    
    vector_dim = 768
    unselected = list(range(num_cands))
    selected = []
    
    def get_safe_vec(cv):
        v = np.array(cv, dtype=np.float32).flatten()[:vector_dim]
        if len(v) < vector_dim:
            v = np.pad(v, (0, vector_dim - len(v)))
        return v

    first_idx = unselected[np.argmax(action_scores)]
    selected.append(first_idx)
    unselected.remove(first_idx)
    
    for _ in range(k - 1):
        if not unselected: break
        best_mmr_score = -float('inf')
        best_idx = -1
        for idx in unselected:
            rl_score = action_scores[idx]
            max_sim = 0
            v_cand = get_safe_vec(cand_vecs[idx]) 
            n_cand = np.linalg.norm(v_cand)
            if n_cand > 0:
                for sel_idx in selected:
                    v_sel = get_safe_vec(cand_vecs[sel_idx]) 
                    n_sel = np.linalg.norm(v_sel)
                    if n_sel > 0:
                        sim = np.dot(v_cand, v_sel) / (n_cand * n_sel)
                        max_sim = max(max_sim, sim)
            
            mmr_score = lambda_mmr * rl_score - (1.0 - lambda_mmr) * max_sim
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx
                
        selected.append(best_idx)
        unselected.remove(best_idx)
        
    return selected

def get_prediction_from_llm(claim_text, selected_sentences, dataset_name):
    # 如果实在没有句子，大模型只能盲猜
    if not selected_sentences:
        evidence_text = "No evidence available."
    else:
        clean_claim = clean_spaced_text(claim_text)
        clean_sentences = [clean_spaced_text(sent) for sent in selected_sentences]
        evidence_text = "\n".join([f"{i+1}. {sent}" for i, sent in enumerate(clean_sentences)])
    
    clean_claim = clean_spaced_text(claim_text)
    
    if dataset_name == "LIAR-RAW":
        options = "['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']"
    else:
        options = "['false', 'true', 'half']"

    prompt = f"""
    You are an elite, highly skeptical fact-checker. 
    Your task is to verify a news claim based STRICTLY on the provided evidence sentences.
    
    [CRITICAL EXAMPLES FOR CALIBRATION]
    - Claim: "The new tax law lowers taxes for everyone."
    - Evidence: "The law lowers taxes for the middle class, but raises them for the top 1%."
    - Prediction: half
    
    - Claim: "The mayor stole $1 million."
    - Evidence: "Audit shows all city funds are fully accounted for."
    - Prediction: false
    
    - Claim: "Water boils at 100C."
    - Evidence: "Scientific consensus proves water boils at 100C at sea level."
    - Prediction: true

    === NOW EVALUATE THE FOLLOWING ===
    
    [NEWS CLAIM]: "{clean_claim}"
    
    [RETRIEVED EVIDENCE]:
    {evidence_text}
    
    STEP 1: Analyze the evidence step-by-step. Does it fully support, partially support, or explicitly contradict the claim? Is there enough context?
    STEP 2: Based on your analysis, classify the claim into exactly ONE of the following categories:
    {options}
    
    Output strictly in the following JSON format:
    {{
        "step_by_step_analysis": "Your detailed reasoning here...",
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
                temperature=0,  
                timeout=20 
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2) 
            else:
                return {"prediction": "unknown", "step_by_step_analysis": str(e)}

def parse_label_to_int(lbl_str, dataset_name):
    lbl_str = str(lbl_str).strip().lower()
    if dataset_name == "LIAR-RAW":
        if 'pants' in lbl_str or 'fire' in lbl_str: return 0
        elif 'false' in lbl_str or 'fake' in lbl_str: return 1
        elif 'barely' in lbl_str: return 2
        elif 'half' in lbl_str: return 3
        elif 'mostly' in lbl_str: return 4
        elif 'true' in lbl_str or 'real' in lbl_str: return 5  
        return -1 
    else:
        if 'half' in lbl_str: return 2
        elif 'true' in lbl_str or 'real' in lbl_str: return 1
        elif 'false' in lbl_str or 'fake' in lbl_str: return 0
        return -1 

def evaluate_results(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f" ➤ {name:<18} | Acc: {acc:.4f} | Prec: {p:.4f} | Recall: {r:.4f} | MacF1: {f1:.4f}")
    return f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="RAWFC", choices=["LIAR-RAW", "RAWFC"])
    args = parser.parse_args()
    dataset_name = args.dataset 
    
    print(f"\n🚀 正在启动【大模型法官】多基线评测流水线 (满血无截断版)，当前数据集: 【{dataset_name}】")
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    setup_logger(log_dir="logs", prefix=f"evaluate_llm_judge_{dataset_name}")

    weight_path = os.path.join("checkpoints", f"{dataset_name}_adaptive_cql_policy_epoch_100.pth")
    if not os.path.exists(weight_path):
        print(f"❌ 找不到权重文件！请确认 {weight_path} 存在. ")
        sys.exit(1)
    policy = load_adaptive_cql_policy(weight_path=weight_path)

    test_path = os.path.join("datasets", dataset_name, "rl_offline_buffer_test_features.json")
    
    y_true_list, y_pred_random, y_pred_cosine, y_pred_cql = [], [], [], []
    case_studies = [] 
    
    with open(test_path, 'r', encoding='utf-8') as f:
        all_items = json.load(f)
    
    random.seed(42) 
    random.shuffle(all_items)
    
    try:
        for item in tqdm(all_items, desc="Evaluating Baselines on Test Set"):
            cand_vecs = item["candidate_vectors"]
            # 🚨 终极修复：彻底删除了 if len(cand_vecs) < 5: continue ！！！
                
            claim_vec = item["claim_vector"]
            candidate_pool = item["candidate_sentences"]
            claim_text = item.get("claim_text", item.get("claim", "UNKNOWN CLAIM"))
            raw_truth = item["ground_truth_label"]
            
            y_true = parse_label_to_int(raw_truth, dataset_name)
            if y_true == -1:
                continue 
            
            idx_random = get_random_top5(len(cand_vecs))
            sent_random = [candidate_pool[i] for i in idx_random] if len(idx_random) > 0 else []
            dict_random = get_prediction_from_llm(claim_text, sent_random, dataset_name)
            pred_random = parse_label_to_int(dict_random.get("prediction", "unknown"), dataset_name)
            
            idx_cosine = get_cosine_top5(claim_vec, cand_vecs)
            sent_cosine = [candidate_pool[i] for i in idx_cosine] if len(idx_cosine) > 0 else []
            dict_cosine = get_prediction_from_llm(claim_text, sent_cosine, dataset_name)
            pred_cosine = parse_label_to_int(dict_cosine.get("prediction", "unknown"), dataset_name)
            
            state_vec = build_state_vector(claim_vec, cand_vecs)
            # 即使小于5句，神经网络依然能输出分数，我们按切片取即可
            action, _ = policy.get_action(state_vec) 
            action_scores = action[:len(cand_vecs)] 
            
            idx_rl = get_rl_top5_mmr(action_scores, cand_vecs, lambda_mmr=1.0   )
            sent_rl = [candidate_pool[i] for i in idx_rl] if len(idx_rl) > 0 else []
            dict_rl = get_prediction_from_llm(claim_text, sent_rl, dataset_name)
            pred_rl = parse_label_to_int(dict_rl.get("prediction", "unknown"), dataset_name)
            
            y_true_list.append(y_true)
            y_pred_random.append(pred_random if pred_random != -1 else 0)
            y_pred_cosine.append(pred_cosine if pred_cosine != -1 else 0)
            y_pred_cql.append(pred_rl if pred_rl != -1 else 0)
            
            case_studies.append({
                "claim_text": clean_spaced_text(claim_text),
                "ground_truth_label": raw_truth,
                "llm_prediction_for_rl": dict_rl.get("prediction", "unknown"),
                "llm_reasoning": dict_rl.get("step_by_step_analysis", "No reasoning provided"),
                "rl_selected_evidence": [clean_spaced_text(s) for s in sent_rl]
            })
            
    except KeyboardInterrupt:
        print("\n\n🛑 收到 Ctrl+C 中断指令！正在紧急抢救...")
        
    if len(case_studies) > 0:
        log_path = os.path.join("logs", f"case_study_{dataset_name}.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(case_studies, f, ensure_ascii=False, indent=4)
        print(f"\n🕵️ 成功抢救了 {len(case_studies)} 条数据！")
        
    print("\n" + "="*65)
    print(f"🏆 最终基线对比 (严格纯净三分类/六分类, 剔除脏数据, 有效样本量: {len(y_true_list)}) 🏆")
    print("="*65)
    evaluate_results("Baseline: Random", y_true_list, y_pred_random)
    evaluate_results("Baseline: Cosine", y_true_list, y_pred_cosine)
    print("-" * 65)
    evaluate_results("Ours: Adaptive CQL", y_true_list, y_pred_cql)
    print("="*65)

if __name__ == '__main__':
    main()