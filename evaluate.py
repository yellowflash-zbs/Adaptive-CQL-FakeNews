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

# 🌟 终极优化 1：锁死所有随机种子，保证每次运行结果 100% 一致！
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ================= 配置区 =================
# 从环境变量读取 API Key，避免把密钥提交到 GitHub。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY，再运行 evaluate.py")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
# ==========================================

def clean_spaced_text(text):
    text = str(text).strip()
    if "   " in text:
        words = text.split("   ")
        cleaned_words = [word.replace(" ", "") for word in words]
        return " ".join(cleaned_words)
    return text

def get_random_top5(num_candidates):
    if num_candidates == 0: return []
    k = min(5, num_candidates)
    indices = np.arange(num_candidates)
    np.random.shuffle(indices)
    return indices[:k]

def get_cosine_top5(claim_vec, cand_vecs):
    num_candidates = len(cand_vecs)
    if num_candidates == 0: return []
    k = min(5, num_candidates)
    
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

def get_rl_top5_mmr(action_scores, cand_vecs, lambda_mmr=0.5):
    num_cands = len(cand_vecs)
    if num_cands == 0: return []
    k = min(5, num_cands)
    
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
    if not selected_sentences:
        evidence_text = "No evidence available."
    else:
        clean_claim = clean_spaced_text(claim_text)
        clean_sentences = [clean_spaced_text(sent) for sent in selected_sentences]
        evidence_text = "\n".join([f"{i+1}. {sent}" for i, sent in enumerate(clean_sentences)])
    
    clean_claim = clean_spaced_text(claim_text)
    
    # 🌟 核心优化：自适应双数据集 Prompt 切换
    if dataset_name == "LIAR-RAW":
        options = "['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']"
        calibration_text = """
    [用于校准你判决尺度的 LIAR 六分类标准]
    - pants-fire (极其荒谬的谎言): 声明完全不准确，且错得离谱。
    - false (错误): 声明不准确。
    - barely-true (勉强真实): 声明包含一点事实，但忽略了会给人留下不同印象的关键事实。
    - half-true (半真半假): 声明部分准确，但遗漏了重要细节或断章取义。
    - mostly-true (基本真实): 声明准确，但需要澄清或补充信息。
    - true (完全真实): 声明准确且没有遗漏任何重要内容。
        """
    else:
        options = "['false', 'true', 'half']"
        calibration_text = """
    [用于校准你判决尺度的经典案例 (RAWFC 三分类)]
    - 声明: "新税法降低了所有人的税收。" | 证据: "该法律降低了中产阶级的税收，但提高了前1%富人的税收。" | 判决: half
    - 声明: "市长贪污了100万美元。" | 证据: "权威审计报告显示，所有市政资金的账目都完全清楚，未见异常。" | 判决: false
        """

    prompt = f"""
    你是一位顶尖的、逻辑极其严密的假新闻核查法官，正在审理一桩复杂的案件。
    你的任务是根据法庭上双方提交的证据，来综合裁决一条新闻声明的真伪。
    {calibration_text}

    === 现在请审理以下案件 ===
    
    [新闻声明]: "{clean_claim}"
    
    [法庭提交的证据]:
    {evidence_text}
    
    步骤 1: 交叉盘问 (Cross-Examination)。
    绝不能把这些证据当成单方面的话术来轻信。你必须主动寻找证据中的【冲突】和【视角差异】。 
    - 证据中是否有强烈【支持】该声明的铁证？
    - 证据中是否有强烈【反驳】该声明的铁证？
    - 如果证据之间互相矛盾、指向不同，请像法官一样权衡它们的可靠性，并明确对比“正方”与“反方”的逻辑。
    (注意：如果提供的证据完全无关或无法证明声明，请根据你的推理，推断最可能的分类，而不是随意乱猜)。
    
    步骤 2: 基于你交叉盘问后的综合判断，将该声明严格归类为以下选项中的【唯一】一个：
    {options}
    
    请严格按照以下 JSON 格式输出（注意：为了系统能正确读取，JSON 的键名必须保持英文）：
    {{
        "step_by_step_analysis": "请写下你交叉盘问正反方证据的详细推理过程...",
        "prediction": "填入你最终选定的类别"
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
                temperature=0.0,  
                timeout=20 
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2) 
            else:
                print(f"  [Error] LLM API 评测请求失败: {e}")
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
    
    print(f"\n🚀 正在启动【大模型法官】多基线评测流水线 (满血自适应版)，当前数据集: 【{dataset_name}】")
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
    
    random.shuffle(all_items)
    
    try:
        for item in tqdm(all_items, desc="Evaluating Baselines on Test Set"):
            raw_cand_vecs = item["candidate_vectors"]
            raw_candidate_pool = item["candidate_sentences"]
            claim_vec = item["claim_vector"]
            claim_text = item.get("claim_text", item.get("claim", "UNKNOWN CLAIM"))
            raw_truth = item["ground_truth_label"]
            
            cand_vecs = []
            candidate_pool = []
            seen_texts = set()
            
            for vec, text in zip(raw_cand_vecs, raw_candidate_pool):
                clean_t = clean_spaced_text(text).lower()
                if clean_t not in seen_texts and len(clean_t) > 10:  
                    seen_texts.add(clean_t)
                    cand_vecs.append(vec)
                    candidate_pool.append(text)
            
            if len(cand_vecs) == 0:
                continue
            
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
            action, _ = policy.get_action(state_vec) 
            action_scores = action[:len(cand_vecs)] 
            
            idx_rl = get_rl_top5_mmr(action_scores, cand_vecs, lambda_mmr=0.5)
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
