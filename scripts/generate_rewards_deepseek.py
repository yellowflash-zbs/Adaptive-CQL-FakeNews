# coding: utf-8
import os
import sys
import json
import random
import time
import argparse
from decimal import Decimal
from openai import OpenAI
from tqdm import tqdm
import ijson 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ================= 配置区 =================
# 🚨 填上你真实的 API Key！
DEEPSEEK_API_KEY = "your_api_key_here" 
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
# ==========================================

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

# 🌟 文本脱水净化器：确保打分导师也能看到干净的文本，不被乱码蒙蔽
def clean_spaced_text(text):
    text = str(text).strip()
    if "   " in text:
        words = text.split("   ")
        cleaned_words = [word.replace(" ", "") for word in words]
        return " ".join(cleaned_words)
    return text

def get_reward_from_llm(claim_idx, claim_text, ground_truth, selected_sentences):
    # 🌟 在这里进行脱水清洗！
    clean_claim = clean_spaced_text(claim_text)
    clean_sentences = [clean_spaced_text(sent) for sent in selected_sentences]
    evidence_text = "\n".join([f"{i+1}. {sent}" for i, sent in enumerate(clean_sentences)])
    
    # 🌟 严苛的导师 Prompt，包含多样性惩罚 (Diversity Penalty)
    prompt = f"""
    You are an elite, highly skeptical AI trainer evaluating an agent's evidence selection.
    The agent selected 5 sentences to verify the following news claim.
    
    [NEWS CLAIM]: "{clean_claim}"
    [TRUE LABEL OF CLAIM]: '{ground_truth}'
    
    [SELECTED EVIDENCE]:
    {evidence_text}
    
    CRITICAL RULES FOR EVALUATION:
    1. DO NOT give high scores just because the evidence contains the same entities, keywords, or topics as the claim.
    2. The evidence must logically, explicitly, and substantially prove why the claim's true label is '{ground_truth}'.
    3. If the evidence is useless background noise, mere overlap, or contradicts the true label, you must ruthlessly penalize it.
    4. 🚨 DIVERSITY PENALTY: You MUST check if the 5 sentences are highly repetitive. If they lack diversity, give -5 or -10 R_global. Good evidence pieces together a diverse picture.
    
    Please evaluate and provide two scores:
    1. R_global (Global Reward): 
       - Give +10 ONLY IF the evidence provides explicit, DIVERSE, and critical proof aligning with the true label.
       - Give -10 if the evidence is misleading, purely useless keyword overlap, or HIGHLY REPETITIVE.
       - Give 0 if it's completely irrelevant but not harmful.
    2. R_fine (Fine-grained Reward): Rate the evidence's true logical density and diversity on a scale of 0 to 5.
    
    Output strictly in the following JSON format:
    {{
        "rationale": "Briefly analyze if the evidence provides diverse, actual logical proof...",
        "R_global": 10,
        "R_fine": 4
    }}
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a strict, logic-driven AI reinforcement learning trainer."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=15 
            )
            result_str = response.choices[0].message.content
            return json.loads(result_str)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2) 
            else:
                print(f"\n[Claim {claim_idx}] API 打分报错: {e}")
                return {"R_global": -10, "R_fine": 0}

def main():
    parser = argparse.ArgumentParser(description="调用 DeepSeek 生成 RL 奖励信号")
    parser.add_argument("--dataset", type=str, default="RAWFC", choices=["LIAR-RAW", "RAWFC"])
    args = parser.parse_args()
    dataset_name = args.dataset
    
    print(f"\n▶️ 开启大模型离线打分 (带文本净化器)，当前目标数据集: 【{dataset_name}】\n")

    input_path = os.path.join(project_root, "datasets", dataset_name, "rl_offline_buffer_train_features.json")
    output_path = os.path.join(project_root, "datasets", dataset_name, "rl_offline_buffer_with_rewards.jsonl") 
    
    if not os.path.exists(input_path):
        print(f"❌ 找不到输入文件: {input_path}")
        return

    processed_claims = set()
    
    if os.path.exists(output_path):
        print(f"🔍 发现已存在的结果文件，正在极速读取进度...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    processed_claims.add(record["claim_index"])
        print(f"✅ 已成功恢复 {len(processed_claims)} 条历史打分记录！将跳过这些数据。")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out:
        
        try:
            items_generator = ijson.items(f_in, 'item', use_float=True)
        except TypeError:
            items_generator = ijson.items(f_in, 'item')
            
        total_items = 10065 if dataset_name == "LIAR-RAW" else 1612
        pbar = tqdm(desc=f"{dataset_name} API 打分进度", initial=len(processed_claims), total=total_items) 
        
        for item in items_generator:
            claim_idx = item["claim_index"]
            if claim_idx in processed_claims:
                continue
                
            truth_label = item["ground_truth_label"]
            candidate_pool = item["candidate_sentences"]
            claim_text = item.get("claim_text", item.get("claim", "UNKNOWN CLAIM")) 
            
            if len(candidate_pool) < 5:
                pbar.update(1)
                continue
                
            if random.random() < 0.5:
                sampled_indices = random.sample(range(len(candidate_pool)), 5)
            else:
                sampled_indices = list(range(5)) 
                
            selected_sentences = [candidate_pool[i] for i in sampled_indices]
            
            reward_scores = get_reward_from_llm(claim_idx, claim_text, truth_label, selected_sentences)
            
            alpha, beta = 1.0, 1.0
            total_reward = alpha * reward_scores.get("R_global", 0) + beta * reward_scores.get("R_fine", 0)
            
            experience = {
                "claim_index": claim_idx,
                "state_claim_vector": item["claim_vector"],
                "state_candidate_vectors": item["candidate_vectors"],
                "action_selected_indices": sampled_indices, 
                "reward_scores": reward_scores,             
                "reward_total": total_reward                
            }
            
            f_out.write(json.dumps(experience, ensure_ascii=False, cls=DecimalEncoder) + '\n')
            f_out.flush() 
            pbar.update(1)
            
        pbar.close()
        
    print(f"\n🎉 大功告成！{dataset_name} 的经验池数据均已打分完毕！\n保存在: {output_path}")

if __name__ == '__main__':
    main()