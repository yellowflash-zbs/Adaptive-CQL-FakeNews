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

# ==========================================
# 1. 终极暴力破解：强行霸占路径解析的第一位 (彻底抛弃旧的 helpers.path_util)
# ==========================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ==========================================

# ================= 配置区 =================
# ⚠️ 确保这个 Key 是有额度的！
DEEPSEEK_API_KEY = "sk-59042b58fd304e24903c8a0f597921f4" 

client = OpenAI(
    api_key=DEEPSEEK_API_KEY, 
    base_url="https://api.deepseek.com"
)
# ==========================================

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def get_reward_from_llm(claim_idx, ground_truth, selected_sentences):
    evidence_text = "\n".join([f"{i+1}. {sent}" for i, sent in enumerate(selected_sentences)])
    
    prompt = f"""
    You are an expert fact-checker evaluating an AI agent's evidence selection.
    
    The AI agent was asked to select exactly 5 sentences to verify a news claim.
    The TRUE label of this claim is: '{ground_truth}'.
    
    Here are the 5 sentences the agent selected:
    {evidence_text}
    
    Please evaluate these 5 sentences and provide two scores:
    1. R_global (Global Reward): If these 5 sentences correctly guide a reader to the true label '{ground_truth}', give a score of 10. If they are misleading or irrelevant, give -10.
    2. R_fine (Fine-grained Reward): Rate the overall logical correlation and information density of these sentences from 0 to 5 (5 is best).
    
    Output strictly in the following JSON format:
    {{
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
                    {"role": "system", "content": "You are a helpful JSON-outputting assistant."},
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
                print(f"\n[Claim {claim_idx}] API 多次调用失败: {e}")
                return {"R_global": 0, "R_fine": 0}

def main():
    # ==========================================
    # 引入智能命令行开关
    # ==========================================
    parser = argparse.ArgumentParser(description="调用 DeepSeek 生成 RL 奖励信号")
    parser.add_argument("--dataset", type=str, default="LIAR-RAW", choices=["LIAR-RAW", "RAWFC"])
    args = parser.parse_args()
    dataset_name = args.dataset
    
    print(f"\n▶️ 开启大模型离线打分，当前目标数据集: 【{dataset_name}】\n")

    # 动态拼接路径
    input_path = os.path.join(project_root, "datasets", dataset_name, "rl_offline_buffer_train_features.json")
    output_path = os.path.join(project_root, "datasets", dataset_name, "rl_offline_buffer_with_rewards.jsonl") 
    
    if not os.path.exists(input_path):
        print(f"❌ 找不到输入文件: {input_path}\n请先运行第1步的 extract_features.py 提取文本特征！")
        return

    processed_claims = set()
    
    # --- 断点续传 ---
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
            
        pbar = tqdm(desc=f"{dataset_name} API 打分进度", initial=len(processed_claims), total=1612 if dataset_name == "RAWFC" else 10065) 
        
        for item in items_generator:
            claim_idx = item["claim_index"]
            
            if claim_idx in processed_claims:
                continue
                
            truth_label = item["ground_truth_label"]
            candidate_pool = item["candidate_sentences"]
            
            if len(candidate_pool) < 5:
                pbar.update(1)
                continue
                
            sampled_indices = random.sample(range(len(candidate_pool)), 5)
            selected_sentences = [candidate_pool[i] for i in sampled_indices]
            
            reward_scores = get_reward_from_llm(claim_idx, truth_label, selected_sentences)
            
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