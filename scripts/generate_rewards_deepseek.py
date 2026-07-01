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
# 从环境变量读取 API Key，避免把密钥提交到 GitHub。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY，再运行 generate_rewards_deepseek.py")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
# ==========================================

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def clean_spaced_text(text):
    text = str(text).strip()
    if "   " in text:
        words = text.split("   ")
        cleaned_words = [word.replace(" ", "") for word in words]
        return " ".join(cleaned_words)
    return text

# 🌟 新增 dataset_name 参数，让大模型教练理解不同数据集的标签体系
def get_reward_from_llm(claim_idx, claim_text, ground_truth, selected_sentences, dataset_name):
    clean_claim = clean_spaced_text(claim_text)
    clean_sentences = [clean_spaced_text(sent) for sent in selected_sentences]
    evidence_text = "\n".join([f"Sentence {i+1}: {sent}" for i, sent in enumerate(clean_sentences)])
    
    # 根据数据集动态提供真实标签的含义说明
    if dataset_name == "LIAR-RAW":
        label_definitions = """
    [LIAR 数据集真实标签含义参考]
    - pants-fire: 极其荒谬的谎言
    - false: 完全错误
    - barely-true: 勉强真实（只包含一丁点事实，忽略了关键事实）
    - half-true: 半真半假
    - mostly-true: 基本真实（大体正确，需细微补充）
    - true: 完全真实无误
        """
    else:
        label_definitions = """
    [RAWFC 数据集真实标签含义参考]
    - false: 声明不准确或虚假。
    - half: 声明半真半假，或存在夸大、脱离上下文。
    - true: 声明完全准确。
        """

    prompt = f"""
    [系统指令]
    你是一个严苛且绝对理性的假新闻事实核查教练。强化学习智能体挑选了 5 句话作为核查证据。
    你的任务是跳出“字面相似度”的陷阱，结合真实标签的含义，【逐句判断】这些证据的立场属性，并给出奖励分数。

    [NEWS CLAIM (新闻声明)]: "{clean_claim}"
    [TRUE LABEL (该声明最终的真实判决)]: "{ground_truth}"
    {label_definitions}

    [SELECTED EVIDENCE (选中的5句证据)]:
    {evidence_text}

    【打分规则】:
    1. 逐句分析立场并计分：
       - 强反驳 (refute)：句子直接提供了能揭穿虚假/片面部分的铁证 (+10分)
       - 强支持 (support)：句子直接提供了证实真实部分的铁证 (+10分)
       - 中立/部分相关 (neutral)：提到了相关实体，但没有给出明确的真假验证信息 (+2分)
       - 无关/废话 (irrelevant)：完全无关，或者纯粹重复声明且无增量信息 (-5分)
    
    2. 计算 R_global (全局奖励)：
       - 基础分：上述 5 句话的得分总和。
       - 🚨 对比式多样性惩罚：如果这 5 句话高度同质化（都在说同一句废话或提供重复视角），在基础分上额外扣除 10 分！好的证据链应该是多角度的。

    3. 给出 R_fine (细粒度奖励)：评估这组证据的整体逻辑密度 (0到5分)。

    请严格输出以下 JSON 格式：
    {{
        "sentence_stances": ["refute", "irrelevant", "support", "neutral", "irrelevant"],
        "rationale": "简短分析这5句话的立场分布以及是否同质化...",
        "R_global": 12,
        "R_fine": 3
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
                temperature=0.0,  
                timeout=15 
            )
            result_str = response.choices[0].message.content
            return json.loads(result_str)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2) 
            else:
                print(f"\n[Claim {claim_idx}] API 打分报错: {e}")
                return {
                    "sentence_stances": ["error"] * 5,
                    "rationale": "API Timeout or Error",
                    "R_global": -10, 
                    "R_fine": 0
                }

def main():
    parser = argparse.ArgumentParser(description="调用 DeepSeek 生成 RL 奖励信号 (自适应对比打分版)")
    parser.add_argument("--dataset", type=str, default="RAWFC", choices=["LIAR-RAW", "RAWFC"])
    args = parser.parse_args()
    dataset_name = args.dataset
    
    print(f"\n▶️ 开启大模型离线打分 (自适应双数据集模式)，当前目标数据集: 【{dataset_name}】\n")

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
            
            # 🌟 传入 dataset_name 参数，动态匹配真实标签含义
            reward_scores = get_reward_from_llm(claim_idx, claim_text, truth_label, selected_sentences, dataset_name)
            
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
        
    print(f"\n🎉 大功告成！{dataset_name} 的双向立场经验池数据均已打分完毕！\n保存在: {output_path}")

if __name__ == '__main__':
    main()
