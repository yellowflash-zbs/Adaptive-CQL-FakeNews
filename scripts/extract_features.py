# coding: utf-8
import os
import sys

# ==========================================
# 1. 终极暴力破解：强行霸占路径解析的第一位
# ==========================================
# 获取当前文件所在目录的上一级绝对路径 (也就是 Adaptive-CQL-FakeNews)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 强行插到环境变量列表的【第 0 位】！优先级最高！
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 把 model 文件夹也强行插到前面，让它能找到 model_exp_fc5
model_path = os.path.join(project_root, 'model')
if model_path not in sys.path:
    sys.path.insert(0, model_path)
# ==========================================

import json
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 现在 Python 绝对无法拒绝下面这两个导入了
from helpers.reader5 import myDataset
from model_exp_fc5 import ExplainFC

# ==========================================
# 2. 强行踢开本地网络代理，直连 HF 国内镜像源
# ==========================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
# ==========================================

BATCH_SIZE = 1 # 务必设为1
REPORT_EACH_CLAIM = 30

def get_label_str(label_idx, label_dict):
    for k, v in label_dict.items():
        if v == label_idx:
            return k
    return "unknown"

def main():
    # ==========================================
    # 3. 引入智能命令行开关
    # ==========================================
    parser = argparse.ArgumentParser(description="抽取文本特征向量")
    parser.add_argument("--dataset", type=str, default="LIAR-RAW", choices=["LIAR-RAW", "RAWFC"])
    # 🌟 新增：支持自由切换 train / test / val 集合
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "val"], help="选择提取训练集还是测试集")
    args = parser.parse_args()
    dataset_name = args.dataset
    split_name = args.split 
    
    print(f"\n🚀 初始化模型与加载数据，当前目标数据集: 【{dataset_name}】 | 数据划分: 【{split_name}】")
    
    # ==========================================
    # 4. 智能切换分类标签 (极其重要！)
    # ==========================================
    if dataset_name == "LIAR-RAW":
        N_TAGS = 6
        LABEL_IDS = {"pants-fire": 0, "false": 1, "barely-true": 2, "half-true": 3, "mostly-true": 4, "true": 5}
    else:  # RAWFC
        N_TAGS = 3
        LABEL_IDS = {"false": 0, "half": 1, "true": 2}  
        
    # 核心越权修复：强行“黑”进底层模块，替换它的死板字典！
    import helpers.reader5 as reader5_module
    reader5_module.label_ids = LABEL_IDS
    # ==========================================
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 实例化模型
    model = ExplainFC(
        hidden_size=384, lstm_layers=1, n_tags=N_TAGS, char_feat_dim=0,
        bidirectional=True, n_embeddings=1000, embedding_dim=300,
        lm_embedding_dim=768, freeze=False, max_doc_num=REPORT_EACH_CLAIM, source_dim=20
    ).to(device)
    model.eval() 

    # ==========================================
    # 5. 动态加载对应数据集的数据 (顺应底层的胃口)
    # ==========================================
    if dataset_name == "LIAR-RAW":
        # LIAR-RAW 读的是单体 JSON 文件 (例如 test.json)
        data_path = os.path.join(project_root, "datasets", dataset_name, f"{split_name}.json")
    else:
        # RAWFC 会动态读取指定的文件夹 (例如 test 文件夹)
        data_path = os.path.join(project_root, "datasets", dataset_name, split_name)

    if not os.path.exists(data_path):
        print(f"❌ 找不到原始数据路径: {data_path}")
        return
        
    # 直接把正确的路径喂给底层数据集类
    dataset = myDataset(data_path, report_each_claim=REPORT_EACH_CLAIM)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.my_collate)

    rl_buffer_data = []

    print(f"⏳ 开始提取【{split_name}】特征数据，请耐心等待...")
    with torch.no_grad(): 
        for i, (oracle_ids, label_ids, raw_text_dict, lm_ids_dict) in enumerate(tqdm(train_loader, desc=f"{split_name} 提取进度")):
            
            # 1. 提取标签
            true_label_idx = label_ids[0].item()
            true_label_str = get_label_str(true_label_idx, LABEL_IDS)

            # 2. 提取并拼接原始候选文本句子
            candidate_sentences = []
            for doc in raw_text_dict['_SRC_TOK'][0]: 
                for sentence_tokens in doc:
                    sentence = " ".join(sentence_tokens)
                    if sentence.strip():
                        candidate_sentences.append(sentence)

            # 3. 提取特征向量
            claim_ids = lm_ids_dict['claim_ids']
            src_ids = lm_ids_dict['src_ids']
            
            claim_repr = [model.bert_embedding(_claim.to(device)).last_hidden_state[:,0,:] for _claim in claim_ids]
            claim_vector = claim_repr[0].cpu().numpy().tolist()

            src_vectors = []
            for _src in src_ids:
                MAX_SENTENCES = 60 
                _src_truncated = _src[:MAX_SENTENCES]
                sent_repr_list = [model.bert_embedding(s.to(device)).last_hidden_state[:,0,:] for s in _src_truncated]
                for tensor in sent_repr_list:
                     src_vectors.append(tensor.cpu().numpy().tolist())

            # 🌟 新增：尝试从 raw_text_dict 中拼凑出原始的 claim 文本
            try:
                # 假设 reader5 存的是 token 列表，比如 ['Says', 'the', 'Annies', 'List', ...]
                if '_CLAIM_TOK' in raw_text_dict:
                    claim_text = " ".join(raw_text_dict['_CLAIM_TOK'][0])
                elif 'claim' in raw_text_dict:
                    claim_text = str(raw_text_dict['claim'])
                else:
                    # 如果都不是，先随便塞个占位符，等报错了我们再看它到底叫啥名字
                    claim_text = "UNKNOWN_CLAIM"
            except Exception:
                claim_text = "UNKNOWN_CLAIM"

            # 4. 打包并存入列表 (🌟 把 claim_text 加进去)
            data_point = {
                "claim_index": i,
                "ground_truth_label": true_label_str,
                "claim_text": claim_text,  # <--- 极其关键的这一行！
                "claim_vector": claim_vector,
                "candidate_vectors": src_vectors,
                "candidate_sentences": candidate_sentences
            }
            rl_buffer_data.append(data_point)

    # 🌟 动态保存路径：文件名会根据 split_name 自动变更为 test_features
    output_path = os.path.join(project_root, "datasets", dataset_name, f"rl_offline_buffer_{split_name}_features.json")
    print(f"💾 正在保存至 {output_path} ... ")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rl_buffer_data, f, ensure_ascii=False)
    print(f"🎉 保存成功！{dataset_name} ({split_name} 集合) 的特征数据已经搞定了！")

if __name__ == '__main__':
    main()