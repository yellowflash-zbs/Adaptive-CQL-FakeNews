# coding: utf-8
import os
import sys

# ==========================================
# 1. 强行霸占路径解析的第一位
# ==========================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

model_path = os.path.join(project_root, 'model')
if model_path not in sys.path:
    sys.path.insert(0, model_path)

import argparse
import json

BATCH_SIZE = 1 
REPORT_EACH_CLAIM = 30

def get_label_str(label_idx, label_dict):
    for k, v in label_dict.items():
        if v == label_idx:
            return k
    return "unknown"

def main():
    parser = argparse.ArgumentParser(description="抽取文本特征向量")
    parser.add_argument("--dataset", type=str, default="RAWFC", choices=["LIAR-RAW", "RAWFC"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "val"])
    parser.add_argument("--limit", type=int, default=0, help="调试用：只抽取前 N 条，0 表示全量")
    parser.add_argument("--output-suffix", default="", help="输出文件后缀，例如 debug10")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有特征文件")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    parser.add_argument("--cuda-visible-devices", default="", help="指定可见 GPU，例如 0；留空则不修改")
    parser.add_argument("--hf-endpoint", default=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))
    args = parser.parse_args()
    dataset_name = args.dataset
    split_name = args.split 

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    import helpers.reader5 as reader5_module
    from helpers.reader5 import myDataset
    from model_exp_fc5 import ExplainFC
    
    print(f"\n初始化模型与加载数据，当前目标数据集: 【{dataset_name}】 | 数据划分: 【{split_name}】")
    
    if dataset_name == "LIAR-RAW":
        N_TAGS = 6
        LABEL_IDS = {"pants-fire": 0, "false": 1, "barely-true": 2, "half-true": 3, "mostly-true": 4, "true": 5}
    else: 
        N_TAGS = 3
        LABEL_IDS = {"false": 0, "half": 1, "true": 2}  
        
    # 双重保险，确保底层类使用正确的字典
    reader5_module.LABEL_IDS = LABEL_IDS

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("你指定了 --device cuda，但当前环境没有可用 CUDA。")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    
    model = ExplainFC(
        hidden_size=384, lstm_layers=1, n_tags=N_TAGS, char_feat_dim=0,
        bidirectional=True, n_embeddings=1000, embedding_dim=300,
        lm_embedding_dim=768, freeze=False, max_doc_num=REPORT_EACH_CLAIM, source_dim=20
    ).to(device)
    model.eval() 

    if dataset_name == "LIAR-RAW":
        data_path = os.path.join(project_root, "datasets", dataset_name, f"{split_name}.json")
    else:
        data_path = os.path.join(project_root, "datasets", dataset_name, split_name)

    if not os.path.exists(data_path):
        print(f"找不到原始数据路径: {data_path}")
        return
        
    dataset = myDataset(data_path, report_each_claim=REPORT_EACH_CLAIM)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.my_collate)

    rl_buffer_data = []

    print(f"开始提取【{split_name}】特征数据，请耐心等待...")
    with torch.no_grad(): 
        for i, (oracle_ids, label_ids, raw_text_dict, lm_ids_dict) in enumerate(tqdm(train_loader, desc=f"{split_name} 提取进度")):
            if args.limit > 0 and i >= args.limit:
                break
            
            # 1. 提取被完美修复的标签
            true_label_idx = label_ids[0].item()
            true_label_str = get_label_str(true_label_idx, LABEL_IDS)

            candidate_sentences = []
            for doc in raw_text_dict['_SRC_TOK'][0]: 
                for sentence_tokens in doc:
                    sentence = " ".join(sentence_tokens)
                    if sentence.strip():
                        candidate_sentences.append(sentence)

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

            try:
                if '_CLAIM_TOK' in raw_text_dict:
                    claim_text = " ".join(raw_text_dict['_CLAIM_TOK'][0])
                elif 'claim' in raw_text_dict:
                    claim_text = str(raw_text_dict['claim'])
                else:
                    claim_text = "UNKNOWN_CLAIM"
            except Exception:
                claim_text = "UNKNOWN_CLAIM"

            data_point = {
                "claim_index": i,
                "ground_truth_label": true_label_str,
                "claim_text": claim_text,
                "claim_vector": claim_vector,
                "candidate_vectors": src_vectors,
                "candidate_sentences": candidate_sentences
            }
            rl_buffer_data.append(data_point)

    output_suffix = args.output_suffix
    if args.limit > 0 and not output_suffix:
        output_suffix = f"debug{args.limit}"
    suffix = f"_{output_suffix}" if output_suffix else ""
    output_path = os.path.join(project_root, "datasets", dataset_name, f"rl_offline_buffer_{split_name}_features{suffix}.json")
    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，请加 --overwrite 或换 --output-suffix: {output_path}")
    print(f"正在保存至 {output_path} ... ")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rl_buffer_data, f, ensure_ascii=False)
    print(f"保存成功，共写入 {len(rl_buffer_data)} 条样本。")

if __name__ == '__main__':
    main()
