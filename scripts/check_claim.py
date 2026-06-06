# coding: utf-8
"""
案例分析与调试探针 (Case Study & Debugging)
用于通过 Claim Index 钻取离线 JSON 数据库，将晦涩的特征向量还原为人类可读的英文原文。
"""
import os
import sys
import ijson

# 动态获取项目根目录，确保不论在哪个盘符运行都不会报错
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def main():
    # 自动指向重构后的 datasets 文件夹
    raw_data_path = os.path.join(project_root, "datasets", "LIAR-RAW", "rl_offline_buffer_train_features.json")
    
    # 💡 你可以随时修改这里的 target_index，去扒出任何一条你想看的新闻原文！
    target_index = 0 
    
    print(f"📖 正在通过流式读取寻找 Claim Index {target_index} 的明文内容...\n")
    print(f"📁 目标数据文件: {raw_data_path}")
    
    if not os.path.exists(raw_data_path):
        print(f"\n❌ 错误: 找不到数据文件！\n请检查是否已将 .json 文件正确放入了 {raw_data_path}")
        return

    with open(raw_data_path, 'r', encoding='utf-8') as f:
        # 兼容不同版本的 ijson 库
        try:
            items_generator = ijson.items(f, 'item', use_float=True)
        except TypeError:
            items_generator = ijson.items(f, 'item')
            
        for item in items_generator:
            if item.get("claim_index") == target_index:
                print("\n" + "="*70)
                print(f"🏆 成功锁定 Claim Index: {target_index}")
                print("-" * 70)
                
                # 1. 打印真实标签
                if "ground_truth_label" in item:
                    print(f"🏷️ 【真实标签 (Ground Truth)】: {item['ground_truth_label']}\n")
                
                # 2. 智能寻找并打印新闻原文
                for key in ["claim", "claim_text", "statement", "text", "news_text"]:
                    if key in item:
                        print(f"🗣️ 【新闻论断原文 ({key})】:\n{item[key]}\n")
                        break
                
                # 3. 打印部分候选证据池 (只打前3句，避免刷屏)
                if "candidate_sentences" in item:
                    print(f"📚 【候选证据池 (节选前 3 句)】:")
                    for i, sent in enumerate(item["candidate_sentences"][:3]):
                        print(f"  [{i+1}] {sent}")
                        
                print("="*70)
                return # 找到后直接结束运行
                
        print(f"\n⚠️ 遍历完整个文件，未找到 Claim Index 为 {target_index} 的数据。")

if __name__ == '__main__':
    main()