# coding: utf-8
"""
特征拼接工厂 (Feature Utilities)
主要负责将文本特征向量进行维度对齐、截断、填充，并拼接成强化学习智能体与下游评估模型所需的固定维度状态向量。
"""
import numpy as np

def build_state_vector(claim_vector, candidate_vectors, max_sentences=60, vector_dim=768):
    """
    将特征拼装成 RL 智能体需要的 46848 维全局状态向量 (State Vector)。
    维度计算: 768 (Claim) + 60 * 768 (Candidates) = 46848 维
    
    Args:
        claim_vector: 新闻论断的 768 维特征向量
        candidate_vectors: 原始候选报道句子的特征向量列表
        max_sentences: 最大候选句子数量 (默认 60)
        vector_dim: 单个特征向量的维度 (默认 768)
        
    Returns:
        numpy.ndarray: 形状为 (46848,) 的一维状态向量
    """
    # 1. 提取并对齐 Claim 向量
    claim_vec = np.array(claim_vector, dtype=np.float32).flatten()[:vector_dim]
    if len(claim_vec) < vector_dim:
        claim_vec = np.pad(claim_vec, (0, vector_dim - len(claim_vec)))

    # 2. 初始化一个全零的矩阵，用于存放 60 句话的向量 (60, 768)
    padded_cand_vecs = np.zeros((max_sentences, vector_dim), dtype=np.float32)
    valid_idx = 0
    
    # 3. 将真实存在的候选句子向量填入矩阵，多退少补
    for vec in candidate_vectors:
        if valid_idx >= max_sentences: break
        flat_vec = np.array(vec, dtype=np.float32).flatten()
        if len(flat_vec) == 0: continue
        elif len(flat_vec) >= vector_dim: 
            padded_cand_vecs[valid_idx, :] = flat_vec[:vector_dim]
        else: 
            padded_cand_vecs[valid_idx, :len(flat_vec)] = flat_vec
        valid_idx += 1

    # 4. 展平并首尾拼接
    return np.concatenate([claim_vec, padded_cand_vecs.flatten()])


def build_downstream_features(claim_vector, candidate_vectors, selected_indices):
    """
    为下游分类器构建推理特征：将 Claim 和 强化学习选中的 5 句话的向量拼接在一起。
    维度计算: 768 (Claim) + 5 * 768 (Top-5 Evidences) = 4608 维
    
    Args:
        claim_vector: 新闻论断的特征向量
        candidate_vectors: 所有的候选句子特征池
        selected_indices: 算法挑选出的 Top-5 句子的索引列表
        
    Returns:
        numpy.ndarray: 形状为 (4608,) 的推理特征向量
    """
    vector_dim = 768
    
    # 1. 对齐 Claim 向量
    claim_vec = np.array(claim_vector, dtype=np.float32).flatten()[:vector_dim]
    if len(claim_vec) < vector_dim: 
        claim_vec = np.pad(claim_vec, (0, vector_dim - len(claim_vec)))
    
    # 2. 依次提取选中的 5 句话的向量
    evidence_vecs = []
    for idx in selected_indices:
        if idx < len(candidate_vectors):
            v = np.array(candidate_vectors[idx], dtype=np.float32).flatten()[:vector_dim]
            if len(v) < vector_dim: 
                v = np.pad(v, (0, vector_dim - len(v)))
            evidence_vecs.append(v)
        else:
            # 如果索引超出了范围（极少发生），用全零向量填充作为安全容错
            evidence_vecs.append(np.zeros(vector_dim, dtype=np.float32))
            
    # 3. 首尾拼接成 4608 维的向量
    return np.concatenate([claim_vec] + evidence_vecs)