# coding: utf-8
"""
代理验证器核心 (MLP Proxy Classifier)
用于模拟下游复杂的真假预测任务，快速验证提取的 Top-5 证据特征的数学可分性。
"""
from sklearn.neural_network import MLPClassifier

def get_proxy_classifier(hidden_layer_sizes=(256, 128), max_iter=200, random_state=42):
    """
    获取配置好的多层感知机 (MLP) 分类器。
    
    Args:
        hidden_layer_sizes (tuple): 隐藏层神经元数量配置
        max_iter (int): 最大训练迭代次数
        random_state (int): 随机种子，确保控制变量实验的绝对公平
        
    Returns:
        MLPClassifier: 实例化后的代理分类器
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state
    )