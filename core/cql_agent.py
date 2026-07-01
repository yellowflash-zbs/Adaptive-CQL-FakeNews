# coding: utf-8
"""
强化学习智能体核心 (RL Agent Core)
包含自适应 CQL (Adaptive CQL) 的策略网络加载与动作推理逻辑。
"""
import os
import sys
import types
import torch
import numpy as np

# 为了确保能正确引入你原来的 d4rl 库，将项目根目录加入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
d4rl_path = os.path.join(project_root, "d4rl")
if d4rl_path not in sys.path:
    sys.path.append(d4rl_path)

# rlkit 的部分训练模块会导入 gtimer；评估阶段并不真正使用它。
# 在没有安装 gtimer 的 Windows 环境中放一个轻量兼容模块，避免导入失败。
if "gtimer" not in sys.modules:
    gtimer_stub = types.ModuleType("gtimer")
    gtimer_stub.stamp = lambda *args, **kwargs: None
    gtimer_stub.timed_for = lambda iterable, *args, **kwargs: iterable
    gtimer_stub.reset = lambda *args, **kwargs: None
    sys.modules["gtimer"] = gtimer_stub

# 这里只需要加载策略网络，不需要真实 gym/d4rl 环境。
if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym")
if "d4rl" not in sys.modules:
    sys.modules["d4rl"] = types.ModuleType("d4rl")

try:
    import rlkit.torch.pytorch_util as ptu
    from rlkit.torch.sac.policies import TanhGaussianPolicy
except ImportError as exc:
    raise ImportError("找不到 rlkit 模块，请确保 d4rl/rlkit 库正确配置在项目中。") from exc


def load_adaptive_cql_policy(weight_path, obs_dim=46848, action_dim=60, hidden_sizes=[256, 256, 256], use_gpu=True):
    """
    加载训练好的自适应 CQL 策略网络 (Actor Network)
    
    Args:
        weight_path (str): 模型权重 (.pth) 的绝对路径
        obs_dim (int): 状态空间维度 (默认 46848)
        action_dim (int): 动作空间维度 (默认 60 句候选集)
        hidden_sizes (list): 隐藏层结构
        use_gpu (bool): 是否使用 GPU
        
    Returns:
        policy: 处于 eval 推理模式的策略网络
    """
    if use_gpu and torch.cuda.is_available():
        ptu.set_gpu_mode(True)
        device = ptu.device
    else:
        ptu.set_gpu_mode(False)
        device = torch.device("cpu")
        
    print("🧠 正在加载自适应 CQL 模型大脑 (Actor Network)...")
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim, 
        action_dim=action_dim, 
        hidden_sizes=hidden_sizes
    )
    
    # 挂载权重
    policy.load_state_dict(torch.load(weight_path, map_location=device))
    policy.to(device)
    policy.eval()  # 冻结模型参数，进入推理模式
    
    return policy


def get_rl_top5(policy, state_vec):
    """
    Ours: 使用训练好的 Adaptive CQL 策略抽取最优的 5 句核心证据。
    
    Args:
        policy: load_adaptive_cql_policy 返回的策略网络
        state_vec: 46848 维的特征拼接状态向量
        
    Returns:
        numpy.ndarray: 模型选出的 Top-5 证据的索引数组
    """
    # 将一维状态向量转为 Tensor，并增加 Batch 维度，送入设备
    state_tensor = ptu.from_numpy(state_vec).unsqueeze(0)
    
    with torch.no_grad(): # 推理阶段不需要计算梯度
        # 1. 智能体观察状态，输出动作分布
        action_dist = policy(state_tensor)
        # 2. 提取连续动作空间的确定性输出 (范围 [-1, 1] 的 60 维向量)
        action_continuous = action_dist.mean.cpu().numpy()[0]
        
    # 3. argsort 排序截断：找到激活值最大的 5 个句子的索引
    return np.argsort(action_continuous)[-5:][::-1]
