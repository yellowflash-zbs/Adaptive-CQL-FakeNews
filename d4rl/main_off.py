import os
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import random
import numpy as np
import d4rl, gym
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.make_env import make
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import *
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.exploration_strategies import OUStrategy, GaussianAndEpsilonStrategy, PolicyWrappedWithExplorationStrategy
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer

# 导入所有 Trainer
from rlkit.torch.sac import (
    IQLTrainer,
    S4RLTrainer,
    SQLTrainer,
    BCTrainer,
    IKLTrainer,
    IKL2Trainer
)
# 导入你的对比实验 Trainer 类
from rlkit.torch.sac.cql_trainer import CQLTrainer  # 原本组
from rlkit.torch.sac.adaptive_cql import AdaptiveCQLTrainer  # 自适应实验组

from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.launchers.launcher_util import setup_logger, omegaconf_to_dict
from rlkit.core import logger
from rlkit.util.io import save_model

PolicyPool = {
    'GaussianPolicy': GaussianPolicy,
    'TanhGaussianPolicy': TanhGaussianPolicy,
    'VaePolicy': VaePolicy,
}

# 注册所有可用的训练器
TrainerPool = {
    'IQL': IQLTrainer,
    'CQL': CQLTrainer,  # 原本组
    'AdaptiveCQL': AdaptiveCQLTrainer,  # 自适应实验组
    'S4RL': S4RLTrainer,
    'SQL': SQLTrainer,
    'BC': BCTrainer,
    'IKL': IKLTrainer,
    'IKL2': IKL2Trainer,
}


@hydra.main(config_path='configs', config_name='base.yaml')
def main(args):
    torch.set_num_threads(2)

    if args.spec is not None:
        assert args.spec == args.env.name, 'env mismatches with the specific config'

    # 日志目录配置
    exp_prefix = args.trainer.exp_prefix
    base_log_dir = os.path.join(args.logger.log_dir, args.env.name, args.trainer.name)

    log_dir, variant_log_path = setup_logger(
        exp_prefix=exp_prefix,
        variant=omegaconf_to_dict(args),
        seed=args.device.seed,
        base_log_dir=base_log_dir,
        include_exp_prefix_sub_dir=True
    )

    ### 设备与种子设置
    if args.device.cuda:
        ptu.set_gpu_mode(True, args.device.gpu_idx)
    seed = args.device.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ### 环境与数据缓存设置
    eval_env = make(args.env.name, None, None, normalize_env=args.trainer.obs_norm)
    if args.env.max_episode_steps > 0:
        eval_env._max_episode_steps = args.env.max_episode_steps
    eval_env.seed(seed)
    eval_env.action_space.seed(seed)

    if args.rlalg.online_finetune:
        expl_env = make(args.env.name, None, None, normalize_env=args.trainer.obs_norm)
        expl_env.seed(seed)
        expl_env.action_space.seed(seed)
        if args.env.max_episode_steps > 0:
            expl_env._max_episode_steps = args.env.max_episode_steps
    else:
        expl_env = None

    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    max_action = float(eval_env.action_space.high[0])
    if args.trainer.policy_kwargs.get('max_action') is not None:
        args.trainer.policy_kwargs.max_action = max_action

    # 加载 D4RL 数据集
    replay_buffer = EnvReplayBuffer(
        args.buffer.max_replay_buffer_size,
        eval_env,
        online_finetune=args.rlalg.online_finetune
    )
    replay_buffer.load_hdf5()

    if args.trainer.obs_norm:
        replay_buffer.normalize_states()
        eval_env.set_obs_stats(replay_buffer.mean, replay_buffer.std)
        if args.rlalg.online_finetune:
            expl_env.set_obs_stats(replay_buffer.mean, replay_buffer.std)
        logger.log("Normalization enabled for observations.")

    if args.trainer.reward_norm:
        reward_scale = replay_buffer.reward_scale_by_traj_returns()
        args.trainer.trainer_kwargs.reward_scale = reward_scale

    ### 网络架构初始化
    qf_kwargs = args.trainer.qf_kwargs
    if qf_kwargs is not None:
        # 只有在非 BC 任务时才创建 Q 网络
        qf1 = ConcatMlp(input_size=obs_dim + action_dim, output_size=1, **qf_kwargs)
        qf2 = ConcatMlp(input_size=obs_dim + action_dim, output_size=1, **qf_kwargs)
        target_qf1 = ConcatMlp(input_size=obs_dim + action_dim, output_size=1, **qf_kwargs)
        target_qf2 = ConcatMlp(input_size=obs_dim + action_dim, output_size=1, **qf_kwargs)
    else:
        # BC 模式下 Q 网络设为空
        qf1 = qf2 = target_qf1 = target_qf2 = None

    if args.trainer.vf_kwargs is not None:
        vf = ConcatMlp(input_size=obs_dim, output_size=1, **args.trainer.vf_kwargs)
    else:
        vf = None

    # Actor 网络初始化
    policy = PolicyPool[args.trainer.policy_type](
        obs_dim=obs_dim,
        action_dim=action_dim,
        **args.trainer.policy_kwargs
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(eval_env, eval_policy)

    if args.rlalg.online_finetune:
        expl_path_collector = MdpPathCollector(eval_env, policy)
    else:
        expl_path_collector = None

    ### 实例化训练器
    trainer = TrainerPool[args.trainer.name](
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vf=vf,
        **args.trainer.trainer_kwargs
    )

    ### 启动算法
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=eval_env._max_episode_steps,
        total_training_steps=args.trainer.trainer_kwargs.total_training_steps,
        log_dir=log_dir,
        **args.rlalg
    )

    logger.log_variant(variant_log_path, omegaconf_to_dict(args))

    algorithm.to(ptu.device)
    algorithm.train()

    save_model(log_dir, trainer, name='final_policy.pth')


if __name__ == '__main__':
    main()