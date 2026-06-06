#!/bin/bash
# [对照组] 原版 Original CQL - 并行执行版

# ↓↓↓ [用户修改区] ↓↓↓
SEEDS=(0 1 2)
ENV_NAME="hopper-medium-replay-v2"
GPU_ID=0
BASE_LOG_DIR="/root/autodl-tmp/logs"
# ↑↑↑ [用户修改区] ↑↑↑

TIMESTAMP=$(date +%m%d_%H%M)

echo "########################################################"
echo "🚀 启动 [对照组 Original CQL] 并行实验 (Seeds: ${SEEDS[*]})..."
echo "########################################################"

for SEED in "${SEEDS[@]}"; do

    echo "▶️ [正在后台启动 Seed: $SEED]..."

    # 【关键修改】：行末加上 & 放入后台
    python main_off.py \
        env.name=$ENV_NAME \
        trainer.name="CQL" \
        trainer.policy_type='TanhGaussianPolicy' \
        ~trainer.policy_kwargs.max_log_std \
        ~trainer.policy_kwargs.min_log_std \
        ~trainer.policy_kwargs.std_architecture \
        +seed=$SEED \
        device.seed=$SEED \
        device.gpu_idx=$GPU_ID \
        logger.log_dir=$BASE_LOG_DIR \
        ~trainer.trainer_kwargs.policy_weight_decay \
        ~trainer.trainer_kwargs.q_weight_decay \
        ~trainer.trainer_kwargs.cosine_lr_decay \
        ~trainer.trainer_kwargs.quantile \
        ~trainer.trainer_kwargs.clip_score \
        ~trainer.trainer_kwargs.lambda_ \
        ~trainer.trainer_kwargs.beta \
        trainer.exp_prefix="Original_CQL_Seed${SEED}_${TIMESTAMP}" &

    # 缓冲 3 秒
    sleep 3
done

# 【关键修改】：等待所有后台任务完成
wait

echo "########################################################"
echo "🎉 所有种子（${#SEEDS[@]}个）已全部并行运行结束！"
echo "########################################################"