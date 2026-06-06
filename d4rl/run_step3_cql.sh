#!/bin/bash
# [自适应组] Adaptive CQL - 种子强制对齐 & 并行执行版

# ↓↓↓ [用户修改区] ↓↓↓
SEEDS=(0 1 2)
ENV_NAME="hopper-medium-replay-v2"
GPU_ID=0
KL_SCALE=0.3
BASE_LOG_DIR="/root/autodl-tmp/logs"
# ↑↑↑ [用户修改区] ↑↑↑

TIMESTAMP=$(date +%m%d_%H%M)

echo "########################################################"
echo "🚀 启动 [自适应组 Adaptive CQL] 强制对齐并行实验 (Seeds: ${SEEDS[*]})..."
echo "########################################################"

for SEED in "${SEEDS[@]}"; do

    # 1. 严格动态匹配：仅搜索当前种子对应的文件夹
    BC_MODEL_DIR="${BASE_LOG_DIR}/${ENV_NAME}/BC/BC-Pretrain-Seed${SEED}/BC_Pretrain_Seed${SEED}_seed=${SEED}_*"
    BC_MODEL_PATH=$(ls $BC_MODEL_DIR/final_policy.pth 2>/dev/null | head -n 1)

    # 2. 严格路径检查
    if [ -z "$BC_MODEL_PATH" ] || [ ! -f "$BC_MODEL_PATH" ]; then
        echo "❌ 错误：未找到 Seed ${SEED} 专属的 BC 预训练模型，跳过此种子！"
        continue
    fi

    echo "🎯 匹配成功！正在后台启动 Seed: $SEED ..."

    # 3. 运行自适应版本 (【关键修改】：行末加上 & 放入后台)
    python main_off.py \
        env.name=$ENV_NAME \
        trainer.name="AdaptiveCQL" \
        trainer.policy_type='TanhGaussianPolicy' \
        ~trainer.policy_kwargs.max_log_std \
        ~trainer.policy_kwargs.min_log_std \
        ~trainer.policy_kwargs.std_architecture \
        +seed=$SEED \
        device.seed=$SEED \
        device.gpu_idx=$GPU_ID \
        logger.log_dir=$BASE_LOG_DIR \
        +trainer.trainer_kwargs.bc_model_path="'$BC_MODEL_PATH'" \
        +trainer.trainer_kwargs.kl_scale=$KL_SCALE \
        trainer.exp_prefix="Adaptive_CQL_Seed${SEED}_${TIMESTAMP}" &

    # 缓冲 3 秒
    sleep 3

done

# 【关键修改】：等待所有后台任务完成
wait

echo "########################################################"
echo "🎉 所有指定的种子实验已全部并行运行完毕！"
echo "########################################################"