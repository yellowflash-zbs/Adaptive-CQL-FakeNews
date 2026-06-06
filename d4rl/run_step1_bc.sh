#!/bin/bash

# =========================================================
# [Step 1] BC 预训练专用脚本 (多种子 并行 执行版) - 遵从 YAML 默认配置
# =========================================================

# ↓↓↓ [用户修改区] ↓↓↓
SEEDS=(0 1 2)  # 填入你要同时跑的种子
ENV_NAME="hopper-medium-replay-v2"
GPU_ID=0
# ↑↑↑ [用户修改区] ↑↑↑

BASE_LOG_DIR="/root/autodl-tmp/logs"

if [[ ! -f "main_off.py" ]]; then
    echo "❌ 错误：请先 cd 到代码根目录下！"
    exit 1
fi

echo "########################################################"
echo "🚀 准备启动环境 [$ENV_NAME] 的 BC 并行预训练..."
echo "########################################################"

# 开始循环遍历所有种子
for SEED in "${SEEDS[@]}"; do
    echo "▶️ 正在后台启动 Seed: $SEED ..."

    # 【关键修改】：去掉了所有强行覆盖训练步数和 Epoch 的参数
    python main_off.py \
        env.name=$ENV_NAME \
        trainer=bc \
        trainer.name="BC" \
        +seed=$SEED \
        device.seed=$SEED \
        device.gpu_idx=$GPU_ID \
        logger.log_dir=$BASE_LOG_DIR \
        rlalg.start_epoch=-100 \
        trainer.exp_prefix="BC_Pretrain_Seed${SEED}" &

    # 缓冲 3 秒，错开不同种子的初始化显存峰值
    sleep 3
done

# 等待所有后台任务完成
wait

echo "########################################################"
echo "🎉 太棒了！环境 [$ENV_NAME] 的所有种子 (${SEEDS[*]}) BC 预训练已全部并行跑完！"
echo "👉 建议：打开 viskit 确认一下它们的 eval/Returns Mean 曲线是否都正常收敛。"
echo "########################################################"