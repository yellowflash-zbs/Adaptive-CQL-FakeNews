import os
import pandas as pd
import numpy as np
from rlkit.util.io import collect_file_folder
import matplotlib.pyplot as plt
import matplotlib.style as style

# ================= 配置区域 =================
RADIUS = 10
LAST_EPI = 10
TOT_LEN = 100
convkernel = np.ones(2 * RADIUS + 1)

# 基础日志路径
BASE_LOG_DIR = '/home/zy/zz/all_logs/offorrl'

# 3种任务环境
env_names = ['halfcheetah', 'hopper', 'walker2d']
# 4种数据集质量
env_types = ['medium', 'medium-replay', 'medium-expert', 'expert']

# 需要提取和绘制的3个核心指标
METRICS = [
    'eval/Normalized Returns',
    'trainer/PBRS KL Divergence',
    'trainer/Q1 Predictions Mean'
]

# 对比算法配置
ALGOS = {
    'AdaptiveCQL': ('Adaptive', 'tab:blue', 'Adaptive CQL (Ours)'),
    'CQL': ('Original', 'tab:orange', 'Original CQL (Baseline)')
}


# ============================================

def load_total_results(file_path, config_prefix, items):
    file_dir_list, num_seeds = collect_file_folder(file_path, config_prefix)
    total_seed_data = []
    curr_min_length = np.inf

    for file_dir in file_dir_list:
        filename = os.path.join(file_dir, 'progress.csv')
        if not os.path.exists(filename):
            continue

        df = pd.read_csv(filename, encoding='utf-8')

        extracted_data = []
        for item in items:
            if item in df.columns:
                extracted_data.append(df[item].values)
            else:
                extracted_data.append(np.full(len(df), np.nan))

        one_seed_data = np.column_stack(extracted_data)
        total_seed_data.append(one_seed_data)
        curr_min_length = np.minimum(curr_min_length, one_seed_data.shape[0]).astype(int)

    return total_seed_data, curr_min_length


def smooth_plot(ax, total_data, curr_len, item_idx, color, label):
    if not total_data or curr_len == 0:
        return 0, 0

    tot_smooth_data = []
    for single_data in total_data:
        data_col = single_data[:, item_idx]
        if np.isnan(data_col).all():
            continue

        smooth_data = np.convolve(data_col, convkernel, mode='same') \
                      / np.convolve(np.ones_like(data_col), convkernel, mode='same')
        tot_smooth_data.append(smooth_data[:curr_len])

    if not tot_smooth_data:
        return 0, 0

    x_ = np.arange(-curr_len, 0)
    y_mean = np.mean(tot_smooth_data, axis=0)
    y_std = np.std(tot_smooth_data, axis=0)

    ax.plot(x_, y_mean, color=color, linestyle='-', label=label, linewidth=2.5)
    ax.fill_between(x_, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)

    return np.mean(y_mean[-LAST_EPI:]), np.mean(y_std[-LAST_EPI:])


if __name__ == '__main__':
    style.use('seaborn-v0_8-darkgrid')

    # 【终极优化】：不仅设置总画布尺寸，更通过 gridspec 强制要求每行之间有足够的空隙
    # 我们将总高度设置为 72 (相当于每个子图高度约 6 英寸)，宽度 24
    fig, axes = plt.subplots(nrows=12, ncols=3,
                             figsize=(24, 72), dpi=150,
                             gridspec_kw={'hspace': 0.5, 'wspace': 0.25})

    # 因为我们用了 gridspec_kw 指定了间距，所以这里不再使用 constrained_layout=True

    row_idx = 0

    for env_name in env_names:
        for env_type in env_types:
            task_name = f"{env_name}-{env_type}-v2"
            print(f"正在绘制任务: {task_name}...")

            for algo_dir, (prefix_name, color, label) in ALGOS.items():
                file_path = os.path.join(BASE_LOG_DIR, task_name, algo_dir)

                try:
                    total_data, curr_len = load_total_results(file_path, prefix_name, METRICS)

                    if curr_len < np.inf and curr_len > 0:
                        for col_idx in range(len(METRICS)):
                            ax = axes[row_idx, col_idx]
                            smooth_plot(ax, total_data, curr_len, col_idx, color, label)
                except Exception as e:
                    print(f"  [跳过] {task_name} {algo_dir}")

            # =============== 设置子图的美化标签 ===============
            for col_idx, metric_name in enumerate(METRICS):
                ax = axes[row_idx, col_idx]
                short_metric = metric_name.split('/')[-1]

                # 增大刻度字体，让数字清晰可见
                ax.tick_params(axis='both', which='major', labelsize=14)

                if col_idx == 0:
                    # 将任务名称和指标名称分行显示，极大地防止重叠
                    ax.set_ylabel(f"[{task_name}]\n\n{short_metric}",
                                  fontsize=18, fontweight='bold', labelpad=20)
                else:
                    ax.set_ylabel(short_metric, fontsize=16, labelpad=15)

                ax.set_xlabel("Epoch", fontsize=16, labelpad=10)

                # 第一行显示全局大标题
                if row_idx == 0:
                    ax.set_title(short_metric, fontsize=24, fontweight='bold', pad=30)

                # 添加图例，并让图例背景半透明以免遮挡曲线
                if col_idx == 0:
                    ax.legend(loc='upper left', fontsize=14, framealpha=0.8)

            row_idx += 1

    # 导出高清完整大图
    output_filename = "all_12_tasks_36_metrics_plot.png"
    # 使用 bbox_inches='tight' 自动裁掉多余白边
    plt.savefig(output_filename, bbox_inches='tight', format='png')
    print(f"\n✅ 绘图完成！完整大图已保存为: {output_filename}")