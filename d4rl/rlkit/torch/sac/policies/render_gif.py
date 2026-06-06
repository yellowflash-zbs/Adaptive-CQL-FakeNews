import gym
import d4rl
import torch
import imageio
import numpy as np
import cv2
from rlkit.torch.sac.policies import TanhGaussianPolicy
import matplotlib.pyplot as plt


def get_frames_and_distance(env_name, model_path, max_steps=1000):
    """获取运行画面，并实时提取物理引擎的 X 轴位移"""
    print(f"\n[加载模型] 正在获取帧与物理坐标: {model_path}")
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[256, 256],
    )

    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    policy.load_state_dict(state_dict)
    policy.eval()

    frames = []
    distances = []
    obs = env.reset()

    for step in range(max_steps):
        frame = env.render(mode='rgb_array')
        frame = np.copy(frame)

        # 【核心技巧】直接从底层物理引擎获取 X 轴坐标作为真实距离
        current_x = env.unwrapped.sim.data.qpos[0]

        frames.append(frame)
        distances.append(current_x)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = policy(obs_tensor).mean.squeeze(0).numpy()

        obs, reward, done, info = env.step(action)
        if done:
            break

    env.close()
    return frames, distances


def draw_racing_ui(frame, title, current_dist, max_dist, color):
    """在画面上绘制带有进度条的竞速 UI"""
    h, w, c = frame.shape

    # 画一个半透明的黑色顶栏背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # 写上算法名字
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, title, (20, 30), font, 0.7, color, 2, cv2.LINE_AA)

    # 动态显示跑了多少米
    dist_text = f"Dist: {max(0, current_dist):.1f} m"
    cv2.putText(frame, dist_text, (w - 200, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # 绘制赛道底条（灰色）
    bar_y = 45
    cv2.line(frame, (20, bar_y), (w - 20, bar_y), (100, 100, 100), 4)

    # 绘制当前进度的彩色条
    progress_w = int(max(0, min(1.0, current_dist / max_dist)) * (w - 40))
    if progress_w > 0:
        cv2.line(frame, (20, bar_y), (20 + progress_w, bar_y), color, 4)

    return frame


def create_racing_assets(env_name, cql_path, adapt_path, output_gif_name, output_png_name, num_snapshots=7):
    print("=== 开始渲染原版 CQL (底侧赛道) ===")
    frames_cql, dists_cql = get_frames_and_distance(env_name, cql_path)

    print("=== 开始渲染自适应 CQL (顶侧赛道) ===")
    frames_adapt, dists_adapt = get_frames_and_distance(env_name, adapt_path)

    print("\n=== 开始合成竞速画面 ===")
    max_len = max(len(frames_cql), len(frames_adapt))

    # 找出全局跑得最远的距离，用来做进度条的满格刻度
    max_dist = max(max(dists_cql), max(dists_adapt))
    if max_dist <= 0: max_dist = 10  # 防止除以0

    combined_frames = []

    for i in range(max_len):
        # 画面保持逻辑：如果某个算法提前跑完/摔倒，则后续一直显示它最后那一帧
        f_cql = frames_cql[i] if i < len(frames_cql) else frames_cql[-1]
        d_cql = dists_cql[i] if i < len(dists_cql) else dists_cql[-1]

        f_adapt = frames_adapt[i] if i < len(frames_adapt) else frames_adapt[-1]
        d_adapt = dists_adapt[i] if i < len(dists_adapt) else dists_adapt[-1]

        # 绘制 UI
        # 绿色代表你的自适应算法，红色代表原版
        f_adapt_ui = draw_racing_ui(f_adapt, 'Adaptive CQL (Ours)', d_adapt, max_dist, (50, 255, 50))
        f_cql_ui = draw_racing_ui(f_cql, 'Original CQL (Baseline)', d_cql, max_dist, (255, 50, 50))

        # 上下垂直拼接 (Vstack)
        h, w, c = f_adapt_ui.shape
        separator = np.zeros((10, w, c), dtype=np.uint8)  # 中间的黑色分割线
        combined_frame = np.concatenate([f_adapt_ui, separator, f_cql_ui], axis=0)

        combined_frames.append(combined_frame)

    # ------------------ 1. 保存 GIF 动图 ------------------
    print(f"💾 正在保存竞速动图: {output_gif_name}")
    imageio.mimsave(output_gif_name, combined_frames, duration=1000 / 30)

    # ------------------ 2. 截取连续帧并拼接为静态长图 ------------------
    print(f"📸 正在抽取 {num_snapshots} 张关键帧用于论文配图...")

    # 计算等间距的帧索引 (从0开始，到max_len-1结束)
    indices = np.linspace(0, max_len - 1, num_snapshots, dtype=int)

    snapshot_frames = []
    for idx in indices:
        # 获取我们拼接好的上下竞速画面
        frame = combined_frames[idx]

        # 为了在静态图上体现时间流逝，我们在图片右下角加个时间戳/步数标记
        time_labeled_frame = frame.copy()
        text = f"Step: {idx}"
        # 获取文字长宽，画个小黑底白字框放在右下角
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        h, w, _ = time_labeled_frame.shape
        cv2.rectangle(time_labeled_frame, (w - text_w - 20, h - text_h - 20), (w, h), (0, 0, 0), -1)
        cv2.putText(time_labeled_frame, text, (w - text_w - 10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        snapshot_frames.append(time_labeled_frame)

    # 将 7 张图水平拼接成一张大长图 (Hstack)
    # 也可以根据你的论文排版需要，改成上下垂直拼接 (使用 axis=0)
    final_static_img = np.concatenate(snapshot_frames, axis=1)

    print(f"💾 正在保存静态分镜长图: {output_png_name}")
    # OpenCV 存图是 BGR 格式，而我们渲染的镜子是 RGB，存为图片时需要转换一下通道
    final_static_img_bgr = cv2.cvtColor(final_static_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_png_name, final_static_img_bgr)

    print("✅ 全部处理完成！GIF和静态分镜图均已生成！")


if __name__ == "__main__":
    # =============== 配置区域 ===============
    ENV_NAME = 'walker2d-expert-v2'

    # 请核对你的真实模型路径
    CQL_MODEL_PATH = '/home/zy/zz/all_logs/offorrl/walker2d-expert-v2/CQL/Original-CQL-Seed0/final_policy.pth'
    ADAPTIVE_MODEL_PATH = '/home/zy/zz/all_logs/offorrl/walker2d-expert-v2/AdaptiveCQL/Adaptive-CQL-Seed0/final_policy.pth'

    OUTPUT_GIF_NAME = 'walker2d-expert-v2_racing.gif'
    OUTPUT_PNG_NAME = 'walker2d-expert-v2_snapshots.png'  # 生成的论文用截图

    # 你可以通过修改这个值来决定截取多少帧，论文一般放 5~7 帧比较合适
    NUM_SNAPSHOTS = 7
    # ========================================

    create_racing_assets(ENV_NAME, CQL_MODEL_PATH, ADAPTIVE_MODEL_PATH, OUTPUT_GIF_NAME, OUTPUT_PNG_NAME, NUM_SNAPSHOTS)