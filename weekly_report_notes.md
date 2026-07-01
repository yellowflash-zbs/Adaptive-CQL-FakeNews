# 周报素材记录

> 这个文件不是正式周报，而是每次开发和实验时的过程记录。每周一生成周报时，从这里整理成 `F:\Study\研究生\周报\佘振邦_周报_YYYYMMDD.md`。

## 2026.07.01

### 代码管理规范
- 确认采用分支开发流程：`main` 保持稳定版本，实验代码放在 `codex/bundle-rl-experiments`。
- 已将实验分支推送到 GitHub，避免直接污染 `main`。
- 每次修改前先创建本地 `backups/` 备份目录，并通过 `.gitignore` 防止备份上传。

### 安全修复
- 移除了 `evaluate.py` 和 `scripts/generate_rewards_deepseek.py` 中的硬编码 DeepSeek API Key。
- 改为从环境变量 `DEEPSEEK_API_KEY` 读取密钥，避免上传 GitHub 后泄漏。

### Bundle-level RL 方向
- 新增证据包候选生成脚本：`scripts/generate_evidence_bundles.py`。
- 新增 bundle-level 离线 CQL 控制器训练脚本：`scripts/train_bundle_policy.py`。
- 新增公共模块：`core/evidence_selection.py`、`core/bundle_policy.py`、`core/label_utils.py`、`core/llm_judge.py`。
- 将下一阶段 RL 从“直接输出 60 维句子选择动作”扩展为“选择证据包策略”的离线 contextual bandit / CQL 控制器。

### 多模式评估
- 重构 `evaluate.py`，支持 `random`、`cosine`、`current_cql`、`claim_only`、`defense_judge`、`bundle_rl` 等单模式评估。
- 输出指标包括 Accuracy、Precision、Recall、Macro F1、混淆矩阵、分类报告、预测分布和 case study。
- 修复评估脚本顶层依赖过重的问题，使 `python evaluate.py --help` 可以在未安装 OpenAI SDK 或 RLKit 环境不完整时正常显示。

### 当前问题
- 项目中 `d4rl/rlkit` 目录包含大量与本课题无关的强化学习算法、环境、示例和历史输出，代码结构过重。
- 下一步需要整理强化学习代码，只保留当前项目真正需要的 Adaptive CQL / bundle-level CQL 相关代码。
