# Bundle-level Offline RL 实验运行步骤

本文档记录接下来 RAWFC / LIAR-RAW 实验的推荐运行顺序。原则是：代码修改由 Codex 完成，真正跑实验、消耗 API、生成大文件和得到指标时，由你在自己电脑上执行命令。

## 0. 开始前检查

确认当前分支是实验分支：

```powershell
git branch --show-current
```

应该输出：

```text
codex/bundle-rl-experiments
```

确认没有未提交代码：

```powershell
git status --short
```

如果没有输出，说明工作区是干净的。

安装或更新依赖：

```powershell
pip install -r requirements.txt
```

如果 `pip` 对应的不是当前 Python，可以使用：

```powershell
python -m pip install -r requirements.txt
```

## 1. 设置 DeepSeek Key

只在需要调用 DeepSeek 的步骤设置。PowerShell 示例：

```powershell
$env:DEEPSEEK_API_KEY="你的API Key"
```

不要把 API Key 写进代码、日志或日报。

## 2. 生成验证集特征

当前仓库已有部分 train/test 特征，但后续调参需要 val 特征。先用小样本检查：

```powershell
python scripts\extract_features.py --dataset RAWFC --split val --limit 2 --overwrite
```

这会生成调试特征文件：

```text
datasets\RAWFC\rl_offline_buffer_val_features_debug2.json
```

如果小样本正常，再生成全量 RAWFC 验证集特征：

```powershell
python scripts\extract_features.py --dataset RAWFC --split val --overwrite
```

LIAR-RAW 同理：

```powershell
python scripts\extract_features.py --dataset LIAR-RAW --split val --limit 2 --overwrite
python scripts\extract_features.py --dataset LIAR-RAW --split val --overwrite
```

如果你的电脑没有 GPU，或 CUDA 环境不稳定，可以加：

```powershell
--device cpu
```

如果你要指定第 0 张 GPU，可以加：

```powershell
--device cuda --cuda-visible-devices 0
```

## 3. 生成证据包候选

先不调用 DeepSeek，只检查格式：

```powershell
python scripts\generate_evidence_bundles.py --dataset RAWFC --split val --feature-suffix debug2 --limit 2 --skip-llm --overwrite
```

确认没有报错后，再小样本调用 DeepSeek 检查 reward 字段：

```powershell
python scripts\generate_evidence_bundles.py --dataset RAWFC --split val --limit 5 --overwrite
```

最后再跑全量训练集候选。注意：这一步会调用 DeepSeek，可能耗时和消耗额度：

```powershell
python scripts\generate_evidence_bundles.py --dataset RAWFC --split train --overwrite
```

LIAR-RAW 同理：

```powershell
python scripts\generate_evidence_bundles.py --dataset LIAR-RAW --split train --overwrite
```

## 4. 训练 bundle-level offline CQL 控制器

RAWFC：

```powershell
python scripts\train_bundle_policy.py --dataset RAWFC --split train --epochs 80 --batch-size 128 --cql-alpha 0.2
```

LIAR-RAW：

```powershell
python scripts\train_bundle_policy.py --dataset LIAR-RAW --split train --epochs 80 --batch-size 128 --cql-alpha 0.2
```

## 5. 先 dry-run 检查证据路由

dry-run 不调用 DeepSeek，只检查是否能成功选证据：

```powershell
python evaluate.py --dataset RAWFC --split val --mode bundle_rl --limit 10 --dry-run-selection
```

如果使用的是调试特征文件，加上：

```powershell
--feature-suffix debug2
```

LIAR-RAW：

```powershell
python evaluate.py --dataset LIAR-RAW --split val --mode bundle_rl --limit 10 --dry-run-selection
```

## 6. 正式验证集评估

先小样本验证：

```powershell
python evaluate.py --dataset RAWFC --split val --mode bundle_rl --limit 20
```

再全量验证集：

```powershell
python evaluate.py --dataset RAWFC --split val --mode bundle_rl
```

对比基线至少跑：

```powershell
python evaluate.py --dataset RAWFC --split val --mode random
python evaluate.py --dataset RAWFC --split val --mode cosine
python evaluate.py --dataset RAWFC --split val --mode current_cql
python evaluate.py --dataset RAWFC --split val --mode claim_only
python evaluate.py --dataset RAWFC --split val --mode defense_judge
python evaluate.py --dataset RAWFC --split val --mode bundle_rl
```

LIAR-RAW 同理，把 `--dataset RAWFC` 换成 `--dataset LIAR-RAW`。

## 7. 查看结果文件

日志和 case study 会保存在：

```text
logs/
```

重点看：

- Accuracy
- Macro F1
- per-class F1
- 混淆矩阵
- 预测分布
- case study 中的 `selected_action`

LIAR-RAW 特别要看 `true`、`mostly-true`、`pants-fire` 是否仍然被压低。

## 8. 不要过早跑 test

现阶段只用 `val` 调参。`test` 只在确定最终设置后跑一次，避免反复调测试集导致结果不可信。
