# 现场演示脚本（约 6–8 分钟）

## 准备

在仓库根目录、Python 3.11 虚拟环境中执行。完整工作流只读取 B/P 模拟数据，不访问 G 企业数据，也不调用外部服务。

```bash
source .venv/bin/activate
```

演示开始时先展示 README 顶部自动生成的作业答案区、B/P 两个结构化 CSV、数据 Schema 和质量审计。实际结果以当前文件为准，不要照读旧终端截图。

## 演示顺序

### 1. 原始 Markdown 与数据边界（约 40 秒）

**操作：** 在文件浏览器中展示 `data/raw/b企业.zip`、`data/raw/p市场.zip`；只读打开一份模拟 Markdown，指出姓名、年龄、经历、ACM、技能和来源标签等字段。不要展示或读取其他企业的数据。

**解释：** 两个数据集分别编号，B 有监督标签，P 没有企业标签。Schema 把个人属性保留在整理数据中，但这些字段不进入模型。

**对应作业：** 问题 1。

### 2. 解析与质量审计（约 50 秒）

```bash
python run_pipeline.py
```

**展示：** `data/processed/b_train.csv`、`p_predict.csv`、`data/schema.json`、`data/processed/data_quality.json`。可用表格查看器展示少量记录，不需要把整份 CSV 打印到终端。

**正常输出形式：** Pipeline 打印成功状态、Run ID、B/P 行数和正类数量；数值应直接读取当次输出。

**解释：** ZIP 中文路径恢复、macOS 元数据排除、Markdown 字段边界、多行文本和源 SHA-256。缺失的薪资值保持缺失，不做推断。

**对应作业：** 问题 1、端到端流程。

### 3. 训练定义和交叉验证（约 60 秒）

```bash
cat src/train.py
cat outputs/model_comparison.csv
```

**展示：** E0–E3 对照、固定 E2 参数、`FEATURES` 白名单、LOOCV 汇总、逐样本预测和 `outputs/confusion_matrix.png`。

**解释：** A/B→1、C→0 是作业建模假设。E1 是 ACM 单特征决策树，E2 是 ACM+工作年限 GBDT；算法和特征同时变化，所以 E1/E2 不能隔离工作年限的边际增益。E2/E3 是相同 GBDT 配置下的 ACM 特征探索性消融。

**对应作业：** 问题 2。

### 4. 独立训练并保存模型（约 40 秒）

```bash
mkdir -p /tmp/gbdt-demo
python scripts/train_model.py \
  --data data/processed/b_train.csv \
  --output /tmp/gbdt-demo/gbdt.joblib
```

**展示：** 成功摘要、使用特征、固定参数、模型 SHA-256 和旁边的 `gbdt.manifest.json`。此演示输出使用临时位置，不覆盖正式 Pipeline 产物。

**解释：** 保存的是已拟合 Pipeline，包含中位数填补器和 GBDT。训练程序只接收 B CSV。

**对应作业：** 问题 2。

### 5. 独立加载模型预测 P（约 50 秒）

```bash
python scripts/predict_market.py \
  --model /tmp/gbdt-demo/gbdt.joblib \
  --data data/processed/p_predict.csv \
  --output /tmp/gbdt-demo/p_predictions.csv
```

**展示：** 输出中的 260 行预测 CSV 和 `p_predictions_positive_candidates.csv`；按需展示 `outputs/p_predictions.csv` 中的正式完整实验结果。

**解释：** 程序加载刚才保存的模型，检查唯一 ID、P 无评级和来源字段，用固定 0.5 阈值生成类别。分数未经校准，不是录用概率。

**对应作业：** 问题 3。

### 6. 展示全部预测及稳定性（约 40 秒）

**展示：** README 自动生成的全部正类 P 原始编号；`outputs/p_predictions.csv`、`p_positive_candidates.csv`、`stability_analysis.csv`。

**解释：** 正类名单源自模型输出而非手写规则。`work_years_out_of_range` 和未见精确特征组合是两个不同的覆盖指标，稳定性一致不代表准确概率。

**对应作业：** 问题 3。

### 7. 展示模型文件与真实树图（约 60 秒）

```bash
python scripts/visualize_gbdt.py \
  --model models/gbdt.joblib \
  --manifest outputs/experiment_manifest.json
```

**展示：** 模型文件、集成总览 PNG、真实 Tree 001 PNG、Tree 022/040，以及 5 页各 8 棵树的 SVG 图集和结构 CSV。

**解释：** 40 棵最大深度为 1 的回归树逐步修正模型 raw score。叶节点输出经过 learning rate 加权后加入总分，不是各自作出的录用决定。

**对应作业：** 问题 2 的模型解释与图表交付。

### 8. 收尾：复现和限制（约 30 秒）

```bash
python scripts/update_readme.py --check
python -m pytest -q
```

**解释：** README 结果区检查正式实验 CSV 和清单；CI 使用合成夹具。只有 20 条 B 标签、无独立测试集，不能把实验结果推广为真实招聘能力。

## 演示失败时

遇到错误就停止该步骤，查看错误类型和 Run ID。不要继续展示前一次成功运行的输出冒充本次结果。`run_pipeline.py` 只有在所有验收完成后才更新正式输出引用。
