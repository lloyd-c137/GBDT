# B 企业简历模式学习实验 V4.0

离线、可追溯的 B 企业历史评级模式实验。项目只从 `data/raw/b企业.zip` 与 `data/raw/p市场.zip` 读取原始 Markdown；不读取或搜索 G 企业数据。原始 ZIP 保持不变。

## 运行

要求 Python 3.11+。在项目目录执行：

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
pytest
```

单命令 `python run_pipeline.py` 会解析 ZIP、恢复错误解码的 UTF-8 中文路径、过滤 macOS 元数据并检查编号和标签，再输出 B/P 独立数据集、质量审计、四组 LOOCV 对照、20 轮稳定性分析、固定 E2 最终模型、P 全量预测与报告。输入数据不完整、标签错误、ACM 状态无法识别或核心编号异常会停止训练。

## 模型

目标是 B 标签 A/B=1、C=0。E0 Dummy、E1 深度 1 决策树（ACM）、E2 固定参数 GBDT（ACM 与工作年限）、E3 固定参数 GBDT（仅工作年限）均采用 Leave-One-Out 验证。E2 的缺失值处理器在每个训练折单独拟合。最终阈值固定为 0.5；P 不参与拟合或参数选择。P 的 score 是未校准模型分数，不是录用概率。

## 输出

- `data/processed/b_train.csv`、`p_predict.csv`：解析后独立数据集，含原始字段与来源哈希。
- `data/processed/data_quality.json`：行数、缺失、编号、标签、ACM 分布、重复哈希和解析错误。
- `models/gbdt.joblib`：完整 B 标签训练的正式 E2 Pipeline。
- `outputs/model_comparison.csv`、`loocv_predictions.csv`、`confusion_matrix.png`。
- `outputs/stability_analysis.csv`、`p_predictions.csv`、`p_positive_candidates.csv`。
- `outputs/feature_analysis.md`、`experiment_manifest.json`、`final_report.md`。

LOOCV 是探索性估计，不是独立测试结果。本模型仅学习小样本历史标注关联，不可用于自动招聘决策。
