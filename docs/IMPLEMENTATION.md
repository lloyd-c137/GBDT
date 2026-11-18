# 实现说明

## 输入与来源

项目只读取 `data/raw/b企业.zip` 和 `data/raw/p市场.zip`。数据集经请求者确认为模拟简历并具有公开发布许可。ZIP 路径进行 UTF-8/CP437 恢复，跳过 `__MACOSX` 与点开头元数据，拒绝路径穿越，校验 CRC，并对原始 Markdown 字节计算 SHA-256。流水线不搜索或加载 G 企业档案。

## Schema 与 Markdown 解析

`src/extract.py` 将 Markdown 解析为两张独立数据表。字段识别在前，Markdown 包装字符规范化在后；支持标题边界、引用块、列表、表格、加粗字段、同行中文分号字段和多行内容。代码围栏中的业务值不会被 `#`、`+` 等技术字符清理破坏。

公共字段保留姓名、年龄原文与数值、当前月薪原文、参加工作时间、工作年限原文与数值、ACM 状态/获奖标记、目标岗位、技能、工作内容、项目职责、团队协作、职业方向、项目背景、GitHub、学校、源文件路径和源文件 SHA-256。B 另有原始 A/B/C 评级和二元标签；P 不存在标签列。字段定义、类型、隐私分类和模型用途在 `data/schema.json`。

无法解析的 ACM 状态标为 `unknown` 并记入质量错误，不能默认为 0。明确未获奖、未参赛、没有获奖记录和获奖状态保留为不同语义。缺失薪资等属性留空；结构化字段不因模型暂未使用而丢弃。

## 训练与防泄漏

二分类目标为 A/B=1、C=0。正式 E2 特征顺序固定为 `acm_award`, `work_years`；`src/features.py` 对所有其他字段拒绝作为模型输入，包括姓名、年龄、薪资、工作日期和文本字段。

`src/train.py` 保留四个已冻结实验：E0 无特征多数类基线；E1 ACM 单特征深度 1 决策树；E2 固定 GBDT；E3 相同 GBDT 参数但只用工作年限。E2 参数、随机种子和 0.5 阈值写入实验清单，不运行参数搜索。

`src/evaluate.py` 对 B 的 20 条记录执行 LOOCV。每折的缺失填补器只在 19 条训练折拟合。E1/E2 同时更换算法和特征集合，不能据此单独判断工作年限的边际预测价值。E2/E3 是固定算法配置下的探索性 ACM 特征消融。LOOCV 不是独立测试集。

## 独立训练与预测

- `scripts/train_model.py` 读取且只读取 B CSV，校验行数、评级分布、标签映射和 E2 特征，拟合正式 Pipeline，保存 `joblib` 模型与伴随清单。清单包含版本、Python/scikit-learn 版本、参数、特征、阈值、B CSV 哈希和模型 SHA-256。
- `scripts/predict_market.py` 校验模型清单、输入列、P 无标签、260 个唯一 ID 和来源信息；加载模型而不重新拟合，对全部 P 记录生成分数、阈值类别和正类子集。
- 两个脚本均不读取姓名、年龄、薪资等字段作为特征。独立预测输出保留 ID、来源、哈希和模型版本。

## GBDT 可解释性与图集

`src/visualize.py` 加载已拟合 `Pipeline`，从已拟合填补器获取转换后特征名，再读取全部 40 个回归树节点。结构 CSV 记录树编号、节点、实际分裂特征/阈值、子节点、样本数、raw value、叶节点 learning-rate 加权值、模型版本与模型哈希。

总览图从真实模型参数及初始类别先验生成：初始 raw score 加上 40 棵树的 `learning_rate × leaf_value`，经过 sigmoid 产生未校准分数，最后和 0.5 阈值比较。代表树和 5 页图集由 scikit-learn 的实际 `DecisionTreeRegressor` 结构渲染，不手工虚构节点。树叶是残差更新，不是独立分类器。

## 稳定性与报告

`src/stability.py` 对 E2 做 20 次留一训练样本扰动，为每个 P ID 保存分数、预测类别、翻转数、区间和缺失/范围标记。工作年限数值越界与精确特征组合未出现分开报告，后者不参与拟合且不单独触发复核。

`src/report.py` 从行级 LOOCV 预测和指标 CSV 生成 E1/E2、E2/E3 结论；如果 CSV 指标与行级结果不同则拒绝生成报告。预测结果、模型、混淆矩阵、树结构、数据审计和报告都在独立 Run ID 的 staging 目录生成；全部一致性检查成功后才发布当前成功版本。

`scripts/update_readme.py` 只从已验证的已提交实验产物生成 README 标记区域。`--check` 检查指标、候选编号、运行 ID、Schema 和树覆盖；不会读取 CI 合成运行目录，也不会用 CI 合成结果覆盖正式答案。

## 自动化测试与 CI

pytest 覆盖真实 ZIP 数量/标签、Markdown 字段解析、个人字段留存但不入模、四组 LOOCV、独立 train/predict、joblib 往返、报告边界、40 棵树的每个节点图/CSV 对照、Pipeline 原子发布和预测一致性。

GitHub Actions 使用 sparse checkout 跳过原始 ZIP 和含个人字段的完整 CSV，安装锁定依赖、跑 pytest、核验 README，并在 Runner 临时目录创建合成 B/P ZIP 后执行端到端流程。不会上传简历或 CI 工件。

## 复现

完整实验：

```bash
python run_pipeline.py
```

独立训练、预测、绘图和 README 检查命令见 [README](../README.md)。每次完整运行的输入哈希、参数、版本、Run ID、模型哈希和输出数量均记录在正式实验产物中。
