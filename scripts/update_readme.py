from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- AUTO:RESULTS:START -->"
END = "<!-- AUTO:RESULTS:END -->"


def _format_ids(ids: list[int], per_line: int = 20) -> str:
    return "\n".join(
        ", ".join(map(str, ids[start : start + per_line]))
        for start in range(0, len(ids), per_line)
    )


def render_results(root: Path = ROOT) -> str:
    outputs = root / "outputs"
    processed = root / "data/processed"
    manifest = json.loads((outputs / "experiment_manifest.json").read_text(encoding="utf-8"))
    run_summary = json.loads((outputs / "run_summary.json").read_text(encoding="utf-8"))
    quality = json.loads((processed / "data_quality.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(outputs / "model_comparison.csv")
    cv = pd.read_csv(outputs / "loocv_predictions.csv")
    predictions = pd.read_csv(outputs / "p_predictions.csv")
    candidates = pd.read_csv(outputs / "p_positive_candidates.csv")
    stability = pd.read_csv(outputs / "stability_analysis.csv")
    trees = pd.read_csv(outputs / "gbdt_tree_structure.csv")
    preview = pd.read_csv(outputs / "data_preview.csv")

    run_id = manifest.get("run_id")
    if (
        not run_id
        or quality.get("run_id") != run_id
        or run_summary.get("run_id") != run_id
        or run_summary.get("status") != "validated_for_publication"
    ):
        raise ValueError("quality data and experiment manifest are from different runs")
    if run_summary.get("input_sha256") != manifest.get("source_sha256"):
        raise ValueError("run summary input hashes disagree with the experiment manifest")
    if run_summary.get("model_sha256") != manifest.get("model_sha256"):
        raise ValueError("run summary model hash disagrees with the experiment manifest")
    pointer_path = root / "current_success.json"
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if pointer.get("run_id") != run_id:
            raise ValueError("README source is not the current successful run")
        run_manifest_path = root / pointer["run_manifest"]
        run_record = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_record.get("status") != "succeeded":
            raise ValueError("current run is not marked succeeded")

    b_count, p_count = int(quality["input_files"]["b"]["rows"]), int(quality["input_files"]["p"]["rows"])
    if run_summary.get("rows") != {"b": b_count, "p": p_count}:
        raise ValueError("run summary counts disagree with data quality audit")
    label_counts = quality["b_label_distribution"]
    ids = predictions.id.astype(int).tolist()
    if len(ids) != p_count or len(set(ids)) != p_count:
        raise ValueError("P prediction IDs are incomplete or duplicated")
    if "label_grade" in predictions or "label_binary" in predictions:
        raise ValueError("P predictions must remain unlabeled")
    if not (predictions.predicted_label == (predictions.model_score >= float(manifest["threshold"])).astype(int)).all():
        raise ValueError("README source predictions do not match the fixed model threshold")
    positive_ids = predictions.loc[predictions.predicted_label == 1, "id"].astype(int).tolist()
    if candidates.id.astype(int).tolist() != positive_ids:
        raise ValueError("positive candidate CSV disagrees with full predictions")
    if not set(range(1, 41)) <= set(trees.tree_number.astype(int)):
        raise ValueError("tree structure CSV does not cover all 40 model trees")
    personal_columns = {"candidate_name", "age_raw", "age_years", "current_monthly_salary_raw"}
    if personal_columns & set(preview.columns):
        raise ValueError("README preview includes personal data fields")
    if len(cv) != 80 or set(cv.experiment) != {"E0", "E1", "E2", "E3"}:
        raise ValueError("LOOCV CSV must contain all 80 E0–E3 predictions")
    if stability.id.astype(int).tolist() != ids:
        raise ValueError("stability IDs do not match P predictions")

    metrics = comparison.set_index("experiment")
    metric_rows = "\n".join(
        f"| {name} | {row['features']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | "
        f"{'; '.join(map(str, sorted(cv.loc[(cv.experiment == name) & (cv.y_true != cv.y_pred), 'id'].astype(int)))) or '无'} |"
        for name, row in metrics.iterrows()
    )
    parameters = manifest["parameters"]
    salary_missing = int(quality["missing_fields"]["b"].get("current_monthly_salary_raw", 0)) + int(
        quality["missing_fields"]["p"].get("current_monthly_salary_raw", 0)
    )
    sample = predictions.merge(stability, on="id", validate="one_to_one").head(5)
    sample_rows = "\n".join(
        f"| {int(row.id)} | {row.model_score:.6f} | {int(row.predicted_label)} | "
        f"{int(row.stability_positive_count)} / 20 | {int(row.review_flag)} |"
        for row in sample.itertuples(index=False)
    )
    preview_rows = "\n".join(
        f"| {row.dataset} | {int(row.id)} | {row.work_years:g} | {row.acm_status} | "
        f"{int(row.acm_award)} | {row.target_role} | {row.skills_raw} | {row.source_file} |"
        for row in preview.itertuples(index=False)
    )
    return f"""{START}
## 作业问题：直接答案（由本次正式实验结果生成）

| 作业问题 | 直接答案 | 对应产物 |
|---|---|---|
| 1. 整理简历为结构化数据 | B **{b_count}** 份、P **{p_count}** 份；B 标签 A/B/C = {label_counts.get('A', 0)}/{label_counts.get('B', 0)}/{label_counts.get('C', 0)}。两组 ID 独立保存。 | [B CSV](data/processed/b_train.csv) · [P CSV](data/processed/p_predict.csv) · [Schema](data/schema.json) |
| 2. 使用 B 企业标注训练 GBDT | E2：`GradientBoostingClassifier`，特征 `{', '.join(manifest['features'])}`，参数 `{json.dumps(parameters, ensure_ascii=False, sort_keys=True)}`；LOOCV Accuracy **{metrics.loc['E2', 'accuracy']:.4f}**，Balanced Accuracy **{metrics.loc['E2', 'balanced_accuracy']:.4f}**。 | [训练程序](scripts/train_model.py) · [模型](models/gbdt.joblib) · [混淆矩阵](outputs/confusion_matrix.png) · [树图](outputs/figures/gbdt_ensemble_overview.png) |
| 3. 对 P 市场 260 份简历预测 | 正类 **{len(positive_ids)}** 份、负类 **{p_count - len(positive_ids)}** 份；固定阈值 {manifest['threshold']:.1f}。分数未经校准，不是录用概率。 | [完整预测](outputs/p_predictions.csv) · [正类名单](outputs/p_positive_candidates.csv) · [独立预测程序](scripts/predict_market.py) |

**结构化 CSV 的真实行样例**（仅展示非姓名、年龄、薪资字段）：

| 数据集 | ID | 工作年限 | ACM 状态 | ACM 获奖 | 目标岗位 | 技能 | 源文件 |
|---|---:|---:|---|---:|---|---|---|
{preview_rows}

**B 标签 LOOCV 对照**（探索性验证，不是独立测试集）：

| 实验 | 特征 | Accuracy | Balanced Accuracy | 错误 ID |
|---|---|---:|---:|---|
{metric_rows}

E1 是 ACM 单特征浅层决策树；E2 是 ACM + 工作年限 GBDT。两者算法和特征集合都不同，不能用 E1/E2 单独估计工作年限的增量效果。E2/E3 使用同一 GBDT 配置，可作为当前 20 条标注上的探索性 ACM 特征消融；不代表因果关系或确定招聘规则。

P 预测正类的**全部原始编号**（与 B 编号空间分开）：

```text
{_format_ids(positive_ids)}
```

代表性 P 结果（按原始 ID 顺序；同分保持并列）：

| P ID | 未校准分数 | 预测类别 | 20 轮扰动正类次数 | 人工复核标记 |
|---:|---:|---:|---:|---:|
{sample_rows}

本次正式运行：`{run_id}`；输入 ZIP 哈希、模型版本和运行环境见 [实验清单](outputs/experiment_manifest.json)。
原文未提供当前月薪的记录按缺失保留（B/P 合计 {salary_missing} 条），未推断或补造薪资值。
{END}"""


def update_readme(readme_path: Path = ROOT / "README.md", root: Path = ROOT, check: bool = False) -> bool:
    text = readme_path.read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("README must contain exactly one generated-results marker pair")
    start = text.index(START)
    end = text.index(END) + len(END)
    replacement = render_results(root)
    current = text[start:end]
    if check:
        return current == replacement
    readme_path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh or verify the README's generated results block")
    parser.add_argument("--check", action="store_true", help="exit nonzero when README results are stale")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    args = parser.parse_args()
    try:
        matches = update_readme(args.readme, ROOT, args.check)
    except Exception as error:
        print(f"README update failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    if args.check and not matches:
        print("README generated results are stale", file=sys.stderr)
        return 1
    print("README generated results are current" if args.check else "README generated results updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
