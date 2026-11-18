from __future__ import annotations

from math import isclose
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


def _model_comparison_conclusions(
    comparison: pd.DataFrame, loocv_predictions: pd.DataFrame
) -> tuple[str, str]:
    """Summarize the E1/E2 model contrast and E2/E3 ACM feature ablation."""
    metrics = comparison.set_index("experiment")
    predictions = loocv_predictions.set_index(["experiment", "id"])

    def pair(left: str, right: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        left_rows = predictions.loc[left].sort_index()
        right_rows = predictions.loc[right].sort_index()
        if not left_rows.index.equals(right_rows.index) or not left_rows.y_true.equals(right_rows.y_true):
            raise ValueError(f"{left}/{right} LOOCV rows or true labels do not match")
        return left_rows, right_rows

    def metric_values(experiment: str, rows: pd.DataFrame) -> tuple[float, float]:
        values = (
            float(accuracy_score(rows.y_true, rows.y_pred)),
            float(balanced_accuracy_score(rows.y_true, rows.y_pred)),
        )
        for name, actual in zip(("accuracy", "balanced_accuracy"), values):
            if not isclose(actual, float(metrics.loc[experiment, name]), abs_tol=1e-12):
                raise ValueError(f"{experiment} {name} does not match its LOOCV predictions")
        return values

    def errors(rows: pd.DataFrame) -> set[int]:
        return set(rows.loc[rows.y_true != rows.y_pred].index.astype(int))

    def metric_text(left_name: str, right_name: str, left: tuple[float, float], right: tuple[float, float]) -> str:
        return (
            f"Accuracy {left_name}={left[0]:.4f}, {right_name}={right[0]:.4f} "
            f"(change {right_name}−{left_name} {right[0] - left[0]:+.4f}); Balanced Accuracy "
            f"{left_name}={left[1]:.4f}, {right_name}={right[1]:.4f} "
            f"(change {right_name}−{left_name} {right[1] - left[1]:+.4f})."
        )

    def metric_direction(deltas: tuple[float, float]) -> str:
        accuracy_delta, balanced_delta = deltas
        if accuracy_delta == 0 and balanced_delta == 0:
            return "Neither metric changed."
        if accuracy_delta >= 0 and balanced_delta >= 0:
            return "E2 is no worse on either metric and improves at least one."
        if accuracy_delta <= 0 and balanced_delta <= 0:
            return "E2 is no better on either metric and declines on at least one."
        return "The metrics move in opposite directions; this is a trade-off, not an overall improvement."

    e1, e2 = pair("E1", "E2")
    e1_metrics, e2_metrics = metric_values("E1", e1), metric_values("E2", e2)
    deltas = (e2_metrics[0] - e1_metrics[0], e2_metrics[1] - e1_metrics[1])
    e1_errors, e2_errors = errors(e1), errors(e2)
    changed_labels = int((e1.y_pred.to_numpy() != e2.y_pred.to_numpy()).sum())
    if changed_labels == 0:
        label_result = (
            "E1 and E2 produced identical LOOCV predictions, so no classification gain was observed "
            "for the complete E2 configuration on these folds. This does not establish future generalization."
        )
    else:
        label_result = (
            f"E1 and E2 predictions differed for {changed_labels} of {len(e1)} LOOCV records; "
            "the difference is exploratory and does not identify which configuration change caused it."
        )
    error_result = (
        f"Their error-ID sets are identical ({sorted(e1_errors)})."
        if e1_errors == e2_errors
        else f"Error IDs: E1={sorted(e1_errors)}; E2={sorted(e2_errors)}."
    )
    e1_e2 = (
        "E1 is a DecisionTreeClassifier using acm_award; E2 is a GradientBoostingClassifier using "
        "acm_award and work_years. "
        f"{metric_text('E1', 'E2', e1_metrics, e2_metrics)} {metric_direction(deltas)} "
        f"{label_result} {error_result} Since both the algorithm and feature set change, this comparison "
        "cannot isolate the incremental effect of work_years."
    )

    e2, e3 = pair("E2", "E3")
    e2_metrics, e3_metrics = metric_values("E2", e2), metric_values("E3", e3)
    e2_errors, e3_errors = errors(e2), errors(e3)
    e2_e3_changed = int((e2.y_pred.to_numpy() != e3.y_pred.to_numpy()).sum())
    e2_e3 = (
        "E2 and E3 use the same GBDT configuration; E2 includes acm_award and work_years, while E3 "
        "uses work_years only. "
        f"{metric_text('E2', 'E3', e2_metrics, e3_metrics)} "
        f"Their LOOCV labels differ for {e2_e3_changed} records; error-ID sets "
        f"{'are identical' if e2_errors == e3_errors else f'differ (E2={sorted(e2_errors)}; E3={sorted(e3_errors)})'}. "
        f"This is an exploratory ACM-feature ablation on the current {len(e2)} labeled records; any association is "
        "not causal and does not establish ACM awards as a hiring criterion."
    )
    return e1_e2, e2_e3


def write_report(
    b: pd.DataFrame,
    p: pd.DataFrame,
    comparison: pd.DataFrame,
    loocv_predictions: pd.DataFrame,
    predictions: pd.DataFrame,
    stability: pd.DataFrame,
    manifest: dict,
    quality: dict,
    out: Path,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    e2_predictions = loocv_predictions.loc[loocv_predictions.experiment == "E2"]
    matrix = confusion_matrix(e2_predictions.y_true, e2_predictions.y_pred, labels=[0, 1])
    _write_confusion_png(out / "confusion_matrix.png", matrix)

    by_experiment = {
        name: set(group.loc[group.y_true != group.y_pred, "id"].astype(int).tolist())
        for name, group in loocv_predictions.groupby("experiment")
    }
    e1_e2_conclusion, e2_e3_conclusion = _model_comparison_conclusions(comparison, loocv_predictions)
    labels = b.label_grade.value_counts().reindex(["A", "B", "C"], fill_value=0).to_dict()
    acm = _plain_table(pd.crosstab(b.acm_status, b.label_binary).to_string())
    roles = _plain_table(pd.crosstab(b.target_role.fillna("missing"), b.label_grade).to_string())
    github_group = b.github_url.fillna("").str.match(r"^https?://", case=False).map(
        {True: "Link provided", False: "No link"}
    )
    github = _plain_table(pd.crosstab(github_group, b.label_grade).to_string())
    positive_ids = predictions.loc[predictions.predicted_label == 1, "id"].astype(int).tolist()
    prediction_stability = predictions.merge(stability, on=["id", "source_file", "source_sha256"], validate="one_to_one")
    prediction_stability = prediction_stability.sort_values("id")
    out_of_range = int(stability.work_years_out_of_range.sum())
    unseen = int(stability.unseen_feature_combination.sum())
    flips = int(stability.prediction_changed.sum())
    missing_features = int(stability.key_feature_missing.sum())
    score_tiers = int(predictions.model_score.nunique())
    errors_md = "\n".join(
        f"- **{exp}**: {', '.join(map(str, sorted(ids))) if ids else 'none'}"
        for exp, ids in sorted(by_experiment.items())
    )
    metrics_columns = [
        "experiment", "features", "accuracy", "balanced_accuracy", "precision",
        "recall", "f1", "tn", "fp", "fn", "tp",
    ]
    metrics = comparison.loc[:, [c for c in metrics_columns if c in comparison]].copy()
    numeric_columns = [c for c in metrics.columns if c in {"accuracy", "balanced_accuracy", "precision", "recall", "f1"}]
    for column in numeric_columns:
        metrics[column] = metrics[column].map(lambda value: f"{float(value):.4f}")

    prediction_rows = []
    for row in prediction_stability.itertuples(index=False):
        prediction_rows.append(
            f"|{int(row.id)}|{float(row.model_score):.6f}|{int(row.predicted_label)}|"
            f"{int(row.stability_positive_count)}|{int(row.prediction_changed)}|{float(row.score_std):.6f}|"
            f"{float(row.score_min):.6f}|{float(row.score_max):.6f}|{int(row.work_years_out_of_range)}|"
            f"{int(row.unseen_feature_combination)}|{int(row.review_flag)}|"
        )

    report = f"""# B Company Resume Matching · GBDT V4.1

## Data source and audit

Parsed original Markdown from the B enterprise and P market ZIP files. B contains {len(b)} labeled records and P contains {len(p)} unlabeled records; P was not used for supervised training. Label counts are {labels}; the input ZIP SHA-256 values are recorded in `experiment_manifest.json`. Parser errors: {len(quality.get('parse_errors', []))}; missing key-feature rows in P: {missing_features}. No G-company data was read.

Work-years training range: {manifest.get('work_years_range', {}).get('b', 'not available')}. P rows outside that numeric interval: **{out_of_range}**. Exact `(acm_award, work_years)` combinations absent from B: **{unseen}**. The latter is a descriptive feature-coverage count only; it is not an out-of-distribution detector, a model failure signal, or a hiring criterion. It does not set `review_flag`.

## E0–E3 Leave-One-Out validation

The 20-fold LOOCV results are exploratory because these labels and configurations have been examined before. They are not independent test-set metrics.

{_markdown_table(metrics)}

Errors are computed from this run's row-level predictions:

{errors_md}

### E1 and E2: complete model comparison

{e1_e2_conclusion}

### E2 and E3: exploratory ACM feature ablation

{e2_e3_conclusion}

The E2 confusion matrix is in `confusion_matrix.png`; its counts are TN={int(matrix[0,0])}, FP={int(matrix[0,1])}, FN={int(matrix[1,0])}, TP={int(matrix[1,1])}.

## Observed B-label patterns

ACM status by binary label:

```text
{acm}
```

Target role by original grade:

```text
{roles}
```

GitHub link presence by original grade:

```text
{github}
```

These are associations in a 20-record sample, not official company policy or causal findings. Work-years means are {b.loc[b.label_binary == 1, 'work_years'].mean():.2f} for A/B and {b.loc[b.label_binary == 0, 'work_years'].mean():.2f} for C. Schools, age, salary, skills, GitHub, and resume text are not E2 inputs and must not be treated as automatic rejection criteria.

## E2 fitted rules and P predictions

The fitted transformed-feature importances and split thresholds are in `feature_analysis.md`. Model scores are uncalibrated scores, not hiring probabilities. Threshold is fixed at 0.5. P has {len(positive_ids)} positive predictions and {score_tiers} distinct score tiers; equal scores remain tied.

|id|score|predicted_label|positive perturbation runs|prediction_changed|score_std|score_min|score_max|work_years_out_of_range|unseen_feature_combination|review_flag|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(prediction_rows)}

## Stability

Twenty leave-one-training-sample-out E2 models were evaluated on each P record. Threshold labels changed for {flips} records. Mean score standard deviation was {stability.score_std.mean():.4f}. `stability_analysis.csv` keeps diagnostics separate from `p_predictions.csv`; join them by `id`. `review_flag` is set for missing key features, work-years outside the B numeric range, prediction flips, or unknown ACM status. Unseen exact feature combinations are reported separately and do not by themselves trigger review.

## Limits and reproducibility

- The training set has only {len(b)} labeled resumes and no independent test set; LOOCV remains exploratory.
- E1/E2 comparison: {e1_e2_conclusion}
- E2/E3 comparison: {e2_e3_conclusion}
- P predictions reflect B historical labels, not job competence or hiring probability. Do not use them to automate screening or rejection.
- Each run is preserved under `runs/<run_id>/`; `current_success.json` identifies the latest fully validated run. A failed run is recorded as failed and does not publish partial outputs.
- Run ID: `{manifest.get('run_id', 'unavailable')}`; model version: `{manifest.get('model_version', 'unavailable')}`. B ZIP SHA-256: `{manifest.get('source_sha256', {}).get('b企业.zip', 'unavailable')}`. P ZIP SHA-256: `{manifest.get('source_sha256', {}).get('p市场.zip', 'unavailable')}`. Runtime: Python `{manifest.get('python_version', 'unavailable')}`, scikit-learn `{manifest.get('scikit_learn_version', 'unavailable')}`.
- Reproduce with `python run_pipeline.py`. The run manifest records dependency versions, fixed model parameters, feature names, and threshold.
"""
    (out / "final_report.md").write_text(report, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = frame.astype(str).values.tolist()
    return "|" + "|".join(columns) + "|\n|" + "|".join(["---"] * len(columns)) + "|\n" + "\n".join(
        "|" + "|".join(row) + "|" for row in rows
    )


def _plain_table(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.splitlines())


def _write_confusion_png(path: Path, matrix) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Negative (0)", "Positive (1)"],
        yticklabels=["Negative (0)", "Positive (1)"],
        xlabel="Predicted Label", ylabel="True Label",
        title="E2 LOOCV Confusion Matrix",
    )
    labels = [["TN", "FP"], ["FN", "TP"]]
    midpoint = float(matrix.max()) / 2 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            count = int(matrix[row, column])
            color = "white" if count > midpoint else "black"
            ax.text(column, row, f"{labels[row][column]}\n{count}", ha="center", va="center", color=color, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
