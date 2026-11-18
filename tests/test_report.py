import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.evaluate import loocv
from src.extract import extract_zip
from src.report import write_report
from src.stability import analyze
from tests.helpers import write_archives


def _inputs(tmp_path):
    bzip, pzip = write_archives(tmp_path / "raw")
    b, _ = extract_zip(bzip, "B")
    p, _ = extract_zip(pzip, "P")
    comparison, cv = loocv(b)
    stability, _, scores, labels = analyze(b, p)
    predictions = pd.DataFrame({
        "id": p.id.astype(int), "model_score": scores, "predicted_label": labels,
        "source_file": p.source_file, "source_sha256": p.source_sha256,
    })
    return b, p, comparison, cv, predictions, stability


def _set_predictions(cv, experiment, values):
    mask = cv.experiment == experiment
    ids = cv.loc[mask, "id"].astype(int)
    cv.loc[mask, "y_pred"] = ids.map(values).astype(int).to_numpy()


def _sync_metrics(comparison, cv):
    for experiment in ("E1", "E2", "E3"):
        rows = cv.loc[cv.experiment == experiment]
        mask = comparison.experiment == experiment
        comparison.loc[mask, "accuracy"] = accuracy_score(rows.y_true, rows.y_pred)
        comparison.loc[mask, "balanced_accuracy"] = balanced_accuracy_score(rows.y_true, rows.y_pred)


def _report(tmp_path, case):
    b, p, comparison, cv, predictions, stability = _inputs(tmp_path)
    truth = cv.loc[cv.experiment == "E1"].set_index("id").y_true.astype(int)
    e1 = truth.copy()
    e2 = truth.copy()
    positive_ids = truth[truth == 1].index.tolist()
    negative_ids = truth[truth == 0].index.tolist()

    if case == "identical":
        e1.loc[positive_ids[:2]] = 0
        e2 = e1.copy()
    elif case == "better":
        e1.loc[positive_ids[:5]] = 0
        e2.loc[positive_ids[:1]] = 0
    elif case == "worse":
        e2.loc[positive_ids[:1]] = 0
    elif case == "tradeoff":
        e1.loc[positive_ids[:4]] = 0
        e2.loc[negative_ids[:3]] = 1
    else:
        raise AssertionError(case)

    _set_predictions(cv, "E1", e1)
    _set_predictions(cv, "E2", e2)
    _sync_metrics(comparison, cv)
    out = tmp_path / case
    write_report(b, p, comparison, cv, predictions, stability, {"work_years_range": {"b": [1, 19]}}, {}, out)
    return (out / "final_report.md").read_text(), comparison, cv


def test_identical_e1_e2_predictions_have_caveat_and_consistent_body_and_summary(tmp_path):
    report, _, _ = _report(tmp_path, "identical")
    assert "E1 and E2 produced identical LOOCV predictions" in report
    assert "no classification gain was observed" in report
    assert "This does not establish future generalization." in report
    assert "cannot isolate the incremental effect of work_years" in report
    assert "E1/E2 comparison: " + _comparison_sentence(report) in report
    assert "Adding work_years to E1" not in report
    assert "These are associations in a 20-record sample" in report


def test_e2_improvement_and_error_ids_are_dynamic(tmp_path):
    report, comparison, cv = _report(tmp_path, "better")
    e1_errors = sorted(cv.loc[(cv.experiment == "E1") & (cv.y_true != cv.y_pred), "id"].astype(int))
    e2_errors = sorted(cv.loc[(cv.experiment == "E2") & (cv.y_true != cv.y_pred), "id"].astype(int))
    e1 = comparison.set_index("experiment").loc["E1"]
    e2 = comparison.set_index("experiment").loc["E2"]
    assert f"Accuracy E1={e1.accuracy:.4f}, E2={e2.accuracy:.4f}" in report
    assert f"Balanced Accuracy E1={e1.balanced_accuracy:.4f}, E2={e2.balanced_accuracy:.4f}" in report
    assert "E2 is no worse on either metric and improves at least one." in report
    assert f"Error IDs: E1={e1_errors}; E2={e2_errors}." in report
    assert "E2 did not show an advantage over the simple tree on this run" not in report


def test_e2_decline_is_reported_from_current_data(tmp_path):
    report, _, _ = _report(tmp_path, "worse")
    assert "E2 is no better on either metric and declines on at least one." in report
    assert "Accuracy E1=" in report and "Balanced Accuracy E1=" in report


def test_opposing_metric_directions_are_reported_as_tradeoff(tmp_path):
    report, comparison, _ = _report(tmp_path, "tradeoff")
    rows = comparison.set_index("experiment")
    assert rows.loc["E2", "accuracy"] > rows.loc["E1", "accuracy"]
    assert rows.loc["E2", "balanced_accuracy"] < rows.loc["E1", "balanced_accuracy"]
    assert "The metrics move in opposite directions" in report
    assert "overall improvement" in report


def test_report_rejects_metrics_that_disagree_with_row_predictions(tmp_path):
    b, p, comparison, cv, predictions, stability = _inputs(tmp_path)
    comparison.loc[comparison.experiment == "E2", "accuracy"] = 0.123
    with pytest.raises(ValueError, match="E2 accuracy does not match its LOOCV predictions"):
        write_report(b, p, comparison, cv, predictions, stability, {"work_years_range": {"b": [1, 19]}}, {}, tmp_path)


def test_e2_e3_explains_acm_ablation_as_exploratory_not_causal(tmp_path):
    report, _, _ = _report(tmp_path, "identical")
    assert "same GBDT configuration" in report
    assert "E2 includes acm_award and work_years, while E3 uses work_years only" in report
    assert "exploratory ACM-feature ablation" in report
    assert "any association is not causal" in report
    assert "does not establish ACM awards as a hiring criterion" in report


def test_report_predictions_counts_and_summary_are_dynamic(tmp_path):
    report, comparison, cv = _report(tmp_path, "better")
    e1_e2 = _comparison_sentence(report)
    assert report.count(e1_e2) == 2
    assert "E2/E3 comparison: " in report
    assert "|E2|" in report
    assert "P has " in report and " distinct score tiers" in report
    assert "ACM carries most observed separation" not in report
    assert "Adding work_years" not in report
    assert report.count("E1 and E2 predictions differed") == 2
    assert len(cv.loc[cv.experiment == "E2"]) == 20
    assert comparison.set_index("experiment").loc["E2", "accuracy"] == accuracy_score(
        cv.loc[cv.experiment == "E2", "y_true"], cv.loc[cv.experiment == "E2", "y_pred"]
    )


def test_confusion_matrix_counts_and_dynamic_prediction_count(tmp_path):
    b, p, comparison, cv, predictions, stability = _inputs(tmp_path)
    predictions.loc[predictions.id <= 4, "predicted_label"] = 1
    predictions.loc[predictions.id > 4, "predicted_label"] = 0
    write_report(b, p, comparison, cv, predictions, stability, {"work_years_range": {"b": [1, 19]}}, {}, tmp_path)
    report = (tmp_path / "final_report.md").read_text()
    assert f"P has {int(predictions.predicted_label.sum())} positive predictions" in report
    assert "TN=" in report and "FP=" in report and "FN=" in report and "TP=" in report
    assert (tmp_path / "confusion_matrix.png").stat().st_size > 1000


def _comparison_sentence(report: str) -> str:
    section = report.split("### E1 and E2: complete model comparison\n\n", 1)[1]
    return section.split("\n\n### E2 and E3", 1)[0]
