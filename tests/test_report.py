from pathlib import Path

import pandas as pd

from src.evaluate import loocv
from src.extract import extract_zip
from src.predict import fit_final
from src.report import write_report
from src.stability import analyze
from src.features import model_matrix
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


def test_report_error_set_conclusion_is_derived_from_current_predictions(tmp_path):
    b, p, comparison, cv, predictions, stability = _inputs(tmp_path)
    e1 = cv.loc[cv.experiment == "E1"].set_index("id").y_pred
    same = cv.copy()
    mask = same.experiment == "E2"
    same.loc[mask, "y_pred"] = same.loc[mask, "id"].map(e1)
    write_report(b, p, comparison, same, predictions, stability, {"work_years_range": {"b": [1, 19]}}, {}, tmp_path / "same")
    same_text = (tmp_path / "same/final_report.md").read_text()
    assert "E1 and E2 produced the same LOOCV error-ID set." in same_text
    assert "E2 and E1 produced identical LOOCV labels" in same_text
    assert sum(1 for line in same_text.splitlines() if line.startswith("|")) >= 260

    changed = cv.copy()
    mask = changed.experiment == "E2"
    selected = changed.loc[mask].index[0]
    changed.loc[selected, "y_pred"] = 1 - int(changed.loc[selected, "y_true"])
    write_report(b, p, comparison, changed, predictions, stability, {"work_years_range": {"b": [1, 19]}}, {}, tmp_path / "different")
    changed_text = (tmp_path / "different/final_report.md").read_text()
    assert "E1 and E2 produced different LOOCV error-ID sets." in changed_text
    assert "E1 and E2 produced the same LOOCV error-ID set." not in changed_text


def test_confusion_matrix_counts_and_dynamic_prediction_count(tmp_path):
    b, p, comparison, cv, predictions, stability = _inputs(tmp_path)
    predictions.loc[predictions.id <= 4, "predicted_label"] = 1
    predictions.loc[predictions.id > 4, "predicted_label"] = 0
    write_report(b, p, comparison, cv, predictions, stability, {"work_years_range": {"b": [1, 19]}}, {}, tmp_path)
    report = (tmp_path / "final_report.md").read_text()
    assert f"P has {int(predictions.predicted_label.sum())} positive predictions" in report
    assert "TN=" in report and "FP=" in report and "FN=" in report and "TP=" in report
    assert (tmp_path / "confusion_matrix.png").stat().st_size > 1000
