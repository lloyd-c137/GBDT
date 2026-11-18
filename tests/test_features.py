from pathlib import Path

import pandas as pd
import pytest

from src.explain import explain_gbdt
from src.features import model_matrix
from src.predict import fit_final
from src.train import build_model
from tests.helpers import write_archives
from src.extract import extract_zip


def test_feature_order_experiments_and_forbidden_fields():
    df = pd.DataFrame({"acm_award": [1], "work_years": [4], "id": [1], "label_grade": ["A"]})
    assert model_matrix(df).columns.tolist() == ["acm_award", "work_years"]
    assert build_model("E0")[0] == []
    assert build_model("E1")[0] == ["acm_award"]
    assert build_model("E2")[0] == ["acm_award", "work_years"]
    assert build_model("E3")[0] == ["work_years"]
    with pytest.raises(ValueError):
        model_matrix(df, ["id"])
    with pytest.raises(ValueError):
        model_matrix(df, ["acm_award", "label_grade"])
    for personal_or_context_field in (
        "candidate_name", "age_raw", "age_years", "current_monthly_salary_raw",
        "employment_start_raw", "work_years_raw", "teamwork_summary", "career_direction", "project_background",
    ):
        with pytest.raises(ValueError):
            model_matrix(df, [personal_or_context_field])


def test_missing_years_add_transformed_indicator_name_without_index_error(tmp_path):
    bzip, _ = write_archives(tmp_path / "raw")
    b, _ = extract_zip(bzip, "B")
    b.loc[0, "work_years"] = float("nan")
    fitted = fit_final(b)
    importance, splits = explain_gbdt(fitted, ["acm_award", "work_years"])
    names = list(fitted.named_steps["impute"].get_feature_names_out(["acm_award", "work_years"]))
    assert names == ["acm_award", "work_years", "missingindicator_work_years"]
    assert len(importance) == len(fitted.named_steps["model"].feature_importances_) == 3
    assert set(splits.feature) <= set(names)


def test_all_missing_training_feature_is_explicitly_rejected(tmp_path):
    bzip, _ = write_archives(tmp_path / "raw")
    b, _ = extract_zip(bzip, "B")
    b["work_years"] = float("nan")
    with pytest.raises(ValueError, match="entirely missing"):
        fit_final(b)
