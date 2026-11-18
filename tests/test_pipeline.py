import json
import os
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_pipeline import run_pipeline
from src.extract import extract_zip
from src.stability import analyze
from tests.helpers import write_archives

ROOT = Path(__file__).resolve().parents[1]


def test_stability_separates_numeric_range_from_exact_combinations(tmp_path):
    bzip, pzip = write_archives(tmp_path / "raw")
    b, _ = extract_zip(bzip, "B")
    p, _ = extract_zip(pzip, "P")
    stable, rounds, scores, labels = analyze(b, p)
    assert len(stable) == 260 and len(rounds) == 20 * 260
    assert rounds.held_out_training_id.nunique() == 20
    assert stable.id.tolist() == p.id.tolist()
    assert np.isfinite(scores).all() and set(np.unique(labels)) <= {0, 1}
    assert stable.score_std.between(0, 1).all()
    assert stable.work_years_out_of_range.sum() > 0
    assert stable.unseen_feature_combination.sum() > stable.work_years_out_of_range.sum()
    expected_review = (
        stable.key_feature_missing | stable.work_years_out_of_range
        | stable.prediction_changed | stable.acm_status_unknown
    )
    assert stable.review_flag.equals(expected_review)
    # Exact combinations alone are descriptive; they do not trigger manual review.
    only_unseen = stable.unseen_feature_combination & ~stable.key_feature_missing & ~stable.work_years_out_of_range & ~stable.prediction_changed
    assert not stable.loc[only_unseen, "review_flag"].any()


def test_missing_market_key_feature_is_not_marked_as_numeric_overflow(tmp_path):
    bzip, pzip = write_archives(tmp_path / "raw")
    b, _ = extract_zip(bzip, "B")
    p, _ = extract_zip(pzip, "P")
    p.loc[0, "work_years"] = float("nan")
    p.loc[0, "acm_award"] = float("nan")
    p.loc[0, "acm_status"] = "unknown"
    stable, _, _, _ = analyze(b, p)
    row = stable.loc[stable.id == 1].iloc[0]
    assert row.work_years_missing and row.acm_award_missing and row.acm_status_unknown
    assert not row.work_years_out_of_range and not row.unseen_feature_combination
    assert row.key_feature_missing and row.review_flag


def test_complete_pipeline_publishes_verified_separate_outputs(tmp_path):
    raw = tmp_path / "raw"
    raw_files = write_archives(raw)
    raw_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw_files}
    artifact_root = tmp_path / "artifacts"
    record = run_pipeline(raw, artifact_root)
    assert record["status"] == "succeeded"
    assert json.loads((artifact_root / "current_success.json").read_text())["run_id"] == record["run_id"]
    run_record = json.loads((artifact_root / "runs" / record["run_id"] / "run_manifest.json").read_text())
    assert run_record["status"] == "succeeded"
    predictions = pd.read_csv(artifact_root / "outputs/p_predictions.csv")
    stability = pd.read_csv(artifact_root / "outputs/stability_analysis.csv")
    rounds = pd.read_csv(artifact_root / "outputs/stability_perturbation_scores.csv")
    assert len(predictions) == len(stability) == 260
    assert predictions.id.nunique() == stability.id.nunique() == 260
    assert len(rounds) == 5200 and rounds.held_out_training_id.nunique() == 20
    assert (predictions.predicted_label == (predictions.model_score >= 0.5).astype(int)).all()
    assert "model_score" not in stability and "score_std" in stability
    assert set(predictions.id) == set(stability.id)
    assert raw_hashes == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw_files}


@pytest.mark.skipif(os.environ.get("GBDT_SKIP_REAL_DATA") == "1" or not (ROOT / "data/raw/b企业.zip").exists(), reason="private raw data excluded from CI")
def test_real_data_range_regression_reference():
    b, _ = extract_zip(ROOT / "data/raw/b企业.zip", "B")
    p, _ = extract_zip(ROOT / "data/raw/p市场.zip", "P")
    stability, _, _, _ = analyze(b, p)
    assert int(stability.work_years_out_of_range.sum()) == 17
    assert int(stability.unseen_feature_combination.sum()) == 148
