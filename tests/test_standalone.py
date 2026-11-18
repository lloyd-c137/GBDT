import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.predict_market import predict_market
from scripts.train_model import train_model
from src.extract import extract_zip
from src.features import model_matrix
from tests.helpers import write_archives


def test_independent_train_then_loaded_model_predicts_all_market_rows(tmp_path):
    bzip, pzip = write_archives(tmp_path / "raw")
    b, b_errors = extract_zip(bzip, "B")
    p, p_errors = extract_zip(pzip, "P")
    assert not b_errors and not p_errors
    b_path, p_path = tmp_path / "b_train.csv", tmp_path / "p_predict.csv"
    b.to_csv(b_path, index=False)
    p.to_csv(p_path, index=False)
    model_path = tmp_path / "models/gbdt.joblib"
    meta = train_model(b_path, model_path)
    assert meta["training_rows"] == 20
    assert meta["features"] == ["acm_award", "work_years"]
    assert meta["model_sha256"]
    assert model_path.with_suffix(".manifest.json").is_file()

    output_path = tmp_path / "outputs/p_predictions.csv"
    result = predict_market(model_path, p_path, output_path)
    saved = pd.read_csv(output_path)
    positive = pd.read_csv(tmp_path / "outputs/p_predictions_positive_candidates.csv")
    loaded = joblib.load(model_path)
    expected = loaded.predict_proba(model_matrix(p))[:, list(loaded.classes_).index(1)]

    assert len(result) == len(saved) == 260 and saved.id.tolist() == p.id.tolist()
    assert np.array_equal(result.model_score.to_numpy(), expected)
    assert (saved.predicted_label == (saved.model_score >= 0.5).astype(int)).all()
    assert positive.id.tolist() == saved.loc[saved.predicted_label == 1, "id"].tolist()
    assert "label_binary" not in saved and "candidate_name" not in saved


def test_independent_training_rejects_label_mismatch(tmp_path):
    bzip, _ = write_archives(tmp_path / "raw")
    b, _ = extract_zip(bzip, "B")
    b.loc[0, "label_binary"] = 1 - int(b.loc[0, "label_binary"])
    path = tmp_path / "bad.csv"
    b.to_csv(path, index=False)
    with pytest.raises(ValueError, match="label_binary"):
        train_model(path, tmp_path / "models/gbdt.joblib")
