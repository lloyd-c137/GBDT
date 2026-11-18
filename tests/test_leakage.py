from pathlib import Path

from src.evaluate import loocv
from src.extract import extract_zip
from src.features import model_matrix
from src.predict import fit_final
from src.train import PARAMS, build_model
from tests.helpers import write_archives


def test_no_labels_or_ids_in_model_matrix_and_p_is_unlabeled(tmp_path):
    bzip, pzip = write_archives(tmp_path / "raw")
    b, _ = extract_zip(bzip, "B")
    p, _ = extract_zip(pzip, "P")
    features, _ = build_model("E2")
    x, xp = model_matrix(b, features), model_matrix(p, features)
    assert x.columns.tolist() == xp.columns.tolist() == ["acm_award", "work_years"]
    assert "label_grade" not in xp and "label_binary" not in xp
    assert "id" not in x and "label_grade" not in x
    assert len(x) == 20 and len(xp) == 260


def test_e0_to_e3_loocv_and_fixed_parameters(tmp_path):
    bzip, _ = write_archives(tmp_path / "raw")
    b, _ = extract_zip(bzip, "B")
    comparison, predictions = loocv(b)
    assert len(predictions) == 80
    assert all(predictions.groupby("experiment").size() == 20)
    assert set(comparison.experiment) == {"E0", "E1", "E2", "E3"}
    assert PARAMS == {
        "loss": "log_loss", "n_estimators": 40, "learning_rate": 0.05,
        "max_depth": 1, "min_samples_leaf": 2, "subsample": 1.0, "random_state": 42,
    }


def test_fixed_input_is_reproducible_and_joblib_round_trips(tmp_path):
    import joblib
    import numpy as np

    bzip, pzip = write_archives(tmp_path / "raw")
    b, _ = extract_zip(bzip, "B")
    p, _ = extract_zip(pzip, "P")
    x = model_matrix(p)
    first, second = fit_final(b), fit_final(b)
    path = tmp_path / "gbdt.joblib"
    joblib.dump(first, path)
    loaded = joblib.load(path)
    expected = first.predict_proba(x)
    assert np.array_equal(expected, second.predict_proba(x))
    assert np.array_equal(expected, loaded.predict_proba(x))
