from __future__ import annotations

import numpy as np
import pandas as pd

from .features import model_matrix
from .schema import FEATURES
from .train import build_model, validate_training_matrix

THRESHOLD = 0.5


def analyze(
    b: pd.DataFrame, p: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Fit E2 plus 20 leave-one-training-sample-out models.

    The exact-combination flag is descriptive coverage only. It does not alter
    fitting, scores, labels, or the review flag.
    """
    y = b.label_binary.astype(int).to_numpy()
    X = model_matrix(b, FEATURES)
    XP = model_matrix(p, FEATURES)
    validate_training_matrix(X)

    _, full = build_model("E2")
    full.fit(X, y)
    full_scores = _positive_scores(full, XP)
    full_labels = (full_scores >= THRESHOLD).astype(int)

    perturb_scores = np.empty((len(b), len(p)), dtype=float)
    for held_out in range(len(b)):
        keep = np.arange(len(b)) != held_out
        X_train = X.iloc[keep]
        validate_training_matrix(X_train)
        _, model = build_model("E2")
        model.fit(X_train, y[keep])
        perturb_scores[held_out] = _positive_scores(model, XP)

    observed_years = X.work_years.dropna()
    if observed_years.empty:
        raise ValueError("E2 training feature work_years is entirely missing")
    year_min, year_max = float(observed_years.min()), float(observed_years.max())
    observed_combinations = {
        (int(row.acm_award), float(row.work_years))
        for row in X.itertuples(index=False)
        if pd.notna(row.acm_award) and pd.notna(row.work_years)
    }

    stability_rows = []
    perturb_rows = []
    for i, row in enumerate(p.itertuples(index=False)):
        years = row.work_years
        award = row.acm_award
        missing_years, missing_award = pd.isna(years), pd.isna(award)
        acm_status_unknown = row.acm_status == "unknown"
        work_years_out_of_range = bool(
            not missing_years and (float(years) < year_min or float(years) > year_max)
        )
        unseen_feature_combination = bool(
            not missing_years
            and not missing_award
            and (int(award), float(years)) not in observed_combinations
        )
        scores = perturb_scores[:, i]
        labels = (scores >= THRESHOLD).astype(int)
        prediction_changed = bool(np.any(labels != full_labels[i]))
        key_feature_missing = bool(missing_years or missing_award)
        review_flag = bool(
            key_feature_missing
            or work_years_out_of_range
            or prediction_changed
            or acm_status_unknown
        )
        stability_rows.append(
            {
                "id": int(row.id),
                "stability_positive_count": int(labels.sum()),
                "prediction_changed": prediction_changed,
                "score_std": float(np.std(scores, ddof=0)),
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
                "work_years_missing": bool(missing_years),
                "acm_award_missing": bool(missing_award),
                "acm_status_unknown": bool(acm_status_unknown),
                "key_feature_missing": key_feature_missing,
                "work_years_out_of_range": work_years_out_of_range,
                "unseen_feature_combination": unseen_feature_combination,
                "review_flag": review_flag,
                "source_file": row.source_file,
                "source_sha256": row.source_sha256,
                "source_dataset": "P market raw markdown",
                "model_version": "b-company-gbdt-v4.1",
            }
        )
        for j, score in enumerate(scores):
            perturb_rows.append(
                {
                    "id": int(row.id),
                    "held_out_training_id": int(b.iloc[j].id),
                    "model_score": float(score),
                    "predicted_label": int(score >= THRESHOLD),
                    "source_dataset": "P market raw markdown",
                    "model_version": "b-company-gbdt-v4.1",
                }
            )

    return pd.DataFrame(stability_rows), pd.DataFrame(perturb_rows), full_scores, full_labels


def _positive_scores(model, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_ if hasattr(model, "classes_") else model.named_steps["model"].classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    proba = model.predict_proba(X)
    return np.asarray(proba[:, classes.index(1)], dtype=float)
