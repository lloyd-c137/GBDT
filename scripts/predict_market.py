from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import model_matrix
from src.schema import FEATURES

THRESHOLD = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_for(model_path: Path, manifest_path: Path | None) -> dict:
    if manifest_path:
        path = manifest_path
    else:
        candidates = [
            model_path.with_suffix(".manifest.json"),
            model_path.parent.parent / "outputs/experiment_manifest.json",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None or not path.is_file():
        raise FileNotFoundError("model manifest not found; provide --manifest or train with scripts/train_model.py")
    return json.loads(path.read_text(encoding="utf-8"))


def predict_market(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    positive_path: Path | None = None,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    model_path, data_path, output_path = model_path.resolve(), data_path.resolve(), output_path.resolve()
    manifest = _manifest_for(model_path, manifest_path)
    expected_model_hash = manifest.get("model_sha256")
    if expected_model_hash and sha256(model_path) != expected_model_hash:
        raise ValueError("model file hash does not match its manifest")
    if manifest.get("features") and manifest["features"] != FEATURES:
        raise ValueError("model manifest features do not match the approved E2 input order")
    if float(manifest.get("threshold", THRESHOLD)) != THRESHOLD:
        raise ValueError("model manifest threshold does not match the fixed 0.5 threshold")

    frame = pd.read_csv(data_path)
    required = {"id", "source_file", "source_sha256", *FEATURES}
    if required - set(frame.columns):
        raise ValueError(f"P prediction CSV is missing required columns: {sorted(required - set(frame.columns))}")
    if "label_grade" in frame or "label_binary" in frame:
        raise ValueError("P prediction CSV must not contain enterprise labels")
    if len(frame) != 260 or frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("P prediction CSV must contain all 260 unique, non-missing resume IDs")
    if frame.source_file.isna().any() or frame.source_sha256.isna().any():
        raise ValueError("P source provenance fields must be present for every row")

    model = joblib.load(model_path)
    X = model_matrix(frame, FEATURES)
    class_index = list(model.classes_).index(1)
    scores = model.predict_proba(X)[:, class_index]
    result = pd.DataFrame(
        {
            "id": frame.id.astype(int),
            "model_score": scores.astype(float),
            "predicted_label": (scores >= THRESHOLD).astype(int),
            "threshold": THRESHOLD,
            "source_dataset": "P market raw markdown",
            "model_version": manifest.get("model_version", "unavailable"),
            "source_file": frame.source_file,
            "source_sha256": frame.source_sha256,
        }
    )
    if not result.model_score.between(0, 1).all():
        raise ValueError("model emitted a score outside [0, 1]")
    positive_path = positive_path or output_path.with_name(f"{output_path.stem}_positive_candidates.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positive_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    result.loc[result.predicted_label == 1].to_csv(positive_path, index=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Load the trained E2 model and predict unlabeled P resumes")
    parser.add_argument("--model", type=Path, default=ROOT / "models/gbdt.joblib")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/p_predict.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/p_predictions.csv")
    parser.add_argument("--positive-output", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = predict_market(args.model, args.data, args.output, args.positive_output, args.manifest)
    except Exception as error:
        print(f"Prediction failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(
        f"Predicted P={len(result)}; positive={int(result.predicted_label.sum())}; "
        f"threshold={THRESHOLD:.1f}; model_version={result.model_version.iloc[0]}"
    )
    print(f"Full predictions: {args.output}")
    print(f"Positive candidates: {args.positive_output or args.output.with_name(f'{args.output.stem}_positive_candidates.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
