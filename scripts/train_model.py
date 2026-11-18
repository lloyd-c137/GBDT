from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import uuid
from pathlib import Path

import joblib
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import fit_final
from src.train import PARAMS

MODEL_VERSION = "b-company-gbdt-v4.1"
FEATURES = ["acm_award", "work_years"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_model(data_path: Path, output_path: Path) -> dict:
    data_path, output_path = data_path.resolve(), output_path.resolve()
    frame = pd.read_csv(data_path)
    required = {"id", "label_grade", "label_binary", *FEATURES}
    if required - set(frame.columns):
        raise ValueError(f"training CSV is missing required columns: {sorted(required - set(frame.columns))}")
    if len(frame) != 20 or frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("B training CSV must contain 20 unique, non-missing resume IDs")
    if frame.label_grade.value_counts().to_dict() != {"B": 8, "C": 8, "A": 4}:
        raise ValueError("B labels must preserve the source A/B/C distribution 4/8/8")
    expected = frame.label_grade.isin(["A", "B"]).astype(int)
    if frame.label_binary.isna().any() or not expected.equals(frame.label_binary.astype(int)):
        raise ValueError("label_binary must equal A/B=1 and C=0 from label_grade")

    model = fit_final(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(".manifest.json")
    token = uuid.uuid4().hex
    temp_model = output_path.with_name(f".{output_path.stem}.{token}.joblib")
    temp_manifest = manifest_path.with_name(f".{manifest_path.name}.{token}.tmp")
    try:
        joblib.dump(model, temp_model)
        manifest = {
            "model_version": MODEL_VERSION,
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "features": FEATURES,
            "parameters": PARAMS,
            "threshold": 0.5,
            "random_seed": PARAMS["random_state"],
            "training_rows": len(frame),
            "label_distribution": frame.label_grade.value_counts().to_dict(),
            "training_csv_name": data_path.name,
            "training_csv_sha256": sha256(data_path),
            "model_file_name": output_path.name,
            "model_sha256": sha256(temp_model),
        }
        temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_model, output_path)
        os.replace(temp_manifest, manifest_path)
    finally:
        temp_model.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit the fixed E2 GBDT using labeled B resumes only")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/b_train.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models/gbdt.joblib")
    args = parser.parse_args()
    try:
        manifest = train_model(args.data, args.output)
    except Exception as error:
        print(f"Training failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(
        f"Trained {manifest['model_version']} on B={manifest['training_rows']} rows; "
        f"features={','.join(manifest['features'])}; model_sha256={manifest['model_sha256']}"
    )
    print(f"Parameters: {json.dumps(manifest['parameters'], sort_keys=True)}")
    print(f"Manifest: {args.output.with_suffix('.manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
