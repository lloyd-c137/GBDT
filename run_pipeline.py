from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from src.evaluate import loocv
from src.explain import explain_gbdt
from src.extract import extract_zip
from src.features import model_matrix
from src.predict import fit_final
from src.report import write_report
from src.schema import FEATURES
from src.stability import analyze
from src.train import PARAMS

ROOT = Path(__file__).resolve().parent
MODEL_VERSION = "b-company-gbdt-v4.1"
THRESHOLD = 0.5
EXPERIMENTS = ("E0", "E1", "E2", "E3")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_file_counts(path: Path) -> dict:
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise ValueError(f"input is missing or not a readable ZIP: {path.name}")
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"ZIP CRC check failed: {path.name}")
        names = archive.namelist()
    usable = [
        name for name in names
        if not name.endswith("/")
        and not name.startswith("__MACOSX/")
        and not any(part.startswith(".") for part in PurePosixPath(name.replace("\\", "/")).parts)
    ]
    markdown = [name for name in usable if name.lower().endswith((".md", ".markdown"))]
    return {
        "archive_entries": len(names),
        "markdown_resumes": len(markdown),
        "macos_metadata": sum(name.startswith("__MACOSX/") for name in names),
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)


def _write_run_record(run_dir: Path, record: dict) -> None:
    _atomic_json(run_dir / "run_manifest.json", record)


def _quality_report(b: pd.DataFrame, p: pd.DataFrame, bzip: Path, pzip: Path, run_id: str) -> dict:
    expected_b, expected_p = set(range(1, 21)), set(range(1, 261))
    return {
        "run_id": run_id,
        "input_files": {
            "b": {"sha256": sha(bzip), "rows": len(b)},
            "p": {"sha256": sha(pzip), "rows": len(p)},
        },
        "file_counts": {"b": zip_file_counts(bzip), "p": zip_file_counts(pzip)},
        "parse_errors": [],
        "b_missing_ids": sorted(expected_b - set(b.id)),
        "b_extra_ids": sorted(set(b.id) - expected_b),
        "p_missing_ids": sorted(expected_p - set(p.id)),
        "p_extra_ids": sorted(set(p.id) - expected_p),
        "b_label_distribution": b.label_grade.value_counts(dropna=False).to_dict(),
        "b_acm_distribution": b.acm_status.value_counts(dropna=False).to_dict(),
        "p_acm_distribution": p.acm_status.value_counts(dropna=False).to_dict(),
        "missing_fields": {"b": b.isna().sum().to_dict(), "p": p.isna().sum().to_dict()},
        "data_types": {"b": b.dtypes.astype(str).to_dict(), "p": p.dtypes.astype(str).to_dict()},
        "work_years_range": {
            "b": [float(b.work_years.min()), float(b.work_years.max())] if b.work_years.notna().any() else None,
            "p": [float(p.work_years.min()), float(p.work_years.max())] if p.work_years.notna().any() else None,
        },
        "duplicate_hashes": {
            "b": int(b.source_sha256.duplicated().sum()),
            "p": int(p.source_sha256.duplicated().sum()),
        },
        "duplicate_text_rate": {
            "b": float(b.duplicated(subset=["work_summary", "project_duties", "skills_raw"]).mean()),
            "p": float(p.duplicated(subset=["work_summary", "project_duties", "skills_raw"]).mean()),
        },
    }


def _validate_data(b: pd.DataFrame, p: pd.DataFrame, errors: list[dict]) -> None:
    expected_b, expected_p = set(range(1, 21)), set(range(1, 261))
    if errors:
        raise ValueError(f"input parsing failed for {len(errors)} archive member(s)")
    if len(b) != 20 or set(b.id) != expected_b:
        raise ValueError("B must contain exactly one parsed resume for each ID 1–20")
    if len(p) != 260 or set(p.id) != expected_p:
        raise ValueError("P must contain exactly one parsed resume for each ID 1–260")
    if b.label_grade.isna().any() or b.label_grade.value_counts().to_dict() != {"A": 4, "B": 8, "C": 8}:
        raise ValueError("B labels are missing or do not match the source 4/8/8 grade counts")
    expected_binary = b.label_grade.isin(["A", "B"]).astype(int)
    if not np.array_equal(expected_binary.to_numpy(), b.label_binary.astype(int).to_numpy()):
        raise ValueError("B binary labels do not match the preserved A/B/C grades")
    if "label_grade" in p or "label_binary" in p:
        raise ValueError("P must not contain enterprise labels")
    required = ["acm_status", "acm_award", "target_role", "skills_raw", "work_summary", "project_duties", "github_url", "school_name", "source_file", "source_sha256"]
    for dataset, frame in (("B", b), ("P", p)):
        missing = {column: int(frame[column].isna().sum()) for column in required if frame[column].isna().any()}
        if missing:
            raise ValueError(f"{dataset} required-field parse gaps: {missing}")


def _build_feature_analysis(model, p: pd.DataFrame, predictions: pd.DataFrame, out: Path) -> None:
    importance, splits = explain_gbdt(model, FEATURES)
    grouped = p[["id", "acm_award"]].merge(predictions[["id", "model_score", "predicted_label"]], on="id", validate="one_to_one")
    grouped = grouped.groupby("acm_award").agg(
        count=("id", "size"), score_min=("model_score", "min"), score_max=("model_score", "max"),
        predicted_positive=("predicted_label", "sum"),
    )
    importance_table = _to_markdown(importance.assign(importance=importance.importance.map(lambda x: f"{x:.6f}")))
    split_table = _to_markdown(splits.assign(threshold=splits.threshold.map(lambda x: f"{x:.6g}")))
    groups = grouped.reset_index()
    groups["score_min"] = groups.score_min.map(lambda x: f"{x:.6f}")
    groups["score_max"] = groups.score_max.map(lambda x: f"{x:.6f}")
    group_table = _to_markdown(groups)
    report = (
        "# E2 Model Structure\n\n"
        "Importances are computed over the fitted transformed feature matrix. They describe this fitted model, not causal effects.\n\n"
        f"## Feature importance\n\n{importance_table}\n\n"
        f"## Split nodes\n\n{split_table}\n\n"
        f"## P scores by ACM award indicator\n\n{group_table}\n\n"
        f"P produced {predictions.model_score.nunique()} distinct uncalibrated score tiers; equal scores remain tied.\n"
    )
    out.write_text(report, encoding="utf-8")


def _to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = frame.astype(str).values.tolist()
    return "|" + "|".join(columns) + "|\n|" + "|".join(["---"] * len(columns)) + "|\n" + "\n".join(
        "|" + "|".join(row) + "|" for row in rows
    )


def _validate_outputs(artifact_dir: Path, b: pd.DataFrame, p: pd.DataFrame, run_id: str) -> None:
    proc, outputs, models = artifact_dir / "data/processed", artifact_dir / "outputs", artifact_dir / "models"
    required_files = [
        proc / "b_train.csv", proc / "p_predict.csv", proc / "data_quality.json", models / "gbdt.joblib",
        outputs / "model_comparison.csv", outputs / "loocv_predictions.csv", outputs / "confusion_matrix.png",
        outputs / "stability_analysis.csv", outputs / "stability_perturbation_scores.csv",
        outputs / "p_predictions.csv", outputs / "p_positive_candidates.csv", outputs / "feature_analysis.md",
        outputs / "experiment_manifest.json", outputs / "final_report.md",
    ]
    missing_files = [str(path.relative_to(artifact_dir)) for path in required_files if not path.is_file()]
    if missing_files:
        raise RuntimeError(f"required outputs missing: {missing_files}")
    pred = pd.read_csv(outputs / "p_predictions.csv")
    stability = pd.read_csv(outputs / "stability_analysis.csv")
    positive = pd.read_csv(outputs / "p_positive_candidates.csv")
    cv = pd.read_csv(outputs / "loocv_predictions.csv")
    comparison = pd.read_csv(outputs / "model_comparison.csv")
    rounds = pd.read_csv(outputs / "stability_perturbation_scores.csv")
    expected_ids = list(range(1, 261))
    if pred.id.tolist() != expected_ids or pred.id.nunique() != 260:
        raise RuntimeError("P predictions do not contain each source ID exactly once and in source order")
    if stability.id.tolist() != expected_ids or stability.id.nunique() != 260:
        raise RuntimeError("stability rows do not align one-to-one with P IDs")
    if positive.id.tolist() != pred.loc[pred.predicted_label == 1, "id"].tolist():
        raise RuntimeError("positive-candidate file disagrees with full P predictions")
    if not pred.model_score.between(0, 1).all() or not (pred.predicted_label == (pred.model_score >= THRESHOLD).astype(int)).all():
        raise RuntimeError("P scores or labels violate the fixed 0.5 threshold")
    if set(cv.experiment) != set(EXPERIMENTS) or not (cv.groupby("experiment").size() == 20).all():
        raise RuntimeError("LOOCV output must have 20 predictions for each E0–E3 experiment")
    if set(comparison.experiment) != set(EXPERIMENTS):
        raise RuntimeError("model comparison does not contain exactly E0–E3")
    e2 = comparison.loc[comparison.experiment == "E2"].iloc[0]
    e2_predictions = cv.loc[cv.experiment == "E2"]
    confusion = {
        "tn": int(((e2_predictions.y_true == 0) & (e2_predictions.y_pred == 0)).sum()),
        "fp": int(((e2_predictions.y_true == 0) & (e2_predictions.y_pred == 1)).sum()),
        "fn": int(((e2_predictions.y_true == 1) & (e2_predictions.y_pred == 0)).sum()),
        "tp": int(((e2_predictions.y_true == 1) & (e2_predictions.y_pred == 1)).sum()),
    }
    if any(int(e2[key]) != value for key, value in confusion.items()):
        raise RuntimeError("E2 comparison confusion counts disagree with LOOCV predictions")
    if rounds.id.nunique() != 260 or rounds.held_out_training_id.nunique() != 20 or len(rounds) != 5200:
        raise RuntimeError("stability perturbation file is incomplete")
    if not rounds.model_score.between(0, 1).all():
        raise RuntimeError("perturbation scores are outside [0, 1]")
    if not set(b.id).issubset(set(cv.id)) or set(pred.id) != set(p.id):
        raise RuntimeError("source IDs do not match prediction outputs")
    model = joblib.load(models / "gbdt.joblib")
    reloaded = model.predict_proba(model_matrix(p))[:, list(model.classes_).index(1)]
    if not np.allclose(reloaded, pred.model_score.to_numpy(dtype=float), rtol=0, atol=1e-15):
        raise RuntimeError("saved model predictions do not reproduce p_predictions.csv")
    report = (outputs / "final_report.md").read_text(encoding="utf-8")
    if report.count("\n|") < 260:
        raise RuntimeError("final report does not include all P prediction rows")
    if f"TN={confusion['tn']}, FP={confusion['fp']}, FN={confusion['fn']}, TP={confusion['tp']}" not in report:
        raise RuntimeError("report confusion matrix disagrees with LOOCV output")
    manifest = json.loads((outputs / "experiment_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_id or manifest.get("parameters") != PARAMS:
        raise RuntimeError("experiment manifest Run ID or fixed model parameters do not match")


def _publish_success(artifact_root: Path, run_dir: Path, run_id: str, record: dict) -> None:
    """Publish output directories as a journaled, rollback-capable transaction."""
    mapping = {
        "processed": (run_dir / "artifacts/data/processed", artifact_root / "data/processed"),
        "models": (run_dir / "artifacts/models", artifact_root / "models"),
        "outputs": (run_dir / "artifacts/outputs", artifact_root / "outputs"),
    }
    staging = run_dir / "publish_staging"
    backups = run_dir / "rollback"
    staging.mkdir(exist_ok=True)
    backups.mkdir(exist_ok=True)
    for name, (source, _) in mapping.items():
        shutil.copytree(source, staging / name)

    pointer = artifact_root / "current_success.json"
    previous_pointer = pointer.read_bytes() if pointer.exists() else None
    journal_path = artifact_root / ".publish_transaction.json"
    journal = {
        "run_id": run_id,
        "status": "publishing",
        "previous_pointer": base64.b64encode(previous_pointer).decode("ascii") if previous_pointer is not None else None,
        "targets": [
            {
                "name": name,
                "destination": destination.relative_to(artifact_root).as_posix(),
                "backup": (backups / name).relative_to(artifact_root).as_posix(),
                "had_original": destination.exists(),
            }
            for name, (_, destination) in mapping.items()
        ],
        "staging": staging.relative_to(artifact_root).as_posix(),
    }
    _atomic_json(journal_path, journal)
    try:
        for name, (_, destination) in mapping.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup = backups / name
            if destination.exists():
                os.replace(destination, backup)
            os.replace(staging / name, destination)

        _atomic_json(pointer, {"run_id": run_id, "run_manifest": f"runs/{run_id}/run_manifest.json"})
        journal["status"] = "committed"
        _atomic_json(journal_path, journal)
        record.update(status="succeeded", completed_at_utc=datetime.now(timezone.utc).isoformat(), error=None)
        _write_run_record(run_dir, record)
    except Exception:
        if journal_path.exists():
            persisted = json.loads(journal_path.read_text(encoding="utf-8"))
            was_committed = persisted.get("status") == "committed"
            _recover_interrupted_publish(artifact_root, persisted)
            if was_committed:
                return
        raise
    shutil.rmtree(backups, ignore_errors=True)
    shutil.rmtree(staging, ignore_errors=True)
    journal_path.unlink(missing_ok=True)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _recover_interrupted_publish(artifact_root: Path, journal: dict | None = None) -> None:
    journal_path = artifact_root / ".publish_transaction.json"
    if journal is None:
        if not journal_path.exists():
            return
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
    run_id = journal["run_id"]
    run_dir = artifact_root / "runs" / run_id
    backups = run_dir / "rollback"
    staging = artifact_root / journal["staging"] if not Path(journal["staging"]).is_absolute() else Path(journal["staging"])
    if journal.get("status") == "committed":
        run_manifest = run_dir / "run_manifest.json"
        if run_manifest.exists():
            record = json.loads(run_manifest.read_text(encoding="utf-8"))
            record.update(status="succeeded", completed_at_utc=record.get("completed_at_utc") or datetime.now(timezone.utc).isoformat(), error=None)
            _write_run_record(run_dir, record)
    else:
        for entry in reversed(journal["targets"]):
            destination = artifact_root / entry["destination"]
            backup = artifact_root / entry["backup"]
            if backup.exists():
                _remove_path(destination)
                os.replace(backup, destination)
            elif not entry["had_original"]:
                _remove_path(destination)
        pointer = artifact_root / "current_success.json"
        old = journal.get("previous_pointer")
        if old is None:
            pointer.unlink(missing_ok=True)
        else:
            temp = pointer.with_name(f".{pointer.name}.{uuid.uuid4().hex}.restore")
            temp.write_bytes(base64.b64decode(old))
            os.replace(temp, pointer)
        run_manifest = run_dir / "run_manifest.json"
        if run_manifest.exists():
            record = json.loads(run_manifest.read_text(encoding="utf-8"))
            record.update(
                status="failed",
                completed_at_utc=datetime.now(timezone.utc).isoformat(),
                error={"type": "InterruptedPublish", "message": "incomplete publication was rolled back"},
            )
            _write_run_record(run_dir, record)
    shutil.rmtree(backups, ignore_errors=True)
    shutil.rmtree(staging, ignore_errors=True)
    journal_path.unlink(missing_ok=True)


def run_pipeline(input_root: Path = ROOT / "data/raw", artifact_root: Path = ROOT, run_id: str | None = None) -> dict:
    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    input_root, artifact_root = Path(input_root).resolve(), Path(artifact_root).resolve()
    runs_root = artifact_root / "runs"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": None,
        "input_root": str(input_root),
        "artifact_root": str(artifact_root),
        "error": None,
    }
    _write_run_record(run_dir, record)
    artifact_dir = run_dir / "artifacts"
    bzip, pzip = input_root / "b企业.zip", input_root / "p市场.zip"
    lock_stream = None
    lock_acquired = False
    try:
        lock_stream = (artifact_root / ".pipeline.lock").open("a+")
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_acquired = True
        except BlockingIOError as error:
            raise RuntimeError("another pipeline run is active for this artifact root") from error
        _recover_interrupted_publish(artifact_root)
        b_counts, p_counts = zip_file_counts(bzip), zip_file_counts(pzip)
        if b_counts["markdown_resumes"] != 20 or p_counts["markdown_resumes"] != 260:
            raise ValueError("ZIP structure has unexpected Markdown resume counts")
        b, b_errors = extract_zip(bzip, "B")
        p, p_errors = extract_zip(pzip, "P")
        errors = b_errors + p_errors
        quality = _quality_report(b, p, bzip, pzip, run_id)
        quality["parse_errors"] = errors
        artifact_dir.mkdir(parents=True, exist_ok=True)
        processed = artifact_dir / "data/processed"
        outputs = artifact_dir / "outputs"
        models = artifact_dir / "models"
        processed.mkdir(parents=True)
        outputs.mkdir(parents=True)
        models.mkdir(parents=True)
        b.to_csv(processed / "b_train.csv", index=False)
        p.to_csv(processed / "p_predict.csv", index=False)
        (processed / "data_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        _validate_data(b, p, errors)

        comparison, cv_predictions = loocv(b)
        comparison["source_dataset"] = "B enterprise labeled markdown"
        comparison["model_version"] = MODEL_VERSION
        cv_predictions["model_version"] = MODEL_VERSION
        cv_predictions.to_csv(outputs / "loocv_predictions.csv", index=False)
        comparison.to_csv(outputs / "model_comparison.csv", index=False)

        stability, perturbations, scores, labels = analyze(b, p)
        final_model = fit_final(b)
        joblib.dump(final_model, models / "gbdt.joblib")
        predictions = pd.DataFrame(
            {
                "id": p.id.astype(int),
                "model_score": scores.astype(float),
                "predicted_label": labels.astype(int),
                "threshold": THRESHOLD,
                "source_dataset": "P market raw markdown",
                "model_version": MODEL_VERSION,
                "source_file": p.source_file,
                "source_sha256": p.source_sha256,
            }
        )
        predictions.to_csv(outputs / "p_predictions.csv", index=False)
        predictions.loc[predictions.predicted_label == 1].to_csv(outputs / "p_positive_candidates.csv", index=False)
        stability.to_csv(outputs / "stability_analysis.csv", index=False)
        perturbations.to_csv(outputs / "stability_perturbation_scores.csv", index=False)

        manifest = {
            "run_id": run_id,
            "model_version": MODEL_VERSION,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "joblib_version": joblib.__version__,
            "parameters": PARAMS,
            "features": FEATURES,
            "threshold": THRESHOLD,
            "random_seed": PARAMS["random_state"],
            "source_sha256": {"b企业.zip": sha(bzip), "p市场.zip": sha(pzip)},
            "rows": {"b": len(b), "p": len(p)},
            "label_distribution": quality["b_label_distribution"],
            "work_years_range": quality["work_years_range"],
        }
        (outputs / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _build_feature_analysis(final_model, p, predictions, outputs / "feature_analysis.md")
        write_report(b, p, comparison, cv_predictions, predictions, stability, manifest, quality, outputs)
        _validate_outputs(artifact_dir, b, p, run_id)

        record.update(
            input_sha256=manifest["source_sha256"],
            output_root=str(artifact_root),
            prediction_count=len(predictions),
            positive_count=int(predictions.predicted_label.sum()),
        )
        _publish_success(artifact_root, run_dir, run_id, record)
        return record
    except Exception as error:
        record.update(
            status="failed",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        _write_run_record(run_dir, record)
        raise
    finally:
        if lock_stream is not None:
            if lock_acquired:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            lock_stream.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the reproducible B-company GBDT experiment")
    parser.add_argument("--input-root", type=Path, default=ROOT / "data/raw", help="directory containing b企业.zip and p市场.zip")
    parser.add_argument("--artifact-root", type=Path, default=ROOT, help="root for processed data, models, outputs, and run history")
    args = parser.parse_args()
    try:
        record = run_pipeline(args.input_root, args.artifact_root)
    except Exception as error:
        print(f"Pipeline failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(
        f"Pipeline succeeded: run_id={record['run_id']} B=20 P={record['prediction_count']} "
        f"positive={record['positive_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
