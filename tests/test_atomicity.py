import json
from pathlib import Path

import pytest

import run_pipeline as pipeline
from tests.helpers import replace_resume, write_archives, _resume


def _seed_previous_success(root: Path) -> None:
    (root / "outputs").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "data/processed").mkdir(parents=True)
    (root / "outputs/previous.txt").write_text("previous output")
    (root / "models/previous.txt").write_text("previous model")
    (root / "data/processed/previous.txt").write_text("previous data")
    (root / "current_success.json").write_text('{"run_id":"previous"}')


def _assert_failed_without_publish(root: Path, run_id: str) -> None:
    record = json.loads((root / "runs" / run_id / "run_manifest.json").read_text())
    assert record["status"] == "failed"
    assert (root / "outputs/previous.txt").read_text() == "previous output"
    assert (root / "models/previous.txt").read_text() == "previous model"
    assert (root / "data/processed/previous.txt").read_text() == "previous data"
    assert json.loads((root / "current_success.json").read_text())["run_id"] == "previous"


def test_missing_zip_records_failure_and_keeps_published_result(tmp_path):
    raw = tmp_path / "raw"
    bzip, pzip = write_archives(raw)
    pzip.unlink()
    artifact_root = tmp_path / "artifacts"
    _seed_previous_success(artifact_root)
    with pytest.raises(ValueError, match="not a readable ZIP"):
        pipeline.run_pipeline(raw, artifact_root, run_id="missing-input")
    _assert_failed_without_publish(artifact_root, "missing-input")


def test_abnormal_b_label_fails_before_training_and_publish(tmp_path):
    raw, artifact_root = tmp_path / "raw", tmp_path / "artifacts"
    bzip, _ = write_archives(raw)
    replace_resume(bzip, "简历_001.md", _resume(1, 2, "获得 ACM-ICPC 区域赛银奖", "D"))
    _seed_previous_success(artifact_root)
    with pytest.raises(ValueError, match="B labels"):
        pipeline.run_pipeline(raw, artifact_root, run_id="bad-label")
    _assert_failed_without_publish(artifact_root, "bad-label")


def test_unknown_acm_fails_before_training_and_publish(tmp_path):
    raw, artifact_root = tmp_path / "raw", tmp_path / "artifacts"
    bzip, _ = write_archives(raw)
    replace_resume(bzip, "简历_001.md", _resume(1, 2, "ACM 经历待核实", "A"))
    _seed_previous_success(artifact_root)
    with pytest.raises(ValueError, match="parsing failed"):
        pipeline.run_pipeline(raw, artifact_root, run_id="bad-acm")
    _assert_failed_without_publish(artifact_root, "bad-acm")


@pytest.mark.parametrize("failure_point", ["model", "report"])
def test_model_or_report_failure_never_publishes_partial_run(tmp_path, monkeypatch, failure_point):
    raw, artifact_root = tmp_path / "raw", tmp_path / "artifacts"
    write_archives(raw)
    _seed_previous_success(artifact_root)
    if failure_point == "model":
        monkeypatch.setattr(pipeline, "fit_final", lambda _: (_ for _ in ()).throw(RuntimeError("injected model failure")))
    else:
        monkeypatch.setattr(pipeline, "write_report", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected report failure")))
    with pytest.raises(RuntimeError, match=f"injected {failure_point} failure"):
        pipeline.run_pipeline(raw, artifact_root, run_id=f"{failure_point}-failure")
    _assert_failed_without_publish(artifact_root, f"{failure_point}-failure")


def test_directory_promotion_rolls_back_if_a_swap_fails(tmp_path, monkeypatch):
    root, run_id = tmp_path / "artifact-root", "rollback-case"
    run_dir = root / "runs" / run_id
    for relative in ("data/processed", "models", "outputs"):
        (run_dir / "artifacts" / relative).mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts" / relative / "new.txt").write_text("new")
        (root / relative).mkdir(parents=True, exist_ok=True)
        (root / relative / "old.txt").write_text("old")
    pointer = root / "current_success.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text('{"run_id":"old"}')
    record = {"run_id": run_id, "status": "running"}
    original_replace = pipeline.os.replace
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        if not failed and str(destination).endswith("/models") and "/publish_staging/" in str(source):
            failed = True
            raise OSError("injected promotion failure")
        return original_replace(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", fail_once)
    with pytest.raises(OSError, match="promotion failure"):
        pipeline._publish_success(root, run_dir, run_id, record)
    for relative in ("data/processed", "models", "outputs"):
        assert (root / relative / "old.txt").read_text() == "old"
        assert not (root / relative / "new.txt").exists()
    assert json.loads(pointer.read_text())["run_id"] == "old"
