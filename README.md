# B Company Resume Matching · GBDT V4.1

Offline experiment of historical B enterprise rating patterns. The pipeline reads only `data/raw/b企业.zip` and `data/raw/p市场.zip`; it does not read G enterprise data. Source ZIP files are never modified.

## Run

Requires Python 3.11. From the project directory:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
python -m pytest -q
```

Each invocation gets a unique Run ID and writes first to `runs/<run_id>/artifacts`. It parses and validates both archives, produces the four LOOCV experiments, 20 perturbation models, the final E2 model, all P predictions, diagnostics, and report. Only a run whose outputs pass consistency checks is published to the standard `data/processed`, `models`, and `outputs` convenience paths. `current_success.json` atomically identifies the authoritative, fully validated run under `runs/<run_id>/artifacts`. Failed runs are recorded as failed and do not publish partial results; interrupted directory promotion is rolled back from its transaction journal on the next invocation.

Use `--input-root` and `--artifact-root` for isolated data and output locations:

```bash
python run_pipeline.py --input-root /path/to/raw --artifact-root /tmp/gbdt-run
```

## Model

Target is B label A/B=1, C=0. E0 is a featureless `DummyClassifier`; E1 is a depth-1 decision tree on `acm_award`; E2 is the fixed-parameter GBDT on `acm_award` and `work_years`; E3 is the same GBDT on `work_years` only. Imputers are fit inside each training fold. An all-missing training feature fails clearly. The E2 threshold is fixed at 0.5. P is not used for supervised training or parameter selection. Scores are uncalibrated outputs, not hiring probabilities.

`work_years_out_of_range` means only that work years fall outside the observed B training minimum and maximum. `unseen_feature_combination` means an exact `(acm_award, work_years)` pair was not observed in B. This is a descriptive coverage count, not an out-of-distribution detector, a model failure, or a hiring criterion. It does not by itself set `review_flag`. Review flags cover missing key features, numeric range overflow, unknown ACM status, or prediction label changes in the stability experiment.

## Outputs

- `data/processed/b_train.csv`, `p_predict.csv`: separate B/P data tables with original source hashes.
- `data/processed/data_quality.json`: row counts, IDs, missing fields, label and ACM counts, hashes, and parsing errors.
- `models/gbdt.joblib`: final E2 pipeline fitted on all B labels.
- `outputs/model_comparison.csv`, `loocv_predictions.csv`: E0–E3 metrics and per-resume validation predictions.
- `outputs/p_predictions.csv`, `p_positive_candidates.csv`: full P results and the positive subset.
- `outputs/stability_analysis.csv`: perturbation and coverage diagnostics; join to predictions by `id`.
- `outputs/stability_perturbation_scores.csv`: scores from each of the 20 held-out-training-sample models.
- `outputs/confusion_matrix.png`: labeled axes and TN, FP, FN, TP counts.
- `outputs/feature_analysis.md`, `experiment_manifest.json`, `final_report.md`.

The report contains all P predictions and separates numeric range overflow from exact-combination coverage. Tied model scores remain tied.

LOOCV is exploratory, not independent test performance. This experiment learns associations in a tiny historical sample and must not be used for automated hiring decisions.

## CI and data protection

GitHub Actions runs pytest and the end-to-end pipeline with synthetic resumes in an isolated temporary directory. Sparse checkout omits raw archives and processed resume records. The workflow uploads no artifacts and prints no resume content.
