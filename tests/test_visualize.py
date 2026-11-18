import json

import joblib
import numpy as np
import pandas as pd

from src.extract import extract_zip
from src.predict import fit_final
from src.schema import FEATURES
from src.train import PARAMS
from src.visualize import generate_tree_artifacts
from tests.helpers import write_archives


def test_visualizations_and_structure_are_from_all_fitted_trees(tmp_path):
    bzip, _ = write_archives(tmp_path / "raw")
    b, errors = extract_zip(bzip, "B")
    assert not errors
    model = fit_final(b)
    model_path = tmp_path / "models/gbdt.joblib"
    model_path.parent.mkdir(parents=True)
    joblib.dump(model, model_path)
    manifest_path = model_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "model_version": "test-e2",
        "features": FEATURES,
        "parameters": PARAMS,
    }), encoding="utf-8")

    figure_dir = tmp_path / "outputs/figures"
    result = generate_tree_artifacts(model_path, figure_dir, manifest_path)
    structure = pd.read_csv(tmp_path / "outputs/gbdt_tree_structure.csv")
    estimators = model.named_steps["model"].estimators_[:, 0]
    names = list(model.named_steps["impute"].get_feature_names_out(FEATURES))

    assert result["tree_count"] == len(estimators) == 40
    assert result["node_rows"] == sum(estimator.tree_.node_count for estimator in estimators)
    assert set(structure.tree_number) == set(range(1, 41))
    for tree_number, estimator in enumerate(estimators, start=1):
        rows = structure[structure.tree_number == tree_number].sort_values("node_id")
        tree = estimator.tree_
        assert rows.node_id.tolist() == list(range(tree.node_count))
        for row in rows.itertuples(index=False):
            node = row.node_id
            if tree.children_left[node] == -1:
                assert row.node_type == "leaf"
                assert np.isclose(row.weighted_leaf_output, model.named_steps["model"].learning_rate * tree.value[node].reshape(-1)[0])
            else:
                assert row.node_type == "split"
                assert row.feature == names[tree.feature[node]]
                assert np.isclose(row.threshold, tree.threshold[node])

    for filename in (
        "gbdt_ensemble_overview.svg", "gbdt_ensemble_overview.png",
        "gbdt_tree_001.svg", "gbdt_tree_001.png", "gbdt_tree_022.svg", "gbdt_tree_022.png",
        "gbdt_tree_040.svg", "gbdt_tree_040.png",
    ):
        assert (figure_dir / filename).is_file()
    pages = sorted(figure_dir.glob("gbdt_trees_page_*.svg"))
    assert len(pages) == 5
    assert all(path.stat().st_size > 1000 for path in [*pages, figure_dir / "gbdt_ensemble_overview.svg"])
