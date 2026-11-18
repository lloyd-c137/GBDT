from __future__ import annotations

import pandas as pd


def explain_gbdt(pipeline, input_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Name importance and split features after the fitted imputer transform."""
    imputer = pipeline.named_steps["impute"]
    model = pipeline.named_steps["model"]
    names = list(imputer.get_feature_names_out(input_features))
    importances = model.feature_importances_
    if len(names) != len(importances):
        raise ValueError(
            "transformed feature names do not match model importances: "
            f"{len(names)} names for {len(importances)} values"
        )
    importance = pd.DataFrame({"feature": names, "importance": importances.astype(float)})
    rows = []
    for estimator_index, tree_model in enumerate(model.estimators_[:, 0], start=1):
        tree = tree_model.tree_
        for node in range(tree.node_count):
            feature_index = int(tree.feature[node])
            if feature_index >= 0:
                if feature_index >= len(names):
                    raise ValueError(f"tree feature index {feature_index} is out of range")
                rows.append(
                    {
                        "estimator": estimator_index,
                        "feature": names[feature_index],
                        "threshold": float(tree.threshold[node]),
                        "samples": int(tree.n_node_samples[node]),
                    }
                )
    return importance, pd.DataFrame(rows, columns=["estimator", "feature", "threshold", "samples"])
