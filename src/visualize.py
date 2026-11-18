from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trim_svg_whitespace(output_dir: Path) -> None:
    for path in output_dir.glob("*.svg"):
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _tree_value(tree, node: int) -> float:
    return float(np.asarray(tree.value[node]).reshape(-1)[0])


def _leaf_rows(tree, learning_rate: float) -> str:
    values = []
    for node in range(tree.node_count):
        if tree.children_left[node] == -1:
            raw = _tree_value(tree, node)
            values.append(f"leaf {node}: {raw:+.5f} × η = {learning_rate * raw:+.5f}")
    return "; ".join(values)


def _draw_tree(axis, tree_model, feature_names: list[str], tree_number: int, learning_rate: float, fontsize: float = 8) -> None:
    tree = tree_model.tree_
    artists = plot_tree(
        tree_model,
        ax=axis,
        feature_names=feature_names,
        filled=True,
        impurity=False,
        proportion=False,
        node_ids=True,
        precision=5,
        rounded=True,
        fontsize=fontsize,
    )
    node_artists = {}
    for artist in artists:
        match = re.match(r"node #(\d+)", artist.get_text())
        if match:
            node_artists[int(match.group(1))] = artist
    if set(node_artists) != set(range(tree.node_count)):
        raise RuntimeError(f"Tree {tree_number} plot nodes do not match its fitted structure")
    for node, artist in node_artists.items():
        if tree.children_left[node] == -1:
            raw = _tree_value(tree, node)
            artist.set_text(f"{artist.get_text()}\nη·value={learning_rate * raw:+.5f}")
    axis.set_title(f"Tree {tree_number:03d} · fitted regression tree · η={learning_rate:g}", fontsize=11)


def _overview(model, features: list[str], model_version: str, output_dir: Path) -> None:
    gbdt = model.named_steps["model"]
    prior = np.asarray(gbdt.init_.class_prior_, dtype=float)
    positive_index = list(gbdt.classes_).index(1)
    initial_probability = float(prior[positive_index])
    initial_score = math.log(initial_probability / (1.0 - initial_probability))
    fig, axis = plt.subplots(figsize=(16, 4.5))
    axis.axis("off")
    positions = [0.08, 0.245, 0.41, 0.575, 0.74, 0.905]
    labels = [
        "E2 inputs\n" + "\n".join(features),
        "Fitted imputer\n(train-fold fit)",
        f"Initial raw score\nF₀={initial_score:+.4f}\nP₀={initial_probability:.4f}",
        f"40 regression trees\nη={gbdt.learning_rate:g}\ndepth={gbdt.max_depth}",
        "Add leaf updates\nF₀(x) + η·Σhₘ(x)",
        "Sigmoid → score\nclass 1 if ≥ 0.5",
    ]
    colors = ["#eaf2f8", "#f4ecf7", "#fcf3cf", "#d5f5e3", "#d6eaf8", "#fadbd8"]
    for index, (x, label, color) in enumerate(zip(positions, labels, colors)):
        axis.text(
            x, 0.56, label, transform=axis.transAxes, ha="center", va="center", fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.75", "facecolor": color, "edgecolor": "#44546a", "linewidth": 1.2},
        )
        if index < len(positions) - 1:
            axis.annotate(
                "", xy=(positions[index + 1] - 0.065, 0.56), xytext=(x + 0.065, 0.56),
                xycoords=axis.transAxes, arrowprops={"arrowstyle": "->", "color": "#44546a", "lw": 1.6},
            )
    axis.text(
        0.5, 0.08,
        f"{model_version} · GradientBoostingClassifier · {gbdt.n_estimators} fitted regression trees · "
        "each leaf is a residual update, not an independent hiring decision",
        transform=axis.transAxes, ha="center", fontsize=8.5, color="#34495e",
    )
    axis.set_title("E2 · 40-tree additive model", fontsize=13, pad=18)
    fig.tight_layout()
    fig.savefig(output_dir / "gbdt_ensemble_overview.svg", bbox_inches="tight")
    fig.savefig(output_dir / "gbdt_ensemble_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _tree_structure(model, features: list[str], model_version: str, model_hash: str) -> pd.DataFrame:
    gbdt = model.named_steps["model"]
    names = list(model.named_steps["impute"].get_feature_names_out(features))
    if len(gbdt.estimators_) != gbdt.n_estimators:
        raise ValueError("fitted estimator count does not match n_estimators")
    rows = []
    for tree_number, estimator in enumerate(gbdt.estimators_[:, 0], start=1):
        tree = estimator.tree_
        for node in range(tree.node_count):
            feature_index = int(tree.feature[node])
            is_leaf = tree.children_left[node] == -1
            raw_value = _tree_value(tree, node)
            rows.append({
                "tree_number": tree_number,
                "node_id": node,
                "node_type": "leaf" if is_leaf else "split",
                "feature": None if is_leaf else names[feature_index],
                "threshold": None if is_leaf else float(tree.threshold[node]),
                "left_child_id": None if is_leaf else int(tree.children_left[node]),
                "right_child_id": None if is_leaf else int(tree.children_right[node]),
                "samples": int(tree.n_node_samples[node]),
                "impurity": float(tree.impurity[node]),
                "raw_output": raw_value,
                "learning_rate": float(gbdt.learning_rate),
                "weighted_leaf_output": float(gbdt.learning_rate * raw_value) if is_leaf else None,
                "model_version": model_version,
                "model_sha256": model_hash,
            })
    return pd.DataFrame(rows)


def generate_tree_artifacts(
    model_path: Path,
    output_dir: Path,
    manifest_path: Path | None = None,
) -> dict:
    model_path, output_dir = model_path.resolve(), output_dir.resolve()
    if manifest_path is None:
        candidates = [
            model_path.with_suffix(".manifest.json"),
            model_path.parent.parent / "outputs/experiment_manifest.json",
        ]
        manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError("model manifest not found; provide --manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("model_sha256")
    if expected_hash and _sha256(model_path) != expected_hash:
        raise ValueError("model file hash does not match its experiment manifest")
    model = joblib.load(model_path)
    gbdt = model.named_steps["model"]
    if (gbdt.n_estimators, float(gbdt.learning_rate), int(gbdt.max_depth)) != (40, 0.05, 1):
        raise ValueError("visualization only accepts the fixed V4.1 E2 model configuration")
    features = list(manifest.get("features", ["acm_award", "work_years"]))
    names = list(model.named_steps["impute"].get_feature_names_out(features))
    if names != features:
        raise ValueError(f"unexpected transformed E2 features: {names}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_hash = _sha256(model_path)
    model_version = manifest.get("model_version", "unavailable")
    structure = _tree_structure(model, features, model_version, model_hash)
    structure.to_csv(output_dir.parent / "gbdt_tree_structure.csv", index=False)
    _overview(model, features, model_version, output_dir)

    estimators = list(gbdt.estimators_[:, 0])
    representative = {1, 22, 40}
    for tree_number in sorted(representative):
        fig, axis = plt.subplots(figsize=(9, 5.5))
        _draw_tree(axis, estimators[tree_number - 1], names, tree_number, gbdt.learning_rate, fontsize=10)
        fig.text(0.5, 0.02, "Regression-tree values are residual updates; weighted contributions use the learning rate.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        for suffix, kwargs in (("svg", {}), ("png", {"dpi": 180})):
            fig.savefig(output_dir / f"gbdt_tree_{tree_number:03d}.{suffix}", bbox_inches="tight", **kwargs)
        plt.close(fig)

    for page, start in enumerate(range(0, len(estimators), 8), start=1):
        fig, axes = plt.subplots(4, 2, figsize=(14, 13))
        for offset, axis in enumerate(axes.flat):
            tree_number = start + offset + 1
            if tree_number <= len(estimators):
                _draw_tree(axis, estimators[tree_number - 1], names, tree_number, gbdt.learning_rate, fontsize=7)
            else:
                axis.axis("off")
        fig.suptitle(
            f"{model_version} · GBDT trees {start + 1:03d}–{min(start + 8, len(estimators)):03d} "
            "(each is a shallow residual regression tree)", fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(output_dir / f"gbdt_trees_page_{page:02d}.svg", bbox_inches="tight")
        plt.close(fig)

    _trim_svg_whitespace(output_dir)
    return {
        "model_version": model_version,
        "model_sha256": model_hash,
        "tree_count": len(estimators),
        "node_rows": len(structure),
        "feature_names": names,
        "output_dir": str(output_dir),
        "structure_csv": str(output_dir.parent / "gbdt_tree_structure.csv"),
    }
