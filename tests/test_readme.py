from pathlib import Path

import pandas as pd

from scripts.update_readme import ROOT, render_results, update_readme


def test_readme_results_block_matches_committed_predictions_and_metrics():
    assert update_readme(ROOT / "README.md", ROOT, check=True)
    block = render_results(ROOT)
    predictions = pd.read_csv(ROOT / "outputs/p_predictions.csv")
    candidates = predictions.loc[predictions.predicted_label == 1, "id"].astype(int).tolist()
    assert f"正类 **{len(candidates)}** 份" in block
    assert all(str(candidate) in block for candidate in candidates)
    assert "E1/E2 单独估计工作年限的增量效果" in block
    assert "本次正式运行：`" in block


def test_readme_update_preserves_hand_written_sections(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(
        "# Example\n\n<!-- AUTO:RESULTS:START -->\n<!-- AUTO:RESULTS:END -->\n\nHand-written note.\n",
        encoding="utf-8",
    )
    assert update_readme(path, ROOT, check=False)
    assert "Hand-written note." in path.read_text(encoding="utf-8")
    assert update_readme(path, ROOT, check=True)
