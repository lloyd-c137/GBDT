from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualize import generate_tree_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the fitted V4.1 E2 GBDT ensemble and all 40 trees")
    parser.add_argument("--model", type=Path, default=ROOT / "models/gbdt.joblib")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figures")
    args = parser.parse_args()
    try:
        result = generate_tree_artifacts(args.model, args.output_dir, args.manifest)
    except Exception as error:
        print(f"Visualization failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(
        f"Rendered {result['tree_count']} actual trees for {result['model_version']}; "
        f"structure rows={result['node_rows']}; figures={result['output_dir']}"
    )
    print(f"Tree structure: {result['structure_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
