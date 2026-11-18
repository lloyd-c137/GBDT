import json
from pathlib import Path

from src.extract import extract_zip
from src.schema import FEATURES
from tests.helpers import write_archives

ROOT = Path(__file__).resolve().parents[1]


def test_schema_covers_b_and_p_fields_and_excludes_personal_fields_from_features(tmp_path):
    schema = json.loads((ROOT / "data/schema.json").read_text(encoding="utf-8"))
    bzip, pzip = write_archives(tmp_path / "raw")
    b, _ = extract_zip(bzip, "B")
    p, _ = extract_zip(pzip, "P")
    definitions = {column["name"]: column for column in schema["columns"]}
    assert set(b.columns) == set(definitions)
    assert set(p.columns) == set(definitions) - {"label_grade", "label_binary"}
    assert schema["model_input_features"] == FEATURES
    for name in ("candidate_name", "age_raw", "age_years", "current_monthly_salary_raw", "employment_start_raw"):
        assert definitions[name]["sensitive"] is True
        assert definitions[name]["model_feature"] is False
    assert "label_grade" in b and "label_binary" in b
    assert "label_grade" not in p and "label_binary" not in p
    assert b.id.is_unique and p.id.is_unique
