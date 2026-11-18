import pandas as pd
from .schema import FEATURES

FORBIDDEN = {"id", "source_file", "source_sha256", "label_grade", "label_binary", "school_name", "github_url", "work_summary", "project_duties", "skills_raw"}

def model_matrix(df: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    if set(features) & FORBIDDEN: raise ValueError(f"forbidden model feature(s): {sorted(set(features) & FORBIDDEN)}")
    if not set(features) <= set(FEATURES): raise ValueError("feature set is not an approved experiment")
    return df.loc[:, features].copy()
