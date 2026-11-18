from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from .schema import FEATURES

PARAMS = dict(loss="log_loss", n_estimators=40, learning_rate=0.05, max_depth=1, min_samples_leaf=2, subsample=1.0, random_state=42)

def build_model(experiment: str) -> tuple[list[str], BaseEstimator]:
    if experiment == "E0": return [], DummyClassifier(strategy="most_frequent")
    if experiment == "E1": return ["acm_award"], Pipeline([("impute", SimpleImputer(strategy="most_frequent", add_indicator=True, keep_empty_features=True)), ("model", DecisionTreeClassifier(max_depth=1, random_state=42))])
    if experiment == "E2": return FEATURES.copy(), Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("model", GradientBoostingClassifier(**PARAMS))])
    if experiment == "E3": return ["work_years"], Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("model", GradientBoostingClassifier(**PARAMS))])
    raise ValueError(f"unknown experiment: {experiment}")


def validate_training_matrix(X):
    empty = [str(column) for column in X.columns if X[column].isna().all()]
    if empty:
        raise ValueError(f"training feature(s) are entirely missing and unsupported: {empty}")
