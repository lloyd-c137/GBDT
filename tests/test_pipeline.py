from pathlib import Path
from src.extract import extract_zip
from src.evaluate import loocv
from src.stability import analyze
ROOT=Path(__file__).resolve().parents[1]

def test_loocv_and_stability_real_data():
    b,_=extract_zip(ROOT/'data/raw/b企业.zip','B'); p,_=extract_zip(ROOT/'data/raw/p市场.zip','P')
    comparison,preds=loocv(b)
    assert len(preds)==80 and all(preds.groupby('experiment').size()==20)
    assert set(comparison.experiment)=={'E0','E1','E2','E3'}
    stability,perturb=analyze(b,p)
    assert len(stability)==260 and len(perturb)==20
    assert stability.model_score.between(0,1).all()

def test_fixed_environment_predictions_are_reproducible():
    from src.predict import fit_final
    from src.features import model_matrix
    import numpy as np
    b,_=extract_zip(ROOT/'data/raw/b企业.zip','B'); p,_=extract_zip(ROOT/'data/raw/p市场.zip','P')
    x=model_matrix(p)
    a=fit_final(b).predict_proba(x); c=fit_final(b).predict_proba(x)
    assert np.array_equal(a,c)
