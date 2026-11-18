from pathlib import Path
from src.extract import extract_zip
from src.features import model_matrix
from src.train import build_model
ROOT=Path(__file__).resolve().parents[1]

def test_p_is_unlabeled_and_does_not_enter_supervised_train():
    b,_=extract_zip(ROOT/'data/raw/b企业.zip','B'); p,_=extract_zip(ROOT/'data/raw/p市场.zip','P')
    x=model_matrix(b,build_model('E2')[0]); xp=model_matrix(p,build_model('E2')[0])
    assert x.columns.tolist()==xp.columns.tolist()==['acm_award','work_years']
    assert 'label_grade' not in xp and 'label_binary' not in xp
    assert len(x)==20 and len(xp)==260
