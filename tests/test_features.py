import pandas as pd
import pytest
from src.features import model_matrix

def test_feature_order_and_forbidden_fields():
    df=pd.DataFrame({'acm_award':[1], 'work_years':[4], 'id':[1], 'label_grade':['A']})
    assert model_matrix(df).columns.tolist()==['acm_award','work_years']
    with pytest.raises(ValueError): model_matrix(df,['id'])
    with pytest.raises(ValueError): model_matrix(df,['acm_award','label_grade'])
