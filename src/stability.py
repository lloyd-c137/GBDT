import numpy as np
import pandas as pd
from .features import model_matrix
from .schema import FEATURES
from .train import build_model


def analyze(df: pd.DataFrame, p: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    y=df.label_binary.astype(int).to_numpy(); X=model_matrix(df,FEATURES); XP=model_matrix(p,FEATURES)
    _, full=build_model("E2"); full.fit(X,y); full_labels=full.predict(XP).astype(int); full_scores=full.predict_proba(XP)[:,list(full.classes_).index(1)]
    scores=[]
    for i in range(len(df)):
        keep=np.arange(len(df))!=i; _, model=build_model("E2"); model.fit(X.iloc[keep],y[keep]); scores.append(model.predict_proba(XP)[:,list(model.classes_).index(1)])
    arr=np.asarray(scores); lows=X.work_years.min(); highs=X.work_years.max(); seen=set(map(tuple,X.dropna().to_numpy()))
    rows=[]
    for j, r in p.iterrows():
        combination=tuple(XP.iloc[j].to_numpy())
        outside=bool(pd.notna(r.work_years) and (r.work_years<lows or r.work_years>highs)) or (not any(pd.isna(v) for v in combination) and combination not in seen)
        vals=arr[:,j]; pred=int(full_labels[j]); changed=bool(np.any((vals>=0.5)!=pred))
        rows.append({"id":int(r.id),"model_score":float(full_scores[j]),"predicted_label":pred,"stability_positive_count":int(np.sum(vals>=0.5)),"prediction_changed":changed,"score_std":float(np.std(vals,ddof=0)),"score_min":float(vals.min()),"score_max":float(vals.max()),"out_of_training_range":outside,"review_flag":bool(outside or changed or r.acm_status=="unknown" or pd.isna(r.work_years))})
    models=[{"held_out_id":int(df.iloc[i].id),"p_predictions":arr[i].astype(float).tolist()} for i in range(len(df))]
    return pd.DataFrame(rows),models
