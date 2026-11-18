import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .features import model_matrix
from .train import build_model, validate_training_matrix


def loocv(df: pd.DataFrame, experiments=("E0", "E1", "E2", "E3")) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df.label_binary.astype(int).to_numpy(); pred_rows=[]; summaries=[]
    for exp in experiments:
        features, _ = build_model(exp)
        X = pd.DataFrame({"_constant_baseline":np.zeros(len(df))}) if exp=="E0" else model_matrix(df, features)
        if exp != "E0": validate_training_matrix(X)
        preds=[]; scores=[]
        for tr, te in LeaveOneOut().split(X):
            if exp != "E0": validate_training_matrix(X.iloc[tr])
            _, model = build_model(exp); model.fit(X.iloc[tr], y[tr]); p=int(model.predict(X.iloc[te])[0])
            score=float(model.predict_proba(X.iloc[te])[0, list(model.classes_).index(1)]) if 1 in model.classes_ else 0.0
            preds.append(p); scores.append(score)
            pred_rows.append({"experiment":exp,"id":int(df.iloc[te[0]].id),"source_dataset":"B enterprise labeled markdown","source_file":df.iloc[te[0]].source_file,"source_sha256":df.iloc[te[0]].source_sha256,"model_version":"b-company-gbdt-v4.0","y_true":int(y[te[0]]),"y_pred":p,"model_score":score})
        cm=confusion_matrix(y,preds,labels=[0,1])
        summaries.append({"experiment":exp,"features":"none (constant Dummy)" if exp=="E0" else ";".join(features),"accuracy":accuracy_score(y,preds),"balanced_accuracy":balanced_accuracy_score(y,preds),"precision":precision_score(y,preds,zero_division=0),"recall":recall_score(y,preds,zero_division=0),"f1":f1_score(y,preds,zero_division=0),"tn":int(cm[0,0]),"fp":int(cm[0,1]),"fn":int(cm[1,0]),"tp":int(cm[1,1]),"error_ids":";".join(str(df.iloc[i].id) for i,(a,b) in enumerate(zip(y,preds)) if a!=b)})
    return pd.DataFrame(summaries), pd.DataFrame(pred_rows)
