from __future__ import annotations
import hashlib, json, platform, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
import joblib, pandas as pd, sklearn, numpy, scipy
from src.extract import extract_zip
from src.evaluate import loocv
from src.features import model_matrix
from src.predict import fit_final
from src.report import write_report
from src.stability import analyze
from src.schema import FEATURES
from src.train import PARAMS

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data/raw'; PROC=ROOT/'data/processed'; OUT=ROOT/'outputs'; MODELS=ROOT/'models'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def zip_file_counts(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
    return {"archive_entries":len(names),"markdown_resumes":sum(n.lower().endswith((".md",".markdown")) and not n.startswith("__MACOSX/") for n in names),"macos_metadata":sum(n.startswith("__MACOSX/") for n in names)}
def main():
    bzip=RAW/'b企业.zip'; pzip=RAW/'p市场.zip'
    if not bzip.exists() or not pzip.exists(): raise FileNotFoundError(f"请将原始 ZIP 放入 {RAW}（需要 b企业.zip、p市场.zip）")
    b,be=extract_zip(bzip,'B'); p,pe=extract_zip(pzip,'P')
    errors=be+pe
    expected_b=set(range(1,21)); expected_p=set(range(1,261))
    quality={"input_files":{"b":{"sha256":sha(bzip),"rows":len(b)},"p":{"sha256":sha(pzip),"rows":len(p)}},"file_counts":{"b":zip_file_counts(bzip),"p":zip_file_counts(pzip)},"parse_errors":errors,
       "b_missing_ids":sorted(expected_b-set(b.id)) if len(b) else sorted(expected_b),"b_extra_ids":sorted(set(b.id)-expected_b) if len(b) else [],
       "p_missing_ids":sorted(expected_p-set(p.id)) if len(p) else sorted(expected_p),"p_extra_ids":sorted(set(p.id)-expected_p) if len(p) else [],
       "b_label_distribution":b.label_grade.value_counts(dropna=False).to_dict() if len(b) else {},"b_acm_distribution":b.acm_status.value_counts(dropna=False).to_dict() if len(b) else {},
       "p_acm_distribution":p.acm_status.value_counts(dropna=False).to_dict() if len(p) else {},
       "missing_fields":{"b":b.isna().sum().to_dict() if len(b) else {},"p":p.isna().sum().to_dict() if len(p) else {}},
       "data_types":{"b":b.dtypes.astype(str).to_dict() if len(b) else {},"p":p.dtypes.astype(str).to_dict() if len(p) else {}},
       "type_anomalies":{"b_invalid_work_years":int(b.work_years.notna().sum()-pd.to_numeric(b.work_years,errors="coerce").notna().sum()) if len(b) else 0,"p_invalid_work_years":int(p.work_years.notna().sum()-pd.to_numeric(p.work_years,errors="coerce").notna().sum()) if len(p) else 0},
       "work_years_range":{"b":[float(b.work_years.min()),float(b.work_years.max())] if b.work_years.notna().any() else None,"p":[float(p.work_years.min()),float(p.work_years.max())] if p.work_years.notna().any() else None},
       "duplicate_hashes":{"b":int(b.source_sha256.duplicated().sum()),"p":int(p.source_sha256.duplicated().sum())},
       "duplicate_text_rate":{"b":float(b.duplicated(subset=['work_summary','project_duties','skills_raw']).mean()),"p":float(p.duplicated(subset=['work_summary','project_duties','skills_raw']).mean())}}
    PROC.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
    b.to_csv(PROC/'b_train.csv',index=False); p.to_csv(PROC/'p_predict.csv',index=False)
    (PROC/'data_quality.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    if errors or len(b)!=20 or len(p)!=260 or set(b.id)!=expected_b or set(p.id)!=expected_p: raise ValueError(f"核心数据质量失败，已保存审计报告：{PROC/'data_quality.json'}")
    if b.label_grade.isna().any() or b.label_grade.value_counts().to_dict()!={'B':8,'C':8,'A':4}: raise ValueError("B 标签缺失或分布不符合 4/8/8")
    if b.acm_award.isna().any() or p.acm_award.isna().any(): raise ValueError("ACM 状态无法解析；检查数据质量 JSON")
    required=["target_role","skills_raw","work_summary","project_duties","github_url","school_name","source_file","source_sha256"]
    missing={"b":{k:int(b[k].isna().sum()) for k in required if b[k].isna().any()},"p":{k:int(p[k].isna().sum()) for k in required if p[k].isna().any()}}
    if missing["b"] or missing["p"]: raise ValueError(f"核心文本字段解析缺失：{missing}")
    if p.label_grade.notna().any() if 'label_grade' in p else False: raise ValueError("P 不应包含企业真实评级")
    comparison,preds=loocv(b)
    comparison["source_dataset"]="B enterprise labeled markdown"; comparison["model_version"]="b-company-gbdt-v4.0"
    comparison.to_csv(OUT/'model_comparison.csv',index=False); preds.to_csv(OUT/'loocv_predictions.csv',index=False)
    stability,perturb=analyze(b,p); final=fit_final(b); joblib.dump(final,MODELS/'gbdt.joblib')
    final_input=model_matrix(p); final_scores=final.predict_proba(final_input)[:,list(final.classes_).index(1)]
    stability['model_score']=final_scores; stability['predicted_label']=(final_scores>=0.5).astype(int)
    base={"source_dataset":"P market raw markdown","model_version":"b-company-gbdt-v4.0","threshold":0.5}
    for key,val in base.items(): stability[key]=val
    stability["source_file"]=p.source_file.to_numpy(); stability["source_sha256"]=p.source_sha256.to_numpy()
    stability.to_csv(OUT/'p_predictions.csv',index=False)
    stability[stability.predicted_label==1].to_csv(OUT/'p_positive_candidates.csv',index=False)
    stability.to_csv(OUT/'stability_analysis.csv',index=False)
    p_rounds=pd.DataFrame([{"held_out_id":r['held_out_id'],**{str(int(i+1)):float(v) for i,v in enumerate(r['p_predictions'])}} for r in perturb]); p_rounds.to_csv(OUT/'_stability_rounds.csv',index=False)
    manifest={"model_version":"b-company-gbdt-v4.0","built_at_utc":datetime.now(timezone.utc).isoformat(),"python_version":platform.python_version(),"scikit_learn_version":sklearn.__version__,"pandas_version":pd.__version__,"numpy_version":numpy.__version__,"scipy_version":scipy.__version__,"joblib_version":joblib.__version__,"parameters":PARAMS,"features":["acm_award","work_years"],"threshold":0.5,"random_seed":42,"source_sha256":{"b企业.zip":sha(bzip),"p市场.zip":sha(pzip)},"rows":{"b":len(b),"p":len(p)},"label_distribution":quality['b_label_distribution']}
    (OUT/'experiment_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    model=final.named_steps['model']; trees=[]
    for i,tree in enumerate(model.estimators_[:,0]):
        t=tree.tree_;
        for n in range(t.node_count):
            if t.feature[n]>=0: trees.append({"estimator":i+1,"feature":FEATURES[t.feature[n]],"threshold":float(t.threshold[n]),"samples":int(t.n_node_samples[n])})
    importance=dict(zip(FEATURES,map(float,model.feature_importances_)))
    score_counts=stability.model_score.value_counts().sort_index()
    group=p.assign(model_score=stability.model_score.to_numpy(),predicted_label=stability.predicted_label.to_numpy()).groupby("acm_award").agg(n=("id","size"),score_min=("model_score","min"),score_max=("model_score","max"),predicted_positive=("predicted_label","sum"))
    (OUT/'feature_analysis.md').write_text("# E2 模型结构分析\n\n特征重要性（拟合后分裂贡献）：\n\n"+"\n".join(f"- `{k}`: {v:.6f}" for k,v in importance.items())+"\n\n实际分裂节点：\n\n|树|特征|阈值|样本数|\n|---:|---|---:|---:|\n"+"\n".join(f"|{x['estimator']}|{x['feature']}|{x['threshold']:.6g}|{x['samples']}|" for x in trees)+"\n\nP 市场按 ACM 获奖指示的分数范围与预测：\n\n|acm_award|n|score_min|score_max|predicted_positive|\n|---:|---:|---:|---:|---:|\n"+"\n".join(f"|{idx}|{int(row.n)}|{row.score_min:.6f}|{row.score_max:.6f}|{int(row.predicted_positive)}|" for idx,row in group.iterrows())+f"\n\nP 市场出现 {len(score_counts)} 个不同的原始模型分数档位；同分候选保留并列，未生成排序。\n",encoding='utf-8')
    write_report(b,p,comparison,preds,stability,manifest,quality,OUT)
    print(f"Pipeline 完成：B={len(b)}, P={len(p)}, positive={int((stability.predicted_label==1).sum())}")
    print(comparison.to_string(index=False))
    print('positive_ids=',','.join(map(str,stability.loc[stability.predicted_label==1,'id'].tolist())))
if __name__=='__main__': main()
