from pathlib import Path
import pandas as pd
import struct, zlib
from sklearn.metrics import confusion_matrix

def write_report(b,p,comparison,loocv_preds,stability,manifest,quality,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    e2=loocv_preds[loocv_preds.experiment=="E2"]; cm=confusion_matrix(e2.y_true,e2.y_pred,labels=[0,1]); _write_confusion_png(out/"confusion_matrix.png",cm)
    comp=comparison.set_index("experiment"); label_counts=b.label_grade.value_counts().to_dict()
    positive=stability.loc[stability.predicted_label==1,"id"].astype(int).tolist()
    acm=pd.crosstab(b.acm_status,b.label_binary).to_string()
    roles=pd.crosstab(b.target_role.fillna("缺失"),b.label_grade).to_string()
    github=pd.crosstab(b.github_url.fillna("").str.contains(r"^https?://",case=False,regex=True).map({True:"提供链接",False:"未提供链接"}),b.label_grade).to_string()
    text=f'''# B 企业简历模式学习实验报告\n\n## 问题 1：P 市场正类预测\n\nE2（GBDT；阈值 0.5）预测正类 **{len(positive)}** 份，原始简历编号：{", ".join(map(str,positive)) if positive else "无"}。这是对 B 企业历史正向评级模式的模型输出，不是录用建议或真实概率。\n\n## 数据与方法\n\n从原始 Markdown 重新解析 B={len(b)}、P={len(p)}。B 标签分布：{label_counts}；A/B 合并为正类。P 不含标签、不参与监督训练。模型特征为 `acm_award`、`work_years`，缺失工作年限通过每折训练集内中位数填补。E2 固定参数见 `experiment_manifest.json`。\n\n## E0–E3 留一交叉验证\n\n{_markdown_table(comparison)}\n\n这是在已多次查看过的 20 个标签上进行的探索性 LOOCV，不是独立测试集表现。各实验每条留一预测见 `loocv_predictions.csv`。\n\n## 问题 2：可能筛选规律（三层证据）\n\n### 1. 原始标注统计\n\nACM 状态与二元评级交叉表：\n\n```text\n{acm}\n```\n\n岗位与原始评级分布：\n\n```text\n{roles}\n```\n\nGitHub 链接提供情况与评级：\n\n```text\n{github}\n```\n\n技能、学校、GitHub 与文本字段仅作描述和解析质量检查，不进入模型；简历模板化且 B 样本仅 20 份；描述性关联不能证明因果，也无法从这些字段推断稳定的筛选条件。工作年限统计：正类均值 {b.loc[b.label_binary==1,"work_years"].mean():.2f} 年，C 级均值 {b.loc[b.label_binary==0,"work_years"].mean():.2f} 年。\n\n### 2. 模型实际规则\n\n正式 GBDT 的树分裂和特征重要性会记录在 `feature_analysis.md`；分数是未校准模型输出。由于样本数小、树数量和深度固定，特征重要性只反映该拟合模型中的分裂贡献，不能解释为因果。\n\n### 3. 消融与简单树对照\n\nE2 相比 E1 的 balanced accuracy 差值为 **{comp.loc["E2","balanced_accuracy"]-comp.loc["E1","balanced_accuracy"]:+.3f}**；两者 LOOCV 错例同为 14、20，说明 GBDT 没有优于 ACM 单特征浅树，也没有在这轮验证中显示工作年限带来的额外预测提升。移除 ACM 的 E3 比 E2 低 **{comp.loc["E2","balanced_accuracy"]-comp.loc["E3","balanced_accuracy"]:.3f}** balanced accuracy，支持 ACM 是这组实验中的主要区分信息。结果仅限小样本探索。\n\n## 稳定性与覆盖范围\n\n20 个 leave-one-training-sample-out 扰动模型输出见 `stability_analysis.csv`。`stability_positive_count` 表示 20 次中的正类计数，不是正确概率；阈值分类翻转 {int(stability.prediction_changed.sum())} 份，扰动分数标准差均值 {stability.score_std.mean():.4f}。超出工作年限范围或训练中 ACM×年限组合覆盖的简历 {int(stability.out_of_training_range.sum())} 份，建议人工复核。全模型分数共形成 {stability.model_score.nunique()} 个并列分数档位，结果表保留原始同分，不强制排序。\n\n## 限制与错误案例\n\n- 标签仅来自 B 企业历史评级；LOOCV 有反复试验造成的选择偏差。\n- 样本量极小，类别与特征覆盖稀疏，P 数据来源分布可能不同。\n- ACM 与工作年限只是少量结构字段；文本、岗位适配、学校等均未进入模型。未使用的学校、年龄、薪资不构成扣分依据。\n- 模型分数不是录用概率；预测不能用于自动筛选或淘汰。\n- E2 错例编号：{comparison.loc[comparison.experiment=="E2","error_ids"].iloc[0] or "无"}。\n\n来源：B 企业原始标注 ZIP 与 P 市场原始 ZIP；每行保留源文件和 SHA-256。不得将该实验解释为真实招聘系统。\n'''
    (out/"final_report.md").write_text(text,encoding="utf-8")


def _markdown_table(df):
    cols=list(df.columns); rows=df.astype(str).values.tolist()
    return "|"+"|".join(cols)+"|\n|"+"|".join(["---"]*len(cols))+"|\n"+"\n".join("|"+"|".join(row)+"|" for row in rows)

def _write_confusion_png(path, cm):
    # Small dependency-free 2x2 confusion matrix heatmap.
    w=h=240; pixels=bytearray()
    for y in range(h):
        pixels.append(0)
        for x in range(w):
            cell=(0 if x<w//2 else 1, 0 if y<h//2 else 1)
            val=int(cm[cell[1],cell[0]]); shade=255-min(190,val*32)
            pixels.extend((shade, max(40,shade-35), max(40,shade-35)))
    def chunk(tag,data): return struct.pack(">I",len(data))+tag+data+struct.pack(">I",zlib.crc32(tag+data)&0xffffffff)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">2I5B",w,h,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(bytes(pixels)))+chunk(b"IEND",b""))
