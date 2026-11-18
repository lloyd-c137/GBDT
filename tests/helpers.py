from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ACM_STATUSES = [
    "获得 ACM-ICPC 区域赛银奖",
    "获得 ACM-ICPC 区域赛铜奖",
    "获得 ACM-ICPC 区域赛荣誉奖",
    "参加 ACM-ICPC 区域赛，未获奖",
    "未参加 ACM 竞赛",
    "无 ACM 竞赛获奖记录",
]


def write_archives(raw_dir: Path) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    bzip, pzip = raw_dir / "b企业.zip", raw_dir / "p市场.zip"
    labels = ["A"] * 4 + ["B"] * 8 + ["C"] * 8
    with ZipFile(bzip, "w", ZIP_DEFLATED) as archive:
        archive.writestr("__MACOSX/._metadata", "ignored")
        for resume_id, label in enumerate(labels, start=1):
            years = 1 + resume_id % 18
            acm = ACM_STATUSES[(resume_id - 1) % len(ACM_STATUSES)]
            award = "ACM" in acm and ("奖" in acm and not ("未获奖" in acm or "无 ACM" in acm))
            if label != "C" and not award:
                acm = ACM_STATUSES[(resume_id - 1) % 3]
            archive.writestr(f"b企业/简历_{resume_id:03}.md", _resume(resume_id, years, acm, label))
    with ZipFile(pzip, "w", ZIP_DEFLATED) as archive:
        archive.writestr("__MACOSX/._metadata", "ignored")
        for resume_id in range(1, 261):
            years = 1 + (resume_id - 1) % 23
            acm = ACM_STATUSES[(resume_id - 1) % len(ACM_STATUSES)]
            archive.writestr(f"p市场/简历_{resume_id:03}.md", _resume(resume_id, years, acm))
    return bzip, pzip


def _resume(resume_id: int, years: int, acm: str, label: str | None = None) -> str:
    text = f"""# Resume {resume_id}

### Work history
> 姓名：测试候选人{resume_id:03d}；年龄：{24 + resume_id % 15} 岁；当前月薪：15,000 元/月
> 参加工作时间：2018 年 1 月；近 {years} 年工作经历：技术工程师
> 工作内容：负责 API 与 C# 服务
> 延续说明：第二行工作正文
> 团队协作：参与跨团队评审
> 职业方向：持续深耕后端工程

### Personal information
> 应聘方向：后端开发工程师

### Competition and skills
> **ACM**：{acm}；GitHub：https://example.test/{resume_id}；技能：C#、C++、TypeScript

### Education
> 本科：示例大学，计算机相关专业

## 项目经历
- 项目背景：测试用例项目
- 主要职责：持续维护项目
  支持第二行项目职责。
"""
    if label is not None:
        text += f"\n## 企业考评结果\n\n**{label} 级**\n"
    return text


def replace_resume(zip_path: Path, member_suffix: str, text: str) -> None:
    temp = zip_path.with_suffix(".replacement.zip")
    with ZipFile(zip_path) as old, ZipFile(temp, "w", ZIP_DEFLATED) as new:
        for info in old.infolist():
            if info.filename.endswith(member_suffix):
                new.writestr(info.filename, text.encode("utf-8"))
            else:
                new.writestr(info.filename, old.read(info.filename))
    temp.replace(zip_path)
