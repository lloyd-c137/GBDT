from __future__ import annotations
import hashlib, re, zipfile
from pathlib import Path, PurePosixPath
import pandas as pd


def _decode_name(name: str) -> str:
    # ZIP's UTF-8 flag is set, but some archives stored UTF-8 bytes as CP437 text.
    try:
        candidate = name.encode("cp437").decode("utf-8")
        if any("\u4e00" <= c <= "\u9fff" for c in candidate): return candidate
    except (UnicodeEncodeError, UnicodeDecodeError): pass
    return name


def _safe_member(name: str) -> bool:
    decoded = _decode_name(name).replace("\\", "/")
    path = PurePosixPath(decoded)
    return not path.is_absolute() and ".." not in path.parts


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*`>#]", "", s)).strip(" ：:|；;\t")


def _fields(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    aliases = {
        "work_years": r"工作年限|工作经历|工作经验|工作时间",
        "target_role": r"应聘方向|目标岗位|求职意向|应聘岗位|岗位方向",
        "skills_raw": r"技能|技术栈|专业技能",
        "work_summary": r"工作内容|工作职责",
        "project_duties": r"主要职责|项目职责|负责内容",
        "github_url": r"GitHub|Github|github",
        "school_name": r"本科|学校|毕业院校|教育背景",
        "acm": r"ACM",
    }
    # Convert markdown table rows and quoted/list lines into one field stream.
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|"):
            line = line.strip("|").replace("|", "；")
        line = re.sub(r"^\s*(?:>+\s*|[-*+]\s+|\d+[.)]\s+)", "", line)
        if line and not line.startswith("#"): lines.append(line)
    stream = "\n".join(lines)
    # Split same-line Chinese semicolon attributes as well as separate lines.
    keys = [(key, re.compile(rf"(?P<k>{pat})\s*[:：]\s*", re.I)) for key, pat in aliases.items()]
    found = []
    for key, pat in keys:
        for m in pat.finditer(stream): found.append((m.start(), m.end(), key))
    found.sort()
    for i, (start, end, key) in enumerate(found):
        stop = found[i+1][0] if i+1 < len(found) else len(stream)
        line_end = stream.find("\n", end)
        if line_end >= 0: stop = min(stop, line_end)
        val = _clean(stream[end:stop])
        if val and key not in out: out[key] = val
    # Work-years commonly appear as a clause: “近 7 年工作经历”.
    m = re.search(r"(?:近\s*)?(\d+(?:\.\d+)?)\s*年\s*(?:工作经历|工作经验|工作年限)", text)
    if m: out["work_years"] = m.group(1)
    else:
        m = re.search(r"工作年限\s*[:：]\s*(\d+(?:\.\d+)?)\s*年?", stream)
        if m: out["work_years"] = m.group(1)
    return out


def _acm(value: str | None) -> tuple[str, int | None]:
    if not value: return "unknown", None
    v = value.lower()
    if "银奖" in v: return "silver", 1
    if "铜奖" in v: return "bronze", 1
    if "荣誉奖" in v or "honorable mention" in v: return "honorable_mention", 1
    if "未获奖" in v or "没有获奖" in v: return "participated_no_award", 0
    if "未参加" in v or "未参赛" in v or "没有参加" in v: return "not_participated", 0
    if "无" in v and ("获奖记录" in v or "竞赛" in v): return "no_award_record", 0
    return "unknown", None


def _label(text: str) -> str | None:
    m = re.search(r"企业考评结果\s*(?:\n|\s)*\*{0,2}\s*([ABC])\s*级", text, re.I)
    return m.group(1).upper() if m else None


def parse_markdown(text: str, resume_id: int, source_file: str, sha256: str, business: str) -> dict:
    f = _fields(text)
    status, award = _acm(f.get("acm"))
    years = f.get("work_years")
    try: years_value = float(years) if years is not None else None
    except ValueError: years_value = None
    row = {"id": int(resume_id), "work_years": years_value, "acm_status": status, "acm_award": award,
           "target_role": f.get("target_role"), "skills_raw": f.get("skills_raw"),
           "work_summary": f.get("work_summary"), "project_duties": f.get("project_duties"),
           "github_url": f.get("github_url"), "school_name": f.get("school_name"),
           "source_file": source_file, "source_sha256": sha256}
    if business == "B":
        row["label_grade"] = _label(text)
        row["label_binary"] = (1 if row["label_grade"] in ("A", "B") else 0) if row["label_grade"] else None
    return row


def extract_zip(path: Path, business: str) -> tuple[pd.DataFrame, list[dict]]:
    rows, errors = [], []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            name = _decode_name(info.filename).replace("\\", "/")
            if info.is_dir(): continue
            if not _safe_member(info.filename):
                errors.append({"source_file": name, "reason": "unsafe zip path"}); continue
            if name.startswith("__MACOSX/") or any(p.startswith(".") for p in PurePosixPath(name).parts): continue
            if not name.lower().endswith((".md", ".markdown")): continue
            basename = PurePosixPath(name).name
            m = re.search(r"(\d+)(?=\.(?:md|markdown)$)", basename, re.I)
            if not m:
                errors.append({"source_file": name, "reason": "missing numeric resume id"}); continue
            raw = zf.read(info)
            digest = hashlib.sha256(raw).hexdigest()
            try: text = raw.decode("utf-8-sig")
            except UnicodeDecodeError as e:
                errors.append({"source_file": name, "reason": f"utf-8 decode: {e}"}); continue
            row = parse_markdown(text, int(m.group(1)), name, digest, business)
            rows.append(row)
            if row["acm_status"] == "unknown": errors.append({"source_file": name, "reason": "ACM status could not be parsed"})
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True), errors
