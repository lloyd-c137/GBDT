import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from src.extract import _acm, _decode_name, extract_zip, parse_markdown
from tests.helpers import ACM_STATUSES, write_archives

ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = os.environ.get("GBDT_SKIP_REAL_DATA") != "1"


@pytest.mark.skipif(not REAL_DATA or not (ROOT / "data/raw/b企业.zip").exists(), reason="real resume data excluded from CI")
def test_real_archives_have_expected_rows_and_labels():
    b, errors = extract_zip(ROOT / "data/raw/b企业.zip", "B")
    p, p_errors = extract_zip(ROOT / "data/raw/p市场.zip", "P")
    assert not errors and not p_errors
    assert len(b) == 20 and b.id.tolist() == list(range(1, 21))
    assert b.label_grade.value_counts().to_dict() == {"B": 8, "C": 8, "A": 4}
    assert len(p) == 260 and p.id.tolist() == list(range(1, 261))
    assert "label_grade" not in p.columns and "label_binary" not in p.columns
    extended = ["candidate_name", "age_raw", "age_years", "employment_start_raw", "teamwork_summary", "career_direction", "project_background"]
    assert not b[extended].isna().any().any() and not p[extended].isna().any().any()


def test_zip_utf8_name_recovery_and_macos_metadata_exclusion(tmp_path):
    assert _decode_name("bΣ╝üΣ╕Ü/τ«ÇσÄå_001.md") == "b企业/简历_001.md"
    bzip, pzip = write_archives(tmp_path / "raw")
    b, errors = extract_zip(bzip, "B")
    p, p_errors = extract_zip(pzip, "P")
    assert not errors and not p_errors
    assert len(b) == 20 and len(p) == 260


def test_acm_categories_unknown_and_negative_phrases():
    expected = ["silver", "bronze", "honorable_mention", "participated_no_award", "not_participated", "no_award_record"]
    assert [_acm(value)[0] for value in ACM_STATUSES] == expected
    assert _acm("参加竞赛，未获奖") == ("participated_no_award", 0)
    assert _acm("有无ACM经历待核实")[1] is None


def test_inline_fields_preserve_csharp_cpp_and_multiline_values():
    text = """### 竞赛
> **ACM**：无 ACM 竞赛获奖记录；GitHub：https://example.test/u；技能：C#、C++、TypeScript
### 工作内容
> 工作内容：实现 C# API
> 继续维护第二行正文。
> **项目职责**：服务某产品；支持下一代改造。
### 岗位
> 应聘方向：后端开发工程师
### 教育
|本科|某大学，计算机专业|
"""
    row = parse_markdown(text, 1, "简历_001.md", "abc", "P")
    assert row["acm_status"] == "no_award_record" and row["acm_award"] == 0
    assert row["github_url"] == "https://example.test/u"
    assert row["skills_raw"] == "C#、C++、TypeScript"
    assert "第二行正文" in row["work_summary"] and "C#" in row["work_summary"]
    assert "下一代改造" in row["project_duties"]
    assert row["target_role"] == "后端开发工程师"
    assert row["school_name"] == "某大学，计算机专业"


def test_personal_and_context_fields_are_preserved_without_model_feature_changes():
    text = """### 个人信息
> **姓名**：测试候选人001；年龄：28岁；当前月薪：18,000元/月
### 工作经历
> 参加工作时间：2019年3月；近 7 年工作经历：工程师
> 团队协作：参与跨团队交付
> 职业方向：后端平台
### 项目经历
> 项目背景：支付服务改造
> 主要职责：开发 API
"""
    row = parse_markdown(text, 1, "简历_001.md", "abc", "P")
    assert row["candidate_name"] == "测试候选人001"
    assert row["age_raw"] == "28岁" and row["age_years"] == 28.0
    assert row["current_monthly_salary_raw"] == "18,000元/月"
    assert row["employment_start_raw"] == "2019年3月"
    assert row["work_years"] == 7.0
    assert row["work_years_raw"] == "近 7 年工作经历：工程师"
    assert row["teamwork_summary"] == "参与跨团队交付"
    assert row["career_direction"] == "后端平台"
    assert row["project_background"] == "支付服务改造"


def test_unparseable_acm_and_zip_slip_are_reported(tmp_path):
    source = tmp_path / "bad.zip"
    with ZipFile(source, "w", ZIP_DEFLATED) as archive:
        archive.writestr("__MACOSX/._简历_001.md", b"junk")
        archive.writestr("../简历_002.md", b"junk")
        archive.writestr("简历_001.md", "应聘方向：测试开发工程师\nACM：经历未知\n".encode())
    frame, errors = extract_zip(source, "P")
    assert len(frame) == 1 and frame.iloc[0].id == 1
    assert any("ACM status" in error["reason"] for error in errors)
    assert any("unsafe zip path" in error["reason"] for error in errors)
