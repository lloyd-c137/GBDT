from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from src.extract import _acm, _decode_name, extract_zip, parse_markdown

ROOT=Path(__file__).resolve().parents[1]

def test_real_archives_have_expected_rows_and_labels():
    b,_=extract_zip(ROOT/'data/raw/b企业.zip','B'); p,_=extract_zip(ROOT/'data/raw/p市场.zip','P')
    assert len(b)==20 and b.id.tolist()==list(range(1,21))
    assert b.label_grade.value_counts().to_dict()=={'B':8,'C':8,'A':4}
    assert len(p)==260 and p.id.tolist()==list(range(1,261))
    assert 'label_grade' not in p.columns and 'label_binary' not in p.columns

def test_zip_utf8_name_recovery():
    assert _decode_name('bΣ╝üΣ╕Ü/τ«ÇσÄå_001.md')=='b企业/简历_001.md'

def test_acm_categories_and_negative_phrases():
    expected={'银奖':'silver','铜奖':'bronze','荣誉奖':'honorable_mention','参加竞赛未获奖':'participated_no_award','未参加竞赛':'not_participated','无竞赛获奖记录':'no_award_record'}
    for phrase,status in expected.items(): assert _acm(phrase)[0]==status
    assert _acm('参加竞赛，未获奖')==('participated_no_award',0)
    assert _acm('有无ACM经历待核实')[1] is None

def test_parse_same_line_fields_and_years():
    text='''## 竞赛\n> ACM：无 ACM 竞赛获奖记录；GitHub：https://example.test/u；技能：Python、SQL\n## 工作\n> 近 6 年工作经历：工程师\n## 个人\n> 应聘方向：数据工程师\n本科：某大学\n主要职责：负责数据管道\n'''
    row=parse_markdown(text,1,'简历_001.md','abc','P')
    assert row['acm_status']=='no_award_record' and row['acm_award']==0
    assert row['github_url']=='https://example.test/u' and 'Python' in row['skills_raw']
    assert row['work_years']==6 and row['target_role']=='数据工程师'

def test_macos_metadata_excluded_and_unparseable_acm_reported(tmp_path):
    source=tmp_path/'bad.zip'
    with ZipFile(source,'w',ZIP_DEFLATED) as z:
        z.writestr('__MACOSX/._简历_001.md',b'junk')
        z.writestr('../简历_002.md',b'junk')
        z.writestr('简历_001.md','应聘方向：测试开发工程师\nACM：经历未知\n'.encode())
    frame,errors=extract_zip(source,'P')
    assert len(frame)==1 and frame.iloc[0].id==1
    assert any('ACM status' in e['reason'] for e in errors)
    assert any('unsafe zip path' in e['reason'] for e in errors)
