import csv
import json
from pathlib import Path

import pytest
from openpyxl import Workbook

from oscardp.shots.discovery import discover_movies
from tools import dataset_preflight as dp


def fixture(tmp_path: Path, *, second_video=False, second_srt=False):
    movies = tmp_path / 'movies'
    scripts = tmp_path / 'scripts'
    folder = movies / 'tt1234567_Title'
    folder.mkdir(parents=True)
    scripts.mkdir()
    video = folder / 'original.mkv'
    srt = folder / 'aligned.srt'
    video.write_bytes(b'video')
    srt.write_text('unaltered subtitle')
    (scripts / 'Title (2020).pdf').write_bytes(b'pdf')
    if second_video:
        (folder / 'other.mp4').write_bytes(b'other')
    if second_srt:
        (folder / 'other.srt').write_text('other subtitle')
    book = Workbook()
    sheet = book.active
    sheet.append(['Year', 'FilmId', 'Film', 'video', 'script'])
    sheet.append([2020, 'tt1234567', 'Title', 'Y', 'Y'])
    workbook = tmp_path / 'inventory.xlsx'
    book.save(workbook)
    return workbook, movies, scripts, video, srt


def test_workbook_parsing_and_duplicate_ids(tmp_path: Path):
    workbook, *_ = fixture(tmp_path)
    parsed = dp.read_workbook(workbook)
    assert parsed['tt1234567'].title == 'Title'
    assert parsed['tt1234567'].year == 2020
    book = Workbook()
    sheet = book.active
    sheet.append(['FilmId', 'Film', 'Year'])
    sheet.append(['tt1234567', 'Title', 2020])
    sheet.append(['tt1234567', 'Other', 2021])
    book.save(workbook)
    with pytest.raises(ValueError, match='Duplicate FilmId'):
        dp.read_workbook(workbook)


def test_inventory_parent_id_srt_and_title_year(tmp_path: Path):
    workbook, movies, scripts, video, srt = fixture(tmp_path)
    records, rows, extras, summary, videos, srts = dp.inventory(workbook, movies, scripts)
    assert summary['complete_sets'] == 1
    assert rows[0]['video_match'] == 'parent_id'
    assert rows[0]['srt_status'] == 'FOUND'
    assert rows[0]['screenplay_match'] == 'title_year'
    assert not extras
    assert discover_movies(movies)[0].movie_key == 'tt1234567'
    plan = dp.plan_rename(records, rows, videos, srts)
    assert len(plan) == 2
    assert {item['classification'] for item in plan} == {'SAFE'}
    assert {item['proposed_filename'] for item in plan} == {'tt1234567.mkv', 'tt1234567.srt'}
    assert video.exists() and srt.exists()  # Planning is read-only.
    approved = dp.approve_safe_plan(plan, set(records))
    assert len(approved) == 2
    assert all(item['approval_status'] == 'APPROVED' for item in approved)


def test_missing_multiple_and_unmatched(tmp_path: Path):
    workbook, movies, scripts, video, srt = fixture(tmp_path, second_video=True, second_srt=True)
    (movies / 'unknown.mp4').write_bytes(b'extra')
    (scripts / 'tt9999999_Extra.pdf').write_bytes(b'extra')
    _, rows, extras, summary, _, _ = dp.inventory(workbook, movies, scripts)
    assert rows[0]['video_status'] == 'MULTIPLE'
    assert rows[0]['srt_status'] == 'MULTIPLE'
    assert summary['unexpected_files'] >= 2
    assert any('unknown.mp4' in item['path'] for item in extras)
    srt.unlink()
    (srt.parent / 'other.srt').unlink()
    _, rows, _, _, _, _ = dp.inventory(workbook, movies, scripts)
    assert rows[0]['srt_status'] == 'MISSING'


def test_orphan_srt_is_ambiguous_not_missing(tmp_path: Path):
    workbook, movies, scripts, video, srt = fixture(tmp_path)
    video.unlink()
    _, rows, extras, summary, _, _ = dp.inventory(workbook, movies, scripts)
    assert rows[0]['video_status'] == 'MISSING'
    assert rows[0]['srt_status'] == 'AMBIGUOUS'
    assert rows[0]['srt_candidates'] == 1
    assert any(item['path'] == str(srt) for item in extras)
    assert summary['srts_ambiguous'] == 1


def test_title_year_video_is_not_rename_eligible(tmp_path: Path):
    workbook, movies, scripts, video, srt = fixture(tmp_path)
    video.unlink()
    srt.unlink()
    other = movies / 'Title.2020.1080p.mkv'
    other.write_bytes(b'video')
    (movies / 'aligned.srt').write_text('subtitle')
    records, rows, _, _, videos, srts = dp.inventory(workbook, movies, scripts)
    assert rows[0]['video_match'] == 'title_year'
    assert rows[0]['manual_review_required']
    assert all(item['classification'] != 'SAFE' for item in dp.plan_rename(records, rows, videos, srts))


def test_collision_and_already_standardised(tmp_path: Path):
    workbook, movies, scripts, video, srt = fixture(tmp_path)
    (video.parent / 'TT1234567.MKV').write_bytes(b'collision')
    records, rows, _, _, videos, srts = dp.inventory(workbook, movies, scripts)
    plan = dp.plan_rename(records, rows, videos, srts)
    assert any(item['classification'] == 'CONFLICT' for item in plan)
    with pytest.raises(ValueError, match='Unresolved'):
        dp.approve_safe_plan(plan, set(records))
    (video.parent / 'TT1234567.MKV').unlink()
    video.rename(video.with_name('tt1234567.mkv'))
    records, rows, _, _, videos, srts = dp.inventory(workbook, movies, scripts)
    plan = dp.plan_rename(records, rows, videos, srts)
    assert any(item['classification'] == 'ALREADY_STANDARDISED' for item in plan)


def test_apply_requires_approval_and_rolls_back(tmp_path: Path):
    workbook, movies, scripts, video, srt = fixture(tmp_path)
    records, rows, _, _, videos, srts = dp.inventory(workbook, movies, scripts)
    plan = dp.plan_rename(records, rows, videos, srts)
    approved = tmp_path / 'approved.csv'
    dp._write_csv(approved, dp.PLAN_FIELDS, plan)
    journal = tmp_path / 'journal.jsonl'
    with pytest.raises(ValueError, match='confirm'):
        dp.apply_rename(approved, journal, False)
    with pytest.raises(ValueError, match='No APPROVED'):
        dp.apply_rename(approved, journal, True)
    assert video.exists() and srt.exists() and not journal.exists()
    for item in plan:
        item['approval_status'] = 'APPROVED'
    dp._write_csv(approved, dp.PLAN_FIELDS, plan)
    assert dp.apply_rename(approved, journal, True)['applied_or_recovered'] == 2
    assert not video.exists() and not srt.exists()
    assert (video.parent / 'tt1234567.mkv').read_bytes() == b'video'
    assert (srt.parent / 'tt1234567.srt').read_text() == 'unaltered subtitle'
    assert discover_movies(movies)[0].movie_key == 'tt1234567'
    assert dp.apply_rename(approved, journal, True)['applied_or_recovered'] == 0
    assert dp.verify_journal(journal)['checked'] == 2
    with pytest.raises(ValueError, match='confirm'):
        dp.verify_journal(journal, rollback=True)
    assert dp.verify_journal(journal, rollback=True, confirm=True)['rolled_back'] == 2
    assert video.exists() and srt.exists()


def test_interrupted_rename_recovery(tmp_path: Path):
    workbook, movies, scripts, video, srt = fixture(tmp_path)
    records, rows, _, _, videos, srts = dp.inventory(workbook, movies, scripts)
    plan = dp.plan_rename(records, rows, videos, srts)
    plan[0]['approval_status'] = 'APPROVED'
    approved = tmp_path / 'approved.csv'
    dp._write_csv(approved, dp.PLAN_FIELDS, plan)
    journal = tmp_path / 'journal.jsonl'
    item = plan[0]
    operation_id = dp._operation_id(item)
    dp._journal(journal, {'operation_id': operation_id, 'state': 'PREPARED', 'row': item})
    dp._rename_noreplace(Path(item['original_path']), Path(item['proposed_path']))
    assert dp.apply_rename(approved, journal, True)['applied_or_recovered'] == 1
    assert dp.verify_journal(journal)['checked'] == 1
    assert dp.verify_journal(journal, rollback=True, confirm=True)['rolled_back'] == 1


def test_rollback_rejects_conflicting_original_name(tmp_path: Path):
    workbook, movies, scripts, video, _ = fixture(tmp_path)
    records, rows, _, _, videos, srts = dp.inventory(workbook, movies, scripts)
    plan = dp.plan_rename(records, rows, videos, srts)
    plan[0]['approval_status'] = 'APPROVED'
    approved = tmp_path / 'approved.csv'
    dp._write_csv(approved, dp.PLAN_FIELDS, plan)
    journal = tmp_path / 'journal.jsonl'
    dp.apply_rename(approved, journal, True)
    video.write_bytes(b'new file')
    with pytest.raises(RuntimeError, match='Rollback conflict'):
        dp.verify_journal(journal, rollback=True, confirm=True)
    assert (video.parent / 'tt1234567.mkv').exists()


def test_stage01_title_mapping_shape(tmp_path: Path):
    workbook, movies, scripts, *_ = fixture(tmp_path)
    assert dp.main(['plan-rename', '--workbook', str(workbook), '--movies-root', str(movies),
                    '--scripts-root', str(scripts), '--reports-dir', str(tmp_path / 'reports'),
                    '--titles-output', str(tmp_path / 'titles.json')]) == 0
    assert json.loads((tmp_path / 'titles.json').read_text()) == {'tt1234567': 'Title'}
    assert (tmp_path / 'reports' / 'rename_plan.csv').is_file()
    assert not (tmp_path / 'reports' / 'rename_journal.jsonl').exists()


def test_rmvb_is_inventory_and_discovery_video(tmp_path: Path):
    workbook, movies, scripts, video, _ = fixture(tmp_path)
    video.rename(video.with_suffix('.rmvb'))
    _, rows, _, summary, _, _ = dp.inventory(workbook, movies, scripts)
    assert rows[0]['video_status'] == 'FOUND'
    assert summary['complete_sets'] == 1
    assert discover_movies(movies)[0].source_path.suffix == '.rmvb'
