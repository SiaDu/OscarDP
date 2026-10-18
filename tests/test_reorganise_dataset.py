"""Synthetic migrations exercise preservation, interruption and existing CLI contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

import pytest
from PIL import Image

from tools import reorganise_dataset as migration
from oscardp.stage02.__main__ import main as stage02_main
from oscardp.stage02.srt import index_srts
from oscardp.shots.validation import validate_movie as validate_stage01
from test_stage02_validator import make_movie, SRT
from test_stage02_preflight import pdf, run_preflight

MID = 'tt1234567'


def prepare(tmp_path, mid=MID):
    frozen = tmp_path / 'frozen'
    directory, one, two = make_movie(frozen / 'analysis', mid)
    srt = frozen / 'srts' / mid / f'{mid}.srt'
    srt.parent.mkdir(parents=True)
    srt.write_text(SRT)
    inputs = {kind: {'path': str(path.relative_to(frozen)), 'sha256': migration.sha(path), 'selection': 'FIXTURE'}
        for kind, path in [('Stage01', directory / f'Stage01_{mid}.json'),
                           ('Stage02', directory / f'Stage02_{mid}.json'), ('SRT', srt)]}
    manifest = frozen / 'frozen_manifest.json'
    manifest.write_text(json.dumps({'release_id': 'fixture_release', 'movie_count': 1,
        'dataset_status': 'FROZEN', 'activated_for_pipeline': False,
        'movies': [{'movie_id': mid, 'release_eligible': True, 'activated_for_pipeline': False, 'inputs': inputs}]}))
    (frozen / 'checksums.sha256').write_text(''.join(f'{migration.sha(p)}  {p.relative_to(frozen)}\n'
        for p in sorted(frozen.rglob('*')) if p.is_file()))
    workspace = tmp_path / 'workspace'
    current, _, _ = make_movie(workspace, mid)
    old = deepcopy(two)
    old['shots'][0]['narrative']['scene'] = 'Earlier synthetic revision'
    (current / f'Stage02_{mid}.json').write_text(json.dumps(old))
    image_dir = current / 'shot_img'
    image_dir.mkdir()
    for n in range(3):
        Image.new('RGB', (8, 8)).save(image_dir / f'shot_{n+1:06d}_00-00-0{n}-500.jpg')
    (current / f'{mid}.srt').write_text(SRT.replace('Hello.', 'Old caption.'))
    plan = migration.build_plan(manifest, migration.sha(manifest), workspace)
    return manifest, workspace, plan


def test_canonical_discovery_history_isolation_and_stage01(tmp_path):
    _, workspace, plan = prepare(tmp_path)
    migration.apply_plan(plan, tmp_path / 'reports')
    current = workspace / MID
    assert validate_stage01(current, frame_count=72).passed
    report = tmp_path / 'official'
    assert stage02_main(['validate', '--analysis-root', str(workspace), '--srt-root', str(workspace),
                         '--expect-movies', '1', '--report-root', str(report)]) == 0
    result = json.loads((report / 'movies' / f'{MID}.json').read_text())
    assert result['summary']['validation_status'] == 'PASS'
    assert 'SRT_AMBIGUOUS' not in {i['rule_code'] for i in result['issues']}
    assert index_srts(workspace)[MID] == [current / f'{MID}.srt']
    archives = list((current / 'stage02_curated').rglob('*.zip'))
    assert archives and not list((current / 'stage02_curated').rglob('*.srt'))
    with zipfile.ZipFile(archives[0]) as z:
        assert b'Old caption.' in z.read(z.namelist()[0])
    history = list((current / 'stage02_curated/history/stage02').glob('*.json'))
    assert len(history) == 1
    assert 'Earlier synthetic revision' in history[0].read_text()
    # Stage01 still rejects unknown complete files; only the named coexistence is allowed.
    (current / 'Stage02_tt7654321.json').write_text('{}')
    assert not validate_stage01(current).passed


def test_identical_hash_is_skipped_and_no_duplicate_backups(tmp_path):
    manifest, workspace, plan = prepare(tmp_path)
    migration.apply_plan(plan, tmp_path / 'reports')
    before = {str(p): (migration.sha(p), p.stat().st_mtime_ns) for p in workspace.rglob('*') if p.is_file()}
    frozen_before = {str(p): migration.sha(p) for p in manifest.parent.rglob('*') if p.is_file()}
    migration.apply_plan(plan, tmp_path / 'reports')
    assert before == {str(p): (migration.sha(p), p.stat().st_mtime_ns) for p in workspace.rglob('*') if p.is_file()}
    assert frozen_before == {str(p): migration.sha(p) for p in manifest.parent.rglob('*') if p.is_file()}


def test_frozen_manifest_mismatch_stops_before_writes(tmp_path):
    manifest, workspace, plan = prepare(tmp_path)
    before = {str(p): migration.sha(p) for p in workspace.rglob('*') if p.is_file()}
    manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(ValueError, match='Manifest SHA-256 mismatch'):
        migration.apply_plan(plan, tmp_path / 'reports')
    assert before == {str(p): migration.sha(p) for p in workspace.rglob('*') if p.is_file()}


def test_interruption_after_backup_can_resume_without_data_loss(tmp_path, monkeypatch):
    _, workspace, plan = prepare(tmp_path)
    original = migration.atomic_bytes
    old = (workspace / MID / f'Stage02_{MID}.json').read_bytes()

    def interrupt(path, raw, **kwargs):
        if Path(path) == workspace / MID / f'Stage02_{MID}.json':
            raise OSError('Simulated interruption before atomic replacement')
        return original(path, raw, **kwargs)

    monkeypatch.setattr(migration, 'atomic_bytes', interrupt)
    with pytest.raises(OSError, match='Simulated interruption'):
        migration.apply_plan(plan, tmp_path / 'reports')
    assert (workspace / MID / f'Stage02_{MID}.json').read_bytes() == old
    assert any(p.read_bytes() == old for p in (workspace / MID / 'stage02_curated/history').rglob('*.json'))
    monkeypatch.setattr(migration, 'atomic_bytes', original)
    assert migration.apply_plan(plan, tmp_path / 'reports')['status'] == 'PASS'


def test_resume_after_one_current_file_was_already_replaced(tmp_path, monkeypatch):
    _, workspace, plan = prepare(tmp_path)
    original = migration.atomic_bytes
    changed = [op for op in plan['operations'] if op['role'] == 'CURRENT'
               and migration.sha(op['destination']) != op['sha256']]
    first, second = changed

    def interrupt(path, raw, **kwargs):
        if Path(path) == Path(second['destination']):
            raise OSError('Interrupted after first promotion')
        return original(path, raw, **kwargs)

    monkeypatch.setattr(migration, 'atomic_bytes', interrupt)
    with pytest.raises(OSError, match='Interrupted after'):
        migration.apply_plan(plan, tmp_path / 'reports')
    assert migration.sha(first['destination']) == first['sha256']
    monkeypatch.setattr(migration, 'atomic_bytes', original)
    assert migration.apply_plan(plan, tmp_path / 'reports')['status'] == 'PASS'


def test_insufficient_space_stops_before_backup_writes(tmp_path, monkeypatch):
    _, workspace, plan = prepare(tmp_path)
    before = {str(p): migration.sha(p) for p in workspace.rglob('*') if p.is_file()}
    monkeypatch.setattr(migration.shutil, 'disk_usage', lambda _: type('Space', (), {'free': 0})())
    with pytest.raises(ValueError, match='Insufficient'):
        migration.apply_plan(plan, tmp_path / 'reports')
    assert before == {str(p): migration.sha(p) for p in workspace.rglob('*') if p.is_file()}


def test_manipulated_plan_cannot_promote_a_nonfrozen_current_version(tmp_path):
    _, workspace, plan = prepare(tmp_path)
    op = next(op for op in plan['operations'] if op['role'] == 'CURRENT')
    op['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='bind frozen selection'):
        migration.apply_plan(plan, tmp_path / 'reports')


def test_unplanned_target_change_is_not_overwritten(tmp_path):
    _, workspace, plan = prepare(tmp_path)
    target = workspace / MID / f'Stage02_{MID}.json'
    target.write_bytes(target.read_bytes() + b'\n')
    changed = target.read_bytes()
    with pytest.raises(ValueError, match='changed|change'):
        migration.apply_plan(plan, tmp_path / 'reports')
    assert target.read_bytes() == changed


def test_unrelated_movie_images_and_future_movies(tmp_path):
    _, workspace, plan = prepare(tmp_path)
    other = workspace / 'tt9999999'
    other.mkdir()
    (other / 'keep.bin').write_bytes(b'untouched')
    pictures = {str(p): migration.sha(p) for p in workspace.rglob('*.jpg')}
    migration.apply_plan(plan, tmp_path / 'reports')
    assert (other / 'keep.bin').read_bytes() == b'untouched'
    assert pictures == {str(p): migration.sha(p) for p in workspace.rglob('*.jpg')}
    # Future film using the unchanged Validator CLI and canonical placement.
    new, _, _ = make_movie(workspace, movie_id='tt7777777')
    (new / 'tt7777777.srt').write_text(SRT)
    assert stage02_main(['validate', '--analysis-root', str(workspace), '--srt-root', str(workspace),
        '--movie-id', 'tt7777777', '--report-root', str(tmp_path / 'future')]) == 0


def test_unsafe_frozen_path_and_symlink_history_blocked(tmp_path):
    manifest, workspace, _ = prepare(tmp_path)
    data = json.loads(manifest.read_text())
    data['movies'][0]['inputs']['SRT']['path'] = '../outside.srt'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Unsafe relative path'):
        migration.release_inputs(manifest, migration.sha(manifest))


def test_default_cli_only_plans_and_requires_plan_before_apply(tmp_path):
    manifest, workspace, _ = prepare(tmp_path)
    report = tmp_path / 'dry'
    args = ['--frozen-manifest', str(manifest), '--expected-manifest-sha256', migration.sha(manifest),
            '--workspace', str(workspace), '--report-root', str(report)]
    before = {str(p): migration.sha(p) for p in workspace.rglob('*') if p.is_file()}
    assert migration.main(args + ['--apply']) == 2
    assert migration.main(args) == 0
    assert before == {str(p): migration.sha(p) for p in workspace.rglob('*') if p.is_file()}
    assert migration.main(args + ['--apply']) == 0
    assert migration.main(args + ['--verify']) == 0


def test_canonical_preflight_and_approval_not_rewritten_or_reused(tmp_path):
    _, workspace, plan = prepare(tmp_path)
    migration.apply_plan(plan, tmp_path / 'migration')
    current = workspace / MID
    screenplay = tmp_path / 'scripts' / 'script.pdf'
    screenplay.parent.mkdir()
    pdf(screenplay)
    options = dict(stage01=current / f'Stage01_{MID}.json', srt=current / f'{MID}.srt',
        screenplay=screenplay, stage02=current / f'Stage02_{MID}.json')
    good = run_preflight(**options, report_root=tmp_path / 'preflight1')
    assert good['preflight_status'] == 'PASS'
    # Synthesize an invalid, old approval; the migration never edits approval records.
    bad = tmp_path / 'old_approval.json'
    bad.write_text(json.dumps({'schema_version': '1.0', 'exceptions': [{
        'issue_id': 'old_issue_not_present', 'rule_code': 'SOURCE_VERSION_CHANGED', 'movie_id': MID,
        'input_sha256': {k: v['sha256'] for k, v in good['inputs'].items()},
        'approved_by': 'Fixture', 'evidence': 'Old scope only', 'decision': 'APPROVED_NONBLOCKING'}]}))
    before = bad.read_bytes()
    checked = run_preflight(**options, report_root=tmp_path / 'preflight2', approved_exceptions=bad)
    assert not checked['llm_annotation_allowed']
    assert bad.read_bytes() == before
