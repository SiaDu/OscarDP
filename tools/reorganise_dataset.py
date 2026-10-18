"""Frozen-manifest-pinned workspace migration; no subtitle or annotation edits.

Default mode is a read-only dry-run. Plans and journals live outside dataset roots.
Existing sources, shared archives and unrelated stages are never deleted.
"""
from __future__ import annotations

import argparse
import ctypes
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import fcntl
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import zipfile

IMDB = re.compile(r'tt\d{7,10}')
STAGE = re.compile(r'(Stage0[12])_(tt\d{7,10})\.json')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def safe(root, relative):
    root = Path(root).absolute()
    relative = PurePosixPath(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe relative path: {relative}')
    path = root.joinpath(*relative.parts)
    for p in [root, *root.parents, path, *path.parents]:
        if p.is_symlink():
            raise ValueError(f'Symlink path rejected: {p}')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes root: {path}')
    return path


def release_inputs(manifest_path, expected_sha256):
    manifest_path = Path(manifest_path).absolute()
    if sha(manifest_path) != expected_sha256:
        raise ValueError('Frozen Manifest SHA-256 mismatch')
    frozen = load(manifest_path)
    root = manifest_path.parent
    if frozen.get('dataset_status') != 'FROZEN' or frozen.get('activated_for_pipeline') is not False:
        raise ValueError('Manifest is not an inactive FROZEN release')
    movies = frozen['movies']
    ids = [m['movie_id'] for m in movies]
    if len(set(ids)) != len(ids) or len(ids) != frozen['movie_count'] or not all(IMDB.fullmatch(x) for x in ids):
        raise ValueError('Invalid or duplicate frozen movie selection')
    files = {}
    for movie in movies:
        if not movie['release_eligible'] or movie['activated_for_pipeline']:
            raise ValueError('Frozen movie is not eligible/inactive')
        mid = movie['movie_id']
        if set(movie['inputs']) != {'Stage01', 'Stage02', 'SRT'}:
            raise ValueError('Expected exactly Stage01, Stage02 and SRT')
        for kind, item in movie['inputs'].items():
            path = safe(root, item['path'])
            if not path.is_file() or sha(path) != item['sha256']:
                raise ValueError(f'Frozen input mismatch: {path}')
            if kind != 'SRT' and load(path).get('movie_id') != mid:
                raise ValueError(f'Movie identity mismatch: {path}')
            files[mid, kind] = {**item, 'source_path': str(path)}
    # Verify the complete frozen checksum inventory, not only the selected inputs.
    listed = set()
    for line in (root / 'checksums.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if name in listed or sha(safe(root, name)) != expected:
            raise ValueError(f'Frozen checksum mismatch: {name}')
        listed.add(name)
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'checksums.sha256'}
    if listed != actual:
        raise ValueError('Incomplete frozen checksum inventory')
    return frozen, files


def identify(path, movie_ids):
    match = STAGE.fullmatch(path.name)
    if match and match[2] in movie_ids:
        if load(path).get('movie_id') != match[2]:
            raise ValueError(f'History identity conflict: {path}')
        return match[2], match[1]
    if path.suffix.lower() == '.srt':
        for text in [path.stem, *(p.name for p in path.parents)]:
            match = re.search(r'(?<![A-Za-z0-9])(tt\d{7,10})(?!\d)', text)
            if match:
                return (match[1], 'SRT') if match[1] in movie_ids else (None, 'OTHER_MOVIE')
    return None, 'PRESERVED_UNRELATED'


def build_plan(manifest_path, expected_sha256, workspace, history_roots=(), movie_root=None):
    frozen, selected = release_inputs(manifest_path, expected_sha256)
    workspace = Path(workspace).absolute()
    frozen_root = Path(manifest_path).absolute().parent
    if workspace.resolve().is_relative_to(frozen_root) or frozen_root.is_relative_to(workspace.resolve()):
        raise ValueError('Workspace and frozen release must be disjoint')
    safe(workspace, '.')
    ids = {m['movie_id'] for m in frozen['movies']}
    inventory, sources, shared_archives = [], [], []
    seen = set()

    def record(path, identity, *, forced=None):
        path = Path(path).absolute()
        if path in seen:
            return
        seen.add(path)
        if path.is_symlink():
            raise ValueError(f'Inventory symlink requires review: {path}')
        mid, kind = forced or identify(path, ids)
        h = sha(path)
        managed = path.is_relative_to(workspace) and ('stage02_curated' in path.relative_to(workspace).parts
                    or '.dataset_history' in path.relative_to(workspace).parts)
        target_hash = selected.get((mid, kind), {}).get('sha256')
        item = dict(movie_id=mid, current_absolute_path=str(path), filename=path.name, sha256=h,
            bytes=path.stat().st_size, identity=identity, kind=kind, managed_history=managed,
            matches_frozen_selected=(h == target_hash if target_hash else None),
            will_migrate=bool(mid and kind in ['Stage01', 'Stage02', 'SRT'] and not managed),
            history_authorship='UNVERIFIED_HISTORY' if h != target_hash else 'FROZEN_SELECTED_BYTES',
            name_conflict=False, belongs_to_other_movie_or_stage=mid is None)
        inventory.append(item)
        if item['will_migrate']:
            sources.append(item)

    for root, identity in [(workspace, 'CURRENT_WORKSPACE'), *[(Path(p).absolute(), 'LEGACY_CURATED_LOCATION_NOT_AUTHENTICATION') for p in history_roots]]:
        if not root.is_dir():
            raise ValueError(f'Inventory root missing: {root}')
        if root.resolve().is_relative_to(frozen_root):
            raise ValueError('History-root scan must not treat frozen data as legacy workspace')
        for p in sorted(root.rglob('*')):
            if p.is_symlink():
                raise ValueError(f'Inventory symlink requires review: {p}')
            if not p.is_file():
                continue
            record(p, identity)
            if p.suffix.lower() == '.zip' and not p.is_relative_to(workspace):
                members = []
                with zipfile.ZipFile(p) as archive:
                    for member in archive.infolist():
                        if member.is_dir():
                            continue
                        rel = PurePosixPath(member.filename)
                        if rel.is_absolute() or '..' in rel.parts:
                            raise ValueError(f'Unsafe historical ZIP member: {member.filename}')
                        raw = archive.read(member)
                        match = STAGE.fullmatch(rel.name)
                        mid = match[2] if match else None
                        kind = match[1] if match else None
                        if mid not in ids or json.loads(raw).get('movie_id') != mid:
                            raise ValueError(f'UNVERIFIED_HISTORY assignment requires review: {p}:{rel}')
                        h = hashlib.sha256(raw).hexdigest()
                        # Unknown unique members must be reviewed, not dropped by archive consolidation.
                        if h != selected[mid, kind]['sha256'] and not any(x['sha256'] == h and x['movie_id'] == mid for x in sources):
                            raise ValueError(f'Unique historical ZIP member requires review: {p}:{rel}')
                        members.append({'name': member.filename, 'movie_id': mid, 'kind': kind, 'sha256': h, 'bytes': len(raw)})
                shared_archives.append({'source_path': str(p), 'sha256': sha(p), 'members': members})
    # Read subtitles from the source-video root, but never write or copy video bytes.
    readonly_stats = []
    if movie_root:
        movie_root = Path(movie_root).absolute()
        if not movie_root.is_dir() or movie_root.resolve().is_relative_to(workspace):
            raise ValueError('Invalid read-only movie source root')
        for p in sorted(movie_root.rglob('*')):
            if p.is_file():
                readonly_stats.append({'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns})
                mid, kind = identify(p, ids)
                if mid and kind == 'SRT':
                    record(p, 'READ_ONLY_MOVIE_SUBTITLE_SOURCE')
    # Source provenance explicitly identifies the film and each bundled raw subtitle hash.
    evidence_by_source = {(x['source_path'], x['sha256']): x for x in frozen.get('evidence_files', [])}
    for evidence in frozen.get('evidence_files', []):
        if not evidence['relative_path'].endswith('.json'):
            continue
        data = load(safe(frozen_root, evidence['relative_path']))
        if not isinstance(data, dict) or data.get('movie_id') not in ids:
            continue
        for source in data.get('source_files', {}).values():
            bundled = evidence_by_source.get((source.get('path'), source.get('sha256')))
            if bundled:
                record(safe(frozen_root, bundled['relative_path']), 'FROZEN_ARCHIVED_SOURCE_EVIDENCE', forced=(data['movie_id'], 'SRT'))

    grouped = defaultdict(list)
    for item in sources:
        grouped[item['movie_id'], item['kind'], item['sha256']].append(item)
    duplicate_hashes = defaultdict(list)
    for item in inventory:
        duplicate_hashes[item['sha256']].append(item['current_absolute_path'])
    for item in inventory:
        item['identical_content_paths'] = [p for p in duplicate_hashes[item['sha256']] if p != item['current_absolute_path']]
    operations, mappings = [], []

    def add(destination, expected, *, source=None, payload=None, replace=False, role='HISTORY'):
        path = safe(workspace, destination)
        if path.exists() and (not path.is_file() or (sha(path) != expected and not replace)):
            raise ValueError(f'Unsafe destination conflict: {path}')
        op = {'destination': str(path), 'relative_destination': destination, 'sha256': expected,
              'role': role, 'allow_replace_after_preservation': replace,
              'observed_before_sha256': sha(path) if path.is_file() else None}
        if source:
            op['source_path'] = str(source)
        else:
            op['payload_utf8'] = payload.decode('utf-8')
        operations.append(op)

    for archive in shared_archives:
        relative = f'.dataset_history/legacy_archives/{archive["sha256"]}.zip'
        add(relative, archive['sha256'], source=archive['source_path'], role='SHARED_LEGACY_ARCHIVE')
        mappings.append({'source_path': archive['source_path'], 'source_sha256': archive['sha256'],
                         'destination': str(safe(workspace, relative)), 'destination_sha256': archive['sha256'],
                         'identity': 'VERIFIED_ARCHIVE_BYTES_NO_AUTHORSHIP_CLAIM', 'members': archive['members']})
    movie_mappings = defaultdict(list)
    subtitle_groups = defaultdict(list)
    for (mid, kind, h), versions in sorted(grouped.items()):
        target = f'{mid}/{mid}.srt' if kind == 'SRT' else f'{mid}/{kind}_{mid}.json'
        if h != selected[mid, kind]['sha256']:
            if kind == 'SRT':
                subtitle_groups[mid].append((h, versions))
                continue
            target = f'{mid}/stage02_curated/history/{kind.lower()}/{kind}_{mid}_sha256_{h}.json'
            add(target, h, source=versions[0]['current_absolute_path'])
        for version in versions:
            mapping = {'movie_id': mid, 'kind': kind, 'source_path': version['current_absolute_path'],
                'source_sha256': h, 'destination': str(safe(workspace, target)), 'destination_sha256': h,
                'identity': version['identity'], 'history_authorship': version['history_authorship']}
            mappings.append(mapping)
            movie_mappings[mid].append(mapping)
    for mid, groups in subtitle_groups.items():
        payload = io.BytesIO()
        zip_records = []
        with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for h, versions in sorted(groups):
                source = Path(versions[0]['current_absolute_path'])
                raw = source.read_bytes()
                if hashlib.sha256(raw).hexdigest() != h:
                    raise ValueError(f'Changed subtitle source: {source}')
                name = f'{mid}_sha256_{h}.srt'
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, raw)
                zip_records.append({'member': name, 'sha256': h, 'source_paths': [x['current_absolute_path'] for x in versions],
                    'identity': [x['identity'] for x in versions], 'history_authorship': 'UNVERIFIED_HISTORY_NO_NEW_SOURCE_APPROVAL'})
        raw = payload.getvalue()
        hzip = hashlib.sha256(raw).hexdigest()
        relative = f'{mid}/stage02_curated/subtitle_sources/original_subtitle_sources_{hzip}.zip'
        # Binary payload is stored in report-owned plan assets, not loose scan-visible SRTs.
        operations.append({'destination': str(safe(workspace, relative)), 'relative_destination': relative,
            'sha256': hzip, 'payload_hex': raw.hex(), 'role': 'SUBTITLE_ARCHIVE',
            'allow_replace_after_preservation': False,
            'observed_before_sha256': sha(safe(workspace, relative)) if safe(workspace, relative).is_file() else None,
            'members': zip_records})
        for item in zip_records:
            for source in item['source_paths']:
                mapping = {'movie_id': mid, 'kind': 'SRT', 'source_path': source, 'source_sha256': item['sha256'],
                    'destination': str(safe(workspace, relative)), 'zip_member': item['member'],
                    'destination_sha256': hzip, 'member_sha256': item['sha256'], 'identity': item['history_authorship']}
                mappings.append(mapping)
                movie_mappings[mid].append(mapping)

    # Revision records are useful only for films with legacy curation or non-selected history.
    for mid, entries in movie_mappings.items():
        if not any(e['source_sha256'] != selected[mid, e['kind']]['sha256'] or 'CURATED' in e['identity'] for e in entries):
            continue
        body = encoded({'movie_id': mid, 'release_id': frozen['release_id'], 'frozen_manifest_sha256': expected_sha256,
            'scope': 'Byte-preserving migration only; does not authenticate LLM authorship or create a new source approval.',
            'source_approval_references': next(m for m in frozen['movies'] if m['movie_id'] == mid).get('source_preflight'),
            'history_mappings': entries})
        add(f'{mid}/stage02_curated/revision_history.json', hashlib.sha256(body).hexdigest(), payload=body, role='REVISION_RECORD')
    for (mid, kind), selected_source in sorted(selected.items()):
        relative = f'{mid}/{mid}.srt' if kind == 'SRT' else f'{mid}/{kind}_{mid}.json'
        add(relative, selected_source['sha256'], source=selected_source['source_path'], replace=True, role='CURRENT')
    mapping_body = encoded({mid: f'{mid}/{mid}.srt' for mid in sorted(ids)})
    old_map = safe(workspace, 'srt_map.json')
    if old_map.is_file() and sha(old_map) != hashlib.sha256(mapping_body).hexdigest():
        old_hash = sha(old_map)
        relative = f'.dataset_history/workspace_metadata/srt_map_sha256_{old_hash}.json'
        add(relative, old_hash, source=str(old_map), role='SHARED_LEGACY_ARCHIVE')
        mappings.append({'source_path': str(old_map), 'source_sha256': old_hash,
                         'destination': str(safe(workspace, relative)), 'destination_sha256': old_hash,
                         'identity': 'PREVIOUS_WORKSPACE_MAP'})
    add('srt_map.json', hashlib.sha256(mapping_body).hexdigest(), payload=mapping_body, replace=True, role='SRT_MAP')
    current_destinations = {op['destination'] for op in operations if op['role'] == 'CURRENT'}
    for item in inventory:
        if item['current_absolute_path'] in current_destinations and item['matches_frozen_selected'] is False:
            item['name_conflict'] = True
    return {'plan_version': '1.0', 'release_id': frozen['release_id'], 'frozen_manifest_path': str(Path(manifest_path).absolute()),
        'frozen_manifest_sha256': expected_sha256, 'workspace': str(workspace),
        'movie_ids': sorted(ids), 'inventory': inventory, 'operations': operations, 'mappings': mappings,
        'readonly_movie_source_stats': readonly_stats, 'history_authorship_policy': 'UNVERIFIED_HISTORY unless actual evidence confirms identity; never inferred from filenames.',
        'allow_legacy_deletion': False, 'activated_for_pipeline': False}


def atomic_bytes(path, raw, *, replace=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.reorganise-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if sha(temporary) != hashlib.sha256(raw).hexdigest():
            raise ValueError('Temporary copy hash mismatch')
        if replace:
            os.replace(temporary, path)
        else:
            libc = ctypes.CDLL(None, use_errno=True)
            rename = libc.renameat2
            rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            if rename(-100, os.fsencode(temporary), -100, os.fsencode(path), 1):
                code = ctypes.get_errno()
                raise OSError(code, os.strerror(code), str(path))
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialized(op):
    if 'source_path' in op:
        raw = Path(op['source_path']).read_bytes()
    elif 'payload_hex' in op:
        raw = bytes.fromhex(op['payload_hex'])
    else:
        raw = op['payload_utf8'].encode()
    if hashlib.sha256(raw).hexdigest() != op['sha256']:
        raise ValueError(f'Operation source changed: {op["destination"]}')
    return raw


def validate_plan(plan):
    _, selected = release_inputs(plan['frozen_manifest_path'], plan['frozen_manifest_sha256'])
    workspace = Path(plan['workspace'])
    frozen_root = Path(plan['frozen_manifest_path']).parent
    if workspace.resolve().is_relative_to(frozen_root) or frozen_root.resolve().is_relative_to(workspace.resolve()):
        raise ValueError('Workspace and frozen release must be disjoint')
    destinations = set()
    for op in plan['operations']:
        path = safe(workspace, op['relative_destination'])
        if str(path) != op['destination'] or path in destinations:
            raise ValueError('Invalid or duplicate planned destination')
        destinations.add(path)
        relative = Path(op['relative_destination'])
        if op['role'] == 'CURRENT':
            mid = relative.parts[0]
            kind = 'SRT' if relative.name == f'{mid}.srt' else relative.name.split('_')[0]
            expected_name = f'{mid}.srt' if kind == 'SRT' else f'{kind}_{mid}.json'
            if len(relative.parts) != 2 or relative.name != expected_name or (mid, kind) not in selected or op['sha256'] != selected[mid, kind]['sha256']:
                raise ValueError('Current target does not bind frozen selection')
        elif op['role'] == 'SRT_MAP':
            if str(relative) != 'srt_map.json':
                raise ValueError('Invalid workspace map target')
        elif not (relative.parts[0] == '.dataset_history' or
                  (len(relative.parts) >= 3 and relative.parts[0] in plan['movie_ids'] and relative.parts[1] == 'stage02_curated')):
            raise ValueError('History target outside approved history locations')


def verify_plan(plan):
    validate_plan(plan)
    for op in plan['operations']:
        path = safe(plan['workspace'], op['relative_destination'])
        if not path.is_file() or sha(path) != op['sha256']:
            raise ValueError(f'Migration target mismatch: {path}')
        if op['role'] == 'SUBTITLE_ARCHIVE':
            with zipfile.ZipFile(path) as archive:
                if set(archive.namelist()) != {item['member'] for item in op['members']}:
                    raise ValueError('Subtitle ZIP membership mismatch')
                for item in op['members']:
                    if hashlib.sha256(archive.read(item['member'])).hexdigest() != item['sha256']:
                        raise ValueError('Subtitle ZIP content mismatch')
    writable = {op['destination'] for op in plan['operations']}
    preserved = 0
    for item in plan['inventory']:
        path = item['current_absolute_path']
        if path not in writable:
            if sha(path) != item['sha256']:
                raise ValueError(f'Unrelated/historical input changed: {path}')
            preserved += 1
    for item in plan['readonly_movie_source_stats']:
        stat = Path(item['path']).stat()
        if stat.st_size != item['bytes'] or stat.st_mtime_ns != item['mtime_ns']:
            raise ValueError(f'Read-only movie-root file changed: {item["path"]}')
    return {'status': 'PASS', 'current_file_count': sum(op['role'] == 'CURRENT' for op in plan['operations']),
        'all_targets_verified': len(plan['operations']), 'preserved_inventory_files': preserved,
        'readonly_movie_root_metadata_unchanged': True, 'frozen_release_unchanged': True}


def _apply_locked(plan, report_root):
    validate_plan(plan)
    workspace = Path(plan['workspace'])
    safe(workspace, '.')
    workspace.mkdir(parents=True, exist_ok=True)
    # All operations are fully materialized and checked before the first replacement.
    for op in plan['operations']:
        path = safe(workspace, op['relative_destination'])
        if path.exists() and sha(path) == op['sha256']:
            continue
        if path.exists() and (not op['allow_replace_after_preservation'] or sha(path) != op['observed_before_sha256']):
            raise ValueError(f'Unplanned destination change: {path}')
        materialized(op)
    need = sum(len(materialized(op)) for op in plan['operations'] if not Path(op['destination']).is_file() or sha(op['destination']) != op['sha256'])
    if shutil.disk_usage(workspace).free < need + 1024 * 1024:
        raise ValueError('Insufficient staging/backup space')
    # Probe write permission without relying on os.access, including external volume masks.
    probe_fd, probe = tempfile.mkstemp(prefix='.reorganise-write-check-', dir=workspace)
    os.close(probe_fd)
    os.unlink(probe)
    report_root = Path(report_root)
    if report_root.resolve().is_relative_to(workspace.resolve()) or report_root.resolve().is_relative_to(Path(plan['frozen_manifest_path']).parent):
        raise ValueError('Report root must be outside dataset roots')
    report_root.mkdir(parents=True, exist_ok=True)
    statuses = []
    for number, op in enumerate(plan['operations']):
        target = safe(workspace, op['relative_destination'])
        if target.exists() and sha(target) == op['sha256']:
            status = 'SKIPPED_IDENTICAL'
        else:
            raw = materialized(op)
            if target.exists():
                before = sha(target)
                if not op['allow_replace_after_preservation'] or before != op['observed_before_sha256']:
                    raise ValueError(f'Concurrent target change: {target}')
                backups = [item for item in plan['mappings'] if item['source_path'] == str(target) and item['source_sha256'] == before
                           and (item.get('zip_member') or item['destination'] != str(target))]
                if not backups:
                    raise ValueError(f'No verified backup for replacement: {target}')
                for backup in backups:
                    if backup.get('zip_member'):
                        with zipfile.ZipFile(backup['destination']) as archive:
                            good = hashlib.sha256(archive.read(backup['zip_member'])).hexdigest() == before
                    else:
                        good = sha(backup['destination']) == before
                    if not good:
                        raise ValueError('Historical backup hash mismatch')
            atomic_bytes(target, raw, replace=target.exists())
            if sha(target) != op['sha256']:
                raise ValueError('Post-copy hash mismatch')
            status = 'COPIED_OR_REPLACED_AFTER_VERIFIED_BACKUP'
        statuses.append({'operation': number, 'destination': str(target), 'sha256': op['sha256'], 'status': status})
        with (report_root / 'migration_journal.jsonl').open('a') as journal:
            journal.write(json.dumps(statuses[-1]) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
    result = verify_plan(plan)
    atomic_bytes(report_root / 'migration_manifest.json', encoded({'release_id': plan['release_id'], 'frozen_manifest_sha256': plan['frozen_manifest_sha256'],
        'mappings': plan['mappings'], 'operations': statuses, 'verification': result,
        'old_curated_directories_deleted': False, 'activated_for_pipeline': False}))
    return result


def apply_plan(plan, report_root):
    workspace = Path(plan['workspace'])
    safe(workspace, '.')
    # A shared directory lock also serializes invocations using different report roots.
    if not workspace.is_dir():
        raise ValueError('Canonical workspace must already exist')
    fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _apply_locked(plan, report_root)
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-manifest', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--history-root', type=Path, action='append', default=[])
    parser.add_argument('--movie-root', type=Path)
    parser.add_argument('--report-root', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--apply', action='store_true')
    modes.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    try:
        workspace = args.workspace.absolute()
        report_root = args.report_root.absolute()
        roots = [workspace, args.frozen_manifest.absolute().parent, *(p.absolute() for p in args.history_root)]
        if args.movie_root:
            roots.append(args.movie_root.absolute())
        if any(report_root.resolve().is_relative_to(p.resolve()) for p in roots):
            raise ValueError('Reports must be outside all dataset/source roots')
        release_inputs(args.frozen_manifest, args.expected_manifest_sha256)
        plan_path = report_root / 'migration_plan.json'
        if plan_path.exists():
            plan = load(plan_path)
            if plan['workspace'] != str(workspace) or plan['frozen_manifest_sha256'] != args.expected_manifest_sha256 or plan['frozen_manifest_path'] != str(args.frozen_manifest.absolute()):
                raise ValueError('Existing plan does not bind these inputs')
        else:
            if args.apply or args.verify:
                raise ValueError('Run --dry-run first to save an auditable plan')
            plan = build_plan(args.frozen_manifest, args.expected_manifest_sha256, workspace, args.history_root, args.movie_root)
            report_root.mkdir(parents=True, exist_ok=True)
            for name, payload in [('migration_plan.json', plan), ('pre_migration_inventory.json', {'release_id': plan['release_id'], 'files': plan['inventory']})]:
                with (report_root / name).open('xb') as stream:
                    stream.write(encoded(payload))
            text = '# Frozen-selected Canonical migration plan\n\n'
            text += f'Release: {plan["release_id"]}; movies: {len(plan["movie_ids"])}; inventory files: {len(plan["inventory"])}.\n\n'
            text += 'SAFE byte-copy plan: historical bytes and ZIP members are preserved before current files can change. No movie/source directories are renamed or deleted. No authorship or approval is inferred from locations.\n\n'
            text += '| Role | Destination | SHA-256 |\n|---|---|---|\n'
            text += '\n'.join(f'| {op["role"]} | {op["destination"]} | {op["sha256"]} |' for op in plan['operations']) + '\n'
            with (report_root / 'migration_plan.md').open('x') as stream:
                stream.write(text)
        if args.apply:
            result = apply_plan(plan, report_root)
        elif args.verify:
            result = verify_plan(plan)
        else:
            result = {'status': 'DRY_RUN', 'movie_count': len(plan['movie_ids']), 'operation_count': len(plan['operations']), 'no_dataset_writes': True}
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f'Reorganisation stopped: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
