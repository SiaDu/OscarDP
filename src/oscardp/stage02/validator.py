"""Deterministic validation; all dataset access is read-only."""
from __future__ import annotations

from collections import Counter
import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, validators

from .srt import parse_srt
from .text import compare_entry
from .issues import aggregate, domain

SCOPE = 'Structural and Source Consistency Validation'
LIMITATION = 'Does not validate screenplay semantics, character identity, emotion, intent or actual movie visuals.'
# JSON integers are represented by Python int: reject bool and 1.0 for source indices.
TYPE_CHECKER = Draft202012Validator.TYPE_CHECKER.redefine('integer', lambda _, v: type(v) is int)
SchemaValidator = validators.extend(Draft202012Validator, type_checker=TYPE_CHECKER)
SCHEMA_BYTES = files('oscardp.stage02').joinpath('stage02_v1.schema.json').read_bytes()
SCHEMA = json.loads(SCHEMA_BYTES)
Draft202012Validator.check_schema(SCHEMA)
SCHEMA_VALIDATOR = SchemaValidator(SCHEMA)
RULES = ('files_identity', 'stage01_contract', 'stage02_schema', 'shot_coverage', 'shot_times',
         'srt_discovery', 'srt_parse', 'source_indices', 'source_text', 'source_overlap')


def read_json(path: Path, *, raw: bytes | None = None) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f'Non-finite JSON number: {value}')
    def floating(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f'Non-finite JSON number: {value}')
        return parsed
    return json.loads((raw if raw is not None else path.read_bytes()).decode('utf-8-sig'), object_pairs_hook=unique, parse_constant=constant, parse_float=floating)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class MovieValidation:
    def __init__(self, movie_dir: Path, tolerance: float):
        self.movie_dir = movie_dir
        self.tolerance = tolerance
        self.issues = []
        self.results = {rule: {'status': 'PASS', 'issue_count': 0} for rule in RULES}
        self.incomplete = set()
        self.inputs = {}
        self.matches = []
        self.source_diagnostics = {}
        self.stats = dict(movie_id=movie_dir.name, movie_name=None, stage01_shots=0, stage02_shots=0,
                          shot_coverage_pct=0.0, srt_status='NOT_CHECKED', srt_cue_count=0,
                          dialogue_shot_count=0, dialogue_entry_count=0, unknown_speaker_count=0,
                          performance_cue_count=0, shots_without_performance_cues=0,
                          exact_match_count=0, normalized_match_count=0, mismatch_count=0)
        self.stats['segment_match_count'] = 0

    def issue(self, rule, code, message, *, severity='ERROR', shot_id=None, srt_index=None, expected=None, actual=None, field=None, entry=None):
        self.issues.append(dict(movie_id=self.movie_dir.name, severity=severity, rule_code=code,
                                shot_id=shot_id, srt_index=srt_index, expected=expected, actual=actual, message=message,
                                domain=domain(rule, code), field=field, entry=entry))
        result = self.results[rule]
        result['issue_count'] += 1
        if severity == 'ERROR' or (severity == 'WARNING' and result['status'] == 'PASS'):
            result['status'] = 'FAIL' if severity == 'ERROR' else 'WARNING'

    def skip(self, *rules):
        self.incomplete.update(rules)
        for rule in rules:
            if self.results[rule]['status'] == 'PASS':
                self.results[rule]['status'] = 'INCOMPLETE'

    def load(self, stage):
        path = self.movie_dir / f'{stage}_{self.movie_dir.name}.json'
        self.inputs[stage] = {'path': str(path.resolve()), 'sha256': None}
        # Detect wrong complete filenames; never accept partNNN as a substitute.
        for candidate in sorted(self.movie_dir.glob(f'{stage}_*.json')):
            if candidate != path and '_part' not in candidate.stem:
                self.issue('files_identity', 'FILENAME_ID', 'Unexpected complete-stage filename in movie directory', expected=path.name, actual=candidate.name)
        try:
            raw = path.read_bytes()
            self.inputs[stage]['sha256'] = hashlib.sha256(raw).hexdigest()
            data = read_json(path, raw=raw)
        except (OSError, ValueError, UnicodeError) as exc:
            self.issue('files_identity', 'INPUT_MISSING' if not path.exists() else 'JSON_INVALID', str(exc), actual=str(path))
            return None
        if not isinstance(data, dict):
            self.issue('files_identity', 'ROOT_TYPE', f'{stage} root must be an object', actual=data)
            return None
        if data.get('movie_id') != self.movie_dir.name:
            self.issue('files_identity', 'MOVIE_ID', f'{stage} ID differs from directory', expected=self.movie_dir.name, actual=data.get('movie_id'))
        name = data.get('movie_name')
        if 'movie_name' not in data or (name is None and stage == 'Stage02') or (name is not None and (not isinstance(name, str) or not name.strip())):
            self.issue('files_identity', 'MOVIE_NAME', f'{stage} movie_name missing or invalid', actual=name)
        return data

    def shots(self, data, stage):
        shots = data.get('shots') if data else None
        if not isinstance(shots, list) or not shots:
            self.issue('shot_coverage', 'SHOTS_INVALID', f'{stage} requires nonempty shots array', actual=shots)
            return []
        self.stats[f'{stage.lower()}_shots'] = len(shots)
        return shots

    def coverage(self, shots1, shots2):
        sequences = []
        for stage, shots in [('Stage01', shots1), ('Stage02', shots2)]:
            ids = [s.get('shot_id') if isinstance(s, dict) else None for s in shots]
            valid_ids = [i for i in ids if isinstance(i, str)]
            for shot_id, count in Counter(valid_ids).items():
                if count > 1:
                    self.issue('shot_coverage', 'SHOT_DUPLICATE', f'{stage} repeats shot ID', shot_id=shot_id, actual=count, expected=1)
            if len(valid_ids) != len(ids):
                self.issue('shot_coverage', 'SHOT_ID_INVALID', f'{stage} contains invalid shot objects/IDs')
            sequences.append(valid_ids)
        ids1, ids2 = sequences
        a, b = set(ids1), set(ids2)
        self.stats['shot_coverage_pct'] = round(100 * len(a & b) / len(a), 6) if a else 0.0
        for shot_id in sorted(a - b):
            self.issue('shot_coverage', 'SHOT_MISSING', 'Stage01 shot missing in Stage02', shot_id=shot_id)
        for shot_id in sorted(b - a):
            self.issue('shot_coverage', 'SHOT_EXTRA', 'Stage02 shot absent from Stage01', shot_id=shot_id)
        if len(shots1) != len(shots2):
            self.issue('shot_coverage', 'SHOT_COUNT', 'Shot counts differ', expected=len(shots1), actual=len(shots2))
        if ids1 != ids2:
            self.issue('shot_coverage', 'SHOT_ORDER', 'Shot ID order differs from Stage01', expected=ids1, actual=ids2)
        for label, shot_id in [('first', ids1[0] if ids1 else None), ('last', ids1[-1] if ids1 else None)]:
            if shot_id and shot_id not in b:
                self.issue('shot_coverage', 'ENDPOINT_MISSING', f'{label} Stage01 shot is missing', shot_id=shot_id)

    def times(self, shots1, shots2):
        reference = {}
        previous = None
        previous_frame = 0
        for stage, shots in [('Stage01', shots1), ('Stage02', shots2)]:
            for pos, shot in enumerate(shots):
                if not isinstance(shot, dict):
                    continue
                shot_id = shot.get('shot_id')
                temporal = shot.get('temporal')
                t = temporal if isinstance(temporal, dict) else {}
                start, end = t.get('start'), t.get('end')
                valid = finite(start) and finite(end) and 0 <= start < end
                if not valid:
                    self.issue('shot_times', 'TIME_INVALID', f'{stage} requires finite 0 <= start < end', shot_id=shot_id, actual=t)
                if stage == 'Stage01':
                    if set(shot) != {'shot_id', 'temporal'} or shot_id != f'shot_{pos + 1:06d}' or set(t) != {'start', 'end', 'start_frame', 'end_frame'}:
                        self.issue('stage01_contract', 'STAGE01_STRUCTURE', 'Stage01 shot violates upstream contract', shot_id=shot_id, actual=shot)
                    sf, ef = t.get('start_frame'), t.get('end_frame')
                    if type(sf) is not int or type(ef) is not int or sf != previous_frame or sf < 0 or sf >= ef:
                        self.issue('stage01_contract', 'STAGE01_FRAMES', 'Invalid or noncontiguous exclusive frame boundaries', shot_id=shot_id, actual=t)
                    if type(ef) is int:
                        previous_frame = ef
                    expected_start = previous if previous is not None else 0
                    if valid and abs(start - expected_start) > self.tolerance:
                        self.issue('stage01_contract', 'STAGE01_CONTINUITY', 'Stage01 has a time gap/overlap or nonzero origin', shot_id=shot_id, expected=expected_start, actual=start)
                    if valid:
                        previous = end
                    if isinstance(shot_id, str):
                        reference[shot_id] = t
                elif isinstance(shot_id, str) and shot_id in reference and valid:
                    for key in ('start', 'end'):
                        expected = reference[shot_id].get(key)
                        if finite(expected) and abs(t[key] - expected) > self.tolerance:
                            self.issue('shot_times', 'TIME_MISMATCH', f'{key} differs from authoritative Stage01', shot_id=shot_id, expected=expected, actual=t[key])

    def schema(self, data):
        for error in SCHEMA_VALIDATOR.iter_errors(data):
            path = list(error.absolute_path)
            shot_id = None
            if len(path) > 1 and path[0] == 'shots' and isinstance(data.get('shots'), list):
                shot = data['shots'][path[1]]
                shot_id = shot.get('shot_id') if isinstance(shot, dict) else None
            self.issue('stage02_schema', 'SCHEMA_INVALID', f'/{"/".join(map(str, path))}: {error.message}', shot_id=shot_id,
                       expected=error.schema, actual=error.instance)

    def sources(self, shots, cues, blocked_indices=frozenset()):
        groups = {}
        for cue in cues:
            groups.setdefault(cue.index, []).append(cue)
        positions = {cue.index: i for i, cue in enumerate(cues)}
        for shot in shots:
            if not isinstance(shot, dict):
                continue
            shot_id = shot.get('shot_id')
            dialogue = shot.get('dialogue', [])
            if isinstance(dialogue, list):
                self.stats['dialogue_entry_count'] += len(dialogue)
                self.stats['dialogue_shot_count'] += bool(dialogue)
                self.stats['unknown_speaker_count'] += sum(isinstance(d, dict) and d.get('speaker') is None for d in dialogue)
            performance = shot.get('performance_cues', [])
            if isinstance(performance, list):
                self.stats['performance_cue_count'] += len(performance)
                self.stats['shots_without_performance_cues'] += not performance
            for field in ('dialogue', 'sound_cues', 'song_lyrics'):
                entries = shot.get(field, [])
                if not isinstance(entries, list):
                    continue
                for entry_no, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        continue
                    indices = entry.get('srt_indices') if field == 'dialogue' else [entry.get('srt_index')]
                    if not isinstance(indices, list) or not indices or any(type(i) is not int or i < 1 for i in indices):
                        self.issue('source_indices', 'SRT_INDICES_INVALID', 'Source indices must be nonempty positive integer arrays', shot_id=shot_id, actual=indices)
                        continue
                    selected = []
                    for index in indices:
                        if index not in groups:
                            if index in blocked_indices:
                                self.issue('source_indices', 'SRT_REFERENCE_BLOCKED', 'Invalid source block prevents source verification; not a missing annotation index', severity='INFO', shot_id=shot_id, srt_index=index, field=field, entry=entry_no)
                            else:
                                self.issue('source_indices', 'SRT_INDEX_MISSING', 'Referenced subtitle does not exist', shot_id=shot_id, srt_index=index)
                        elif len(groups[index]) != 1:
                            timing = self.authoritative_times.get(shot_id, {}) if isinstance(shot_id, str) else {}
                            candidates = [c for c in groups[index] if isinstance(entry.get('text'), str)
                                and compare_entry(entry['text'], [c], field)['status'] != 'mismatch'
                                and finite(timing.get('start')) and finite(timing.get('end'))
                                and min(c.end, timing['end']) > max(c.start, timing['start'])]
                            if len(candidates) == 1 and len(indices) == 1:
                                selected.append(candidates[0])
                                self.issue('source_indices', 'SRT_DUPLICATE_REFERENCE_DISAMBIGUATED',
                                           'One duplicate occurrence matches both text and authoritative shot overlap; original source remains invalid',
                                           severity='INFO', shot_id=shot_id, srt_index=index, actual=vars(candidates[0]), field=field, entry=entry_no)
                            else:
                                self.issue('source_indices', 'SRT_INDEX_AMBIGUOUS', 'Referenced subtitle number is duplicated in source', shot_id=shot_id, srt_index=index, field=field, entry=entry_no)
                        elif index in blocked_indices:
                            self.issue('source_indices', 'SRT_REFERENCE_BLOCKED', 'Empty or invalid source block prevents verification', severity='INFO', shot_id=shot_id, srt_index=index, field=field, entry=entry_no)
                        else:
                            selected.append(groups[index][0])
                    if len(selected) != len(indices):
                        self.skip('source_text', 'source_overlap')
                        continue
                    if len(set(indices)) != len(indices) or any(positions[b] != positions[a] + 1 for a, b in zip(indices, indices[1:])):
                        self.issue('source_indices', 'SRT_MERGE_ORDER', 'Merged cues must be consecutive in original source order', shot_id=shot_id, actual=indices)
                    text = entry.get('text')
                    if isinstance(text, str):
                        evidence = compare_entry(text, selected, field)
                        self.stats['segment_match_count'] += evidence['match_rule'] != 'whole_cue' and evidence['status'] != 'mismatch'
                        self.stats[evidence['status'] + '_match_count' if evidence['status'] != 'mismatch' else 'mismatch_count'] += 1
                        self.matches.append(dict(shot_id=shot_id, field=field, entry=entry_no, srt_indices=indices,
                                                 source_cues=[vars(c) for c in selected], **evidence))
                        if evidence['status'] != 'exact':
                            self.issue('source_text', 'TEXT_NORMALIZED' if evidence['status'] == 'normalized' else 'TEXT_MISMATCH',
                                       'Formatting-only differences: ' + ', '.join(evidence['reasons']) if evidence['status'] == 'normalized' else 'Text differs beyond permitted normalization; inspect subtitle version/source',
                                       severity='WARNING' if evidence['status'] == 'normalized' else 'ERROR', shot_id=shot_id,
                                       srt_index=indices, expected=evidence['expected'], actual=text, field=field, entry=entry_no)
                        if evidence['classification_error']:
                            self.issue('source_text', 'FIELD_CLASSIFICATION', evidence['classification_error'], shot_id=shot_id,
                                       srt_index=indices, expected={'field': field, 'source_segments': evidence['source_segments']},
                                       actual=text, field=field, entry=entry_no)
                    # Stage01 boundaries remain authoritative even if Stage02 time is corrupt.
                    t = self.authoritative_times.get(shot_id, {}) if isinstance(shot_id, str) else {}
                    if finite(t.get('start')) and finite(t.get('end')):
                        for cue in selected:
                            if min(cue.end, t['end']) <= max(cue.start, t['start']):
                                gap = max(t['start'] - cue.end, cue.start - t['end'], 0)
                                self.issue('source_overlap', 'SRT_NO_OVERLAP', 'Cue has no positive temporal overlap with authoritative shot; no offset applied',
                                           shot_id=shot_id, srt_index=cue.index, expected=t, actual=vars(cue))
                                if gap > 0:
                                    self.issue('source_overlap', 'SRT_POSSIBLE_OFFSET', 'Possible subtitle offset or version mismatch; requires manual review', severity='WARNING', shot_id=shot_id, srt_index=cue.index, actual={'gap_seconds': gap})
                    else:
                        self.skip('source_overlap')

    def finish(self):
        roots = aggregate(self.issues)
        self.stats['errors'] = sum(i['severity'] == 'ERROR' for i in self.issues)
        self.stats['warnings'] = sum(i['severity'] == 'WARNING' for i in self.issues)
        self.stats['validation_status'] = 'FAIL' if self.stats['errors'] or self.incomplete else 'WARNING' if self.stats['warnings'] else 'PASS'
        self.stats['root_errors'] = sum(r['severity'] == 'ERROR' for r in roots)
        self.stats['root_warnings'] = sum(r['severity'] == 'WARNING' for r in roots)
        self.stats['validation_complete'] = not self.incomplete
        for name in ('SOURCE_INTEGRITY', 'ANNOTATION_CONSISTENCY'):
            findings = [i for i in self.issues if i['domain'] == name]
            if any(i['severity'] == 'ERROR' for i in findings):
                status = 'FAIL'
            elif self.incomplete & (set(RULES) - {'srt_discovery', 'srt_parse'} if name == 'ANNOTATION_CONSISTENCY' else {'srt_discovery', 'srt_parse'}):
                status = 'INCOMPLETE'
            else:
                status = 'WARNING' if any(i['severity'] == 'WARNING' for i in findings) else 'PASS'
            self.stats[name.lower() + '_status'] = status
        return dict(scope=SCOPE, limitation=LIMITATION, validator_version='1.1', tolerance_seconds=self.tolerance,
                    schema_sha256=hashlib.sha256(SCHEMA_BYTES).hexdigest(), summary=self.stats, inputs=self.inputs,
                    rule_results=self.results, issues=self.issues, root_problems=roots,
                    incomplete_checks=sorted(self.incomplete), source_diagnostics=self.source_diagnostics,
                    text_comparisons=self.matches)


def validate_movie(movie_dir: Path, srt_candidates: list[Path], *, srt_override: Path | None = None,
                   tolerance: float = 1e-6, discovery_error: str | None = None,
                   ambiguous_ownership: bool = False) -> dict:
    if not finite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    v = MovieValidation(movie_dir, tolerance)
    one, two = v.load('Stage01'), v.load('Stage02')
    if one is None:
        v.skip('stage01_contract')
    if one is not None and set(one) != {'movie_id', 'movie_name', 'shots'}:
        v.issue('stage01_contract', 'STAGE01_ROOT', 'Stage01 root differs from upstream contract', actual=one)
    if two is not None:
        v.stats['movie_name'] = two.get('movie_name')
        v.schema(two)
    else:
        v.skip('stage02_schema')
    if one and two and one.get('movie_name') is not None and one.get('movie_name') != two.get('movie_name'):
        v.issue('files_identity', 'MOVIE_NAME_CONFLICT', 'Movie titles differ', expected=one.get('movie_name'), actual=two.get('movie_name'))
    shots1, shots2 = v.shots(one, 'Stage01'), v.shots(two, 'Stage02')
    if not shots1:
        v.skip('stage01_contract')
    if not shots1 or not shots2:
        v.skip('shot_coverage', 'shot_times')
    v.coverage(shots1, shots2)
    v.times(shots1, shots2)
    v.authoritative_times = {s['shot_id']: s['temporal'] for s in shots1 if isinstance(s, dict) and isinstance(s.get('shot_id'), str) and isinstance(s.get('temporal'), dict)}
    candidates = [srt_override.resolve()] if srt_override else sorted(set(srt_candidates))
    v.inputs['srt_candidates'] = [str(p) for p in candidates]
    parsed = None
    if discovery_error and not srt_override:
        v.stats['srt_status'] = 'DISCOVERY_FAILED'
        v.issue('srt_discovery', 'SRT_DISCOVERY_FAILED', discovery_error)
    elif len(candidates) != 1 or (ambiguous_ownership and not srt_override):
        code = 'SRT_NOT_FOUND' if not candidates else 'SRT_AMBIGUOUS'
        v.stats['srt_status'] = code.removeprefix('SRT_')
        v.issue('srt_discovery', code, 'A unique confirmed subtitle source is required', actual=[str(p) for p in candidates])
    else:
        path = candidates[0]
        v.inputs['SRT'] = {'path': str(path.resolve()), 'sha256': None, 'selection': 'explicit_mapping' if srt_override else 'imdb_id'}
        try:
            raw = path.read_bytes()
            v.inputs['SRT']['sha256'] = hashlib.sha256(raw).hexdigest()
            parsed = parse_srt(raw.decode('utf-8-sig'))
            v.stats['srt_status'] = 'INVALID' if parsed.issues else 'FOUND'
            v.stats['srt_cue_count'] = len(parsed.cues)
            v.source_diagnostics = {'parse_issues': parsed.issues, 'numbered_block_count': len(parsed.blocks),
                                    'valid_timed_cue_count': len(parsed.cues)}
            diagnostics = parsed.issues
            if not parsed.cues and len(diagnostics) > 1:
                diagnostics = [dict(rule_code='SRT_PARSE_FAILED', message='No usable timed cues; dependent source errors aggregated',
                                    actual={'diagnostic_count': len(parsed.issues), 'examples': parsed.issues[:5]})]
            for issue in diagnostics:
                v.issue('srt_parse', issue['rule_code'], issue['message'], srt_index=issue.get('srt_index'), actual=issue.get('actual'))
        except (OSError, UnicodeError) as exc:
            v.stats['srt_status'] = 'UNREADABLE'
            v.issue('srt_parse', 'SRT_UNREADABLE', str(exc), actual=str(path))
    if parsed is None:
        v.skip('srt_parse', 'source_indices', 'source_text', 'source_overlap')
    elif not parsed.cues:
        v.skip('source_indices', 'source_text', 'source_overlap')
    if not shots2:
        v.skip('source_indices', 'source_text', 'source_overlap')
    # Collect annotation statistics even when SRT cannot be checked.
    if parsed and parsed.cues:
        blocked = {b['index'] for b in parsed.blocks if not b['time_valid'] or not b['text'].strip()}
        v.sources(shots2, parsed.cues, blocked)
    else:
        _annotation_stats(v, shots2)
    if v.stats['unknown_speaker_count']:
        v.issue('source_indices', 'UNKNOWN_SPEAKERS', 'Unknown speakers are permitted', severity='INFO', actual=v.stats['unknown_speaker_count'])
    if v.stats['shots_without_performance_cues']:
        v.issue('stage02_schema', 'NO_PERFORMANCE_CUES', 'Optional performance cues absent/empty', severity='INFO', actual=v.stats['shots_without_performance_cues'])
    return v.finish()


def _annotation_stats(v, shots):
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        d, p = shot.get('dialogue', []), shot.get('performance_cues', [])
        if isinstance(d, list):
            v.stats['dialogue_entry_count'] += len(d)
            v.stats['dialogue_shot_count'] += bool(d)
            v.stats['unknown_speaker_count'] += sum(isinstance(e, dict) and e.get('speaker') is None for e in d)
        if isinstance(p, list):
            v.stats['performance_cue_count'] += len(p)
            v.stats['shots_without_performance_cues'] += not p
