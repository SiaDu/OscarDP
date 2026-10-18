"""Stable root-problem grouping while retaining every affected occurrence."""
from __future__ import annotations

import hashlib
import json

SOURCE_CODES = {'SRT_DUPLICATE_INDEX', 'SRT_INDEX_AMBIGUOUS', 'SRT_REFERENCE_BLOCKED', 'SRT_DUPLICATE_REFERENCE_DISAMBIGUATED'}


def domain(rule: str, code: str) -> str:
    if code in SOURCE_CODES or rule in ('srt_discovery', 'srt_parse'):
        return 'SOURCE_INTEGRITY'
    return 'ANNOTATION_CONSISTENCY'


def aggregate(issues: list[dict]) -> list[dict]:
    roots = {}
    for number, issue in enumerate(issues, 1):
        issue['issue_id'] = f'issue_{number:06d}'
        indices = issue.get('srt_index')
        indices = indices if isinstance(indices, list) else [indices] if indices is not None else []
        if issue['domain'] == 'SOURCE_INTEGRITY' and indices:
            key = [issue['movie_id'], 'source_index_integrity', indices]
        elif issue['rule_code'] == 'SRT_UNSUPPORTED_FORMAT':
            key = [issue['movie_id'], 'source_format']
        else:
            key = [issue['movie_id'], issue['domain'], issue['rule_code'], indices,
                   issue.get('field'), issue.get('expected'), issue.get('actual')]
        digest = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
        root_id = 'root_' + digest
        issue['root_problem_id'] = root_id
        root = roots.setdefault(root_id, dict(root_problem_id=root_id, movie_id=issue['movie_id'],
            severity='INFO', domain=issue['domain'], rule_codes=[], affected_shots=[], srt_indices=indices,
            occurrence_count=0, issue_ids=[]))
        root['occurrence_count'] += 1
        root['issue_ids'].append(issue['issue_id'])
        if issue['rule_code'] not in root['rule_codes']:
            root['rule_codes'].append(issue['rule_code'])
        if issue.get('shot_id') and issue['shot_id'] not in root['affected_shots']:
            root['affected_shots'].append(issue['shot_id'])
        if {'INFO': 0, 'WARNING': 1, 'ERROR': 2}[issue['severity']] > {'INFO': 0, 'WARNING': 1, 'ERROR': 2}[root['severity']]:
            root['severity'] = issue['severity']
    return list(roots.values())
