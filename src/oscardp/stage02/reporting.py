"""Batch report serialization outside the official movie directories."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .validator import SCOPE, LIMITATION, VERSION

SUMMARY_FIELDS = ['movie_id', 'movie_name', 'stage01_shots', 'stage02_shots', 'shot_coverage_pct',
                  'srt_status', 'srt_cue_count', 'dialogue_shot_count', 'dialogue_entry_count',
                  'unknown_speaker_count', 'performance_cue_count', 'exact_match_count',
                  'normalized_match_count', 'mismatch_count', 'errors', 'warnings', 'validation_status']
SUMMARY_FIELDS += ['source_integrity_status', 'annotation_consistency_status', 'validation_complete',
                   'root_errors', 'root_warnings', 'segment_match_count']
ISSUE_FIELDS = ['movie_id', 'severity', 'rule_code', 'shot_id', 'srt_index', 'expected', 'actual', 'message',
                'domain', 'field', 'entry', 'issue_id', 'root_problem_id']


def write_reports(root: Path, reports: list[dict], batch_issues: list[dict], *, tolerance: float) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    movies = root / 'movies'
    movies.mkdir(exist_ok=True)
    outputs = [root / 'summary.json', root / 'summary.csv', root / 'issues.csv', root / 'root_problems.csv',
               *(movies / f'{r["summary"]["movie_id"]}.json' for r in reports)]
    if any(p.exists() or p.is_symlink() for p in outputs):
        raise FileExistsError('Reports already exist; choose a new --report-root (no silent overwrite)')
    counts = Counter(r['summary']['validation_status'] for r in reports)
    summary = dict(scope=SCOPE, limitation=LIMITATION, validator_version=VERSION, created_at=datetime.now(timezone.utc).isoformat(),
                   tolerance_seconds=tolerance, movie_count=len(reports), counts={k: counts[k] for k in ['PASS', 'WARNING', 'FAIL']},
                   validation_status='FAIL' if counts['FAIL'] or batch_issues else 'WARNING' if counts['WARNING'] else 'PASS',
                   batch_issues=batch_issues, movies=[r['summary'] for r in reports])
    def dump(path, data):
        with path.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    for report in reports:
        dump(movies / f'{report["summary"]["movie_id"]}.json', report)
    dump(root / 'summary.json', summary)
    with (root / 'summary.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[*SUMMARY_FIELDS, 'scope', 'limitation'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(dict(r['summary'], scope=SCOPE, limitation=LIMITATION) for r in reports)
    with (root / 'issues.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[*ISSUE_FIELDS, 'scope', 'limitation'])
        writer.writeheader()
        for issue in [*batch_issues, *(i for r in reports for i in r['issues'])]:
            row = {k: json.dumps(issue.get(k), ensure_ascii=False) if isinstance(issue.get(k), (dict, list)) else issue.get(k, '') for k in ISSUE_FIELDS}
            writer.writerow(dict(row, scope=SCOPE, limitation=LIMITATION))
    root_fields = ['movie_id', 'root_problem_id', 'severity', 'domain', 'rule_codes', 'srt_indices', 'affected_shots', 'occurrence_count', 'issue_ids']
    with (root / 'root_problems.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[*root_fields, 'scope', 'limitation'])
        writer.writeheader()
        for report in reports:
            for problem in report.get('root_problems', []):
                row = {key: json.dumps(problem[key], ensure_ascii=False) if isinstance(problem[key], list) else problem[key] for key in root_fields}
                writer.writerow(dict(row, scope=SCOPE, limitation=LIMITATION))
    return summary
