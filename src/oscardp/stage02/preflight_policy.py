"""Source version comparison, fail-closed review approvals and report export."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re

from .validator import read_json

SHA256 = re.compile(r'[0-9a-f]{64}')


def compare_baselines(paths, movie_id, inputs, add):
    """Only explicit manifests and matching paths supply historical fingerprints."""
    comparisons = []
    for path in paths:
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError('Baseline manifest must be a JSON object')
        records = data.get('movies', [data])
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise ValueError('Baseline movies must be objects')
        matched = 0
        for record in records:
            summary = record.get('summary', {})
            if not isinstance(summary, dict):
                raise ValueError('Baseline summary must be an object')
            if record.get('movie_id', summary.get('movie_id')) != movie_id:
                continue
            if not isinstance(record.get('inputs', {}), dict):
                raise ValueError('Baseline inputs must be an object')
            sources = dict(record.get('inputs', {}))
            for kind, label in [('Stage01','stage01'),('Stage02','stage02'),('SRT','srt')]:
                for prefix in ('original', 'curated'):
                    key = f'{prefix}_{label}_path'
                    if key in record:
                        sources[f'{prefix}_{kind}'] = dict(path=record[key], sha256=record.get(f'{prefix}_{label}_sha256'))
            for label, old in sources.items():
                kind = label.removeprefix('original_').removeprefix('curated_')
                if kind not in inputs or not isinstance(old, dict) or not isinstance(old.get('path'), str):
                    continue
                source_path = Path(old['path']).expanduser()
                if not source_path.is_absolute():
                    source_path = path.parent / source_path
                if source_path.resolve() != Path(inputs[kind]['path']):
                    continue
                old_sha = old.get('sha256')
                if not isinstance(old_sha, str) or not SHA256.fullmatch(old_sha):
                    raise ValueError(f'Invalid baseline SHA-256 for {kind}: {path}')
                matched += 1
                changed = old_sha != inputs[kind]['sha256']
                row = dict(manifest_path=str(path), input=kind, path=inputs[kind]['path'],
                           historical_sha256=old_sha, current_sha256=inputs[kind]['sha256'], changed=changed)
                comparisons.append(row)
                if changed:
                    add('SOURCE_VERSION_CHANGED', 'WARNING', kind,
                        'Current bytes differ from the explicit historical manifest. Historical data and hashes were not modified.', row, True)
        if not matched:
            add('SOURCE_BASELINE_UNMATCHED', 'WARNING', 'SRT', 'No baseline record matches this movie and actual input path.',
                dict(manifest_path=str(path), movie_id=movie_id), True)
    return dict(requested=bool(paths), comparisons=comparisons,
                status='CHANGED' if any(c['changed'] for c in comparisons) else 'UNCHANGED' if comparisons else 'NOT_COMPARED')


def apply_approvals(path, movie_id, inputs, issues, add):
    result = dict(provided=path is not None, approved_issue_ids=[], rejected_records=[])
    if path is None:
        return result
    data = read_json(path)
    if not isinstance(data, dict) or data.get('schema_version') != '1.0' or not isinstance(data.get('exceptions'), list):
        raise ValueError('Approval file requires schema_version "1.0" and an exceptions array')
    expected = {kind: source['sha256'] for kind, source in inputs.items()}
    by_id = {issue['issue_id']: issue for issue in issues}
    seen = set()
    for record in data['exceptions']:
        if not isinstance(record, dict) or any(not isinstance(record.get(k), str) or not record[k].strip()
                for k in ['issue_id','rule_code','movie_id','approved_by','evidence','decision']):
            raise ValueError('Each approval requires issue_id, rule_code, movie_id, approved_by, evidence and decision strings')
        hashes = record.get('input_sha256')
        if not isinstance(hashes, dict) or not hashes or any(not isinstance(v,str) or not SHA256.fullmatch(v) for v in hashes.values()):
            raise ValueError('Each approval requires explicit input_sha256 fingerprints')
        issue = by_id.get(record['issue_id'])
        reason = None
        if record['issue_id'] in seen:
            reason = 'Duplicate approval record'
        elif record['movie_id'] != movie_id or hashes != expected:
            reason = 'Movie identity or exact set of current input hashes differs'
        elif not issue or issue['rule_code'] != record['rule_code']:
            reason = 'Issue ID / rule does not identify a current finding'
        elif issue['severity'] != 'WARNING' or not issue['requires_approval']:
            reason = 'Only explicitly reviewable WARNINGs can be approved; integrity/schema blockers cannot be waived'
        elif record['decision'] != 'APPROVED_NONBLOCKING':
            reason = 'Decision must explicitly approve a nonblocking exception'
        seen.add(record['issue_id'])
        if reason:
            rejection = dict(issue_id=record['issue_id'], reason=reason)
            result['rejected_records'].append(rejection)
            add('APPROVAL_BINDING_INVALID', 'BLOCKED', 'approved_exceptions', 'Approval rejected; gate remains closed.', rejection)
        else:
            issue['approval'] = dict(status='APPROVED', approved_by=record['approved_by'], evidence=record['evidence'],
                                     decision=record['decision'], input_sha256=hashes)
            result['approved_issue_ids'].append(record['issue_id'])
    return result


def write_preflight_reports(root, report):
    root.mkdir(parents=True, exist_ok=True)
    with (root/'source_preflight.json').open('x',encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    fields = ['issue_id','rule_code','severity','input','requires_approval','message','evidence','approval']
    with (root/'issues.csv').open('x',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for issue in report['issues']:
            writer.writerow({k:json.dumps(issue[k],ensure_ascii=False) if isinstance(issue[k],(dict,list)) else issue[k] for k in fields})
    lines = ['# Stage02 Source Preflight v1.0', '', f"Movie: `{report['movie_id']}`",
             f"Quality: **{report['preflight_status']}**; `llm_annotation_allowed={str(report['llm_annotation_allowed']).lower()}`.",
             f"Exit code: `{report['exit_code']}`. No input repair, release or activation performed.", '',
             '## Inputs', '', '| Input | Path | SHA-256 |', '| --- | --- | --- |']
    for kind, source in report['inputs'].items():
        # Keep Markdown links/data unambiguous even for unusual input filenames.
        safe = source['path'].replace('|','\\|').replace('\n',' ')
        lines.append(f"| {kind} | {safe} | {source['sha256']} |")
    lines += ['', '## Check completeness', '', '| Check | Completed |', '| --- | --- |']
    lines += [f'| {name} | {value} |' for name,value in report['check_completeness'].items()]
    lines += ['', '## Findings', '']
    for issue in report['issues']:
        lines.append(f"- `{issue['severity']} / {issue['rule_code']} / {issue['issue_id']}`: {issue['message']} Approval: {issue['approval']['status']}.")
    if not report['issues']:
        lines.append('No deterministic source findings.')
    compatibility = report['existing_stage02_compatibility']
    if compatibility['requested']:
        lines += ['', '## Existing Stage02 compatibility', '',
                  f"Checks completed: `{compatibility['completed']}`. Stale source references: `{compatibility.get('stale_reference_count',0)}`."]
    lines += ['', '## Limits', '',
              'The exact movie edition and screenplay/movie semantic match remain UNVERIFIED. '
              'Boundary checks cannot establish audio synchronization or authoritative replacement times. '
              'PDF checks inspect the text layer only; no OCR or LLM is run. '
              'Approved warnings stay visible and bind every selected input hash and one exact issue. '
              'A preflight admission is separate from post-annotation Validator v1.1.1 and dataset activation.', '',
              'Full issue-specific evidence, parser versions, executed rules and provenance are in `source_preflight.json`. '
              'Keep this report local; anomalous subtitle blocks may contain source language.', '']
    with (root/'PREFLIGHT.md').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
