"""Reconcile saved validation runs without modifying any annotation or source.

All curation and JSON Patch artifacts remain unapproved review proposals.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from .curation import dump, investigate, sha
from .srt import parse_srt, normalized_text
from .validator import LIMITATION, SCOPE, read_json

DECISIONS = ('RESOLVED_VALIDATOR_RULE', 'RESOLVED_SOURCE_CURATION', 'STAGE02_CORRECTION_PROPOSED',
             'ACCEPTED_FORMAT_DIFFERENCE', 'NEEDS_HUMAN_REVIEW', 'UNRESOLVED')


def indices(value):
    return value if isinstance(value, list) else [] if value is None else [value]


def _root_key(movie_id, issue, field, unsupported):
    code = issue['rule_code']
    if unsupported and code.startswith('SRT_'):
        return [movie_id, 'unconfirmed_source_format']
    if code.startswith('SRT_') and code not in ('SRT_NO_OVERLAP', 'SRT_POSSIBLE_OFFSET', 'SRT_MERGE_ORDER'):
        return [movie_id, 'source_quality', indices(issue.get('srt_index'))]
    return [movie_id, code, indices(issue.get('srt_index')), field, issue.get('expected'), issue.get('actual')]


def patches_for(report: dict) -> list[dict]:
    """Propose exact text replacements only for explicitly typed contamination."""
    stage = read_json(Path(report['inputs']['Stage02']['path']))
    positions = {shot['shot_id']: i for i, shot in enumerate(stage['shots'])}
    result = []
    for comparison in report['text_comparisons']:
        if not comparison.get('classification_error'):
            continue
        field = comparison['field']
        parts = [p for source in comparison['source_segments'] for p in source['segments'] if p['kind'] == field]
        before = comparison['actual']
        after = '\n'.join(p['text'] for p in parts)
        shot_id, entry = comparison['shot_id'], comparison['entry']
        affected = [i for i in report['issues'] if i.get('shot_id') == shot_id and i.get('field') == field and i.get('entry') == entry and i['rule_code'] == 'FIELD_CLASSIFICATION']
        if not parts or after == before:
            continue
        pointer = f'/shots/{positions[shot_id]}/{field}/{entry}/text'
        complementary = [dict(p, srt_index=source['srt_index']) for source in comparison['source_segments'] for p in source['segments'] if p['kind'] in ('dialogue', 'song_lyrics') and p['kind'] != field]
        shot = stage['shots'][positions[shot_id]]
        missing = [p for p in complementary if not any(normalized_text(e['text'])[0] == normalized_text(p['text'])[0] for e in shot.get(p['kind'], []))]
        result.append(dict(movie_id=stage['movie_id'], shot_id=shot_id, srt_indices=comparison['srt_indices'],
            field=field, entry=entry, root_problem_ids=sorted({i['root_problem_id'] for i in affected}),
            path=pointer, before=before, after=after, source_evidence=comparison['source_cues'], source_segments=comparison['source_segments'],
            reason='Keep the complete explicitly delimited target-field component(s); remove spoken dialogue/lyrics assigned to another field. No paraphrase or speaker inference.',
            requires_human_confirmation=True, applied=False, stage02_sha256=report['inputs']['Stage02']['sha256'],
            missing_complementary_components=missing, approval_constraint='Review complementary components before approval; this proposal alone does not add missing dialogue or infer speakers.',
            json_patch=[dict(op='test', path='/movie_id', value=stage['movie_id']),
                        dict(op='test', path=f'/shots/{positions[shot_id]}/shot_id', value=shot_id),
                        dict(op='test', path=pointer, value=before), dict(op='replace', path=pointer, value=after)]))
    return result


def file_patch_bundles(patches: list[dict], validation: Path) -> list[dict]:
    """Consolidate overlapping shot proposals into coherent, guarded file plans."""
    from copy import deepcopy
    from .validator import SCHEMA_VALIDATOR
    grouped = defaultdict(list)
    for proposal in patches:
        grouped[proposal['movie_id']].append(proposal)
    bundles = []
    for movie_id, proposals in sorted(grouped.items()):
        report = read_json(validation / 'movies' / f'{movie_id}.json')
        source_path = Path(report['inputs']['Stage02']['path'])
        before = read_json(source_path)
        after = deepcopy(before)
        positions = {s['shot_id']: i for i, s in enumerate(after['shots'])}
        changes = set()
        companions = []
        for proposal in proposals:
            pos, field, entry = positions[proposal['shot_id']], proposal['field'], proposal['entry']
            assert after['shots'][pos][field][entry]['text'] == proposal['before']
            after['shots'][pos][field][entry]['text'] = proposal['after']
            changes.add((pos, field))
            for component in proposal['missing_complementary_components']:
                target = component['kind']
                value = dict(srt_indices=[component['srt_index']], speaker=None, text=component['text']) if target == 'dialogue' else dict(srt_index=component['srt_index'], text=component['text'])
                entries = after['shots'][pos].setdefault(target, [])
                if value not in entries:
                    entries.append(value)
                    changes.add((pos, target))
                    companions.append(dict(shot_id=proposal['shot_id'], field=target, value=value,
                        reason='Preserve the complete source component displaced from the wrong field; speaker remains null pending screenplay review'))
        schema_errors = list(SCHEMA_VALIDATOR.iter_errors(after))
        if schema_errors:
            raise ValueError(f'Proposed file bundle violates fixed schema: {movie_id}: {schema_errors[0].message}')
        operations = [dict(op='test', path='/movie_id', value=movie_id)]
        before_after = []
        for pos, field in sorted(changes):
            pointer = f'/shots/{pos}/{field}'
            old_value, new_value = before['shots'][pos].get(field), after['shots'][pos][field]
            operations.append(dict(op='test', path=f'/shots/{pos}/shot_id', value=before['shots'][pos]['shot_id']))
            if field in before['shots'][pos]:
                operations.append(dict(op='test', path=pointer, value=old_value))
                operations.append(dict(op='replace', path=pointer, value=new_value))
            else:
                operations.append(dict(op='add', path=pointer, value=new_value))
            before_after.append(dict(path=pointer, before=old_value, after=new_value))
        bundles.append(dict(movie_id=movie_id, stage02_path=str(source_path), stage02_sha256=sha(source_path),
            applied=False, requires_human_confirmation=True, schema_checked_in_memory=True,
            classification_occurrences=len(proposals), root_problem_ids=sorted({r for p in proposals for r in p['root_problem_ids']}),
            companion_entries=companions, changes=before_after, json_patch=operations,
            note='Consolidated file plan includes companion dialogue/lyrics; no output annotation JSON is written. Null speakers require human/context review.'))
    return bundles


def reconcile(old: dict, current: dict, investigation: dict) -> tuple[list[dict], list[dict]]:
    movie_id = current['summary']['movie_id']
    old_comparisons = old.get('text_comparisons', [])
    new_comparisons = current.get('text_comparisons', [])
    unsupported = any(i['rule_code'] == 'SRT_UNSUPPORTED_FORMAT' for i in current['issues'])
    source = current['inputs'].get('SRT')
    blocks = parse_srt(Path(source['path']).read_text(encoding='utf-8-sig')).blocks if source else []
    curation = [p for p in investigation.get('proposals', []) if p['type'] == 'DERIVED_SRT_CURATION']
    rows = []
    roots = {}
    unsupported_primary_seen = False
    for number, issue in enumerate(old['issues'], 1):
        if issue['severity'] not in ('ERROR', 'WARNING'):
            continue
        refs = indices(issue.get('srt_index'))
        matches = [c for c in old_comparisons if c['shot_id'] == issue.get('shot_id') and c['srt_indices'] == refs and c['actual'] == issue.get('actual')]
        fields = sorted({c['field'] for c in matches})
        field = fields[0] if len(fields) == 1 else None
        updated = [c for c in new_comparisons if c['shot_id'] == issue.get('shot_id') and c['srt_indices'] == refs and c['actual'] == issue.get('actual') and (field is None or c['field'] == field)]
        decision, reason = 'UNRESOLVED', 'No independently justified resolution found'
        confirmation, source_change, candidate_only = True, False, False
        code = issue['rule_code']
        if code.startswith('TEXT_'):
            if updated and any(c.get('classification_error') for c in updated):
                decision, reason = 'STAGE02_CORRECTION_PROPOSED', 'Explicit source components are assigned to the wrong field; see unapplied JSON Patch proposal'
            elif updated and all(c['status'] == 'normalized' for c in updated):
                decision, reason = 'ACCEPTED_FORMAT_DIFFERENCE', 'Only the documented presentation/Unicode/whitespace transformations differ; raw evidence retained'
                confirmation = False
            elif updated and all(c['status'] == 'exact' for c in updated):
                decision, reason = 'RESOLVED_VALIDATOR_RULE', 'A complete typed source segment/projection is reproduced exactly; arbitrary substrings are still rejected'
                confirmation = False
            elif updated:
                decision, reason = 'NEEDS_HUMAN_REVIEW', 'Text still differs beyond permitted whole-cue/typed-segment rules'
        elif code == 'SRT_INDEX_AMBIGUOUS' and any(i['rule_code'] == 'SRT_DUPLICATE_REFERENCE_DISAMBIGUATED' and i.get('shot_id') == issue.get('shot_id') and indices(i.get('srt_index')) == refs for i in current['issues']):
            decision, reason = 'RESOLVED_VALIDATOR_RULE', 'Unique duplicate occurrence satisfies both source text and Stage01 overlap; original source duplicate remains ERROR'
            confirmation = False
        elif unsupported and code in ('SRT_FORMAT', 'SRT_EMPTY', 'SRT_INDEX_MISSING'):
            if not unsupported_primary_seen and code != 'SRT_INDEX_MISSING':
                decision, reason = 'UNRESOLVED', 'Primary unsupported-source problem: original cue numbers cannot be confirmed. Subsequent parser/index findings share this root; annotation checks remain INCOMPLETE'
                source_change = True
                unsupported_primary_seen = True
            else:
                decision, reason = 'RESOLVED_VALIDATOR_RULE', 'Dependent parser/index cascade suppressed under one unsupported-format root; source numbering remains unresolved and annotation checks INCOMPLETE'
                confirmation = False
        elif any(any(b['index'] in refs for b in p['removed_blocks']) for p in curation):
            decision, reason = 'RESOLVED_SOURCE_CURATION', 'A separately hashed derivative removes only evidenced advertising/empty blocks; candidate revalidation recorded. Original-source run remains unchanged; promotion requires approval'
            source_change, candidate_only = True, True
        elif code == 'SRT_TIMESTAMP':
            decision, reason = 'NEEDS_HUMAN_REVIEW', 'Nonempty zero/invalid-duration source cue has no authoritative replacement timing; no deletion or timing guess proposed'
            source_change = True
        elif unsupported:
            decision, reason = 'UNRESOLVED', 'No authoritative numbered subtitle found in the inspected movie directory; content/time links do not prove original cue numbers'
            source_change = True
        key = _root_key(movie_id, issue, field, unsupported)
        root_id = 'baseline_root_' + hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
        old_id = f'{movie_id}:issue_{number:06d}'
        root = roots.setdefault(root_id, dict(root_problem_id=root_id, primary_record=old_id, affected_shots=[], occurrence_count=0, decisions=[]))
        if issue.get('shot_id') and issue['shot_id'] not in root['affected_shots']:
            root['affected_shots'].append(issue['shot_id'])
        root['occurrence_count'] += 1
        if decision not in root['decisions']:
            root['decisions'].append(decision)
        rows.append(dict(baseline_issue_id=old_id, movie_id=movie_id, shot_id=issue.get('shot_id'), srt_indices=refs,
            old_rule_code=code, old_severity=issue['severity'], field=field, old_issue=issue,
            original_text_evidence=dict(expected=issue.get('expected'), actual=issue.get('actual'),
                source_blocks=[b for b in blocks if b['index'] in refs], comparisons=matches,
                source_path=source['path'] if source else None, source_sha256=source['sha256'] if source else None),
            decision=decision, reason=reason, requires_source_change=source_change,
            requires_human_confirmation=confirmation, candidate_scenario_only=candidate_only,
            applied=False, current_comparisons=updated, new_movie_status=current['summary']['validation_status'],
            source_change_mode='DERIVATIVE_OR_AUTHORITATIVE_SOURCE_ONLY' if source_change else 'NONE', original_source_overwrite_required=False,
            root_problem_id=root_id, duplicate_of=root['primary_record'] if root['primary_record'] != old_id else None,
            independent_problem=root['primary_record'] == old_id))
    return rows, list(roots.values())


def _csv(path, rows, fields):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in fields})


def run_review(baseline: Path, validation: Path, output: Path, *, historical: Path | None = None) -> dict:
    baseline, validation, output = (p.expanduser().resolve() for p in (baseline, validation, output))
    old_summary, new_summary = read_json(baseline / 'summary.json'), read_json(validation / 'summary.json')
    if output.is_relative_to(baseline):
        raise ValueError('Review output must be outside the preserved baseline directory')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Review output must be a fresh directory')
    if {r['movie_id'] for r in old_summary['movies']} != {r['movie_id'] for r in new_summary['movies']}:
        raise ValueError('Baseline/current movie sets differ; cannot claim a full matched review')
    for r in new_summary['movies']:
        movie = read_json(validation / 'movies' / f'{r["movie_id"]}.json')
        for source in (movie['inputs'].get(k) for k in ('Stage01', 'Stage02', 'SRT')):
            if source and output.is_relative_to(Path(source['path']).resolve().parent):
                raise ValueError('Review artifacts must be outside source movie/subtitle directories')
    # Verify baseline CSV and JSON agree, and fingerprint everything before analysis.
    csv_fields = ('movie_id', 'severity', 'rule_code', 'shot_id', 'srt_index', 'expected', 'actual', 'message')
    def exported(value):
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return '' if value is None else str(value)
    with (baseline / 'issues.csv').open(encoding='utf-8', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
        csv_counts = Counter((r['movie_id'], r['severity'], r['rule_code']) for r in csv_rows if r['severity'] in ('ERROR', 'WARNING'))
        csv_evidence = Counter(tuple(r.get(k, '') for k in csv_fields) for r in csv_rows)
    json_counts = Counter()
    json_evidence = Counter()
    fingerprints = {}
    input_records = []
    for root in (baseline, validation):
        for path in [root / 'summary.json', root / 'issues.csv', *sorted((root / 'movies').glob('*.json'))]:
            fingerprints[str(path)] = sha(path)
    for row in old_summary['movies']:
        mid = row['movie_id']
        old, new = (read_json(root / 'movies' / f'{mid}.json') for root in (baseline, validation))
        json_counts.update((mid, i['severity'], i['rule_code']) for i in old['issues'] if i['severity'] in ('ERROR', 'WARNING'))
        json_evidence.update(tuple(exported(i.get(k)) for k in csv_fields) for i in old['issues'])
        for kind in ('Stage01', 'Stage02', 'SRT'):
            source = new['inputs'].get(kind)
            if source:
                path = Path(source['path'])
                actual = sha(path)
                prior = old['inputs'].get(kind, {})
                if actual != source['sha256'] or actual != prior.get('sha256'):
                    raise ValueError(f'Input hash differs from baseline/current validation: {path}')
                fingerprints[str(path)] = actual
                input_records.append(dict(movie_id=mid, input_kind=kind, path=str(path),
                    baseline_sha256=prior['sha256'], v1_1_sha256=source['sha256'], actual_sha256_before=actual))
    if csv_counts != json_counts or csv_evidence != json_evidence:
        raise ValueError('Baseline issues.csv and movies/*.json disagree')
    output.mkdir(parents=True, exist_ok=True)
    all_rows, all_roots, investigations, patches, movie_rows, new_findings = [], [], [], [], [], []
    unresolved = []
    for row in new_summary['movies']:
        mid = row['movie_id']
        old, current = (read_json(root / 'movies' / f'{mid}.json') for root in (baseline, validation))
        source_review = investigate(current, output)
        investigations.append(source_review)
        movie_patches = patches_for(current)
        patches.extend(movie_patches)
        rows, roots = reconcile(old, current, source_review)
        all_rows.extend(rows)
        all_roots.extend(roots)
        unresolved.extend(r for r in roots if any(d in ('NEEDS_HUMAN_REVIEW', 'UNRESOLVED', 'STAGE02_CORRECTION_PROPOSED', 'RESOLVED_SOURCE_CURATION') for d in r['decisions']))
        movie_rows.append(dict(movie_id=mid, movie_name=row['movie_name'], baseline_status=old['summary']['validation_status'],
            v1_1_status=row['validation_status'], source_integrity=row['source_integrity_status'],
            annotation_consistency=row['annotation_consistency_status'], root_errors=row['root_errors'], root_warnings=row['root_warnings'],
            occurrence_errors=row['errors'], occurrence_warnings=row['warnings'], proposed_patch_count=len(movie_patches)))
        for root in current['root_problems']:
            if root['severity'] not in ('ERROR', 'WARNING'):
                continue
            findings = [i for i in current['issues'] if i['root_problem_id'] == root['root_problem_id']]
            is_new = any(i['rule_code'] == 'FIELD_CLASSIFICATION' for i in findings)
            finding = dict(root, findings=findings, newly_detected_in_v1_1=is_new,
                proposed_patch_count=sum(root['root_problem_id'] in p['root_problem_ids'] for p in movie_patches))
            if is_new:
                finding['decision'] = 'STAGE02_CORRECTION_PROPOSED'
            elif all(i['rule_code'] == 'TEXT_NORMALIZED' for i in findings):
                finding['decision'] = 'ACCEPTED_FORMAT_DIFFERENCE'
            elif any(i['rule_code'] == 'SRT_UNSUPPORTED_FORMAT' for i in findings):
                finding['decision'] = 'UNRESOLVED'
            elif any(p['type'] == 'DERIVED_SRT_CURATION' for p in source_review['proposals']):
                finding['decision'] = 'RESOLVED_SOURCE_CURATION'
            else:
                finding['decision'] = 'NEEDS_HUMAN_REVIEW'
            finding['requires_human_confirmation'] = finding['decision'] != 'ACCEPTED_FORMAT_DIFFERENCE'
            finding['applied'] = False
            new_findings.append(finding)
    assert len(all_rows) == sum(json_counts.values())
    decision_counts = Counter(r['decision'] for r in all_rows)
    audit = dict(scope=SCOPE, limitation=LIMITATION, baseline_root=str(baseline), validation_root=str(validation),
        baseline_counts_on_disk=old_summary['counts'],
        baseline_count_note=f'The required on-disk baseline counts are {old_summary["counts"]}. No baseline files were rewritten.',
        v1_1_counts=new_summary['counts'], reviewed_issue_occurrences=len(all_rows), independent_baseline_roots=len(all_roots),
        duplicate_occurrences=len(all_rows) - len(all_roots), decision_counts={d: decision_counts[d] for d in DECISIONS},
        all_source_files_accessible=all(i['source_accessible'] for i in investigations),
        source_curation_policy='Derivatives and candidate validation only; no promotion or original/Stage02 edits. RESOLVED_SOURCE_CURATION is conditional on approval and applies only to the candidate scenario.',
        proposed_stage02_patch_count=len(patches), verified_input_count=len(input_records), movies=movie_rows)
    dump(output / 'issue_resolution.json', dict(scope=SCOPE, rows=all_rows, roots=all_roots))
    _csv(output / 'issue_resolution.csv', all_rows, ['baseline_issue_id', 'movie_id', 'shot_id', 'srt_indices', 'old_rule_code', 'old_severity', 'field',
          'decision', 'reason', 'requires_source_change', 'requires_human_confirmation', 'candidate_scenario_only', 'root_problem_id', 'duplicate_of', 'independent_problem', 'original_text_evidence'])
    dump(output / 'source_investigations.json', dict(scope=SCOPE, investigations=investigations))
    bundles = file_patch_bundles(patches, validation)
    audit['file_patch_proposal_count'] = len(bundles)
    audit['companion_entry_count'] = sum(len(b['companion_entries']) for b in bundles)
    audit['current_root_decision_counts'] = dict(Counter(r['decision'] for r in new_findings))
    dump(output / 'proposed_stage02_patches.json', dict(scope=SCOPE, applied=False, requires_human_confirmation=True, proposals=patches,
        file_proposals=bundles, policy='Use consolidated file_proposals when reviewing an entire file; individual text proposals are supporting evidence. No patch has been applied.'))
    dump(output / 'current_findings.json', dict(scope=SCOPE, roots=new_findings))
    dump(output / 'unresolved.json', dict(scope=SCOPE, baseline_roots_requiring_action=unresolved,
        current_roots_requiring_action=[r for r in new_findings if r['severity'] == 'ERROR'],
        proposed_source_fixes=[dict(movie_id=i['movie_id'], proposals=i['proposals']) for i in investigations if i['proposals']]))
    if historical:
        historical = historical.expanduser().resolve()
        for name in ('summary.json', 'issues.csv'):
            fingerprints[str(historical / name)] = sha(historical / name)
        audit['historical_first_run_counts'] = read_json(historical / 'summary.json')['counts']
        audit['baseline_count_note'] += f' The separately audited historical first-run counts are {audit["historical_first_run_counts"]}; the primary review uses the on-disk baseline.'
        supplemental_rows, supplemental_roots = [], []
        for row in new_summary['movies']:
            mid = row['movie_id']
            old_path = historical / 'movies' / f'{mid}.json'
            fingerprints[str(old_path)] = sha(old_path)
            current = read_json(validation / 'movies' / f'{mid}.json')
            investigation = next(i for i in investigations if i['movie_id'] == mid)
            historical_movie = read_json(old_path)
            for key in ('Stage01', 'Stage02', 'SRT'):
                if historical_movie['inputs'].get(key, {}).get('sha256') != current['inputs'].get(key, {}).get('sha256'):
                    raise ValueError(f'Historical input fingerprint differs: {mid}: {key}')
            rows, roots = reconcile(historical_movie, current, investigation)
            supplemental_rows.extend(rows)
            supplemental_roots.extend(roots)
        dump(output / 'historical_first_run_resolution.json', dict(scope=SCOPE, baseline_root=str(historical),
            note='Supplemental earlier first-run comparison; the required on-disk baseline remains the primary review.', rows=supplemental_rows, roots=supplemental_roots))
        audit['historical_first_run_issue_occurrences'] = len(supplemental_rows)
    changed = [path for path, fingerprint in fingerprints.items() if sha(Path(path)) != fingerprint]
    if changed:
        raise ValueError(f'Inputs changed during review: {changed}')
    for record in input_records:
        record['actual_sha256_after'] = sha(Path(record['path']))
        record['unchanged'] = record['actual_sha256_before'] == record['actual_sha256_after']
    implementation_paths = [*Path(__file__).resolve().parent.glob('*.py'), *Path(__file__).resolve().parent.glob('*.schema.json')]
    repository = Path(__file__).resolve().parents[3]
    implementation_paths += [p for p in [repository / 'docs/stage02_spec_v1.md', repository / 'docs/prompts/stage02_prompt_v1.txt', repository / 'pyproject.toml', repository / 'uv.lock'] if p.is_file()]
    dump(output / 'verification.json', dict(scope=SCOPE, baseline_csv_json_agree=True, inputs=input_records,
        implementation_and_contract_hashes={str(p): sha(p) for p in implementation_paths},
        baseline_and_validation_artifact_hashes={p: h for p, h in fingerprints.items() if p not in {r['path'] for r in input_records}},
        all_hashes_unchanged=True, numbered_source_indices_invented=False, stage02_patches_applied=False, original_sources_modified=False))
    dump(output / 'review_summary.json', audit)
    _write_markdown(output / 'REVIEW.md', audit, investigations, new_findings)
    return audit


def _write_markdown(path, audit, investigations, findings):
    lines = ['# Stage02 Validator v1.1 review', '', SCOPE + '. ' + LIMITATION, '',
        audit['baseline_count_note'], '', f'v1.1 original-source results: {audit["v1_1_counts"]}.', '',
        f'Reviewed {audit["reviewed_issue_occurrences"]} baseline ERROR/WARNING occurrences, '
        f'{audit["independent_baseline_roots"]} independent roots and {audit["duplicate_occurrences"]} duplicate occurrences.', '',
        '## Movie results', '', '| Movie | Baseline → v1.1 | Source integrity | Annotation consistency | Root errors / warnings | Patch proposals |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in audit['movies']:
        lines.append(f'| {r["movie_id"]} {r["movie_name"]} | {r["baseline_status"]} → {r["v1_1_status"]} | {r["source_integrity"]} | {r["annotation_consistency"]} | {r["root_errors"]} / {r["root_warnings"]} | {r["proposed_patch_count"]} |')
    lines.extend(['', '## Baseline issue dispositions', '', *[f'- {k}: {v}' for k, v in audit['decision_counts'].items()], '',
        'Every occurrence, old rule, source text, affected shot, decision and duplicate link is in issue_resolution.json/.csv.', '',
        '## Source investigations and pending approval', '', audit['source_curation_policy'], ''])
    for item in investigations:
        if not item['source_issues']:
            continue
        lines.append(f'### {item["movie_id"]}')
        lines.extend(['', f'Source: {item.get("source_path")}; SHA-256: {item.get("source_sha256")}.', ''])
        for proposal in item['proposals']:
            if proposal['type'] == 'DERIVED_SRT_CURATION':
                lines.append(f'- Unapproved derivative: `{proposal["derived_path"]}`. Candidate result: {proposal["candidate_validation"]["validation_status"]}; source: {proposal["candidate_validation"]["source_integrity_status"]}. Removed original block numbers: {[b["block_number"] for b in proposal["removed_blocks"]]}. Original numbering retained for all other cues.')
            else:
                lines.append(f'- {proposal["type"]}: {proposal["reason"]}')
        if item.get('webvtt'):
            v = item['webvtt']
            lines.append(f'- {v["timed_block_count"]} timed blocks; {v["unique_signature_count"]} distinct time/text signatures. Diagnostic cross-links do not establish original indices. No renumbered SRT was generated.')
            lines.append(f'- {v["numeric_identifier_count"]} numeric cue identifiers; {v["unique_link_count"]} Stage02 reference indices have a unique text/time candidate, {v["ambiguous_link_count"]} have multiple candidates, {v["unmatched_link_count"]} have no candidate. Even unique content/time links do not establish original numbering.')
        lines.append('')
    new = [r for r in findings if r['newly_detected_in_v1_1']]
    lines.extend(['## Newly detected classification problems', '',
        f'{len(new)} independent field-classification roots were absent from the baseline rules. These remain errors; counts are not minimized to force passing movies.', '',
        f'proposed_stage02_patches.json contains {audit["proposed_stage02_patch_count"]} text proposals consolidated into {audit["file_patch_proposal_count"]} file bundles, with {audit["companion_entry_count"]} companion entries preserving displaced dialogue/lyrics. Exact before/after arrays, source evidence, file hashes and JSON Patch test guards are included. No patch has been applied. Null companion speakers require contextual review.', '',
        '## Verification and artifacts', '',
        f'- verification.json records all {audit["verified_input_count"]} input hashes before/after, baseline/current report fingerprints and CSV/JSON evidence agreement.',
        '- source_investigations.json preserves rejected cue blocks, neighboring timing evidence, candidate-source lineage and complete original index maps.',
        '- current_findings.json and unresolved.json retain current error roots, affected shots and pending source/annotation actions.',
        '- Source candidate scenarios are separate from the 18-movie original-source validation. The baseline, original SRTs and Stage02 JSON remain unchanged.', ''])
    with path.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
