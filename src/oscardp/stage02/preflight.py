"""Read-only pre-annotation quality gate, independent of Stage02 generation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

from .curation import references
from .preflight_policy import apply_approvals, compare_baselines, write_preflight_reports
from .preflight_sources import inspect_pdf, inspect_subtitles, inspect_timeline
from .validator import MovieValidation, SCHEMA_BYTES, SCHEMA_VALIDATOR, VERSION as VALIDATOR_VERSION, read_json

VERSION = '1.0'
MOVIE_ID = re.compile(r'tt[0-9]{7,10}')


def fingerprint(path):
    raw = path.read_bytes()
    stat = path.stat()
    return raw, dict(path=str(path), bytes=len(raw), mtime_ns=stat.st_mtime_ns,
                     mtime_utc=datetime.fromtimestamp(stat.st_mtime,timezone.utc).isoformat(),
                     sha256=hashlib.sha256(raw).hexdigest())


def stage01_quality(path, raw, add):
    result = dict(completed=False, shot_count=0, duration_seconds=None, total_frames=None,
                  shared_rules='MovieValidation.shots / MovieValidation.times, Validator 1.1.1')
    if raw is None:
        return result, None, None
    try:
        one = read_json(path, raw=raw)
    except (ValueError, UnicodeError) as error:
        add('STAGE01_JSON_INVALID', 'BLOCKED', 'Stage01', str(error), {})
        return result, None, None
    if not isinstance(one, dict):
        add('STAGE01_ROOT_TYPE', 'BLOCKED', 'Stage01', 'Stage01 root must be an object.', {})
        return result, None, None
    mid = one.get('movie_id')
    if not isinstance(mid,str) or not MOVIE_ID.fullmatch(mid):
        add('STAGE01_MOVIE_ID', 'BLOCKED', 'Stage01', 'Invalid canonical IMDb ID.', {'actual': mid})
        mid = None
    if mid and (path.name != f'Stage01_{mid}.json' or (MOVIE_ID.fullmatch(path.parent.name) and path.parent.name != mid)):
        add('STAGE01_FILE_IDENTITY', 'BLOCKED', 'Stage01', 'Stage01 filename or IMDb directory differs from movie_id.',
            dict(movie_id=mid, filename=path.name, parent=path.parent.name))
    if set(one) != {'movie_id','movie_name','shots'}:
        add('STAGE01_ROOT', 'BLOCKED', 'Stage01', 'Stage01 root differs from the fixed upstream contract.', {'keys': sorted(one)})
    name = one.get('movie_name')
    if 'movie_name' not in one or (name is not None and (not isinstance(name,str) or not name.strip())):
        add('STAGE01_MOVIE_NAME', 'BLOCKED', 'Stage01', 'Stage01 movie_name missing or invalid.', {'actual': name})
    validator = MovieValidation(Path(mid or 'unknown'),1e-6)
    try:
        shots = validator.shots(one,'Stage01')
        for pos, shot in enumerate(shots):
            if not isinstance(shot,dict):
                validator.issue('stage01_contract','STAGE01_STRUCTURE','Stage01 shot must be an object.',actual={'position':pos,'value':shot})
        validator.times(shots,[])
    except (ValueError, TypeError, OverflowError) as error:
        add('STAGE01_CONTRACT_UNCHECKABLE', 'BLOCKED', 'Stage01', 'Malformed Stage01 prevents complete contract checks.', {'error':str(error)})
        return result, one, mid
    for issue in validator.issues:
        add(issue['rule_code'], 'BLOCKED', 'Stage01', issue['message'],
            dict(shot_id=issue['shot_id'], expected=issue['expected'], actual=issue['actual']))
    result.update(completed=True,shot_count=len(shots))
    if shots and not validator.issues:
        result.update(duration_seconds=shots[-1]['temporal']['end'],total_frames=shots[-1]['temporal']['end_frame'])
    return result, one, mid


def existing_compatibility(one, two, parsed, movie_id, add):
    result = dict(requested=True,completed=False,stale_reference_count=0,modifications_applied=False,
                  validator_version=VALIDATOR_VERSION)
    if not one or not two or parsed is None or not parsed.cues:
        return result
    v = MovieValidation(Path(movie_id or 'unknown'),1e-6)
    v.schema(two)
    if v.issues:
        for issue in v.issues:
            add('STAGE02_'+issue['rule_code'],'BLOCKED','Stage02',issue['message'],
                dict(shot_id=issue['shot_id'],actual=issue['actual']))
        return result
    if two['movie_id'] != movie_id or (one.get('movie_name') is not None and one['movie_name'] != two['movie_name']):
        add('STAGE02_IDENTITY_MISMATCH','BLOCKED','Stage02','Stage02 movie identity differs from Stage01.',
            dict(expected_movie_id=movie_id,actual_movie_id=two['movie_id']))
    v.coverage(one['shots'],two['shots'])
    v.times(one['shots'],two['shots'])
    v.authoritative_times = {s['shot_id']:s['temporal'] for s in one['shots']}
    blocked = {b['index'] for b in parsed.blocks if not b['time_valid'] or not b['text'].strip()}
    v.sources(two['shots'],parsed.cues,blocked)
    result.update(completed=not v.incomplete,exact_match_count=v.stats['exact_match_count'],
                  normalized_match_count=v.stats['normalized_match_count'],mismatch_count=v.stats['mismatch_count'],
                  incomplete_checks=sorted(v.incomplete))
    for issue in v.issues:
        if issue['severity']=='INFO':
            continue
        evidence = dict(shot_id=issue['shot_id'],srt_index=issue['srt_index'],field=issue['field'],entry=issue['entry'],
                        expected=issue['expected'],actual=issue['actual'],validator_rule=issue['rule_code'])
        code = 'STAGE02_'+issue['rule_code']
        if issue['rule_code']=='SRT_NO_OVERLAP':
            code='STAGE02_STALE_CUE_REFERENCE'
            result['stale_reference_count'] += 1
            cue=issue['actual']
            evidence['overlapping_stage01_candidates']=[s['shot_id'] for s in one['shots']
                if min(s['temporal']['end'],cue['end'])>max(s['temporal']['start'],cue['start'])]
            evidence['candidate_assignment_applied']=False
        severity='BLOCKED' if issue['severity']=='ERROR' else 'WARNING'
        add(code,severity,'Stage02',issue['message'],evidence,severity=='WARNING')
    return result


def run_preflight(*,stage01:Path,srt:Path,screenplay:Path,report_root:Path,stage02:Path|None=None,
                  approved_exceptions:Path|None=None,baseline_manifests=()):
    paths={kind:p.expanduser().resolve() for kind,p in [('Stage01',stage01),('SRT',srt),('Screenplay',screenplay)]}
    if stage02 is not None:
        paths['Stage02']=stage02.expanduser().resolve()
    output=report_root.expanduser()
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise FileExistsError('Preflight report directory must be fresh/empty; no silent overwrite')
    output=output.resolve()
    if any(output.is_relative_to(p.parent) for p in paths.values()):
        raise ValueError('Report directory must be outside the input directories')
    baselines=[p.expanduser().resolve() for p in baseline_manifests]
    approval_path=approved_exceptions.expanduser().resolve() if approved_exceptions else None
    auxiliary_paths=[*baselines,*([approval_path] if approval_path else [])]
    if any(p.is_relative_to(output) for p in auxiliary_paths):
        raise ValueError('Reports must not contain approval/baseline inputs')
    inputs,raws,issues={}, {},[]
    def add(code,severity,kind,message,evidence,requires_approval=False):
        identity=json.dumps(dict(version=VERSION,rule=code,input=kind,
            input_sha256=inputs.get(kind,{}).get('sha256'),evidence=evidence),sort_keys=True,ensure_ascii=False,allow_nan=False)
        issue_id='preflight_'+hashlib.sha256(identity.encode()).hexdigest()[:24]
        record=dict(issue_id=issue_id,rule_code=code,severity=severity,input=kind,
            requires_approval=requires_approval,message=message,evidence=evidence,approval={'status':'NOT_APPROVED' if requires_approval else 'NOT_APPLICABLE'})
        if not any(i['issue_id']==issue_id for i in issues):
            issues.append(record)
    for kind,path in paths.items():
        try:
            stat=path.stat()
            raw,meta=fingerprint(path)
            inputs[kind]=meta
            raws[kind]=raw
            if (stat.st_mtime_ns,stat.st_size)!=(meta['mtime_ns'],meta['bytes']):
                add('INPUT_CHANGED_DURING_READ','BLOCKED',kind,'Input changed during provenance capture.',{})
        except OSError as error:
            inputs[kind]=dict(path=str(path),bytes=None,mtime_ns=None,mtime_utc=None,sha256=None)
            raws[kind]=None
            add('INPUT_UNREADABLE','BLOCKED',kind,'Input cannot be read.',{'error':str(error)})
    quality,one,movie_id=stage01_quality(paths['Stage01'],raws['Stage01'],add)
    for kind,path in paths.items():
        if kind=='Stage01':
            continue
        ids=MOVIE_ID.findall(path.name)
        if movie_id and ids and any(value!=movie_id for value in ids):
            add('INPUT_MOVIE_IDENTITY','BLOCKED',kind,'Filename identifies a different movie.',{'filename':path.name,'expected':movie_id,'found':ids})
    two=None
    stage02_valid=False
    if stage02 is not None and raws['Stage02'] is not None:
        try:
            two=read_json(paths['Stage02'],raw=raws['Stage02'])
            errors=list(SCHEMA_VALIDATOR.iter_errors(two))
            stage02_valid=not errors
            for error in errors:
                add('STAGE02_SCHEMA_INVALID','BLOCKED','Stage02','Existing Stage02 violates fixed Schema v1.0.',
                    dict(pointer='/'+ '/'.join(map(str,error.absolute_path)),message=error.message))
        except (ValueError,UnicodeError) as error:
            add('STAGE02_JSON_INVALID','BLOCKED','Stage02',str(error),{})
    refs=references(two) if stage02_valid else None
    subtitle,parsed=inspect_subtitles(raws['SRT'],quality['duration_seconds'],refs,add)
    timeline=inspect_timeline(parsed,quality['duration_seconds'],add)
    pdf=inspect_pdf(raws['Screenplay'],add)
    compat=existing_compatibility(one,two,parsed,movie_id,add) if stage02 is not None and stage02_valid and quality['duration_seconds'] is not None else dict(requested=stage02 is not None,completed=False)
    auxiliary=[]
    for p in auxiliary_paths:
        _,meta=fingerprint(p)
        auxiliary.append(meta)
    versions=compare_baselines(baselines,movie_id,inputs,add)
    stability=True
    for kind,path in paths.items():
        if raws[kind] is None:
            stability=False
            continue
        try:
            _,after=fingerprint(path)
            if after['sha256']!=inputs[kind]['sha256'] or after['mtime_ns']!=inputs[kind]['mtime_ns']:
                stability=False
                add('INPUT_CHANGED_DURING_PREFLIGHT','BLOCKED',kind,'Input changed while checks ran; rerun against stable files.',{'after':after})
        except OSError as error:
            stability=False
            add('INPUT_CHANGED_DURING_PREFLIGHT','BLOCKED',kind,str(error),{})
    approvals=apply_approvals(approval_path,movie_id,inputs,issues,add)
    for prior in auxiliary:
        _,after=fingerprint(Path(prior['path']))
        if prior!=after:
            add('REVIEW_INPUT_CHANGED','BLOCKED','approved_exceptions','Baseline / approval changed during validation.',{'path':prior['path']})
    checks=dict(input_provenance=all(v['sha256'] for v in inputs.values()),stage01_contract=quality['completed'],
                srt_format_and_integrity=subtitle['completed'],subtitle_timeline=timeline['completed'],
                screenplay_readability=pdf['completed'],input_stability=stability)
    checks={k:bool(v) for k,v in checks.items()}
    if stage02 is not None:
        checks['existing_stage02_compatibility']=compat['completed']
    for check,complete in checks.items():
        if not complete:
            add('CHECK_INCOMPLETE','BLOCKED','preflight','A required check could not complete.',{'check':check})
    blocked=any(i['severity']=='BLOCKED' for i in issues)
    status='BLOCKED' if blocked else 'WARNING' if any(i['severity']=='WARNING' for i in issues) else 'PASS'
    pending=[i['issue_id'] for i in issues if i['requires_approval'] and i['approval']['status']!='APPROVED']
    allowed=not blocked and not pending and all(checks.values())
    implementation={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path(__file__).with_name('__main__.py'),Path(__file__).with_name('preflight_policy.py'),Path(__file__).with_name('preflight_sources.py'),
         Path(__file__).with_name('validator.py'),Path(__file__).with_name('srt.py'),Path(__file__).with_name('text.py'),Path(__file__).with_name('curation.py')]}
    report=dict(preflight_version=VERSION,validator_version=VALIDATOR_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),movie_id=movie_id,inputs=inputs,review_inputs=auxiliary,
        provenance_hash_algorithm='SHA-256',python_version=sys.version.split()[0],implementation_sha256=implementation,
        stage02_schema_version='1.0',stage02_schema_sha256=hashlib.sha256(SCHEMA_BYTES).hexdigest(),
        stage01_quality=quality,srt_format_and_integrity=subtitle,subtitle_timeline=timeline,screenplay_readability=pdf,
        source_version_comparison=versions,existing_stage02_compatibility=compat,approvals=approvals,
        issues=issues,executed_issue_rules=sorted({i['rule_code'] for i in issues}),check_completeness=checks,
        executed_rules=dict(stage01=['strict_json','movie_identity','root_shape','ordered_contiguous_shot_ids','finite_contiguous_seconds','zero_based_contiguous_exclusive_frames'],
            subtitles=['actual_content_format','encoding_bom_newlines','shared_srt_parser','unique_positive_original_indices','empty_vs_nonempty_invalid_duration','promotional_content_heuristic'],
            timeline=['nonoverlapping_regressions','isolated_neighbor_outliers','movie_extent_conflicts','long_cues','possible_offset_or_incomplete_extent'],
            screenplay=['pdf_signature','encryption','page_count','all_page_text_extraction','usable_text_quantity'],
            gate=['mandatory_check_completeness','immutable_input_recheck','per_issue_all_input_sha256_approval']),
        pending_review_issue_ids=pending,preflight_status=status,llm_annotation_allowed=allowed,
        exit_code=0 if allowed else 1,
        limitations=['Movie edition and screenplay semantic match are UNVERIFIED; no SCRIPT_MATCH_PASS.',
                    'No subtitle repairs, OCR, LLM calls, Stage02 generation, dataset release or activation.',
                    'Warning approval binds a precise issue and the exact hash set of every selected input.'])
    write_preflight_reports(output,report)
    return report
