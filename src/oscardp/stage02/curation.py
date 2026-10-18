"""Evidence-only source investigations and unapproved derivative proposals.

Never writes source data. Derivatives are review artifacts, not promoted inputs.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from .srt import Cue, parse_srt
from .text import compare_entry
from .validator import read_json, validate_movie

AD = re.compile(r'Advertise your product or brand here|FilthyRichFutures\.com', re.I)
VTT_CUE = re.compile(r'(?m)^(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n(.*?)(?=\n\s*\n|\Z)', re.S)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def references(stage: dict) -> dict[int, list[dict]]:
    refs = defaultdict(list)
    for shot in stage['shots']:
        for field in ('dialogue', 'sound_cues', 'song_lyrics'):
            for entry_no, entry in enumerate(shot.get(field, [])):
                indices = entry.get('srt_indices', [entry.get('srt_index')])
                for index in indices:
                    refs[index].append(dict(shot_id=shot['shot_id'], field=field, entry=entry_no,
                                            text=entry['text'], indices=indices))
    return dict(refs)


def webvtt_investigation(text: str, refs: dict, stage01: dict) -> dict:
    """Content/time cross-links are diagnostics, never proof of original numbering."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    signatures = {}
    numeric_identifiers = []
    def seconds(value):
        h, m, s = value.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    for number, match in enumerate(VTT_CUE.finditer(text), 1):
        previous_line = text[:match.start()].removesuffix('\n').rsplit('\n', 1)[-1].strip()
        if previous_line.isdigit():
            numeric_identifiers.append(dict(timed_block=number, identifier=previous_line))
        start, end, body = match.groups()
        key = (start, end, body)
        if key not in signatures:
            signatures[key] = dict(signature_id=f'vtt_signature_{len(signatures) + 1:06d}', start=seconds(start),
                                   end=seconds(end), text=body, block_occurrences=[])
        signatures[key]['block_occurrences'].append(number)
    timeline = {s['shot_id']: s['temporal'] for s in stage01['shots']}
    # Index by exact normalized text first; no fuzzy search or cue renumbering.
    from .srt import normalized_text
    by_text = defaultdict(list)
    for item in signatures.values():
        by_text[normalized_text(item['text'])[0]].append(item)
    links = []
    for index, entries in sorted(refs.items()):
        candidate_sets = []
        unsupported = False
        for entry in entries:
            if len(entry['indices']) != 1:
                unsupported = True
                continue
            candidates = set()
            for item in by_text.get(normalized_text(entry['text'])[0], []):
                temporal = timeline[entry['shot_id']]
                if min(item['end'], temporal['end']) > max(item['start'], temporal['start']):
                    candidates.add(item['signature_id'])
            candidate_sets.append(candidates)
        candidates = set.intersection(*candidate_sets) if candidate_sets else set()
        links.append(dict(stage02_reference_index=index, candidate_signature_ids=sorted(candidates),
                          candidate_count=len(candidates), affected_shots=sorted({e['shot_id'] for e in entries}),
                          merged_entries_not_mapped=unsupported, proves_original_index=False))
    return dict(format='WebVTT', header_count=text.count('WEBVTT'), numeric_identifier_count=len(numeric_identifiers),
                numeric_identifiers=numeric_identifiers, timed_block_count=sum(len(v['block_occurrences']) for v in signatures.values()),
                unique_signature_count=len(signatures), duplicate_timed_block_count=sum(len(v['block_occurrences']) - 1 for v in signatures.values()),
                original_numbered_source_found=False, original_indices_recoverable=False,
                reason='No cue identifiers survive in the source. Stage02 numbering and content/time links cannot prove original SRT index provenance; sequential numbering is not authorized.',
                candidate_links=links, signatures=list(signatures.values()),
                unique_link_count=sum(link['candidate_count'] == 1 for link in links),
                ambiguous_link_count=sum(link['candidate_count'] > 1 for link in links),
                unmatched_link_count=sum(link['candidate_count'] == 0 for link in links),
                mapping_scope='Exact normalized whole-cue text plus intersection of every referenced shot overlap; mixed/merged excerpts may have no candidates. Diagnostic only.')


def investigate(report: dict, root: Path) -> dict:
    movie_id = report['summary']['movie_id']
    source = report['inputs'].get('SRT')
    result = dict(movie_id=movie_id, source_accessible=False, proposals=[], source_issues=[], references={}, approval_status='NOT_APPROVED')
    if not source:
        result['reason'] = 'No confirmed SRT path in report'
        return result
    path = Path(source['path'])
    try:
        raw = path.read_bytes()
        text = raw.decode('utf-8-sig')
    except (OSError, UnicodeError) as exc:
        result['reason'] = str(exc)
        return result
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != source['sha256']:
        raise ValueError(f'Source changed since validation: {path}')
    result.update(source_accessible=True, source_path=str(path), source_sha256=actual_hash,
                  alternate_subtitle_paths=sorted(str(p) for p in path.parent.rglob('*') if p.is_file() and p.suffix.lower() in ('.srt', '.vtt')),
                  alternate_search_scope=str(path.parent))
    stage = read_json(Path(report['inputs']['Stage02']['path']))
    stage01 = read_json(Path(report['inputs']['Stage01']['path']))
    refs = references(stage)
    parsed = parse_srt(text)
    result['source_issues'] = parsed.issues
    relevant = {i.get('srt_index') for i in parsed.issues if i.get('srt_index') is not None}
    result['references'] = {str(i): refs.get(i, []) for i in sorted(relevant)}
    if any(i['rule_code'] == 'SRT_UNSUPPORTED_FORMAT' for i in parsed.issues):
        result['webvtt'] = webvtt_investigation(text, refs, stage01)
        result['proposals'].append(dict(type='RECOVER_AUTHORITATIVE_NUMBERED_SOURCE', requires_human_confirmation=True,
                                        status='UNRESOLVED', proposed_after=None, reason=result['webvtt']['reason']))
        return result
    removals = {}
    grouped = defaultdict(list)
    for block in parsed.blocks:
        grouped[block['index']].append(block)
    for block in parsed.blocks:
        index = block['index']
        if index is None:
            continue
        before = dict(block)
        if not block['time_valid']:
            result['proposals'].append(dict(type='RESTORE_TIMESTAMPS_FROM_AUTHORITATIVE_SOURCE', index=index, before=before,
                after=None, requires_human_confirmation=True, status='NEEDS_HUMAN_REVIEW',
                source_references=refs.get(index, []), reason='Nonempty zero/invalid-duration caption; no timing authority. Do not delete meaningful text or infer offsets.',
                neighboring_blocks=[b for b in parsed.blocks if b['block_number'] in (block['block_number'] - 1, block['block_number'] + 1)]))
        elif not block['text'].strip() and not refs.get(index):
            removals[block['block_number']] = 'Unreferenced empty cue; no linguistic content removed'
        elif len(grouped[index]) > 1 and AD.search(block['text']):
            other = [b for b in grouped[index] if b['block_number'] != block['block_number'] and b['time_valid']]
            # Curation is proposed only when every reference points to a retained caption.
            retained_supported = all(any(compare_entry(e['text'], [Cue(index, b['start'], b['end'], b['text'])], e['field'])['status'] != 'mismatch'
                                         for b in other) for e in refs.get(index, []))
            if retained_supported:
                removals[block['block_number']] = 'Explicit advertising text duplicates a retained film-caption number; every Stage02 reference matches retained text'
    if not removals:
        return result
    destination = root / 'proposed_sources' / f'{movie_id}.srt'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n\n'.join(b['raw'].rstrip('\n') for b in parsed.blocks if b['block_number'] not in removals) + '\n')
    mapping = [dict(original_block=b['block_number'], original_index=b['index'],
                    derived_index=b['index'] if b['block_number'] not in removals else None,
                    disposition='REMOVAL_PROPOSED' if b['block_number'] in removals else 'RETAINED',
                    reason=removals.get(b['block_number']), original_block_sha256=hashlib.sha256(b['raw'].encode()).hexdigest()) for b in parsed.blocks]
    proposal = dict(type='DERIVED_SRT_CURATION', status='PROPOSED_ONLY', requires_human_confirmation=True,
                    original_path=str(path), original_sha256=actual_hash, derived_path=str(destination.resolve()),
                    derived_sha256=sha(destination), encoding='UTF-8 LF; source cue numbers and retained raw block text preserved',
                    removed_blocks=[b for b in parsed.blocks if b['block_number'] in removals],
                    reasons=removals, original_index_mapping=mapping, applied_to_dataset=False)
    candidate = validate_movie(Path(report['inputs']['Stage02']['path']).parent, [], srt_override=destination)
    proposal['candidate_validation'] = candidate['summary']
    proposal['candidate_report'] = str((root / 'source_scenarios' / f'{movie_id}.json').resolve())
    dump(root / 'source_scenarios' / f'{movie_id}.json', candidate)
    result['proposals'].append(proposal)
    assert sha(path) == actual_hash
    return result
