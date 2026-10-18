"""Deterministic source diagnostics; never repair, renumber or discard captions."""
from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import re

from .srt import parse_srt

# Heuristic thresholds are serialized in every report, not hidden repair rules.
TIMELINE_POLICY = dict(major_displacement_seconds=30.0, minor_regression_seconds=2.0,
                       long_cue_seconds=120.0, boundary_tolerance_seconds=1e-6)
ADVERTISING = re.compile(
    r'https?://\S+|www\.\S+|\b[\w-]+\.(?:com|org|net)\b|'
    r'advertise (?:your|here)|contact us|earning money with crypto|'
    r'关注微博|字幕(?:组|下载)', re.I)


def block_evidence(block):
    """Full raw evidence is restricted to the anomalous block, never the film."""
    result = {key: block[key] for key in ('block_number', 'index', 'start', 'end', 'text', 'raw')}
    result['text_empty'] = not block['text'].strip()
    for key in ('text','raw'):
        if len(result[key]) > 4000:
            result[key+'_sha256'] = hashlib.sha256(result[key].encode()).hexdigest()
            result[key+'_characters'] = len(result[key])
            result[key] = result[key][:4000]
            result[key+'_truncated'] = True
    return result


def inspect_subtitles(raw, duration, references, add):
    result = dict(completed=False, actual_format='UNREADABLE', encoding=None, bom=None,
                  newline_counts={}, cue_count=0, numbered_block_count=0,
                  parser='oscardp.stage02.srt.parse_srt (Validator 1.1.1)',
                  index_mapping_created=False, content_modified=False)
    if raw is None:
        return result, None
    bom = next((name for marker, name in [(b'\xff\xfe\x00\x00', 'UTF-32-LE'),
        (b'\x00\x00\xfe\xff', 'UTF-32-BE'), (b'\xff\xfe', 'UTF-16-LE'),
        (b'\xfe\xff', 'UTF-16-BE'), (b'\xef\xbb\xbf', 'UTF-8')] if raw.startswith(marker)), None)
    result['bom'] = bom
    if bom and bom != 'UTF-8':
        result['encoding'] = bom
        add('SRT_ENCODING_UNSUPPORTED', 'BLOCKED', 'SRT',
            'Only UTF-8 / UTF-8 BOM is supported by the shared source parser; no guessed transcoding.', {'bom': bom})
        return result, None
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as error:
        add('SRT_UNDECODABLE', 'BLOCKED', 'SRT', 'Subtitle is not decodable UTF-8.',
            dict(byte_start=error.start, byte_end=error.end, reason=error.reason))
        return result, None
    result['encoding'] = 'UTF-8-BOM' if bom else 'UTF-8'
    crlf = text.count('\r\n')
    result['newline_counts'] = dict(CRLF=crlf, LF=text.count('\n')-crlf, CR=text.count('\r')-crlf)
    if '\x00' in text or '\ufffd' in text:
        add('SRT_ENCODING_CORRUPT', 'BLOCKED', 'SRT', 'NUL or replacement characters suggest damaged decoding.',
            dict(nul_count=text.count('\x00'), replacement_count=text.count('\ufffd')))
    if sum(v > 0 for v in result['newline_counts'].values()) > 1:
        add('SRT_MIXED_NEWLINES', 'WARNING', 'SRT', 'Mixed newline styles require source review.', result['newline_counts'], True)
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    if re.search(r'(?m)^\s*WEBVTT(?:\s|$)', normalized) or re.search(r'(?m)^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->', normalized):
        result['actual_format'] = 'WebVTT'
        result['numeric_identifier_count'] = len(re.findall(r'(?m)^\d+\n\d{2}:\d{2}:\d{2}\.\d{3}\s+-->', normalized))
        result['webvtt_header_count'] = len(re.findall(r'(?m)^WEBVTT(?:\s|$)', normalized))
        add('SRT_UNSUPPORTED_FORMAT', 'BLOCKED', 'SRT',
            'Actual content is WebVTT; original numbered SRT identity is not authenticated. No index mapping is invented.',
            dict(actual_format='WebVTT', numeric_identifier_count=result['numeric_identifier_count']))
        return result, None
    if not re.search(r'(?m)^\s*\d+\s*\n', normalized) and '-->' not in normalized:
        result['actual_format'] = 'UNSUPPORTED'
        add('SRT_UNSUPPORTED_FORMAT', 'BLOCKED', 'SRT', 'No numbered SRT structure detected.',
            dict(content_bytes=len(raw), content_sha256=hashlib.sha256(raw).hexdigest()))
        return result, None
    result['actual_format'] = 'SRT'
    try:
        parsed = parse_srt(text)  # The exact same parser as the post-annotation validator.
    except (ValueError, OverflowError) as error:
        add('SRT_PARSE_FAILED', 'BLOCKED', 'SRT', 'Shared SRT parser could not parse the source.', {'error': str(error)})
        return result, None
    result.update(completed=True, cue_count=len(parsed.cues), numbered_block_count=len(parsed.blocks),
                  presentation_markers=dict(html_tag_count=len(re.findall(r'</?\w+[^>]*>', normalized)),
                                            ass_alignment_count=len(re.findall(r'\{\\an[1-9]\}', normalized))))
    groups = defaultdict(list)
    for block in parsed.blocks:
        groups[block['index']].append(block)
    # Preserve parser diagnostic counts without exporting an entire malformed film.
    result['shared_parser_issue_counts'] = dict(Counter(i['rule_code'] for i in parsed.issues))
    for problem in parsed.issues:
        if problem['rule_code'] == 'SRT_DUPLICATE_INDEX' or problem['message'] == 'Empty subtitle text':
            continue  # More specific, evidence-rich preflight dispositions below.
        block = next((b for b in parsed.blocks if b['raw'] == problem.get('actual')), None)
        code = problem['rule_code']
        if block and block['start'] is not None and block['end'] is not None and block['start'] >= block['end']:
            code = 'SRT_ZERO_DURATION' if block['start'] == block['end'] else 'SRT_REVERSED_TIME'
        add(code, 'BLOCKED', 'SRT', problem['message'],
            dict(block=block_evidence(block)) if block else dict(block_number=problem.get('block'),
                srt_index=problem.get('srt_index'), diagnostic=problem['message']))
    for index, blocks in groups.items():
        if index is not None and len(blocks) > 1:
            add('SRT_DUPLICATE_INDEX', 'BLOCKED', 'SRT', 'Original cue number identifies multiple source blocks.',
                dict(srt_index=index, blocks=[block_evidence(b) for b in blocks]))
    for block in parsed.blocks:
        if block['index'] is not None and not block['text'].strip():
            used = (references or {}).get(block['index'], [])
            severity = 'BLOCKED' if used or not block['time_valid'] else 'WARNING'
            add('SRT_EMPTY_CUE', severity, 'SRT',
                'Empty cue retained. Approval is possible only for a valid-timed, unreferenced empty caption.',
                dict(block=block_evidence(block), existing_stage02_references=used,
                     references_checked=references is not None), severity == 'WARNING')
        if ADVERTISING.search(block['text']):
            add('SRT_SUSPECT_ADVERTISING', 'WARNING', 'SRT',
                'Heuristic website/promotional content; may be genuine film text. Review, never auto-delete.',
                dict(block=block_evidence(block), matched_strings=ADVERTISING.findall(block['text'])), True)
    if not parsed.cues:
        add('SRT_NO_USABLE_CUES', 'BLOCKED', 'SRT', 'No valid timed subtitle cues available.', {})
    return result, parsed


def inspect_timeline(parsed, duration, add):
    result = dict(completed=False, thresholds=TIMELINE_POLICY, correction_applied=False,
                  exact_movie_edition_match='UNVERIFIED', global_offset_seconds=None,
                  limitation='Stage01 provides movie extent, not audio. Offset/edition identity cannot be proved from subtitle boundaries alone.')
    if parsed is None or duration is None:
        return result
    blocks = [b for b in parsed.blocks if b['time_valid'] and b['index'] is not None]
    if not blocks:
        return result
    result.update(completed=True, movie_start=0, movie_end=duration,
                  first_cue_start=min(b['start'] for b in blocks), last_cue_end=max(b['end'] for b in blocks),
                  timed_blocks_checked=len(blocks), overlaps_allowed=True)
    major = TIMELINE_POLICY['major_displacement_seconds']
    for pos, current in enumerate(blocks):
        previous = blocks[pos-1] if pos else None
        following = blocks[pos+1] if pos+1 < len(blocks) else None
        evidence = dict(current=block_evidence(current), previous=block_evidence(previous) if previous else None,
                        following=block_evidence(following) if following else None)
        if previous:
            backward = previous['start']-current['start']
            evidence['start_regression_seconds'] = backward
            evidence['gap_from_previous_end_seconds'] = current['start']-previous['end']
            # Unordered, genuinely overlapping captions are permitted, including nested captions.
            if current['end'] <= previous['start'] and backward >= major:
                add('SRT_TIMELINE_BACKWARD_JUMP', 'BLOCKED', 'SRT',
                    'Large nonoverlapping backward jump in original block order; no timing correction inferred.', evidence)
            elif current['end'] <= previous['start'] and backward >= TIMELINE_POLICY['minor_regression_seconds']:
                add('SRT_TIMELINE_REGRESSION', 'WARNING', 'SRT', 'Nonoverlapping subtitle order regression requires review.', evidence, True)
        if previous and following:
            backward_outlier = previous['start']-current['end'] >= major and following['start']-current['end'] >= major
            forward_outlier = current['start']-previous['end'] >= major and current['start']-following['end'] >= major
            if (backward_outlier or forward_outlier) and abs(following['start']-previous['end']) <= major:
                evidence['neighbor_bridge_seconds'] = following['start']-previous['end']
                evidence['isolated_direction'] = 'backward' if backward_outlier else 'forward'
                add('SRT_TIMELINE_ISOLATED_OUTLIER', 'BLOCKED', 'SRT',
                    'Cue position conflicts severely with both neighbors; possible timestamp digit/hour error. No interpolation.', evidence)
        overrun = current['end']-duration
        if overrun > TIMELINE_POLICY['boundary_tolerance_seconds']:
            severity = 'BLOCKED' if current['start'] >= duration or overrun >= major else 'WARNING'
            add('SRT_OUTSIDE_MOVIE', severity, 'SRT', 'Cue exceeds the Stage01 movie extent.',
                dict(block=block_evidence(current), movie_end=duration, overrun_seconds=overrun), severity == 'WARNING')
        if current['end']-current['start'] > TIMELINE_POLICY['long_cue_seconds']:
            add('SRT_UNUSUALLY_LONG_CUE', 'WARNING', 'SRT', 'Unusually long cue duration; verify title card / source timing.',
                dict(block=block_evidence(current), duration_seconds=current['end']-current['start']), True)
    if duration > 300 and len(blocks) >= 3 and (result['first_cue_start'] > .8*duration or result['last_cue_end'] < .5*duration or result['first_cue_start'] >= duration):
        add('SRT_POSSIBLE_OFFSET_OR_INCOMPLETE', 'WARNING', 'SRT',
            'Subtitle extent may indicate a source offset or incomplete track; boundaries alone do not establish an offset.',
            {k: result[k] for k in ['movie_start','movie_end','first_cue_start','last_cue_end','timed_blocks_checked']}, True)
    return result


def inspect_pdf(raw, add):
    from pypdf import PdfReader, __version__
    result = dict(completed=False, parser='pypdf', parser_version=__version__, encrypted=None,
                  page_count=None, page_text_stats=[], text_characters=0, alphanumeric_characters=0,
                  usable_text=False, minimum_alphanumeric_characters=200,
                  minimum_context_page_characters=100, screenplay_movie_match='UNVERIFIED', ocr_performed=False)
    if raw is None:
        return result
    if not raw.startswith(b'%PDF-'):
        add('SCREENPLAY_NOT_PDF', 'BLOCKED', 'Screenplay', 'Input content does not begin with a PDF header.', {})
        return result
    try:
        reader = PdfReader(BytesIO(raw), strict=True)
        result['encrypted'] = reader.is_encrypted
        if reader.is_encrypted:
            add('SCREENPLAY_ENCRYPTED', 'BLOCKED', 'Screenplay', 'Encrypted PDF is not accepted; no password or decryption attempted.', {})
            return result
        result['page_count'] = len(reader.pages)
        failed = []
        for number, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text() or ''
                count = sum(c.isalnum() for c in text)
                result['page_text_stats'].append(dict(page=number, characters=len(text), alphanumeric_characters=count,
                                                       extracted_text_sha256=hashlib.sha256(text.encode()).hexdigest()))
                result['text_characters'] += len(text)
                result['alphanumeric_characters'] += count
            except Exception as error:
                failed.append(dict(page=number, error=f'{type(error).__name__}: {error}'))
        if failed:
            add('SCREENPLAY_EXTRACTION_FAILED', 'BLOCKED', 'Screenplay', 'Text extraction failed on one or more pages.', {'pages': failed})
            return result
        result['completed'] = True
        result['usable_text'] = result['alphanumeric_characters'] >= 200 and any(p['alphanumeric_characters'] >= 100 for p in result['page_text_stats'])
        if not result['usable_text']:
            add('SCREENPLAY_TEXT_UNUSABLE', 'BLOCKED', 'Screenplay',
                'No sufficiently usable text layer; scanned/empty/sparse PDF suspected. No OCR performed.',
                dict(page_count=result['page_count'], alphanumeric_characters=result['alphanumeric_characters']))
        elif sum(p['alphanumeric_characters'] < 20 for p in result['page_text_stats']) > result['page_count']/2:
            add('SCREENPLAY_SPARSE_TEXT', 'WARNING', 'Screenplay', 'Most pages have little extractable text; verify continuous context.',
                dict(page_count=result['page_count'], sparse_pages=[p['page'] for p in result['page_text_stats'] if p['alphanumeric_characters'] < 20]), True)
    except Exception as error:
        add('SCREENPLAY_INVALID_PDF', 'BLOCKED', 'Screenplay', 'PDF parsing failed.', {'error': f'{type(error).__name__}: {error}'})
    return result
