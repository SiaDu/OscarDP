"""Bounded mixed-caption extraction; never accept arbitrary substring matches.

Offsets refer to the rendered source (known presentation tags removed), not raw HTML.
Only explicitly delimited music/sound spans split a cue into typed segments.
"""
from __future__ import annotations

import html
import re
import unicodedata

from .srt import ALIGNMENT, BREAK, MARKUP, Cue, compare_text, normalized_text

MUSIC = re.compile(r'[♪♫][^♪♫]+[♪♫]')
BRACKET = re.compile(r'\([^()]+\)|\[[^\[\]]+\]')
SOUND = re.compile(
    r'\b(?:music|playing|applause|silence|inaudible|indistinct\w*|laughter|throat|'
    r'gas\w*|sigh\w*|laugh\w*|chuckl\w*|cry\w*|cries|sob\w*|groan\w*|grunt\w*|'
    r'sniff\w*|snort\w*|pant\w*|breath\w*|inhale\w*|exhale\w*|scream\w*|'
    r'yell\w*|shout\w*|cheer\w*|clap\w*|whistl\w*|beep\w*|buzz\w*|ring\w*|'
    r'slam\w*|engine\w*|rev\w*|whirr\w*|pour\w*|clatter\w*|strum\w*|burbl\w*|'
    r'trembl\w*|cough\w*|chatter\w*|click\w*|knock\w*|squeak\w*|rustl\w*|bang\w*)\b', re.I)
LABEL = re.compile(r"^\[([A-Za-z][A-Za-z0-9 .'-]*)\]\s+(?=\S)")
TURN = re.compile(r'^\s*-(?=\s|[A-Za-z"\'(\[♪♫])\s*')
CAPTION_NAMES = r"[A-Z][A-Z0-9 .'-]*(?:(?:and |or |& |/ )[A-Z][A-Z0-9 .'-]*)*"
SDH_LABEL = re.compile(r'^(MAN(?: \d+)?|WOMAN(?: \d+)?):\s*(?=\S)')


def rendered(text: str) -> str:
    return unicodedata.normalize('NFC', html.unescape(ALIGNMENT.sub('', MARKUP.sub('', BREAK.sub('\n', text)))))


def _dialogue_body(text: str) -> tuple[str, list[str]]:
    """Remove evidenced turn layout and explicit bracketed or SDH speaker labels."""
    rules = []
    new = TURN.sub('', text.strip(), count=1)
    if new != text.strip():
        rules.append('source_turn_delimiter')
    label = LABEL.match(new)
    if label and not SOUND.search(label.group(1)):
        new = new[label.end():]
        rules.append('explicit_bracketed_speaker_label')
    label = SDH_LABEL.match(new)
    if label:
        new = new[label.end():]
        rules.append('explicit_sdh_speaker_label:' + label.group(1))
    return new.strip(), rules


def segments(cue: Cue) -> dict:
    plain = rendered(cue.text)
    typed = []
    for match in BRACKET.finditer(plain):
        # Parentheses are complete SDH annotations; bracketed bare names before
        # a turn/lyric are speaker metadata, never invented spoken text.
        following = plain[match.end():].lstrip()
        kind = 'sound_cues' if match.group().startswith('(') or SOUND.search(match.group()[1:-1]) else 'metadata'
        if kind == 'metadata' and not following:
            kind = 'sound_cues'  # A standalone bracket annotation, e.g. [door opens].
        typed.append(dict(kind=kind, start=match.start(), end=match.end(), text=match.group(),
                          rules=['complete_sound_annotation' if kind == 'sound_cues' else 'explicit_bracketed_speaker_label']))
    for match in MUSIC.finditer(plain):
        if any(match.start() < s['end'] and match.end() > s['start'] for s in typed):
            continue
        kind = 'song_lyrics' if match.group()[1:-1].strip() else 'sound_cues'
        typed.append(dict(kind=kind, start=match.start(), end=match.end(), text=match.group(),
                          rules=['paired_music_delimiters' if kind == 'song_lyrics' else 'instrumental_music_marker']))
    typed.sort(key=lambda s: s['start'])
    all_segments = []
    cursor = 0
    for part in [*typed, dict(start=len(plain), end=len(plain), kind=None)]:
        if cursor < part['start']:
            residue = plain[cursor:part['start']]
            # A residual turn marker at the end precedes the next typed span.
            residue = re.sub(r'(?:^|\n)\s*-\s*$', '', residue)
            if part['kind'] == 'song_lyrics':
                residue = re.sub(r'[ \t]+-[ \t]*$', '', residue)
            body, rules = _dialogue_body(residue)
            # Separators following a sound annotation (e.g. "(laughing):") are layout.
            if body.startswith(':') and any(s['end'] == cursor and s['kind'] in ('sound_cues', 'metadata') for s in typed):
                body = body[1:].strip()
                rules.append('annotation_separator')
            standalone_label = re.fullmatch(CAPTION_NAMES + ':', body)
            label_before_annotation = (re.fullmatch(CAPTION_NAMES, body)
                and part['kind'] == 'sound_cues' and plain[part['end']:].lstrip().startswith(':'))
            if standalone_label or label_before_annotation:
                all_segments.append(dict(kind='metadata', start=cursor, end=part['start'], text=body,
                                         rules=['explicit_caption_speaker_label']))
                body = ''
            if body and not re.fullmatch(r'[-:\s]+', body):
                all_segments.append(dict(kind='dialogue', start=cursor, end=part['start'], text=body, rules=rules))
        if part['kind']:
            all_segments.append(part)
        cursor = part['end']
    return dict(srt_index=cue.index, source_text=cue.text, rendered_source=plain,
                segments=all_segments, has_explicit_segments=bool(typed))


def compare_entry(actual: str, cues: list[Cue], field: str, *, decisions=()) -> dict:
    evidence = compare_text(actual, cues)
    evidence.update(match_rule='whole_cue', source_segments=[segments(c) for c in cues],
                    classification_error=None, extraction=None)
    parsed = evidence['source_segments']
    evidence['human_decision_ids'] = []
    for p in parsed:
        for decision in decisions:
            if decision['srt_index'] == p['srt_index'] and decision['source_text'] == p['source_text']:
                for part in p['segments']:
                    if part['text'] == decision['text']:
                        part['kind'] = decision['classification']
                        part['rules'] = [*part['rules'], 'user_confirmed_audiovisual_review']
                        p['has_explicit_segments'] = True
                        evidence['human_decision_ids'].append(decision['decision_id'])
    # Known lexical delimiters establish classification; plain prose/italics alone do not.
    kinds = {s['kind'] for p in parsed if p['has_explicit_segments'] for s in p['segments'] if s['kind'] != 'metadata'}
    if evidence['status'] != 'mismatch':
        forbidden = (field == 'song_lyrics' and 'dialogue' in kinds) or (
            field == 'sound_cues' and bool(kinds - {'sound_cues'})) or (
            field == 'dialogue' and 'song_lyrics' in kinds)
        if forbidden:
            evidence['classification_error'] = 'Full entry contains explicitly delimited content belonging to another field'
        return evidence
    # Single complete typed segment, or the complete field projection of every cue.
    # No word slicing, arbitrary line selection, case/punctuation changes or fuzzy match.
    candidates = []
    if len(cues) == 1 and parsed[0]['has_explicit_segments']:
        for part in parsed[0]['segments']:
            if part['kind'] == field:
                candidates.append(dict(text=part['text'], parts=[part], rule='complete_typed_segment'))
    projection = []
    for p in parsed:
        parts = [s for s in p['segments'] if s['kind'] == field] if p['has_explicit_segments'] else []
        if not parts:
            projection = []
            break
        projection.extend(parts)
    if projection:
        candidates.append(dict(text='\n'.join(s['text'] for s in projection), parts=projection, rule='complete_field_projection'))
        # Older annotations may retain an explicit source label. Both complete
        # presentations remain traceable; never treat the label as an identity.
        labeled = []
        for part in projection:
            labels = [r.split(':', 1)[1] for r in part['rules'] if r.startswith('explicit_sdh_speaker_label:')]
            labeled.append((labels[0] + ': ' if labels else '') + part['text'])
        if labeled != [s['text'] for s in projection]:
            candidates.append(dict(text='\n'.join(labeled), parts=projection, rule='complete_field_projection'))
    normalized_actual = normalized_text(actual)[0]
    for candidate in candidates:
        expected, reasons = normalized_text(candidate['text'])
        if expected == normalized_actual:
            if not any('user_confirmed_audiovisual_review' in p['rules'] for p in candidate['parts']):
                evidence['human_decision_ids'] = []
            evidence.update(status='exact' if actual == candidate['text'] else 'normalized',
                            match_rule=candidate['rule'], extraction=candidate,
                            normalized_expected=expected, reasons=sorted(set(reasons + normalized_text(actual)[1])))
            return evidence
    # A complete segment from a different explicit field is a classification error,
    # not a paraphrase or an invented source text.
    for p in parsed:
        if not p['has_explicit_segments']:
            continue
        for part in p['segments']:
            if part['kind'] != field and normalized_text(part['text'])[0] == normalized_actual:
                evidence['classification_error'] = f'Exact complete {part["kind"]} segment stored in {field}'
    return evidence
