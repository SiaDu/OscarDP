"""Explicit, source-bound human decisions; no lexical semantic defaults."""
import re

from .text import segments


def bind_decisions(records, movie_id, sha256, cues, shots):
    """Return valid records and errors. Invalid records never alter comparison."""
    bound, errors, seen = [], [], set()
    shot_ids = {s.get('shot_id') for s in shots if isinstance(s, dict)}
    for record in records:
        if not isinstance(record, dict):
            errors.append('Decision must be an object')
            continue
        if record.get('movie_id') != movie_id:
            continue
        index = record.get('srt_index')
        source = [c for c in cues if c.index == index]
        key = (index, record.get('text'))
        valid = (type(index) is int and index > 0 and len(source) == 1
                 and record.get('source') == 'user_confirmed_audiovisual_review'
                 and isinstance(record.get('decision_id'), str) and bool(record['decision_id'])
                 and isinstance(record.get('original_srt_sha256'), str)
                 and re.fullmatch(r'[a-f0-9]{64}', record['original_srt_sha256'])
                 and record['original_srt_sha256'] == sha256
                 and record.get('source_text') == source[0].text
                 and record.get('classification') in ('dialogue', 'sound_cues', 'song_lyrics')
                 and isinstance(record.get('evidence'), str) and bool(record['evidence'])
                 and isinstance(record.get('affected_shots'), list) and bool(record['affected_shots'])
                 and all(isinstance(s, str) and s in shot_ids for s in record['affected_shots'])
                 and len(set(record['affected_shots'])) == len(record['affected_shots'])
                 and isinstance(record.get('text'), str) and bool(record['text'])
                 and sum(p['text'] == record['text'] for p in segments(source[0])['segments']) == 1
                 and key not in seen)
        if valid:
            bound.append(record)
            seen.add(key)
        else:
            errors.append(f'Unbound or duplicate human decision: {record.get("decision_id", index)}')
    return bound, errors


def check_consistency(records, shots, matches):
    """Require each adjudicated complete segment exactly once in its field per shot."""
    errors = []
    for record in records:
        for shot_id in record['affected_shots']:
            occurrences = []
            for match in matches:
                if match['shot_id'] != shot_id or record['srt_index'] not in match['srt_indices']:
                    continue
                if match['status'] == 'mismatch':
                    continue
                parts = (match['extraction']['parts'] if match['extraction'] else
                         [s for p in match['source_segments'] for s in p['segments']])
                if any(p['text'] == record['text'] for p in parts):
                    occurrences.append(match)
            if len(occurrences) != 1 or occurrences[0]['field'] != record['classification']:
                errors.append((shot_id, record, 'Adjudicated segment must occur exactly once in the confirmed field'))
            elif record['classification'] == 'dialogue' and record.get('speaker') is None:
                match = occurrences[0]
                shot = next(s for s in shots if s.get('shot_id') == shot_id)
                if shot['dialogue'][match['entry']].get('speaker') is not None:
                    errors.append((shot_id, record, 'No confirmed speaker identity; speaker must remain null'))
    return errors
