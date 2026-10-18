"""Strict SRT parsing, conservative text comparison and IMDb-based discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
import html
import os
from pathlib import Path
import re
import unicodedata

from oscardp.shots.discovery import IMDB_PATTERN, VIDEO_EXTENSIONS

TIMING = re.compile(r'^(\d{2,}):([0-5]\d):([0-5]\d),(\d{3})\s+-->\s+(\d{2,}):([0-5]\d):([0-5]\d),(\d{3})$')
# Strip only recognised subtitle presentation tags; preserve other angle-bracket text.
MARKUP = re.compile(r'</?(?:i|b|u|s|font|span)(?:\s+[^<>]*)?\s*>', re.I)
BREAK = re.compile(r'<br\s*/?>', re.I)
# Recognised ASS alignment overrides only; never discard arbitrary brace text.
ALIGNMENT = re.compile(r'\{\\an[1-9]\}')


@dataclass(frozen=True)
class Cue:
    index: int
    start: float
    end: float
    text: str


@dataclass
class ParsedSRT:
    cues: list[Cue] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)


def parse_srt(text: str) -> ParsedSRT:
    """Keep cue order, numbers and original multiline text; never renumber/re-time."""
    result = ParsedSRT()
    text = text.removeprefix('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
    if text.lstrip().startswith('WEBVTT'):
        result.issues.append(dict(rule_code='SRT_UNSUPPORTED_FORMAT',
                                 message='WebVTT content is not numbered SRT; original SRT indices cannot be confirmed',
                                 actual=text[:1000]))
        return result
    seen = set()
    for block_number, block in enumerate(re.split(r'\n[ \t]*\n+', text), 1):
        if not block.strip():
            continue
        lines = block.split('\n')
        if not re.fullmatch(r'\d+', lines[0].strip()):
            result.issues.append(dict(rule_code='SRT_FORMAT', message='Missing integer cue index', block=block_number, actual=block))
            continue
        index = int(lines[0])
        if index in seen:
            result.issues.append(dict(rule_code='SRT_DUPLICATE_INDEX', message='Duplicate subtitle index', srt_index=index, actual=block))
        seen.add(index)
        match = TIMING.fullmatch(lines[1].strip()) if len(lines) > 1 else None
        if index < 1 or not match:
            result.issues.append(dict(rule_code='SRT_TIMESTAMP', message='Invalid index or timestamp', srt_index=index, actual=block))
            continue
        values = list(map(int, match.groups()))
        start, end = (h * 3600 + m * 60 + s + ms / 1000 for h, m, s, ms in (values[:4], values[4:]))
        if start >= end:
            result.issues.append(dict(rule_code='SRT_TIMESTAMP', message='Cue start must precede end', srt_index=index, actual=block))
            continue
        body = '\n'.join(lines[2:]).rstrip('\n')
        if not body.strip():
            result.issues.append(dict(rule_code='SRT_FORMAT', message='Empty subtitle text', srt_index=index, actual=block))
        # Detect missing blank separators instead of swallowing a second cue as text.
        if any('-->' in line and TIMING.fullmatch(line.strip()) for line in lines[2:]):
            result.issues.append(dict(rule_code='SRT_FORMAT', message='Missing separator between cues', srt_index=index, actual=block))
        result.cues.append(Cue(index, start, end, body))
    if not result.cues:
        result.issues.append(dict(rule_code='SRT_EMPTY', message='No valid cues found', actual=text[:1000]))
    return result


def normalized_text(text: str) -> tuple[str, list[str]]:
    reasons = []
    steps = [
        ('subtitle_markup', lambda s: MARKUP.sub('', BREAK.sub('\n', s))),
        ('subtitle_alignment', lambda s: ALIGNMENT.sub('', s)),
        ('html_entities', html.unescape),
        ('unicode_nfc', lambda s: unicodedata.normalize('NFC', s)),
        ('whitespace', lambda s: ' '.join(s.split())),
    ]
    for reason, transform in steps:
        new = transform(text)
        if new != text:
            reasons.append(reason)
        text = new
    return text, reasons


def compare_text(actual: str, cues: list[Cue]) -> dict:
    """Reconstruct in supplied index order. Exact merge separator is a newline."""
    expected = '\n'.join(cue.text for cue in cues)
    normalized_expected, er = normalized_text(expected)
    normalized_actual, ar = normalized_text(actual)
    status = 'exact' if actual == expected else 'normalized' if normalized_actual == normalized_expected else 'mismatch'
    return dict(status=status, expected=expected, actual=actual,
                reasons=sorted(set(er + ar)) if status != 'exact' else [],
                normalized_expected=normalized_expected, normalized_actual=normalized_actual)


def _path_id(path: Path, root: Path) -> str | None:
    """Same filename-then-nearest-parent ID precedence as dataset_preflight.file_id."""
    for name in [path.name, *(p.name for p in path.relative_to(root).parents[:-1])]:
        match = IMDB_PATTERN.search(name)
        if match:
            return match.group(1).lower()
    return None


class SRTIndex(dict):
    """Candidates plus uncertain shared-directory ownership."""
    def __init__(self):
        super().__init__()
        self.ambiguous_ids = set()


def index_srts(root: Path) -> SRTIndex:
    """One traversal; no workbook, fuzzy titles, renames or video decoding."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'SRT root not found: {root}')
    subtitles = []
    directory_ids: dict[Path, set[str]] = {}
    def scan_error(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=scan_error):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if name.startswith('.') or path.is_symlink():
                continue
            if path.suffix.lower() == '.srt':
                subtitles.append(path)
            elif path.suffix.lower() in VIDEO_EXTENSIONS:
                movie_id = _path_id(path, root)
                if movie_id:
                    directory_ids.setdefault(path.parent, set()).add(movie_id)
    result = SRTIndex()
    for path in subtitles:
        direct_id = _path_id(path, root)
        # Unnamed subtitle may belong to a video with an explicit ID in that directory.
        owners = {direct_id} if direct_id else directory_ids.get(path.parent, set())
        if len(owners) > 1:
            result.ambiguous_ids.update(owners)
        for movie_id in owners:
            result.setdefault(movie_id, []).append(path)
    return result
