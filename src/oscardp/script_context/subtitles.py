from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

from oscardp.shots.schema import format_timestamp, rounded_seconds

from .schema import CleanSubtitle


TAG_RE = re.compile(r"<[^>]+>|\{\\[^}]+\}")
MUSIC_RE = re.compile(r"[♪♫]+")
CJK_RE = re.compile(r"[\u3400-\u9fff]")
WEBVTT_HEADER_RE = re.compile(r"(?mi)^\s*WEBVTT(?:[^\r\n]*)?$")
WEBVTT_TIMESTAMP_RE = re.compile(
    r"(?m)^(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{3})[ \t]*-->[ \t]*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{3})(?:[ \t]+[^\r\n]*)?$"
)
RELEASE_CREDIT_RE = re.compile(
    r"(?i)^(?:(?:downloaded\s+from|official\s+yify\s+movies\s+site:)\s+)?(?:www\.)?"
    r"(?:yts\.(?:bz|lt|mx)|opensubtitles(?:\.org)?|hamiltonshare)$"
)


def clean_text(text: str) -> str:
    value = html.unescape(TAG_RE.sub("", text))
    value = MUSIC_RE.sub(" ", value)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def detect_language(text: str) -> str:
    letters = re.findall(r"[A-Za-z]", text)
    cjk = CJK_RE.findall(text)
    if cjk and len(cjk) >= len(letters):
        return "zh"
    if letters:
        return "en"
    return "und"


def _timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_webvtt_fragments(raw: str) -> list[tuple[float, float, str]]:
    matches = list(WEBVTT_TIMESTAMP_RE.finditer(raw))
    cues: list[tuple[float, float, str]] = []
    for index, match in enumerate(matches):
        segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        lines = raw[match.end():segment_end].splitlines()
        content: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if content:
                    break
                continue
            if WEBVTT_HEADER_RE.fullmatch(stripped):
                if content:
                    break
                continue
            if stripped.isdigit() and not content:
                continue
            content.append(line)
        text = "\n".join(content).strip()
        if text:
            cues.append((_timestamp_seconds(match["start"]), _timestamp_seconds(match["end"]), text))
    return cues


def _language_text(text: str, language: str) -> str:
    if not language:
        return clean_text(text)
    selected: list[str] = []
    for raw_line in text.splitlines():
        line = clean_text(raw_line)
        if not line:
            continue
        if language == "en" and re.search(r"[A-Za-z]", line):
            line = clean_text(CJK_RE.sub("", line))
            if re.search(r"[A-Za-z]", line):
                selected.append(line)
        elif language == "zh" and CJK_RE.search(line):
            selected.append(line)
        elif detect_language(line) == language:
            selected.append(line)
    return clean_text(" ".join(selected))


def load_clean_subtitles(path: Path, language: str = "en") -> list[CleanSubtitle]:
    try:
        import srt
    except ImportError as exc:
        raise RuntimeError('srt is required; install with pip install -e ".[context]"') from exc
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    candidates: list[tuple[float, float, str, str, str]] = []
    if WEBVTT_HEADER_RE.search(raw):
        parsed = _parse_webvtt_fragments(raw)
    else:
        parsed = [
            (item.start.total_seconds(), item.end.total_seconds(), item.content)
            for item in srt.parse(raw)
        ]
    for start, end, content in parsed:
        original = re.sub(r"\s+", " ", content).strip()
        cleaned = _language_text(content, language)
        if not cleaned:
            continue
        if RELEASE_CREDIT_RE.fullmatch(cleaned):
            continue
        detected = detect_language(cleaned)
        candidates.append((start, end, original, cleaned, detected))
    candidates.sort(key=lambda row: (row[0], row[1], row[3]))
    seen: set[tuple[float, float, str]] = set()
    rows: list[CleanSubtitle] = []
    for start, end, original, cleaned, detected in candidates:
        key = (start, end, unicodedata.normalize("NFKC", cleaned).casefold())
        if key in seen:
            continue
        seen.add(key)
        ordinal = len(rows) + 1
        rows.append(CleanSubtitle(
            subtitle_id=f"subtitle_{ordinal:06d}", start=format_timestamp(start), end=format_timestamp(end),
            start_sec=rounded_seconds(start), end_sec=rounded_seconds(end), original_text=original,
            cleaned_text=cleaned, detected_language=detected,
        ))
    return rows
