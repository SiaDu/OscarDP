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


def load_clean_subtitles(path: Path, language: str = "en") -> list[CleanSubtitle]:
    try:
        import srt
    except ImportError as exc:
        raise RuntimeError('srt is required; install with pip install -e ".[context]"') from exc
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    candidates: list[tuple[float, float, str, str, str]] = []
    for item in srt.parse(raw):
        original = re.sub(r"\s+", " ", item.content).strip()
        cleaned = clean_text(item.content)
        if not cleaned:
            continue
        detected = detect_language(cleaned)
        if language and detected != language:
            continue
        candidates.append((item.start.total_seconds(), item.end.total_seconds(), original, cleaned, detected))
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
