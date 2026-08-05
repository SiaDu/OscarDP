from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CleanSubtitle:
    subtitle_id: str
    start: str
    end: str
    start_sec: float
    end_sec: float
    original_text: str
    cleaned_text: str
    detected_language: str


@dataclass(frozen=True)
class AlignmentConfig:
    alignment_threshold: float = 0.82
    review_threshold: float = 0.65
    candidate_margin: float = 0.08
    dialogue_window: int = 100
    semantic_model: str | None = None


@dataclass(frozen=True)
class ContextOptions:
    movie_key: str
    screenplay: Path
    subtitle: Path
    shots: Path
    output_dir: Path
    subtitle_language: str = "en"
    alignment_threshold: float = 0.82
    review_threshold: float = 0.65
    semantic_model: str | None = None
    disable_semantic: bool = False
    llm_mode: str = "none"
    llm_responses: Path | None = None
    scene_interpolation_max_gap: float = 10.0
    resume: bool = True
    overwrite: bool = False
    dry_run: bool = False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}: {exc}") from exc
    return rows
