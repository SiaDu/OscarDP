from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from .schema import MovieRef

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".webm"}
PARTIAL_SUFFIXES = (".part", ".partial", ".crdownload")
IMDB_PATTERN = re.compile(r"(?i)(?<![a-z0-9])(tt\d{7,10})(?!\d)")


def _has_hidden_component(relative_path: Path) -> bool:
    return any(part.startswith(".") for part in relative_path.parts)


def _normalized_fallback(relative_path: Path) -> str:
    without_suffix = relative_path.with_suffix("").as_posix()
    normalized = unicodedata.normalize("NFKC", without_suffix).lower()
    normalized = re.sub(r"[^a-z0-9/]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    parts = [part.strip("_") for part in normalized.split("/") if part.strip("_")]
    key = "__".join(parts)
    if not key:
        digest = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:12]
        return f"movie_{digest}"
    return key


def derive_movie_key(video_path: Path, input_root: Path) -> str:
    root = input_root.resolve()
    path = video_path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Video is outside input root: {video_path}") from exc

    match = IMDB_PATTERN.search(path.name)
    if match:
        return match.group(1).lower()
    for parent in relative.parents[:-1]:
        match = IMDB_PATTERN.search(parent.name)
        if match:
            return match.group(1).lower()
    return _normalized_fallback(relative)


def discover_movies(
    input_root: Path,
    *,
    limit: int | None = None,
    movie_key: str | None = None,
) -> list[MovieRef]:
    root = input_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    refs: list[MovieRef] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root)
        lower_name = path.name.lower()
        if _has_hidden_component(relative):
            continue
        if lower_name.endswith(PARTIAL_SUFFIXES) or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        key = derive_movie_key(path, root)
        if movie_key and key != movie_key.lower():
            continue
        refs.append(MovieRef(key, path.resolve(), relative.as_posix()))

    collisions: dict[str, list[str]] = {}
    for ref in refs:
        collisions.setdefault(ref.movie_key, []).append(ref.source_video_relpath)
    duplicates = {key: paths for key, paths in collisions.items() if len(paths) > 1}
    if duplicates:
        details = "; ".join(f"{key}: {', '.join(paths)}" for key, paths in sorted(duplicates.items()))
        raise ValueError(f"Duplicate movie_key values: {details}")
    return refs[:limit] if limit is not None else refs
