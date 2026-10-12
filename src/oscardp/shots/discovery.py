from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from .schema import MovieRef

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".webm", ".rmvb"}
PARTIAL_SUFFIXES = (".part", ".partial", ".crdownload")
IMDB_PATTERN = re.compile(r"(?i)(?<![a-z0-9])(tt\d{7,10})(?!\d)")
CANONICAL_ID = re.compile(r"tt\d{7,10}")


def load_movie_titles(path: Path) -> dict[str, str]:
    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate movie ID in title JSON: {key}")
            result[key] = value
        return result

    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read movie titles from {path}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError("Movie titles must be a nonempty JSON object")
    for movie_id, title in data.items():
        if not CANONICAL_ID.fullmatch(movie_id):
            raise ValueError(f"Invalid IMDb ID in title JSON: {movie_id}")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Missing or invalid movie title for {movie_id}")
    return data


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
    allowed_ids: set[str] | None = None,
    require_all: bool = False,
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
        if allowed_ids is not None and key not in allowed_ids:
            continue
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
    if require_all and allowed_ids is not None:
        expected = allowed_ids if movie_key is None else allowed_ids & {movie_key.lower()}
        missing = expected - {ref.movie_key for ref in refs}
        if missing:
            raise ValueError(f"Missing allowlisted movie IDs: {', '.join(sorted(missing))}")
    return refs[:limit] if limit is not None else refs
