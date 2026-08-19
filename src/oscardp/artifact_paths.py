"""Explicit relocation support for immutable artifact manifests.

Frozen OscarDP manifests retain the absolute paths that were valid when they
were produced.  A consumer may relocate a mounted dataset, but it must opt in
and retain the declared path for provenance rather than rewriting the manifest.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


PathMap = tuple[Path, Path]


def parse_path_maps(specifications: Iterable[str]) -> tuple[PathMap, ...]:
    """Parse repeatable ``OLD=NEW`` absolute-prefix mappings.

    The longest matching source prefix wins.  Duplicate source prefixes are
    rejected so that a command has one unambiguous relocation contract.
    """
    mappings: list[PathMap] = []
    seen: set[Path] = set()
    for specification in specifications:
        if not isinstance(specification, str) or specification.count("=") != 1:
            raise ValueError("path map must be OLD=NEW")
        old_text, new_text = specification.split("=", 1)
        if not old_text or not new_text:
            raise ValueError("path map OLD and NEW must be non-empty")
        old, new = Path(old_text), Path(new_text)
        if not old.is_absolute() or not new.is_absolute():
            raise ValueError("path map OLD and NEW must be absolute paths")
        old, new = old.resolve(strict=False), new.resolve(strict=False)
        if old in seen:
            raise ValueError(f"duplicate path map source: {old}")
        seen.add(old)
        mappings.append((old, new))
    return tuple(sorted(mappings, key=lambda item: len(item[0].parts), reverse=True))


def resolve_artifact_path(declared_path: str | Path, path_maps: Iterable[PathMap] = ()) -> Path:
    """Return the locally accessible path for an absolute manifest path.

    A mapping applies only at a complete directory boundary; e.g. ``/mnt/i``
    cannot rewrite ``/mnt/inside``.  The returned path is always below the
    explicit destination root when a mapping is used.
    """
    declared = Path(declared_path)
    if not declared.is_absolute():
        raise ValueError(f"artifact path must be absolute: {declared_path}")
    declared = declared.resolve(strict=False)
    for old, new in path_maps:
        try:
            relative = declared.relative_to(old)
        except ValueError:
            continue
        resolved = (new / relative).resolve(strict=False)
        try:
            resolved.relative_to(new)
        except ValueError as exc:  # Defensive: reject any attempted escape.
            raise ValueError(f"path map resolves outside destination: {declared_path}") from exc
        return resolved
    return declared
