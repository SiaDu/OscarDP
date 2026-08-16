from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from oscardp.shots.schema import json_dumps


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")  # noqa: TRY004
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{number}")  # noqa: TRY004
            rows.append(value)
    return rows


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    write_atomic(path, json_dumps(value, pretty=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_atomic(path, "".join(json_dumps(row) + "\n" for row in rows))
