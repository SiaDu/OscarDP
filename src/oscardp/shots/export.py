from __future__ import annotations

import csv
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from .schema import ShotRecord, json_dumps

SHOT_CSV_FIELDS = [field.name for field in dataclasses.fields(ShotRecord)]


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json_dumps(value)
    return value


def export_shots_csv(movie_dir: Path, output_path: Path | None = None) -> Path:
    shots_path = movie_dir / "shots.jsonl"
    if not shots_path.is_file():
        raise FileNotFoundError(f"Missing shots.jsonl: {shots_path}")
    rows: list[dict[str, Any]] = []
    with shots_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            missing = [field for field in SHOT_CSV_FIELDS if field not in value]
            if missing:
                raise ValueError(
                    f"shots.jsonl line {line_number} is missing fields: {', '.join(missing)}"
                )
            rows.append({field: _csv_value(value[field]) for field in SHOT_CSV_FIELDS})
    if not rows:
        raise ValueError("Cannot export an empty shots.jsonl")
    destination = output_path or movie_dir / "shots.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHOT_CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination
