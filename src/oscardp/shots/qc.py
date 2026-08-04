from __future__ import annotations

from pathlib import Path
from statistics import mean, median

from .schema import ShotRecord
from .transnet import Boundary


def _evenly_sample(items: list[Boundary], count: int) -> list[Boundary]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    positions = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[position] for position in positions]


def select_qc_boundaries(
    boundaries: list[Boundary],
    shots: list[ShotRecord],
    *,
    threshold: float,
    save_all: bool = False,
    sample_count: int = 24,
    max_boundaries: int = 100,
    very_short_sec: float = 0.5,
) -> list[Boundary]:
    if save_all:
        return list(boundaries)
    by_frame = {boundary.frame: boundary for boundary in boundaries}
    low = sorted(
        (boundary for boundary in boundaries if boundary.confidence < min(1.0, threshold + 0.1)),
        key=lambda boundary: (boundary.confidence, boundary.frame),
    )
    short_frames: set[int] = set()
    for shot in shots:
        if shot.duration_sec < very_short_sec:
            if shot.start_frame in by_frame:
                short_frames.add(shot.start_frame)
            if shot.end_frame in by_frame:
                short_frames.add(shot.end_frame)
    short = sorted((by_frame[frame] for frame in short_frames), key=lambda boundary: boundary.frame)
    sampled = _evenly_sample(boundaries, sample_count)
    selected: list[Boundary] = []
    seen: set[int] = set()
    for group in (low, short, sampled):
        for boundary in group:
            if boundary.frame not in seen:
                selected.append(boundary)
                seen.add(boundary.frame)
            if len(selected) >= max_boundaries:
                return sorted(selected, key=lambda boundary: boundary.frame)
    return sorted(selected, key=lambda boundary: boundary.frame)


def build_contact_sheet(
    pairs: list[tuple[Path, Path, str]],
    output_path: Path,
    *,
    max_pairs: int = 24,
    thumb_width: int = 320,
) -> None:
    from PIL import Image, ImageDraw

    visible = pairs[:max_pairs]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not visible:
        image = Image.new("RGB", (640, 80), "white")
        ImageDraw.Draw(image).text((12, 28), "No QC boundaries selected", fill="black")
        image.save(output_path, quality=90)
        return
    rows = []
    for before_path, after_path, label in visible:
        with Image.open(before_path) as before_source, Image.open(after_path) as after_source:
            before = before_source.convert("RGB")
            after = after_source.convert("RGB")
            thumb_height = max(1, round(before.height * thumb_width / before.width))
            before.thumbnail((thumb_width, thumb_height))
            after.thumbnail((thumb_width, thumb_height))
            row_height = max(before.height, after.height) + 32
            row = Image.new("RGB", (thumb_width * 2, row_height), "white")
            row.paste(before, (0, 32))
            row.paste(after, (thumb_width, 32))
            draw = ImageDraw.Draw(row)
            draw.text((8, 8), f"{label} before", fill="black")
            draw.text((thumb_width + 8, 8), f"{label} after", fill="black")
            rows.append(row)
    sheet = Image.new("RGB", (thumb_width * 2, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(output_path, quality=90)


def build_qc_summary(
    shots: list[ShotRecord],
    *,
    missing_keyframes: int,
    validation_passed: bool,
    very_short_sec: float = 0.5,
) -> dict[str, int | float | bool]:
    durations = [shot.duration_sec for shot in shots]
    return {
        "shot_count": len(shots),
        "mean_shot_duration_sec": round(mean(durations), 6) if durations else 0.0,
        "median_shot_duration_sec": round(median(durations), 6) if durations else 0.0,
        "very_short_shot_count": sum(duration < very_short_sec for duration in durations),
        "missing_keyframes": int(missing_keyframes),
        "validation_passed": bool(validation_passed),
    }
