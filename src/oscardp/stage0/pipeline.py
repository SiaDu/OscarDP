from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .media import GIB, MediaInfo, classification, probe, select_audio_stream, supports_hdr_tonemap, target_dimensions, transcode_reasons, video_filter

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".webm"}
FIELDS = [
    "movie_id", "movie_title", "source_path", "processing_source_path", "source_container", "source_codec",
    "source_width", "source_height", "source_fps", "source_bit_depth", "source_dynamic_range", "source_size_gib",
    "source_duration_sec", "classification", "transcode_required", "transcode_reasons", "target_codec",
    "target_width", "target_height", "target_fps", "target_dynamic_range", "target_size_limit_gib", "output_path",
    "output_exists", "output_size_gib", "ffmpeg_status", "validation_status", "error_message",
]
ID_PATTERN = re.compile(r"(?i)(tt\d{7,10})")


@dataclass(frozen=True)
class NormalizeOptions:
    input_root: Path
    output_root: Path
    inventory: Path | None = None
    execute: bool = False
    movie_id: str | None = None
    limit: int | None = None
    force: bool = False
    cq: int = 25
    max_size_gib: float = 4.5


def discover_videos(root: Path, movie_id: str | None, limit: int | None, inventory: Path | None) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    paths: list[Path] = []
    # Inventory paths are relative by design, so an inventory created against a
    # different mount point can safely be reused with this input root.
    if inventory is not None:
        with inventory.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                relative = row.get("source_video_relpath") or row.get("source_path")
                if not relative:
                    continue
                candidate = Path(relative)
                candidate = candidate if candidate.is_absolute() else root / candidate
                try:
                    candidate.relative_to(root)
                except ValueError:
                    continue
                if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
                    paths.append(candidate)
        paths = sorted(set(paths), key=lambda item: item.as_posix().lower())
    if not paths:
        paths = [path for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()) if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and not any(part.startswith(".") for part in path.relative_to(root).parts)]
    if movie_id:
        wanted = movie_id.lower()
        paths = [path for path in paths if wanted in path.as_posix().lower()]
    return paths[:limit] if limit is not None else paths


def movie_identity(path: Path, input_root: Path) -> tuple[str, str]:
    relative = path.relative_to(input_root)
    match = ID_PATTERN.search(relative.as_posix())
    movie_id = match.group(1).lower() if match else relative.parent.name
    title = relative.parent.name
    return movie_id, title


def output_path(path: Path, input_root: Path, output_root: Path) -> Path:
    relative = path.relative_to(input_root)
    return output_root / relative.parent / f"{path.stem}_standardized.mp4"


def source_fields(info: MediaInfo) -> dict[str, object]:
    return {
        "source_container": info.container, "source_codec": info.codec, "source_width": info.width,
        "source_height": info.height, "source_fps": round(info.fps, 6), "source_bit_depth": info.bit_depth,
        "source_dynamic_range": info.dynamic_range, "source_size_gib": round(info.size_gib, 6),
        "source_duration_sec": round(info.duration_sec, 6),
    }


def validate_output(path: Path, source: MediaInfo, max_size_gib: float) -> tuple[bool, str, MediaInfo | None]:
    try:
        output = probe(path)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        return False, f"ffprobe validation failed: {exc}", None
    problems: list[str] = []
    if output.width > 1920 or output.height > 1080: problems.append("resolution exceeds 1920x1080")
    if output.bit_depth != 8: problems.append(f"bit depth is {output.bit_depth}, expected 8")
    if output.pixel_format != "yuv420p": problems.append(f"pixel format is {output.pixel_format}, expected yuv420p")
    if output.dynamic_range != "SDR": problems.append("output is not SDR")
    if (output.color_primaries, output.color_transfer, output.color_space) != ("bt709", "bt709", "bt709"):
        problems.append("output color metadata is not BT.709")
    if output.size_gib > max_size_gib: problems.append(f"size exceeds {max_size_gib} GiB")
    if abs(output.duration_sec - source.duration_sec) > 0.5: problems.append("duration differs by over 0.5 sec")
    if source.fps and abs(output.fps - source.fps) > max(0.01, source.fps * 0.001): problems.append("fps changed")
    return not problems, "; ".join(problems), output


def ffmpeg_command(source: Path, partial: Path, info: MediaInfo, cq: int, bitrate: int | None = None) -> list[str]:
    audio_stream = select_audio_stream(info)
    command = ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", "0:v:0"]
    if audio_stream is not None:
        command.extend(["-map", f"0:{audio_stream}", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"])
    command.extend(["-vf", video_filter(info), "-c:v", "hevc_nvenc", "-preset", "p6", "-pix_fmt", "yuv420p"])
    if bitrate is None:
        command.extend(["-rc", "vbr", "-cq", str(cq), "-b:v", "0"])
    else:
        command.extend(["-rc", "vbr", "-b:v", str(bitrate), "-maxrate", str(bitrate), "-bufsize", str(bitrate * 2)])
    command.extend(["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart", str(partial)])
    return command


def encode(source: Path, final: Path, info: MediaInfo, options: NormalizeOptions) -> tuple[str, str]:
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name(final.stem + ".partial.mp4")
    if partial.exists():
        partial.unlink()
    try:
        subprocess.run(ffmpeg_command(source, partial, info, options.cq), text=True, capture_output=True, check=True)
        if partial.stat().st_size / GIB > options.max_size_gib:
            partial.unlink()
            target_bits = 4.3 * GIB * 8
            bitrate = int(target_bits / info.duration_sec - 192_000)
            if bitrate <= 0:
                return "FAILED_OVERSIZE", "computed non-positive retry bitrate"
            subprocess.run(ffmpeg_command(source, partial, info, options.cq, bitrate), text=True, capture_output=True, check=True)
        valid, message, _ = validate_output(partial, info, options.max_size_gib)
        if not valid:
            return ("FAILED_OVERSIZE" if partial.exists() and partial.stat().st_size / GIB > options.max_size_gib else "VALIDATION_FAILED"), message
        os.replace(partial, final)
        return "PASS", ""
    except subprocess.CalledProcessError as exc:
        return "FAILED_FFMPEG", exc.stderr.strip() or str(exc)
    finally:
        if partial.exists():
            partial.unlink()


def _write_reports(rows: list[dict[str, object]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "stage0_media_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    with (output_root / "stage0_media_inventory.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    errors = [row for row in rows if row.get("error_message")]
    with (output_root / "stage0_errors.jsonl").open("w", encoding="utf-8") as handle:
        for row in errors: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"movie_count": len(rows), "by_classification": {}, "by_ffmpeg_status": {}, "error_count": len(errors)}
    for row in rows:
        for field, key in (("classification", "by_classification"), ("ffmpeg_status", "by_ffmpeg_status")):
            value = str(row[field]); summary[key][value] = summary[key].get(value, 0) + 1
    (output_root / "stage0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(options: NormalizeOptions) -> list[dict[str, object]]:
    if options.limit is not None and options.limit < 1: raise ValueError("--limit must be at least 1")
    rows: list[dict[str, object]] = []
    for source in discover_videos(options.input_root, options.movie_id, options.limit, options.inventory):
        movie_id, title = movie_identity(source, options.input_root)
        final = output_path(source, options.input_root, options.output_root)
        row: dict[str, object] = {"movie_id": movie_id, "movie_title": title, "source_path": str(source), "output_path": str(final), "target_codec": "hevc", "target_dynamic_range": "SDR", "target_size_limit_gib": options.max_size_gib, "output_exists": final.exists(), "output_size_gib": round(final.stat().st_size / GIB, 6) if final.exists() else "", "error_message": ""}
        try:
            info = probe(source); row.update(source_fields(info))
            width, height = target_dimensions(info); row.update({"target_width": width, "target_height": height, "target_fps": round(info.fps, 6)})
            reasons = transcode_reasons(info, options.max_size_gib); row.update({"classification": classification(reasons), "transcode_required": bool(reasons), "transcode_reasons": ";".join(reasons)})
            if not reasons:
                row.update({"processing_source_path": str(source), "ffmpeg_status": "NOT_REQUIRED", "validation_status": "KEEP"})
            elif info.dynamic_range == "HDR" and not supports_hdr_tonemap()[0]:
                row.update({"processing_source_path": "", "ffmpeg_status": "FAILED_HDR_TONEMAP_UNAVAILABLE", "validation_status": "NOT_RUN", "error_message": supports_hdr_tonemap()[1]})
            elif final.exists() and not options.force:
                valid, message, output = validate_output(final, info, options.max_size_gib)
                row.update({"processing_source_path": str(final) if valid else "", "ffmpeg_status": "SKIPPED_EXISTING_VALID" if valid else "SKIPPED_EXISTING_INVALID", "validation_status": "PASS" if valid else "FAIL", "output_size_gib": round(output.size_gib, 6) if output else "", "error_message": message})
            elif not options.execute:
                row.update({"processing_source_path": str(final), "ffmpeg_status": "DRY_RUN", "validation_status": "NOT_RUN"})
            else:
                status, error = encode(source, final, info, options)
                valid, message, output = validate_output(final, info, options.max_size_gib) if status == "PASS" else (False, error, None)
                row.update({"processing_source_path": str(final) if valid else "", "ffmpeg_status": status, "validation_status": "PASS" if valid else "FAIL", "output_exists": final.exists(), "output_size_gib": round(output.size_gib, 6) if output else "", "error_message": message})
        except Exception as exc:  # Per-movie isolation is intentional for batch processing.
            row.update({"processing_source_path": "", "classification": "ERROR", "transcode_required": "", "transcode_reasons": "", "ffmpeg_status": "FAILED_PROBE", "validation_status": "NOT_RUN", "error_message": str(exc)})
        rows.append(row)
    _write_reports(rows, options.output_root)
    return rows
