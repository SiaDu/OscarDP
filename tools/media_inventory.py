#!/usr/bin/env python3
"""Stage 0B movie-level inventory built from a Stage 0A preflight report."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oscardp.stage0.media import GIB, is_english, inventory_classification, profile_reasons, probe, stream_language, video_bitrate
from oscardp.stage0.preflight import SUBTITLE_EXTENSIONS, _subtitle_language

FIELDS = [
    "movie_id", "movie_key", "movie_folder", "source_video_relpath", "source_video_filename", "resolution", "width", "height", "fps", "video_codec", "container", "pixel_format", "bit_depth", "dynamic_range", "duration_sec", "video_bitrate_mbps", "video_bitrate_source", "audio_codecs", "audio_tracks", "audio_languages", "embedded_subtitle_count", "embedded_subtitle_languages", "external_subtitle_count", "external_subtitle_formats", "external_subtitle_languages", "english_subtitle_count", "english_subtitle_paths", "has_english_subtitle", "filesize_bytes", "filesize_gib", "classification", "classification_reasons", "manual_review_required", "manual_review_reasons", "ffprobe_error",
]


def _jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _external_subtitles(folder: Path, root: Path) -> list[tuple[Path, str]]:
    return [(path, _subtitle_language(path)) for path in sorted(folder.rglob("*"), key=lambda item: item.as_posix().lower()) if path.is_file() and path.suffix.lower() in SUBTITLE_EXTENSIONS]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 0B movie-level inventory; requires Stage 0A dry-run report.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True, help="stage0_source_preflight.jsonl from Stage 0A")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); root = args.input_root.resolve()
    if not root.is_dir(): parser.error(f"Input root does not exist: {root}")
    if not args.preflight.is_file(): parser.error(f"Preflight report does not exist: {args.preflight}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader()
        for preflight in _jsonl(args.preflight):
            movie_folder = str(preflight["movie_folder"]); folder = root / movie_folder
            source_relpath = str(preflight.get("primary_video_path") or "")
            row: dict[str, object] = {"movie_id": preflight.get("movie_id", ""), "movie_key": preflight.get("movie_key", ""), "movie_folder": movie_folder, "source_video_relpath": source_relpath, "source_video_filename": Path(source_relpath).name if source_relpath else "", "manual_review_required": preflight.get("manual_review_required", False), "manual_review_reasons": preflight.get("manual_review_reasons", ""), "ffprobe_error": ""}
            subtitles = _external_subtitles(folder, root) if folder.is_dir() else []
            english_paths = [path.relative_to(root).as_posix() for path, language in subtitles if is_english(language)]
            row.update({"external_subtitle_count": len(subtitles), "external_subtitle_formats": ";".join(sorted({path.suffix.lower().lstrip(".") for path, _ in subtitles})), "external_subtitle_languages": ";".join(language for _path, language in subtitles), "english_subtitle_count": len(english_paths), "english_subtitle_paths": ";".join(english_paths)})
            if not source_relpath:
                row.update({"classification": "MANUAL_REVIEW", "classification_reasons": "no unambiguous primary video", "has_english_subtitle": bool(english_paths)})
                writer.writerow(row); continue
            source = root / source_relpath
            try:
                info = probe(source); bitrate, bitrate_source = video_bitrate(info, source)
                embedded_languages = [stream_language(stream) for stream in info.subtitle_streams]
                row.update({"resolution": f"{info.width}x{info.height}", "width": info.width, "height": info.height, "fps": f"{info.fps:.3f}", "video_codec": info.codec, "container": info.container, "pixel_format": info.pixel_format, "bit_depth": info.bit_depth, "dynamic_range": info.dynamic_range, "duration_sec": f"{info.duration_sec:.3f}", "video_bitrate_mbps": f"{bitrate / 1_000_000:.3f}" if bitrate else "", "video_bitrate_source": bitrate_source, "audio_codecs": ";".join(str(stream.get("codec_name") or "unknown") for stream in info.audio_streams), "audio_tracks": len(info.audio_streams), "audio_languages": ";".join(stream_language(stream) for stream in info.audio_streams), "embedded_subtitle_count": len(info.subtitle_streams), "embedded_subtitle_languages": ";".join(embedded_languages), "has_english_subtitle": bool(english_paths or any(is_english(language) for language in embedded_languages)), "filesize_bytes": source.stat().st_size, "filesize_gib": f"{source.stat().st_size / GIB:.3f}", "classification": inventory_classification(info), "classification_reasons": "; ".join(profile_reasons(info)) or "meets target media profile"})
            except Exception as exc:
                row.update({"classification": "ERROR", "classification_reasons": "ffprobe failed or no video stream", "has_english_subtitle": bool(english_paths), "ffprobe_error": str(exc)})
            writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
