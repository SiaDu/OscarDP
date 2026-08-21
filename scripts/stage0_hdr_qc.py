#!/usr/bin/env python3
"""Create a bounded visual/numeric comparison of the two CPU HDR filters.

This tool intentionally processes still frames only.  It is a pre-batch review
artifact, not a Stage 0 normalization command.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oscardp.stage0.media import probe, video_filter


def _timestamps(duration: float, count: int) -> list[float]:
    # Avoid opening/closing credits while covering the whole feature evenly.
    start = min(60.0, duration * 0.05)
    end = max(start, duration - start)
    if count == 1 or end == start:
        return [round(duration / 2, 3)]
    return [round(start + (end - start) * index / (count - 1), 3) for index in range(count)]


def _frame_name(index: int, seconds: float) -> str:
    return f"frame_{index:02d}_{seconds:010.3f}s.png"


def _extract(video: Path, seconds: float, filter_graph: str, output: Path) -> None:
    command = [
        "ffmpeg", "-y", "-v", "error", "-ss", f"{seconds:.3f}", "-i", str(video),
        "-map", "0:v:0", "-an", "-vf", filter_graph, "-frames:v", "1", str(output),
    ]
    subprocess.run(command, text=True, capture_output=True, check=True)


def _metrics(path: Path) -> dict[str, float]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return {
        "mean_luma": round(float(luma.mean()), 6),
        "luma_stddev": round(float(luma.std()), 6),
        "luma_p01": round(float(np.percentile(luma, 1)), 6),
        "luma_p50": round(float(np.percentile(luma, 50)), 6),
        "luma_p99": round(float(np.percentile(luma, 99)), 6),
        "black_clip_fraction": round(float((luma <= 0.01).mean()), 6),
        "white_clip_fraction": round(float((luma >= 0.99).mean()), 6),
    }


def _contact_sheet(pairs: list[tuple[Path, Path, str]], output: Path) -> None:
    tile_width, tile_height, label_height = 480, 320, 28
    canvas = Image.new("RGB", (tile_width * 2, (tile_height + label_height) * len(pairs)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (reference, candidate, label) in enumerate(pairs):
        y = index * (tile_height + label_height)
        for x, image_path in ((0, reference), (tile_width, candidate)):
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((tile_width, tile_height))
            offset_x = x + (tile_width - image.width) // 2
            offset_y = y + (tile_height - image.height) // 2
            canvas.paste(image, (offset_x, offset_y))
        draw.text((4, y + tile_height + 5), f"A tonemap-first | B resize-first | {label}", fill="black")
    canvas.save(output, quality=95)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create paired HDR tone-map QC frames and diagnostics.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/g/datasets/oscar_movie_processed/stage0_hdr_qc"))
    parser.add_argument("--frame-count", type=int, default=12)
    parser.add_argument("--timestamps", nargs="*", type=float, help="Optional exact source timestamps in seconds")
    parser.add_argument("--clip-delta", type=float, default=0.01, help="Absolute clipping-fraction increase that flags review")
    parser.add_argument("--median-luma-delta", type=float, default=0.08, help="Absolute median-luma difference that flags review")
    args = parser.parse_args()
    if args.frame_count < 10:
        parser.error("--frame-count must be at least 10")
    source = args.video.resolve()
    info = probe(source)
    if info.dynamic_range != "HDR":
        parser.error("--video must be an HDR source")
    root = args.output_dir.resolve()
    reference_dir, candidate_dir, sheets_dir = root / "tonemap_first", root / "resize_first", root / "contact_sheets"
    for directory in (reference_dir, candidate_dir, sheets_dir):
        directory.mkdir(parents=True, exist_ok=True)
    timestamps = args.timestamps if args.timestamps else _timestamps(info.duration_sec, args.frame_count)
    if len(timestamps) < 10:
        parser.error("provide at least 10 timestamps")
    reference_filter = video_filter(info, hdr_filter_order="tonemap-first")
    candidate_filter = video_filter(info, hdr_filter_order="resize-first")
    frames: list[dict[str, object]] = []
    pairs: list[tuple[Path, Path, str]] = []
    for index, seconds in enumerate(timestamps, start=1):
        if not 0 <= seconds < info.duration_sec:
            parser.error(f"timestamp outside source duration: {seconds}")
        name = _frame_name(index, seconds)
        reference, candidate = reference_dir / name, candidate_dir / name
        _extract(source, seconds, reference_filter, reference)
        _extract(source, seconds, candidate_filter, candidate)
        baseline, optimized = _metrics(reference), _metrics(candidate)
        flags: list[str] = []
        if optimized["black_clip_fraction"] - baseline["black_clip_fraction"] >= args.clip_delta:
            flags.append("black_clip_increase")
        if optimized["white_clip_fraction"] - baseline["white_clip_fraction"] >= args.clip_delta:
            flags.append("white_clip_increase")
        if abs(optimized["luma_p50"] - baseline["luma_p50"]) >= args.median_luma_delta:
            flags.append("median_luma_shift")
        frames.append({"timestamp_sec": seconds, "tonemap_first": baseline, "resize_first": optimized, "manual_review_flags": flags, "reference_frame": reference.relative_to(root).as_posix(), "candidate_frame": candidate.relative_to(root).as_posix()})
        pairs.append((reference, candidate, f"{seconds:.3f}s"))
    sheet = sheets_dir / "paired_contact_sheet.jpg"
    _contact_sheet(pairs, sheet)
    report = {
        "source_video": str(source), "source_duration_sec": info.duration_sec,
        "source_fps": info.fps, "frame_count": len(frames), "timestamps_sec": timestamps,
        "reference_filter_order": "tonemap-first", "candidate_filter_order": "resize-first",
        "reference_filter": reference_filter, "candidate_filter": candidate_filter,
        "manual_review_thresholds": {"clip_delta": args.clip_delta, "median_luma_delta": args.median_luma_delta},
        "frames": frames, "contact_sheet": sheet.relative_to(root).as_posix(),
    }
    (root / "qc_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(root / "qc_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
