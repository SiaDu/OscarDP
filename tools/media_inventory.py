#!/usr/bin/env python3
"""Create a reproducible ffprobe inventory for the Oscar movie source tree."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".webm"}
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


def probe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size,bit_rate:stream=index,codec_type,codec_name,"
        "width,height,avg_frame_rate,r_frame_rate,bit_rate,bits_per_raw_sample,"
        "pix_fmt,color_transfer,color_primaries,color_space,side_data_list",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator) if float(denominator) else None


def bit_depth(stream: dict) -> int | None:
    raw = stream.get("bits_per_raw_sample")
    if raw and raw.isdigit():
        return int(raw)
    pixel_format = stream.get("pix_fmt", "")
    for depth in (16, 14, 12, 10, 9, 8):
        if str(depth) in pixel_format:
            return depth
    return 8 if pixel_format else None


def is_hdr(stream: dict) -> bool:
    if stream.get("color_transfer", "").lower() in HDR_TRANSFERS:
        return True
    return any(
        side_data.get("side_data_type") in {
            "DOVI configuration record", "Mastering display metadata", "Content light level metadata"
        }
        for side_data in stream.get("side_data_list", [])
    )


def decimal(value: float | int | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def classify(video: dict, size_bytes: int, fps: float | None, hdr: bool, depth: int | None, bitrate: int | None) -> tuple[str, str]:
    # Default delivery profile: SDR 1080p or lower, H.264/H.265 8-bit, <=30 fps,
    # <=10 Mbps video bitrate, <=20 GiB file size. First matching class wins.
    reasons: list[str] = []
    if hdr:
        reasons.append("HDR metadata detected")
    if size_bytes > 20 * 1024**3:
        reasons.append("filesize exceeds 20 GiB")
    if (video.get("width") or 0) > 1920 or (video.get("height") or 0) > 1080:
        reasons.append("resolution exceeds 1920x1080")
    if video.get("codec_name") not in {"h264", "hevc"}:
        reasons.append(f"unsupported video codec: {video.get('codec_name') or 'unknown'}")
    if depth not in {None, 8}:
        reasons.append(f"bit depth is {depth}-bit")
    if fps is not None and fps > 30.01:
        reasons.append(f"fps exceeds 30: {fps:.3f}")
    if bitrate is not None and bitrate > 10_000_000:
        reasons.append(f"video bitrate exceeds 10 Mbps: {bitrate / 1_000_000:.3f}")
    if hdr:
        return "HDR_TONEMAP", "; ".join(reasons)
    if size_bytes > 20 * 1024**3:
        return "OVERSIZE", "; ".join(reasons)
    return ("TRANSCODE", "; ".join(reasons)) if reasons else ("PASS", "meets default delivery profile")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_root.resolve()
    paths = sorted(path for path in root.rglob("*") if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts) and path.suffix.lower() in VIDEO_EXTENSIONS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_video_relpath", "resolution", "fps", "video_codec", "bit_depth", "dynamic_range", "duration_sec", "video_bitrate_mbps", "video_bitrate_source", "audio_codecs", "audio_tracks", "filesize_bytes", "filesize_gib", "classification", "classification_reason", "ffprobe_error"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, path in enumerate(paths, 1):
            row = {"source_video_relpath": path.relative_to(root).as_posix()}
            try:
                data = probe(path)
                streams = data.get("streams", [])
                video = next(stream for stream in streams if stream.get("codec_type") == "video")
                audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
                size = int(data.get("format", {}).get("size", path.stat().st_size))
                fps = fraction(video.get("avg_frame_rate")) or fraction(video.get("r_frame_rate"))
                depth = bit_depth(video)
                hdr = is_hdr(video)
                duration = float(data.get("format", {}).get("duration", 0))
                stream_bitrate = int(video["bit_rate"]) if str(video.get("bit_rate", "")).isdigit() else None
                # Matroska often omits per-stream bit rates.  Its container
                # average is retained as a clearly labelled fallback instead
                # of leaving the requested bitrate column blank.
                bitrate = stream_bitrate or (round(size * 8 / duration) if duration > 0 else None)
                bitrate_source = "video_stream" if stream_bitrate else "container_average_estimate"
                label, reason = classify(video, size, fps, hdr, depth, bitrate)
                row.update({
                    "resolution": f"{video.get('width', '')}x{video.get('height', '')}",
                    "fps": decimal(fps), "video_codec": video.get("codec_name", ""),
                    "bit_depth": depth or "", "dynamic_range": "HDR" if hdr else "SDR",
                    "duration_sec": decimal(duration),
                    "video_bitrate_mbps": decimal((bitrate / 1_000_000) if bitrate else None),
                    "video_bitrate_source": bitrate_source if bitrate else "unavailable",
                    "audio_codecs": ";".join(stream.get("codec_name", "unknown") for stream in audio),
                    "audio_tracks": len(audio), "filesize_bytes": size,
                    "filesize_gib": decimal(size / 1024**3), "classification": label,
                    "classification_reason": reason,
                })
            except (subprocess.CalledProcessError, StopIteration, json.JSONDecodeError, OSError) as error:
                row.update({"classification": "TRANSCODE", "classification_reason": "ffprobe failed or no video stream", "ffprobe_error": str(error)})
            writer.writerow(row)
            print(f"[{position}/{len(paths)}] {row['source_video_relpath']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
