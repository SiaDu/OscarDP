from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Iterator
from itertools import pairwise
from pathlib import Path
from statistics import median

from .schema import FrameTimeline, VideoMetadata

TRANSNET_WIDTH = 48
TRANSNET_HEIGHT = 27
TRANSNET_FRAME_BYTES = TRANSNET_WIDTH * TRANSNET_HEIGHT * 3


class ExternalToolError(RuntimeError):
    pass


def require_media_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise ExternalToolError(f"Missing required system tools: {', '.join(missing)}")


def parse_fraction(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(value)


def _run_text(command: list[str]) -> str:
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode:
        raise ExternalToolError(process.stderr.strip() or f"Command failed: {command[0]}")
    return process.stdout


def probe_video(video_path: Path, source_video_relpath: str) -> tuple[VideoMetadata, dict[str, object]]:
    require_media_tools()
    output = _run_text([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration",
        "-of", "json", str(video_path),
    ])
    payload = json.loads(output)
    streams = payload.get("streams") or []
    if not streams:
        raise ExternalToolError(f"No video stream found: {video_path}")
    stream = streams[0]
    fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(stream.get("r_frame_rate"))
    if fps <= 0:
        raise ExternalToolError(f"Invalid video frame rate: {video_path}")
    raw_count = stream.get("nb_frames")
    metadata = VideoMetadata(
        source_video_relpath=source_video_relpath,
        duration_sec=float((payload.get("format") or {}).get("duration") or 0.0),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        codec_name=str(stream.get("codec_name") or ""),
        fps=fps,
        frame_count=int(raw_count) if raw_count not in (None, "", "N/A") else None,
        is_vfr=False,
        timestamp_source="frame_index_cfr",
    )
    return metadata, stream


def iter_frame_timestamps(
    video_path: Path,
    progress: Callable[[int], None] | None = None,
) -> Iterator[tuple[float, float | None]]:
    require_media_tools()
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time,pkt_duration_time",
        "-of", "compact=p=0:nk=0", str(video_path),
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout is not None
    frame_count = 0
    for line in process.stdout:
        fields: dict[str, str] = {}
        for item in line.strip().split("|"):
            key, separator, value = item.partition("=")
            if separator:
                fields[key] = value
        raw_pts = fields.get("best_effort_timestamp_time")
        if raw_pts in (None, "N/A"):
            continue
        raw_duration = fields.get("pkt_duration_time")
        duration = None if raw_duration in (None, "N/A") else float(raw_duration)
        frame_count += 1
        if progress is not None:
            progress(frame_count)
        yield float(raw_pts), duration
    stderr = process.stderr.read() if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise ExternalToolError(stderr.strip() or "ffprobe frame scan failed")


def build_timeline(
    raw_timestamps: list[tuple[float, float | None]],
    nominal_fps: float,
) -> FrameTimeline:
    if not raw_timestamps:
        raise ExternalToolError("ffprobe returned no decoded frame timestamps")
    first_pts = raw_timestamps[0][0]
    pts = [max(0.0, item[0] - first_pts) for item in raw_timestamps]
    durations = [item[1] if item[1] and item[1] > 0 else None for item in raw_timestamps]
    deltas = [right - left for left, right in pairwise(pts) if right > left]
    nominal_delta = 1.0 / nominal_fps
    tolerance = max(1e-6, nominal_delta * 0.001)
    matching = sum(abs(delta - nominal_delta) <= tolerance for delta in deltas)
    ratio = matching / len(deltas) if deltas else 1.0
    is_vfr = ratio < 0.999
    fallback_duration = median(deltas) if deltas else nominal_delta
    last_duration = durations[-1] or fallback_duration
    exclusive_end = pts[-1] + last_duration if is_vfr else len(pts) / nominal_fps
    return FrameTimeline(pts, durations, is_vfr, nominal_fps, exclusive_end)


def scan_timeline(
    video_path: Path,
    nominal_fps: float,
    progress: Callable[[int], None] | None = None,
) -> FrameTimeline:
    return build_timeline(list(iter_frame_timestamps(video_path, progress)), nominal_fps)


def decode_transnet_frames(video_path: Path) -> Iterator[bytes]:
    require_media_tools()
    command = [
        "ffmpeg", "-v", "error", "-i", str(video_path), "-map", "0:v:0",
        "-vf", f"scale={TRANSNET_WIDTH}:{TRANSNET_HEIGHT}", "-pix_fmt", "rgb24",
        "-f", "rawvideo", "-vsync", "0", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    while True:
        frame = process.stdout.read(TRANSNET_FRAME_BYTES)
        if not frame:
            break
        if len(frame) != TRANSNET_FRAME_BYTES:
            process.kill()
            raise ExternalToolError("FFmpeg returned a truncated raw frame")
        yield frame
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise ExternalToolError(stderr.strip() or "FFmpeg frame decode failed")


def _select_expression(frame_indices: list[int]) -> str:
    if not frame_indices:
        raise ValueError("At least one frame index is required")
    return "+".join(f"eq(n\\,{index})" for index in frame_indices)


def extract_selected_frames(
    video_path: Path,
    frame_indices: list[int],
    target_dir: Path,
    progress: Callable[[int], None] | None = None,
) -> list[Path]:
    require_media_tools()
    unique_indices = sorted(set(frame_indices))
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = target_dir / "frame_%08d.jpg"
    command = [
        "ffmpeg", "-y", "-v", "error", "-progress", "pipe:1", "-nostats",
        "-i", str(video_path), "-map", "0:v:0",
        "-vf", f"select={_select_expression(unique_indices)}", "-vsync", "0",
        "-q:v", "2", str(output_pattern),
    ]
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    for line in process.stdout:
        key, separator, value = line.strip().partition("=")
        if progress is not None and separator and key == "frame":
            try:
                progress(int(value))
            except ValueError:
                pass
    stderr = process.stderr.read() if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise ExternalToolError(stderr.strip() or "FFmpeg selected-frame extraction failed")
    outputs = sorted(target_dir.glob("frame_*.jpg"))
    if len(outputs) != len(unique_indices):
        raise ExternalToolError(
            f"Expected {len(unique_indices)} selected frames, FFmpeg wrote {len(outputs)}"
        )
    return outputs
