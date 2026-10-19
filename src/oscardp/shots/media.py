from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import os
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
        "-show_entries", "frame=media_type,best_effort_timestamp_time,pkt_duration_time",
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
        # FFprobe may emit blank/side-data-only records (not decoded frames),
        # notably H.264 SEI. Identify frames explicitly before requiring PTS.
        if fields.get("media_type") != "video":
            continue
        raw_pts = fields.get("best_effort_timestamp_time")
        if raw_pts in (None, "N/A"):
            process.kill()
            process.wait()
            raise ExternalToolError("Decoded frame timing unavailable (missing PTS); refusing to shift frame indices")
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


def extract_selected_frames(
    video_path: Path,
    frame_indices: list[int],
    target_dir: Path,
    progress: Callable[[int], None] | None = None,
) -> list[Path]:
    """One sequential decode, bounded memory, JPEG encoding only for selected frames.

    Fixed-size RGB avoids large select expressions and linear filter evaluation.
    The tradeoff is full-resolution pipe bandwidth. Per-frame cache commits allow
    interruption recovery; a complete cache requires no FFmpeg invocation.
    """
    from PIL import Image
    from .frame_samples import atomic_json, image_info, load_json, sync_dir, safe_path, rename_noreplace, append_receipt, read_receipts

    indices = sorted(set(frame_indices))
    if not indices or any(type(n) is not int or n < 0 for n in indices):
        raise ValueError("Nonnegative selected frame indices required")
    require_media_tools()
    stat = video_path.stat()
    identity = {"path": str(video_path.resolve()), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns}
    safe_path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_path = target_dir / "extraction.json"
    safe_path(target_dir, cache_path.name)
    cache = load_json(cache_path) if cache_path.exists() else {"source": identity, "frames": {}}
    if cache.get("source") != identity:
        raise ExternalToolError("Extraction cache belongs to a different video")
    started = set()
    for receipt in read_receipts(target_dir / "frames.jsonl"):
        if receipt.get("status") == "encoding":
            started.add(receipt["frame_index"])
            continue
        key = str(receipt["frame_index"])
        if key in cache["frames"] and cache["frames"][key] != receipt["sha256"]:
            raise ExternalToolError("Conflicting frame extraction receipts")
        cache["frames"][key] = receipt["sha256"]
    if not cache_path.exists():
        atomic_json(cache_path, {"source": identity, "frames": {}})
    paths = {n: target_dir / f"frame_{n:08d}.jpg" for n in indices}
    for n, path in paths.items():
        safe_path(target_dir, path.name)
        tmp = safe_path(target_dir, path.with_suffix(".tmp").name)
        if not path.exists() and tmp.exists() and str(n) not in cache["frames"] and n in started:
            # An fsynced intent owns this incomplete encode; re-decode this frame.
            tmp.unlink()
        if not path.exists() and tmp.exists():
            if cache["frames"].get(str(n)) != image_info(tmp)["sha256"]:
                raise ExternalToolError(f"Unverified interrupted temporary image: {tmp}")
            rename_noreplace(tmp, path)
            sync_dir(target_dir)
        if path.exists() and cache["frames"].get(str(n)) != image_info(path)["sha256"]:
            raise ExternalToolError(f"Unverified/conflicting extraction cache: {path}")
    pending = {n for n, path in paths.items() if not path.exists()}
    if not pending:
        if progress:
            progress(len(indices))
        return list(paths.values())
    metadata, _ = probe_video(video_path, video_path.name)
    frame_bytes = metadata.width * metadata.height * 3
    if frame_bytes <= 0:
        raise ExternalToolError("Invalid decoded frame dimensions")
    command = ["ffmpeg", "-v", "error", "-noautorotate", "-i", str(video_path),
               "-map", "0:v:0", "-vsync", "0", "-pix_fmt", "rgb24",
               "-frames:v", str(max(pending) + 1), "-f", "rawvideo", "pipe:1"]
    completed = len(indices) - len(pending)
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        assert process.stdout is not None
        try:
            for n in range(max(pending) + 1):
                raw = process.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    raise ExternalToolError(f"Incomplete decode at exact frame {n}")
                if n not in pending:
                    continue
                path = paths[n]
                tmp = path.with_suffix(".tmp")
                append_receipt(target_dir / "frames.jsonl", {"frame_index": n, "status": "encoding"})
                with tmp.open("xb") as stream:
                    Image.frombytes("RGB", (metadata.width, metadata.height), raw).save(
                        stream, format="JPEG", quality=95)
                    stream.flush()
                    os.fsync(stream.fileno())
                digest = image_info(tmp)["sha256"]
                cache["frames"][str(n)] = digest
                append_receipt(target_dir / "frames.jsonl", {"frame_index": n, "sha256": digest})
                rename_noreplace(tmp, path)
                sync_dir(target_dir)
                completed += 1
                if progress:
                    progress(completed)
            process.stdout.close()
            code = process.wait()
            if code:
                errors.seek(0)
                raise ExternalToolError(errors.read().decode(errors="replace"))
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()
    return list(paths.values())
