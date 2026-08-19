from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


GIB = 1024 ** 3
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


@dataclass(frozen=True)
class MediaInfo:
    container: str
    codec: str
    width: int
    height: int
    fps: float
    bit_depth: int
    dynamic_range: str
    size_gib: float
    duration_sec: float
    pixel_format: str
    color_primaries: str
    color_transfer: str
    color_space: str
    audio_streams: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["audio_streams"] = list(value["audio_streams"])
        return value


def _fraction(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator) if float(denominator) else 0.0


def _bit_depth(stream: dict[str, object]) -> int:
    raw = str(stream.get("bits_per_raw_sample") or "")
    if raw.isdigit():
        return int(raw)
    pixel_format = str(stream.get("pix_fmt") or "")
    for depth in (16, 14, 12, 10, 9, 8):
        if str(depth) in pixel_format:
            return depth
    return 8


def _is_hdr(stream: dict[str, object]) -> bool:
    if str(stream.get("color_transfer") or "").lower() in HDR_TRANSFERS:
        return True
    for item in stream.get("side_data_list") or []:
        if isinstance(item, dict) and item.get("side_data_type") in {
            "DOVI configuration record", "Mastering display metadata", "Content light level metadata",
        }:
            return True
    return False


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=True)


def probe(path: Path) -> MediaInfo:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=format_name,duration,size:stream=index,codec_type,codec_name,width,height,"
        "avg_frame_rate,r_frame_rate,bits_per_raw_sample,pix_fmt,color_transfer,"
        "color_primaries,color_space,disposition:stream_tags=language:stream_side_data",
        "-of", "json", str(path),
    ]
    payload = json.loads(run_checked(command).stdout)
    streams = payload.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if not isinstance(video, dict):
        raise ValueError(f"No video stream: {path}")
    audio = tuple(item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio")
    form = payload.get("format") or {}
    duration = float(form.get("duration") or 0.0)
    size = int(form.get("size") or path.stat().st_size)
    fps = _fraction(str(video.get("avg_frame_rate") or "")) or _fraction(str(video.get("r_frame_rate") or ""))
    return MediaInfo(
        container=str(form.get("format_name") or ""), codec=str(video.get("codec_name") or ""),
        width=int(video.get("width") or 0), height=int(video.get("height") or 0), fps=fps,
        bit_depth=_bit_depth(video), dynamic_range="HDR" if _is_hdr(video) else "SDR",
        size_gib=size / GIB, duration_sec=duration, pixel_format=str(video.get("pix_fmt") or ""),
        color_primaries=str(video.get("color_primaries") or ""),
        color_transfer=str(video.get("color_transfer") or ""),
        color_space=str(video.get("color_space") or ""),
        audio_streams=audio,
    )


def target_dimensions(info: MediaInfo) -> tuple[int, int]:
    factor = min(1.0, 1920 / info.width, 1080 / info.height)
    return max(2, int(info.width * factor) // 2 * 2), max(2, int(info.height * factor) // 2 * 2)


def transcode_reasons(info: MediaInfo, max_size_gib: float) -> list[str]:
    reasons: list[str] = []
    if info.size_gib > max_size_gib:
        reasons.append("TRANSCODE_SIZE")
    if info.width > 1920 or info.height > 1080:
        reasons.append("TRANSCODE_RESOLUTION")
    if info.dynamic_range != "SDR":
        reasons.append("TRANSCODE_HDR")
    if info.bit_depth > 8:
        reasons.append("TRANSCODE_BIT_DEPTH")
    return reasons


def classification(reasons: list[str]) -> str:
    if not reasons:
        return "KEEP"
    return reasons[0] if len(reasons) == 1 else "TRANSCODE_MULTIPLE"


def supports_hdr_tonemap() -> tuple[bool, str]:
    try:
        output = run_checked(["ffmpeg", "-hide_banner", "-filters"]).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return False, f"Cannot inspect FFmpeg filters: {exc}"
    missing = [name for name in ("zscale", "tonemap") if name not in output]
    return (not missing, "" if not missing else f"Missing FFmpeg filters: {', '.join(missing)}")


def select_audio_stream(info: MediaInfo) -> int | None:
    if not info.audio_streams:
        return None
    for stream in info.audio_streams:
        tags = stream.get("tags") if isinstance(stream.get("tags"), dict) else {}
        if str(tags.get("language") or "").lower() in {"eng", "en", "english"}:
            return int(stream["index"])
    for stream in info.audio_streams:
        disposition = stream.get("disposition") if isinstance(stream.get("disposition"), dict) else {}
        if int(disposition.get("default") or 0):
            return int(stream["index"])
    return int(info.audio_streams[0]["index"])


def video_filter(info: MediaInfo) -> str:
    width, height = target_dimensions(info)
    scale = f"scale={width}:{height}"
    if info.dynamic_range == "HDR":
        return (
            "zscale=t=linear:npl=100,format=gbrpf32le,tonemap=mobius:desat=0,"
            f"zscale=p=bt709:t=bt709:m=bt709:r=tv,{scale},format=yuv420p"
        )
    return f"{scale},format=yuv420p"
