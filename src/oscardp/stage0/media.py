from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


GIB = 1024 ** 3
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".webm", ".ts", ".m2ts"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt", ".sub", ".idx"}
TARGET_CODECS = {"h264", "hevc"}
TARGET_MAX_SIZE_GIB = 4.5


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
    subtitle_streams: tuple[dict[str, object], ...] = ()
    video_bitrate: int | None = None

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


def probe_payload(path: Path) -> dict[str, object]:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=format_name,duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,bit_rate,"
        "avg_frame_rate,r_frame_rate,bits_per_raw_sample,pix_fmt,color_transfer,"
        "color_primaries,color_space,disposition:stream_tags=language:stream_side_data",
        "-of", "json", str(path),
    ]
    return json.loads(run_checked(command).stdout)


def probe(path: Path) -> MediaInfo:
    payload = probe_payload(path)
    streams = payload.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if not isinstance(video, dict):
        raise ValueError(f"No video stream: {path}")
    audio = tuple(item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio")
    subtitles = tuple(item for item in streams if isinstance(item, dict) and item.get("codec_type") == "subtitle")
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
        subtitle_streams=subtitles,
        video_bitrate=int(video["bit_rate"]) if str(video.get("bit_rate") or "").isdigit() else None,
    )


def stream_language(stream: dict[str, object]) -> str:
    tags = stream.get("tags") if isinstance(stream.get("tags"), dict) else {}
    raw = str(tags.get("language") or "").strip().lower()
    return {"en": "en", "eng": "en", "english": "en", "und": "und"}.get(raw, raw or "und")


def is_english(language: str) -> bool:
    return language.lower() in {"en", "eng", "english"}


def video_bitrate(info: MediaInfo, path: Path) -> tuple[int | None, str]:
    if info.video_bitrate is not None:
        return info.video_bitrate, "video_stream"
    if info.duration_sec <= 0:
        return None, "unavailable"
    return round(path.stat().st_size * 8 / info.duration_sec), "container_average_estimate"


def target_dimensions(info: MediaInfo) -> tuple[int, int]:
    factor = min(1.0, 1920 / info.width, 1080 / info.height)
    return max(2, int(info.width * factor) // 2 * 2), max(2, int(info.height * factor) // 2 * 2)


def profile_reasons(info: MediaInfo, max_size_gib: float = TARGET_MAX_SIZE_GIB) -> list[str]:
    reasons: list[str] = []
    if info.dynamic_range != "SDR":
        reasons.append("HDR")
    if info.size_gib > max_size_gib:
        reasons.append(f"filesize exceeds {max_size_gib:g} GiB")
    if info.width > 1920 or info.height > 1080:
        reasons.append("resolution exceeds 1920x1080")
    if info.bit_depth > 8:
        reasons.append(f"bit depth is {info.bit_depth}-bit")
    if info.codec not in TARGET_CODECS:
        reasons.append(f"unsupported video codec: {info.codec or 'unknown'}")
    return reasons


def inventory_classification(info: MediaInfo, max_size_gib: float = TARGET_MAX_SIZE_GIB) -> str:
    if info.dynamic_range != "SDR":
        return "HDR_TONEMAP"
    if info.size_gib > max_size_gib:
        return "OVERSIZE"
    return "TRANSCODE" if profile_reasons(info, max_size_gib) else "PASS"


def transcode_reasons(info: MediaInfo, max_size_gib: float) -> list[str]:
    """Compatibility adapter for Stage 0C's existing report schema."""
    return profile_reasons(info, max_size_gib)


def classification(reasons: list[str]) -> str:
    """Compatibility adapter; Stage 0B should use inventory_classification."""
    if not reasons:
        return "KEEP"
    if "HDR" in reasons:
        return "HDR_TONEMAP"
    if any(reason.startswith("filesize") for reason in reasons):
        return "OVERSIZE"
    return "TRANSCODE"


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
