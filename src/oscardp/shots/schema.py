from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON values must not contain NaN or Infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def json_dumps(value: Any, *, pretty: bool = False) -> str:
    kwargs: dict[str, Any] = {
        "ensure_ascii": False,
        "allow_nan": False,
        "sort_keys": False,
    }
    if pretty:
        kwargs.update(indent=2)
    return json.dumps(_json_safe(value), **kwargs)


@dataclass(frozen=True)
class MovieRef:
    movie_key: str
    source_path: Path
    source_video_relpath: str


@dataclass
class VideoMetadata:
    source_video_relpath: str
    duration_sec: float
    width: int
    height: int
    codec_name: str
    fps: float
    frame_count: int | None
    is_vfr: bool
    timestamp_source: str


@dataclass
class FrameTimeline:
    pts_sec: list[float]
    frame_duration_sec: list[float | None]
    is_vfr: bool
    nominal_fps: float
    exclusive_end_sec: float

    @property
    def frame_count(self) -> int:
        return len(self.pts_sec)

    def timestamp(self, frame_index: int) -> float:
        if not 0 <= frame_index < self.frame_count:
            raise IndexError(frame_index)
        if self.is_vfr:
            return self.pts_sec[frame_index]
        return frame_index / self.nominal_fps

    def exclusive_timestamp(self, frame_index: int) -> float:
        if frame_index == self.frame_count:
            return self.exclusive_end_sec
        return self.timestamp(frame_index)


@dataclass
class ShotRecord:
    movie_key: str
    shot_id: str
    source_video_relpath: str
    start_frame: int
    end_frame: int
    frame_count: int
    start_time: str
    end_time: str
    start_sec: float
    end_sec: float
    duration_sec: float
    keyframe_frame: int
    keyframe_time_sec: float
    keyframe_relpath: str
    boundary_before_confidence: float | None
    boundary_after_confidence: float | None
    shot_scale: None
    camera_movement: None
    model: dict[str, Any]


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str]
    shot_count: int
    missing_keyframes: int


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def rounded_seconds(value: float) -> float:
    return round(float(value), 6)
