from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oscardp.shots.media import ExternalToolError, require_media_tools


@dataclass(frozen=True)
class FrameRequest:
    shot_id: str
    fraction: float
    frame_index: int
    time_sec: float
    output_path: Path


def sample_requests(shot: dict[str, Any], output_dir: Path) -> list[FrameRequest]:
    start_frame = int(shot["frame_range"]["start_frame"])
    end_frame = int(shot["frame_range"]["end_frame"])
    frame_count = end_frame - start_frame
    start_sec = float(shot["time"]["start_sec"])
    end_sec = float(shot["time"]["end_sec"])
    requests = []
    for fraction in (0.2, 0.5, 0.8):
        frame = min(end_frame - 1, start_frame + int((frame_count - 1) * fraction))
        timestamp = start_sec + (end_sec - start_sec) * fraction
        requests.append(FrameRequest(
            shot_id=shot["shot_id"], fraction=fraction, frame_index=frame,
            time_sec=timestamp,
            output_path=output_dir / shot["shot_id"] / f"sample_{int(fraction * 100):02d}.jpg",
        ))
    return requests


def _chunks(requests: list[FrameRequest], max_gap: float = 30.0, max_span: float = 60.0) -> list[list[FrameRequest]]:
    result: list[list[FrameRequest]] = []
    for request in sorted(requests, key=lambda item: item.time_sec):
        if not result or request.time_sec - result[-1][-1].time_sec > max_gap or request.time_sec - result[-1][0].time_sec > max_span:
            result.append([request])
        else:
            result[-1].append(request)
    return result


def extract_sparse_frames(video: Path, requests: list[FrameRequest]) -> None:
    require_media_tools()
    for request in requests:
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
    for chunk in _chunks(requests):
        seek = max(0.0, chunk[0].time_sec - 0.5)
        labels = []
        filters = (
            [f"[0:v]split={len(chunk)}" + "".join(f"[v{i}]" for i in range(len(chunk)))]
            if len(chunk) > 1 else ["[0:v]null[v0]"]
        )
        for index, request in enumerate(chunk):
            relative = max(0.0, request.time_sec - seek)
            label = f"out{index}"
            labels.append(label)
            filters.append(
                f"[v{index}]select='gte(t,{relative:.6f})',scale='min(640,iw)':-2[{label}]"
            )
        command = ["ffmpeg", "-y", "-v", "error", "-ss", f"{seek:.6f}", "-i", str(video), "-filter_complex", ";".join(filters)]
        for label, request in zip(labels, chunk, strict=True):
            command.extend(["-map", f"[{label}]", "-frames:v", "1", "-q:v", "3", str(request.output_path)])
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode:
            raise ExternalToolError(process.stderr.strip() or "FFmpeg sparse-frame extraction failed")
        missing = [str(request.output_path) for request in chunk if not request.output_path.is_file()]
        if missing:
            raise ExternalToolError(f"FFmpeg did not create sparse frames: {missing[:3]}")


class YuNetDetector:
    def __init__(self, model_path: Path, score_threshold: float = 0.8) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Stage 3 CV requires the 'performance' optional dependency (opencv-python-headless)") from exc
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("Installed OpenCV does not provide FaceDetectorYN")
        self.cv2 = cv2
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.detector = self.cv2.FaceDetectorYN.create(
            str(self.model_path), "", (320, 320), self.score_threshold, 0.3, 5000,
        )

    def analyze(self, image_path: Path) -> dict[str, Any]:
        image = self.cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read sampled frame: {image_path}")
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _status, faces = self.detector.detect(image)
        face_count = 0 if faces is None else len(faces)
        maximum_area = 0.0
        if faces is not None:
            maximum_area = max(float(face[2] * face[3]) / max(1.0, width * height) for face in faces)
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        luma = float(np.mean(gray))
        blur_variance = float(self.cv2.Laplacian(gray, self.cv2.CV_64F).var())
        too_dark = luma < 20.0
        too_bright = luma > 240.0
        too_blurry = blur_variance < 20.0
        usable = not (too_dark or too_bright or too_blurry)
        return {
            "face_count": face_count,
            "max_face_area_ratio": round(maximum_area, 6),
            "quality": {
                "mean_luma": round(luma, 6), "blur_variance": round(blur_variance, 6),
                "too_dark": too_dark, "too_bright": too_bright, "too_blurry": too_blurry,
                "usable": usable,
            },
        }


def analyze_shot_frames(
    detector: YuNetDetector, requests: list[FrameRequest], run_dir: Path,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for request in requests:
        result = detector.analyze(request.output_path)
        frames.append({
            "fraction": request.fraction, "frame": request.frame_index,
            "time_sec": round(request.time_sec, 6),
            "path": request.output_path.relative_to(run_dir).as_posix(), **result,
        })
    usable = [row for row in frames if row["quality"]["usable"]]
    face_frames = [row for row in usable if row["face_count"] > 0]
    usable_ratio = len(usable) / len(frames) if frames else 0.0
    visible_ratio = len(face_frames) / len(usable) if usable else 0.0
    maximum_area = max((float(row["max_face_area_ratio"]) for row in face_frames), default=0.0)
    size_score = min(1.0, maximum_area / 0.10)
    stability = min(1.0, len(face_frames) / 2.0)
    face_score = 0.50 * visible_ratio + 0.25 * size_score + 0.15 * usable_ratio + 0.10 * stability
    aggregate = {
        "sampled_frame_count": len(frames), "usable_frame_count": len(usable),
        "face_visible_frame_count": len(face_frames), "usable_frame_ratio": round(usable_ratio, 6),
        "face_visible_frame_ratio": round(visible_ratio, 6),
        "max_face_area_ratio": round(maximum_area, 6), "stability": round(stability, 6),
    }
    return frames, round(min(1.0, face_score), 6), aggregate
