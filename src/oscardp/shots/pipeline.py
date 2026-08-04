from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from .discovery import derive_movie_key
from .media import (
    decode_transnet_frames,
    extract_selected_frames,
    probe_video,
    scan_timeline,
)
from .progress import ProgressReporter
from .qc import build_contact_sheet, build_qc_summary, select_qc_boundaries
from .schema import (
    FrameTimeline,
    MovieRef,
    ShotRecord,
    VideoMetadata,
    format_timestamp,
    json_dumps,
    rounded_seconds,
)
from .transnet import (
    Boundary,
    TransNetRunner,
    infer_stream,
    predictions_to_boundaries,
    sha256_file,
)
from .validation import validate_movie


@dataclass(frozen=True)
class ProcessOptions:
    input_root: Path
    output_root: Path
    weights: Path
    threshold: float = 0.5
    device: str = "auto"
    resume: bool = True
    overwrite: bool = False
    dry_run: bool = False
    save_all_boundary_frames: bool = False
    save_raw_predictions: bool = False
    progress: bool = True


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_text_atomic(path, json_dumps(value, pretty=True) + "\n")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _make_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"oscardp.{log_path.parent.parent.name}.{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _fingerprint(movie: MovieRef, options: ProcessOptions, model_sha256: str) -> dict[str, Any]:
    stat = movie.source_path.stat()
    return {
        "source_path": str(movie.source_path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "threshold": options.threshold,
        "model_sha256": model_sha256,
    }


def _timeline_to_json(timeline: FrameTimeline) -> dict[str, Any]:
    return {
        "pts_sec": timeline.pts_sec,
        "frame_duration_sec": timeline.frame_duration_sec,
        "is_vfr": timeline.is_vfr,
        "nominal_fps": timeline.nominal_fps,
        "exclusive_end_sec": timeline.exclusive_end_sec,
    }


def _timeline_from_json(value: dict[str, Any]) -> FrameTimeline:
    return FrameTimeline(
        pts_sec=[float(item) for item in value["pts_sec"]],
        frame_duration_sec=[None if item is None else float(item) for item in value["frame_duration_sec"]],
        is_vfr=bool(value["is_vfr"]),
        nominal_fps=float(value["nominal_fps"]),
        exclusive_end_sec=float(value["exclusive_end_sec"]),
    )


def build_shot_records(
    movie: MovieRef,
    metadata: VideoMetadata,
    timeline: FrameTimeline,
    boundaries: list[Boundary],
    threshold: float,
) -> list[ShotRecord]:
    if metadata.frame_count is None or metadata.frame_count < 1:
        raise ValueError("Decoded frame_count must be known before building shots")
    boundary_by_frame = {boundary.frame: boundary.confidence for boundary in boundaries}
    endpoints = [0] + [boundary.frame for boundary in boundaries] + [metadata.frame_count]
    shots: list[ShotRecord] = []
    for index, (start, end) in enumerate(pairwise(endpoints), 1):
        if start >= end:
            continue
        start_sec = timeline.timestamp(start)
        end_sec = timeline.exclusive_timestamp(end)
        keyframe = (start + end - 1) // 2
        shot_id = f"shot_{index:06d}"
        shots.append(
            ShotRecord(
                movie_key=movie.movie_key,
                shot_id=shot_id,
                source_video_relpath=movie.source_video_relpath,
                start_frame=start,
                end_frame=end,
                frame_count=end - start,
                start_time=format_timestamp(start_sec),
                end_time=format_timestamp(end_sec),
                start_sec=rounded_seconds(start_sec),
                end_sec=rounded_seconds(end_sec),
                duration_sec=rounded_seconds(end_sec - start_sec),
                keyframe_frame=keyframe,
                keyframe_time_sec=rounded_seconds(timeline.timestamp(keyframe)),
                keyframe_relpath=f"keyframes/{shot_id}.jpg",
                boundary_before_confidence=(
                    rounded_seconds(boundary_by_frame[start]) if start in boundary_by_frame else None
                ),
                boundary_after_confidence=(
                    rounded_seconds(boundary_by_frame[end]) if end in boundary_by_frame else None
                ),
                shot_scale=None,
                camera_movement=None,
                model={"name": "TransNetV2", "threshold": float(threshold)},
            )
        )
    return shots


def _write_shots(path: Path, shots: list[ShotRecord]) -> None:
    content = "".join(json_dumps(shot) + "\n" for shot in shots)
    _write_text_atomic(path, content)


def _update_index(output_root: Path, row: dict[str, Any]) -> None:
    index_path = output_root / "index.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing = json.loads(line)
                    rows[str(existing["movie_key"])] = existing
    rows[str(row["movie_key"])] = row
    content = "".join(json_dumps(rows[key]) + "\n" for key in sorted(rows))
    _write_text_atomic(index_path, content)


def _backup_path(output_root: Path, movie_key: str, suffix: str = "") -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return output_root / ".backup" / f"{movie_key}-{stamp}{suffix}"


def _prepare_work_dir(work_dir: Path, expected_state: dict[str, Any], overwrite: bool) -> None:
    if not work_dir.exists():
        work_dir.mkdir(parents=True)
        _write_json(work_dir / "state.json", {"fingerprint": expected_state, "phase": "started"})
        return
    state_path = work_dir / "state.json"
    existing = _read_json(state_path) if state_path.is_file() else None
    if existing and existing.get("fingerprint") == expected_state:
        return
    if not overwrite:
        raise RuntimeError("Existing work state does not match this input/options; use --overwrite")
    backup = _backup_path(work_dir.parent.parent, work_dir.name, "-work")
    backup.parent.mkdir(parents=True, exist_ok=True)
    work_dir.replace(backup)
    work_dir.mkdir(parents=True)
    _write_json(work_dir / "state.json", {"fingerprint": expected_state, "phase": "started"})


def _load_or_run_inference(
    movie: MovieRef,
    options: ProcessOptions,
    work_dir: Path,
    model_sha256: str,
    logger: logging.Logger,
    progress: ProgressReporter,
) -> tuple[VideoMetadata, FrameTimeline, list[float]]:
    resume_dir = work_dir / ".resume"
    metadata_cache = resume_dir / "metadata.json"
    timeline_cache = resume_dir / "timeline.json"
    predictions_cache = resume_dir / "predictions.npy"
    if metadata_cache.is_file() and timeline_cache.is_file() and predictions_cache.is_file():
        import numpy as np

        logger.info("Resuming cached metadata, timeline, and predictions")
        progress.stage("[1/5] Resume inference cache")
        metadata = VideoMetadata(**_read_json(metadata_cache))
        timeline = _timeline_from_json(_read_json(timeline_cache))
        predictions = [float(value) for value in np.load(predictions_cache, allow_pickle=False)]
        progress.finish(f"{len(predictions):,} frames cached")
        return metadata, timeline, predictions

    progress.stage("[1/5] Probe video metadata")
    metadata, _ = probe_video(movie.source_path, movie.source_video_relpath)
    progress.finish(
        f"{metadata.width}x{metadata.height} @ {metadata.fps:.3f} fps"
    )
    logger.info("Scanning decoded frame timestamps")
    estimated_count = metadata.frame_count
    estimated = False
    if estimated_count is None and metadata.duration_sec > 0:
        estimated_count = round(metadata.duration_sec * metadata.fps)
        estimated = True
    progress.stage("[2/5] Scan decoded timestamps", estimated_count, estimated=estimated)
    timeline = scan_timeline(movie.source_path, metadata.fps, progress.update)
    progress.finish(f"{timeline.frame_count:,} decoded frames")
    logger.info("Loading TransNetV2 model sha256=%s", model_sha256)
    progress.stage("[3/5] Load TransNetV2 model")
    runner = TransNetRunner.load(options.weights, options.device, model_sha256)
    progress.finish(f"device={runner.device}")
    import torch

    gpu_name = None
    gpu_capability = None
    if runner.device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(runner.device)
        gpu_capability = torch.cuda.get_device_capability(runner.device)
    logger.info(
        "Runtime python=%s torch=%s cuda_runtime=%s requested_device=%s "
        "selected_device=%s gpu_name=%s gpu_capability=%s",
        sys.executable,
        torch.__version__,
        torch.version.cuda,
        options.device,
        runner.device,
        gpu_name,
        gpu_capability,
    )
    logger.info(
        "Model weights=%s sha256=%s model_device=%s model_load_sec=%.6f",
        options.weights.resolve(),
        runner.model_sha256,
        runner.model_device,
        runner.model_load_sec,
    )
    progress.stage("[4/5] TransNetV2 inference", timeline.frame_count)
    runner.reset_inference_metrics()
    inference_stage_started = time.perf_counter()
    predictions = infer_stream(
        decode_transnet_frames(movie.source_path), runner, progress.update
    )
    inference_stage_sec = time.perf_counter() - inference_stage_started
    peak_memory_bytes = runner.peak_memory_allocated()
    logger.info(
        "Inference input_device=%s model_inference_sec=%.6f "
        "inference_stage_sec=%.6f peak_memory_allocated_bytes=%d",
        runner.input_device,
        runner.model_inference_sec,
        inference_stage_sec,
        peak_memory_bytes,
    )
    progress.finish(f"{len(predictions):,} predictions")
    if len(predictions) != timeline.frame_count:
        raise RuntimeError(
            f"PTS count ({timeline.frame_count}) differs from inference decode count ({len(predictions)})"
        )
    metadata.frame_count = len(predictions)
    metadata.is_vfr = timeline.is_vfr
    metadata.timestamp_source = "decoded_pts_vfr" if timeline.is_vfr else "frame_index_cfr"
    metadata.duration_sec = rounded_seconds(timeline.exclusive_end_sec)
    resume_dir.mkdir(parents=True, exist_ok=True)
    _write_json(metadata_cache, dataclasses.asdict(metadata))
    _write_json(timeline_cache, _timeline_to_json(timeline))
    import numpy as np

    np.save(predictions_cache, np.asarray(predictions, dtype=np.float32), allow_pickle=False)
    _write_json(work_dir / "state.json", {
        "fingerprint": _fingerprint(movie, options, model_sha256),
        "phase": "inference_complete",
    })
    return metadata, timeline, predictions


def _materialize_frames(
    movie: MovieRef,
    work_dir: Path,
    shots: list[ShotRecord],
    boundaries: list[Boundary],
    selected_qc: list[Boundary],
    progress: ProgressReporter,
) -> list[tuple[Path, Path, str]]:
    keyframes_dir = work_dir / "keyframes"
    boundaries_dir = work_dir / "qc" / "boundaries"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    boundaries_dir.mkdir(parents=True, exist_ok=True)
    needed = [shot.keyframe_frame for shot in shots]
    for boundary in selected_qc:
        needed.extend([boundary.frame - 1, boundary.frame])
    extracted_dir = work_dir / ".selected_frames"
    outputs = extract_selected_frames(
        movie.source_path, needed, extracted_dir, progress=progress.update
    )
    frame_map = dict(zip(sorted(set(needed)), outputs))
    for shot in shots:
        shutil.copy2(frame_map[shot.keyframe_frame], work_dir / shot.keyframe_relpath)
    boundary_ordinals = {boundary.frame: index for index, boundary in enumerate(boundaries, 1)}
    pairs: list[tuple[Path, Path, str]] = []
    for boundary in selected_qc:
        ordinal = boundary_ordinals[boundary.frame]
        label = f"boundary_{ordinal:06d}"
        before = boundaries_dir / f"{label}_before.jpg"
        after = boundaries_dir / f"{label}_after.jpg"
        shutil.copy2(frame_map[boundary.frame - 1], before)
        shutil.copy2(frame_map[boundary.frame], after)
        pairs.append((before, after, label))
    shutil.rmtree(extracted_dir)
    return pairs


def process_one(video_path: Path, options: ProcessOptions) -> dict[str, Any]:
    pipeline_started = time.perf_counter()
    progress = ProgressReporter(enabled=options.progress)
    root = options.input_root.resolve()
    video = video_path.resolve()
    relative = video.relative_to(root)
    movie = MovieRef(derive_movie_key(video, root), video, relative.as_posix())
    final_dir = options.output_root.resolve() / movie.movie_key
    if options.dry_run:
        return {
            "movie_key": movie.movie_key,
            "source_video_relpath": movie.source_video_relpath,
            "status": "dry_run",
            "output_relpath": movie.movie_key,
        }
    if not 0.0 < options.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if options.overwrite and options.resume:
        options = dataclasses.replace(options, resume=False)
    if final_dir.is_dir() and not options.overwrite:
        progress.stage("[1/1] Validate existing output")
        existing = validate_movie(final_dir)
        if existing.passed:
            row = {
                "movie_key": movie.movie_key,
                "source_video_relpath": movie.source_video_relpath,
                "status": "completed",
                "output_relpath": movie.movie_key,
                "shot_count": existing.shot_count,
                "validation_passed": True,
                "error": None,
                "resumed": True,
            }
            index_row = {key: value for key, value in row.items() if key != "resumed"}
            _update_index(options.output_root.resolve(), index_row)
            progress.finish(f"{existing.shot_count:,} shots; already complete")
            return row
        raise RuntimeError("Existing output is incomplete or invalid; use --overwrite")

    options.output_root.mkdir(parents=True, exist_ok=True)
    model_sha256 = sha256_file(options.weights)
    expected_state = _fingerprint(movie, options, model_sha256)
    work_dir = options.output_root.resolve() / ".work" / movie.movie_key
    _prepare_work_dir(work_dir, expected_state, options.overwrite)
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger = _make_logger(work_dir / "logs" / "process.log")
    backup: Path | None = None
    try:
        logger.info("Processing %s", movie.source_video_relpath)
        metadata, timeline, predictions = _load_or_run_inference(
            movie, options, work_dir, model_sha256, logger, progress
        )
        boundaries = predictions_to_boundaries(predictions, options.threshold)
        shots = build_shot_records(movie, metadata, timeline, boundaries, options.threshold)
        logger.info(
            "Detection transitions=%d shots=%d threshold=%.6f",
            len(boundaries),
            len(shots),
            options.threshold,
        )
        _write_json(work_dir / "video_metadata.json", metadata)
        _write_shots(work_dir / "shots.jsonl", shots)
        selected_qc = select_qc_boundaries(
            boundaries,
            shots,
            threshold=options.threshold,
            save_all=options.save_all_boundary_frames,
        )
        needed_count = len({shot.keyframe_frame for shot in shots} | {
            frame
            for boundary in selected_qc
            for frame in (boundary.frame - 1, boundary.frame)
        })
        progress.stage("[5/5] Extract keyframes and QC", needed_count)
        pairs = _materialize_frames(
            movie, work_dir, shots, boundaries, selected_qc, progress
        )
        progress.update(needed_count)
        progress.finish(f"{len(shots):,} shots; validating and publishing")
        build_contact_sheet(pairs, work_dir / "qc" / "boundary_contact_sheet.jpg")
        if options.save_raw_predictions:
            import numpy as np

            debug_dir = work_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                debug_dir / "raw_predictions.npz",
                single_frame_predictions=np.asarray(predictions, dtype=np.float32),
            )
        validation = validate_movie(work_dir)
        summary = build_qc_summary(
            shots,
            missing_keyframes=validation.missing_keyframes,
            validation_passed=validation.passed,
        )
        _write_json(work_dir / "qc" / "qc_summary.json", summary)
        if not validation.passed:
            raise RuntimeError("Validation failed: " + "; ".join(validation.errors))
        logger.info("Validation passed with %d shots", len(shots))
        _close_logger(logger)
        resume_dir = work_dir / ".resume"
        if resume_dir.exists():
            shutil.rmtree(resume_dir)
        state_path = work_dir / "state.json"
        if state_path.exists():
            state_path.unlink()
        if final_dir.exists():
            backup = _backup_path(options.output_root.resolve(), movie.movie_key)
            backup.parent.mkdir(parents=True, exist_ok=True)
            final_dir.replace(backup)
        try:
            work_dir.replace(final_dir)
        except Exception:
            if backup is not None and backup.exists() and not final_dir.exists():
                backup.replace(final_dir)
            raise
        row = {
            "movie_key": movie.movie_key,
            "source_video_relpath": movie.source_video_relpath,
            "status": "completed",
            "output_relpath": movie.movie_key,
            "shot_count": len(shots),
            "validation_passed": True,
            "error": None,
        }
        try:
            _update_index(options.output_root.resolve(), row)
        except Exception:
            final_dir.replace(work_dir)
            if backup is not None and backup.exists():
                backup.replace(final_dir)
            raise
        total_sec = time.perf_counter() - pipeline_started
        published_logger = _make_logger(final_dir / "logs" / "process.log")
        published_logger.info("Pipeline total_sec=%.6f", total_sec)
        _close_logger(published_logger)
        progress.stage("Complete")
        progress.finish(f"{len(shots):,} shots published to {final_dir}")
        return row
    except Exception as exc:
        logger.info("Pipeline failed total_sec=%.6f", time.perf_counter() - pipeline_started)
        progress.fail(str(exc))
        if logger.handlers:
            logger.exception("Processing failed")
            _close_logger(logger)
        failure = {
            "movie_key": movie.movie_key,
            "source_video_relpath": movie.source_video_relpath,
            "status": "failed",
            "output_relpath": None,
            "shot_count": 0,
            "validation_passed": False,
            "error": str(exc),
        }
        try:
            _update_index(options.output_root.resolve(), failure)
        except Exception as index_error:  # noqa: BLE001 - preserve the original processing error
            log_path = work_dir / "logs" / "process.log"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"Failed to update failure index: {index_error}\n")
        raise
