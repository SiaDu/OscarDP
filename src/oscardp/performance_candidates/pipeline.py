from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .cv import (
    YuNetDetector,
    analyze_shot_frames,
    extract_sparse_frames,
    sample_requests,
)
from .grouping import group_events
from .io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from .schema import PIPELINE_VERSION, RULESET_VERSION, SCHEMA_VERSION, MiningOptions
from .semantic import mine_shot_semantics

DetectorFactory = Callable[[Path], YuNetDetector]


def _verify(path: Path, expected: str | None, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if expected and actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {label}: {path}")
    return {"path": path.resolve().as_posix(), "sha256": actual, "size": path.stat().st_size}


def _artifact(movie: dict[str, Any], group: str, name: str) -> tuple[Path, str | None]:
    value = movie.get(group, {}).get(name)
    if not isinstance(value, dict) or not value.get("path"):
        raise ValueError(f"Release movie lacks {group}.{name}")
    return Path(value["path"]), value.get("sha256")


def _verify_video_artifact(movie: dict[str, Any]) -> dict[str, Any]:
    """Reuse the frozen release's completed video hash plus stat fingerprint.

    Stage 2 deliberately avoids re-hashing multi-gigabyte protected videos at
    release-resume time.  Stage 3 consumes that same immutable release contract
    rather than making every pilot read an entire source movie before CV begins.
    Older synthetic fixtures lacking a fingerprint retain full SHA verification.
    """
    value = movie.get("protected_artifacts", {}).get("video")
    if not isinstance(value, dict) or not value.get("path") or not value.get("sha256"):
        raise ValueError("Release movie lacks protected_artifacts.video")
    path = Path(value["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing video: {path}")
    stat = path.stat()
    current = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    recorded = value.get("stat_fingerprint")
    if recorded is None:
        return _verify(path, value["sha256"], "video")
    if current != recorded:
        raise RuntimeError(f"Protected video stat fingerprint changed: {path}")
    return {
        "path": path.resolve().as_posix(), "sha256": value["sha256"], "size": stat.st_size,
        "stat_fingerprint": current,
        "verification_method": "frozen_release_sha256_plus_stat_fingerprint",
    }


def _release_inputs(options: MiningOptions) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    release_info = _verify(options.release_manifest, None, "release manifest")
    release = read_json(options.release_manifest)
    if release.get("status") != "FROZEN":
        raise RuntimeError("Stage 3 requires a FROZEN Stage 2 release")
    matches = [row for row in release.get("movies", []) if row.get("movie_id") == options.movie_key]
    if len(matches) != 1:
        raise ValueError(f"Movie is not a unique completed release member: {options.movie_key}")
    movie = matches[0]
    if movie.get("status") != "COMPLETE":
        raise RuntimeError(f"Stage 3 refuses non-COMPLETE movie: {options.movie_key}")

    paths = {
        "reviewed_shot_context": _artifact(movie, "artifacts", "reviewed_shot_context"),
        "screenplay_context": _artifact(movie, "protected_artifacts", "deterministic_screenplay_context"),
        "shots": _artifact(movie, "protected_artifacts", "shots"),
    }
    inputs: dict[str, Any] = {"release_manifest": release_info}
    for label, (path, expected) in paths.items():
        inputs[label] = _verify(path, expected, label)
    inputs["video"] = _verify_video_artifact(movie)
    inputs["face_model"] = _verify(options.face_model, options.face_model_sha256, "YuNet face model")
    return release, movie, inputs


def _pending_subtitle_ids(release: dict[str, Any], movie_key: str) -> set[str]:
    artifact = release.get("pending_human_ambiguities")
    if not isinstance(artifact, dict) or not artifact.get("path"):
        return set()
    path = Path(artifact["path"])
    _verify(path, artifact.get("sha256"), "pending human ambiguities")
    return {
        str(row["subtitle_id"])
        for row in read_jsonl(path)
        if row.get("movie_id") == movie_key and row.get("subtitle_id")
    }


def _validate_source_pairing(context_rows: list[dict[str, Any]], shots: list[dict[str, Any]]) -> None:
    if len(context_rows) != len(shots):
        raise RuntimeError("Reviewed shot context and shots.jsonl have different row counts")
    for index, (context, shot) in enumerate(zip(context_rows, shots, strict=True)):
        if context.get("shot_id") != shot.get("shot_id"):
            raise RuntimeError(f"Shot ID mismatch at source row {index}")
        frame_range = context.get("frame_range") or {}
        if (frame_range.get("start_frame"), frame_range.get("end_frame")) != (shot.get("start_frame"), shot.get("end_frame")):
            raise RuntimeError(f"Frame range mismatch for {shot.get('shot_id')}")


def _fingerprint(options: MiningOptions, release: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "ruleset_version": RULESET_VERSION,
        "release_id": release["release_id"],
        "movie_id": options.movie_key,
        "inputs": inputs,
        "semantic_threshold": options.semantic_threshold,
        "semantic_override_threshold": options.semantic_override_threshold,
        "max_event_duration_sec": options.max_event_duration_sec,
        "cv": {"detector": "OpenCV FaceDetectorYN YuNet", "sample_fractions": [0.2, 0.5, 0.8], "maximum_input_edge": 640},
    }


def _safe_overwrite(run_dir: Path, release_id: str, movie_id: str) -> None:
    if not run_dir.exists():
        return
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RuntimeError(f"Refusing to overwrite unsafe Stage 3 target: {run_dir}")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Refusing to overwrite an unrecognized directory: {run_dir}")
    manifest = read_json(manifest_path)
    if manifest.get("release_id") != release_id or manifest.get("movie_id") != movie_id:
        raise RuntimeError(f"Refusing to overwrite a different Stage 3 run: {run_dir}")
    shutil.rmtree(run_dir)


def _shot_output(
    semantic: dict[str, Any], frames: list[dict[str, Any]], face_score: float,
    aggregate: dict[str, Any], movie_id: str, release_id: str, model_sha: str,
    provenance: dict[str, str],
) -> tuple[dict[str, Any], str | None]:
    shot = semantic["shot"]
    semantic_score = float(semantic["semantic_score"])
    context_confidence = float(semantic["context_confidence"])
    shot_score = 0.65 * semantic_score + 0.25 * face_score + 0.10 * context_confidence
    visible = int(aggregate["face_visible_frame_count"]) > 0
    basis = "semantic_and_cv" if visible else ("semantic_override" if semantic_score >= 0.75 else None)
    rejection = None if basis else "no_face_evidence_and_below_semantic_override_threshold"
    row = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "movie_id": movie_id,
        "source_shot_id": shot["shot_id"],
        "source_index": semantic["source_index"],
        "frame_range": shot["frame_range"],
        "time": shot["time"],
        "scene": shot.get("scene"),
        "scene_transition": bool(shot.get("scene_transition")),
        "semantic": {
            "ruleset_version": semantic["ruleset_version"],
            "semantic_score": semantic["semantic_score"],
            "categories": semantic["categories"],
            "evidence": semantic["evidence"],
        },
        "cv": {
            "detector": {"name": "OpenCV FaceDetectorYN", "model": "YuNet", "model_sha256": model_sha},
            "sample_frames": frames,
            "aggregate": aggregate,
            "face_score": face_score,
        },
        "context_confidence": semantic["context_confidence"],
        "shot_score": round(min(1.0, shot_score), 6),
        "selection_basis": basis,
        "provenance": provenance,
    }
    return row, rejection


def mine(
    options: MiningOptions, *, detector_factory: DetectorFactory = YuNetDetector,
    extractor: Callable[[Path, list[Any]], None] = extract_sparse_frames,
) -> dict[str, Any]:
    if not 0 <= options.semantic_threshold <= options.semantic_override_threshold <= 1:
        raise ValueError("Require 0 <= semantic-threshold <= semantic-override-threshold <= 1")
    if options.max_event_duration_sec <= 0:
        raise ValueError("max-event-duration-sec must be positive")
    release, _movie, inputs = _release_inputs(options)
    release_id = str(release["release_id"])
    run_dir = options.output_root / release_id / PIPELINE_VERSION / options.movie_key
    fingerprint = _fingerprint(options, release, inputs)
    if options.dry_run:
        return {"status": "dry_run", "movie_id": options.movie_key, "run_dir": run_dir.as_posix(), "fingerprint": fingerprint}

    manifest_path = run_dir / "manifest.json"
    if options.resume and not options.overwrite and manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("fingerprint") == fingerprint:
            from .validation import validate_run
            report = validate_run(run_dir)
            if report.passed:
                return {"status": "completed", "resumed": True, "movie_id": options.movie_key, "run_dir": run_dir.as_posix(), **manifest["counts"]}
    if run_dir.exists() and not options.overwrite:
        raise RuntimeError(f"Stage 3 output exists but cannot be resumed; use --overwrite: {run_dir}")
    if options.overwrite:
        _safe_overwrite(run_dir, release_id, options.movie_key)

    context_path = Path(inputs["reviewed_shot_context"]["path"])
    screenplay_path = Path(inputs["screenplay_context"]["path"])
    shots_path = Path(inputs["shots"]["path"])
    video_path = Path(inputs["video"]["path"])
    context_rows = read_jsonl(context_path)
    screenplay = read_json(screenplay_path)
    source_shots = read_jsonl(shots_path)
    _validate_source_pairing(context_rows, source_shots)
    pending_ids = _pending_subtitle_ids(release, options.movie_key)
    excluded_shots = {
        row["shot_id"] for row in context_rows
        if any(subtitle.get("subtitle_id") in pending_ids for subtitle in row.get("subtitles", []))
    }
    semantics = mine_shot_semantics(context_rows, screenplay, excluded_shots)
    seeds = [row for row in semantics if row["semantic_score"] >= options.semantic_threshold and row["excluded_reason"] is None]

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{options.movie_key}.stage3.", dir=run_dir.parent))
    try:
        all_requests = [request for semantic in seeds for request in sample_requests(semantic["shot"], temporary / "cv_frames")]
        extractor(video_path, all_requests)
        detector = detector_factory(options.face_model)
        row_provenance = {
            "reviewed_shot_context_sha256": inputs["reviewed_shot_context"]["sha256"],
            "screenplay_context_sha256": inputs["screenplay_context"]["sha256"],
            "shots_sha256": inputs["shots"]["sha256"],
            "video_sha256": inputs["video"]["sha256"],
        }
        request_by_shot = {
            semantic["shot"]["shot_id"]: [request for request in all_requests if request.shot_id == semantic["shot"]["shot_id"]]
            for semantic in seeds
        }
        selected: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        for semantic in semantics:
            shot_id = semantic["shot"]["shot_id"]
            if semantic["excluded_reason"]:
                audit.append({
                    "movie_id": options.movie_key, "source_shot_id": shot_id,
                    "source_index": semantic["source_index"], "semantic_score": semantic["semantic_score"],
                    "status": "excluded", "reason": semantic["excluded_reason"],
                })
                continue
            if semantic["semantic_score"] < options.semantic_threshold:
                continue
            requests = request_by_shot[shot_id]
            frames, face_score, aggregate = analyze_shot_frames(detector, requests, temporary)
            row, rejection = _shot_output(
                semantic, frames, face_score, aggregate, options.movie_key, release_id, inputs["face_model"]["sha256"],
                row_provenance,
            )
            if rejection is None:
                selected.append(row)
                status = "selected"
            else:
                status = "rejected"
            audit.append({
                "movie_id": options.movie_key, "source_shot_id": shot_id,
                "source_index": semantic["source_index"], "semantic_score": semantic["semantic_score"],
                "face_score": face_score, "shot_score": row["shot_score"], "status": status,
                "reason": rejection, "selection_basis": row["selection_basis"],
                "semantic": row["semantic"], "cv": row["cv"], "scene": row["scene"], "time": row["time"],
            })

        selected.sort(key=lambda row: int(row["source_index"]))
        for ordinal, row in enumerate(selected, 1):
            row["performance_shot_id"] = f"perfshot_{options.movie_key}_{ordinal:06d}"
        ranked = sorted(selected, key=lambda row: (-float(row["shot_score"]), float(row["time"]["start_sec"]), row["performance_shot_id"]))
        for rank, row in enumerate(ranked, 1):
            row["movie_rank"] = rank
        events = group_events(selected, context_rows, options.movie_key, release_id, options.max_event_duration_sec)

        write_jsonl(temporary / "performance_shots.jsonl", selected)
        write_jsonl(temporary / "performance_events.jsonl", events)
        write_jsonl(temporary / "screening_audit.jsonl", audit)
        qc = {
            "schema_version": SCHEMA_VERSION, "movie_id": options.movie_key,
            "source_shot_count": len(context_rows), "semantic_seed_count": len(seeds),
            "excluded_pending_ambiguity_shot_count": len(excluded_shots),
            "performance_shot_count": len(selected), "performance_event_count": len(events),
            "semantic_override_count": sum(row["selection_basis"] == "semantic_override" for row in selected),
            "rejected_seed_count": sum(row["status"] == "rejected" for row in audit),
        }
        write_json(temporary / "qc_summary.json", qc)
        outputs = {
            name: {"path": name, "sha256": sha256_file(temporary / name)}
            for name in ("performance_shots.jsonl", "performance_events.jsonl", "screening_audit.jsonl", "qc_summary.json")
        }
        manifest = {
            "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
            "ruleset_version": RULESET_VERSION, "status": "COMPLETE", "release_id": release_id,
            "movie_id": options.movie_key, "fingerprint": fingerprint, "inputs": inputs,
            "pending_subtitle_ids": sorted(pending_ids), "excluded_shot_ids": sorted(excluded_shots),
            "outputs": outputs,
            "counts": {"performance_shot_count": len(selected), "performance_event_count": len(events), "screening_audit_count": len(audit)},
            "constraints": {
                "openai_used": False, "openface_used": False, "face_identity_used": False,
                "emotion_classification_used": False, "visual_detector_scope": "face_presence_only",
            },
        }
        write_json(temporary / "manifest.json", manifest)
        temporary.replace(run_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    from .validation import validate_run
    report = validate_run(run_dir)
    if not report.passed:
        raise RuntimeError("Stage 3 validation failed: " + "; ".join(report.errors[:20]))
    return {
        "status": "completed", "resumed": False, "movie_id": options.movie_key,
        "run_dir": run_dir.as_posix(), "performance_shot_count": len(selected),
        "performance_event_count": len(events), "screening_audit_count": len(audit),
    }


__all__ = ["MiningOptions", "mine"]
