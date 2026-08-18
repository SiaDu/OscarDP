"""Manual-gallery target-face verification for frozen Stage 3 v2.1 candidates.

This module deliberately compares every detected face only with the one
nominated performer supplied by a human-labelled same-film gallery.  It never
assigns identities to other cast members and never uses screenplay speakers as
an identity signal.
"""
from __future__ import annotations

import math
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .cv import FrameRequest, YuNetDetector, extract_sparse_frames
from .grouping import group_events
from .io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from .schema import SCHEMA_VERSION

PIPELINE_VERSION = "performance_candidates_v2_2"
BACKEND = "insightface_buffalo_l"
GALLERY_VERSION = "target_face_gallery_v1"
DECISION_RULESET_VERSION = "target_face_verification_rules_v1"
SPARSE_FRACTIONS = (0.2, 0.5, 0.8)


def normalize_embedding(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    return [] if norm <= 0 else [value / norm for value in values]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return -1.0
    return sum(a * b for a, b in zip(left, right, strict=True))


def _hash(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": path.resolve().as_posix(), "sha256": sha256_file(path), "size": path.stat().st_size}


def _v21_inputs(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("pipeline_version") != "performance_candidates_v2_1":
        raise ValueError("Target face verification requires a frozen performance_candidates_v2_1 run")
    for name in ("performance_shots.jsonl", "performance_events.jsonl"):
        expected = (manifest.get("outputs", {}).get(name) or {}).get("sha256")
        path = run_dir / name
        if not expected or sha256_file(path) != expected:
            raise RuntimeError(f"v2.1 output hash mismatch: {name}")
    return manifest, read_jsonl(run_dir / "performance_shots.jsonl"), read_jsonl(run_dir / "performance_events.jsonl")


def _crop(image_path: Path, bbox: list[float], output: Path) -> None:
    with Image.open(image_path) as image:
        width, height = image.size
        x, y, face_width, face_height = bbox
        left, top = max(0, math.floor(x)), max(0, math.floor(y))
        right, bottom = min(width, math.ceil(x + face_width)), min(height, math.ceil(y + face_height))
        if right <= left or bottom <= top:
            raise ValueError(f"invalid face crop bounds in {image_path}")
        output.parent.mkdir(parents=True, exist_ok=True)
        image.crop((left, top, right, bottom)).convert("RGB").save(output, quality=95)


def _contact_sheet(rows: list[dict[str, Any]], output: Path) -> None:
    cell_width, cell_height, columns = 240, 220, 4
    sheet = Image.new("RGB", (columns * cell_width, max(1, math.ceil(len(rows) / columns)) * cell_height), "white")
    for index, row in enumerate(rows):
        x, y = index % columns * cell_width, index // columns * cell_height
        crop = Path(row["crop_path"])
        with Image.open(crop) as image:
            image = image.convert("RGB")
            image.thumbnail((cell_width - 8, 170))
            sheet.paste(image, (x + (cell_width - image.width) // 2, y))
        draw = ImageDraw.Draw(sheet)
        draw.text((x + 4, y + 174), str(row["candidate_id"]), fill="black")
        draw.text((x + 4, y + 190), f"{row['source_shot_id']} / {int(row['sample_fraction'] * 100)}% / face {row['face_index']}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def prepare_target_gallery(
    v21_run: Path, output_dir: Path, face_model: Path, candidate_count: int = 20,
    face_model_sha256: str | None = None, overwrite: bool = False,
    detector_factory: Callable[[Path], YuNetDetector] = YuNetDetector,
) -> dict[str, Any]:
    """Create manually selectable same-film face crops without assigning identity."""
    if candidate_count <= 0:
        raise ValueError("candidate-count must be positive")
    manifest, shots, _events = _v21_inputs(v21_run)
    model = _hash(face_model)
    if face_model_sha256 and model["sha256"] != face_model_sha256:
        raise RuntimeError("SHA-256 mismatch for YuNet model")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Gallery output already exists and is non-empty: {output_dir}")
        if not (output_dir / "gallery_manifest.json").is_file():
            raise RuntimeError(f"Refusing to overwrite unrecognized gallery directory: {output_dir}")
        shutil.rmtree(output_dir)
    detector = detector_factory(face_model)
    candidates: list[dict[str, Any]] = []
    # A score/scene ordering is only a browsing aid.  Every candidate remains
    # unlabelled until a reviewer explicitly chooses target/non_target.
    ordered = sorted(shots, key=lambda row: (-float(row["shot_score"]), float(row["time"]["start_sec"]), row["source_shot_id"]))
    for pass_index, rows in enumerate((ordered, ordered)):
        for shot in rows:
            if len(candidates) >= candidate_count:
                break
            shot_id = shot["source_shot_id"]
            frames = sorted(shot.get("cv", {}).get("sample_frames", []), key=lambda frame: float(frame["fraction"]))
            # Centre frames maximize legibility.  The second pass deliberately
            # adds 20/80% alternatives only if distinct shots did not fill the
            # requested contact sheet, preserving visual/scene diversity.
            frames = [frame for frame in frames if float(frame["fraction"]) == .5] if pass_index == 0 else frames
            for frame in frames:
                if len(candidates) >= candidate_count:
                    break
                path = v21_run / frame["path"]
                detections, _metadata = detector.detect_faces(path)
                for detection in detections:
                    if len(candidates) >= candidate_count:
                        break
                    candidate_id = f"gallery_candidate_{len(candidates) + 1:04d}"
                    crop_path = output_dir / "crops" / f"{candidate_id}.jpg"
                    _crop(path, detection["bbox"], crop_path)
                    candidates.append({
                        "candidate_id": candidate_id, "gallery_version": GALLERY_VERSION,
                        "performer_id": (manifest.get("target") or {}).get("performer_id"),
                        "performer_name": (manifest.get("target") or {}).get("performer_name"),
                        "source_shot_id": shot_id, "performance_shot_id": shot["performance_shot_id"],
                        "sample_fraction": frame["fraction"], "timestamp_sec": frame["time_sec"],
                        "frame_path": path.as_posix(), "face_index": detection["face_index"],
                        "bbox": detection["bbox"], "face_area_ratio": detection["face_area_ratio"],
                        "crop_path": crop_path.resolve().as_posix(), "crop_sha256": sha256_file(crop_path),
                        "label": None, "manual_notes": None,
                    })
        if len(candidates) >= candidate_count:
            break
    if not candidates:
        raise RuntimeError("YuNet did not find any face crops in v2.1 sparse frames")
    write_jsonl(output_dir / "gallery_candidates.jsonl", candidates)
    write_jsonl(output_dir / "gallery_candidates.labeled.jsonl", candidates)
    _contact_sheet(candidates, output_dir / "gallery_candidates.contact_sheet.jpg")
    manifest_out = {
        "gallery_version": GALLERY_VERSION, "purpose": "manual same-film target and hard-negative gallery selection",
        "target": manifest.get("target"), "v2_1_run": _hash(v21_run / "manifest.json"),
        "v2_1_performance_shots": _hash(v21_run / "performance_shots.jsonl"), "yunet_model": model,
        "candidate_count": len(candidates),
        "manual_labels_path": (output_dir / "gallery_candidates.labeled.jsonl").resolve().as_posix(),
        "instructions": {"target": "Select 8-15 clear live same-film target face crops.", "non_target": "Optionally select 5-10 definitely non-target face crops.", "exclude": "Use skip for screen/photo/artwork, blur, tiny, occluded, or extreme-profile faces."},
    }
    write_json(output_dir / "gallery_manifest.json", manifest_out)
    return {"status": "prepared", "gallery_dir": output_dir.as_posix(), "candidate_count": len(candidates), "labeled_manifest": manifest_out["manual_labels_path"]}


def _create_analyzer(model_root: Path, provider: str = "CUDAExecutionProvider") -> Any:
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError("Target face verification requires optional dependency insightface") from exc
    app = FaceAnalysis(name="buffalo_l", root=str(model_root), providers=[provider])
    app.prepare(ctx_id=0 if "CUDA" in provider.upper() else -1, det_size=(640, 640))
    if "CUDA" in provider.upper():
        active = {
            name
            for model in app.models.values()
            for name in model.session.get_providers()
        }
        if "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                "CUDAExecutionProvider was requested but InsightFace fell back to CPU; "
                "install a CUDA-compatible ONNX Runtime GPU runtime or pass "
                "--provider CPUExecutionProvider explicitly."
            )
    return app


def _bbox_iou(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1, y1, x2, y2 = max(lx, rx), max(ly, ry), min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def _embed(analyzer: Any, image_path: Path, expected_bbox: list[float]) -> list[float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Target face verification requires OpenCV") from exc
    image = cv2.imread(str(image_path))
    faces = analyzer.get(image) if image is not None else []
    if not faces:
        return []
    # InsightFace bboxes are xyxy; YuNet records are xywh.  Matching on the
    # full source frame avoids a second detector failing on a tight face crop.
    face = max(
        faces,
        key=lambda item: _bbox_iou(
            expected_bbox,
            [float(item.bbox[0]), float(item.bbox[1]), float(item.bbox[2] - item.bbox[0]), float(item.bbox[3] - item.bbox[1])],
        ),
    )
    face_bbox = [float(face.bbox[0]), float(face.bbox[1]), float(face.bbox[2] - face.bbox[0]), float(face.bbox[3] - face.bbox[1])]
    if _bbox_iou(expected_bbox, face_bbox) < 0.10:
        return []
    return normalize_embedding([float(value) for value in face.embedding.flatten().tolist()]) if hasattr(face, "embedding") else []


def _model_provenance(model_root: Path, provider: str) -> dict[str, Any]:
    paths = sorted(path for path in model_root.rglob("*.onnx") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No InsightFace ONNX models found under {model_root}")
    return {"backend": BACKEND, "name": "buffalo_l", "provider": provider, "model_root": model_root.resolve().as_posix(), "files": [_hash(path) for path in paths]}


def _labels(gallery_dir: Path, labels_path: Path | None) -> list[dict[str, Any]]:
    path = labels_path or gallery_dir / "gallery_candidates.labeled.jsonl"
    rows = read_jsonl(path)
    invalid = [row.get("candidate_id") for row in rows if row.get("label") not in {"target", "non_target", "skip", None}]
    if invalid:
        raise ValueError(f"Invalid gallery label(s): {invalid[:5]}")
    selected = [row for row in rows if row.get("label") in {"target", "non_target"}]
    if not selected:
        raise ValueError("No target/non_target crops are labelled in gallery manifest")
    return selected


def calibrate_target_face(
    gallery_dir: Path, labels_path: Path | None, model_root: Path, provider: str = "CUDAExecutionProvider",
    analyzer_factory: Callable[[Path, str], Any] = _create_analyzer,
) -> dict[str, Any]:
    rows = _labels(gallery_dir, labels_path)
    analyzer = analyzer_factory(model_root, provider)
    embeddings = []
    for row in rows:
        vector = _embed(analyzer, Path(row["frame_path"]), [float(value) for value in row["bbox"]])
        if vector:
            embeddings.append({**row, "embedding": vector})
    targets = [row for row in embeddings if row["label"] == "target"]
    negatives = [row for row in embeddings if row["label"] == "non_target"]
    if len(targets) < 2:
        raise ValueError("Calibration requires at least two usable manually labelled target gallery crops")
    positive = [max(cosine_similarity(row["embedding"], other["embedding"]) for other in targets if other is not row) for row in targets]
    negative = [max(cosine_similarity(row["embedding"], target["embedding"]) for target in targets) for row in negatives]
    # Precision-first: every manually labelled positive must clear the
    # threshold, while negatives supply an observed upper bound.  This is
    # derived entirely from the current same-film gallery rather than copied
    # from a previous project.
    threshold = min(positive)
    positive_margins = [
        value - max((cosine_similarity(row["embedding"], negative_row["embedding"]) for negative_row in negatives), default=-1.0)
        for row, value in zip(targets, positive, strict=True)
    ]
    negative_margins = [
        value - max((cosine_similarity(row["embedding"], other["embedding"]) for other in negatives if other is not row), default=-1.0)
        for row, value in zip(negatives, negative, strict=True)
    ]
    margin = min(positive_margins) if positive_margins else 0.0
    if negative_margins:
        margin = max(margin, max(negative_margins))
    result = {"gallery_version": GALLERY_VERSION, "selection_objective": "precision_first", "positive_count": len(positive), "negative_count": len(negative), "positive_similarity": positive, "negative_target_similarity": negative, "positive_margins": positive_margins, "negative_margins": negative_margins, "recommended_threshold": round(threshold, 6), "recommended_margin": round(margin, 6), "model": _model_provenance(model_root, provider)}
    write_json(gallery_dir / "calibration.json", result)
    write_jsonl(gallery_dir / "gallery_embeddings.jsonl", embeddings)
    return result


def _score(vector: list[float], targets: list[dict[str, Any]], negatives: list[dict[str, Any]]) -> dict[str, Any]:
    target_scores = sorted(((cosine_similarity(vector, row["embedding"]), row["candidate_id"]) for row in targets), reverse=True)
    negative_scores = [cosine_similarity(vector, row["embedding"]) for row in negatives]
    best_target, prototype = target_scores[0] if target_scores else (-1.0, None)
    top_k = target_scores[: min(3, len(target_scores))]
    best_negative = max(negative_scores, default=None)
    return {
        "best_target_similarity": round(best_target, 6),
        "top_k_target_mean": round(sum(value for value, _id in top_k) / len(top_k), 6) if top_k else None,
        "best_target_prototype_id": prototype,
        "best_negative_similarity": round(best_negative, 6) if best_negative is not None else None,
        "target_negative_margin": round(best_target - best_negative, 6) if best_negative is not None else None,
    }


def _decision(evidence: list[dict[str, Any]], threshold: float, margin: float) -> tuple[str, int]:
    supported = [row for row in evidence if row.get("best_target_similarity", -1.0) >= threshold and (row.get("target_negative_margin") is None or row["target_negative_margin"] >= margin)]
    frames = {float(row["sample_fraction"]) for row in supported}
    if len(frames) >= 2:
        return "verified", len(frames)
    usable = [row for row in evidence if row.get("embedding_status") == "ok"]
    near = any(abs(float(row["best_target_similarity"]) - threshold) <= 0.03 for row in usable)
    if len(frames) == 1 or near or len(usable) < 2:
        return "uncertain", len(frames)
    return "not_verified", 0


def _frame_evidence(
    detector: YuNetDetector, analyzer: Any, frame_path: Path, fraction: float, timestamp: float,
) -> list[dict[str, Any]]:
    detections, _metadata = detector.detect_faces(frame_path)
    rows = []
    for detection in detections:
        vector = _embed(analyzer, frame_path, [float(value) for value in detection["bbox"]])
        rows.append({"sample_fraction": fraction, "timestamp_sec": round(timestamp, 6), "face_index": detection["face_index"], "bbox": detection["bbox"], "face_area_ratio": detection["face_area_ratio"], "embedding_status": "ok" if vector else "embedding_failed", "_embedding": vector})
    return rows


def verify_target_faces(
    v21_run: Path, gallery_dir: Path, model_root: Path, face_model: Path, threshold: float | None = None,
    margin_threshold: float | None = None, provider: str = "CUDAExecutionProvider", overwrite: bool = False,
    detector_factory: Callable[[Path], YuNetDetector] = YuNetDetector,
    analyzer_factory: Callable[[Path, str], Any] = _create_analyzer,
    dense_extractor: Callable[[Path, list[FrameRequest]], None] = extract_sparse_frames,
) -> dict[str, Any]:
    """Run sparse verification and bounded dense retry only for uncertain shots."""
    manifest, shots, old_events = _v21_inputs(v21_run)
    calibration = read_json(gallery_dir / "calibration.json")
    threshold = float(calibration["recommended_threshold"] if threshold is None else threshold)
    margin_threshold = float(calibration["recommended_margin"] if margin_threshold is None else margin_threshold)
    gallery = read_jsonl(gallery_dir / "gallery_embeddings.jsonl")
    target = manifest.get("target") or {}
    if any(row.get("performer_id") != target.get("performer_id") for row in gallery):
        raise ValueError("Gallery performer_id does not match v2.1 run target")
    targets, negatives = [row for row in gallery if row["label"] == "target"], [row for row in gallery if row["label"] == "non_target"]
    if len(targets) < 2:
        raise ValueError("Verification requires at least two target gallery embeddings")
    output = v21_run.parents[2] / PIPELINE_VERSION / manifest["movie_id"] / v21_run.name
    if output.exists() and not overwrite:
        raise RuntimeError(f"v2.2 output exists: {output}; use --overwrite")
    if output.exists():
        if not (output / "manifest.json").is_file():
            raise RuntimeError(f"Refusing to overwrite unrecognized directory: {output}")
        shutil.rmtree(output)
    detector, analyzer = detector_factory(face_model), analyzer_factory(model_root, provider)
    temporary = Path(tempfile.mkdtemp(prefix=".stage3-v22-", dir=output.parent))
    try:
        sparse_status: dict[str, tuple[str, list[dict[str, Any]], int]] = {}
        for shot in shots:
            evidence = []
            for frame in shot.get("cv", {}).get("sample_frames", []):
                frame_evidence = _frame_evidence(detector, analyzer, v21_run / frame["path"], float(frame["fraction"]), float(frame["time_sec"]))
                for item in frame_evidence:
                    vector = item.pop("_embedding")
                    if vector:
                        item.update(_score(vector, targets, negatives))
                evidence.extend(frame_evidence)
            status, supported = _decision(evidence, threshold, margin_threshold)
            sparse_status[shot["performance_shot_id"]] = (status, evidence, supported)
        # Dense decoding is deliberately deferred to only uncertain shots.
        video = Path(manifest["inputs"]["video"]["path"])
        uncertain = [shot for shot in shots if sparse_status[shot["performance_shot_id"]][0] == "uncertain"]
        dense_requests: list[FrameRequest] = []
        for shot in uncertain:
            start, end = float(shot["time"]["start_sec"]), float(shot["time"]["end_sec"])
            count = min(12, max(1, math.ceil(end - start)))
            for index in range(count):
                fraction = (index + 0.5) / count
                dense_requests.append(FrameRequest(shot["source_shot_id"], fraction, 0, start + (end - start) * fraction, temporary / "dense_frames" / shot["source_shot_id"] / f"dense_{index:02d}.jpg"))
        if dense_requests:
            dense_extractor(video, dense_requests)
        final_rows, audit = [], []
        for shot in shots:
            sparse, evidence, supported = sparse_status[shot["performance_shot_id"]]
            dense = [request for request in dense_requests if request.shot_id == shot["source_shot_id"]]
            if sparse == "uncertain":
                for request in dense:
                    for item in _frame_evidence(detector, analyzer, request.output_path, request.fraction, request.time_sec):
                        vector = item.pop("_embedding")
                        if vector:
                            item.update(_score(vector, targets, negatives))
                        item["sampling_pass"] = "dense"
                        evidence.append(item)
                final_status, supported = _decision(evidence, threshold, margin_threshold)
                if final_status == "uncertain":
                    final_status = "uncertain"
            else:
                final_status = sparse
            verified_evidence = [row for row in evidence if row.get("best_target_similarity", -1.0) >= threshold and (row.get("target_negative_margin") is None or row["target_negative_margin"] >= margin_threshold)]
            summary = {"status": final_status, "backend": BACKEND, "gallery_version": GALLERY_VERSION, "decision_ruleset_version": DECISION_RULESET_VERSION, "threshold": threshold, "margin_threshold": margin_threshold, "sparse_sample_count": 3, "matched_sparse_sample_count": supported if not dense else len({row["sample_fraction"] for row in verified_evidence if row.get("sampling_pass") != "dense"}), "dense_resampling_used": bool(dense), "dense_sample_count": len(dense), "best_similarity": max((row.get("best_target_similarity", -1.0) for row in evidence), default=None), "best_negative_similarity": max((row.get("best_negative_similarity") for row in evidence if row.get("best_negative_similarity") is not None), default=None), "best_margin": max((row.get("target_negative_margin") for row in evidence if row.get("target_negative_margin") is not None), default=None), "evidence": evidence}
            audit.append({"performance_shot_id": shot["performance_shot_id"], "source_shot_id": shot["source_shot_id"], "status": final_status, "dense_retry_used": bool(dense), "reason": "criteria_met" if final_status == "verified" else "current_visual_evidence_not_sufficient", "target_face_verification": summary})
            if final_status == "verified":
                row = {**shot, "target_face_verification": summary, "verified_target_person_id": target.get("performer_id"), "target_face_verified": True, "live_performance_verified": None}
                final_rows.append(row)
        ranked = sorted(final_rows, key=lambda row: (-float(row["shot_score"]), float(row["time"]["start_sec"]), row["performance_shot_id"]))
        for rank, row in enumerate(ranked, 1):
            row["target_rank"] = rank
        context = read_jsonl(Path(manifest["inputs"]["reviewed_shot_context"]["path"]))
        old_by_source = {source: event["event_id"] for event in old_events for source in event.get("source_shot_ids", [])}
        events = group_events(final_rows, context, manifest["movie_id"], manifest["release_id"], float(manifest["fingerprint"]["max_event_duration_sec"]), None)
        for event in events:
            event["target"] = target
            event["event_id"] = event["event_id"].replace("_target_", f"_{target.get('performer_id') or 'target'}_")
            event["source_v2_1_event_ids"] = sorted({old_by_source[source] for source in event["source_shot_ids"] if source in old_by_source})
        write_jsonl(temporary / "performance_shots.jsonl", sorted(final_rows, key=lambda row: int(row["source_index"])))
        write_jsonl(temporary / "performance_events.jsonl", events)
        write_jsonl(temporary / "target_face_verification.jsonl", audit)
        write_jsonl(temporary / "verification_audit.jsonl", audit)
        counts = Counter(row["status"] for row in audit)
        qc = {"v2_1_input_shot_count": len(shots), "sparse_verified_count": sum(value[0] == "verified" for value in sparse_status.values()), "sparse_uncertain_count": sum(value[0] == "uncertain" for value in sparse_status.values()), "sparse_not_verified_count": sum(value[0] == "not_verified" for value in sparse_status.values()), "dense_retry_shot_count": len(uncertain), "dense_verified_count": sum(row["status"] == "verified" and row["dense_retry_used"] for row in audit), "dense_not_verified_count": sum(row["status"] == "not_verified" and row["dense_retry_used"] for row in audit), "dense_still_uncertain_count": sum(row["status"] == "uncertain" and row["dense_retry_used"] for row in audit), "final_verified_shot_count": counts["verified"], "final_uncertain_count": counts["uncertain"], "final_not_verified_count": counts["not_verified"], "static_risk_verified_count": sum(bool((row.get("static_depiction_risk") or {}).get("flagged")) for row in final_rows), "performance_event_count": len(events)}
        write_json(temporary / "qc_summary.json", qc)
        outputs = {name: {"path": name, "sha256": sha256_file(temporary / name)} for name in ("performance_shots.jsonl", "performance_events.jsonl", "target_face_verification.jsonl", "verification_audit.jsonl", "qc_summary.json")}
        out_manifest = {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION, "status": "COMPLETE", "release_id": manifest["release_id"], "movie_id": manifest["movie_id"], "target": target, "v2_1_input": {"run_dir": v21_run.resolve().as_posix(), "manifest": _hash(v21_run / "manifest.json"), "performance_shots": _hash(v21_run / "performance_shots.jsonl")}, "gallery": {"manifest": _hash(gallery_dir / "gallery_manifest.json"), "calibration": _hash(gallery_dir / "calibration.json"), "embeddings": _hash(gallery_dir / "gallery_embeddings.jsonl")}, "model": _model_provenance(model_root, provider), "verification_rules": {"version": DECISION_RULESET_VERSION, "threshold": threshold, "margin_threshold": margin_threshold, "sparse_min_distinct_samples": 2, "dense_max_frames": 12}, "inputs": manifest["inputs"], "outputs": outputs, "counts": {"performance_shot_count": len(final_rows), "performance_event_count": len(events), "verification_audit_count": len(audit)}}
        write_json(temporary / "manifest.json", out_manifest)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"status": "completed", "run_dir": output.as_posix(), **qc}


__all__ = ["BACKEND", "DECISION_RULESET_VERSION", "GALLERY_VERSION", "PIPELINE_VERSION", "calibrate_target_face", "cosine_similarity", "normalize_embedding", "prepare_target_gallery", "verify_target_faces"]
