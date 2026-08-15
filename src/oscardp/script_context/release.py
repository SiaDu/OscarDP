from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .pipeline import _write_atomic, _write_json, _write_jsonl
from .schema import read_jsonl

PRODUCTION_REVIEWER_VERSION = "v3.2.1-production.3-retrieval-v3-validator-v3"
RELEASE_ID = "v3_2_1_production_3_final_seven"
BLOCKED_MOVIE_ID = "tt30144839"
EXPECTED_MOVIE_IDS = (
    "tt12300742", "tt1312221", "tt14905854", "tt18382850",
    "tt27714581", BLOCKED_MOVIE_ID, "tt30343021", "tt31193180",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")  # noqa: TRY004
    return value


def _movie_index(document: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    movies = document.get("movies")
    if not isinstance(movies, list):
        raise ValueError(f"{label} has no movies list")  # noqa: TRY004
    result = {row.get("movie_id"): row for row in movies if isinstance(row, dict)}
    if None in result or len(result) != len(movies):
        raise ValueError(f"{label} contains missing or duplicate movie IDs")
    if tuple(result) != EXPECTED_MOVIE_IDS:
        raise ValueError(f"{label} movie order/scope does not match the eight-movie release target")
    return result


def _manifest_path(
    output_root: Path, movie_id: str, status_movie: dict[str, Any], inventory_movie: dict[str, Any],
) -> Path:
    candidates = [
        status_movie.get("production_manifest_path"),
        (status_movie.get("completion_artifacts") or {}).get("production_manifest", {}).get("path"),
        (inventory_movie.get("production_review") or {}).get("production_manifest"),
    ]
    for value in candidates:
        if value:
            return Path(value)
    directory = output_root / movie_id / "review/openai/production_v3_2_1_retrieval_v3"
    matches = []
    for path in sorted(directory.glob("production_manifest*.json")):
        try:
            manifest = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if manifest.get("production_reviewer_version") == PRODUCTION_REVIEWER_VERSION:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"{movie_id} has {len(matches)} eligible final production manifests")
    return matches[0]


def _verify_artifact(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    actual = _sha(path)
    if actual != expected_sha256:
        raise RuntimeError(f"hash mismatch for {label}: {path}")
    result: dict[str, Any] = {"path": path.as_posix(), "sha256": actual}
    if path.suffix == ".jsonl":
        result["line_count"] = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    return result


def _verify_protected(label: str, source: dict[str, Any]) -> dict[str, Any]:
    path = Path(source["path"])
    if not path.is_file() or not source.get("unchanged"):
        raise RuntimeError(f"protected artifact is missing or was not previously verified: {label}")
    expected = source.get("expected_sha256")
    if not isinstance(expected, str) or not expected:
        raise RuntimeError(f"protected artifact has no expected SHA-256: {label}")
    stat = path.stat()
    current_stat = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    recorded_stat = source.get("expected_stat_fingerprint")
    if label == "video":
        if recorded_stat is not None and current_stat != recorded_stat:
            raise RuntimeError(f"protected video stat fingerprint changed: {path}")
        if source.get("actual_sha256") not in {None, expected}:
            raise RuntimeError(f"protected video prior SHA-256 verification is inconsistent: {path}")
        return {
            "path": path.as_posix(), "sha256": expected, "stat_fingerprint": current_stat,
            "verification_method": (
                "prior_completed_manifest_sha256_plus_recorded_stat_fingerprint"
                if recorded_stat is not None else
                "prior_completed_manifest_sha256_plus_release_stat_capture"
            ),
        }
    actual = _sha(path)
    if actual != expected:
        raise RuntimeError(f"protected artifact changed: {path}")
    return {
        "path": path.as_posix(), "sha256": actual, "stat_fingerprint": current_stat,
        "verification_method": "release_sha256",
    }


def _batch_records(production_dir: Path) -> list[dict[str, Any]]:
    fetch_by_batch: dict[str, dict[str, Any]] = {}
    for path in production_dir.glob("result*/fetch_metadata.json"):
        value = _load_json(path)
        if value.get("batch_id"):
            fetch_by_batch[value["batch_id"]] = {**value, "path": path.as_posix(), "sha256": _sha(path)}
    records = []
    for path in sorted(production_dir.glob("chunks/**/batch_job*.json")):
        job = _load_json(path)
        batch_id = job.get("batch_id")
        fetch = fetch_by_batch.get(batch_id)
        if fetch is None or fetch.get("status") != "completed":
            raise RuntimeError(f"production Batch lacks completed fetch evidence: {path}")
        records.append({
            "batch_id": batch_id, "input_file_id": job.get("input_file_id"),
            "output_file_id": fetch.get("output_file_id"), "error_file_id": fetch.get("error_file_id"),
            "model": job.get("model"), "request_count": job.get("request_count"),
            "status": "completed", "job_file": path.as_posix(), "job_file_sha256": _sha(path),
            "fetch_metadata": fetch["path"], "fetch_metadata_sha256": fetch["sha256"],
        })
    if not records:
        raise RuntimeError(f"no production Batch records found under {production_dir}")
    if len({row["batch_id"] for row in records}) != len(records):
        raise RuntimeError(f"duplicate production Batch IDs under {production_dir}")
    return records


def _ambiguity_rows(movie_id: str, title: str, audit_path: Path, release_id: str) -> list[dict[str, Any]]:
    result = []
    for row in read_jsonl(audit_path):
        classification = (row.get("diagnostics") or {}).get("no_candidate_match_classification")
        diagnosis = (row.get("evidence_review") or {}).get("diagnosis")
        if classification != "ambiguous_needs_review" and diagnosis != "genuine_ambiguity":
            continue
        item = deepcopy(row)
        item.update({
            "release_id": release_id, "movie_id": movie_id, "movie_title": title,
            "source_high_risk_audit": audit_path.as_posix(),
            "human_decision": None, "human_block_ids": None, "reviewer_notes": None,
            "review_status": "pending",
        })
        result.append(item)
    return result


def _stage3_handoff(release_id: str, movies: list[dict[str, Any]], ambiguity_count: int) -> str:
    lines = [
        f"# OscarDP Stage 3 handoff — {release_id}", "",
        "Stage 2 is frozen for seven movies. Stage 3 must consume only the reviewed shot-context paths below.",
        "The blocked movie and pending ambiguities are excluded from resolved-quality claims.", "",
        "| Movie | Status | Reviewed shot context |", "|---|---|---|",
    ]
    for movie in movies:
        lines.append(f"| {movie['movie_id']} | COMPLETE | `{movie['artifacts']['reviewed_shot_context']['path']}` |")
    lines.extend([
        f"| {BLOCKED_MOVIE_ID} | BLOCKED_WITH_EXPLICIT_REASON | — |", "",
        f"Pending isolated human ambiguities: {ambiguity_count}.", "",
        "Do not start OpenFace, face tracking, AU extraction, gaze, or acting annotation as part of this release.",
        "When the correct tt30144839 screenplay becomes available, process it separately and create a new eight-movie release.", "",
    ])
    return "\n".join(lines)


def _resume_frozen_release(
    release_dir: Path, release_id: str, code_commit: str,
    inventory: dict[str, Any], status: dict[str, Any], experiments_path: Path,
) -> dict[str, Any] | None:
    manifest_path = release_dir / "release_manifest.json"
    validation_path = release_dir / "release_validation.json"
    if not manifest_path.exists():
        return None
    manifest = _load_json(manifest_path)
    if (manifest.get("release_id"), manifest.get("code_commit")) != (release_id, code_commit):
        raise FileExistsError(f"refusing to overwrite a different release: {manifest_path}")
    if not validation_path.is_file():
        return None
    validation = _load_json(validation_path)
    if not validation.get("passed") or validation.get("release_id") != release_id:
        return None
    if (inventory.get("release") or {}).get("release_id") != release_id or (status.get("release") or {}).get("release_id") != release_id:
        return None
    for movie in manifest.get("movies", []):
        for name, artifact in movie.get("artifacts", {}).items():
            _verify_artifact(Path(artifact["path"]), artifact["sha256"], f"resume {movie.get('movie_id')} {name}")
    for label in ("pending_human_ambiguities", "stage3_handoff"):
        artifact = manifest[label]
        _verify_artifact(Path(artifact["path"]), artifact["sha256"], f"resume {label}")
    experiment_rows = read_jsonl(experiments_path)
    if not any(row.get("event") == "production_release_frozen" and row.get("release_id") == release_id for row in experiment_rows):
        return None
    summary = manifest["summary"]
    return {
        "release_id": release_id, "status": "FROZEN", "passed": True, "resumed": True,
        "manifest": manifest_path.as_posix(), "manifest_sha256": _sha(manifest_path),
        "validation": validation_path.as_posix(), "validation_sha256": _sha(validation_path),
        "pending_human_ambiguities": summary["pending_human_ambiguities"],
        "production_paid_batch_count": summary["production_paid_batch_count"],
        "complete_movies": summary["complete_movies"], "terminal_movies": summary["terminal_movies"],
        "blocked_movies": summary["blocked_movies"], "target_movies": summary["target_movies"],
    }


def freeze_stage2_release(
    inventory_path: Path, status_path: Path, experiments_path: Path, output_root: Path,
    release_dir: Path, code_commit: str, *, release_id: str = RELEASE_ID,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ValueError("code commit must be a full 40-character Git SHA")
    inventory, status = _load_json(inventory_path), _load_json(status_path)
    resumed = _resume_frozen_release(
        release_dir, release_id, code_commit, inventory, status, experiments_path,
    )
    if resumed is not None:
        return resumed
    inventory_before_sha, status_before_sha = _sha(inventory_path), _sha(status_path)
    inventory_movies = _movie_index(inventory, "inventory")
    status_movies = _movie_index(status, "status")
    completed = [movie_id for movie_id in EXPECTED_MOVIE_IDS if status_movies[movie_id].get("final_qc_status") == "COMPLETE"]
    blocked = [movie_id for movie_id in EXPECTED_MOVIE_IDS if status_movies[movie_id].get("final_qc_status") == "BLOCKED_WITH_EXPLICIT_REASON"]
    if completed != [movie_id for movie_id in EXPECTED_MOVIE_IDS if movie_id != BLOCKED_MOVIE_ID] or blocked != [BLOCKED_MOVIE_ID]:
        raise RuntimeError(f"release requires exactly seven expected COMPLETE movies and blocked {BLOCKED_MOVIE_ID}")
    blocked_status = status_movies[BLOCKED_MOVIE_ID]
    blocked_inventory = inventory_movies[BLOCKED_MOVIE_ID]
    if blocked_inventory.get("screenplay") is not None or blocked_status.get("source_status") != "blocked_missing_screenplay":
        raise RuntimeError(f"{BLOCKED_MOVIE_ID} is not consistently blocked for a missing screenplay")

    movie_records, ambiguities = [], []
    reviewer_manifest_hashes: set[str] = set()
    for movie_id in completed:
        status_movie, inventory_movie = status_movies[movie_id], inventory_movies[movie_id]
        manifest_path = _manifest_path(output_root, movie_id, status_movie, inventory_movie)
        manifest = _load_json(manifest_path)
        if manifest.get("movie_id") != movie_id or manifest.get("status") != "COMPLETE":
            raise RuntimeError(f"invalid final production manifest for {movie_id}")
        if manifest.get("production_reviewer_version") != PRODUCTION_REVIEWER_VERSION:
            raise RuntimeError(f"reviewer version mismatch for {movie_id}")
        counts = manifest.get("counts") or {}
        if int(counts.get("unresolved_ambiguities", -1)) > 5:
            raise RuntimeError(f"too many unresolved ambiguities for {movie_id}")
        artifacts = {
            name: _verify_artifact(Path(source["path"]), source["sha256"], f"{movie_id} {name}")
            for name, source in manifest.get("artifacts", {}).items()
        }
        for name in ("reviewed_alignment", "reviewed_shot_context", "high_risk_audit", "qc_report", "reviewer_manifest"):
            if name not in artifacts:
                raise RuntimeError(f"{movie_id} manifest is missing {name}")
        if artifacts["reviewed_alignment"].get("line_count") != counts.get("alignment_rows"):
            raise RuntimeError(f"reviewed alignment count mismatch for {movie_id}")
        if artifacts["reviewed_shot_context"].get("line_count") != counts.get("shot_rows"):
            raise RuntimeError(f"reviewed shot-context count mismatch for {movie_id}")
        if artifacts["high_risk_audit"].get("line_count") != counts.get("high_risk_audit_rows"):
            raise RuntimeError(f"high-risk audit count mismatch for {movie_id}")
        qc = _load_json(Path(artifacts["qc_report"]["path"]))
        if not qc.get("passed") or qc.get("errors"):
            raise RuntimeError(f"final QC did not pass for {movie_id}")
        for key in ("invalid_response_count", "missing_response_count", "foreign_response_count", "unresolved_candidate_recall_risk_count", "unresolved_reviewer_selection_risk_count"):
            if qc.get(key) != 0:
                raise RuntimeError(f"final QC {key} is not zero for {movie_id}")
        if qc.get("unresolved_ambiguity_count") != counts.get("unresolved_ambiguities"):
            raise RuntimeError(f"final QC ambiguity count mismatch for {movie_id}")
        protected = {name: _verify_protected(name, source) for name, source in manifest.get("protected_hashes", {}).items()}
        reviewer_manifest_hashes.add(artifacts["reviewer_manifest"]["sha256"])
        production_dir = manifest_path.parent
        batches = _batch_records(production_dir)
        ambiguity = _ambiguity_rows(movie_id, inventory_movie.get("title") or movie_id, Path(artifacts["high_risk_audit"]["path"]), release_id)
        if len(ambiguity) != counts.get("unresolved_ambiguities"):
            raise RuntimeError(f"pending ambiguity extraction mismatch for {movie_id}")
        ambiguities.extend(ambiguity)
        movie_records.append({
            "movie_id": movie_id, "title": inventory_movie.get("title"), "status": "COMPLETE",
            "production_manifest": {"path": manifest_path.as_posix(), "sha256": _sha(manifest_path)},
            "production_reviewer_version": manifest["production_reviewer_version"],
            "hard_validation_contract_version": manifest.get("hard_validation_contract_version"),
            "counts": counts, "artifacts": artifacts, "protected_artifacts": protected,
            "production_batches": batches, "production_batch_count": len(batches),
        })
    if len(reviewer_manifest_hashes) != 1:
        raise RuntimeError("completed movies do not share one frozen reviewer manifest")

    release_dir.mkdir(parents=True, exist_ok=True)
    ambiguity_path = release_dir / "pending_human_ambiguities.jsonl"
    ambiguity_manifest_path = release_dir / "pending_human_ambiguities_manifest.json"
    handoff_path = release_dir / "stage3_handoff.md"
    manifest_path = release_dir / "release_manifest.json"
    validation_path = release_dir / "release_validation.json"
    existing_manifest = _load_json(manifest_path) if manifest_path.exists() else None

    _write_jsonl(ambiguity_path, ambiguities)
    ambiguity_manifest = {
        "schema_version": "1.0", "release_id": release_id, "record_count": len(ambiguities),
        "movie_counts": {movie_id: sum(row["movie_id"] == movie_id for row in ambiguities) for movie_id in completed},
        "audit_sha256": _sha(ambiguity_path), "human_labels_present": False,
        "review_status_counts": {"pending": len(ambiguities)},
    }
    _write_json(ambiguity_manifest_path, ambiguity_manifest)
    _write_atomic(handoff_path, _stage3_handoff(release_id, movie_records, len(ambiguities)))
    total_batches = sum(row["production_batch_count"] for row in movie_records)
    manifest = {
        "schema_version": "1.0", "release_id": release_id, "status": "FROZEN",
        "code_commit": code_commit, "production_reviewer_version": PRODUCTION_REVIEWER_VERSION,
        "reviewer_manifest_sha256": next(iter(reviewer_manifest_hashes)),
        "summary": {"complete_movies": 7, "terminal_movies": 8, "blocked_movies": 1, "target_movies": 8, "pending_human_ambiguities": len(ambiguities), "production_paid_batch_count": total_batches},
        "movies": movie_records,
        "blocked_movies": [{
            "movie_id": BLOCKED_MOVIE_ID, "title": blocked_inventory.get("title"),
            "status": "BLOCKED_WITH_EXPLICIT_REASON", "reason": blocked_status.get("blocked_reason"),
            "source_status": blocked_status.get("source_status"), "screenplay": None,
        }],
        "governing_specification": status.get("active_spec_revision"),
        "registry_inputs": {"inventory": {"path": inventory_path.as_posix(), "sha256_before_release": inventory_before_sha}, "status": {"path": status_path.as_posix(), "sha256_before_release": status_before_sha}, "experiments": {"path": experiments_path.as_posix(), "sha256_before_release": _sha(experiments_path)}},
        "pending_human_ambiguities": {"path": ambiguity_path.as_posix(), "sha256": _sha(ambiguity_path), "manifest": ambiguity_manifest_path.as_posix(), "manifest_sha256": _sha(ambiguity_manifest_path)},
        "stage3_handoff": {"path": handoff_path.as_posix(), "sha256": _sha(handoff_path)},
        "frozen_at": existing_manifest.get("frozen_at") if existing_manifest else datetime.now(UTC).isoformat(),
    }
    _write_json(manifest_path, manifest)

    inventory = deepcopy(inventory); status = deepcopy(status)
    inventory_movies = _movie_index(inventory, "inventory")
    status_movies = _movie_index(status, "status")
    for movie in movie_records:
        movie_id = movie["movie_id"]
        production_review = inventory_movies[movie_id].setdefault("production_review", {})
        production_review.update({
            "status": "COMPLETE", "reviewer_version": PRODUCTION_REVIEWER_VERSION,
            "production_manifest": movie["production_manifest"]["path"], "production_manifest_sha256": movie["production_manifest"]["sha256"],
            "final_qc": movie["artifacts"]["qc_report"]["path"], "final_qc_sha256": movie["artifacts"]["qc_report"]["sha256"],
            "reviewed_alignment": movie["artifacts"]["reviewed_alignment"]["path"], "reviewed_alignment_sha256": movie["artifacts"]["reviewed_alignment"]["sha256"],
            "reviewed_shot_context": movie["artifacts"]["reviewed_shot_context"]["path"], "reviewed_shot_context_sha256": movie["artifacts"]["reviewed_shot_context"]["sha256"],
            "pending_ambiguity_count": movie["counts"]["unresolved_ambiguities"], "protected_hashes_unchanged": True,
        })
        status_movie = status_movies[movie_id]
        prior_active = {key: status_movie.get(key) for key in ("production_active_batch_id", "production_active_batch_status", "production_active_chunk_index", "production_active_request_counts") if status_movie.get(key) is not None}
        if prior_active:
            status_movie["release_archived_active_batch"] = prior_active
        for key in ("production_active_batch_id", "production_active_batch_status", "production_active_chunk_index", "production_active_request_counts"):
            if key in status_movie:
                status_movie[key] = None
        status_movie.update({
            "final_qc_status": "COMPLETE", "production_batch_status": "completed_validated_applied_final_qc_passed",
            "production_reviewer_version": PRODUCTION_REVIEWER_VERSION,
            "production_manifest_path": movie["production_manifest"]["path"], "production_manifest_sha256": movie["production_manifest"]["sha256"],
            "final_qc_path": movie["artifacts"]["qc_report"]["path"], "final_qc_sha256": movie["artifacts"]["qc_report"]["sha256"],
            "reviewed_alignment_path": movie["artifacts"]["reviewed_alignment"]["path"], "reviewed_alignment_sha256": movie["artifacts"]["reviewed_alignment"]["sha256"],
            "reviewed_shot_context_path": movie["artifacts"]["reviewed_shot_context"]["path"], "reviewed_shot_context_sha256": movie["artifacts"]["reviewed_shot_context"]["sha256"],
            "pending_ambiguity_count": movie["counts"]["unresolved_ambiguities"], "candidate_recall_risk_count": 0, "reviewer_selection_risk_count": 0,
            "protected_hashes_verified_unchanged": True,
        })
    release_summary = {"complete_movies": 7, "terminal_movies": 8, "blocked_movies": 1, "target_movies": 8}
    for document in (inventory, status):
        document["current_goal_commit"] = code_commit
        document["release"] = {"release_id": release_id, "status": "FROZEN", "manifest": manifest_path.as_posix(), "summary": release_summary}
        document["updated_at"] = datetime.now(UTC).isoformat()
    global_reviewer = status.setdefault("global_reviewer", {})
    active_snapshot = {key: value for key, value in global_reviewer.items() if key.startswith("active_") and value is not None}
    if active_snapshot:
        global_reviewer["release_archived_active_batch"] = active_snapshot
    for key in list(global_reviewer):
        if key.startswith("active_"):
            global_reviewer[key] = None
    global_reviewer.update({"release_status": "frozen", "release_id": release_id, "production_reviewer_version": PRODUCTION_REVIEWER_VERSION})
    _write_json(inventory_path, inventory)
    _write_json(status_path, status)

    experiment_rows = read_jsonl(experiments_path)
    if not any(row.get("event") == "production_release_frozen" and row.get("release_id") == release_id for row in experiment_rows):
        experiment_rows.append({
            "schema_version": "1.0", "event": "production_release_frozen", "release_id": release_id,
            "code_commit": code_commit, "reviewer_version": PRODUCTION_REVIEWER_VERSION,
            "complete_movies": 7, "terminal_movies": 8, "blocked_movies": 1,
            "manifest": manifest_path.as_posix(), "recorded_at": datetime.now(UTC).isoformat(),
        })
        _write_jsonl(experiments_path, experiment_rows)

    validation = {
        "schema_version": "1.0", "release_id": release_id, "passed": True, "errors": [],
        "code_commit": code_commit, "summary": release_summary,
        "release_manifest": {"path": manifest_path.as_posix(), "sha256": _sha(manifest_path)},
        "pending_human_ambiguities": {"path": ambiguity_path.as_posix(), "sha256": _sha(ambiguity_path), "record_count": len(ambiguities)},
        "stage3_handoff": {"path": handoff_path.as_posix(), "sha256": _sha(handoff_path)},
        "registry_outputs": {"inventory_sha256": _sha(inventory_path), "status_sha256": _sha(status_path), "experiments_sha256": _sha(experiments_path)},
        "checks": {
            "seven_complete": True, "eight_terminal": True, "one_blocked": True,
            "reviewer_version_consistent": True, "artifact_hashes_match": True,
            "protected_artifacts_unchanged": True, "final_qc_passed": True,
            "pending_ambiguities_self_contained": True, "no_paid_api_call": True,
        },
        "validated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(validation_path, validation)
    return {
        "release_id": release_id, "status": "FROZEN", "passed": True,
        "manifest": manifest_path.as_posix(), "manifest_sha256": _sha(manifest_path),
        "validation": validation_path.as_posix(), "validation_sha256": _sha(validation_path),
        "pending_human_ambiguities": len(ambiguities), "production_paid_batch_count": total_batches,
        **release_summary,
    }
