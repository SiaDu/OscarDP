from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oscardp.artifact_paths import parse_path_maps
from oscardp.script_context.release import (
    EXPECTED_MOVIE_IDS,
    PRODUCTION_REVIEWER_VERSION,
    freeze_stage2_release,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def release_fixture(tmp_path: Path, *, ambiguity_movie: str = "tt14905854") -> tuple[Path, Path, Path, Path]:
    output_root = tmp_path / "processed"
    inventory_movies, status_movies = [], []
    reviewer = output_root / "reviewer.json"
    write_json(reviewer, {"production_reviewer_version": PRODUCTION_REVIEWER_VERSION})
    for movie_id in EXPECTED_MOVIE_IDS:
        title = f"Title {movie_id}"
        if movie_id == "tt30144839":
            inventory_movies.append({"movie_id": movie_id, "title": "One Battle After Another", "screenplay": None})
            status_movies.append({
                "movie_id": movie_id, "final_qc_status": "BLOCKED_WITH_EXPLICIT_REASON",
                "source_status": "blocked_missing_screenplay", "blocked_reason": "Required screenplay source is unavailable.",
            })
            continue
        movie_dir = output_root / movie_id
        production_dir = movie_dir / "review/openai/production_v3_2_1_retrieval_v3"
        reviewed = movie_dir / "reviewed.jsonl"; shots = movie_dir / "reviewed-shots.jsonl"
        risk = production_dir / "risk.jsonl"; qc = production_dir / "qc.json"
        source = movie_dir / "source.pdf"; video = movie_dir / "video.mkv"
        write_jsonl(reviewed, [{"subtitle_id": "s1"}]); write_jsonl(shots, [{"shot_id": "shot_000001"}])
        ambiguity_count = 1 if movie_id == ambiguity_movie else 0
        risk_rows = [] if not ambiguity_count else [{
            "request_id": "r1", "subtitle_id": "s1", "subtitle_text": "Maybe.",
            "diagnostics": {"no_candidate_match_classification": "ambiguous_needs_review"},
            "evidence_review": {"diagnosis": "genuine_ambiguity"},
            "human_decision": "must be cleared", "review_status": "completed",
        }]
        write_jsonl(risk, risk_rows)
        write_json(qc, {
            "passed": True, "errors": [], "invalid_response_count": 0, "missing_response_count": 0,
            "foreign_response_count": 0, "unresolved_candidate_recall_risk_count": 0,
            "unresolved_reviewer_selection_risk_count": 0, "unresolved_ambiguity_count": ambiguity_count,
        })
        source.write_bytes(b"screenplay"); video.write_bytes(b"video")
        job = production_dir / "chunks/batch_job.chunk_001.json"
        write_json(job, {"batch_id": f"batch-{movie_id}", "input_file_id": f"input-{movie_id}", "model": "gpt-test", "request_count": 1})
        fetch = production_dir / "result_chunk_001/fetch_metadata.json"
        write_json(fetch, {"batch_id": f"batch-{movie_id}", "status": "completed", "output_file_id": f"output-{movie_id}", "error_file_id": None})
        artifacts = {
            "reviewed_alignment": {"path": str(reviewed), "sha256": sha(reviewed)},
            "reviewed_shot_context": {"path": str(shots), "sha256": sha(shots)},
            "validated_responses": {"path": str(reviewed), "sha256": sha(reviewed)},
            "high_risk_audit": {"path": str(risk), "sha256": sha(risk)},
            "high_risk_summary": {"path": str(qc), "sha256": sha(qc)},
            "qc_report": {"path": str(qc), "sha256": sha(qc)},
            "reviewer_manifest": {"path": str(reviewer), "sha256": sha(reviewer)},
        }
        manifest = production_dir / "production_manifest.json"
        write_json(manifest, {
            "movie_id": movie_id, "status": "COMPLETE", "production_reviewer_version": PRODUCTION_REVIEWER_VERSION,
            "hard_validation_contract_version": "candidate_task_v3_structure_v3", "artifacts": artifacts,
            "protected_hashes": {
                "video": {"path": str(video), "expected_sha256": sha(video), "actual_sha256": sha(video), "unchanged": True},
                "screenplay": {"path": str(source), "expected_sha256": sha(source), "actual_sha256": sha(source), "unchanged": True},
            },
            "counts": {"requests": 1, "resolutions": 1, "alignment_rows": 1, "shot_rows": 1, "high_risk_audit_rows": ambiguity_count, "unresolved_ambiguities": ambiguity_count},
        })
        inventory_movies.append({"movie_id": movie_id, "title": title, "screenplay": {"path": str(source)}})
        status_movies.append({
            "movie_id": movie_id, "final_qc_status": "COMPLETE", "production_manifest_path": str(manifest),
            "production_active_batch_id": f"batch-{movie_id}", "production_active_batch_status": "in_progress",
        })
    inventory = output_root / "stage2_goal_inventory.json"
    status = output_root / "stage2_goal_status.json"
    experiments = output_root / "stage2_reviewer_experiments.jsonl"
    write_json(inventory, {"movies": inventory_movies, "current_goal_commit": "old"})
    write_json(status, {"movies": status_movies, "current_goal_commit": "old", "active_spec_revision": {"sha256": "spec"}, "global_reviewer": {"active_batch_id": "stale", "active_batch_status": "in_progress"}})
    experiments.write_text("", encoding="utf-8")
    return inventory, status, experiments, output_root


def test_freeze_release_writes_self_contained_release_and_repairs_registries(tmp_path: Path) -> None:
    inventory, status, experiments, output_root = release_fixture(tmp_path)
    release_dir = output_root / "stage2_releases/release"
    commit = "a" * 40

    result = freeze_stage2_release(inventory, status, experiments, output_root, release_dir, commit)

    assert result["passed"] and result["complete_movies"] == 7
    assert result["terminal_movies"] == 8 and result["blocked_movies"] == 1
    assert result["pending_human_ambiguities"] == 1 and result["production_paid_batch_count"] == 7
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    assert manifest["summary"]["production_paid_batch_count"] == 7
    assert manifest["blocked_movies"][0]["movie_id"] == "tt30144839"
    ambiguity = json.loads((release_dir / "pending_human_ambiguities.jsonl").read_text())
    assert ambiguity["human_decision"] is None and ambiguity["review_status"] == "pending"
    updated_inventory = json.loads(inventory.read_text())
    updated_status = json.loads(status.read_text())
    assert updated_inventory["current_goal_commit"] == commit
    assert all(row.get("production_review", {}).get("status") == "COMPLETE" for row in updated_inventory["movies"] if row["movie_id"] != "tt30144839")
    assert updated_status["global_reviewer"]["active_batch_id"] is None
    assert updated_status["global_reviewer"]["release_status"] == "frozen"
    assert json.loads((release_dir / "release_validation.json").read_text())["passed"]
    assert "production_release_frozen" in experiments.read_text()

    frozen_hashes = {path.name: sha(path) for path in release_dir.iterdir() if path.is_file()}
    resumed = freeze_stage2_release(inventory, status, experiments, output_root, release_dir, commit)
    assert resumed["resumed"] is True
    assert frozen_hashes == {path.name: sha(path) for path in release_dir.iterdir() if path.is_file()}


def test_freeze_release_rejects_artifact_hash_mismatch_before_registry_updates(tmp_path: Path) -> None:
    inventory, status, experiments, output_root = release_fixture(tmp_path)
    original_inventory = inventory.read_bytes(); original_status = status.read_bytes()
    status_doc = json.loads(status.read_text())
    manifest = Path(status_doc["movies"][0]["production_manifest_path"])
    manifest_doc = json.loads(manifest.read_text())
    Path(manifest_doc["artifacts"]["reviewed_alignment"]["path"]).write_text("changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        freeze_stage2_release(inventory, status, experiments, output_root, output_root / "release", "b" * 40)

    assert inventory.read_bytes() == original_inventory and status.read_bytes() == original_status


def test_freeze_release_requires_exact_terminal_population(tmp_path: Path) -> None:
    inventory, status, experiments, output_root = release_fixture(tmp_path)
    status_doc = json.loads(status.read_text())
    next(row for row in status_doc["movies"] if row["movie_id"] == "tt30144839")["final_qc_status"] = "COMPLETE"
    write_json(status, status_doc)

    with pytest.raises(RuntimeError, match="exactly seven"):
        freeze_stage2_release(inventory, status, experiments, output_root, output_root / "release", "c" * 40)


def test_freeze_release_rejects_noncompleted_batch_evidence(tmp_path: Path) -> None:
    inventory, status, experiments, output_root = release_fixture(tmp_path)
    fetch = next(output_root.glob("tt12300742/review/openai/production_v3_2_1_retrieval_v3/result*/fetch_metadata.json"))
    value = json.loads(fetch.read_text()); value["status"] = "in_progress"; write_json(fetch, value)

    with pytest.raises(RuntimeError, match="completed fetch evidence"):
        freeze_stage2_release(inventory, status, experiments, output_root, output_root / "release", "d" * 40)


def test_frozen_release_resume_supports_explicit_historical_mount_map(tmp_path: Path) -> None:
    inventory, status, experiments, output_root = release_fixture(tmp_path)
    release_dir = output_root / "release"
    commit = "e" * 40
    freeze_stage2_release(inventory, status, experiments, output_root, release_dir, commit)

    historical_root = "/historical/processed"
    actual_root = output_root.as_posix()
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for movie in manifest["movies"]:
        for group in ("artifacts", "protected_artifacts"):
            for artifact in movie.get(group, {}).values():
                artifact["path"] = artifact["path"].replace(actual_root, historical_root, 1)
                artifact["declared_path"] = artifact["path"]
    for artifact in (manifest["pending_human_ambiguities"], manifest["stage3_handoff"]):
        artifact["path"] = artifact["path"].replace(actual_root, historical_root, 1)
        artifact["declared_path"] = artifact["path"]
    write_json(manifest_path, manifest)
    before = manifest_path.read_bytes()

    resumed = freeze_stage2_release(
        inventory, status, experiments, output_root, release_dir, commit,
        path_maps=parse_path_maps([f"{historical_root}={output_root}"]),
    )

    assert resumed["resumed"] is True
    assert manifest_path.read_bytes() == before
