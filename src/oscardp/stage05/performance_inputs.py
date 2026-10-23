"""Read-only join of corrected observations and tentative identity provenance."""
from pathlib import Path

import numpy as np

from .identity_inputs import file_hash, load_corrected
from .output import identity, load_json


def snapshot(paths):
    return {str(p.resolve()): file_hash(p) for p in paths if p.is_file()}


def load_inputs(tracking_run, identity_run, movie_id, shot_ids):
    source, tracks, observations, tracking_files = load_corrected(tracking_run, movie_id, shot_ids)
    config = load_json(identity_run/"run_config.json")
    result = load_json(identity_run/"identity_clusters.json")
    qc = load_json(identity_run/"qc_report.json")
    if (result.get("movie_id") != movie_id or config.get("movie_id") != movie_id
            or Path(config["source_corrected_run"]).resolve() != tracking_run.resolve()
            or Path(result["source_corrected_run"]).resolve() != tracking_run.resolve()):
        raise ValueError("Identity/tracking movie or source run mismatch")
    for document in [config, result]:
        if document["source_files"] != tracking_files or document["boundary_sidecar"] != source["boundary_sidecar"]:
            raise ValueError("Identity/corrected tracking provenance mismatch")
    if qc["boundary_sidecar"] != source["boundary_sidecar"] or not qc["source_hashes_unchanged"]:
        raise ValueError("Identity QC provenance mismatch")
    if result.get("human_verified") is not False:
        raise ValueError("Pilot identity verification status must remain tentative")
    if set(config["selected_shot_ids"]) != set(shot_ids) or set(config["track_ids"]) != set(tracks):
        raise ValueError("Identity Track selection mismatch")
    if set(result["track_representations"]) != set(tracks):
        raise ValueError("Identity Track representation mismatch")
    mapping = {}
    for cluster in result["clusters"]:
        if (cluster.get("human_verified") is not False or cluster.get("verified_identity") is not False
                or cluster["status"] not in {"tentative_identity_cluster", "unknown"}):
            raise ValueError("Expected tentative, unverified identity clusters")
        for tid in cluster["track_ids"]:
            if tid not in tracks or tid in mapping:
                raise ValueError("Unknown/duplicate identity Track")
            mapping[tid] = {"identity_cluster_id": cluster["cluster_id"], "identity_status": cluster["status"],
                            "identity_human_verified": False}
    if set(mapping) != set(tracks):
        raise ValueError("Identity cluster missing a Track")
    for tid, track in tracks.items():
        info = result["track_representations"][tid]
        if any(info[k] != track[k] for k in ["shot_id", "original_temporal", "effective_processing_temporal"]):
            raise ValueError("Identity effective/original Track range mismatch")
    if result["model"] != config["model"]:
        raise ValueError("Identity model provenance mismatch")
    with np.load(identity_run/"track_embeddings.npz", allow_pickle=False) as data:
        if str(data["model_sha256"]) != config["model"]["sha256"] or set(data["track_ids"]) != set(tracks):
            raise ValueError("Identity NPZ model/Track mismatch")
        if not np.isfinite(data["track_embeddings"]).all():
            raise ValueError("Invalid source identity embeddings")
    for sid in shot_ids:
        if sum(t["shot_id"] == sid for t in tracks.values()) != 1:
            raise ValueError("Single-person pilot requires exactly one source Track per Shot")
    earlier = Path(source["stage01"]["path"]).parent.glob(f"Stage0[1-4]_{movie_id}.json")
    paths = [p for root in [tracking_run, identity_run] for p in root.rglob("*") if p.is_file()]
    paths += list(earlier)+[Path(source["boundary_sidecar"]["path"]), Path(source["boundary_sidecar"]["audit"]["path"])]
    protected = snapshot(paths)
    provenance = {"tracking_run": str(tracking_run.resolve()), "tracking_files": tracking_files,
                  "identity_run": str(identity_run.resolve()),
                  "identity_files": {p.name: {**identity(p), "sha256": file_hash(p)} for p in identity_run.iterdir() if p.is_file()},
                  "boundary_sidecar": source["boundary_sidecar"], "video": source["video"],
                  "protected_source_sha256": protected}
    return source, tracks, observations, mapping, provenance
