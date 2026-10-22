"""Build review artifacts from measured decode evidence and explicit visual reviews.

This script does not discover cuts, infer review decisions, apply corrections,
rewrite old QC artifacts, or modify Stage01–04. Visual decisions are supplied in
review_annotations.json after inspecting the actual evidence images.
"""
import argparse
import hashlib
import subprocess
from pathlib import Path

import jsonschema

from oscardp.stage05.output import atomic_json, identity, load_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--review-file", type=Path, required=True)
    p.add_argument("--legacy-qc", type=Path, required=True)
    p.add_argument("--proposal-shot-ids", nargs="+", default=[])
    args = p.parse_args()
    root = args.audit_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    config = load_json(root/"audit_config.json")
    compare = load_json(root/"decoder_comparison.json")
    reference = load_json(root/"reference_frames.json")
    reviews = load_json(args.review_file)
    if compare["status"] != "passed":
        raise ValueError("Independent decoder comparison must pass before report generation")
    boundaries = config["boundaries"]
    if len({b["frame"] for b in boundaries}) != len(boundaries):
        raise ValueError("Review dataset must identify unique physical boundaries")
    reviewed = {r["original_boundary_frame"]: r for r in reviews["boundaries"]}
    if set(reviewed) != {b["frame"] for b in boundaries}:
        raise ValueError("Visual review coverage does not match audit boundaries")
    for name, source in config["sources"].items():
        if sha(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"Source changed during audit: {name}")
    if identity(Path(config["video"]["path"])) != config["video"]:
        raise ValueError("Video identity changed during audit")
    for r in reviewed.values():
        if r["review_status"] != "model_reviewed_pending_human" or r["human_verified"] is not False:
            raise ValueError("This report builder accepts only the documented model-review evidence")
        n = r["first_new_view_frame"]
        if n is not None and str(n) not in compare["frames"]:
            raise ValueError("Visual review refers to a frame that was not actually extracted")
    movie_id = config["movie_id"]
    stage01 = config["sources"][f"Stage01_{movie_id}.json"]
    lookup = {s["shot_id"]: s for s in load_json(Path(stage01["path"]))["shots"]}
    manifest_path = root/f"Stage05_{movie_id}_boundary_audit.json"
    correction_path = root/f"Stage05_{movie_id}_boundary_corrections.proposed.json"
    erratum_path = root/"legacy_qc_provenance_review.json"
    for path in [manifest_path, correction_path, erratum_path, root/"boundary_audit_report.md"]:
        if path.exists():
            raise FileExistsError(f"Preserve existing review result; choose a new audit directory: {path}")
    code_paths = [repo/"src/oscardp/shots/transnet.py", repo/"src/oscardp/shots/pipeline.py",
                  repo/"src/oscardp/shots/media.py", repo/"src/oscardp/stage05/video_reader.py",
                  repo/"src/oscardp/stage05/boundary_audit.py", Path(__file__).resolve()]
    code = [{"path": str(path.relative_to(repo)), "sha256": sha(path)} for path in code_paths]
    fingerprint_path = Path(stage01["path"]).parents[1]/".internal"/movie_id/"fingerprint.json"
    old_metadata_path = fingerprint_path.with_name("video_metadata.json")
    historical = {"fingerprint": load_json(fingerprint_path), "video_metadata": load_json(old_metadata_path),
                  "historical_prediction_arrays": "Not retained in inspected .internal/.work directories; no TransNet rerun was performed."}
    old = historical["fingerprint"]
    if (old["source_path"], old["source_size"], old["source_mtime_ns"]) != (
            config["video"]["path"], config["video"]["size"], config["video"]["mtime_ns"]):
        raise ValueError("Historical Stage01 source identity differs from the audited video")
    audit = {"schema_version": "stage05.boundary_audit.v1", "movie_id": movie_id,
             "basis": "source_video_frames_with_independent_ordinal_and_decoder_checks",
             "review_status": "model_reviewed_pending_human", "human_verified": False,
             "source_stage01": stage01, "source_video": config["video"], "historical_stage01_provenance": historical,
             "source_stage01_to_04_hashes_unchanged": True, "code_provenance": code,
             "selection_summary": {"start_boundary_count": sum(b["edge"] == "start" for b in boundaries),
                                   "end_boundary_count": sum(b["edge"] == "end" for b in boundaries),
                                   "distinct_scene_contexts": len({b["scene_context"].split(":")[0] for b in boundaries}),
                                   "source_frame_count": compare["frame_count"],
                                   "reviewed_first_new_at_N_plus_1_count": sum(r["first_new_view_frame"] == n+1 for n, r in reviewed.items())},
             "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
             "upstream": load_json(root/"upstream/sources.json"),
             "decoder_comparison": {k: v for k, v in compare.items() if k != "frames"},
             "reference_decode": {k: v for k, v in reference.items() if k != "frames"},
             "reference_evidence": "reference_frames.json", "decoder_evidence": "decoder_comparison.json",
             "contact_sheets": [str(path.relative_to(root)) for path in sorted((root/"contact_sheets").glob("*.jpg"))],
             "boundary_reviews": [dict(b, boundary_id=f"frame_{b['frame']:08d}",
                                       **{k: v for k, v in reviewed[b["frame"]].items() if k != "original_boundary_frame"}) for b in boundaries],
             "interpretation_limits": ["Both decode paths share FFmpeg codecs; this is an application-index audit.",
                                       "Only selected boundary windows were reviewed, not every shot interior.",
                                       "Hard-cut label semantics do not imply every prediction peak is a correct cut.",
                                       "No global offset or original-data correction has been applied."]}
    changes = []
    for sid in args.proposal_shot_ids:
        original = lookup[sid]["temporal"]
        a, b = original["start_frame"], original["end_frame"]
        if a not in reviewed or b not in reviewed:
            raise ValueError(f"Both edges require independent review: {sid}")
        start = reviewed[a]["first_new_view_frame"]
        end = reviewed[b]["first_new_view_frame"]
        if start is None or end is None or not 0 <= start < end <= historical["video_metadata"]["frame_count"]:
            raise ValueError(f"Unresolved/out-of-range proposed processing interval: {sid}")
        fps = config["fps"]
        cfr = historical["video_metadata"]["timestamp_source"] == "frame_index_cfr"
        def timestamp(frame):
            return frame/fps if cfr else compare["frames"][str(frame)]["reference_pts_sec"]
        changes.append({"shot_id": sid, "original_temporal": original,
                        "proposed_processing_temporal": {"start": timestamp(start), "end": timestamp(end),
                                                         "start_frame": start, "end_frame": end},
                        "boundary_changes": {"start_delta_frames": start-a, "end_delta_frames": end-b},
                        "reason": "Each original edge is the last old-view frame in the reviewed source window; proposed boundaries use the model-reviewed first new-view frame.",
                        "evidence": {"basis": audit["basis"], "start_boundary_id": f"frame_{a:08d}",
                                     "end_boundary_id": f"frame_{b:08d}", "source_frame_indices": sorted({a-1,a,a+1,b-1,b,b+1})},
                        "review": {"status": "model_reviewed_pending_human", "reviewer": reviews["reviewer"],
                                   "reviewed_on": reviews["reviewed_on"], "human_verified": False, "approved_for_processing": False},
                        "neighbor_effect": "Original neighbors remain unchanged. If only this proposal is later applied, it excludes the original first frame and includes the frame at the original exclusive end; coordinate with any adjacent selected-shot proposal to avoid overlaps."})
    proposal = {"schema_version": "stage05.boundary_correction.v1", "movie_id": movie_id,
                "application_status": "proposal_only_not_applied", "scope": "explicit_audited_shots_only_no_global_offset",
                "frame_semantics": "zero_based_start_inclusive_end_exclusive",
                "timestamp_rule": historical["video_metadata"]["timestamp_source"],
                "source_stage01": stage01, "source_video": config["video"],
                "audit_manifest": manifest_path.name, "corrections": changes}
    jsonschema.validate(proposal, load_json(repo/"docs/schemas/stage05_boundary_correction_v1.schema.json"))
    legacy = load_json(args.legacy_qc)
    jsonschema.validate(legacy, load_json(repo/"docs/schemas/shot_boundary_qc_v1.schema.json"))
    erratum = {"schema_version": "stage05.qc_provenance_review.v1", "legacy_file": identity(args.legacy_qc),
               "legacy_sha256": sha(args.legacy_qc), "legacy_schema_valid": True, "legacy_file_modified": False,
               "assessment": "Notes disclose Stage05 preview sampling, but the Stage03 filename, sampled_images basis and Early/Middle/Late roles can be confused with original Stage01 representative JPG evidence.",
               "actual_legacy_evidence": "Stage05 low-resolution QC video sampled first/middle/late frames; not the original Stage03 image triplet.",
               "new_evidence": {"manifest": manifest_path.name, "basis": audit["basis"]},
               "consumption_policy": "Use the new Stage05 audit for video-boundary evidence; do not merge this legacy file into Stage03 representative-image QC. Preserve the original as history. Neither old visual discontinuity nor decoder index verification alone establishes a human-confirmed interior cut."}
    atomic_json(manifest_path, audit)
    atomic_json(correction_path, proposal)
    atomic_json(erratum_path, erratum)
    print(manifest_path)
    print(correction_path)


if __name__ == "__main__":
    main()
