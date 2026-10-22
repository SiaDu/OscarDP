"""Explicit, source-bound processing overrides; never edit Stage01 records."""
import copy
import hashlib
import json
import math
from importlib.resources import files
from pathlib import Path

import jsonschema

from .output import identity, load_json


def apply_sidecar(args, stage01, source, video, metadata, cfr):
    path = getattr(args, "boundary_sidecar", None)
    opt_in = getattr(args, "allow_experimental_boundaries", False)
    if path is None:
        if opt_in:
            raise ValueError("--allow-experimental-boundaries requires --boundary-sidecar")
        return None, None
    path = path.resolve(strict=True)
    sidecar = load_json(path)
    schema = json.loads(files("oscardp.stage05").joinpath("boundary_correction_v1.schema.json").read_text())
    jsonschema.validate(sidecar, schema)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    source_ref = sidecar["source_stage01"]
    if (sidecar["movie_id"] != args.movie_id or source_ref["sha256"] != sha
            or Path(source_ref["path"]).resolve() != source.resolve()):
        raise ValueError("Boundary sidecar Stage01 identity/movie_id/SHA256 mismatch")
    if sidecar["source_video"] != identity(video):
        raise ValueError("Boundary sidecar source video identity mismatch")
    rule = "frame_index_cfr" if cfr else "decoded_pts_relative_to_first_frame"
    if sidecar["timestamp_rule"] != rule:
        raise ValueError("Boundary sidecar timestamp rule mismatch")
    audit_path = (path.parent/sidecar["audit_manifest"]).resolve(strict=True)
    audit = load_json(audit_path)
    if (audit["movie_id"] != args.movie_id or audit["source_stage01"]["sha256"] != sha
            or audit["source_video"] != identity(video) or audit["decoder_comparison"]["status"] != "passed"):
        raise ValueError("Boundary audit provenance/decoder verification mismatch")
    reviews = {b["boundary_id"]: b for b in audit["boundary_reviews"]}
    if len(reviews) != len(audit["boundary_reviews"]):
        raise ValueError("Duplicate boundary IDs in audit evidence")
    ordered = stage01["shots"]
    lookup = {s["shot_id"]: s for s in ordered}
    movie_end = ordered[-1]["temporal"]["end_frame"]
    changes = {}
    experimental = False
    for correction in sidecar["corrections"]:
        sid = correction["shot_id"]
        if sid in changes or sid not in lookup:
            raise ValueError(f"Duplicate/unknown sidecar Shot: {sid}")
        original = lookup[sid]["temporal"]
        if correction["original_temporal"] != original:
            raise ValueError(f"Boundary sidecar original temporal mismatch: {sid}")
        review = correction["review"]
        approved = review["status"] == "approved" and review["approved_for_processing"] and review["human_verified"]
        pending = review["status"] == "model_reviewed_pending_human" and not review["approved_for_processing"] and not review["human_verified"]
        if not approved:
            if not opt_in or not pending:
                raise ValueError(f"Unapproved boundary correction {sid}; model-reviewed drafts require --allow-experimental-boundaries")
            experimental = True
        effective = correction["proposed_processing_temporal"]
        a, b = effective["start_frame"], effective["end_frame"]
        if (type(a) is not int or type(b) is not int or not 0 <= a < b <= movie_end
                or (metadata.frame_count is not None and b > metadata.frame_count)):
            raise ValueError(f"Invalid corrected inclusive/exclusive frame range: {sid}")
        if not all(math.isfinite(effective[k]) for k in ["start", "end"]) or not 0 <= effective["start"] < effective["end"] <= metadata.duration_sec+.1:
            raise ValueError(f"Invalid corrected time range: {sid}")
        evidence_frames = correction["evidence"]["source_frame_indices"]
        if any(type(n) is not int or not 0 <= n < movie_end for n in evidence_frames):
            raise ValueError(f"Audited source frame indices out of range: {sid}")
        for edge in ["start", "end"]:
            old_frame, new_frame = original[f"{edge}_frame"], effective[f"{edge}_frame"]
            if correction["boundary_changes"][f"{edge}_delta_frames"] != new_frame-old_frame:
                raise ValueError(f"Sidecar delta mismatch: {sid} {edge}")
            evidence_id = correction["evidence"][f"{edge}_boundary_id"]
            evidence = reviews.get(evidence_id)
            if evidence is None or evidence["frame"] != old_frame or evidence["first_new_view_frame"] != new_frame:
                raise ValueError(f"Audited boundary evidence mismatch: {sid} {edge}")
            required_frames = {n if n < movie_end else movie_end-1 for n in [old_frame, new_frame]}
            if not required_frames <= set(evidence_frames):
                raise ValueError(f"Audited frame evidence missing for {sid} {edge}")
            if cfr:
                expected_time = new_frame/metadata.fps
            elif new_frame == movie_end:
                expected_time = ordered[-1]["temporal"]["end"]
            else:
                decoder_path = (audit_path.parent/audit["decoder_evidence"]).resolve(strict=True)
                expected_time = load_json(decoder_path)["frames"][str(new_frame)]["reference_pts_sec"]
            if abs(expected_time-effective[edge]) > 1e-6:
                raise ValueError(f"Corrected boundary timestamp mismatch: {sid} {edge}")
        changes[sid] = correction
    # Check the full declared plan, not just today's selected subset.
    effective_all = [changes[s["shot_id"]]["proposed_processing_temporal"] for s in ordered if s["shot_id"] in changes]
    if any(left["end_frame"] > right["start_frame"] for left, right in zip(effective_all, effective_all[1:])):
        raise ValueError("Conflicting corrected ranges overlap")
    for left, right in zip(ordered, ordered[1:]):
        lsid, rsid = left["shot_id"], right["shot_id"]
        if lsid in changes and rsid in changes:
            if changes[lsid]["proposed_processing_temporal"]["end_frame"] != changes[rsid]["proposed_processing_temporal"]["start_frame"]:
                raise ValueError("Adjacent corrected shots have a gap/conflicting boundary")
    if experimental:
        root = (args.output_root/args.movie_id).resolve()
        if (args.output_root.resolve() == Path("reports/stage05_2").resolve()
                or root.is_relative_to(args.analysis_root.resolve())):
            raise ValueError("Experimental boundaries require an independent experimental output directory")
        old = root/"run_config.json"
        if old.exists() and not load_json(old).get("boundary_sidecar"):
            raise ValueError("Cannot reuse/overwrite an original-range run for an experimental correction")
    selected = set(args.shot_ids)
    shots = []
    for s in ordered:
        if s["shot_id"] not in selected:
            continue
        original = copy.deepcopy(s["temporal"])
        effective = copy.deepcopy(changes[s["shot_id"]]["proposed_processing_temporal"] if s["shot_id"] in changes else original)
        shots.append(dict(s, temporal=effective, original_temporal=original, effective_processing_temporal=effective))
    # Selected ranges must not overlap, including a corrected/uncorrected mix.
    for left, right in zip(shots, shots[1:]):
        if left["temporal"]["end_frame"] > right["temporal"]["start_frame"]:
            raise ValueError("Selected effective ranges overlap; uncorrected selected neighbor would be processed twice")
    conflicts = []
    for i, s in enumerate(ordered):
        sid = s["shot_id"]
        if sid not in selected or sid not in changes:
            continue
        effective = changes[sid]["proposed_processing_temporal"]
        for neighbor_index, edge in [(i-1, "start"), (i+1, "end")]:
            if not 0 <= neighbor_index < len(ordered):
                continue
            neighbor = ordered[neighbor_index]
            nid = neighbor["shot_id"]
            # A correction only takes effect when that neighbor is also selected.
            other = changes[nid]["proposed_processing_temporal"] if nid in selected and nid in changes else neighbor["temporal"]
            delta = effective["start_frame"]-other["end_frame"] if edge == "start" else other["start_frame"]-effective["end_frame"]
            if delta:
                conflicts.append({"shot_id": sid, "edge": edge, "neighbor_shot_id": nid,
                                  "neighbor_selected": nid in selected, "kind": "gap" if delta > 0 else "overlap",
                                  "frame_count": abs(delta), "neighbor_modified": False})
    meta = {**identity(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "audit": {**identity(audit_path), "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest()},
            "mode": "experimental_model_reviewed_pending_human" if experimental else "approved",
            "allow_experimental_boundaries": opt_in, "human_verified": not experimental,
            "applied_shot_ids": [s["shot_id"] for s in shots if s["shot_id"] in changes],
            "unchanged_selected_shot_ids": [s["shot_id"] for s in shots if s["shot_id"] not in changes],
            "reviews": {s["shot_id"]: changes[s["shot_id"]]["review"] for s in shots if s["shot_id"] in changes},
            "neighbor_conflicts": conflicts}
    return shots, meta
