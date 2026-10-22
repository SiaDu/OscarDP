import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from oscardp.shots.media import probe_video, iter_frame_timestamps, build_timeline
from oscardp.stage05 import pipeline
from oscardp.stage05.__main__ import parser
from oscardp.stage05.boundaries import apply_sidecar
from oscardp.stage05.output import atomic_json, identity, load_json
from oscardp.stage05.video_reader import iter_frames


@pytest.fixture
def case(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    video = tmp_path/"movie.mkv"
    pixels = [np.full((64, 96, 3), 30+i*5, np.uint8) for i in range(12)]
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "96x64",
                    "-framerate", "25", "-i", "pipe:0", "-c:v", "ffv1", str(video)],
                   input=b"".join(a.tobytes() for a in pixels), check=True)
    movie_id = "tt12300742"
    analysis = tmp_path/"analysis"
    source = analysis/movie_id/f"Stage01_{movie_id}.json"
    source.parent.mkdir(parents=True)
    stage01 = {"movie_id": movie_id, "shots": [{"shot_id": f"shot_{i+1:06d}",
               "temporal": {"start_frame": i*3, "end_frame": i*3+3, "start": i*3/25, "end": (i*3+3)/25}} for i in range(4)]}
    atomic_json(source, stage01)
    model = tmp_path/"model.onnx"
    model.write_bytes(b"mock model")
    args = parser().parse_args(["detect-track", "--movie-id", movie_id, "--analysis-root", str(analysis),
                               "--video", str(video), "--shot-ids", "shot_000002", "shot_000003", "--model", str(model),
                               "--output-root", str(tmp_path/"experiment"), "--allow-experimental-boundaries"])
    sidecar_path = tmp_path/"corrections.json"
    args.boundary_sidecar = sidecar_path
    source_ref = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    audit_path = tmp_path/"audit.json"
    audit = {"movie_id": movie_id, "source_stage01": source_ref, "source_video": identity(video),
             "decoder_comparison": {"status": "passed"},
             "boundary_reviews": [{"boundary_id": f"frame_{n:08d}", "frame": n, "first_new_view_frame": n+1} for n in [3,6,9]]}
    atomic_json(audit_path, audit)
    changes = []
    for i in [1,2]:
        original = stage01["shots"][i]["temporal"]
        a,b = original["start_frame"], original["end_frame"]
        changes.append({"shot_id": f"shot_{i+1:06d}", "original_temporal": original,
                        "proposed_processing_temporal": {"start_frame": a+1, "end_frame": b+1, "start": (a+1)/25, "end": (b+1)/25},
                        "boundary_changes": {"start_delta_frames": 1, "end_delta_frames": 1}, "reason": "synthetic audited correction",
                        "evidence": {"basis": "source_video_frames_with_independent_ordinal_and_decoder_checks",
                                     "start_boundary_id": f"frame_{a:08d}", "end_boundary_id": f"frame_{b:08d}",
                                     "source_frame_indices": [a-1,a,a+1,b-1,b,b+1]},
                        "review": {"status": "model_reviewed_pending_human", "human_verified": False,
                                   "approved_for_processing": False, "reviewer": "test model", "reviewed_on": "2026-10-22"},
                        "neighbor_effect": "report only"})
    sidecar = {"schema_version": "stage05.boundary_correction.v1", "movie_id": movie_id,
               "application_status": "proposal_only_not_applied", "scope": "explicit_audited_shots_only_no_global_offset",
               "frame_semantics": "zero_based_start_inclusive_end_exclusive", "timestamp_rule": "frame_index_cfr",
               "source_stage01": source_ref, "source_video": identity(video), "audit_manifest": "audit.json", "corrections": changes}
    atomic_json(sidecar_path, sidecar)
    class Detector:
        def __init__(self, *a, **kw):
            pass
        def detect(self, image):
            return [{"bbox_xywh": [10.,10.,40.,40.], "detection_score": .95, "landmarks5": None}]
    monkeypatch.setattr(pipeline, "YuNetDetector", Detector)
    return args, stage01, source, video, sidecar, audit


def preflight(case):
    args, *_ = case
    return pipeline.preflight(args)


def test_without_sidecar_preserves_old_config_and_ranges(case):
    args, stage01, *_ = case
    args.boundary_sidecar = None
    args.allow_experimental_boundaries = False
    _, shots, _, _, _, config = preflight(case)
    assert shots == stage01["shots"][1:3]
    assert config["implementation_version"] == 2 and "boundary_sidecar" not in config
    assert all("original_temporal" not in shot for shot in shots)


def test_pending_requires_opt_in_and_independent_directory(case):
    args, *_ = case
    args.allow_experimental_boundaries = False
    with pytest.raises(ValueError, match="Unapproved boundary"):
        preflight(case)
    args.allow_experimental_boundaries = True
    args.output_root = Path("reports/stage05_2")
    with pytest.raises(ValueError, match="independent experimental"):
        preflight(case)


@pytest.mark.parametrize("field", ["movie_id", "sha256", "temporal", "video", "timestamp", "schema", "audit"])
def test_invalid_sidecars_fail(case, field):
    args, _, _, _, sidecar, audit = case
    if field == "movie_id": sidecar["movie_id"] = "tt0000000"
    if field == "sha256": sidecar["source_stage01"]["sha256"] = "bad"
    if field == "temporal": sidecar["corrections"][0]["original_temporal"]["start"] = 0
    if field == "video": sidecar["source_video"]["size"] += 1
    if field == "timestamp": sidecar["corrections"][0]["proposed_processing_temporal"]["end"] += .01
    if field == "schema": del sidecar["frame_semantics"]
    if field == "audit":
        audit["boundary_reviews"][0]["first_new_view_frame"] += 1
        atomic_json(args.boundary_sidecar.parent/"audit.json", audit)
    atomic_json(args.boundary_sidecar, sidecar)
    with pytest.raises((ValueError, __import__("jsonschema").ValidationError)):
        preflight(case)


def test_effective_ranges_decode_and_output_with_no_duplicate_frames(case):
    args, stage01, source, video, *_ = case
    original_bytes = source.read_bytes()
    _, shots, metadata, cfr, _, config = preflight(case)
    frames = list(iter_frames(video, shots, metadata, cfr))
    assert [f.index for f in frames] == list(range(4,10))
    assert frames[0].timestamp == .16 and frames[-1].next_timestamp == .4
    assert pipeline.run(args) == 0
    root = args.output_root/args.movie_id
    rows = [json.loads(line) for line in (root/"face_detections.jsonl").read_text().splitlines()]
    assert [r["frame_index"] for r in rows] == list(range(4,10))
    assert len({r["frame_index"] for r in rows}) == 6
    for row in rows:
        assert row["timestamp_sec"] == row["frame_index"]/25
        s = next(s for s in shots if s["shot_id"] == row["shot_id"])
        assert row["original_temporal"] == s["original_temporal"]
        assert row["effective_processing_temporal"] == s["temporal"]
    assert len(load_json(root/"face_tracks.json")["shot_ranges"]) == 2
    qc = load_json(root/"qc_report.json")
    assert {c["kind"] for c in qc["boundary_sidecar"]["neighbor_conflicts"]} == {"gap", "overlap"}
    assert all(not c["neighbor_modified"] for c in qc["boundary_sidecar"]["neighbor_conflicts"])
    assert source.read_bytes() == original_bytes and stage01 == load_json(source)


def test_sidecar_change_and_old_receipts_cannot_resume(case):
    args, _, _, _, sidecar, _ = case
    assert pipeline.run(args) == 0
    args.resume = True
    assert pipeline.run(args) == 0
    receipt_path = args.output_root/args.movie_id/"checkpoints/shot_000002.json"
    receipt = load_json(receipt_path)
    del receipt["processing_config_sha256"]
    atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="Checkpoint belongs"):
        pipeline.run(args)
    # Even a semantics-equivalent second sidecar has a different file identity.
    other = args.boundary_sidecar.with_name("different.json")
    atomic_json(other, sidecar)
    args.boundary_sidecar = other
    with pytest.raises(ValueError, match="configuration changed"):
        pipeline.run(args)


def test_unlisted_shot_is_not_shifted(case):
    args, stage01, *_ = case
    args.shot_ids = ["shot_000001", "shot_000002", "shot_000003"]
    _, shots, _, _, _, config = preflight(case)
    assert shots[0]["temporal"] == stage01["shots"][0]["temporal"]
    assert config["boundary_sidecar"]["unchanged_selected_shot_ids"] == ["shot_000001"]


def test_selected_uncorrected_overlap_is_rejected(case):
    args, *_ = case
    args.shot_ids.append("shot_000004")
    with pytest.raises(ValueError, match="Selected effective ranges overlap"):
        preflight(case)


def test_adjacent_corrections_cannot_have_gap(case):
    args, _, _, _, sidecar, audit = case
    # Individually audited endpoints still must agree at their shared boundary.
    sidecar["corrections"][0]["proposed_processing_temporal"].update(end_frame=6, end=6/25)
    sidecar["corrections"][0]["boundary_changes"]["end_delta_frames"] = 0
    audit["boundary_reviews"].append({"boundary_id": "alternate", "frame": 6, "first_new_view_frame": 6})
    sidecar["corrections"][0]["evidence"]["end_boundary_id"] = "alternate"
    atomic_json(args.boundary_sidecar, sidecar)
    atomic_json(args.boundary_sidecar.parent/"audit.json", audit)
    with pytest.raises(ValueError, match="Adjacent corrected shots have a gap"):
        preflight(case)


def test_packaged_schema_matches_design_contract():
    root = Path(__file__).parents[1]
    assert load_json(root/"src/oscardp/stage05/boundary_correction_v1.schema.json") == load_json(root/"docs/schemas/stage05_boundary_correction_v1.schema.json")


def test_vfr_and_eof_ranges_keep_decoded_pts(case):
    args, stage01, source, video, sidecar, audit = case
    replacement = video.with_name("vfr.mkv")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", "setpts=if(lt(N\\,4)\\,N\\,2*N-4)/(25*TB)",
                    "-vsync", "0", "-c:v", "ffv1", str(replacement)], check=True)
    video.write_bytes(replacement.read_bytes())
    metadata, _ = probe_video(video, video.name)
    timeline = build_timeline(list(iter_frame_timestamps(video)), metadata.fps)
    assert timeline.is_vfr
    for s in stage01["shots"]:
        t = s["temporal"]
        t.update(start=timeline.timestamp(t["start_frame"]), end=timeline.exclusive_timestamp(t["end_frame"]))
    atomic_json(source, stage01)
    sidecar["source_stage01"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    sidecar["source_video"] = identity(video)
    sidecar["timestamp_rule"] = "decoded_pts_relative_to_first_frame"
    audit.update(source_video=identity(video), source_stage01=sidecar["source_stage01"], decoder_evidence="pts.json")
    atomic_json(args.boundary_sidecar.parent/"pts.json", {"frames": {str(i): {"reference_pts_sec": timeline.timestamp(i)} for i in range(12)}})
    # One corrected final shot, ending at the original movie EOF, never EOF+1.
    change = copy.deepcopy(sidecar["corrections"][0])
    change.update(shot_id="shot_000004", original_temporal=stage01["shots"][3]["temporal"])
    change["proposed_processing_temporal"] = {"start_frame": 10, "end_frame": 12,
                                               "start": timeline.timestamp(10), "end": timeline.exclusive_end_sec}
    change["boundary_changes"] = {"start_delta_frames": 1, "end_delta_frames": 0}
    change["evidence"].update(start_boundary_id="frame_00000009", end_boundary_id="eof",
                                source_frame_indices=[6,7,8,9,10,11])
    audit["boundary_reviews"].append({"boundary_id": "eof", "frame": 12, "first_new_view_frame": 12})
    sidecar["corrections"] = [change]
    atomic_json(args.boundary_sidecar, sidecar)
    atomic_json(args.boundary_sidecar.parent/"audit.json", audit)
    args.shot_ids = ["shot_000004"]
    _, shots, metadata, cfr, _, config = preflight(case)
    assert not cfr
    frames = list(iter_frames(video, shots, metadata, cfr))
    assert [f.index for f in frames] == [10,11]
    assert frames[0].timestamp == pytest.approx(timeline.timestamp(10))
    assert frames[-1].next_timestamp == pytest.approx(timeline.exclusive_end_sec)
    assert pipeline.run(args) == 0
    bad = load_json(args.boundary_sidecar)
    bad["corrections"][0]["proposed_processing_temporal"]["end_frame"] = 13
    atomic_json(args.boundary_sidecar, bad)
    with pytest.raises(ValueError, match="Invalid corrected"):
        preflight(case)
