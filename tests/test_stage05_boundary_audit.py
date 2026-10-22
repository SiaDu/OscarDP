import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from oscardp.shots.media import probe_video
from oscardp.shots.pipeline import build_shot_records
from oscardp.shots.schema import FrameTimeline, MovieRef, VideoMetadata
from oscardp.shots.transnet import predictions_to_boundaries
from oscardp.stage05.boundary_audit import reference_decode, stage05_decode, selected_indices


def test_hard_cut_last_old_label_is_not_next_inclusive_start():
    # Official hard-cut interpolation uses range <= transition_boundary for old
    # footage; this audit demonstrates the current adapter's semantic mismatch.
    n = 3
    old_mask = np.arange(10) <= n
    assert old_mask[n] and not old_mask[n+1]
    predictions = [0.]*10
    predictions[n] = .99
    boundaries = predictions_to_boundaries(predictions)
    timeline = FrameTimeline([i/25 for i in range(10)], [.04]*10, False, 25, .4)
    metadata = VideoMetadata("test.mkv", .4, 96, 64, "ffv1", 25, 10, False, "frame_index_cfr")
    records = build_shot_records(MovieRef("tt12300742", Path("test.mkv"), "test.mkv"), metadata, timeline, boundaries, .5)
    assert records[1].start_frame == n
    assert records[0].end_frame == n
    assert old_mask[records[1].start_frame]  # Current behavior; not a fix applied by audit.


def test_reference_ordinals_and_stage05_pixels_independent(tmp_path):
    video = tmp_path/"clip.mkv"
    # Every frame has distinct native pixels and a hard cut near the boundary.
    arrays = [np.full((64, 96, 3), (i*19, 20 if i < 5 else 240, 255-i*15), dtype=np.uint8) for i in range(12)]
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
                    "-video_size", "96x64", "-framerate", "25", "-i", "pipe:0", "-c:v", "ffv1", str(video)],
                   input=b"".join(a.tobytes() for a in arrays), check=True)
    root = tmp_path/"audit"
    root.mkdir()
    (root/"frames").mkdir()
    metadata, _ = probe_video(video, video.name)
    indices = selected_indices([{"frame": 4}])
    frames, count = reference_decode(video, metadata, indices, root)
    assert count == 8
    from oscardp.stage05.boundary_audit import digest
    for index in indices:
        assert frames[index]["ordinal_verified"]
        assert frames[index]["native_bgr_sha256"] == digest(arrays[index].tobytes())
    stage05_decode(video, metadata, indices, frames, root, True)
    assert all(f["stage05_native_pixels_match"] for f in frames.values())
    assert all(f["timestamp_error_sec"] < 1e-6 for f in frames.values())
    assert not (root/"selected_native.bgr.tmp").exists()


def test_mismatched_reference_pixels_fail(tmp_path):
    from oscardp.stage05 import boundary_audit
    from oscardp.stage05.video_reader import Frame
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    reference = {1: {"native_bgr_sha256": "wrong", "reference_pts_sec": .04}}
    from unittest.mock import patch
    metadata = VideoMetadata("test", .2, 1, 1, "raw", 25, 5, False, "frame_index_cfr")
    with patch.object(boundary_audit, "iter_frames", return_value=iter([Frame(1, .04, .08, image)])):
        with pytest.raises(RuntimeError, match="native pixel mismatch"):
            stage05_decode(Path("unused"), metadata, [1], reference, tmp_path, True)


def test_proposal_schema_refuses_model_self_approval():
    import copy
    import json
    import jsonschema
    schema = json.loads((Path(__file__).parents[1]/"docs/schemas/stage05_boundary_correction_v1.schema.json").read_text())
    temporal = {"start": 1., "end": 2., "start_frame": 25, "end_frame": 50}
    proposal = {"schema_version": "stage05.boundary_correction.v1", "movie_id": "tt12300742",
                "application_status": "proposal_only_not_applied", "scope": "explicit_audited_shots_only_no_global_offset",
                "frame_semantics": "zero_based_start_inclusive_end_exclusive", "timestamp_rule": "frame_index_cfr",
                "source_stage01": {"path": "Stage01.json", "sha256": "example"},
                "source_video": {"path": "movie.mp4", "size": 1, "mtime_ns": 1}, "audit_manifest": "audit.json",
                "corrections": [{"shot_id": "shot_000002", "original_temporal": temporal,
                                 "proposed_processing_temporal": dict(temporal, start_frame=26, end_frame=51),
                                 "boundary_changes": {"start_delta_frames": 1, "end_delta_frames": 1},
                                 "reason": "source frame evidence", "neighbor_effect": "review unselected neighbors",
                                 "evidence": {"basis": "source_video_frames_with_independent_ordinal_and_decoder_checks",
                                              "start_boundary_id": "frame_25", "end_boundary_id": "frame_50",
                                              "source_frame_indices": [24,25,26,49,50,51]},
                                 "review": {"status": "model_reviewed_pending_human", "reviewer": "model",
                                            "reviewed_on": "2026-10-22", "human_verified": False,
                                            "approved_for_processing": False}}]}
    jsonschema.validate(proposal, schema)
    invalid = copy.deepcopy(proposal)
    invalid["corrections"][0]["review"]["approved_for_processing"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)
    invalid = copy.deepcopy(proposal)
    invalid["scope"] = "all_shots_plus_one"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)
