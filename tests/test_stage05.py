import json
import math
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from oscardp.stage05.detectors import YuNetDetector
from oscardp.stage05.output import atomic_json
from oscardp.stage05.qc import DiscontinuityGuard, report
from oscardp.stage05.tracker import ShotTracker
from oscardp.stage05.video_reader import iter_frames
from oscardp.shots.media import probe_video, build_timeline, iter_frame_timestamps


def detection(x=10, y=10):
    return {"bbox_xywh": [x, y, 40, 40], "detection_score": .95, "landmarks5": None}


def step(t, ds, n):
    return t.update(ds, n, n/25, (n+1)/25)


def test_moving_face_and_exclusive_track_end():
    t = ShotTracker("shot_000209")
    records = [step(t, [detection(10+n*2)], n)[0] for n in range(10)]
    assert {r["track_id"] for r in records} == {"shot_000209_track_001"}
    summary = t.summaries()[0]
    assert (summary["start_frame"], summary["end_frame"], summary["end_sec"]) == (0, 10, .4)
    assert summary["tracking_status"] == "confirmed_ended"


def test_two_faces_unique_one_to_one_and_detection_order_changes():
    t = ShotTracker("shot_000001")
    a = step(t, [detection(10), detection(110)], 0)
    b = step(t, [detection(108), detection(12)], 1)
    assert len({r["track_id"] for r in b}) == 2
    assert a[0]["track_id"] == b[1]["track_id"]
    assert a[1]["track_id"] == b[0]["track_id"]


def test_missing_frames_are_not_observations():
    t = ShotTracker("shot_000001", max_missed=2)
    a = step(t, [detection()], 0)[0]
    assert step(t, [], 1) == []
    assert step(t, [], 2) == []
    b = step(t, [detection()], 3)[0]
    assert a["track_id"] == b["track_id"]
    s = t.summaries()[0]
    assert s["missed_frame_count"] == 2
    assert s["detected_frame_count"] == 2
    assert s["valid_detection_ratio"] == .5
    assert b["detection_status"] == "observed"


def test_reentry_after_expiration_and_far_reentry_break_tracks():
    t = ShotTracker("shot_000001", max_missed=1)
    a = step(t, [detection()], 0)[0]
    step(t, [], 1)
    step(t, [], 2)
    b = step(t, [detection()], 3)[0]
    c = step(t, [detection(300)], 4)[0]
    assert len({a["track_id"], b["track_id"], c["track_id"]}) == 3


def test_ambiguous_crossing_breaks_link():
    t = ShotTracker("shot_000001")
    old = step(t, [detection(0), detection(40)], 0)
    new = step(t, [detection(20)], 1)
    assert new[0]["track_id"] not in {r["track_id"] for r in old}
    assert t.ambiguities


def test_shots_reset_ids():
    a = step(ShotTracker("shot_000001"), [detection()], 0)[0]
    b = step(ShotTracker("shot_000002"), [detection()], 1)[0]
    assert a["track_id"] != b["track_id"]
    assert a["track_id"].endswith("001") and b["track_id"].endswith("001")


def test_discontinuity_breaks_geometrically_identical_faces():
    guard = DiscontinuityGuard(.18)
    t = ShotTracker("shot_000001")
    assert guard.update(np.zeros((64, 96, 3), dtype=np.uint8)) is None
    old = step(t, [detection()], 0)[0]
    difference = guard.update(np.full((64, 96, 3), 255, dtype=np.uint8))
    assert difference == 1.
    t.break_links(1, difference)
    new = step(t, [detection()], 1)[0]
    assert new["track_id"] != old["track_id"]
    assert t.ambiguities[0]["reason"] == "visual_discontinuity_association_reset"
    assert guard.update(np.full((64, 96, 3), 254, dtype=np.uint8)) is None


def test_resize_coordinates_landmarks_and_no_faces():
    row = np.array([[10, 20, 30, 40, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, .9]])
    result = YuNetDetector.map_faces(row, 2, 3)[0]
    assert result["bbox_xywh"] == [20, 60, 60, 120]
    assert result["landmarks5"] == [[2, 6], [6, 12], [10, 18], [14, 24], [18, 30]]
    assert YuNetDetector.map_faces(None, 1, 1) == []


def test_empty_shot_qc_and_finite_json(tmp_path):
    t = ShotTracker("shot_000001")
    for n in range(5):
        assert step(t, [], n) == []
    qc = report([{"shot_id": "shot_000001", "status": "complete", "decoded_frame_count": 5,
                  "tracks": t.summaries()}], {})
    assert qc["face_detection_count"] == qc["face_track_count"] == 0
    assert not qc["incomplete_shots"]
    atomic_json(tmp_path/"qc.json", qc)
    assert json.loads((tmp_path/"qc.json").read_text())["decoded_frame_count"] == 5
    with pytest.raises(ValueError):
        atomic_json(tmp_path/"bad.json", {"x": math.nan})


def test_record_required_fields_and_json(tmp_path):
    record = step(ShotTracker("shot_000001"), [detection()], 0)[0]
    assert set(record) == {"shot_id", "frame_index", "timestamp_sec", "track_id", "bbox_xywh",
                           "detection_score", "landmarks5", "detection_status"}
    atomic_json(tmp_path/"record.json", record)
    assert json.loads((tmp_path/"record.json").read_text()) == record


@pytest.fixture
def clip(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg required")
    video = tmp_path/"clip.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=25",
                    "-frames:v", "12", "-c:v", "ffv1", str(video)], check=True)
    return video


def test_exact_frame_bounds_and_pixels(clip):
    metadata, _ = probe_video(clip, clip.name)
    shots = [{"temporal": {"start_frame": 2, "end_frame": 5, "end": .2}},
             {"temporal": {"start_frame": 8, "end_frame": 12, "end": .48}}]
    frames = list(iter_frames(clip, shots, metadata, True))
    assert [f.index for f in frames] == [2, 3, 4, 8, 9, 10, 11]
    assert frames[2].next_timestamp == .2
    assert frames[-1].next_timestamp == .48
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-pix_fmt", "bgr24",
                          "-f", "rawvideo", "pipe:1"], capture_output=True, check=True).stdout
    reference = np.frombuffer(raw, np.uint8).reshape(12, 64, 96, 3)
    for f in frames:
        assert f.timestamp == f.index/25
        np.testing.assert_array_equal(f.image, reference[f.index])


def test_vfr_pts_and_exclusive_timestamp(tmp_path):
    video = tmp_path/"vfr.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=25",
                    "-vf", "setpts=if(lt(N\\,4)\\,N\\,2*N-4)/(25*TB)", "-frames:v", "10",
                    "-vsync", "0", "-c:v", "ffv1", str(video)], check=True)
    metadata, _ = probe_video(video, video.name)
    timeline = build_timeline(list(iter_frame_timestamps(video)), metadata.fps)
    assert timeline.is_vfr
    shots = [{"temporal": {"start_frame": 2, "end_frame": 8, "end": timeline.timestamp(8)}}]
    frames = list(iter_frames(video, shots, metadata, False))
    assert [f.index for f in frames] == list(range(2, 8))
    for f in frames:
        assert f.timestamp == pytest.approx(timeline.timestamp(f.index))
        assert f.next_timestamp == pytest.approx(timeline.timestamp(f.index+1))


def test_missing_model_and_video_fail_clearly(tmp_path):
    pytest.importorskip("cv2")
    with pytest.raises(FileNotFoundError, match="YuNet ONNX"):
        YuNetDetector(tmp_path/"missing.onnx")
    with pytest.raises(Exception, match="No such file"):
        probe_video(tmp_path/"missing.mp4", "missing.mp4")


def test_pipeline_checkpoints_resume_and_config_guard(clip, tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from oscardp.stage05 import pipeline
    metadata, _ = probe_video(clip, clip.name)
    calls = []
    class Detector:
        def detect(self, image):
            calls.append(1)
            return []
    shots = [{"shot_id": "shot_000001", "temporal": {
        "start_frame": 2, "end_frame": 5, "start": .08, "end": .2}}]
    config = {"movie_id": "tt12300742", "frame_boundaries": "start inclusive; end exclusive",
              "detector": "mock", "device": "cpu", "opencv_version": "test", "model": {},
              "model_sha256": "test", "score_threshold": .8, "nms_threshold": .3,
              "inference_max_size": 0, "shot_ids": ["shot_000001"]}
    monkeypatch.setattr(pipeline, "preflight", lambda args: (clip, shots, metadata, True, Detector(), config))
    args = SimpleNamespace(dry_run=False, output_root=tmp_path/"out", movie_id="tt12300742",
                           resume=False, overwrite=False, max_missed=3, min_hits=3, discontinuity_threshold=.18)
    assert pipeline.run(args) == 0
    root = args.output_root/args.movie_id
    qc = json.loads((root/"qc_report.json").read_text())
    assert qc["decoded_frame_count"] == 3 and qc["face_detection_count"] == 0
    assert (root/"face_detections.jsonl").read_text() == ""
    assert len(calls) == 3
    with pytest.raises(FileExistsError):
        pipeline.run(args)
    args.resume = True
    assert pipeline.run(args) == 0
    assert len(calls) == 3
    config["score_threshold"] = .85
    with pytest.raises(ValueError, match="configuration changed"):
        pipeline.run(args)


def test_failed_shot_retry(clip, tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from oscardp.stage05 import pipeline
    metadata, _ = probe_video(clip, clip.name)
    fail = [True]
    class Detector:
        def detect(self, image):
            if fail[0]:
                raise RuntimeError("simulated detector interruption")
            return [detection()]
    shots = [{"shot_id": "shot_000001", "temporal": {
        "start_frame": 2, "end_frame": 5, "start": .08, "end": .2}}]
    config = {"movie_id": "tt12300742", "frame_boundaries": "start inclusive; end exclusive",
              "detector": "mock", "device": "cpu", "opencv_version": "test", "model": {},
              "model_sha256": "test", "score_threshold": .8, "nms_threshold": .3,
              "inference_max_size": 0, "shot_ids": ["shot_000001"]}
    monkeypatch.setattr(pipeline, "preflight", lambda args: (clip, shots, metadata, True, Detector(), config))
    args = SimpleNamespace(dry_run=False, output_root=tmp_path/"out", movie_id="tt12300742",
                           resume=False, overwrite=False, max_missed=3, min_hits=3, discontinuity_threshold=.18)
    assert pipeline.run(args) == 1
    root = args.output_root/args.movie_id
    qc = json.loads((root/"qc_report.json").read_text())
    assert qc["incomplete_shots"] == ["shot_000001"]
    fail[0] = False
    args.resume = True
    assert pipeline.run(args) == 0
    qc = json.loads((root/"qc_report.json").read_text())
    assert qc["decoded_frame_count"] == 3 and qc["face_detection_count"] == 3
    assert not qc["incomplete_shots"]
