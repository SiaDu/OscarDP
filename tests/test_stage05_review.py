"""Independent review contracts, deterministic sampling and real pilot acceptance."""
import copy
import csv
import json
import threading
from http.server import HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from oscardp.stage05_review.annotations import blank_record, validate_records
from oscardp.stage05_review.integration import digest, write, read
from oscardp.stage05_review.sampling import select_pilot
from oscardp.stage05_review.server import make_handler


def item():
    return {"review_id": "review_1", "event_id": "event_1", "shot_id": "shot_1", "track_id": "track_1",
            "start_frame": "10", "end_frame": "20"}


def validate(record):
    return validate_records([record], [item()], "abc", {"shot_1": [5, 25]})


def completed():
    r = blank_record(item(), "abc")
    r.update(observed_event_label="observable eye narrowing", event_presence_decision="uncertain",
             temporal_boundary_decision="needs_adjustment", evidence_quality="limited", reviewer="test fixture human",
             reviewed_at="2026-10-25T12:00:00+00:00", video_review_attested=True, review_status="human_reviewed")
    return r


def test_blank_and_partial_progress():
    assert validate(blank_record(item(), "abc"))
    assert validate(completed())
    assert validate_records([], [item()], "abc", {"shot_1": [5, 25]}) == []


@pytest.mark.parametrize("field,value", [("source_event_id", "event_wrong"), ("source_run_sha256", "wrong"),
    ("track_id", "wrong"), ("human_verified", True), ("reviewer", None), ("reviewed_at", "2026-10-25T12:00:00"),
    ("video_review_attested", False), ("event_presence_decision", "emotion"),
    ("temporal_boundary_decision", None), ("reviewed_frame_range", [4, 20])])
def test_bad_human_records_rejected(field, value):
    r = completed()
    r[field] = value
    with pytest.raises((ValueError, TypeError)):
        validate(r)


def test_suggested_boundaries_and_no_source_mutation():
    source = item()
    before = copy.deepcopy(source)
    r = completed()
    r["suggested_boundaries"] = {"onset_frame": 11, "apex_frame": 15, "offset_frame": 21}
    validate(r)
    assert source == before
    r["suggested_boundaries"]["offset_frame"] = 14
    with pytest.raises(ValueError):
        validate(r)
    r = blank_record(source, "abc")
    r["reviewer"] = "Codex"
    with pytest.raises(ValueError):
        validate(r)


def test_null_event_negative_record():
    q = {**item(), "event_id": ""}
    r = blank_record(q, "abc")
    assert r["source_event_id"] is None
    validate_records([r], [q], "abc", {"shot_1": [5, 25]})


def test_http_partial_save_restore_and_conflicts(tmp_path):
    root, reviews = tmp_path/"run", tmp_path/"reviews"
    (root/"viewer").mkdir(parents=True)
    reviews.mkdir()
    write(root/"run_config.json", {"source_run_sha256": "abc"})
    write(root/"review_items.json", {"items": [{"source_queue_item": item()}]})
    write(root/"scene_performance_packets.json", {"scenes": [{"shots": [{"shot_id": "shot_1",
          "effective_processing_temporal": {"start_frame": 5, "end_frame": 25}}]}]})
    write(reviews/"abc.json", {"schema_version": "stage05.6.human_reviews.v1", "source_run_sha256": "abc",
                              "records": [blank_record(item(), "abc")]})
    server = HTTPServer(("127.0.0.1", 0), make_handler(root, reviews))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/api/reviews"
    try:
        with urlopen(base) as response:
            initial = json.load(response)
        request = {"revision": initial["revision"], "source_run_sha256": "abc", "records": [completed()]}

        def post(payload):
            return urlopen(Request(base, json.dumps(payload).encode(),
                {"Content-Type": "application/json", "X-OscarDP-Review": "1"}))

        with post(request) as response:
            saved = json.load(response)
        assert saved["records"] == [completed()]
        with pytest.raises(HTTPError) as exc:
            post(request)
        assert exc.value.code == 409
        request["revision"] = saved["revision"]
        request["records"][0]["source_event_id"] = "wrong"
        with pytest.raises(HTTPError) as exc:
            post(request)
        assert exc.value.code == 400
        assert read(reviews/"abc.json")["records"][0]["source_event_id"] == "event_1"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


SOURCE = Path("reports/stage05_5/tt12300742/pilot_001")
RESULT = Path("reports/stage05_6/tt12300742/pilot_001")


@pytest.mark.skipif(not SOURCE.exists(), reason="Local research data absent")
def test_real_strata_and_determinism():
    q = list(csv.DictReader((SOURCE/"event_review_queue.csv").open()))
    selected, stats = select_pilot(q)
    again, _ = select_pilot(list(reversed(q)))
    assert selected == again
    assert len(selected) == 48
    selected_ids = {r["event_id"] for r in selected if r["event_id"]}
    events = read(SOURCE/"performance_events.json")["events"]
    for kind, n in [("eye_closure_candidate", 4), ("head_orientation_change", 8), ("camera_relative_gaze_change", 7)]:
        assert len([e for e in events if e["event_type"] == kind]) == n
        assert all(e["event_id"] in selected_ids for e in events if e["event_type"] == kind)
    for flag in ["au_mode_disagreement", "censored_event", "near_shot_boundary"]:
        assert any(flag in r["reasons"] for r in selected)
    assert stats["by_sample_kind"]["rejected_interval"] == 6
    assert stats["by_sample_kind"]["algorithm_negative_interval"] == 5
    assert stats["by_sample_kind"]["au45_without_geometry_support"] == 4
    assert len({r["feature"] for r in selected if r["event_type"] == "au_activation" and r["event_id"]}) >= 12


@pytest.mark.skipif(not (RESULT/"qc_report.json").exists(), reason="Generated local pilot absent")
def test_real_packet_scene_identity_invalid_frame_and_review_separation():
    packet = read(RESULT/"scene_performance_packets.json")
    assert [s["scene_id"] for s in packet["scenes"]] == ["scene_000021"]
    assert len(packet["event_candidates"]) == 126
    scene = packet["scenes"][0]
    assert len(scene["shots"]) == 5
    assert len(read(RESULT/"review_items.json")["items"]) == 205
    s214 = next(s for s in scene["shots"] if s["shot_id"] == "shot_000214")
    assert {d["speaker"] for d in s214["stage02_dialogue"]} == {"MICHELLE", "TEDDY"}
    identity = s214["face_tracks"][0]["identity_cluster"]
    assert identity["candidate_mapping"]["candidates"][0]["character_name"] == "TEDDY"
    assert identity["identity_status"] == "tentative"
    assert identity["character_mapping_status"] == "unverified"
    assert s214["effective_processing_temporal"]["start_frame"] == 34648
    assert s214["original_stage01_temporal"]["start_frame"] == 34647
    assert s214["boundary_review"]["human_verified"] is False
    assert 34648 in s214["face_tracks"][0]["invalid_frames_by_feature"]["eye_geometry"]
    forbidden = {"emotion", "intent", "emotion_label", "acting_intent"}

    def walk(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            if "human_verified" in value:
                assert value["human_verified"] is False
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(packet)
