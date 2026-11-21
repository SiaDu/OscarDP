"""Physical rotation checks and gap-safe within-shot motion tests."""

import numpy as np
import pytest

from oscardp.stage06_2.geometry import angular_change, decompose, head_rotation
from oscardp.stage06_2.inputs import EYES, POSE
from oscardp.stage06_2.temporal import detect, movement_type, prepare_frames

F = np.array([0., 0., -1.])


def test_openface_source_sign_and_noncommuting_order():
    x, y, z = .29, -.46, .37
    s1, s2, s3 = np.sin([x, y, z])
    c1, c2, c3 = np.cos([x, y, z])
    # Individual matrix entries from the pinned C++ RotationHelpers.h.
    expected = [[c2*c3, -c2*s3, s2],
                [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]]
    np.testing.assert_allclose(head_rotation([x, y, z]), expected, atol=1e-14)
    np.testing.assert_allclose(head_rotation([0, np.pi/2, 0]) @ F, [-1, 0, 0], atol=1e-14)
    np.testing.assert_allclose(head_rotation([np.pi/2, 0, 0]) @ F, [0, 1, 0], atol=1e-14)


def test_eyes_move_under_fixed_head():
    pose = [.41, -.24, .33]
    a = decompose(pose, [0, 0, -1], [0, 0, -1])
    b = decompose(pose, [.2, 0, -1], [.2, 0, -1])
    np.testing.assert_allclose(a["head_direction"], b["head_direction"])
    assert angular_change(a["eye_relative_direction"], b["eye_relative_direction"]) == pytest.approx(
        np.degrees(np.arctan(.2)))


def test_head_turn_with_fixed_distinct_monocular_rays():
    angle = .4
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    left, right = np.array([.6, 0, -.8]), np.array([0, .28, -.96])
    a = decompose([0, 0, 0], left, right)
    b = decompose([0, angle, 0], rotation@left, rotation@right)
    assert angular_change(a["gaze_direction"], b["gaze_direction"]) > 20
    np.testing.assert_allclose(b["left_eye_relative_direction"], left, atol=1e-14)
    np.testing.assert_allclose(b["right_eye_relative_direction"], right, atol=1e-14)
    np.testing.assert_allclose(b["eye_relative_direction"], a["eye_relative_direction"], atol=1e-14)


def test_simultaneous_motion_may_cancel_camera_gaze():
    a = decompose([0, 0, 0], F, F)
    b = decompose([0, .4, 0], F, F)
    np.testing.assert_allclose(a["gaze_direction"], b["gaze_direction"])
    np.testing.assert_allclose(b["eye_relative_direction"], [np.sin(.4), 0, -np.cos(.4)])
    assert angular_change(a["head_direction"], b["head_direction"]) == pytest.approx(np.degrees(.4))


def test_invalid_vector_and_missing_pose():
    a = decompose([0, 0, 0], [0, 0, 0], F)
    assert a["gaze_direction"] is a["eye_relative_direction"] is a["left_eye_relative_direction"] is None
    np.testing.assert_allclose(a["right_eye_relative_direction"], F)
    b = decompose([None, 0, 0], F, F)
    assert b["head_direction"] is b["eye_relative_direction"] is None
    np.testing.assert_allclose(b["gaze_direction"], F)


def rows_for_motion(head, eye, closure=False):
    rows = []
    for i in range(32):
        u = np.clip((i-6)/12, 0, 1)
        pose = [0., u*head, 0.]
        ray = [-np.sin(u*(eye+head)), 0., -np.cos(u*(eye+head))]
        rows.append(dict(zip(POSE + EYES[0] + EYES[1], [*pose, *ray, *ray]), shot_id="shot_000001",
                         track_id="track_001", person=None, global_frame_index=i, movie_timestamp_sec=100+i*.04,
                         source_next_timestamp_sec=100+(i+1)*.04, tracking_usable=True,
                         head_pose_valid=True, gaze_valid=True))
    source = [{"event_id": "closure", "event_type": "eye_closure_candidate", "start_frame": 11, "end_frame": 13,
               "start_sec": 100.44, "end_sec": 100.52}] if closure else []
    return prepare_frames(rows, source), source


@pytest.mark.parametrize(("head", "eye", "label"), [
    (0., .25, "eye_dominant"), (.25, 0., "head_dominant"), (.25, -.24, "combined"),
])
def test_detects_movement_without_stage05_gaze_seeds(head, eye, label):
    rows, source = rows_for_motion(head, eye)
    events = detect(rows, source)
    assert [e["movement_type"] for e in events] == [label]
    assert events[0]["valid_frame_ratio"] == 1.
    assert events[0]["start_s"] >= 100


def test_closure_breaks_intervals_and_retains_head_pose():
    rows, source = rows_for_motion(.4, .35, closure=True)
    assert all(rows[i]["head_usable"] and not rows[i]["eye_relative_usable"] for i in (11, 12))
    assert rows[11]["gaze_direction"] is rows[12]["eye_relative_direction"] is None
    for e in detect(rows, source):
        assert e["end_frame"] <= 11 or e["start_frame"] >= 13
    assert movement_type(1., 8., unreliable=True) == "uncertain"


def test_binocular_disagreement_retains_individual_rays():
    rows, _ = rows_for_motion(0., 0.)
    row = {**rows[0], "gaze_0_x": .5, "gaze_0_z": -.8660254}
    rejected = prepare_frames([row], [])[0]
    assert rejected["binocular_disagreement_deg"] == pytest.approx(30.)
    assert "binocular_disagreement" in rejected["qc_flags"]
    assert rejected["left_eye_relative_direction"] is not None
    assert rejected["eye_relative_direction"] is None and not rejected["eye_relative_usable"]
    assert rejected["eye_relative_direction_qc_rejected"] is not None
