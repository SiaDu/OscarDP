"""Physical rotation checks and gap-safe within-shot motion tests."""

import numpy as np
import pytest

from oscardp.stage06_2.geometry import angular_change, decompose, head_rotation
from oscardp.stage06_2.inputs import EYES, POSE
from oscardp.stage06_2.temporal import detect, movement_type, prepare_frames, valid_blocks

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


def test_binocular_disagreement_keeps_all_rays_usable():
    rows, _ = rows_for_motion(0., 0.)
    row = {**rows[0], "gaze_0_x": .5, "gaze_0_z": -.8660254}
    frame = prepare_frames([row], [])[0]
    assert frame["binocular_disagreement_deg"] == pytest.approx(30.)
    assert "binocular_disagreement" in frame["qc_flags"]
    assert frame["left_eye_relative_direction"] is not None
    assert frame["camera_gaze_usable"] and frame["eye_relative_usable"]
    np.testing.assert_allclose(frame["gaze_direction"], frame["eye_relative_direction"])


@pytest.mark.parametrize("bad_gaze", [{"gaze_valid": False}, {"tracking_usable": False},
                                     {"gaze_0_x": float("nan")}, {"gaze_0_z": 0.}])
def test_invalid_gaze_still_blocks_the_mean(bad_gaze):
    rows, _ = rows_for_motion(0., 0.)
    frame = prepare_frames([{**rows[0], **bad_gaze}], [])[0]
    assert frame["gaze_direction"] is frame["eye_relative_direction"] is None
    assert not frame["camera_gaze_usable"] and not frame["eye_relative_usable"]


def test_missing_pose_blocks_eye_relative_but_preserves_camera_gaze():
    rows, _ = rows_for_motion(0., 0.)
    frame = prepare_frames([{**rows[0], "head_pose_valid": False}], [])[0]
    assert frame["camera_gaze_usable"]
    assert not frame["head_usable"] and not frame["eye_relative_usable"]
    assert all(frame[key] is None for key in (
        "eye_relative_direction", "left_eye_relative_direction", "right_eye_relative_direction"))


def test_pts_break_splits_otherwise_usable_eye_frames():
    rows, _ = rows_for_motion(0., .25)
    for row in rows[10:]:
        row["movie_timestamp_sec"] += .1
        row["source_next_timestamp_sec"] += .1
    assert [len(block) for block in valid_blocks(rows, [row["eye_relative_usable"] for row in rows])] == [10, 22]
    assert all(e["end_frame"] <= 10 or e["start_frame"] >= 10 for e in detect(rows, []))


@pytest.mark.parametrize(("increments", "expected"), [
    ((.25, .25), "eye_dominant"),
    ((.75, -.10), "uncertain"),
])
def test_motion_classification_ignores_absolute_offset_but_not_conflicting_change(increments, expected):
    rows, _ = rows_for_motion(0., 0.)
    for i, frame in enumerate(rows):
        ramp = np.clip((i-6)/12, 0, 1)
        for eye, base, change in zip(EYES, (.2, -.2), increments, strict=True):
            angle = base + ramp * change
            frame.update(zip(eye, (np.sin(angle), 0., -np.cos(angle)), strict=True))
    result = detect(prepare_frames(rows, []), [])
    assert [event["movement_type"] for event in result] == [expected]
    event = result[0]
    assert event["eye_relative_change_deg"] >= 6
    assert "binocular_disagreement" in event["qc_flags"]
    assert ("binocular_motion_disagreement" in event["qc_flags"]) == (expected == "uncertain")


def position_sequence(*, gap=False, eyes_stable=False):
    rows, _ = rows_for_motion(0., 0.)
    for i, r in enumerate(rows):
        shift = .08 if i >= 16 else 0
        direction = [np.sin(3*shift), 0., -np.cos(3*shift)]
        r.update(left_eye_x=.45+(0 if eyes_stable else shift),
                 right_eye_x=.55+(0 if eyes_stable else shift), left_eye_y=0., right_eye_y=0.,
                 gaze_direction=direction)
        for eye in EYES:
            r.update(zip(eye, direction, strict=True))
    if gap:
        rows[16].update(camera_gaze_usable=False, eye_relative_usable=False)
    return rows


def test_stable_gaze_shift_requires_sustained_and_continuous_support():
    from oscardp.stage06_2.stable import analyze
    rows = position_sequence(eyes_stable=True)
    states, motion, frames = analyze(rows, [], [])
    assert sum(s["type"]=="stable_gaze_candidate" for s in states) == 2
    assert sum(s["type"]=="eye_position_stable" for s in states) == 1
    [shift] = [e for e in motion if e["type"]=="gaze_direction_shift_candidate"]
    assert shift["start_frame"] <= 16 < shift["end_frame"]
    assert len(frames) < len(rows)
    _, motion, _ = analyze(position_sequence(gap=True, eyes_stable=True), [], [])
    assert all(e["type"]!="gaze_direction_shift_candidate" for e in motion)


def test_eye_position_does_not_require_gaze_and_blink_stays_separate():
    from oscardp.stage06_2.stable import analyze
    rows = position_sequence()
    for r in rows:
        r.update(eye_relative_usable=False, camera_gaze_usable=False)
    rows[8].update(left_eye_x=None, right_eye_x=None)
    states, motion, picks = analyze(rows, [{"event_type":"eye_closure_candidate", "event_id":"blink",
        "start_frame":8, "end_frame":9, "start_sec":100.32,"end_sec":100.36}], [])
    assert states and {s["type"] for s in states} == {"eye_position_stable"}
    assert any(e["type"]=="blink_candidate" for e in motion)
    assert any(e["type"]=="unknown" and e["start_frame"]==8 for e in motion)
    assert not picks


def test_aperture_position_uses_fitted_iris_and_roll_normalization():
    from oscardp.stage06_2.eye_position import position
    points=np.zeros((28,2))
    points[:8]=[5., 1.]
    points[8:20]=[[0,0],[2,-2],[4,-3],[6,-3],[8,-3],[10,-2],[12,0],
                  [10,2],[8,3],[6,3],[4,3],[2,2]]
    reference=position(points)
    angle=.35
    R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    np.testing.assert_allclose(position(points@R.T+[150,200]), [5/12,1/12])
    np.testing.assert_allclose(reference,[5/12,1/12])
    points[:8,1]=8
    assert position(points) is None
