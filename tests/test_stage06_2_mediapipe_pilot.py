import numpy as np

from oscardp.stage06_2.mediapipe_pilot import map_landmarks, reject_single_eye_jumps, roi
from oscardp.stage06_2.stable import analyze


def test_crop_mapping_handles_clipped_roi_and_non_square_scale():
    box = roi([4, 8, 40, 80], (100, 120))
    assert box == (0, 0, 54, 108)
    np.testing.assert_allclose(map_landmarks([[.5,.25]],box), [[27,27]])
    np.testing.assert_allclose(map_landmarks([[.5,.25]],(21,13,81,93)), [[51,33]])


def test_single_eye_jump_keeps_diagnostics_but_removes_position():
    left = dict(xy=[.7,0], jump=True, step_eye_width=.12, centre_px=[100,50])
    right = dict(xy=[.6,0], jump=False, step_eye_width=.01)
    reject_single_eye_jumps([{"image_left":{"mediapipe":left},"image_right":{"mediapipe":right}}])
    assert left["xy"] is None and left["centre_px"] == [100,50]
    assert left["status"] == "unsupported_landmark_jump"
    assert right["xy"] == [.6,0]


def test_monocular_support_is_explicit_and_a_missing_eye_breaks_comparison():
    rows=[]
    for i in range(31):
        r={"global_frame_index":i,"movie_timestamp_sec":i*.04,"source_next_timestamp_sec":(i+1)*.04,
           "left_eye_x":.4 if i<15 else .5,"left_eye_y":0., "right_eye_x":None,"right_eye_y":None,
           "head_usable":True,"camera_gaze_usable":False,"eye_relative_usable":False,
           "pose_Rx":0.,"pose_Ry":0.,"pose_Rz":0.}
        rows.append(r)
    states,events,_=analyze(rows,[],[],allow_monocular=True)
    assert all(s["eye_support"] == ["left"] and s["right_xy"] is None for s in states)
    assert [e["eye_support"] for e in events if e["type"]=="eye_shift_candidate"] == [["left"]]
    rows[15]["left_eye_x"]=None
    _,events,_=analyze(rows,[],[],allow_monocular=True)
    assert not any(e["type"]=="eye_shift_candidate" for e in events)
