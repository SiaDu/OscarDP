
import numpy as np
import pytest

from oscardp.stage05.performance_prepare import isolated_frame, track_span
from oscardp.stage06_2.full_sequence import MOVIE, SEQUENCE, missing_intervals, shot_order, timeline_records


def test_sequence_requires_all_thirteen_stage04_shots():
    shots=[f"shot_{n:06d}" for n in range(211,224)]
    sequence={"movie_id":MOVIE,"sequence_candidates":[{"sequence_id":SEQUENCE,"shot_ids":shots}]}
    assert shot_order(sequence)[1]==shots
    shots.pop()
    with pytest.raises(ValueError):
        shot_order(sequence)


def test_native_coordinate_isolation_is_per_track_and_blocks_overlap():
    image=np.ones((80,120,3),dtype=np.uint8)
    a,b=[12,10,30,20],[65,20,25,25]
    left,status=isolated_frame(image,a,[b])
    right,_=isolated_frame(image,b,[a])
    assert status=="isolated_native_pixels" and left.shape==right.shape==image.shape
    assert left[20,25].all() and not left[30,70].any()
    assert right[30,70].all() and not right[20,25].any()
    blocked,why=isolated_frame(image,a,[[20,20,15,20]])
    assert not blocked.any() and why=="conflicting_face_overlap"
    missing,why=isolated_frame(image,None,[b])
    assert not missing.any() and why=="target_unobserved"


def test_track_lifetime_keeps_original_pts_and_cut_break():
    assert track_span({"start_frame":55,"end_frame":57,"start_sec":1423.25,"end_sec":1423.36})["start"]==1423.25
    evidence=[{"global_frame_index":55,"movie_timestamp_sec":1423.25,"source_next_timestamp_sec":1423.29},
              {"global_frame_index":56,"movie_timestamp_sec":1423.29,"source_next_timestamp_sec":1423.36}]
    event={"start_frame":55,"end_frame":57,"start_s":1423.25,"end_s":1423.36}
    [r]=timeline_records("shot_000222","track_a","motion",[event],evidence,"source")
    assert r["track_id"]=="track_a" and r["start_s"]==1423.25
    for bad in ({**event,"end_frame":58},{**event,"start_s":0.}):
        with pytest.raises(ValueError,match="boundary"):
            list(timeline_records("shot_000222","track_a","motion",[bad],evidence,"source"))


def test_missing_intervals_do_not_mask_other_modalities():
    rows=[]
    for i,(eye,gaze,head) in enumerate(((False,True,True),(False,True,True),(True,False,True))):
        rows.append({"movie_timestamp_sec":1500+i*.04,"source_next_timestamp_sec":1500+(i+1)*.04,
                     "eye_position_status":{"left":{"usable":eye},"right":{"usable":False}},
                     "gaze_direction_status":{"usable":gaze},"head_pose_status":{"usable":head}})
    assert list(missing_intervals(rows)) == [
        {"modality":"eye_position","start_s":1500,"end_s":1500.08,"reason":"frame_qc"},
        {"modality":"gaze_direction","start_s":1500.08,"end_s":1500.12,"reason":"frame_qc"},
    ]
