import copy
from types import SimpleNamespace

import numpy as np
import pytest

from oscardp.stage05.event_au import detect_au
from oscardp.stage05.event_config import load_config, config_hash
from oscardp.stage05.event_detection import detect, validate_events
from oscardp.stage05.event_eye import detect_eyes
from oscardp.stage05.event_motion import detect_motion
from oscardp.stage05.event_records import assign_ids
from oscardp.stage05.event_series import derivative, segments, median_filter, hysteresis


def rows(values,mode="static",times=None):
    result = []
    times = np.arange(len(values))*.04+10 if times is None else np.asarray(times)
    for i,value in enumerate(values):
        row = {"movie_id":"tt1","scene_id":None,"shot_id":"shot_000001","track_id":"track1","identity_cluster_id":"identity_001",
               "identity_status":"tentative_identity_cluster","identity_human_verified":False,"extraction_mode":mode,
               "global_frame_index":100+i,"openface_local_frame":i+1,"parquet_row":i,"movie_timestamp_sec":float(times[i]),
               "source_next_timestamp_sec":float(times[i+1]) if i+1<len(times) else float(times[i]+.04),
               "AU01_r":float(value),"AU01_c":float(value>1),"AU45_r":0.,"AU45_c":0.,
               "pose_Rx":0.,"pose_Ry":0.,"pose_Rz":0.,"gaze_angle_x":0.,"gaze_angle_y":0.,
               "au_valid":True,"head_pose_valid":True,"gaze_valid":True,"tracking_usable":True,"landmarks_2d_valid":True}
        set_eyes(row,3.,3.)
        result.append(row)
    return result


def set_eyes(row,left,right):
    for start,h in [(36,left),(42,right)]:
        points = [[-10,0],[-5,-h],[5,-h],[10,0],[5,h],[-5,h]]
        for i,(x,y) in enumerate(points,start): row[f"x_{i}"],row[f"y_{i}"] = x,y


def test_au_onset_plateau_and_offset_candidates():
    data = rows([0]*6+[2]*8+[0]*6)
    events,_,_ = detect_au(data,"AU01_r",load_config())
    assert len(events)==1
    e = events[0]
    assert e["onset_frame"]==106 and e["offset_frame"]==114
    assert e["apex_frame"] is None and e["apex_plateau"]=={"start_frame":106,"end_frame":114}
    assert not e["left_censored"] and not e["right_censored"]


def test_sustained_observation_no_fabricated_apex_or_onset():
    events,_,_ = detect_au(rows([2.]*20),"AU01_r",load_config())
    assert len(events)==1
    e = events[0]
    assert e["left_censored"] and e["right_censored"]
    assert e["onset_frame"] is None and e["offset_frame"] is None and e["apex_frame"] is None and e["apex_plateau"] is None
    assert e["measurements"]["phase"]=="sustained_observation"


def test_hysteresis_holds_subhigh_jitter_and_respects_low_gap():
    t = np.arange(8)*.04
    assert hysteresis(t,t+.04,np.array([0,1.2,.7,.9,.6,1.1,0,0]),1.,.4)==[(1,6)]
    assert hysteresis(t,t+.04,np.array([0,1.2,0,0,0,1.2,0,0]),1.,.4,.08)==[(1,2),(5,6)]


def test_multi_peak_kept_together_without_baseline_return():
    data = rows([0]*10+[2]*5+[.6]*3+[2]*5+[0]*10)
    events,_,_ = detect_au(data,"AU01_r",load_config())
    assert len(events)==1 and events[0]["apex_plateau"] is not None
    data = rows([0]*10+[2]*5+[0]*6+[2]*5+[0]*10)
    assert len(detect_au(data,"AU01_r",load_config())[0])==2


def test_bilateral_eye_closure_and_reopening_with_au_conflict():
    data = rows([0]*20)
    for r in data[7:10]: set_eyes(r,.5,.5)
    found,_,_,_ = detect_eyes(data,None,load_config())
    assert len(found)==1
    e = found[0]
    assert e["measurements"]["classification"]=="complete_closure_candidate"
    assert e["measurements"]["reopening_observed"]
    assert "eye_geometry_au45_conflict" in e["qc_flags"] and not e["measurements"]["verified_blink"]


def test_partial_eye_closure_and_missing_reopening_are_uncertain():
    data = rows([0]*20)
    for r in data[15:]: set_eyes(r,1.5,2.)
    found,_,_,_ = detect_eyes(data,None,load_config())
    assert found[0]["measurements"]["classification"]=="partial_closure"
    assert found[0]["right_censored"] and found[0]["offset_frame"] is None
    assert found[0]["measurements"]["cycle_evidence"]=="incomplete_uncertain"


def test_persistent_au45_does_not_create_eye_cycles():
    data = rows([0]*20)
    for r in data: r["AU45_r"],r["AU45_c"] = 3.,1.
    found,_,_,info = detect_eyes(data,data,load_config())
    assert not found and info["au45_without_geometry_intervals"]


def test_single_frame_eye_anomaly_not_declared_closure():
    data = rows([0]*20)
    set_eyes(data[10],.1,.1)
    found,bad,_,_ = detect_eyes(data,None,load_config())
    assert not found and bad


def motion_rows():
    data = rows([0]*35)
    angles = np.r_[np.zeros(8),np.linspace(0,.5,12),np.ones(15)*.5]
    for r,a in zip(data,angles): r["pose_Ry"],r["gaze_angle_x"] = float(a),float(a)
    return data


def test_head_and_gaze_motion_keep_relative_units_and_confounds():
    data = motion_rows()
    head,_,_ = detect_motion(data,"head",load_config())
    gaze,_,_ = detect_motion(data,"gaze",load_config())
    assert head and gaze and head[0]["measurements"]["dominant_rotation_axis"]=="yaw"
    assert head[0]["measurements"]["angular_displacement_rad"]>.3
    assert "head_motion_confound_possible" in gaze[0]["qc_flags"]
    assert "head_gaze_simultaneous_change" in gaze[0]["qc_flags"]
    assert not gaze[0]["measurements"]["eye_only_gaze_shift"]


def test_invalid_frame_splits_events_and_is_never_phase_evidence():
    data = rows([2]*25)
    data[12]["au_valid"] = False
    found,_,_ = detect_au(data,"AU01_r",load_config())
    assert len(found)==2
    assert all(not e["start_frame"]<=112<e["end_frame"] for e in found)
    assert all(112 not in [e["onset_frame"],e["apex_frame"],e["offset_frame"]] for e in found)
    validate_events(found,{("track1","static"):data})


def test_dynamic_is_sensitivity_only_no_formal_duplicate_stream():
    static,dynamic = rows([0]*20),rows([2]*20,"dynamic")
    for r in dynamic: r["AU45_r"] = 2.
    comparison = {("track1",n):{"paired_pearson":"","paired_rmse":"2."} for n in ["AU01_r","AU45_r"]}
    found,_,_,sensitivity,_ = detect({("track1","static"):static,("track1","dynamic"):dynamic},comparison,load_config(),True)
    assert not found and any(s["dynamic_sensitivity_count"] for s in sensitivity)
    assert all(not s["is_formal_event_stream"] for s in sensitivity)


def test_vfr_derivatives_and_time_windows_do_not_assume_fps():
    times = np.array([0.,.03,.09,.16,.23,.31])
    np.testing.assert_allclose(derivative(times,2*times),2.)
    values = np.array([0.,0.,10.,0.,0.,0.])
    assert median_filter(times,values,.025)[2]==10
    assert median_filter(times,values,.08)[2]==0
    data = rows([0]*6,times=times)
    assert segments(data,[True]*6,.05)==[[0,1],[2],[3],[4],[5]]


@pytest.mark.parametrize("values",[[3.],[0,0,3,0,0],[0.,0.]])
def test_short_shots_and_isolated_au_spikes(values):
    assert not detect_au(rows(values),"AU01_r",load_config())[0]


def test_ids_sorting_and_tentative_status_are_deterministic():
    data = rows([0]*6+[2]*8+[0]*6)
    e = detect_au(data,"AU01_r",load_config())[0][0]
    second = copy.deepcopy(e)
    second.update(feature="AU04_r",start_frame=e["start_frame"]+1)
    a,b = assign_ids(copy.deepcopy([e,second])),assign_ids(copy.deepcopy([second,e]))
    assert a==b and all(x["identity_status"]=="tentative" and x["human_verified"] is False for x in a)
    assert config_hash(load_config())==config_hash(load_config())


def test_referential_integrity_rejects_phase_on_invalid_frame():
    data = rows([0]*6+[2]*8+[0]*6)
    found = detect_au(data,"AU01_r",load_config())[0]
    data[6]["au_valid"] = False
    with pytest.raises(ValueError,match="invalid data"):
        validate_events(found,{("track1","static"):data})


def test_event_rerun_never_overwrites(tmp_path):
    from oscardp.stage05.event_pipeline import run
    old = tmp_path/"event.json"
    old.write_text("keep")
    with pytest.raises(FileExistsError): run(SimpleNamespace(output_dir=tmp_path))
    assert old.read_text()=="keep"


@pytest.mark.parametrize("key,value",[("min_frames",1),("min_duration_sec",float("nan"))])
def test_bad_event_configuration_rejected(tmp_path,key,value):
    import json
    path = tmp_path/"config.json"
    path.write_text(json.dumps({"au":{key:value}}))
    with pytest.raises(ValueError): load_config(path)


def test_status_and_parquet_qc_masks_must_match():
    from oscardp.stage05.event_inputs import validate_rows,MASKS
    row = rows([0])[0]
    for name in MASKS: row.setdefault(name,True)
    state = copy.deepcopy(row)
    state["reliability"] = {n:row[n] for n in MASKS}
    state["reliability"]["au_valid"] = False
    with pytest.raises(ValueError,match="QC mask mismatch"):
        validate_rows([row],[state],{}, {},{"measurement_rows":1},{},{},None,{})


def test_earlier_identity_provenance_is_required(tmp_path,monkeypatch):
    import json
    from oscardp.stage05.event_inputs import load_measurements
    (tmp_path/"comparisons").mkdir()
    for path in ["frame_features.parquet","frame_status.jsonl","comparisons/au_static_vs_dynamic.csv"]:
        (tmp_path/path).write_text("")
    data = {"schema_version":"stage05.4.frame_measurements.v1","movie_id":"tt1","selected_shot_ids":["shot_000001"],
            "au_modes":["static","dynamic"],"main_au_mode":"static",
            "source_provenance":{"tracking_run":"tracking","identity_run":"identity"}}
    for name in ["run_config.json","track_measurements.json","qc_report.json"]: (tmp_path/name).write_text(json.dumps(data))
    def reject(*a): raise ValueError("identity provenance mismatch")
    monkeypatch.setattr("oscardp.stage05.event_inputs.load_inputs",reject)
    with pytest.raises(ValueError,match="identity provenance mismatch"): load_measurements(tmp_path,"tt1",["shot_000001"])
