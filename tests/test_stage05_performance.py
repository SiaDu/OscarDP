import copy
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from oscardp.stage05.output import atomic_json
from oscardp.stage05.performance_csv import columns_by_modality, parse_csv
from oscardp.stage05.performance_inputs import load_inputs, snapshot
from oscardp.stage05.performance_openface import command
from oscardp.stage05.performance_output import align_rows, save_frames
from oscardp.stage05.performance_prepare import frame_mapping
from oscardp.stage05.performance_qc import assess, missing_intervals


@pytest.fixture
def face():
    angles = np.linspace(0,2*np.pi,68,endpoint=False)
    points = np.column_stack([60+40*np.cos(angles),70+50*np.sin(angles)])
    eye = np.array([[-8,0],[-4,-3],[4,-3],[8,0],[4,3],[-4,3]])
    points[36:42] = eye+[40,50]
    points[42:48] = eye+[80,50]
    points[30],points[48],points[54] = [60,75],[45,100],[75,100]
    raw = {"frame":1,"face_id":0,"timestamp":0.,"confidence":.95,"success":1,
           "AU01_r":1.,"AU04_r":2.,"AU01_c":1.,"AU28_c":0.,"nonfinite_columns":[],
           "pose_Rx":0.,"pose_Ry":0.,"pose_Rz":0.,"pose_Tx":0.,"pose_Ty":0.,"pose_Tz":500.,
           "gaze_angle_x":0.,"gaze_angle_y":0.,"eye_lmk_x_0":30.}
    raw.update({f"{axis}_{i}":float(points[i,j]) for j,axis in enumerate("xy") for i in range(68)})
    raw.update({f"{axis}_{i}":1. for axis in "XYZ" for i in range(68)})
    raw.update({f"gaze_{eye}_{axis}":-1. if axis=="z" else 0. for eye in range(2) for axis in "xyz"})
    observation = {"frame_index":100,"bbox_xywh":[10,10,100,120],
                   "landmarks5":[[40,50],[80,50],[60,75],[45,100],[75,100]]}
    return raw,observation


def test_csv_columns_types_and_au_families(tmp_path,face):
    raw,_ = face
    path = tmp_path/"sample.csv"
    fields = [k for k in raw if k!="nonfinite_columns"]
    with path.open("w",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([" "+n+" " for n in fields])
        writer.writerow([raw[n] for n in fields])
    header,rows = parse_csv(path,1,25)
    assert type(rows[1]["frame"]) is int and type(rows[1]["AU01_r"]) is float
    groups = columns_by_modality(header)
    assert groups["au_intensity"] == ["AU01_r","AU04_r"]
    assert groups["au_occurrence"] == ["AU01_c","AU28_c"]
    assert "AU28_r" not in header


@pytest.mark.parametrize("timestamp",[10.,10.037,10.103])
def test_exact_cfr_and_vfr_frame_mapping_preserves_source_pts(timestamp):
    shot = {"shot_id":"shot_000001","effective_processing_temporal":{"start_frame":100,"end_frame":104,"start":9.,"end":11.}}
    frame = SimpleNamespace(index=101,timestamp=timestamp,next_timestamp=timestamp+.047,image=np.zeros((2,2,3),np.uint8))
    row = frame_mapping(shot,"track",frame,2)
    assert row["global_frame_index"] == 101 and row["movie_timestamp_sec"] == timestamp
    with pytest.raises(ValueError,match="frame or source timestamp"):
        frame_mapping(shot,"track",frame,1)


def make_rows(face,mode="static"):
    raw,obs = face
    groups = columns_by_modality(raw.keys())
    mapping = [{"shot_id":"shot_000001","track_id":"track","openface_local_frame":i+1,
                "global_frame_index":100+i,"movie_timestamp_sec":5.+i*.047,"source_next_timestamp_sec":5.+(i+1)*.047} for i in range(3)]
    failed = copy.deepcopy(raw)
    failed.update(frame=3,success=0)
    rows = align_rows("tt1",mapping,[k for k in raw if k!="nonfinite_columns"],{1:raw,3:failed},
                      [dict(obs,frame_index=100),dict(obs,frame_index=102)],
                      {"identity_cluster_id":"identity_001","identity_status":"tentative_identity_cluster","identity_human_verified":False},
                      mode,groups,.8)
    return rows,groups


def test_missing_and_failed_frames_never_dropped(face):
    rows,_ = make_rows(face)
    assert len(rows)==3 and rows[1]["features"]["AU01_r"] is None
    assert rows[0]["reliability"]["au_valid"]
    assert rows[1]["reliability"]["reasons"] == ["missing_openface_frame"]
    assert "openface_tracking_failed" in rows[2]["reliability"]["reasons"]
    intervals = missing_intervals(rows,"au_valid")
    assert intervals[0]["start_frame"]==101 and intervals[0]["end_frame"]==103


def test_wrong_person_spatial_match_is_not_forced(face):
    raw,obs = face
    obs["bbox_xywh"][0] += 400
    obs["landmarks5"] = (np.asarray(obs["landmarks5"])+[400,0]).tolist()
    flags = assess(raw,columns_by_modality(raw),obs)
    assert not flags["association"]["matched"]
    assert not any(flags[k] for k in ["au_valid","head_pose_valid","gaze_valid"])


def test_success_not_enough_for_all_modalities(face):
    raw,obs = face
    raw["gaze_0_z"] = 0.
    flags = assess(raw,columns_by_modality(raw),obs)
    assert flags["openface_success"] and flags["au_valid"] and flags["head_pose_valid"]
    assert not flags["gaze_valid"]


def test_nan_infinity_preserved_as_explicit_nulls(tmp_path):
    path = tmp_path/"nan.csv"
    path.write_text("frame,face_id,timestamp,confidence,success,AU01_r,AU01_c\n1,0,0,.9,1,nan,inf\n")
    _,rows = parse_csv(path,1,25)
    assert rows[1]["AU01_r"] is None and rows[1]["AU01_c"] is None
    assert rows[1]["nonfinite_columns"] == ["AU01_r","AU01_c"]


@pytest.mark.parametrize("body",["1,0,0,.9,1\n1,0,0,.9,1\n","1,1,0,.9,1\n","2,0,0,.9,1\n","1,0,.04,.9,1\n"])
def test_multiface_duplicates_offsets_or_out_of_range_rejected(tmp_path,body):
    path = tmp_path/"bad.csv"
    path.write_text("frame,face_id,timestamp,confidence,success\n"+body)
    with pytest.raises(ValueError): parse_csv(path,1,25)


def test_modes_separate_and_commands_explicit(tmp_path,face):
    camera = {k:100 for k in ["fx","fy","cx","cy"]}
    static = command(tmp_path/"FeatureExtraction",tmp_path/"video.mkv",tmp_path,"static",camera)
    dynamic = command(tmp_path/"FeatureExtraction",tmp_path/"video.mkv",tmp_path,"dynamic",camera)
    assert "-au_static" in static and "-au_static" not in dynamic
    a,_ = make_rows(face,"static")
    b,_ = make_rows(face,"dynamic")
    assert {r["extraction_mode"] for r in a+b} == {"static","dynamic"}
    with pytest.raises(ValueError): make_rows(face,"mixed")


def test_json_parquet_refs_and_tentative_identity(tmp_path,face):
    pq = pytest.importorskip("pyarrow.parquet")
    rows,_ = make_rows(face)
    header = list(rows[0]["features"])
    info = save_frames(tmp_path,rows,[header],{"test":"immutable"})
    data = pq.read_table(tmp_path/info["parquet"]).to_pylist()
    status = [json.loads(line) for line in (tmp_path/info["status"]).read_text().splitlines()]
    assert len(data)==len(status)==3
    for i,(p,s) in enumerate(zip(data,status)):
        assert p["parquet_row"]==s["parquet_row"]==i
        assert p["movie_timestamp_sec"]==s["movie_timestamp_sec"]
        assert p["global_frame_index"]==s["global_frame_index"]
        assert not p["identity_human_verified"] and p["identity_status"]=="tentative_identity_cluster"
    assert data[1]["AU01_r"] is None


def test_performance_rerun_refused_before_accessing_sources(tmp_path):
    from oscardp.stage05.performance_pipeline import run
    protected = tmp_path/"original.json"
    protected.write_text("keep")
    before = snapshot([protected])
    with pytest.raises(FileExistsError): run(SimpleNamespace(output_dir=tmp_path))
    assert snapshot([protected])==before


def test_corrected_provenance_validator_is_required(tmp_path,monkeypatch):
    def reject(*args):
        raise ValueError("corrected provenance mismatch")
    monkeypatch.setattr("oscardp.stage05.performance_inputs.load_corrected",reject)
    with pytest.raises(ValueError,match="corrected provenance"):
        load_inputs(tmp_path,tmp_path,"tt1",["shot_000001"])


@pytest.mark.parametrize("mutation",["hash","human_verified","track_ids"])
def test_identity_source_hash_and_verification_mismatch_rejected(tmp_path,monkeypatch,mutation):
    boundary = {"mode":"experimental_model_reviewed_pending_human","human_verified":False}
    files = {"source":{"sha256":"same"}}
    source = {"boundary_sidecar":boundary}
    tracks = {"track":{"shot_id":"shot_000001"}}
    monkeypatch.setattr("oscardp.stage05.performance_inputs.load_corrected",lambda *a:(source,tracks,{},files))
    config = {"movie_id":"tt1","source_corrected_run":str(tmp_path),"source_files":copy.deepcopy(files),
              "boundary_sidecar":boundary,"selected_shot_ids":["shot_000001"],"track_ids":["track"]}
    result = dict(config,human_verified=False,track_representations=tracks,clusters=[])
    if mutation=="hash": config["source_files"]={"source":{"sha256":"changed"}}
    if mutation=="human_verified": result["human_verified"]=True
    if mutation=="track_ids": config["track_ids"]=["wrong"]
    for name,value in [("run_config.json",config),("identity_clusters.json",result),
                       ("qc_report.json",{"boundary_sidecar":boundary,"source_hashes_unchanged":True})]:
        atomic_json(tmp_path/name,value)
    with pytest.raises(ValueError): load_inputs(tmp_path,tmp_path,"tt1",["shot_000001"])
