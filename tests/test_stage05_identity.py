import numpy as np
import pytest

from oscardp.stage05.identity_association import AssociationConfig, representation, associate
from oscardp.stage05.identity_model import TEMPLATE, align, normalize
from oscardp.stage05.identity_sampling import candidates, crop_quality
from oscardp.stage05.identity_inputs import load_corrected
from oscardp.stage05.output import atomic_json


def representations(vectors_by_track, config=None):
    config = config or AssociationConfig()
    info, centers, samples = {},{},{}
    for i,(tid,vs) in enumerate(vectors_by_track.items()):
        vs = [normalize(v) for v in vs]
        data,center = representation(tid,vs,["weights-A"]*len(vs),config)
        data["shot_id"] = f"shot_{i+1:06d}"
        info[tid] = data
        if center is not None:
            centers[tid] = center
            samples[tid] = np.stack(vs)
    return info,samples,centers


def test_clear_same_and_different_candidates():
    cfg = AssociationConfig()
    info,samples,centers = representations({"a":[[1,0,0]]*4,"b":[[.99,.1,0]]*4,"c":[[0,0,1]]*4},cfg)
    clusters,pairs,_,_ = associate(info,samples,centers,cfg)
    assert sorted(sorted(c["track_ids"]) for c in clusters) == [["a","b"],["c"]]
    assert {p["decision"] for p in pairs} == {"same_identity_candidate","different_identity_candidate"}
    assert all(not c["verified_identity"] and c["review_needed"] for c in clusters)


def test_uncertain_similarity_and_bad_quality_do_not_force_match():
    cfg = AssociationConfig()
    info,samples,centers = representations({"a":[[1,0]]*3,"b":[[.5,.8660254]]*3,"c":[[1,0]]},cfg)
    clusters,pairs,_,_ = associate(info,samples,centers,cfg)
    assert len(clusters) == 3
    assert next(p for p in pairs if p["track_b"]=="b")["decision"] == "ambiguous"
    assert all(p["decision"]=="insufficient_evidence" for p in pairs if "c" in (p["track_a"],p["track_b"]))


def test_inconsistent_track_not_hidden_by_mean():
    data,center = representation("mixed",[[1,0],[1,0],[-1,0],[-1,0]],["weights-A"]*4,AssociationConfig())
    assert data["status"] == "inconsistent_track" and not data["eligible"]
    assert center is None
    assert data["internal_consistency"]["minimum"] == -1


def test_complete_link_rejects_transitive_conflict():
    # A--B and B--C pass; A--C lies in the rejection band. A connected-component
    # implementation would merge all three and hide that conflict.
    angle = np.deg2rad(35)
    a,b,c = [np.array([np.cos(t),np.sin(t)]) for t in [-angle,0,angle]]
    cfg = AssociationConfig()
    info,samples,centers = representations({"a":[a]*3,"b":[b]*3,"c":[c]*3},cfg)
    clusters,pairs,_,rejected = associate(info,samples,centers,cfg)
    assert sorted(len(c["track_ids"]) for c in clusters) == [1,2]
    assert rejected and rejected[0]["reason"].startswith("complete_link")
    assert not any(len(c["track_ids"]) == 3 for c in clusters)


def test_model_weights_cannot_mix():
    with pytest.raises(ValueError,match="different embedding model"):
        representation("track",[[1,0],[1,0],[1,0]],["A","A","B"],AssociationConfig())
    info,samples,centers = representations({"a":[[1,0]]*3,"b":[[1,0]]*3})
    info["b"]["model_sha256"] = "other weights"
    with pytest.raises(ValueError,match="different model"):
        associate(info,samples,centers,AssociationConfig())


@pytest.mark.parametrize("v",[[0.,0.],[1.,np.nan],[np.inf,0.]])
def test_bad_embeddings_fail(v):
    with pytest.raises(ValueError): normalize(v)
    with pytest.raises(ValueError,match="dimension"): normalize([1,0],3)


def test_similarity_alignment_known_transform_and_landmark_order():
    image = np.ones((600,600,3),np.uint8)*100
    landmarks = TEMPLATE*4+np.array([100,50])
    crop,matrix,error,fraction = align(image,landmarks)
    np.testing.assert_allclose(matrix,[[.25,0,-25],[0,.25,-12.5]],atol=1e-8)
    assert error < 1e-10 and fraction==1 and crop.shape==(112,112,3)
    with pytest.raises(ValueError,match="ordering"):
        align(image,landmarks[[1,0,2,4,3]])


def test_high_confidence_blurred_or_boundary_crop_is_rejected():
    image = np.full((112,112,3),100,np.uint8)
    row = {"bbox_xywh":[0,0,112,112],"detection_score":.99,"landmarks5":TEMPLATE.tolist()}
    crop,metrics,reasons = crop_quality(image,row)
    assert "blurred_aligned_crop" in reasons and metrics["detection_score"]==.99
    row["landmarks5"] = (TEMPLATE-40).tolist()
    _,_,reasons = crop_quality(image,row)
    assert "landmark_outside_source_image" in reasons or "crop_crosses_source_boundary" in reasons


def test_candidates_cover_time_not_just_adjacent_high_scores():
    rows = [{"frame_index":i,"detection_score":.9+i/10000,"bbox_xywh":[0,0,100,100]} for i in range(160)]
    selected = candidates(rows,16)
    assert len(selected)==16 and selected[0]["frame_index"]==9 and selected[-1]["frame_index"]==159
    assert min(np.diff([r["frame_index"] for r in selected])) == 10


def test_original_stage052_provenance_rejected(tmp_path):
    for name,data in [("run_config.json",{"movie_id":"tt12300742"}),("face_tracks.json",{"movie_id":"tt12300742"}),("qc_report.json",{})]:
        atomic_json(tmp_path/name,data)
    (tmp_path/"face_detections.jsonl").write_text("")
    with pytest.raises(ValueError,match="corrected Stage05.2 provenance"):
        load_corrected(tmp_path,"tt12300742",["shot_000209"])


def test_existing_pilot_cannot_be_overwritten(tmp_path):
    from types import SimpleNamespace
    from oscardp.stage05.identity_pipeline import run
    output = tmp_path/"pilot_001"
    output.mkdir()
    protected = output/"result.txt"
    protected.write_text("keep")
    with pytest.raises(FileExistsError,match="already exists"):
        run(SimpleNamespace(output_dir=output))
    assert protected.read_text()=="keep"


def test_corrected_range_or_boundary_provenance_mismatch_rejected(tmp_path):
    # Small copied run metadata with deliberately inconsistent source-sidecar
    # hashes must fail before source video decoding or inference.
    source_ref = {"mode":"experimental_model_reviewed_pending_human","human_verified":False,
                  "path":str(tmp_path/"sidecar.json"),"sha256":"wrong"}
    atomic_json(tmp_path/"sidecar.json",{"review":"pending"})
    atomic_json(tmp_path/"run_config.json",{"movie_id":"tt12300742","boundary_sidecar":source_ref})
    atomic_json(tmp_path/"face_tracks.json",{"movie_id":"tt12300742","boundary_sidecar":source_ref})
    atomic_json(tmp_path/"qc_report.json",{"boundary_sidecar":source_ref})
    (tmp_path/"face_detections.jsonl").write_text("")
    with pytest.raises(ValueError,match="sidecar changed"):
        load_corrected(tmp_path,"tt12300742",["shot_000209"])
