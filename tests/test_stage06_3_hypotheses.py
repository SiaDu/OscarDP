import pytest

from oscardp.stage06_3 import hypotheses
from oscardp.stage06_3.hypotheses import decision, fused_prediction, geometry_prediction, visual_prediction
from oscardp.stage06_3.hypothesis_evaluation import METHODS, evaluate_human_labels, read_human_labels

STATE = {"state_id": "state", "direction_source": "eye_head", "qc_status": "stable",
         "look_direction": {"vector": [0.15, 0.05, -0.98]}}
IDENTITY = {"character_candidates": ["TEDDY"], "mapping_strength": "supported"}


def fit(target, region, point):
    return {"target": target, "camera_branches": {
        name: {"compatible": count, "tested": 100, "median_error_deg": error}
        for name, (count, error) in (("candidate_region", region), ("inherited_point", point))
    }}


def geometry(preferred, alternatives=(), stronger=(), level="gaze_supported"):
    return {
        **decision(preferred, alternatives, level, "Schematic fit."),
        "mapping_strength": "supported",
        "spatial_by_gazer": {},
        "substantially_compatible_targets": stronger,
    }


def visual(preferred, alternatives=(), level="visual_directional"):
    return visual_prediction({
        "preferred_target": preferred, "alternative_targets": alternatives,
        "evidence_level": level, "reason": "Observed at sampled frames.",
    })


def test_geometry_ranks_branch_consistent_fit_without_confirming_it(monkeypatch):
    tests = [
        fit("MICHELLE", (80, 15), (20, 35)),
        fit("DON", (10, 55), (5, 80)),
    ]

    def replay(state, camera, layout, branches, orientation, baseline):
        assert state["character"] == "TEDDY"
        assert baseline == []
        return {"geometry": {"tested_targets": tests}, "selected_target": "UNKNOWN"}

    monkeypatch.setattr(hypotheses, "geometry_for", replay)
    result = geometry_prediction(STATE, IDENTITY, (None, None, None, None))
    assert result["target_ranking"] == [["MICHELLE"], ["DON"]]
    assert result["selected_target"] == "UNKNOWN"

    tests[1] = fit("DON", (10, 55), (90, 8))
    result = geometry_prediction(STATE, IDENTITY, (None, None, None, None))
    assert result["preferred_target"] is None
    assert result["target_ranking"] == [["DON", "MICHELLE"]]
    assert result["decision_status"] == "ambiguous"


def test_fused_agreement_promotes_hypothesis_and_keeps_alternatives():
    v = visual("MICHELLE", ("OTHER",))
    g = geometry("MICHELLE", ("DON",), stronger=("MICHELLE",))
    result = fused_prediction(STATE, v, g)
    assert result["target_ranking"] == [["MICHELLE"], ["DON", "OTHER"]]
    assert result["evidence_level"] == "gaze_supported"
    assert result["decision_status"] == "supported_hypothesis"
    assert result["selected_target"] == "UNKNOWN"


def test_fusion_keeps_both_hypotheses_during_conflict():
    g = geometry("DON", stronger=("DON",))
    g["spatial_by_gazer"] = {"TEDDY": {"geometry": {"tested_targets": [fit("MICHELLE", (0, 80), (0, 80))]}}}
    result = fused_prediction(STATE, visual("MICHELLE"), g)
    assert result["target_ranking"] == [["MICHELLE"], ["DON"]]
    assert result["evidence_level"] == "visual_directional"
    assert result["selected_target"] == "UNKNOWN"
    assert result["visual_geometry_preference"] == "conflict"
    assert {conflict["kind"] for conflict in result["conflicts"]} == {
        "preferred_targets_disagree", "visual_candidate_outside_geometry"
    }
    context_result = fused_prediction(STATE, visual("MICHELLE", level="context_only"), g)
    assert context_result["target_ranking"] == [["DON"], ["MICHELLE"]]
    assert context_result["decision_status"] == "preferred_but_unconfirmed"


def test_head_only_tie_and_context_only_hypothesis_remain_distinct():
    v = visual(None, ("TEDDY", "DON"))
    g = geometry("TEDDY", ("DON",), stronger=("TEDDY",), level="head_supported")
    head = fused_prediction({**STATE, "direction_source": "head_only"}, v, g)
    assert head["preferred_target"] is None
    assert set(head["alternative_targets"]) == {"TEDDY", "DON"}
    assert head["evidence_level"] == "head_supported"

    state = {**STATE, "direction_source": "unobservable", "qc_status": "blink_gap", "look_direction": None}
    context = fused_prediction(state, visual("MICHELLE", level="context_only"),
                               geometry(None, level="insufficient"))
    assert context["preferred_target"] == "MICHELLE"
    assert context["selected_target"] == "UNKNOWN"
    assert context["evidence_level"] == "context_only"


def test_confirmation_requires_both_methods_and_stable_direction():
    v = visual("MICHELLE", level="visual_endpoint")
    g = {**geometry("MICHELLE", stronger=("MICHELLE",)), "selected_target": "MICHELLE"}
    assert fused_prediction(STATE, v, g)["selected_target"] == "MICHELLE"
    state = {**STATE, "qc_status": "heterogeneous_stable_support"}
    assert fused_prediction(state, v, g)["selected_target"] == "UNKNOWN"
    assert fused_prediction(STATE, v, {**g, "selected_target": "UNKNOWN"})["selected_target"] == "UNKNOWN"


def test_human_evaluation_excludes_unclear_and_blanks():
    rows = [
        {"state_id": "a", **decision("MICHELLE", (), "gaze_supported", "", supported=True)},
        {"state_id": "b", **decision(None, ("TEDDY", "DON"), "visual_directional", "")},
    ]
    predictions = {method: rows for method in METHODS}
    pending = evaluate_human_labels(predictions, {"b": "", "c": "UNCLEAR"})
    assert pending == {"labeled_states": 0, "unclear_labels": 1, "blank_labels": 1,
                       "status": "awaiting_evaluable_human_labels"}
    report = evaluate_human_labels(predictions, {"a": "MICHELLE", "b": "DON", "c": "UNCLEAR", "d": ""})
    assert report["labeled_states"] == 2
    assert report["methods"]["fused"]["top1_accuracy"] == 1.0
    assert report["methods"]["fused"]["candidate_recall"] == 1.0
    assert report["methods"]["fused"]["prediction_coverage"] == 0.5
    assert report["methods"]["fused"]["accuracy_coverage_tradeoff"]["selected"]["top1_accuracy"] is None


def test_review_label_unknown_is_invalid(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("state_id,shot_id,track_id,human_target\nstate,shot,track,UNKNOWN\n")
    with pytest.raises(ValueError, match="Invalid human target"):
        read_human_labels(path, [{"state_id": "state", "shot_id": "shot", "track_id": "track"}])
