"""Synthetic structural tests only; never write synthetic inference into research reports."""
import copy

import pytest

from oscardp.stage06.inputs import interval, overlap
from oscardp.stage06.prompts import fusion_input, narrative_input, visual_prompt
from oscardp.stage06.temporal import apply_relations, check_annotations, check_proposals, check_states, coverage, match_events


def proposal(name, modality, start, end, character="MICHELLE"):
    return dict(proposal_id=name,character_candidate=character,source_modality=modality,start=start,end=end,
                speech_intervals=[[start,end]],visible_intervals=[[start,end]])


def test_exclusive_end_no_event_overlap_and_candidate_compatibility():
    states=[dict(state_id="s",character_candidate="MICHELLE",start=10.,end=20.)]
    events=[dict(event_id="touch",character_candidate="MICHELLE",start=20.,end=21.),
            dict(event_id="wrong",character_candidate="TEDDY",start=12.,end=13.),
            dict(event_id="unknown",character_candidate="Unknown",start=12.,end=13.),
            dict(event_id="ok",character_candidate="MICHELLE",start=19.,end=22.)]
    links=match_events(states,events)
    assert [l["event_id"] for l in links]==["unknown","ok"]
    assert links[0]["character_compatibility"]=="unresolved_identity"
    assert links[1]["overlap_seconds"]==1.
    assert all(l["relation"]=="unresolved" for l in links)
    assert overlap((10.,20.),(20.,21.))==0


def test_unknown_identity_cannot_become_expressive_cue():
    links=match_events([dict(state_id="s",character_candidate="MICHELLE",start=10.,end=20.)],
                       [dict(event_id="e",character_candidate="Unknown",start=12.,end=13.)])
    result={"links":[dict(link_id=links[0]["link_id"],relation="possible_expressive_cue",rationale="test",uncertainty=[])]}
    assert apply_relations(links,result)[0]["relation"]=="unresolved"
    with pytest.raises(ValueError,match="omits or invents"):
        apply_relations(links,{"links":[]})


def test_narrative_continuity_does_not_fill_reverse_shot_visibility():
    state=dict(character_candidate="MICHELLE",start=10.,end=30.)
    ps=[proposal("n","narrative",10.,30.),proposal("v1","video",10.,14.),
        proposal("v2","video",20.,25.),proposal("a","audio",15.,16.)]
    result=coverage(state,ps)
    assert result["video"]==[[10.,14.],[20.,25.]]
    assert result["narrative_support"]==[[10.,30.]]
    assert result["no_av_evidence"]==[[14.,15.],[16.,20.],[25.,30.]]


def test_nan_times_and_fabricated_state_evidence_rejected():
    with pytest.raises(ValueError):
        interval(float("nan"),20.)
    p=proposal("audio1","audio",10.,20.)
    state=dict(state_id="s",character_candidate="MICHELLE",start=10.,end=20.,affect="Unknown",
               expression_style=None,transition=None,evidence=["invented"],uncertainty=[])
    with pytest.raises(ValueError,match="unknown proposals"):
        check_states({"states":[state]},[p],(10.,20.),"C")
    state["evidence"]=["audio1"]
    other=copy.deepcopy(state);other.update(state_id="s2",start=15.)
    with pytest.raises(ValueError,match="Overlapping"):
        check_states({"states":[state,other]},[p],(10.,20.),"C")


def test_branch_and_ablation_inputs_are_whitelisted():
    inp=dict(start=10.,end=20.,scene_id="sc",scene_context=dict(start_seconds=0.,end_seconds=30.,summary="SECRET_NARRATIVE"),
             stage02_context=["SECRET_DIALOGUE"],stage04_character_context=[],subtitles=[],video_fps=2.)
    branches={m:{"proposals":[m]} for m in ("audio","video","narrative")}
    b=fusion_input(inp,branches,"B")
    assert set(b["branches"])=={"audio","video"}
    assert "SECRET" not in str(b)
    a=fusion_input(inp,branches,"A")
    assert set(a["branches"])=={"narrative"}
    visual=visual_prompt(inp)
    assert "SECRET" not in visual
    assert set(narrative_input(inp))=={"start","end","scene_context","stage02_shots","stage04_character_context","subtitles"}


def test_existing_results_refused_before_source_access(tmp_path):
    from types import SimpleNamespace
    from oscardp.stage06.pipeline import run
    with pytest.raises(FileExistsError):
        run(SimpleNamespace(output_dir=str(tmp_path)))


def test_audio_without_speech_and_narrative_claim_in_b_rejected():
    p=proposal("a","audio",10.,20.)
    p.update(affect="Unknown",expression_style=None,evidence=["scene_audio.wav"],uncertainty=[])
    p["speech_intervals"]=[]
    with pytest.raises(ValueError,match="speech support"):
        check_proposals({"proposals":[p]},"audio",(10.,20.))
    s=dict(state_id="s",character_candidate="MICHELLE",start=10.,end=20.)
    a=dict(annotation_id="a",status="model_proposal",character_candidate="MICHELLE",start=10.,end=20.,
           state_ids=["s"],event_ids=[],evidence=["a"],interpretation="test",alternative_interpretation=["test"],uncertainty=[],
           observation=[dict(source_modality="narrative",start=10.,end=20.,description="test",evidence_ids=["a"])])
    inp=dict(start=10.,end=20.,stage02_context_references=[],subtitles=[])
    with pytest.raises(ValueError,match="modality excluded"):
        check_annotations({"annotations":[a]},[s],[],[p],inp)
