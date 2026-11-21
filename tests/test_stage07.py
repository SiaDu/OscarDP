"""Synthetic structural tests only; never write synthetic inference into research reports."""
import copy

import pytest

from oscardp.stage07.inputs import interval, overlap
from oscardp.stage07.prompts import fusion_input, narrative_input, visual_prompt
from oscardp.stage07.temporal import apply_relations, check_annotations, check_proposals, check_states, coverage, match_events


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
    from oscardp.stage07.pipeline import run
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


def test_cli_defaults_and_explicit_legacy_weights(monkeypatch):
    import sys
    from oscardp.stage07 import __main__ as cli
    captured = []
    monkeypatch.setattr(cli, "run", lambda args: captured.append(args) or 0)
    for command in ("prepare", "run"):
        argv = ["stage07", command, "--model-dir", "models/stage06"]
        if command == "run":
            argv += ["--analysis-dir", "reports/stage06/tt12300742/pilot_001/assistant_analysis"]
        monkeypatch.setattr(sys, "argv", argv)
        assert cli.main() == 0
    assert all(a.output_dir == "reports/stage07/tt12300742/pilot_001" for a in captured)
    assert all(a.model_dir == "models/stage06" for a in captured)


def test_legacy_prepared_reader_does_not_modify_source(tmp_path):
    from oscardp.stage07.inputs import read_prepared_input, write
    legacy = dict(schema_version="stage06.research_input.v1", start=10., end=20., sources={"stage05_6": "stage06_input_pilot.json"})
    write(tmp_path/"stage06_input.json", legacy)
    before = (tmp_path/"stage06_input.json").read_bytes()
    result = read_prepared_input(tmp_path)
    assert result["schema_version"] == "stage07.research_input.v1"
    assert result["sources"] == legacy["sources"]
    assert "legacy_compatibility" in result
    assert (tmp_path/"stage06_input.json").read_bytes() == before
    current = dict(schema_version="stage07.research_input.v1", start=30., end=40.)
    write(tmp_path/"stage07_input.json", current)
    assert read_prepared_input(tmp_path) == current


def test_historical_reports_are_not_writable_by_stage07():
    from pathlib import Path
    from types import SimpleNamespace
    from oscardp.stage07.pipeline import assemble, local_prepare, run
    old = Path(__file__).resolve().parents[1]/"reports/stage06/tt12300742/new_run"
    with pytest.raises(ValueError, match="read-only"):
        local_prepare(SimpleNamespace(output_dir=old))
    with pytest.raises(ValueError, match="read-only"):
        assemble(old, "unused")
    with pytest.raises(ValueError, match="read-only"):
        run(SimpleNamespace(output_dir=old, prepared=True))


def test_completed_legacy_result_refused_in_custom_directory(tmp_path):
    from oscardp.stage07.pipeline import assemble
    (tmp_path/"stage06_prototype.json").write_text("{}")
    with pytest.raises(FileExistsError, match="results already exist"):
        assemble(tmp_path, "unused")


def test_stage06_namespace_is_reserved_not_acting_alias():
    import importlib.util
    assert importlib.util.find_spec("oscardp.stage06") is None


def test_stage07_dependency_extra_matches_lock():
    import tomllib
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root/"pyproject.toml").read_text())["project"]
    lock = tomllib.loads((root/"uv.lock").read_text())
    package = next(p for p in lock["package"] if p["name"] == "oscardp")
    assert "stage06" not in project["optional-dependencies"]
    assert "stage07" in project["optional-dependencies"]
    assert package["optional-dependencies"]["stage07"] == [{"name": "transformers"}]
    assert "stage07" in package["metadata"]["provides-extras"]


@pytest.fixture
def legacy_pilot():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]/"reports/stage06/tt12300742/pilot_001"
    if not (root/"stage06_prototype.json").exists():
        pytest.skip("Optional local Legacy Prototype artifacts are absent")
    return root


def test_saved_legacy_analysis_replays_identical_core_results(legacy_pilot, tmp_path, monkeypatch):
    import shutil
    from oscardp.stage07 import pipeline
    from oscardp.stage07.inputs import read
    # Only copy small JSONs into pytest's scratch directory; no historical media/output is written.
    for name in ("stage06_input.json", "audio_affect.json", "local_preparation.json"):
        shutil.copyfile(legacy_pilot/name, tmp_path/name)
    rendered = []
    monkeypatch.setattr(pipeline, "annotated_review", lambda output: rendered.append(output))
    def no_inference(*args, **kwargs):
        raise AssertionError("Migration verification must not run a model or decode video")
    for name in ("audio_affect", "extract", "frame_sheets"):
        monkeypatch.setattr(pipeline, name, no_inference)
    assert pipeline.assemble(tmp_path, legacy_pilot/"assistant_analysis") == 0
    actual = read(tmp_path/"stage07_prototype.json")
    expected = read(legacy_pilot/"stage06_prototype.json")
    for key in ("affect_states", "events", "state_event_links", "acting_annotations", "time_range"):
        assert actual[key] == expected[key]
    assert actual["schema_version"] == "stage07.research_prototype.v1"
    assert actual["method"]["stage"] == "Stage07"
    assert read(tmp_path/"stage07_input.json")["schema_version"] == "stage07.research_input.v1"
    assert rendered == [tmp_path]
    review = (tmp_path/"REVIEW.md").read_text()
    assert str(legacy_pilot/"assistant_analysis") in review
    assert "oscardp.stage07 run" in review


def test_actual_stage05_packet_legacy_filename_is_readable(legacy_pilot):
    from oscardp.stage07.inputs import load_sources, selection
    data = load_sources(legacy_pilot.parents[2]/"stage05_6/tt12300742/pilot_001")
    result = selection(data)
    assert result["schema_version"] == "stage07.research_input.v1"
    assert result["sources"]["stage05_6"].endswith("stage06_input_pilot.json")
    assert len(result["stage05_event_references"]) == 81
