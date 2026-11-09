"""Explicit, stateless information boundaries for the three branches and ablations."""
import json

COMMON = """You are analysing a research acting annotation prototype, not ground truth.
Return a single JSON object. All times are movie-global seconds, half-open [start,end).
Do not invent observations, exact affect change instants, identities, calibrated confidence or missing modalities.
Separate observable evidence from inferred affect and inferred intent. Use Mixed or Unknown when appropriate.
Keep Affect distinct from Expression Style. All generated interpretations have status model_proposal.
Use concise English fields. Evidence must use actual IDs supplied in the input.
"""


def pack(instruction, data):
    return COMMON + instruction + "\nINPUT JSON:\n" + json.dumps(data, ensure_ascii=False)


def narrative_input(inp):
    return {"start": inp["start"], "end": inp["end"], "scene_context": inp["scene_context"],
            "stage02_shots": inp["stage02_context"], "stage04_character_context": inp["stage04_character_context"],
            "subtitles": inp["subtitles"]}


def narrative_prompt(inp):
    return pack("""NARRATIVE BRANCH. Use only this text. No audio/video observations or Stage05 events are available.
Stage02 screenplay performance cues are text evidence, not confirmed visual behaviour. Scene summaries extend
beyond this excerpt: infer ONLY clip-local proposals, without leaking later developments into earlier states.
Analyse goals/conflict/relationships, possible affect, style and triggers. Avoid a proposal for every sentence.
Return {"proposals":[{"proposal_id":"narrative_001","character_candidate":"MICHELLE|TEDDY|DON|Unknown",
"start":number,"end":number,"affect":string,"expression_style":string|null,
"evidence":["shot_000215","srt_000376"],"uncertainty":[string],"source_modality":"narrative",
"interpretation":string,"possible_trigger":string|null}]}.
""", narrative_input(inp))


def visual_prompt(inp):
    # Deliberate whitelist: no Stage02/04 text, names, event features, or other predictions.
    return pack("""VISUAL BRANCH. Inspect ONLY the attached silent continuous video, sampled at the specified FPS.
There is no audio. Do not lip-read, infer dialogue, use knowledge of a film or identify actors/characters.
First enumerate anonymous persons using visual_person_001 etc and appearance descriptions. Use those IDs
as character_candidate. Describe OBSERVABLE BEHAVIOUR separately from INFERRED AFFECT. Note restrained,
controlled, subtle etc expression styles, possible changes, visibility and insufficient evidence.
Sparse temporal samples do not establish an AU, a microexpression, continuous gaze, or a precise cut timestamp.
Each proposal must be supported by visible behaviour within its own explicit visible_intervals. Break support
at reverse shots or occlusion. No visual affect while that person is offscreen; never extend one still to a scene.
Use several frames over time; refrain from interpreting an anonymous listener's private thoughts.
Video player time 0 equals input movie_global_start. ADD this offset to every reported timestamp.
Return {"subjects":[{"visual_subject_id":string,"description":string}],
"proposals":[{"proposal_id":"visual_001","character_candidate":"visual_person_001",
"start":number,"end":number,"affect":string,"expression_style":string|null,
"observable_behaviour":string,"inferred_affect":string,"visible_intervals":[[number,number]],
"evidence":["visual_only.mp4"],"uncertainty":[string],"source_modality":"video"}],
"limitations":[string]}.
""", {"movie_global_start": inp["start"], "movie_global_end": inp["end"], "sampling_fps": inp["video_fps"]})


def fusion_input(inp, branches, group):
    allowed = {"A": ("narrative",), "B": ("audio", "video"), "C": ("audio", "video", "narrative")}[group]
    # No events or full narrative context may enter the B experiment.
    return {"group": group, "start": inp["start"], "end": inp["end"], "scene_id": inp["scene_id"],
            "scene_bounds": [inp["scene_context"]["start_seconds"], inp["scene_context"]["end_seconds"]],
            "branches": {m: branches[m] for m in allowed}}


def fusion_prompt(inp, branches, group):
    return pack("""TEMPORAL AFFECT SEGMENTATION. Use ONLY the proposals included for this experiment.
Merge compatible nearby proposals into sustained per-character states, without a state for every subtitle,
shot or event. Preserve contradictions and uncertain transition intervals. Unknown-speaker audio cannot
be attributed to a named person; anonymous visual identity associations are explicitly unverified.
Narrative continuity is NOT visible performance continuity. Do not fill unsupported gaps as observed.
No Stage05 events enter this segmentation. Confidence scores are uncalibrated. Audio's four-class ontology
cannot test Fear, Surprise or subtle strategies; Neutral does not prove an absence of fear.
Return {"states":[{"state_id":"state_001","character_candidate":string,"start":number,"end":number,
"affect":string,"expression_style":string|null,"evidence":["actual_proposal_id"],
"uncertainty":[string],"interpretation":string,
"transition":null|{"start":number,"end":number,"description":string,"uncertainty":[string]}}],
"conflicts":[{"proposal_ids":[string],"description":string}]}.
States must not overlap for the same character, but their uncertain transition intervals may straddle boundaries.
""", fusion_input(inp, branches, group))


def links_prompt(inp, states, candidates):
    return pack("""STATE–EVENT SEMANTIC ASSOCIATION. Python already computed positive temporal overlaps and
checked candidate identity compatibility. Overlap is not causation. All Stage05 events remain unreviewed
algorithm_candidate, their character mapping unverified and their boundary corrections experimental.
Do not claim emotional intent from AU codes alone. Raw mouth AU25/AU26 are speech-sensitive, NOT residual
nonverbal actions. Head/gaze are camera-relative and do not establish body-independent movements or targets.
Use only possible_expressive_cue, contextual_response, co_occurring_observation, conflicting_evidence, unresolved.
Return exactly one assessment for EACH supplied link_id:
{"links":[{"link_id":string,"relation":string,"rationale":string,"uncertainty":[string]}]}.
No new link IDs, events, measurements, identity verification, or causal claims.
""", {"states": states, "overlap_candidates": candidates, "subtitles": inp["subtitles"]})


def annotation_prompt(inp, states, links, branches, group):
    modalities = {"A": ("narrative",), "B": ("audio", "video"), "C": ("narrative", "audio", "video")}[group]
    data = {"group": group, "start": inp["start"], "end": inp["end"], "states": states,
            "branches": {m: branches[m] for m in modalities}, "state_event_links": links if group == "C" else []}
    if group in ("A", "C"):
        data["text_context"] = narrative_input(inp)
    return pack("""EXPLAINABLE ACTING ANNOTATION. Use only the evidence in this experiment.
Keep actual observation, possible interpretation, evidence and alternatives separate. For Narrative Only,
observation means text evidence, NEVER witnessed performance. No emotion/intent is ground truth.
Unknown speakers are not named actors. Sampled-video impressions do not verify AUs. Cite actual state,
proposal, event, shot and subtitle IDs. No invented audio descriptions: SER scores only classify a waveform,
and RMS is not pitch, tremor or verified prosody. Visual style can be perceived without confirming inner affect.
Stage05 mouth events may reflect speech. Head/gaze remain camera-relative. Unreviewed events are not facts.
Aim for one concise annotation per meaningful state/transition rather than one for every event.
Return {"annotations":[{"annotation_id":"annotation_001","status":"model_proposal",
"character_candidate":string,"start":number,"end":number,"state_ids":[string],"event_ids":[string],
"observation":[{"source_modality":"audio|video|narrative|stage05","start":number,"end":number,"description":string,"evidence_ids":[string]}],
"interpretation":string,"evidence":[string],"alternative_interpretation":[string],"uncertainty":[string]}]}.
""", data)
