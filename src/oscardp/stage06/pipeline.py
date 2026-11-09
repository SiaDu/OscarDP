"""Local preparation and assembly of real interactive-assistant annotations. No external API."""
import copy
from pathlib import Path

from .inputs import MOVIE, load_sources, read, selection, write
from .media import annotated_review, extract, frame_sheets
from .models import audio_affect
from .temporal import (apply_relations, check_annotations, check_proposals, check_states,
                       match_events, unique)


METHOD = {
    "authoring": "current_interactive_assistant",
    "external_inference_api_calls": 0,
    "gemini_used": False,
    "experiment_design": "Modality evidence restrictions within one conversation; not blinded independent sessions.",
    "limitations": [
        "The same assistant had prior source-context and SER-label exposure. Within-chat memory cannot be reset.",
        "Initial first-three-sheet preview exposed burned-in subtitles; all 12 sheets were re-inspected with the bottom 22 percent masked.",
        "Visual reasoning uses 230 nominal 2 FPS frames and one tail frame, not continuous full-frame-rate playback.",
        "No direct assistant listening/prosody assessment was performed; Audio is genuine local waveform SER.",
        "Saved proposals can be replayed, but a CLI cannot automatically rerun this interactive assistant's cognition.",
    ],
}


def mapped_branches(branches, identity):
    result = copy.deepcopy(branches)
    mapping = {m["visual_subject_id"]:m["character_candidate"] for m in identity["mappings"]}
    for p in result["video"]["proposals"]:
        p["visual_subject_id"] = p["character_candidate"]
        name = mapping.get(p["character_candidate"], "Unknown")
        if name != "Unknown":
            p["character_candidate"] = name
        p["identity_status"] = "unverified_model_association" if name != "Unknown" else "unresolved_anonymous_subject"
    return result


def proposals_for(branches, group):
    modalities = {"A": ("narrative",), "B": ("audio", "video"), "C": ("audio", "video", "narrative")}[group]
    return [p for m in modalities for p in branches[m]["proposals"]]


def subtitle_time(seconds):
    ms = round(seconds*1000)
    h, rem = divmod(ms,3600000)
    m, rem = divmod(rem,60000)
    s, ms = divmod(rem,1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def review_subtitles(output, inp, states, events, target="review_video.srt"):
    lines=[]
    cursor=inp["start"]
    cue=0
    while cursor < inp["end"]:
        cue+=1
        end=min(cursor+1.,inp["end"])
        active=[s for s in states if s["start"] < end and s["end"] > cursor]
        local=[e for e in events if e["start"] < end and e["end"] > cursor]
        text=[f"Movie {cursor:.3f}–{end:.3f}s | MODEL PROPOSALS / UNREVIEWED EVENTS"]
        text += [f"{s['character_candidate']}: {s['affect']} / {s['expression_style']} [{s['state_id']}]" for s in active]
        if local:
            text += ["Events: "+", ".join(f"{e['event_id']} {e['feature']}" for e in local[:5]) +
                     (f" (+{len(local)-5}; see JSON)" if len(local)>5 else "")]
        lines += [str(cue),f"{subtitle_time(cursor-inp['start'])} --> {subtitle_time(end-inp['start'])}","\n".join(text),""]
        cursor=end
    with (output/target).open("x",encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def local_prepare(args):
    output=Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Output exists; choose a fresh directory: {output}")
    data=load_sources(args.pilot_input)
    inp=selection(data,args.start,args.end)
    inp["video_fps"]=args.video_fps
    output.mkdir(parents=True,exist_ok=False)
    print("Stage06 continuous media extraction",flush=True)
    media=extract(inp,output)
    frame_sheets(output,inp["duration"],args.video_fps)
    write(output/"media_info.json",media)
    inp["audio_video_coverage"].update(audio=media["audio"],video=media["video"],actual_review_duration=media["review_duration"])
    write(output/"stage06_input.json",inp)
    print("Stage06 real local VAD/SER inference",flush=True)
    try:
        audio=check_proposals(audio_affect(inp,output,args.ser_model,args.model_dir),"audio",(inp["start"],inp["end"]))
    except Exception as exc:
        audio=dict(status="failed",model=args.ser_model,proposals=[],error=f"{type(exc).__name__}: {exc}")
    write(output/"audio_affect.json",audio)
    write(output/"local_preparation.json",dict(status="local_preparation",audio_status=audio["status"],external_api_calls=0,
          next_step="Interactive assistant inspects masked frame sequence and source text, then saves actual annotation JSONs for assembly."))
    review_subtitles(output,inp,[],inp["stage05_event_references"])
    print(f"Local preparation saved: {output}; {len(audio['proposals'])} audio proposals",flush=True)
    return output


def load_analysis(output, analysis):
    inp=read(output/"stage06_input.json")
    bounds=inp["start"],inp["end"]
    branches={"audio":read(output/"audio_affect.json"), "video":read(analysis/"visual_affect.json"),
              "narrative":read(analysis/"narrative_affect.json")}
    for modality,doc in branches.items():
        check_proposals(doc,modality,bounds)
    allowed=set(inp["stage02_context_references"])|{f"srt_{s['srt_index']:06d}" for s in inp["subtitles"]}
    for p in branches["narrative"]["proposals"]:
        if not set(p["evidence"]) <= allowed or p["character_candidate"] not in inp["character_candidate"]+["Unknown"]:
            raise ValueError("Narrative evidence/character mismatch")
    identity=read(analysis/"visual_identity_map.json")
    if unique(identity["mappings"],"visual_subject_id") != unique(branches["video"]["subjects"],"visual_subject_id"):
        raise ValueError("Visual identity mapping inventory mismatch")
    for m in identity["mappings"]:
        if m["status"] != "unverified_model_association" or m["character_candidate"] not in inp["character_candidate"]+["Unknown"]:
            raise ValueError("Visual mapping cannot verify or invent identities")
    return inp,branches,identity,mapped_branches(branches,identity)


def group_document(analysis, group, branches, inp):
    doc=read(analysis/(f"ablation_{group}.json" if group in ("A","B") else "affect_states.json"))
    check_states(doc,proposals_for(branches,group),(inp["start"],inp["end"]),group)
    if group in ("A","B"):
        check_annotations(doc,doc["states"],[],proposals_for(branches,group),inp)
    doc["input_modalities"]={"A":["narrative"],"B":["audio","video"],"C":["audio","video","narrative"]}[group]
    doc["status"]="model_proposal"
    doc["method_limitations"]=METHOD["limitations"]
    return doc


def report(output,inp,branches,groups,links,annotations,comparison):
    states=groups["C"]["states"]
    events=inp["stage05_event_references"]
    linked={l["event_id"] for l in links}
    text="# Bugonia Stage06 A/B/C exploratory comparison\n\n"
    text+="This is a modality-restricted comparison authored by the current assistant in one conversation, **not a blinded independent-model experiment**. A/B results were saved before C. The same model had earlier context and SER-label exposure, and a first preview exposed burned-in captions. Final visual evidence was re-inspected with captions masked. These limitations prevent a clean causal ablation claim. No external inference API was used.\n\n"
    text+="| Group | Inputs | States | Transition intervals | Annotations |\n|---|---|---:|---:|---:|\n"
    for g in ("A","B","C"):
        d=groups[g]
        text+=f"| {g} | {', '.join(d['input_modalities'])}{' + Stage05 for links' if g=='C' else ''} | {len(d['states'])} | {sum(s['transition'] is not None for s in d['states'])} | {len(d.get('annotations',[]))} |\n"
    for key,value in comparison["comparison"].items():
        text+=f"\n## {key.replace('_',' ').title()}\n\n{value}\n"
    for g in ("A","B","C"):
        text+=f"\n## {g} state inventory\n\n"
        for s in groups[g]["states"]:
            text+=f"- {s['state_id']} {s['character_candidate']} {s['start']:.3f}–{s['end']:.3f}s: {s['affect']} / {s['expression_style']}. {s['interpretation']}\n"
            if s["transition"]:
                t=s["transition"]
                text+=f"  Transition interval: {t['start']:.3f}–{t['end']:.3f}s; {t['description']}\n"
    (output/"comparison_abc.md").write_text(text,encoding="utf-8")
    review=f"""# Stage06 Bugonia research prototype — model proposals

Continuous excerpt: **{inp['start']:.6f}–{inp['end']:.6f} global seconds**, {inp['duration']:.6f}s, {inp['scene_id']}.
Interruption, attempt to regain conversational control, and a complete negotiation offer.
{inp['boundary_note']}

| Candidate excerpt | Global seconds | Stage05 events | Selection reason |
|---|---|---:|---|
"""
    for c in inp["scene_candidates"]:
        review+=f"| {c['title']} | {c['start']:.3f}–{c['end']:.3f} | {c['event_count']} | {c['reason']} |\n"
    review+=f"\nCharacters: {', '.join(inp['character_candidate'])}. Associations remain unverified; no actor recognition.\n\n"
    review+="## Actual inference\n\n- Audio: Silero JIT VAD (16kHz, 32ms windows, threshold 0.5), then local CPU `superb/wav2vec2-base-superb-er` with Transformers 4.57.6 and existing PyTorch. Only waveform is sent into SER. Subtitle times/speaker hints are unverified text attribution; mixed speakers remain Unknown.\n- Video: the current interactive assistant actually inspected all 12 masked timestamped contact sheets, 230 regular 2 FPS samples plus one tail frame. No Gemini or other external inference API. Original continuous clip is silent for visual evidence; bottom 22 percent was masked to hide burned-in captions.\n- Narrative, state fusion, semantic links, A/B/C interpretations and comparison: current interactive assistant, structured JSON saved in assistant_analysis/. Exact backend API model ID was not independently verified; no API model name is invented.\n\n"
    for m,d in branches.items():
        review+=f"- {m}: {d['status']}, {len(d['proposals'])} proposals.\n"
    review+=f"\n## Results\n\n{len(states)} C affect states; {sum(s['transition'] is not None for s in states)} uncertain transition intervals; {len(linked)} distinct linked Stage05 events; {len(links)} overlap links; {len(annotations)} acting annotations. All annotations are model_proposal. All {len(events)} event references remain algorithm_candidate and human_verified=false. No source measurement values or earlier outputs were modified.\n\n"
    review+="## Artifact fields\n\n- stage06_input.json: movie/scene ID, global start/end, candidate comparison, source paths and actual Stage02/04 context, original subtitle timing/speaker candidates, compact event references, actual media coverage.\n- audio_affect.json: actual four-class labels, all softmax scores, VAD speech intervals, descriptive RMS and uncertainty. No valence/arousal; no verified diarization or pitch/tremor measurements.\n- visual_affect.json: anonymous subjects, global intervals, observable_behaviour distinct from inferred_affect, visible_intervals and exact inspected-sheet references.\n- narrative_affect.json: text-only interpretations, proposed triggers and Shot/subtitle references. Screenplay cues are not visual observations.\n- visual_identity_map.json: subsequent appearance associations to Stage04 candidates; unverified.\n- affect_states.json: character states, affect/style separately, transitions, evidence IDs, conflicts and observation_coverage (audio/video/narrative_support/no_av_evidence). Coverage is proposal support, not continuous verified observation.\n- state_event_links.json: Python overlap and candidate checks, original Event IDs and proposed relations/rationales. No causal or identity verification and no full numeric arrays.\n- acting_annotations.json: observation/interpretation/evidence/alternative_interpretation/uncertainty, state/event IDs and global times.\n- stage06_prototype.json: integrated states, compact event references, links, annotations and method limitations.\n- ablation_A.json / ablation_B.json and comparison_abc.md: evidence-restricted A/B results and actual comparisons; no superiority assumption.\n- review_video.mp4: continuous source excerpt with original soundtrack. review_video_states.srt: optional final state/event overlay; review_video.srt is the initial event-only overlay. Subtitle playback times are necessarily clip-relative, while their labels and all annotation JSON use movie-global seconds. Movie time = player time + selected start.\n- observed_frames_*.jpg and observed_tail.jpg: actual inspected visual evidence; nominal 2 FPS samples and approximate boundaries. Initial frames_*.jpg are unmasked previews and not cited as final visual evidence.\n- media_info.json: source stream time bases, actual encoded duration and extraction commands.\n\n"
    review+="## Required human review and research limits\n\n1. Watch the entire original-audio clip. Confirm whether the proposed Michelle transition is an observable change, a rhetorical turn, or both. Estimated transition intervals are not precise emotion-onset timestamps.\n2. Listen critically to all 44 SER proposals, especially the 26 Happiness labels. These are actual uncalibrated classifier outputs; no tuning/relabeling was done to match the plot. A four-class model cannot test fear/surprise or establish emotional neutrality.\n3. Check facial tension versus sustained anger, and brief mouth-corner changes versus happiness. Reverse shots and the wide-shot resolution change do not prove emotional transitions.\n4. Independently review all Stage05 event presence, QC flags, tentative character mapping and experimental corrected cuts. AU25/AU26 are speech-sensitive; no speech residual was computed. Head/gaze remain camera-relative; no head–body disentanglement or gaze target is claimed.\n5. Only three Shots have Stage05 measurements in this excerpt. Unprocessed intervals are missing coverage, not negative events; no cross-Shot interpolation was performed.\n6. Narrative continuity does not prove visible acting continuity. All inner motives, emotions, strategies and interpretations remain proposals, with alternatives. No human review or ground-truth evaluation was created.\n7. This single-session analysis cannot establish the requested strict blind independence. See method.json and comparison_abc.md for prior-context and subtitle-exposure limitations.\n\n"
    review+="## Reproduce local preparation and replay saved assistant analysis\n\n```sh\nuv --no-cache run --no-sync python -m oscardp.stage06 run \\\n  --analysis-dir reports/stage06/tt12300742/pilot_001/assistant_analysis \\\n  --output-dir reports/stage06/tt12300742/pilot_002\n```\n\nThis reruns real local media/VAD/SER and replays the already authored interactive proposals. It does **not** rerun an LLM/VLM API or automatically reproduce this conversation's cognitive analysis. For new model interpretations, inspect the new evidence interactively and supply a new analysis folder. Existing completed outputs are refused. No API key is needed.\n\n"
    review+="Validation: new Stage06 time/ID/coverage/isolation/refusal tests passed. The initial full repository run passed 394 tests and failed one pre-existing HTTP test because sandbox sockets were prohibited; that exact test passed when loopback access was allowed. Later Stage06 changes were tested separately.\n"
    (output/"REVIEW.md").write_text(review,encoding="utf-8")


def assemble(output,analysis):
    output,analysis=Path(output),Path(analysis)
    names=["visual_affect.json","narrative_affect.json","visual_identity_map.json","ablation_A.json","ablation_B.json",
           "affect_states.json","state_event_links.json","acting_annotations.json","comparison_abc.json","comparison_abc.md",
           "stage06_prototype.json","method.json","modality_quality.json","review_video_states.srt",
           "review_video_annotated.mp4","review_overlay.ass"]
    if any((output/n).exists() for n in names):
        raise FileExistsError("Stage06 results already exist; use a fresh run directory")
    inp,raw,identity,branches=load_analysis(output,analysis)
    groups={g:group_document(analysis,g,branches,inp) for g in ("A","B","C")}
    events=inp["stage05_event_references"]
    candidates=match_events(groups["C"]["states"],events)
    links=apply_relations(candidates,read(analysis/"relation_assessments.json"))
    for link in links:
        event=link.pop("event")
        link.update(feature=event["feature"],event_type=event["event_type"],
                    event_review_status=event["review_status"],event_qc_flags=event["qc_flags"],
                    event_source_reference=event["evidence_reference"],character_mapping_status="unverified")
    ann=read(analysis/"acting_annotations.json")
    check_annotations(ann,groups["C"]["states"],events,proposals_for(branches,"C"),inp)
    comparison=read(analysis/"comparison_abc.json")
    required={"affect_types","boundaries_and_transitions","interpretation_differences","evidence_sufficiency","full_pipeline_added_value","conflicts_and_unsupported_claims"}
    if set(comparison["comparison"]) != required or not all(isinstance(v,str) and v for v in comparison["comparison"].values()):
        raise ValueError("Incomplete A/B/C comparison")
    # Only write final results after all basic structural/time/reference checks pass.
    for name,doc in [("visual_affect.json",raw["video"]),("narrative_affect.json",raw["narrative"]),
                     ("visual_identity_map.json",identity),("ablation_A.json",groups["A"]),("ablation_B.json",groups["B"]),
                     ("affect_states.json",groups["C"]),("state_event_links.json",dict(status="model_proposal",links=links)),
                     ("acting_annotations.json",ann),("comparison_abc.json",comparison),("method.json",METHOD)]:
        write(output/name,doc)
    groups["C"]["annotations"]=ann["annotations"]
    write(output/"stage06_prototype.json",dict(schema_version="stage06.research_prototype.v1",status="model_proposal",movie_id=MOVIE,
          scene_id=inp["scene_id"],character_candidate=inp["character_candidate"],time_range=dict(start=inp["start"],end=inp["end"],unit="movie_global_seconds"),
          affect_states=groups["C"]["states"],events=events,state_event_links=links,acting_annotations=ann["annotations"],human_verified=False,method=METHOD))
    write(output/"modality_quality.json",dict(status={m:d["status"] for m,d in raw.items()},proposal_counts={m:len(d["proposals"]) for m,d in raw.items()},missing_modalities=[],method_limits=METHOD["limitations"]))
    if (output/"REVIEW.md").exists():
        if (output/"LOCAL_REVIEW.md").exists():
            raise FileExistsError("Existing LOCAL_REVIEW.md; refusing overwrite")
        (output/"REVIEW.md").rename(output/"LOCAL_REVIEW.md")
    report(output,inp,raw,groups,links,ann["annotations"],comparison)
    review_subtitles(output,inp,groups["C"]["states"],events,"review_video_states.srt")
    annotated_review(output)
    print(f"Assembled {len(groups['C']['states'])} states, {len({l['event_id'] for l in links})} events, {len(ann['annotations'])} annotations: {output}",flush=True)
    return 0


def run(args):
    output=Path(args.output_dir)
    if output.exists() and not getattr(args,"prepared",False):
        raise FileExistsError(f"Output exists; choose a fresh directory: {output}")
    if not getattr(args,"prepared",False):
        local_prepare(args)
    elif not (output/"local_preparation.json").exists():
        raise ValueError("--prepared requires a completed local preparation")
    if args.command=="prepare":
        return 0
    if not args.analysis_dir:
        raise ValueError("Supply --analysis-dir with actual interactive-assistant annotations; no external inference is performed")
    return assemble(output,Path(args.analysis_dir))
