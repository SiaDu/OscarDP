"""Minimal time/ID checks, evidence coverage and deterministic State–Event overlap."""
from .inputs import interval, overlap

RELATIONS = {"possible_expressive_cue", "contextual_response", "co_occurring_observation", "conflicting_evidence", "unresolved"}


def unique(rows, key):
    ids = [r[key] for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Missing/duplicate {key}")
    return set(ids)


def check_proposals(doc, modality, bounds):
    rows = doc["proposals"]
    unique(rows, "proposal_id")
    for p in rows:
        interval(p["start"], p["end"], bounds)
        for k in ("character_candidate", "affect", "expression_style", "evidence", "uncertainty", "source_modality"):
            if k not in p:
                raise ValueError(f"Missing proposal field {k}")
        if p["source_modality"] != modality or not p["evidence"]:
            raise ValueError("Wrong branch modality or empty evidence")
        if modality == "audio":
            if not p.get("speech_intervals"):
                raise ValueError("Audio proposal has no real speech support")
            for a,b in p["speech_intervals"]:
                interval(a,b,(p["start"],p["end"]))
        if modality == "video":
            if not p.get("observable_behaviour") or not p.get("inferred_affect") or not p.get("visible_intervals"):
                raise ValueError("Video proposal needs explicit observed and inferred evidence")
            for a,b in p["visible_intervals"]:
                interval(a,b,(p["start"],p["end"]))
    return doc


def union(intervals):
    result = []
    for a,b in sorted(intervals):
        if result and a <= result[-1][1] + 1e-6:
            result[-1][1] = max(b,result[-1][1])
        else:
            result.append([a,b])
    return result


def coverage(state, proposals):
    result = {"audio": [], "video": [], "narrative_support": []}
    for p in proposals:
        if p["character_candidate"] != state["character_candidate"]:
            continue
        modality = p["source_modality"]
        windows = p.get("speech_intervals", []) if modality == "audio" else p.get("visible_intervals", []) if modality == "video" else [[p["start"],p["end"]]]
        key = "narrative_support" if modality == "narrative" else modality
        for a,b in windows:
            a,b = max(a,state["start"]),min(b,state["end"])
            if a < b:
                result[key].append([a,b])
    result = {k: union(v) for k,v in result.items()}
    observed = union(result["audio"]+result["video"])
    gaps, cursor = [],state["start"]
    for a,b in observed:
        if a > cursor:
            gaps.append([cursor,a])
        cursor = max(cursor,b)
    if cursor < state["end"]:
        gaps.append([cursor,state["end"]])
    result.update(no_av_evidence=gaps, basis="Coverage of available sampled/VAD-supported proposals, not continuous verified observation. Narrative support never substitutes for AV coverage.")
    return result


def check_states(doc, proposals, bounds, prefix):
    rows = doc["states"]
    unique(rows, "state_id")
    proposal_map = {p["proposal_id"]:p for p in proposals}
    previous = {}
    for s in sorted(rows, key=lambda r:(r["character_candidate"],r["start"])):
        interval(s["start"],s["end"],bounds)
        for k in ("affect", "expression_style", "transition", "evidence", "uncertainty"):
            if k not in s:
                raise ValueError(f"Missing state field {k}")
        if s["character_candidate"] in previous and s["start"] < previous[s["character_candidate"]]-1e-6:
            raise ValueError("Overlapping states for one character")
        previous[s["character_candidate"]] = s["end"]
        if not s["evidence"] or not set(s["evidence"]) <= proposal_map.keys():
            raise ValueError("State references unknown proposals")
        for pid in s["evidence"]:
            p=proposal_map[pid]
            if p["character_candidate"] != s["character_candidate"] or not overlap((s["start"],s["end"]),(p["start"],p["end"])):
                raise ValueError("State evidence has incompatible character or time")
        if s["transition"] is not None:
            interval(s["transition"]["start"],s["transition"]["end"],bounds)
            if not overlap((s["start"],s["end"]),(s["transition"]["start"],s["transition"]["end"])):
                raise ValueError("Transition does not touch its state")
        if not s["state_id"].startswith(prefix+"_"):
            s["state_id"] = prefix+"_"+s["state_id"]
        s["status"] = "model_proposal"
        s["observation_coverage"] = coverage(s,proposals)
    for conflict in doc.get("conflicts", []):
        if not set(conflict["proposal_ids"]) <= proposal_map.keys():
            raise ValueError("Conflict references unknown proposal")
    return doc


def match_events(states, events):
    links = []
    for state in states:
        for event in events:
            shared = overlap((state["start"],state["end"]),(event["start"],event["end"]))
            if not shared:
                continue
            a,b = state["character_candidate"],event["character_candidate"]
            if a != b and a != "Unknown" and b != "Unknown":
                continue
            links.append(dict(link_id=f"link_{len(links)+1:06d}", state_id=state["state_id"],
                  event_id=event["event_id"], character_candidate=a,
                  character_compatibility="candidate_match_unverified" if a==b and a!="Unknown" else "unresolved_identity",
                  start=max(state["start"],event["start"]), end=min(state["end"],event["end"]),
                  overlap_seconds=shared, event=event, relation="unresolved", semantic_assessment_status="not_run"))
    return links


def apply_relations(candidates, result):
    unique(result["links"],"link_id")
    assessments = {r["link_id"]:r for r in result["links"]}
    if assessments.keys() != {r["link_id"] for r in candidates}:
        raise ValueError("Semantic assessment omits or invents overlap links")
    for link in candidates:
        r=assessments[link["link_id"]]
        if r["relation"] not in RELATIONS or not r.get("rationale") or not isinstance(r.get("uncertainty"),list):
            raise ValueError("Invalid semantic relation")
        link.update({k:r[k] for k in ("relation","rationale","uncertainty")},semantic_assessment_status="model_proposal")
        if link["character_compatibility"] == "unresolved_identity":
            link["relation"] = "unresolved"
            link["uncertainty"].append("Unknown identity prevents assigning the event to this character.")
    return candidates


def check_annotations(doc, states, events, proposals, inp):
    unique(doc["annotations"],"annotation_id")
    state_map={s["state_id"]:s for s in states}
    event_map={e["event_id"]:e for e in events}
    proposal_map={p["proposal_id"]:p for p in proposals}
    modalities={p["source_modality"] for p in proposals}
    allowed = set(state_map)|set(event_map)|{p["proposal_id"] for p in proposals}
    allowed |= set(inp["stage02_context_references"])|{f"srt_{s['srt_index']:06d}" for s in inp["subtitles"]}
    allowed |= {"scene_audio.wav","visual_only.mp4","silero_vad_speech_support"}
    for a in doc["annotations"]:
        interval(a["start"],a["end"],(inp["start"],inp["end"]))
        if a["status"] != "model_proposal" or not a["state_ids"] or not set(a["state_ids"]) <= state_map.keys():
            raise ValueError("Invalid annotation state/status")
        if not set(a["event_ids"]) <= event_map.keys() or not set(a["evidence"]) <= allowed:
            raise ValueError("Unknown annotation evidence")
        for sid in a["state_ids"]:
            s=state_map[sid]
            if s["character_candidate"] != a["character_candidate"] or not overlap((s["start"],s["end"]),(a["start"],a["end"])):
                raise ValueError("Annotation state identity/time mismatch")
        for eid in a["event_ids"]:
            e=event_map[eid]
            if e["character_candidate"] not in (a["character_candidate"],"Unknown") or not overlap((e["start"],e["end"]),(a["start"],a["end"])):
                raise ValueError("Annotation event identity/time mismatch")
        if not a["observation"] or not a["interpretation"] or not a["alternative_interpretation"] or not isinstance(a["uncertainty"],list):
            raise ValueError("Annotation needs observations, interpretation, alternatives and uncertainty")
        for obs in a["observation"]:
            interval(obs["start"],obs["end"],(a["start"],a["end"]))
            if not obs["evidence_ids"] or not set(obs["evidence_ids"]) <= allowed:
                raise ValueError("Observation has unknown evidence IDs")
            modality=obs["source_modality"]
            if modality not in modalities|({"stage05"} if events else set()):
                raise ValueError("Observation uses a modality excluded from this experiment")
            supported=False
            for eid in obs["evidence_ids"]:
                if modality=="stage05" and eid in event_map:
                    p=event_map[eid]
                elif eid in proposal_map and proposal_map[eid]["source_modality"]==modality:
                    p=proposal_map[eid]
                elif modality=="narrative" and (eid.startswith("shot_") or eid.startswith("srt_")):
                    supported=True
                    continue
                else:
                    continue
                if p["character_candidate"]==a["character_candidate"] and overlap((obs["start"],obs["end"]),(p["start"],p["end"])):
                    supported=True
            if not supported:
                raise ValueError("Observation lacks compatible modality/character/time evidence")
    return doc
