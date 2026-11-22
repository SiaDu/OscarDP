"""Run the Bugonia seq_006 cross-shot gaze target pilot."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from shutil import copy2

import pyarrow.parquet as pq

from oscardp.shots.transnet import sha256_file
from oscardp.stage06_2.evidence_replay import write_json
from oscardp.stage06_2.inputs import EYES, read
from oscardp.stage06_2.stable import interval
from .inference import CHARACTERS, TARGETS, direction_windows, gaze_states, match, timeline_transitions
from .report import publish_review

MOVIE = "tt12300742"
SEQUENCE = "seq_006"
VERSION = "stage06.3.gaze_target.v1.0"
CLOCK = ("global_frame_index","movie_timestamp_sec","source_next_timestamp_sec")


def load_visual(path, output):
    packet = read(path)
    if packet["movie_id"] != MOVIE:
        raise ValueError("Visual evidence belongs to a different movie")
    records = {}
    extras = defaultdict(list)
    destination = output/"visual_evidence"
    destination.mkdir()
    for review in [*packet["frames"],*packet.get("focus_frames",[])]:
        focus = review in packet.get("focus_frames",[])
        if (any(Path(review[k]).name!=review[k] for k in ("file","review_sheet"))
                or (not focus and review["track_id"] in records)):
            raise ValueError("Invalid visual evidence path or duplicate Track")
        source = path.parent/review["file"]
        if sha256_file(source)!=review["sha256"]:
            raise ValueError(f"Reviewed frame hash changed: {source}")
        if not (destination/source.name).exists():
            copy2(source,destination)
        sheet = path.parent/review["review_sheet"]
        if not (destination/sheet.name).exists():
            copy2(sheet,destination)
        record = {**review,"file":str(Path("visual_evidence")/source.name),
                  "review_sheet":str(Path("visual_evidence")/sheet.name)}
        if focus:
            extras[review["track_id"]].append(record)
        else:
            records[review["track_id"]] = record
    copy2(path,destination/"reviewed_frames.json")
    return packet, records, extras


def map_track(shot, track, review, stage03):
    sid,tid = shot["shot_id"],track["track_id"]
    visual_names = [v["character_name"] for v in shot["visible_character_candidates"]
                    if v["character_name"] in CHARACTERS]
    upstream_name = track.get("person_candidate")
    candidate = review["character_candidate"] if review else upstream_name
    strength = review["review_status"] if review else "upstream_only"
    if review and (review["shot_id"],review["track_id"])!=(sid,tid):
        raise ValueError("Visual record is indexed by the wrong Shot/Track")
    person = candidate if candidate in CHARACTERS and strength!="boundary_residue" else "UNKNOWN"
    possible = [candidate] if candidate in CHARACTERS else sorted(set(visual_names))
    return {"shot_id":sid,"track_id":tid,"character":person,"character_candidates":possible,
            "mapping_strength":strength,"status":"candidate" if person!="UNKNOWN" else "unresolved",
            "human_verified":False,"coverage":track["status"],"measured":"stats" in track,
            "sources":{"visual":review,"stage03":{"sample_roles":[x["role"] for x in stage03["samples"]],
                                                 "visible_subjects":len(stage03["subjects"])},
                       "stage04":shot["visible_character_candidates"],
                       "stage05_identity":{"namespace":track.get("identity_namespace") or shot["stage05"]["identity"],
                                           "cluster_id":track.get("identity_cluster_id"),
                                           "status":track.get("identity_status"),"prior_name":upstream_name}}}


def link_identity_evidence(mapping):
    groups = defaultdict(list)
    for record in mapping:
        identity = record["sources"]["stage05_identity"]
        if identity["status"]=="tentative_identity_cluster" and identity["cluster_id"]:
            groups[(str(Path(identity["namespace"]).resolve()),identity["cluster_id"])].append(record)
    for records in groups.values():
        for record in records:
            same_character = [r["track_id"] for r in records if r is not record and
                              r["character"]==record["character"]!="UNKNOWN"]
            record["sources"]["stage05_identity"]["consistent_same_batch_tracks"] = same_character


def load_tracks(source, qc_root, shot, track, qc_rows, provenance):
    sid,tid = shot["shot_id"],track["track_id"]
    root = source/sid/tid
    saved_motion = qc_root/sid/tid/"motion_events.json"
    if not saved_motion.is_file():
        saved_motion = root/"motion_events.json"
    frame_path = root/"eye_position_frames.parquet"
    paths = (frame_path,root/"evidence_frames.parquet",root/"gaze_states.json",saved_motion,
             root/"selected_gaze_frames.json")
    raw = pq.read_table(frame_path,columns=[*CLOCK,"parquet_row","shot_id","track_id",*sum(EYES,[])]).to_pylist()
    previous = pq.read_table(paths[1],columns=[*CLOCK,"parquet_row","gaze_direction_status"]).to_pylist()
    rows = []
    for a,b,c in zip(raw,previous,qc_rows,strict=True):
        keys = (*CLOCK,"parquet_row")
        if any(a[k]!=b[k] or a[k]!=c[k] for k in keys) or (
                a["shot_id"],c["shot_id"],a["track_id"],c["track_id"])!=(sid,sid,tid,tid):
            raise ValueError(f"Stage06.2 and Eye Observability QC do not align: {tid}")
        if bool(c["gaze_direction"])!=c["gaze_direction_status"]["usable"]:
            raise ValueError(f"Malformed gaze mask: {tid}")
        rows.append({**a,**c})
    if len(rows)!=track["stats"]["frames"]:
        raise ValueError(f"Missing frame evidence: {tid}")
    states,motion,picks = (read(path) for path in paths[2:])
    if any((d["shot_id"],d["track_id"])!=(sid,tid) for d in (states,motion,picks)):
        raise ValueError(f"Stage06.2 state provenance mismatch: {tid}")
    prior_gaze = [s for s in states["states"] if s["type"]=="stable_gaze_candidate"]
    changed = sum(a["gaze_direction_status"]["usable"]!=b["gaze_direction_status"]["usable"]
                  for a,b in zip(previous,rows,strict=True))
    # Keep the archived state IDs and bounds where the mask was identical.
    new_gaze = gaze_states(rows) if changed else prior_gaze
    for path in paths:
        provenance[str(path)] = sha256_file(path)
    return rows, new_gaze, states["states"], motion["events"], picks["frames"], {
        "prior_valid_gaze_frames":sum(r["gaze_direction_status"]["usable"] for r in previous),
        "current_valid_gaze_frames":sum(r["gaze_direction_status"]["usable"] for r in rows),
        "changed_gaze_mask_frames":changed,"stable_gaze_before":len(prior_gaze),"stable_gaze_used":len(new_gaze),
        "gaze_states_origin":"in_memory_stage06_2_same_thresholds" if changed else "saved_stage06_2",
        "motion_source":str(saved_motion)}


def dialogue_context(actor, i, shots, dialogue, layout):
    sid = shots[i]["shot_id"]
    facing = {p["person"]:p["facing_to"] for p in layout["people"]}
    priors = {}
    for person in CHARACTERS:
        if person==actor:
            continue
        reasons = ["scene_participant"]
        if facing.get(actor)==person:
            reasons.append("rough_body_facing")
        if person in dialogue.get(sid,[]):
            reasons.append("stage02_speaker_in_shot")
        if any(v["camera_subject"]==person for v in shots[max(0,i-1):i+2] if v["shot_id"]!=sid):
            reasons.append("adjacent_camera_subject")
        priors[person] = reasons
    return dict(sorted(priors.items(),key=lambda item:(-len(item[1]),item[0])))


def shot_bounds(shot, track):
    sid,tid = shot["shot_id"],track["track_id"]
    observed = next(r for r in shot["reported_faces"] if r["track_id"]==tid)
    if shot["boundary_status"]=="experimental_model_reviewed_pending_human":
        return observed["effective_processing_temporal"]
    return shot["stage01_temporal"]


def representatives(span, picks, visual, focus_frames):
    around = sorted((r for r in picks if span["start_s"]<=r["movie_timestamp_sec"]<span["end_s"]),
                    key=lambda r:abs(r["movie_timestamp_sec"]-(span["start_s"]+span["end_s"])/2))
    result = [{**r,"source":"stage06_2_selected_gaze_frames"} for r in around[:2]]
    if visual:
        result.append({k:visual[k] for k in ("global_frame_index","movie_timestamp_sec","file","bbox_xywh")}
                      | {"source":"inspected_original_movie_frame",
                         "within_interval":span["start_s"]<=visual["movie_timestamp_sec"]<span["end_s"]})
    for f in focus_frames:
        if span["start_s"]<=f["movie_timestamp_sec"]<span["end_s"]:
            result.append({k:f[k] for k in ("global_frame_index","movie_timestamp_sec","file","review_sheet")}
                          | {"source":"inspected_original_movie_frame","within_interval":True})
    return result


def analyze_intervals(*, track, mapping, shot, shot_index, camera, layout, prior, rows, gaze,
                      states, motion, picks, source_stats, focus_frames):
    bounds = shot_bounds(shot,track)
    source_face = next(t for t in shot["reported_faces"] if t["track_id"]==track["track_id"])
    actor = mapping["character"]
    windows = [(mode,interval(rows,indices),ray,[rows[i] for i in indices])
               for mode,indices,ray in direction_windows(rows,gaze)] if rows else []
    if not rows:
        windows.append(("unresolved",{"start_frame":source_face["start_frame"],"end_frame":source_face["end_frame"],
                                      "start_s":source_face["start_sec"],"end_s":source_face["end_sec"]},None,[]))

    result = []
    for j,(mode,span,ray,frames) in enumerate(sorted(windows,key=lambda r:r[1]["start_s"]),1):
        candidates,selected,level,reasons,geometry = match(actor,mode,ray,camera,layout,prior)
        if not frames:
            reasons = [("boundary_residue" if mapping["mapping_strength"]=="boundary_residue" else
                        "track_measurements_unavailable")]
        if mode=="unresolved" and any(r["gaze_direction"] is not None for r in frames):
            reasons.append("valid_gaze_without_sustained_stable_window")
        def overlapping(items):
            return [r for r in items if r["start_s"]<span["end_s"] and span["start_s"]<r["end_s"]]
        eyes = overlapping([s for s in states if s["type"]=="eye_position_stable"])
        eye_events = overlapping([e for e in motion if e["type"] in {
            "weak_eye_motion_candidate","eye_shift_candidate","eye_adjustment_candidate","blink_candidate"}])
        result.append({**span,"interval_id":f"{track['track_id']}_interval_{j:03d}","shot_id":shot["shot_id"],
                       "shot_index":shot_index,"track_id":track["track_id"],"character":actor,
                       "mapping_strength":mapping["mapping_strength"],"layout_id":layout["layout_id"],
                       "shot_start_s":bounds["start"],"shot_end_s":bounds["end"],
                       "boundary_status":shot["boundary_status"],"direction_source":("head_only" if mode=="head" else mode),
                       "target_candidates":candidates,"selected_target":selected,"evidence_level":level,
                       "uncertainty_reason":reasons,"human_verified":False,
                       "representative_frames":representatives(span,picks,mapping["sources"]["visual"],focus_frames),
                       "evidence":{"gaze_direction":ray if mode=="gaze" else None,
                                   "head_facing_direction":ray if mode=="head" else None,
                                   "camera_group":camera["camera_group"],
                                   "camera_group_role":"appearance_continuity_only" if camera["camera_group"] else None,
                                   "axis_side":camera["axis_side"],"axis_reference_shot":camera["axis_reference_shot"],
                                   "layout_basis":camera["layout_basis"],"scene_geometry":geometry,
                                   "plan_usable":geometry is not None,"context_priors":prior,
                                   "measured_frames":len(frames),
                                   "eye_position_usable_frames":sum(any(v["usable"] for v in r["eye_position_status"].values()) for r in frames),
                                   "gaze_usable_frames":sum(r["gaze_direction"] is not None for r in frames),
                                   "head_pose_usable_frames":sum(r["head_facing_direction"] is not None for r in frames),
                                   "stable_eye_positions":eyes,"eye_motion_observations":eye_events,
                                   "direction_status_source":source_stats.get("gaze_states_origin") if mode=="gaze" else
                                                             "eye_observability_qc" if mode=="head" else None}})
    return result


def coverage_gaps(records, start, end):
    """Account for time with no measured Track for this character."""
    gaps = []
    for r in records:
        if not r["evidence"]["measured_frames"]:
            continue
        a,b = max(start,r["start_s"]),min(end,r["end_s"])
        if b<=a:
            continue
        if a>start+1e-6:
            gaps.append({"start_s":start,"end_s":a,"type":"target_uncertain","selected_target":"UNKNOWN",
                         "reason":"no_measured_character_track"})
        start = max(start,b)
    if end>start+1e-6:
        gaps.append({"start_s":start,"end_s":end,"type":"target_uncertain","selected_target":"UNKNOWN",
                     "reason":"no_measured_character_track"})
    return gaps


def run(args):
    output = args.output_dir
    protected = [args.full_sequence,args.eye_qc,args.movie_dir,args.layout.parent]
    if any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve())
           for p in protected):
        raise ValueError("Output must be independent of previous stage results")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    layout_packet = read(args.layout)
    timeline_path = args.full_sequence/"sequence_gaze_timeline.json"
    source = read(timeline_path)
    qc_path = args.eye_qc/"eye_observability_frames.parquet"
    qc_by_track = defaultdict(list)
    for row in pq.read_table(qc_path).to_pylist():
        qc_by_track[row["track_id"]].append(row)
    documents = {key:args.movie_dir/f"Stage{key}_{MOVIE}.json" for key in ("02","03","04","04_3")}
    dialogue,visual03,context04,seq = (read(documents[k]) for k in ("02","03","04","04_3"))
    shots = source["shots"]
    cameras = layout_packet["shots"]
    expected = next(s["shot_ids"] for s in seq["sequence_candidates"] if s["sequence_id"]==SEQUENCE)
    if (source["movie_id"]!=MOVIE or layout_packet["movie_id"]!=MOVIE or source["sequence_id"]!=SEQUENCE
            or layout_packet["sequence_id"]!=SEQUENCE
            or [s["shot_id"] for s in shots]!=expected or [c["shot_id"] for c in cameras]!=expected
            or len(expected)!=13):
        raise ValueError("Expected the same 13 shots in all Bugonia seq_006 inputs")
    if set(qc_by_track)!={t["track_id"] for s in shots for t in s["tracks"] if "stats" in t}:
        raise ValueError("Latest Eye Observability QC does not cover exactly the measured Tracks")
    provenance_paths = [args.layout,timeline_path,qc_path,args.visual_evidence,*documents.values()]
    provenance = {str(p):sha256_file(p) for p in provenance_paths}
    layout_by_id = {l["layout_id"]:l for l in layout_packet["layouts"]}
    stage02 = {s["shot_id"]:list({d["speaker"] for d in s["dialogue"] if d["speaker"] in CHARACTERS})
               for s in dialogue["shots"]}
    stage03 = {s["shot_id"]:s for s in visual03["shots"]}
    stage04 = {s["shot_id"]:s["character_candidates"] for s in context04["shots"]}
    if any(shot["visible_character_candidates"]!=stage04[shot["shot_id"]] for shot in shots):
        raise ValueError("Stage04 candidate context changed since Stage06.2")
    output.mkdir(parents=True,exist_ok=False)
    visual_packet,visual,focus_frames = load_visual(args.visual_evidence,output)
    mapping, all_intervals, shot_stats = [],[],[]
    for i,(shot,camera) in enumerate(zip(shots,cameras,strict=True)):
        layout = layout_by_id[camera["layout_id"]]
        sid = shot["shot_id"]
        diagnostic = {"shot_id":sid,"upstream_status":shot["status"],"available_tracks":0,
                      "camera_group":camera["camera_group"],"camera_point":camera["camera_position"],
                      "camera_region":camera["camera_region"],"tracks":{}}
        for track in shot["tracks"]:
            tid = track["track_id"]
            number = len(all_intervals)
            rec = map_track(shot,track,visual.get(tid),stage03[sid])
            mapping.append(rec)
            if rec["measured"]:
                rows,gaze,states,motion,picks,stats = load_tracks(args.full_sequence,args.eye_qc,shot,track,qc_by_track[tid],provenance)
                diagnostic["available_tracks"] += 1
            else:
                rows,gaze,states,motion,picks,stats = [],[],[],[],[],{}
            priors = dialogue_context(rec["character"],i,cameras,stage02,layout)
            all_intervals.extend(analyze_intervals(track=track,mapping=rec,shot=shot,shot_index=i,camera=camera,
                                                   layout=layout,prior=priors,rows=rows,gaze=gaze,states=states,
                                                   motion=motion,picks=picks,source_stats=stats,focus_frames=focus_frames[tid]))
            diagnostic["tracks"][tid] = {**stats,"interval_count":len(all_intervals)-number}
        shot_stats.append(diagnostic)
    all_intervals.sort(key=lambda r:(r["start_s"],r["shot_index"],r["track_id"]))
    link_identity_evidence(mapping)
    common = {"schema_version":VERSION,"movie_id":MOVIE,"sequence_id":SEQUENCE,"human_verified":False}
    write_json(output/"character_track_mapping.json",{**common,"visual_method":visual_packet["review_method"],
                                                      "review_packet":"visual_evidence/reviewed_frames.json",
                                                      "video":visual_packet["video"],"tracks":mapping})
    write_json(output/"gaze_target_candidates.json",{**common,"targets":list(TARGETS),"shots":shot_stats,
                                                     "intervals":all_intervals,"input_sha256":provenance})
    characters = {}
    for actor in (*CHARACTERS,"UNKNOWN"):
        subset = [r for r in all_intervals if r["character"]==actor]
        transitions = timeline_transitions(subset) if actor!="UNKNOWN" else []
        characters[actor] = {"intervals":[{**{k:r[k] for k in (
            "interval_id","shot_id","track_id","start_s","end_s","selected_target","evidence_level","direction_source")},
            "type":"stable_target_candidate" if r["selected_target"]!="UNKNOWN" else "target_uncertain",
            "target_candidates":[c["target"] for c in r["target_candidates"]]} for r in subset],
            "transitions":transitions,
            "coverage_gaps":coverage_gaps(subset,source["sequence_start_s"],source["sequence_end_s"])
                            if actor!="UNKNOWN" else []}
    write_json(output/"gaze_target_timeline.json",{**common,"characters":characters,
        "timeline_policy":"Shot-local measurements, no inheritance across missing or unresolved intervals",
        "sequence_start_s":source["sequence_start_s"],"sequence_end_s":source["sequence_end_s"]})
    publish_review(output,common,source,shot_stats,mapping,all_intervals,characters)
    print(len(shots),"shots,",len(mapping),"tracks,",len(all_intervals),"intervals",
          dict(Counter(r["evidence_level"] for r in all_intervals)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("layout","full-sequence","eye-qc","movie-dir","visual-evidence","output-dir"):
        parser.add_argument("--"+option,type=Path,required=True)
    run(parser.parse_args())


if __name__=="__main__":
    main()
