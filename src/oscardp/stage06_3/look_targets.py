"""Fuse reviewed visual observations with frozen directional and spatial evidence."""

from collections import Counter,defaultdict
import csv
from pathlib import Path
from shutil import copy2

import numpy as np

from oscardp.shots.transnet import sha256_file
from oscardp.stage06_2.evidence_replay import write_json
from oscardp.stage06_2.inputs import read
from .inference import CHARACTERS, TARGETS, camera_hypotheses, timeline_transitions
from .look_pilot import MOVIE, SEQUENCE, VERSION
from .look_review import SOURCE, track_review
from .look_states import overlaps
from .orientation import evaluate, orientations


def verify_visual(folder, states, qc, observations):
    manifest = read(folder/"visual_packet.json")
    frames = manifest["states"]
    observations = observations["observations"]
    expected={s["state_id"] for s in states}
    if frames.keys()!=expected or observations.keys()!=expected:
        raise ValueError("All Look States need a reviewed original-frame entry")
    clocks={(r["track_id"],r["global_frame_index"]):r["movie_timestamp_sec"] for rr in qc.values() for r in rr}
    all_frames={f["global_frame_index"]:f for ff in frames.values() for f in ff}
    for s in states:
        key=s["state_id"]
        state_frames=frames[key]
        representative=state_frames[0]
        if not representative["within_state"] or representative["role"] not in {"stable","qc_sample"}:
            raise ValueError("No in-state representative frame")
        for f in state_frames:
            n=f["global_frame_index"]
            inside=s["start_frame"]<=n<s["end_frame"] and s["start_s"]<=f["movie_timestamp_sec"]<s["end_s"]
            if ((f["shot_id"],f["track_id"],f["state_id"])!=(s["shot_id"],s["track_id"],key)
                    or f["within_state"]!=inside or (not inside and f["role"]!="pre_motion")
                    or f["source"]!=SOURCE or clocks[s["track_id"],n]!=f["movie_timestamp_sec"]
                    or f["file"]!=f"frames/frame_{n}.jpg" or sha256_file(folder/f["file"])!=f["sha256"]):
                raise ValueError(f"Frame identity, PTS or hash invalid: {key} {n}")
        reviewed=observations[key]
        if (reviewed["local_frame_numbers"]!=list(dict.fromkeys(f["global_frame_index"] for f in state_frames))
                or not set(reviewed["cross_shot_reference_frames"])<=all_frames.keys()
                or any(all_frames[n]["shot_id"]==s["shot_id"] for n in reviewed["cross_shot_reference_frames"])
                or reviewed["human_verified"] is not False):
            raise ValueError(f"Inconsistent actual frame observations: {key}")
        for v in reviewed["target_observations"]:
            if v["target"] not in TARGETS or v["target"] in {s["character"],"UNKNOWN"}:
                raise ValueError(f"Invalid visual target label: {key}")
            if v["target"] in {"AWAY","OTHER"} and not (
                    v["support"]=="fits_observed_landing" and v.get("independent_endpoint_evidence")):
                raise ValueError(f"No independent nonperson evidence: {key}")
    return manifest,all_frames


def geometry_for(s, camera, layout, hypos, orientation, baseline):
    prior=defaultdict(list)
    for r in baseline:
        for p,why in r["evidence"]["context_priors"].items():
            prior[p].extend(v for v in why if v not in prior[p])
    direction=s["look_direction"]
    if not direction:
        return {"status":"no_reliable_look_direction","geometry":None,"target_candidates":[],"selected_target":"UNKNOWN",
                "reasons":["no_sustained_camera_or_head_direction"],"context_priors":dict(prior)}
    mode="head_only" if s["direction_source"]=="head_only" else "gaze"
    record={**s,"interval_id":s["state_id"],"direction_source":mode,"selected_target":"UNKNOWN",
            "target_candidates":[{"target":p,"support":"context_prior"} for p in prior],
            "evidence_level":"context_only","uncertainty_reason":[],
            "evidence":{"gaze_direction":direction["vector"] if mode=="gaze" else None,
                        "head_facing_direction":direction["vector"] if mode=="head_only" else None,
                        "context_priors":dict(prior)}}
    result=evaluate(record,camera,layout,hypos,orientation,[])
    return {"status":orientation["status"],"geometry":result["evidence"]["scene_geometry"],
            "orientation_constraint_ids":result["evidence"]["applied_orientation_ids"],
            "target_candidates":[c for c in result["target_candidates"] if c["target"]!="UNKNOWN"],
            "selected_target":result["selected_target"],"reasons":result["uncertainty_reason"],
            "context_priors":dict(prior)}


def fuse(s, spatial, observation):
    geometric=spatial["geometry"] is not None
    visual={a["target"]:a for a in observation["target_observations"]}
    if s["direction_source"]=="head_only":
        visual={p:v for p,v in visual.items() if p in CHARACTERS}
    options={}
    for r in spatial["target_candidates"]:
        options.setdefault(r["target"],[]).append(r["support"])
    for name,v in visual.items():
        support=v["support"] if v["support"]=="visually_preferred" else "visual_"+v["support"]
        options.setdefault(name,[]).append(support)
    ordered=sorted(options,key=lambda p:(-int(visual.get(p,{}).get("support") in
        {"visually_preferred","fits_observed_landing"}),-len(options[p]),p))
    unresolved=list(spatial["reasons"])
    conflicting=[f["target"] for f in (spatial["geometry"] or {}).get("tested_targets",[])
                 if f["target"] in visual and not any(v["compatible"] for v in f["camera_branches"].values())]
    if conflicting:
        unresolved.append("visual_candidate_not_compatible_with_schematic_geometry")
    selected="UNKNOWN"
    prior=spatial["selected_target"]
    gaze=s["direction_source"] in {"eye_head","gaze_only"}
    if (gaze and s["qc_status"]=="stable" and prior in visual and geometric and not conflicting
            and visual[prior]["support"]=="fits_observed_landing" and len(ordered)==1):
        selected=prior
    # An observed nonperson endpoint still needs independent usable camera gaze,
    # and cannot overrule a competing person without review.
    if (gaze and s["qc_status"]=="stable" and len(ordered)==1 and ordered[0] in {"AWAY","OTHER"}
            and visual[ordered[0]].get("independent_endpoint_evidence")):
        selected=ordered[0]
    if selected=="UNKNOWN":
        if len([p for p in ordered if p in CHARACTERS])>1:
            unresolved.append("multiple_person_candidates")
        unresolved.append("no_independent_convergence_on_one_target")
    if s["qc_status"]=="heterogeneous_stable_support":
        unresolved.append("more_than_one_direction_within_the_event_driven_state")
    if geometric and gaze:
        level="gaze_supported"
    elif geometric and s["direction_source"]=="head_only":
        level="head_supported"
    elif visual:
        level="context_only"
    else:
        level="insufficient"
    final=[{"target":p,"support":options[p]} for p in ordered]
    if selected=="UNKNOWN":
        final.append({"target":"UNKNOWN","support":["unresolved"]})
    return {**s,"selected_target":selected,"possible_targets":final,"evidence_level":level,
            "visual_evidence":observation,"spatial_evidence":spatial,
            "uncertainty_reason":list(dict.fromkeys(unresolved)),"human_verified":False}


def event_targets(events, candidates):
    by_id={s["state_id"]:s for s in candidates}
    transitions=[]
    for e in events:
        pre=by_id.get(e["pre_state_id"])
        post=by_id.get(e["post_state_id"])
        if not (e["split_supported_states"] or e["assessment"] in {"return","adjustment_or_same_direction"}):
            continue
        a=pre["selected_target"] if pre else "UNKNOWN"
        b=post["selected_target"] if post else "UNKNOWN"
        if a==b!="UNKNOWN":
            kind="target_hold_candidate"
        elif (a!=b and a!="UNKNOWN" and b!="UNKNOWN" and e["split_supported_states"]
              and e["assessment"]=="sustained_gaze_change"):
            kind="target_switch_candidate"
        else:
            if not e["split_supported_states"]:
                continue
            kind="target_uncertain"
        transitions.append({"basis_event_id":e["event_id"],"shot_id":e["shot_id"],"track_id":e["track_id"],
                            "type":kind,"start_s":e["start_s"],"end_s":e["end_s"],
                            "pre_state_id":e["pre_state_id"],"post_state_id":e["post_state_id"],
                            "before_target":a,"after_target":b,"human_verified":False})
    return transitions


def cross_cut_targets(candidates, mapping, previous):
    """Match only independently supported boundary states across a cut."""
    bounds={r["track_id"]:r for r in previous["intervals"]}
    identities={r["track_id"]:r["mapping_strength"] for r in mapping["tracks"]}
    groups=defaultdict(list)
    for s in candidates:
        b=bounds[s["track_id"]]
        groups[s["character"]].append({**s,**{k:b[k] for k in ("shot_start_s","shot_end_s","layout_id","shot_index")},
              "interval_id":s["state_id"],"mapping_strength":identities[s["track_id"]],
              "evidence":{"plan_usable":s["spatial_evidence"]["geometry"] is not None and
                          s["qc_status"]=="stable"}})
    return [t for actor,items in groups.items() if actor!="UNKNOWN"
            for t in timeline_transitions(sorted(items,key=lambda r:r["start_s"])) if t["cross_shot"]]


def write_comparison(output, previous, states, events, transitions, frame_count):
    old=previous["intervals"]
    transition_counts=Counter(t["type"] for t in transitions)
    statuses=Counter(r["state_origin"] for r in states)
    written={}
    for s in states:
        labels={t["target"] for t in s["possible_targets"]}
        for p in old:
            if p["track_id"]!=s["track_id"] or not overlaps(s,p) or p["evidence"]["measured_frames"]<=0:
                continue
            seconds=min(s["end_s"],p["end_s"])-max(s["start_s"],p["start_s"])
            old_labels={t["target"] for t in p["target_candidates"]}
            k=s["shot_id"]
            d=written.setdefault(k,Counter())
            d["total"]+=seconds
            d["old_pair"]+=seconds*({"TEDDY","DON"}<=old_labels)
            d["new_pair"]+=seconds*({"TEDDY","DON"}<=labels)
            d["sets_differ"]+=seconds*(old_labels!=labels)
            d["new_unique"]+=seconds*(s["selected_target"]!="UNKNOWN")
            d["old_unique"]+=seconds*(p["selected_target"]!="UNKNOWN")
    with (output/"comparison_v1_1_v1_2.md").open("x",encoding="utf-8") as f:
        f.write("# v1.1 / v1.2 同 Track 可比时间\n\n"
                "按同一 Shot、同一 Track 的旧区间与新状态的 PTS 交集累加。多人同框的各 Track 分别计时。"
                "时间是覆盖量，仍需人工标签评估准确性。\n\n"
                "| Shot | 交集（秒） | Teddy / Don 同列 v1.1 → v1.2（秒） | 候选集合改变（秒） | 有主选 v1.1 → v1.2（秒） |\n"
                "| --- | ---: | ---: | ---: | ---: |\n")
        for shot,d in written.items():
            f.write(f"| {shot} | {d['total']:.3f} | {d['old_pair']:.3f} → {d['new_pair']:.3f} | "
                    f"{d['sets_differ']:.3f} | {d['old_unique']:.3f} → {d['new_unique']:.3f} |\n")
        f.write("\n- 212 的单一人物旧主选来自 v1.0；v1.1 已撤回。实际检查了五个旧主选中点，"
                "以及本次两个方向事件的前后帧。前景 Michelle 是优先复核候选；画外 Don 和低处手势区域仍有歧义。\n"
                "- 217 的重点区间与 219、221 的弱眼动前后原片均已检查，"
                "可见方向变化不足以判定新的对象。近景场景方位仍无法区分 Teddy 与 Don。\n"
                "- 222 的 Michelle 侧脸保留粗头向，宽景证明 Teddy 和 Don 在同侧。"
                "Teddy 003 的宽景朝向与 Michelle 更相容，但瞳孔落点不可读。\n")
    with (output/"review.md").open("x",encoding="utf-8") as f:
        f.write("# Bugonia seq_006 · Event-driven Look States\n\n"
                f"13 个 Shot、21 条实测 Track，生成 {len(states)} 个 Look State。"
                f"状态来源 {dict(statuses)}。方向来源 {dict(Counter(r['direction_source'] for r in states))}。\n\n"
                f"已逐状态查看 {frame_count} 张不同的正片帧。预选近景和前后帧由 Codex 交互视觉检查，"
                "画面观察、空间相容性和最终候选分别保存。身份和目标均为候选。\n\n"
                f"有主选 {sum(r['selected_target']!='UNKNOWN' for r in states)} 段，"
                f"UNKNOWN {sum(r['selected_target']=='UNKNOWN' for r in states)} 段。"
                f"Hold {transition_counts['target_hold_candidate']}，Switch {transition_counts['target_switch_candidate']}。\n\n"
                "| Shot | 状态数 | 已知方向 | 目标主选 |\n| --- | ---: | ---: | ---: |\n")
        for shot in sorted({r["shot_id"] for r in states}):
            items=[r for r in states if r["shot_id"]==shot]
            f.write(f"| {shot} | {len(items)} | {sum(r['look_direction'] is not None for r in items)} "
                    f"| {sum(r['selected_target']!='UNKNOWN' for r in items)} |\n")
        f.write("\n## 已支持的方向状态边界\n\n")
        for e in events:
            if e["split_supported_states"]:
                f.write(f"- {e['shot_id']}，{e['start_s']:.3f}–{e['end_s']:.3f} s，"
                        f"{e['assessment']}，稳定前后方向差 {e['direction_change_deg']:.2f}°。"
                        "候选落点的判定另见时间线。\n")
        f.write("\n逐 Track 卡片位于 review/，review_items.json 将无法区分的目标、身份碎片和"
                "视觉与空间不相容的样本送入人工队列。human_review.csv 中标签为空。\n")


def publish(args, source, mapping, orientation, previous, states, events, qc, provenance):
    observation_document=read(args.observations)
    folder=args.observations.parent
    manifest,all_frames=verify_visual(folder,states,qc,observation_document)
    observations=observation_document["observations"]
    scene=read(args.layout)
    if (scene["movie_id"],scene["sequence_id"])!=(MOVIE,SEQUENCE):
        raise ValueError("Layout movie / sequence mismatch")
    if sha256_file(args.layout)!=orientation["layout_sha256"]:
        raise ValueError("Layout changed since v1.1 camera review")
    inherited={r["shot_id"]:r for r in orientation["shots"]}
    layouts={l["layout_id"]:l for l in scene["layouts"]}
    spatial_inputs={}
    for camera in scene["shots"]:
        sid=camera["shot_id"]
        layout=layouts[camera["layout_id"]]
        if sid in inherited:
            old=inherited[sid]
            _,hypos,relation=orientations(camera,layout,{**old,"objects":old["visible_nonperson_objects"]})
        else:
            pos={p["person"]:np.asarray(p["position"]) for p in layout["people"] if p["position"] is not None}
            hypos=camera_hypotheses(camera,pos)
            relation={"status":"inherited_only" if hypos else "relative_only","applied_constraint_ids":[]}
        spatial_inputs[sid]=(camera,layout,hypos,relation)
    prior=defaultdict(list)
    for r in previous["intervals"]:
        prior[r["track_id"]].append(r)
    candidates=[]
    for s in states:
        observation=observations[s["state_id"]]
        def ref(f):
            keys=("shot_id","track_id","global_frame_index","movie_timestamp_sec","role","within_state","source","sha256")
            return {**{k:f[k] for k in keys},"file":"visual_evidence/"+f["file"]}
        reviewed={**observation,"frames":[ref(f) for f in manifest["states"][s["state_id"]]],
                  "cross_shot_frames":[ref(all_frames[n]) for n in observation["cross_shot_reference_frames"]],
                  "review_status":"reviewed_model_observation"}
        b=[r for r in prior[s["track_id"]] if overlaps(s,r)]
        candidates.append(fuse(s,geometry_for(s,*spatial_inputs[s["shot_id"]],b),reviewed))
    transitions=event_targets(events,candidates)
    cross_transitions=cross_cut_targets(candidates,mapping,previous)
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    asset=out/"visual_evidence"
    (asset/"frames").mkdir(parents=True)
    for f in all_frames.values():
        copy2(folder/f["file"],asset/f["file"])
    copy2(args.observations,asset/"observations.json")
    copy2(folder/"visual_packet.json",asset/"visual_packet.json")
    common={"schema_version":VERSION,"movie_id":MOVIE,"sequence_id":SEQUENCE,"human_verified":False}
    provenance.update({str(args.layout):sha256_file(args.layout),
                       "visual_evidence/observations.json":sha256_file(args.observations),
                       "visual_evidence/visual_packet.json":sha256_file(folder/"visual_packet.json")})
    track_review(out,candidates,manifest["states"],events,folder)
    write_json(out/"look_states.json",{**common,"states":states,"input_sha256":provenance})
    write_json(out/"look_direction_events.json",{**common,"events":events})
    write_json(out/"character_track_mapping.json",{**mapping,"v1_2_mapping_source":str(args.baseline_v1/"character_track_mapping.json"),
                                                 "visual_review":"visual_evidence/observations.json"})
    write_json(out/"gaze_target_candidates.json",{**common,"look_states":candidates,
                                                "actual_visual_assessor":observation_document["method"]})
    shots=[]
    for shot in source["shots"]:
        sid=shot["shot_id"]
        entries=[]
        for t in [r for r in mapping["tracks"] if r["shot_id"]==sid]:
            tid=t["track_id"]
            items=[r for r in candidates if r["track_id"]==tid]
            entries.append({"track_id":tid,"character":t["character"],"coverage":t["coverage"],
                            "measured":t["measured"],"look_states":items,
                            "target_transitions":[r for r in transitions if r["track_id"]==tid]})
        present={r["character"] for r in entries if r["measured"]}
        visible={name for r in candidates if r["shot_id"]==sid for name in r["visual_evidence"]["visible_characters"]}
        shots.append({"shot_id":sid,"tracks":entries,
                      "characters_without_measured_track":[{"character":p,"coverage":"visible_without_measured_track"
                           if p in visible else "not_tracked_or_offscreen"} for p in CHARACTERS if p not in present]})
    write_json(out/"shot_gaze_target_timeline.json",{**common,"shots":shots,
              "cross_shot_target_transitions":cross_transitions,
              "cross_shot_status":"candidate_only" if cross_transitions else
                                  "no_sufficiently_supported_target_continuity_across_camera_cut"})
    queue=[r for r in candidates if r["selected_target"]=="UNKNOWN" or r["direction_source"]=="head_only"]
    write_json(out/"review_items.json",{**common,"review_items":queue,
                                       "identity_review_queue":[t for t in mapping["tracks"] if t["mapping_strength"]!="supported"]})
    with (out/"human_review.csv").open("x",newline="",encoding="utf-8-sig") as stream:
        writer=csv.writer(stream)
        keys=("state_id","shot_id","track_id","character","start_s","end_s","selected_target")
        writer.writerow([*keys,"human_target","review_status","notes"])
        writer.writerows([*[r[k] for k in keys],"","pending",""] for r in candidates)
    write_comparison(out,previous,candidates,events,transitions,len(all_frames))
    print(len(shots),"shots,",len(states),"states,",len(all_frames),"actual reviewed frames,",
          sum(r["selected_target"]!="UNKNOWN" for r in candidates),"selected targets")
