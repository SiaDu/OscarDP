"""Replay the immutable Stage06.3 v1.0 intervals with relative camera constraints."""

import argparse
from collections import Counter
import csv
from pathlib import Path
from shutil import copy2

import pyarrow.parquet as pq

from oscardp.shots.transnet import sha256_file
from oscardp.stage06_2.evidence_replay import write_json
from oscardp.stage06_2.inputs import read
from .aligned_review import choose_intervals, publish_aligned
from .inference import CHARACTERS, timeline_transitions
from .orientation import FOCUS_SHOTS, evaluate, orientations
from .report import figure


def labels(record):
    return {c["target"] for c in record["target_candidates"]}


def summary(records, transitions):
    return {
        "unique_named":sum(r["selected_target"] in CHARACTERS for r in records),
        "multiple_people":sum(len(labels(r).intersection(CHARACTERS))>=2 for r in records),
        "teddy_don_ambiguous":sum({"TEDDY","DON"}<=labels(r) for r in records),
        **{k:sum(k in labels(r) for r in records) for k in ("AWAY","OTHER")},
        "unknown":sum(r["selected_target"]=="UNKNOWN" for r in records),
        **{f"cross_shot_{key}":sum(t["cross_shot"] and t["type"]==f"target_{key}_candidate"
                                  for t in transitions) for key in ("hold","switch")},
        **dict(Counter(r["evidence_level"] for r in records)),
    }


def prediction(record):
    return (record["selected_target"],sorted(labels(record)),record["evidence_level"])


def change(old, new, orientation, unfiltered):
    prior, current = labels(old), labels(new)
    added, removed = sorted(current-prior), sorted(prior-current)
    selection = old["selected_target"]!=new["selected_target"]
    difference = bool(added or removed or selection or old["evidence_level"]!=new["evidence_level"])
    if not orientation:
        explanation = ["existing_evidence_carried_forward"]
    elif orientation["status"]=="relative_only":
        explanation = ["visual_relation_added_without_transferable_scene_heading"]
    elif not new["evidence"].get("scene_geometry"):
        explanation = ["visual_orientation_not_applicable_to_this_interval"]
    else:
        explanation = ["camera_hypotheses_screen_depth_or_bed_filter_applied",
                       "all_compatible_alternatives_preserved"]
        if removed:
            explanation.append("no_compatible_surviving_hypothesis_for_removed_person")
        if added:
            explanation.append("alternative_compatible_in_at_least_one_surviving_hypothesis_or_unresolved")
    if selection and new["selected_target"]=="UNKNOWN":
        explanation.append("v1_1_requires_no_viable_named_rival_and_robust_camera_branch_support")
    return {"v1_0_target":old["selected_target"],"v1_1_target":new["selected_target"],
            "v1_0_candidates":sorted(prior),"v1_1_candidates":sorted(current),
            "v1_0_level":old["evidence_level"],"v1_1_level":new["evidence_level"],
            "added":added,"removed":removed,"changed_prediction":difference,
            "orientation_filter_changes_prediction":prediction(new)!=prediction(unfiltered),
            "v1_1_without_orientation_filter":{"target":unfiltered["selected_target"],
                "candidates":sorted(labels(unfiltered)),"level":unfiltered["evidence_level"]},
            "explanation":explanation,"v1_0_uncertainty":old["uncertainty_reason"]}


def publish_orientation(path, output, constraints, packet, qc_rows):
    image_dir = output/"review_contact_sheets"/"orientation"
    image_dir.mkdir()
    evidence = {}
    batches = [*[(shot,items["frames"]) for shot,items in packet["shots"].items()],
               *[(frame["shot_id"],[frame]) for frame in packet.get("interval_review_frames",[])]]
    for shot,frames in batches:
        for frame in frames:
            name = frame["file"]
            if (shot!=frame["shot_id"] or Path(name).name!=name
                    or sha256_file(path.parent/name)!=frame["sha256"]):
                raise ValueError("Orientation image did not match the reviewed hash")
            key = shot,frame["track_id"],frame["global_frame_index"]
            if qc_rows.get(key)!=frame["movie_timestamp_sec"]:
                raise ValueError("Orientation frame did not match the original Track PTS")
            copy2(path.parent/name,image_dir/name)
            evidence[frame["global_frame_index"]] = {**frame,"file":str((image_dir/name).relative_to(output))}
    sheets = []
    for name in packet["review_sheets"]:
        if Path(name).name!=name:
            raise ValueError("Invalid orientation sheet path")
        copy2(path.parent/name,image_dir/name)
        sheets.append(str((image_dir/name).relative_to(output)))
    for record in constraints.values():
        for row in record["constraints"]+record["visible_nonperson_objects"]:
            if not row["source_frames"] or any(n not in evidence for n in row["source_frames"]):
                raise ValueError("Constraint without a verified source frame")
    for item in [*packet.get("nonperson_interval_observations",[]),*packet.get("nonperson_visibility_notes",[])]:
        if any(n not in evidence for n in item["source_frames"]):
            raise ValueError("Nonperson observation without a reviewed source image")
    packet_copy = image_dir/"observations.json"
    copy2(path,packet_copy)
    return {"review_method":packet["review_method"],"frames":list(evidence.values()),"contact_sheets":sheets,
            "evidence_packet":str(packet_copy.relative_to(output))}


def publish_comparison(output, before, after, old_timeline, new_timeline):
    extract = lambda d:[t for c in d["characters"].values() for t in c["transitions"]]
    old,new = summary(before,extract(old_timeline)),summary(after,extract(new_timeline))
    lost = [r for r in after if r["comparison"]["v1_0_target"] in CHARACTERS and r["selected_target"] not in CHARACTERS]
    gained = [r for r in after if r["comparison"]["v1_0_target"] not in CHARACTERS and r["selected_target"] in CHARACTERS]
    changed = [r for r in after if r["comparison"]["changed_prediction"]]
    spatial = [r for r in after if r["comparison"]["orientation_filter_changes_prediction"]]
    with (output/"comparison_v1_0_v1_1.md").open("x",encoding="utf-8") as f:
        f.write("# Stage06.3 同区间比较\n\n"
                f"使用相同的 {len(before)} 个 interval_id、帧边界与电影 PTS。人工标签留空。\n\n"
                "| 指标（区间数） | v1.0 | v1.1 | Δ |\n| --- | ---: | ---: | ---: |\n")
        names = {"unique_named":"唯一人物主选","multiple_people":"多人物候选",
                 "teddy_don_ambiguous":"Teddy / Don 同时为候选","AWAY":"AWAY 候选","OTHER":"OTHER 候选",
                 "unknown":"UNKNOWN 主选","cross_shot_hold":"跨镜头 Hold","cross_shot_switch":"跨镜头 Switch"}
        for key in (*names,"gaze_supported","head_supported","context_only","insufficient"):
            a,b=old.get(key,0),new.get(key,0)
            f.write(f"| {names.get(key,key)} | {a} | {b} | {b-a:+d} |\n")
        for title,items in (("新增唯一人物主选",gained),("失去唯一人物主选",lost)):
            f.write(f"\n## {title}（{len(items)}）\n\n")
            if not items:
                f.write("无。\n")
            for r in items:
                c=r["comparison"]
                f.write(f"- `{r['interval_id']}`，{r['start_s']:.6f}–{r['end_s']:.6f} s，"
                        f"{c['v1_0_target']} → {c['v1_1_target']}，{', '.join(r['uncertainty_reason'])}。\n")
        f.write(f"\n## 预测或证据等级发生变化的区间（{len(changed)}）\n\n"
                "| interval_id | 全局时间 s | 目标前 → 后 | 证据等级前 → 后 | 理由 |\n"
                "| --- | --- | --- | --- | --- |\n")
        for r in changed:
            c=r["comparison"]
            f.write(f"| {r['interval_id']} | {r['start_s']:.6f}–{r['end_s']:.6f} "
                    f"| {' / '.join(c['v1_0_candidates'])} → {' / '.join(c['v1_1_candidates'])} "
                    f"| {c['v1_0_level']} → {c['v1_1_level']} | {', '.join(r['uncertainty_reason'])} |\n")
        f.write(f"\n## 单独检查机位过滤（{len(spatial)} 段）\n\n"
                "固定 v1.1 候选规则，只比较是否应用新增画面约束。"
                "以下区间的主选、候选集合或证据等级因机位过滤而变。\n\n")
        for r in spatial:
            raw=r["comparison"]["v1_1_without_orientation_filter"]
            f.write(f"- {r['interval_id']}，{raw['target']} → {r['selected_target']}，"
                    f"{' / '.join(raw['candidates'])} → {' / '.join(sorted(labels(r)))}。\n")
        if not spatial:
            f.write("筛选后方位集合收窄，候选输出维持相同。\n")
        f.write("\n逐区间旧候选、变动原因、投影相容性、保留方位分支见 gaze_target_candidates.json。"
                "这些数量衡量候选覆盖和歧义，准确性仍待人工填写。\n")
    return old,new,len(changed)


def run(args):
    baseline,output = args.baseline,args.output_dir
    inputs = (baseline,args.layout.parent,args.eye_qc,args.orientation_evidence.parent)
    if any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve()) for p in inputs):
        raise ValueError("Choose an independent output directory")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    source = read(baseline/"gaze_target_candidates.json")
    old_timeline = read(baseline/"gaze_target_timeline.json")
    mapping = read(baseline/"character_track_mapping.json")
    scene = read(args.layout)
    packet = read(args.orientation_evidence)
    identity = source["movie_id"],source["sequence_id"]
    if any((p["movie_id"],p["sequence_id"])!=identity for p in (scene,packet,mapping,old_timeline)):
        raise ValueError("Movie or sequence mismatch")
    if identity!=("tt12300742","seq_006") or set(packet["shots"])!=set(FOCUS_SHOTS):
        raise ValueError("This pilot requires the six Bugonia focus Shots")
    before = source["intervals"]
    camera = {s["shot_id"]:s for s in scene["shots"]}
    layouts = {l["layout_id"]:l for l in scene["layouts"]}
    if (len(before)!=230 or len({r["interval_id"] for r in before})!=230
            or set(r["shot_id"] for r in before)!=camera.keys() or len(camera)!=13):
        raise ValueError("Incomplete v1.0 interval coverage")
    timeline_sources = [Path(p) for p in source["input_sha256"] if Path(p).name=="sequence_gaze_timeline.json"]
    if (len(timeline_sources)!=1
            or sha256_file(timeline_sources[0])!=source["input_sha256"][str(timeline_sources[0])]):
        raise ValueError("Stage06.2 source timeline reference missing or changed")
    plot_source = read(timeline_sources[0])
    qc_path = args.eye_qc/"eye_observability_frames.parquet"
    original_hashes = {Path(path).resolve():value for path,value in source["input_sha256"].items()}
    for path in (args.layout,qc_path):
        if sha256_file(path)!=original_hashes.get(path.resolve()):
            raise ValueError(f"Baseline source changed: {path}")
    qc = pq.read_table(qc_path).to_pylist()
    qc_clock = {(r["shot_id"],r["track_id"],r["global_frame_index"]):r["movie_timestamp_sec"] for r in qc}
    raw_hypotheses,hypotheses,constraints = {},{},{}
    for sid in FOCUS_SHOTS:
        view = camera[sid]
        raw_hypotheses[sid],hypotheses[sid],constraints[sid] = orientations(view,layouts[view["layout_id"]],packet["shots"][sid])
    nonperson = packet.get("nonperson_interval_observations",[])
    by_id = {r["interval_id"]:r for r in before}
    for note in [*nonperson,*packet.get("nonperson_visibility_notes",[])]:
        span = by_id[note["interval_id"]]
        for frame in note["source_frames"]:
            key = span["shot_id"],span["track_id"],frame
            if (not span["start_frame"]<=frame<span["end_frame"] or key not in qc_clock
                    or not span["start_s"]<=qc_clock[key]<span["end_s"]):
                raise ValueError("Nonperson evidence is outside its interval")
    after = []
    for old in before:
        sid = old["shot_id"]
        if sid in constraints:
            view = camera[sid]
            current = evaluate(old,view,layouts[view["layout_id"]],hypotheses[sid],constraints[sid],nonperson)
            unfiltered = current
            if len(raw_hypotheses[sid])!=len(hypotheses[sid]):
                unfiltered = evaluate(old,view,layouts[view["layout_id"]],raw_hypotheses[sid],
                    {**constraints[sid],"status":"inherited_only","applied_constraint_ids":[]},nonperson)
        else:
            current = unfiltered = old.copy()
        after.append({**current,"representative_frames":[],
                      "comparison":change(old,current,constraints.get(sid),unfiltered)})
    common = {"schema_version":"stage06.3.gaze_target.v1.1","movie_id":identity[0],"sequence_id":identity[1],
              "human_verified":False}
    after_by_id = {r["interval_id"]:r for r in after}
    characters = {}
    for actor,baseline_track in old_timeline["characters"].items():
        records = [after_by_id[r["interval_id"]] for r in baseline_track["intervals"]]
        characters[actor] = {**baseline_track,"transitions":timeline_transitions(records) if actor!="UNKNOWN" else [],
                             "intervals":[{**prior,"selected_target":r["selected_target"],
                                          "target_candidates":[v["target"] for v in r["target_candidates"]],
                                          "evidence_level":r["evidence_level"],
                                          "type":"target_uncertain" if r["selected_target"]=="UNKNOWN" else "stable_target_candidate"}
                                          for prior,r in zip(baseline_track["intervals"],records,strict=True)]}
    transitions = [t for c in characters.values() for t in c["transitions"]]
    new_timeline = {**old_timeline,**common,"characters":characters}
    output.mkdir(parents=True,exist_ok=False)
    items = publish_aligned(output,choose_intervals(before,after,transitions),qc,mapping["video"])
    for item in items:
        r = after_by_id[item["interval_id"]]
        r["representative_frames"] = item["representative_frames"]
        item["comparison"] = r["comparison"]
        item["nonperson_visibility_observations"] = [v for v in packet.get("nonperson_visibility_notes",[])
                                                    if v["interval_id"]==item["interval_id"]]
    visual = publish_orientation(args.orientation_evidence,output,constraints,packet,qc_clock)
    write_json(output/"camera_orientation_constraints.json",{**common,**visual,"shots":list(constraints.values()),
                                                    "interval_nonperson_evidence":nonperson,
                                                    "interval_nonperson_visibility_notes":packet.get("nonperson_visibility_notes",[]),
                                                    "layout_source":str(args.layout),
                                                    "layout_sha256":sha256_file(args.layout)})
    write_json(output/"gaze_target_candidates.json",{**source,**common,"intervals":after,
                "character_track_mapping_source":str(baseline/"character_track_mapping.json"),
                "camera_orientation_source":"camera_orientation_constraints.json",
                "replay_input_sha256":{str(p):sha256_file(p) for p in (
                    baseline/"gaze_target_candidates.json",baseline/"gaze_target_timeline.json",
                    baseline/"character_track_mapping.json",args.layout,args.orientation_evidence,qc_path)}})
    write_json(output/"gaze_target_timeline.json",new_timeline)
    write_json(output/"review_items.json",{**common,"review_items":items})
    with (output/"human_review.csv").open("x",encoding="utf-8-sig",newline="") as f:
        cols = ("interval_id","shot_id","character","start_s","end_s")
        writer = csv.writer(f)
        writer.writerow((*cols,"v1_0_target","v1_1_target","human_target","review_status","notes"))
        writer.writerows((*[r[k] for k in cols],r["comparison"]["v1_0_target"],r["selected_target"],"","pending","")
                         for r in after)
    figure(output,plot_source,after,characters)
    old,new,changed = publish_comparison(output,before,after,old_timeline,new_timeline)
    write_review(output,constraints,items,old,new,changed)
    print(f"{len(after)} fixed intervals, {len(items)} reviewed intervals, {changed} prediction/level changes")
    print(old,new)


def write_review(output,constraints,items,old,new,changed):
    with (output/"review.md").open("x",encoding="utf-8") as f:
        f.write("# Bugonia seq_006 · Camera Orientation v1.1\n\n"
                f"13 个镜头的 230 个区间与 v1.0 一一对应。{changed} 个区间的候选或等级变化。"
                f"唯一人物主选 {old['unique_named']} → {new['unique_named']}，"
                f"Teddy / Don 同列 {old['teddy_don_ambiguous']} → {new['teddy_don_ambiguous']}。"
                "机位变化反映相对证据和采样范围。\n\n"
                "| Shot | 原片约束及边界 | 方位假设数（前 → 后） |\n| --- | --- | --- |\n")
        for sid,r in constraints.items():
            a,b=(sum(r[k].values()) for k in ("hypotheses_before","hypotheses_after"))
            f.write(f"| {sid} | {r['reason']} | {a} → {b} |\n")
        f.write(f"\n{len(items)} 个优先区间使用 {sum(len(i['representative_frames']) for i in items)} 张区间内原片。"
                "每张图保存 PTS、全局帧号和区间角色。样本短于三帧时共享角色。"
                "身份复用 v1.0，对标签无人工确认。\n\n"
                "AWAY / OTHER 必须有区间内独立视觉方向证据。"
                "物件只出现在画面中时，记录于机位文件的可见物件栏。"
                "头向、对白、镜头切换和目标未匹配本身不支持非人物目标。\n\n"
                "人工表 human_target 可填 MICHELLE、TEDDY、DON、AWAY、OTHER、UNKNOWN、UNCLEAR。"
                "标签初始为空，review_status 为 pending。\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("baseline","layout","eye-qc","orientation-evidence","output-dir"):
        parser.add_argument("--"+option,type=Path,required=True)
    run(parser.parse_args())


if __name__=="__main__":
    main()
