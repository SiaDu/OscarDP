"""Replay v1.3 measurements into v1.4; no model or source video is opened."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from oscardp.shots.transnet import sha256_file
from .geometry import head_change
from .inputs import EYES, POSE, read
from .review import plot
from .stable import MAX_TRANSITION_S, POSITION_LIMIT, analyze, interval
from .temporal import prepare_frames, valid_blocks

SHOTS = ("shot_000211","shot_000214","shot_000217")
SIDES = ("left","right")
VERSION = "stage06.2.evidence.v1.4"
WEAK_X = .020
OPPOSITE_X = .010
PERIOD = .5


def assess(rows, closures):
    result = prepare_frames(rows, closures)
    for row in result:
        eyes = {}
        for side in SIDES:
            reason = row[f"{side}_eye_visibility"]
            xy = [row[f"{side}_eye_{axis}"] for axis in "xy"]
            finite = np.isfinite(np.asarray(xy,dtype=float)).all()
            usable = reason=="usable" and finite and not row["closure_event_ids"]
            eyes[side] = {"usable":bool(usable),"source_status":reason,
                          "reason":None if usable else ("closure_candidate" if row["closure_event_ids"]
                                   else "missing_position" if reason=="usable" else reason),
                          "landmark_jump":row[f"{side}_eye_landmark_jump"]}
            if not usable:
                for axis in "xy":
                    row[f"{side}_eye_{axis}"]=None
        row["eye_position_status"] = eyes
        for signal,usable,original,invalid in (
                ("gaze_direction","camera_gaze_usable","gaze_valid","invalid_gaze_vector"),
                ("head_pose","head_usable","head_pose_valid","invalid_pose_vector")):
            reason = None
            if not row[usable]:
                reason = ("tracking_failure" if not row["tracking_usable"] else
                          "closure_candidate" if signal=="gaze_direction" and row["closure_event_ids"] else
                          "source_"+original+"_false" if not row[original] else invalid)
            row[signal+"_status"] = {"usable":row[usable],"source_valid":row[original],"reason":reason}
    return result


def weak_motion(rows, strict_events):
    """Short sustained horizontal displacement with corroboration or explicit limited support."""
    candidates=[]
    for side in SIDES:
        other="left" if side=="right" else "right"
        key=f"{side}_eye_x"
        mask=[r[key] is not None and r["tracking_usable"] and not r["closure_event_ids"]
              and not any(r[f"{s}_eye_landmark_jump"] for s in SIDES) for r in rows]
        for block in valid_blocks(rows,mask):
            for a in range(len(block)-5):
                first=block[a:a+3]
                left=np.array([rows[i][key] for i in first])
                if np.ptp(left)>POSITION_LIMIT:
                    continue
                for b in range(a+3,len(block)-2):
                    last=block[b:b+3]
                    duration=rows[last[-1]]["movie_timestamp_sec"]-rows[first[0]]["movie_timestamp_sec"]
                    if duration>MAX_TRANSITION_S:
                        break
                    if duration<.16:
                        continue
                    right=np.array([rows[i][key] for i in last])
                    centre=float(np.median(left))
                    delta=float(np.median(right)-centre)
                    if (abs(delta)<WEAK_X or np.ptp(right)>POSITION_LIMIT
                            or np.min(np.sign(delta)*(right-centre))<WEAK_X/2):
                        continue
                    i,j=first[0],last[-1]
                    span=interval(rows,[i,j])
                    if any(e["type"]=="eye_shift_candidate" and e["start_s"]<span["end_s"]
                           and e["end_s"]>span["start_s"] for e in strict_events):
                        continue
                    other_key=f"{other}_eye_x"
                    other_delta=None
                    other_before=[rows[k][other_key] for k in first if rows[k][other_key] is not None]
                    other_after=[rows[k][other_key] for k in last if rows[k][other_key] is not None]
                    if len(other_before)>=2 and len(other_after)>=2:
                        other_delta=float(np.median(other_after)-np.median(other_before))
                        if abs(other_delta)>=OPPOSITE_X and other_delta*delta<0:
                            continue
                    # Do not call a lower-confidence event when both eyes already
                    # support the full positional threshold.
                    if abs(delta)>=POSITION_LIMIT and other_delta is not None and abs(other_delta)>=POSITION_LIMIT:
                        continue
                    head=None
                    if rows[i]["head_usable"] and rows[j]["head_usable"]:
                        head=head_change([rows[i][k] for k in POSE],[rows[j][k] for k in POSE])
                        if head>6:
                            continue
                    other_reason=("insufficient_continuous_eye_position" if other_delta is None else
                                  "below_strict_displacement_threshold" if abs(other_delta)<POSITION_LIMIT else
                                  "motion_support_asymmetric")
                    other_qc=sorted({rows[k][f"{other}_eye_visibility"] for k in range(i,j+1)
                                     if rows[k][other_key] is None})
                    candidates.append({"type":"weak_eye_motion_candidate",**span,
                                       "eye_support":[side],"supporting_eye_delta_x":delta,
                                       "other_eye_delta_x":other_delta,"other_eye_reason":other_reason,
                                       "other_eye_unavailable_statuses":other_qc,"head_change_deg":head,
                                       "gaze_direction_evidence":"gaze_direction_unknown"})
    # Retain the largest persistent displacement among overlapping windows.
    chosen=[]
    for event in sorted(candidates,key=lambda e:(-abs(e["supporting_eye_delta_x"]),e["start_s"])):
        if not any(e["start_s"]<event["end_s"] and event["start_s"]<e["end_s"] for e in chosen):
            chosen.append(event)
    return sorted(chosen,key=lambda e:e["start_s"])


def select_frames(rows, initial, weak):
    by_frame={r["global_frame_index"]:i for i,r in enumerate(rows)}
    def modes(r):
        return [k for k,good in (("eye_position",any(v["usable"] for v in r["eye_position_status"].values())),
                                 ("gaze_direction",r["camera_gaze_usable"]),("head_pose",r["head_usable"])) if good]
    eligible={i:modes(r) for i,r in enumerate(rows) if r["tracking_usable"] and not r["closure_event_ids"] and modes(r)}
    picks={}
    for item in initial:
        i=by_frame[item["global_frame_index"]]
        if i in eligible:
            reasons=set(item["reasons"])
            reasons.update("stable_state" if "stable" in r else "motion_event" for r in tuple(reasons)
                           if "stable" in r or "motion" in r or "shift" in r)
            picks[i]=reasons
    for e in weak:
        for n in (e["start_frame"],e["end_frame"]-1):
            i=by_frame[n]
            if i in eligible:
                picks.setdefault(i,set()).update(("motion_event","weak_eye_motion"))
    coverage=[]
    start,end=rows[0]["movie_timestamp_sec"],rows[-1]["source_next_timestamp_sec"]
    for j in range(int(np.ceil((end-start)/PERIOD))):
        a,b=start+j*PERIOD,min(end,start+(j+1)*PERIOD)
        candidates=[i for i in eligible if a<=rows[i]["movie_timestamp_sec"]<b]
        existing=[i for i in candidates if i in picks]
        used=existing or candidates
        entry={"start_s":a,"end_s":b,"status":"missing","representative_frame":None}
        if used:
            i=min(used,key=lambda i:(-len(eligible[i]),"eye_position" not in eligible[i],
                                    abs(rows[i]["movie_timestamp_sec"]-(a+b)/2)))
            picks.setdefault(i,set()).add("periodic_check")
            entry.update(status="covered",representative_frame=rows[i]["global_frame_index"])
        coverage.append(entry)
    frames=[{"global_frame_index":rows[i]["global_frame_index"],"movie_timestamp_sec":rows[i]["movie_timestamp_sec"],
             "reasons":sorted(reason),"available_modalities":eligible[i],
             "unavailable_modalities":sorted({"eye_position","gaze_direction","head_pose"}-set(eligible[i]))}
            for i,reason in sorted(picks.items())]
    return frames,coverage


def write_json(path,data):
    with path.open("x",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")


def run_shot(source,output,shot,track_id=None):
    track_id=track_id or shot+"_track_001"
    paths=[source/name for name in ("eye_position_frames.parquet","gaze_states.json",
                                     "motion_events.json","selected_gaze_frames.json")]
    old_states,old_motion,old_picks=map(read,paths[1:])
    rows=pq.read_table(paths[0]).to_pylist()
    if (not rows or any(d["shot_id"]!=shot for d in (old_states,old_motion,old_picks))
            or any((r["movie_id"],r["shot_id"],r["track_id"])!=("tt12300742",shot,track_id) for r in rows)
            or any(b["global_frame_index"]<=a["global_frame_index"]
                   or b["movie_timestamp_sec"]<a["source_next_timestamp_sec"]-1e-6 for a,b in zip(rows,rows[1:]))
            or any(not np.isfinite([r["movie_timestamp_sec"],r["source_next_timestamp_sec"]]).all()
                   or r["source_next_timestamp_sec"]<=r["movie_timestamp_sec"] for r in rows)):
        raise ValueError(f"Invalid v1.3 source identity or timeline: {shot}")
    closures=[{"event_type":"eye_closure_candidate","event_id":e["source_event_id"],
               "start_frame":e["start_frame"],"end_frame":e["end_frame"],"start_sec":e["start_s"],"end_sec":e["end_s"]}
              for e in old_motion["events"] if e["type"]=="blink_candidate"]
    rows=assess(rows,closures)
    # This auxiliary stream is unchanged by v1.4. Keep its Stage05 links.
    adjunct=old_motion["direction_motion_features"]
    states,motion,initial=analyze(rows,closures,adjunct,allow_monocular=True,independent_gaze=True)
    weak=weak_motion(rows,motion)
    # Preserve v1.3-compatible strict and adjustment events. Weak windows are
    # supplementary evidence, not alternate labels for those events.
    motion=sorted(motion+weak,key=lambda e:e["start_s"])
    for e in motion:
        if e["type"]=="eye_shift_candidate":
            e["gaze_direction_evidence"]="gaze_direction_unknown"
    picks,coverage=select_frames(rows,initial,weak)
    evidence = [{"global_frame_index":r["global_frame_index"],"movie_timestamp_sec":r["movie_timestamp_sec"],
                 "source_next_timestamp_sec":r["source_next_timestamp_sec"],"parquet_row":r["parquet_row"],
                 **{k:r[k] for k in ("eye_position_status","gaze_direction_status","head_pose_status",
                                     "tracking_usable","source_qc_reasons","qc_flags")}} for r in rows]
    modalities={"eye_left":sum(r["eye_position_status"]["left"]["usable"] for r in rows),
                "eye_right":sum(r["eye_position_status"]["right"]["usable"] for r in rows),
                "eye_both":sum(all(e["usable"] for e in r["eye_position_status"].values()) for r in rows),
                "gaze_direction":sum(r["camera_gaze_usable"] for r in rows),
                "head_pose":sum(r["head_usable"] for r in rows)}
    common={"schema_version":VERSION,"movie_id":"tt12300742","sequence_id":"seq_006",
            "shot_id":shot,"track_id":track_id,"status":"model_candidate"}
    stats={"frames":len(rows),"modalities":modalities,"states":dict(Counter(s["type"] for s in states)),
           "events":dict(Counter(e["type"] for e in motion)),"picks":len(picks),
           "bins_covered":sum(e["status"]=="covered" for e in coverage),"bins":len(coverage)}
    output.mkdir()
    pq.write_table(pa.Table.from_pylist(evidence).replace_schema_metadata(
        {"schema_version":VERSION,"time_axis":"movie global PTS","source_parquet":str(paths[0])}),
        output/"evidence_frames.parquet")
    write_json(output/"gaze_states.json",{**common,"states":states,"summary":stats,
                                          "input_sha256":{str(p):sha256_file(p) for p in paths}})
    write_json(output/"motion_events.json",{**common,"events":motion,"direction_motion_features":adjunct,
                                             "weak_threshold_same_eye_width":WEAK_X})
    write_json(output/"selected_gaze_frames.json",{**common,"frames":picks,"total_source_frames":len(rows),
                                                    "periodic_interval_s":PERIOD,"time_coverage":coverage})
    plot(rows,[e for e in motion if e["type"] not in {"unknown","blink_candidate"}],
         output/"eye_motion_review.png",eye_states=states,eye_label="Saved MediaPipe iris fit")
    with (output/"review.md").open("x",encoding="utf-8") as f:
        f.write(f"# Stage06.2 v1.4 · {shot}\n\n"
                "仅重算 v1.3 数值，无图像解码或模型推理。Gaze 独立使用 OpenFace 相机方向，"
                "头姿与 MediaPipe 眼位各有可用状态。\n\n"
                f"- 逐帧可用性：{modalities}\n- 状态：{stats['states']}\n"
                f"- 事件：{stats['events']}\n"
                f"- 代表帧：{len(picks)}/{len(rows)}；0.5 s 窗口覆盖 {stats['bins_covered']}/{stats['bins']}。\n"
                "- 弱眼动只说明拟合位置有持续证据。逐帧拒绝原因和原始 Stage05 关联行见 evidence_frames.parquet。\n"
                "\n## 弱眼动候选（电影全局秒）\n")
        for e in weak:
            f.write(f"- {e['start_s']:.6f}–{e['end_s']:.6f}：{e['eye_support'][0]} "
                    f"水平位移 {e['supporting_eye_delta_x']:+.3f} 眼宽；对侧 {e['other_eye_reason']}。\n")
        if not weak:
            f.write("- 无。\n")
    return {"old_states":Counter(s["type"] for s in old_states["states"]),
            "old_events":Counter(e["type"] for e in old_motion["events"]),
            "old_picks":len(old_picks["frames"]),
            "old_bins":len({int((e["movie_timestamp_sec"]-rows[0]["movie_timestamp_sec"])/PERIOD)
                            for e in old_picks["frames"]}),
            "old_eye":old_states["mediapipe"],"new":stats}


def run(source,output):
    if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
        raise ValueError("Source and output must be independent directories")
    output.mkdir(parents=True,exist_ok=False)
    results={shot:run_shot(source/shot,output/shot,shot) for shot in SHOTS}
    types=("eye_shift_candidate","weak_eye_motion_candidate","eye_adjustment_candidate","gaze_direction_shift_candidate")
    with (output/"comparison_summary.md").open("x",encoding="utf-8") as f:
        f.write("# Stage06.2 v1.3 → v1.4 · 数值复放\n\n"
                "| Shot | 双眼有效（前 → 后） | 稳定 Gaze | Eye Shift | Weak | Adjustment | Gaze Shift | 代表帧 | 0.5 s 窗口覆盖 |\n"
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for shot,r in results.items():
            n=r["new"]
            count=lambda k: f"{r['old_events'][k]} → {n['events'].get(k,0)}"
            f.write(f"| {shot} | {r['old_eye']['both']} → {n['modalities']['eye_both']} / {n['frames']} "
                    f"| {r['old_states']['stable_gaze_candidate']} → {n['states'].get('stable_gaze_candidate',0)} "
                    + "".join(f"| {count(k)} " for k in types)
                    + f"| {r['old_picks']} → {n['picks']} | {r['old_bins']} → {n['bins_covered']} / {n['bins']} |\n")
        f.write("\nGaze 稳定仍要求两眼和综合相机方向在同一有效段内达到现有 3°、0.25 s 门槛；"
                "无需同时有 MediaPipe 眼位或头相对方向。缺失时间窗单独记录，无替代帧。\n")
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    args=parser.parse_args()
    try:
        run(args.source,args.output_dir)
    except (OSError,ValueError,KeyError) as e:
        parser.exit(1,f"evidence_replay: {e}\n")


if __name__=="__main__":
    main()
