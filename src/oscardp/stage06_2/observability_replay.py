"""Apply the Stage06.2 Blink QC fix to saved full-sequence results."""

import argparse
from collections import Counter
import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from oscardp.shots.transnet import sha256_file
from .blink_qc import FACE_EYE_COLUMNS, MAX_BLINK_S, SIDES, blink_qc
from .evidence_replay import write_json
from .inputs import POSE, read
from .temporal import prepare_frames

KEYS = ("movie_id", "shot_id", "track_id", "extraction_mode", "global_frame_index",
        "movie_timestamp_sec", "source_next_timestamp_sec")
CLOCK = ("global_frame_index", "movie_timestamp_sec", "source_next_timestamp_sec")
STATES = ("eye_open", "eye_closed", "eye_unobservable", "eye_unknown")


def saved_rows(directory, source_measurements):
    raw = pq.read_table(directory/"eye_position_frames.parquet").to_pylist()
    evidence = pq.read_table(directory/"evidence_frames.parquet").to_pylist()
    measurements = pq.read_table(source_measurements, columns=[*KEYS,*FACE_EYE_COLUMNS,"parquet_row"],
                                 filters=[("track_id","=",raw[0]["track_id"]),
                                          ("extraction_mode","=",raw[0]["extraction_mode"])]).to_pylist()
    by_row = {r["parquet_row"]:r for r in measurements}
    for row, previous in zip(raw, evidence, strict=True):
        r = by_row[row["parquet_row"]]
        if any(row[k] != r[k] for k in KEYS) or any(row[k] != previous[k] for k in (*CLOCK,"parquet_row")):
            raise ValueError(f"Saved frame identity or PTS mismatch: {directory}")
        row.update({k:r[k] for k in FACE_EYE_COLUMNS})
    if len(raw) != len(by_row):
        raise ValueError(f"Stage05 and Stage06 Track have different frame counts: {directory}")
    return raw, evidence


def evidence_rows(raw, old_evidence, observations):
    result = []
    # Use the original Stage05 gaze/pose QC, without the legacy eye-event mask.
    for row, old, eyes in zip(prepare_frames(raw, []), old_evidence, observations, strict=True):
        head, gaze = row["head_direction"], row["gaze_direction"]
        position = {}
        for side in SIDES:
            previous = old["eye_position_status"][side]
            unobservable = eyes[side]["state"] == "eye_unobservable"
            reason = previous["reason"]
            if not previous["usable"]:
                if unobservable:
                    reason = "eye_unobservable"
                elif reason == "closure_candidate":
                    reason = "legacy_position_masked"
            position[side] = {**previous,"state":"usable" if previous["usable"] else
                              "unobservable" if unobservable else "unavailable",
                              "reason":reason,"prior_reason":previous["reason"]}
        gaze_reason = None
        if gaze is None:
            gaze_reason = ("tracking_failure" if not row["tracking_usable"] else
                           "source_gaze_valid_false" if not row["gaze_valid"] else "invalid_gaze_vector")
        result.append({**{k:row[k] for k in (*KEYS,"parquet_row")}, "eye_position_status":position,
                       "eye_observability":eyes, "blink_status":{
                           s:{"eye_open":"observable","eye_closed":"observable",
                              "eye_unobservable":"unobservable","eye_unknown":"unknown"}[eyes[s]["state"]]
                           for s in SIDES},
                       "head_pose_status":{**old["head_pose_status"],
                                           "state":"usable" if head is not None else "unavailable"},
                       **{key:row[key] if head is not None else None for key in POSE},
                       "head_facing_direction":head, "gaze_direction":gaze,
                       "gaze_direction_status":{"usable":gaze is not None,"source_valid":row["gaze_valid"],
                                                "state":"usable" if gaze is not None else "unavailable",
                                                "reason":gaze_reason},
                       "prior_gaze_direction_status":old["gaze_direction_status"],
                       "source_qc_reasons":row["source_qc_reasons"]})
    return result


def duration(event):
    return None if event is None else event["end_s"]-event["start_s"]


def compare(track, old_blinks, new_blinks, evidence, source_events):
    pending = list(new_blinks)
    for old in old_blinks:
        match = next((e for e in pending if old["source_event_id"] in e["source_event_ids"]), None)
        status = "removed"
        reasons = []
        if match:
            pending.remove(match)
            status = "supported" if all(match[k]==old[k] for k in ("start_frame","end_frame")) else "refined"
            reasons.append("observable_open_closed_open")
        else:
            reasons.append("no_complete_observable_eye_cycle")
        if duration(old) > MAX_BLINK_S:
            reasons.append("exceeds_short_blink_duration")
        rows = [r for r in evidence if old["start_frame"] <= r["global_frame_index"] < old["end_frame"]]
        n_unobservable = [sum(r["eye_observability"][s]["state"]=="eye_unobservable" for r in rows) for s in SIDES]
        if any(n_unobservable):
            reasons.append("unobservable_eye_frames_present")
        source = source_events.get(old["source_event_id"],{})
        cycle = source.get("measurements",{}).get("cycle_evidence")
        yield comparison_row(track, old, match, status, reasons, *n_unobservable, cycle)
    for event in pending:
        yield comparison_row(track, None, event, "added", ["observable_open_closed_open"], None, None, None)


def comparison_row(track, old, new, status, reasons, left, right, prior_cycle):
    return {"shot_id":track.split("_track_")[0],"track_id":track,"decision":status,"reasons":";".join(reasons),
            "before_count":int(old is not None),"after_count":int(new is not None),
            **{f"{prefix}_{k}":e[k] if e else None for prefix,e in (("before",old),("after",new))
               for k in ("start_s","end_s","source_event_id")},
            "before_duration_s":duration(old),"after_duration_s":duration(new),
            "after_eye_support":",".join(new["eye_support"]) if new else None,
            "before_left_unobservable_frames":left,"before_right_unobservable_frames":right,
            "stage05_cycle_evidence":prior_cycle}


def analyze_track(directory, measurement, source_events):
    raw, old_evidence = saved_rows(directory, measurement)
    intervals = list(source_events.values())
    observations, new_blinks = blink_qc(raw, intervals)
    evidence = evidence_rows(raw, old_evidence, observations)
    old_motion = read(directory/"motion_events.json")
    unchanged = [e for e in old_motion["events"] if e["type"]!="blink_candidate"]
    old_blinks = [e for e in old_motion["events"] if e["type"]=="blink_candidate"]
    counts = {side:Counter(row[side]["state"] for row in observations) for side in SIDES}
    comparisons = list(compare(raw[0]["track_id"], old_blinks, new_blinks, evidence, source_events))
    motion = {**old_motion,"events":sorted(unchanged+new_blinks,key=lambda e:e["start_s"])}
    return evidence, counts, comparisons, old_blinks != new_blinks, motion


def run(source, output, focus_track=None, manual_note=None):
    if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
        raise ValueError("Source and output must be independent directories")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    timeline = source/"sequence_gaze_timeline.json"
    shots = read(timeline)["shots"]
    entries = [(s,t) for s in shots for t in s["tracks"]]
    if focus_track and not any(t["track_id"]==focus_track and "stats" in t for _,t in entries):
        raise ValueError(f"Focus Track has no measurements: {focus_track}")
    output.mkdir(parents=True,exist_ok=False)
    summary = {}
    comparisons = []
    frame_rows = []
    event_cache = {}
    hashes = {str(timeline):sha256_file(timeline)}
    for shot, track in sorted(entries, key=lambda pair: pair[1]["track_id"] != focus_track):
        tid, sid = track["track_id"], shot["shot_id"]
        if "stats" not in track:
            continue
        directory = source/sid/tid
        events_path = Path(shot["stage05"]["events"])/"performance_events.json"
        measurement = Path(shot["stage05"]["measurement"])/"frame_features.parquet"
        if events_path not in event_cache:
            event_cache[events_path] = [e for e in read(events_path)["events"] if e["event_type"]=="eye_closure_candidate"]
        events = {e["event_id"]:e for e in event_cache[events_path] if e["track_id"]==tid}
        data, counts, compared, changed, motion = analyze_track(directory, measurement, events)
        frame_rows.extend(data)
        paths = [directory/n for n in ("motion_events.json","evidence_frames.parquet","eye_position_frames.parquet")]
        for path in [*paths, events_path, measurement]:
            if str(path) not in hashes:
                hashes[str(path)] = sha256_file(path)
        summary[tid] = {"counts":counts, "rows":len(data), "changed":changed,
                        "head":sum(r["head_pose_status"]["usable"] for r in data),
                        "old_gaze":sum(r["prior_gaze_direction_status"]["usable"] for r in data),
                        "gaze":sum(r["gaze_direction_status"]["usable"] for r in data),
                        "eye_position":sum(any(e["usable"] for e in r["eye_position_status"].values()) for r in data),
                        "before":sum(r["before_count"] for r in compared),"after":sum(r["after_count"] for r in compared)}
        if tid == focus_track:
            summary[tid]["observability_reasons"] = Counter(
                reason for frame in data for eye in frame["eye_observability"].values() for reason in eye["reasons"])
        if changed:
            dst = output/sid/tid
            dst.mkdir(parents=True,exist_ok=False)
            write_json(dst/"motion_events.json",{**motion,"blink_qc":{
                "source_motion_events":str(paths[0]),"source_motion_events_sha256":hashes[str(paths[0])],
                "max_duration_s":MAX_BLINK_S}})
        comparisons.extend(compared)
    with (output/"blink_qc_comparison.csv").open("x",encoding="utf-8",newline="") as f:
        columns = list(comparison_row("",None,None,"",[],None,None,None))
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted(comparisons,key=lambda r:(r["shot_id"],r["track_id"],r["before_start_s"] or r["after_start_s"])))
    frame_rows.sort(key=lambda r:(r["shot_id"],r["track_id"],r["global_frame_index"]))
    table = pa.Table.from_pylist(frame_rows).replace_schema_metadata({
        "time_axis":"movie global PTS; exclusive source_next_timestamp_sec",
        "eye_sides":"image-left and image-right",
        "camera_axes":"+X screen right, +Y down, +Z away from camera",
        "head_pose":"OpenFace XYZ radians",
    })
    pq.write_table(table,output/"eye_observability_frames.parquet",compression="zstd")
    write_json(output/"input_sha256.json", hashes)
    write_reports(output, shots, summary, comparisons, focus_track, manual_note)


def write_reports(output, shots, summary, comparisons, focus, note):
    old = sum(r["before_count"] for r in comparisons)
    new = sum(r["after_count"] for r in comparisons)
    long = [r for r in comparisons if (r["before_duration_s"] or 0)>MAX_BLINK_S]
    with (output/"eye_observability_summary.md").open("x",encoding="utf-8") as f:
        f.write("# seq_006 · 眼部可观测性与 Blink QC\n\n"
                f"原有 {old} 个 Blink 候选，复核后 {new} 个。旧长候选 {len(long)} 个，"
                f"涉及 {len({r['shot_id'] for r in long})} 个 Shot。"
                "左/右指画面中的左右眼。计数顺序为睁眼、闭眼、不可观测、证据不足。\n\n"
                "| Shot | Track | 帧数 | 左眼 | 右眼 | 眼位 | 头姿 | Gaze 前 → 后 | Blink 前 → 后 |\n"
                "| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |\n")
        for shot in shots:
            for t in shot["tracks"] or [{"track_id":"—"}]:
                tid = t["track_id"]
                prefix = f"| {shot['shot_id']} | {tid} "
                r = summary.get(tid)
                if not r:
                    f.write(prefix+"| 未测 | — | — | — | — | — | — |\n")
                    continue
                eye_counts = [" / ".join(str(r["counts"][side][s]) for s in STATES) for side in SIDES]
                f.write(prefix+f"| {r['rows']} | {eye_counts[0]} | {eye_counts[1]} | {r['eye_position']} "
                        f"| {r['head']} | {r['old_gaze']} → {r['gaze']} | {r['before']} → {r['after']} |\n")
        f.write("\n逐帧状态和拒绝原因见 eye_observability_frames.parquet。眼位帧数为既有可用拟合中任一眼有效的帧数。\n")
    with (output/"review.md").open("x",encoding="utf-8") as f:
        f.write("# Stage06.2 · Eye Observability & Blink QC\n\n"
                "Stage05.5 的相对开度可保留单眼下降、观测边界缺失或长持续区间。旧 Stage06.2 将这些候选直接写成 Blink。"
                "当前规则以原画面像素尺度、侧脸姿态、跟踪与眼睑形状为前提，逐眼查找连续睁眼、闭合和重开。"
                f"短事件上限为 {MAX_BLINK_S:.2f} 秒。持续闭眼保留逐帧状态。\n\n")
        if focus:
            r = summary[focus]
            f.write(f"## {focus}\n\n")
            if note:
                f.write(f"原片人工观察（用户提供）\n\n> {note}\n\n")
            f.write(f"- 复放 {r['rows']} 帧。眼位可用 {r['eye_position']} 帧，头姿可用 {r['head']} 帧，"
                    f"Gaze {r['old_gaze']} → {r['gaze']} 帧，Blink {r['before']} → {r['after']} 个。\n")
            for side in SIDES:
                counts = r["counts"][side]
                f.write(f"- 画面{'左' if side=='left' else '右'}眼不可观测 {counts['eye_unobservable']} 帧，"
                        f"证据不足 {counts['eye_unknown']} 帧。\n")
            reasons = r["observability_reasons"].most_common(4)
            if reasons:
                f.write("- 最常见的逐眼 QC 原因（同一眼可有多项）为 "
                        + "、".join(f"{reason} {n} 次" for reason,n in reasons)+"。\n")
            for item in comparisons:
                if item["track_id"]==focus and item["before_count"]:
                    f.write(f"- 旧候选 {item['before_start_s']:.3f}–{item['before_end_s']:.3f} s，"
                            f"长 {item['before_duration_s']:.4f} s，{item['decision']}，"
                            f"原因 {item['reasons']}。\n")
            f.write("\n")
        f.write("## 与旧时间线的关系\n\n"
                f"- 比较表记录了 {len(long)} 个超过短 Blink 上限的旧区间及其复核结果。"
                "单眼候选只记录支持该判断的眼。\n"
                "- 按 Stage05.4 原数值提供独立逐帧 Gaze。头朝向向量使用 OpenFace 坐标系，"
                "在 Gaze 无效时仍保留为单独的低精度空间证据。\n"
                "- 旧的 Stable Gaze、Eye Shift、Weak Eye Motion 和其他非 Blink 事件逐项保留。"
                "本次只重放保存的数据。\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--focus-track")
    parser.add_argument("--manual-note")
    args = parser.parse_args()
    run(args.source,args.output_dir,args.focus_track,args.manual_note)


if __name__ == "__main__":
    main()
