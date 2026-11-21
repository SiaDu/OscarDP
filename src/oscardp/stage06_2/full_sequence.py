"""Assemble Bugonia seq_006 from saved and newly measured Stage05 Tracks."""

import argparse
from collections import Counter
from pathlib import Path
from shutil import copy2
from types import SimpleNamespace

import pyarrow.parquet as pq

from . import mediapipe_pilot as mp
from .evidence_replay import run_shot, write_json
from .inputs import read
from .review import decode_review_frames

MOVIE = "tt12300742"
SEQUENCE = "seq_006"
REUSE = {"shot_000211","shot_000214","shot_000217"}
FOCUS = {
    "shot_000212":"原片复核：前景为 Michelle 的后脑，无法测量；Track 001 覆盖后面的 Teddy。",
    "shot_000218":"原片复核：Track 002 覆盖前景 Teddy；Track 003 在背景小脸位置只检测到 3 帧，Don 缺少连续数据。",
    "shot_000222":"原片复核：001／002／003 位置分别对应 Don／Michelle／Teddy。004 在 003 断开后接续同一区域，疑似 Teddy 的碎片；保持独立 Track，身份均未确认。",
    "shot_000223":"Stage01 起始帧残留上一镜头；两个单帧 Track 保留为不可用，实际测量从 003 的首帧开始。",
    "shot_000216":"前景疑似人物局部，主要可见候选为 Don。",
}


def shot_order(sequence):
    if sequence["movie_id"] != MOVIE:
        raise ValueError("Movie mismatch")
    seq=next(s for s in sequence["sequence_candidates"] if s["sequence_id"]==SEQUENCE)
    shots=seq["shot_ids"]
    if shots != [f"shot_{i:06d}" for i in range(211,224)]:
        raise ValueError("Stage04.3 did not provide the expected 13-shot contiguous sequence")
    return seq,shots


def initial_coverage(args,shots,output):
    stage=Path(args.stage04).parent
    visual={s["shot_id"]:s for s in read(stage/f"Stage03_{MOVIE}.json")["shots"]}
    context={s["shot_id"]:s for s in read(args.stage04)["shots"]}
    old=read(args.corrected/"face_tracks.json")["tracks"]
    lines=["# seq_006 · 补跑前覆盖","",
           "| Shot | Stage03／04 可见候选 | Stage05.2／05.3 | Stage05.4／05.5 | Stage06.2 | 待办／限制 |",
           "| --- | --- | --- | --- | --- | --- |"]
    for shot in shots:
        note=FOCUS.get(shot,"人物名只作为上游视觉候选。")
        names=sorted({r["character_name"] for r in context[shot]["character_candidates"] if r["character_name"]})
        seen=", ".join(names) or "未命名"
        seen += f"（{len(visual[shot]['subjects'])} 个 Stage03 可见主体条目）"
        if shot in REUSE:
            tids=[t["track_id"] for t in old if t["shot_id"]==shot]
            lines.append(f"| {shot} | {seen} | {', '.join(tids)}／身份未人工核验 | 已有连续测量与候选事件 | 已有 v1.4 | 复用，实验性边界待审。{note} |")
        else:
            lines.append(f"| {shot} | {seen} | 尚缺正式 Track | 尚缺 | 尚缺 | 补跑 Stage05.2–05.5 与 Stage06.2。{note} |")
    with output.open("x",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")


def missing_intervals(rows):
    for name, predicate in {
        "eye_position":lambda r:any(x["usable"] for x in r["eye_position_status"].values()),
        "gaze_direction":lambda r:r["gaze_direction_status"]["usable"],
        "head_pose":lambda r:r["head_pose_status"]["usable"],
    }.items():
        start=None
        for i,row in enumerate(rows):
            if not predicate(row):
                if start is None:
                    start=i
            elif start is not None:
                yield {"modality":name,"start_s":rows[start]["movie_timestamp_sec"],
                       "end_s":rows[i-1]["source_next_timestamp_sec"],"reason":"frame_qc"}
                start=None
        if start is not None:
            yield {"modality":name,"start_s":rows[start]["movie_timestamp_sec"],
                   "end_s":rows[-1]["source_next_timestamp_sec"],"reason":"frame_qc"}


def timeline_records(shot,tid,kind,records,evidence,source):
    by={r["global_frame_index"]:r for r in evidence}
    for record in records:
        lo,hi=record["start_frame"],record["end_frame"]-1
        if (lo not in by or hi not in by or hi<lo
                or abs(record["start_s"]-by[lo]["movie_timestamp_sec"])>1e-6
                or abs(record["end_s"]-by[hi]["source_next_timestamp_sec"])>1e-6):
            raise ValueError(f"Interval invalid or crosses Track/shot boundary: {tid}")
        yield {**record,"shot_id":shot,"track_id":tid,"record_kind":kind,"source":str(source)}


def summarize_track(shot,tid,out,shot_span):
    state=read(out/"gaze_states.json")
    motion=read(out/"motion_events.json")
    selected=read(out/"selected_gaze_frames.json")
    evidence=pq.read_table(out/"evidence_frames.parquet").to_pylist()
    raw=pq.read_table(out/"eye_position_frames.parquet",columns=["identity_cluster_id","identity_status","person",
                      "head_pose_valid","gaze_valid"]).to_pylist()
    first=raw[0]
    stats=state["summary"]
    modalities=stats["modalities"]
    if not any(modalities.values()):
        status="unavailable"
    elif first["person"] is None:
        status="partial_identity_unknown"
    elif not all(modalities[k] for k in ("gaze_direction","head_pose")) or not (
            modalities["eye_left"] or modalities["eye_right"]):
        status="partial_modalities_unavailable"
    else:
        status="processed_with_qc"
    gaps=list(missing_intervals(evidence))
    for a,b in [(shot_span["start"],evidence[0]["movie_timestamp_sec"]),
                (evidence[-1]["source_next_timestamp_sec"],shot_span["end"])]:
        if b-a>1e-5:
            gaps.append({"modality":"all","start_s":a,"end_s":b,"reason":"outside_measured_track_span"})
    result={"track_id":tid,"status":status,"identity_cluster_id":first["identity_cluster_id"],
            "identity_status":first["identity_status"],"person_candidate":first["person"] or "UNKNOWN",
            "human_verified":False,"stats":stats,"missing_intervals":gaps,
            "source_openface":{"pose":sum(r["head_pose_valid"] for r in raw),"gaze":sum(r["gaze_valid"] for r in raw)},
            "source":str(out)}
    records=[*timeline_records(shot,tid,"state",state["states"],evidence,out/"gaze_states.json"),
             *timeline_records(shot,tid,"motion",motion["events"],evidence,out/"motion_events.json")]
    pts={r["global_frame_index"]:r["movie_timestamp_sec"] for r in evidence}
    for r in selected["frames"]:
        n=r["global_frame_index"]
        if n not in pts or abs(pts[n]-r["movie_timestamp_sec"])>1e-6:
            raise ValueError("Representative frame outside Track/PTS")
    return result,records


def run(args):
    seq,shots=shot_order(read(args.sequences))
    stage01=read(args.stage04.parent/f"Stage01_{MOVIE}.json")
    bounds={r["shot_id"]:r["temporal"] for r in stage01["shots"]}
    contexts={r["shot_id"]:r["character_candidates"] for r in read(args.stage04)["shots"]}
    root=args.output_dir
    root.mkdir(parents=True,exist_ok=False)
    initial_coverage(args,shots,root/"coverage_before.md")
    assignments={}
    corrected_tracks=read(args.corrected/"face_tracks.json")["tracks"]
    legacy_stage05={"tracking":str(args.corrected)}
    for key in read(args.reuse_positions/"shot_000211"/"gaze_states.json")["input_sha256"]:
        for filename,stage in {"identity_clusters.json":"identity","track_measurements.json":"measurement",
                               "performance_events.json":"events"}.items():
            if Path(key).name==filename:
                legacy_stage05[stage]=str(Path(key).parent)
    for tracking,identity,measurement,events in args.batch:
        paths=tuple(map(Path,(tracking,identity,measurement,events)))
        for sid in read(paths[2]/"run_config.json")["selected_shot_ids"]:
            if sid not in shots or sid in REUSE or sid in assignments:
                raise ValueError(f"Duplicate or out-of-scope measurement shot: {sid}")
            assignments[sid]=paths
    event_runs={Path(legacy_stage05["events"]),*(paths[3] for paths in assignments.values())}
    event_counts={p:Counter(r["track_id"] for r in read(p/"performance_events.json")["events"])
                  for p in event_runs}
    output_shots,timeline=[],[]
    for shot in shots:
        span=bounds[shot]
        entry={"shot_id":shot,"stage01_temporal":span,"visible_character_candidates":contexts[shot],
               "reported_faces":[], "tracks":[],"focus_note":FOCUS.get(shot)}
        records=[]
        if shot in REUSE:
            tid=shot+"_track_001"
            dst=root/shot/tid
            dst.mkdir(parents=True)
            for name in ("gaze_states.json","motion_events.json","selected_gaze_frames.json","eye_motion_review.png",
                         "review.md","evidence_frames.parquet"):
                copy2(args.reuse/shot/name,dst/name)
            copy2(args.reuse_positions/shot/"eye_position_frames.parquet",dst/"eye_position_frames.parquet")
            result,records=summarize_track(shot,tid,dst,span)
            result.update(origin="reused_v1.4",reused_path=str(args.reuse/shot),
                          identity_namespace=legacy_stage05["identity"],
                          stage05_event_count=event_counts[Path(legacy_stage05["events"])][tid])
            entry.update(status="reused_with_qc",boundary_status="experimental_model_reviewed_pending_human",
                         stage05=legacy_stage05)
            entry["reported_faces"]=[t for t in corrected_tracks if t["shot_id"]==shot]
            entry["tracks"].append(result)
        elif shot in assignments:
            tracking,identity,measurement,events=assignments[shot]
            tracked=[r for r in read(tracking/"face_tracks.json")["tracks"] if r["shot_id"]==shot]
            identities=read(identity/"identity_clusters.json")["clusters"]
            cluster={tid:c for c in identities for tid in c["track_ids"]}
            measured={r["track_id"]:r for r in read(measurement/"track_measurements.json")["tracks"]
                      if r["shot_id"]==shot and r["extraction_mode"]=="static"}
            entry.update(boundary_status="stage01_original",stage05={"tracking":str(tracking),"identity":str(identity),
                         "measurement":str(measurement),"events":str(events)})
            entry["reported_faces"]=tracked
            meta=read(tracking/"run_config.json")
            all_frames=pq.read_table(measurement/"frame_features.parquet",columns=[
                "global_frame_index","movie_timestamp_sec"],
                filters=[("shot_id","=",shot),("extraction_mode","=","static")]).to_pylist()
            pictures=decode_review_frames(Path(meta["video"]["path"]),meta["video"],all_frames) if all_frames else {}
            schema=pq.ParquetFile(measurement/"frame_features.parquet").schema.names
            au=[k for k in schema if k.startswith("AU") and k not in {"AU45_c","AU45_r"}]
            for t in tracked:
                tid=t["track_id"]
                c=cluster.get(tid,{})
                if tid not in measured:
                    entry["tracks"].append({"track_id":tid,"status":"unavailable","reason":"insufficient_confirmed_track",
                                            "identity_cluster_id":c.get("cluster_id"),"identity_status":c.get("status"),
                                            "missing_intervals":[{"modality":"all","start_s":span["start"],"end_s":span["end"],
                                                                  "reason":"insufficient_confirmed_track"}]})
                    continue
                intermediate=root/"stage06_2_numeric"/shot/tid
                dst=root/shot/tid
                dst.parent.mkdir(parents=True,exist_ok=True)
                local=SimpleNamespace(shot_id=shot,track_id=tid,model=args.model,stage04=args.stage04,sequences=args.sequences,
                    tracking_run=tracking,identity_run=identity,measurement_run=measurement,event_run=events,
                    track_output=intermediate,allow_partial=True,au_columns=au)
                mp.run(local,pictures=pictures)
                run_shot(intermediate,dst,shot,tid)
                for name in ("eye_position_frames.parquet","eye_landmark_overlay.png"):
                    copy2(intermediate/name,dst/name)
                result,items=summarize_track(shot,tid,dst,span)
                result["origin"]="new_measurement"
                result["identity_namespace"]=str(identity)
                result["stage05_event_count"]=event_counts[events][tid]
                result["timing_s"]=read(intermediate/"gaze_states.json")["timing_s"]
                entry["tracks"].append(result)
                records.extend(items)
            pictures.clear()
            usable=[t for t in entry["tracks"] if t["status"]!="unavailable"]
            entry["status"]=("unavailable" if not usable else
                             "partial" if any(t["status"]!="processed_with_qc" for t in entry["tracks"])
                             else "processed_with_qc")
        else:
            entry.update(status="unavailable",boundary_status="stage01_original",reason="no_completed_stage05_batch")
        for item in records:
            track=next(t for t in entry["tracks"] if t["track_id"]==item["track_id"])
            item.update(identity_cluster_id=track["identity_cluster_id"],person_candidate=track["person_candidate"],
                        boundary_status=entry["boundary_status"],identity_status=track["identity_status"])
        timeline.extend(records)
        output_shots.append(entry)
    timeline.sort(key=lambda e:(e["start_s"],e["end_s"],e["shot_id"],e["track_id"]))
    write_json(root/"sequence_gaze_timeline.json",{"schema_version":"stage06.2.sequence_observations.v1","movie_id":MOVIE,
               "sequence_id":SEQUENCE,"sequence_start_s":seq.get("start_seconds"),"sequence_end_s":seq.get("end_seconds"),
               "human_verified":False,"shots":output_shots,"timeline":timeline,
               "limitations":["Events computed separately per Shot and Track.",
               "Reused corrected boundaries may overlap adjacent Stage01 frames; no cross-cut merging or interpolation."]})
    write_summary(root,output_shots)


def write_summary(root,shots):
    with (root/"sequence_summary.md").open("x",encoding="utf-8") as f:
        f.write("# Bugonia seq_006 · Stage05 / Stage06.2\n\n"
                "每行是一个 Track；候选身份均未人工确认。相同编号的身份聚类只在各自 Stage05.3 批次内有效。"
                "旧纠偏边界和相邻 Stage01 边界可能有一帧重叠，各镜头独立计算。\n\n"
                "| Shot | Track／状态 | 身份 | 眼位左／右 | OpenFace 原始 Pose／Gaze | 分析 Gaze | Stage05.5 候选 | 稳定 Gaze／Shift／Weak／Unknown | 代表帧／覆盖窗 |\n"
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for s in shots:
            if not s["tracks"]:
                f.write(f"| {s['shot_id']} | unavailable | — | — | — | — | — | — | — |\n")
            for t in s["tracks"]:
                v=t.get("stats")
                if not v:
                    f.write(f"| {s['shot_id']} | {t['track_id']}／{t['reason']} | {t.get('identity_status')} | — | — | — | — | — | — |\n")
                    continue
                m=v["modalities"]
                of=t["source_openface"]
                counts=[v["states"].get("stable_gaze_candidate",0),
                        *[v["events"].get(k,0) for k in ("eye_shift_candidate","weak_eye_motion_candidate","unknown")]]
                f.write(f"| {s['shot_id']} | {t['track_id']}／{t['status']} | {t['person_candidate']}／{t['identity_cluster_id']} "
                        f"| {m['eye_left']}/{m['eye_right']}（共 {v['frames']}） | {of['pose']}/{of['gaze']} "
                        f"| {m['gaze_direction']} | {t['stage05_event_count']} | {'/'.join(map(str,counts))} "
                        f"| {v['picks']}／{v['bins_covered']}/{v['bins']} |\n")
        f.write("\n## 逐镜头覆盖\n\n")
        f.write("| Shot | 总状态 | 跟踪条数 | 有数值证据 | 身份未解析 | 彻底不可用／未测 |\n"
                "| --- | --- | ---: | ---: | ---: | ---: |\n")
        for s in shots:
            counts=Counter(t["status"] for t in s["tracks"])
            f.write(f"| {s['shot_id']} | {s['status']} | {len(s['reported_faces'])} "
                    f"| {len(s['tracks'])-counts['unavailable']} | {counts['partial_identity_unknown']} "
                    f"| {counts['unavailable']} |\n")
        f.write("\n## 重点画面和可见性\n\n")
        for s in shots:
            if s["focus_note"]:
                f.write(f"- {s['shot_id']}：{s['focus_note']}\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("stage04","sequences","corrected","reuse","reuse-positions","model","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--batch",action="append",nargs=4,default=[],metavar=("TRACKING","IDENTITY","MEASUREMENT","EVENTS"))
    run(parser.parse_args())


if __name__=="__main__":
    main()
