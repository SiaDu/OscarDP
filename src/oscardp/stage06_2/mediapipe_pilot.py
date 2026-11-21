"""Stage06.2 v1.3 MediaPipe eye positions on the existing Bugonia Stage05 Tracks."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq

from oscardp.shots.transnet import sha256_file
from .__main__ import frame_table
from .eye_position import LANDMARKS, attach_positions
from .inputs import load, read
from .iris_comparison import COLORS, JUMP, geometry, mark_jumps
from .review import decode_review_frames, plot
from .stable import analyze
from .temporal import detect, prepare_frames

SHOTS = ("shot_000217","shot_000211","shot_000214")
SIDES = ("image_left","image_right")
VERSION = "stage06.2.mediapipe_eye_position.v1.3"
PAD = .25


def roi(bbox,size):
    x,y,w,h = bbox
    return tuple(np.clip(np.r_[np.floor([x-PAD*w,y-PAD*h]),np.ceil([x+(1+PAD)*w,y+(1+PAD)*h])],
                         0,[*size,*size]).astype(int))


def map_landmarks(points, box):
    x,y,a,b = box
    return np.asarray(points)*[a-x,b-y]+[x,y]


def track_rows(args, rows):
    path = args.tracking_run/"face_detections.jsonl"
    provenance = read(args.measurement_run/"track_measurements.json")["source_provenance"]["tracking_files"]
    digest = sha256_file(path)
    if digest != provenance[path.name]["sha256"]:
        raise ValueError("Track observations do not match Stage05.4 provenance")
    index = {}
    with path.open() as stream:
        for line in stream:
            r=json.loads(line)
            if r["shot_id"]==args.shot_id and r["track_id"]==args.track_id:
                n=r["frame_index"]
                if n in index:
                    raise ValueError(f"Duplicate track frame {n}")
                index[n]=r
    for row in rows:
        match=index.get(row["global_frame_index"])
        if match and abs(match["timestamp_sec"]-row["movie_timestamp_sec"])>1e-6:
            raise ValueError("Track PTS mismatch")
    return index,{str(path.resolve()):digest}


def infer(rows,pictures,tracks,model):
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    records=[]
    options=FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)))
    with FaceLandmarker.create_from_options(options) as landmarker:
        for r in rows:
            n=r["global_frame_index"]
            frame=pictures[n]
            track=tracks.get(n)
            points=None
            failure="track_unavailable"
            crop_box=None
            if track and track["detection_status"]=="observed":
                crop_box=roi(track["bbox_xywh"],frame.size)
                x,y,a,b=crop_box
                if a-x>=32 and b-y>=32:
                    crop=frame.crop(crop_box)
                    mesh=landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.asarray(crop))).face_landmarks
                    failure="detection_failure"
                    if len(mesh)==1 and len(mesh[0])==478:
                        cand=map_landmarks([[p.x,p.y] for p in mesh[0]],crop_box)
                        anchors=np.asarray(track["landmarks5"][:2])
                        sep=np.linalg.norm(anchors[0]-anchors[1])
                        corners=(cand[[33,263]]+cand[[133,362]])/2
                        failure="track_alignment_failure"
                        if (np.isfinite(cand).all() and sep>10 and cand[468,0]<cand[473,0]
                                and np.max(np.linalg.norm(corners-anchors,axis=1))<=.4*sep):
                            points=cand
            record={"window":"shot","frame":n,"time_s":r["movie_timestamp_sec"],
                    "next_s":r["source_next_timestamp_sec"],"roi":crop_box}
            for i,side in enumerate(SIDES):
                comp=geometry(r,points,i)
                mp_eye=comp["mediapipe"]
                if points is None:
                    mp_eye["status"]=failure
                elif r["head_usable"] and abs(r["pose_Ry"])>.5 and mp_eye["status"]=="usable":
                    mp_eye["status"]="oblique_fit_unreliable"
                    mp_eye["xy"]=None
                record[side]=comp
            records.append(record)
    mark_jumps(records)
    return records


def reject_single_eye_jumps(records):
    """Keep raw diagnostic landmarks; mask an abrupt step with no fellow-eye support."""
    for r in records:
        eyes=[r[s]["mediapipe"] for s in SIDES]
        for i,eye in enumerate(eyes):
            other=eyes[1-i].get("step_eye_width")
            if eye["jump"] and (other is None or other<JUMP/2):
                eye["xy"]=None
                eye["status"]="unsupported_landmark_jump"


def attach(rows,records):
    for r,fit in zip(rows,records,strict=True):
        r["mediapipe_roi_xyxy"]=list(map(int,fit["roi"])) if fit["roi"] is not None else None
        for short,long in zip(("left","right"),SIDES,strict=True):
            eye=fit[long]["mediapipe"]
            of=fit[long]["openface"]
            xy=eye["xy"] or (None,None)
            r.update({f"{short}_eye_x":xy[0],f"{short}_eye_y":xy[1],f"{short}_eye_visibility":eye["status"],
                      f"{short}_eye_landmark_jump":eye["jump"],
                      f"{short}_mp_candidate_xy":eye.get("candidate_xy"),
                      f"{short}_mp_iris_px":eye.get("iris_px"), f"{short}_mp_eyelids_px":eye.get("eyelids_px"),
                      f"{short}_mp_centre_px":eye.get("centre_px"),
                      f"{short}_comparison_openface_xy":of["xy"]})


def overlay(records,pictures,path,shot):
    samples = ([min(records,key=lambda r:abs(r["time_s"]-t)) for t in
                (1468.008,1472.137,1472.262,1472.721,1475.391,1475.724,1476.016)]
               if shot=="shot_000217" else [records[i] for i in np.linspace(0,len(records)-1,6,dtype=int)])
    panel=Image.new("RGB",(950,30+150*len(samples)),"#19212a")
    draw=ImageDraw.Draw(panel)
    draw.text((10,6),"Amber: MediaPipe | Cyan: OpenFace | Image-left / Image-right",fill="white")
    for idx,r in enumerate(samples):
        y0=30+150*idx
        draw.text((10,y0),f"frame {r['frame']} | {r['time_s']:.6f} s",fill="white")
        frame=pictures[r["frame"]]
        for side_i,side in enumerate(SIDES):
            eyes=r[side]
            points=[p for eye in eyes.values() for lid in eye.get("eyelids_px",[]) for p in lid]
            if not points:
                continue
            x,y=np.maximum(0,np.floor(np.min(points,axis=0)-18)).astype(int)
            a,b=np.minimum(frame.size,np.ceil(np.max(points,axis=0)+18)).astype(int)
            if a<=x or b<=y:
                continue
            crop=frame.crop((x,y,a,b))
            crop.thumbnail((430,120))
            pen=ImageDraw.Draw(crop)
            def cv(p):
                return tuple((np.asarray(p)-[x,y])*np.asarray(crop.size)/[a-x,b-y])
            for model,item in eyes.items():
                if "centre_px" not in item:
                    continue
                for lid in item["eyelids_px"]:
                    pen.line([cv(p) for p in lid],fill=COLORS[model])
                line=[cv(p) for p in item["iris_px"]]
                pen.line([*line,line[0]],fill=COLORS[model],width=2)
                u,v=cv(item["centre_px"])
                pen.ellipse((u-3,v-3,u+3,v+3),fill=COLORS[model])
            panel.paste(crop,(10+side_i*465,y0+19))
            draw.text((10+side_i*465,y0+135),eyes["mediapipe"]["status"],fill="white")
    panel.save(path)


def stats(rows,states,motion,picks):
    left=[r["left_eye_x"] is not None for r in rows]
    right=[r["right_eye_x"] is not None for r in rows]
    return {"frames":len(rows),"left":sum(left),"right":sum(right),
            "both":sum(a and b for a,b in zip(left,right,strict=True)),
            "either":sum(a or b for a,b in zip(left,right,strict=True)),
            "states":dict(Counter(s["type"] for s in states)),
            "events":dict(Counter(e["type"] for e in motion)),"representatives":len(picks)}


def label_direction(motion,states):
    """A 2D shift never inherits a direction change from its head-pose event."""
    for event in motion:
        if event["type"]!="eye_shift_candidate":
            continue
        confirmed=any(e["type"]=="gaze_direction_shift_candidate" and
                      e["start_s"]<event["end_s"] and event["start_s"]<e["end_s"] for e in motion)
        event["gaze_direction_evidence"]="sustained_change_candidate" if confirmed else "gaze_direction_unknown"


def write(path,data):
    with path.open("x",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")


def run(args):
    start=perf_counter()
    args.movie_id,args.sequence_id,args.track_id="tt12300742","seq_006",args.shot_id+"_track_001"
    output=args.output_dir/args.shot_id
    if output.exists():
        raise ValueError(f"Output exists: {output}")
    rows,_,source,_,hashes,video=load(args,[*LANDMARKS,*args.au_columns])
    rows=prepare_frames(rows,source)
    tracks,track_hash=track_rows(args,rows)
    pictures=decode_review_frames(Path(video["path"]),video,rows)
    inference=perf_counter()
    records=infer(rows,pictures,tracks,args.model)
    elapsed_inference=perf_counter()-inference
    reject_single_eye_jumps(records)
    adjunct=detect(rows,source)
    baseline=[dict(r) for r in rows]
    attach_positions(baseline)
    bs,bm,bp=analyze(baseline,source,adjunct)
    attach(rows,records)
    states,motion,picks=analyze(rows,source,adjunct,allow_monocular=True)
    label_direction(motion,states)
    mp=stats(rows,states,motion,picks)
    mp["jumps"]={side:sum(r[side]["mediapipe"]["jump"] for r in records) for side in SIDES}
    mp["qc"]={side:dict(Counter(r[side]["mediapipe"]["status"] for r in records)) for side in SIDES}
    output.mkdir(parents=True,exist_ok=False)
    common={"schema_version":VERSION,"movie_id":args.movie_id,"sequence_id":args.sequence_id,
            "shot_id":args.shot_id,"track_id":args.track_id,"status":"model_candidate"}
    pq.write_table(frame_table(rows,VERSION),output/"eye_position_frames.parquet")
    write(output/"motion_events.json",{**common,"events":motion,"direction_motion_features":adjunct,
                                      "openface_baseline_eye_events":[e for e in bm if e["type"].startswith("eye_")]})
    write(output/"selected_gaze_frames.json",{**common,"frames":picks,"total_source_frames":len(rows)})
    plot(rows,[e for e in motion if e["type"] not in {"unknown","blink_candidate"}],
         output/"eye_motion_review.png",eye_states=states,eye_label="MediaPipe iris fit")
    overlay(records,pictures,output/"eye_landmark_overlay.png",args.shot_id)
    total=perf_counter()-start
    write(output/"gaze_states.json",{**common,"states":states,"mediapipe":mp,"openface_baseline":stats(baseline,bs,bm,bp),
                                     "input_sha256":{**hashes,**track_hash},"model_sha256":sha256_file(args.model),
                                     "mediapipe_version":version("mediapipe"),
                                     "timing_s":{"mediapipe_and_landmark_qc":elapsed_inference,"total":total}})
    with (output/"review.md").open("x",encoding="utf-8") as f:
        f.write(f"# Stage06.2 v1.3 · {args.shot_id}\n\n"
                "MediaPipe 提供虹膜及眼角；OpenFace 提供头姿、AU、相机相对 Gaze 及分解所需向量。"
                "青色为 OpenFace，橙色为 MediaPipe。可用性来自拟合、跟踪及可见度代理门槛。\n\n"
                f"- MediaPipe：{mp}\n- OpenFace v1.2 对照：{stats(baseline,bs,bm,bp)}\n"
                f"- 推理及 Landmark QC：{elapsed_inference:.2f} s；整体：{total:.2f} s。\n"
                "- 代表帧仅是待审清单，尚未调用其他模型。比较三镜头与观测片段见上级 comparison_summary.md。\n")
    print(args.shot_id,mp,round(total,2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot-id",choices=SHOTS,required=True)
    for name in ("stage04","sequences","tracking-run","identity-run","measurement-run","event-run","model","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    schema=pq.ParquetFile(args.measurement_run/"frame_features.parquet").schema.names
    args.au_columns=[k for k in schema if k.startswith("AU") and k not in {"AU45_r","AU45_c"}]
    run(args)


if __name__=="__main__":
    main()
