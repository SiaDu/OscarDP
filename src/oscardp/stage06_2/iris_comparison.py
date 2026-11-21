"""Same-source-frame OpenFace / MediaPipe iris comparison for Bugonia 217."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from oscardp.shots.transnet import sha256_file
from .eye_position import LANDMARKS, eye_points
from .inputs import load
from .review import decode_review_frames
from .temporal import prepare_frames

WINDOWS = {"reference": (1467.925,1468.092), "fit_error": (1472.096,1472.763),
           "visible_motion": (1475.391,1476.058)}
# Same physical eyes: OpenFace 0 is image-left in this shot, MediaPipe calls it right.
MP_EYES = (
    (468, (469,470,471,472), (33,246,161,160,159,158,157,173,133), (33,7,163,144,145,153,154,155,133)),
    (473, (474,475,476,477), (263,466,388,387,386,385,384,398,362), (263,249,390,373,374,380,381,382,362)),
)
COLORS = {"openface":"#38e2ff","mediapipe":"#ffb052"}
JUMP = .08


def measure(center, iris, lids, *, blocked=None):
    """Identical 2D eye-corner frame and aperture check for the two topologies."""
    center, iris = np.asarray(center), np.asarray(iris)
    lids = tuple(map(np.asarray, lids))
    if not np.isfinite(np.vstack((center, iris, *lids))).all():
        return {"xy":None, "status":"nonfinite_landmarks", "aperture":None, "jump":False}
    p,q = sorted([lids[0][0],lids[0][-1]], key=lambda v:v[0])
    w = np.linalg.norm(q-p)
    if w < 10:
        return {"xy":None,"status":"low_visibility","aperture":None,"jump":False}
    x = (q-p)/w
    basis = np.array([x,[-x[1],x[0]]])
    xy = basis @ (center-p)/w

    def edges(t):
        ans=[]
        for lid in lids:
            projected=(lid-p)@basis.T/w
            order=np.argsort(projected[:,0])
            ans.append(np.interp(t,projected[order,0],projected[order,1]))
        return sorted(ans)

    aperture = edges(.5)[1]-edges(.5)[0]
    lo,hi = edges(xy[0])
    status = blocked or ("low_visibility" if aperture < .12 else
                         "centre_outside_lids" if not (0 < xy[0] < 1 and lo <= xy[1] <= hi) else "usable")
    return {"xy":xy.tolist() if status=="usable" else None, "status":status, "aperture":float(aperture),
            "jump":False, "candidate_xy":xy.tolist(), "centre_px":center.tolist(),
            "iris_px":iris.tolist(), "eyelids_px":[v.tolist() for v in lids]}


def geometry(row, mp_points, eye):
    pts=eye_points(row,eye)
    closed="closure_candidate" if row["closure_event_ids"] else None
    of=measure(pts[:8].mean(axis=0),pts[:8],(pts[8:15],pts[[*range(14,20),8]]),
               blocked=closed or (None if row["tracking_usable"] and row["landmarks_2d_valid"] else "tracking_failure"))
    if mp_points is None:
        mp={"xy":None,"status":"detection_failure","aperture":None,"jump":False}
    else:
        c,iris,upper,lower=MP_EYES[eye]
        mp=measure(mp_points[c],mp_points[list(iris)],(mp_points[list(upper)],mp_points[list(lower)]),blocked=closed)
    return {"openface":of,"mediapipe":mp}


def mark_jumps(records):
    for a,b in zip(records,records[1:]):
        if (a["window"]!=b["window"] or b["frame"]!=a["frame"]+1
                or abs(b["time_s"]-a["next_s"])>1e-6 or b["time_s"]-a["time_s"]>.25):
            continue
        for eye in ("image_left","image_right"):
            for model in COLORS:
                p,q=a[eye][model]["xy"],b[eye][model]["xy"]
                if p is not None and q is not None:
                    step=float(np.linalg.norm(np.subtract(p,q)))
                    b[eye][model]["step_eye_width"]=step
                    b[eye][model]["jump"]=step>=JUMP


def overlay(records,pictures,path):
    desired=(1468.008,1472.137,1472.262,1472.722,1475.391,1475.724,1476.016)
    samples=[min(records,key=lambda r:abs(r["time_s"]-t)) for t in desired]
    panel=Image.new("RGB",(1020,25+155*len(samples)),"#17212a")
    draw=ImageDraw.Draw(panel)
    draw.text((10,6),"Cyan: OpenFace  |  Amber: MediaPipe  |  dot: centre  |  image-left / image-right",fill="white")
    for n,r in enumerate(samples):
        frame=pictures[r["frame"]]
        draw.text((10,25+155*n),f"{r['time_s']:.6f} s | frame {r['frame']} | {r['window']}",fill="white")
        for i,eye in enumerate(("image_left","image_right")):
            items=r[eye]
            points=[p for v in items.values() for lid in v.get("eyelids_px",[]) for p in lid]
            if not points:
                continue
            x,y=np.maximum(0,np.floor(np.min(points,axis=0)-22)).astype(int)
            a,b=np.minimum(frame.size,np.ceil(np.max(points,axis=0)+22)).astype(int)
            crop=frame.crop((x,y,a,b))
            crop=crop.resize((round((a-x)*125/(b-y)),125))
            pen=ImageDraw.Draw(crop)
            def project(p):
                return tuple((np.asarray(p)-[x,y])*np.asarray(crop.size)/[a-x,b-y])
            for model,v in items.items():
                if "centre_px" not in v:
                    continue
                color=COLORS[model]
                for lid in v["eyelids_px"]:
                    pen.line([project(p) for p in lid],fill=color,width=1)
                iris=[project(p) for p in v["iris_px"]]
                pen.line([*iris,iris[0]],fill=color,width=2)
                u,w=project(v["centre_px"])
                pen.ellipse((u-3,w-3,u+3,w+3),fill=color)
            panel.paste(crop,(i*510+10,n*155+45))
    panel.save(path)


def plot(records,path):
    fig,axes=plt.subplots(4,3,figsize=(14,8),layout="constrained",sharex="col",sharey="row")
    for col,window in enumerate(WINDOWS):
        rows=[r for r in records if r["window"]==window]
        time=[r["time_s"] for r in rows]
        for i,eye in enumerate(("image_left","image_right")):
            for j,axis in enumerate("xy"):
                ax=axes[2*i+j,col]
                if not j and not i:
                    ax.set_title(window)
                if not col:
                    ax.set_ylabel(f"{eye} {axis} / eye width")
                for k,model in enumerate(COLORS):
                    vals=[r[eye][model]["xy"][j] if r[eye][model]["xy"] is not None else np.nan for r in rows]
                    color=("#167d98","#bd6300")[k]
                    ax.plot(time,vals,color=color,marker=".",ms=3,label=model)
                    for t,r,v in zip(time,rows,vals,strict=True):
                        item=r[eye][model]
                        if item["jump"]:
                            ax.scatter(t,v,marker="^",color=color,s=37)
                        if item["status"]!="usable":
                            ax.scatter(t,.05+k*.08,marker="x" if item["status"]=="detection_failure" else "|",
                                       color=color,transform=ax.get_xaxis_transform())
                ax.grid(alpha=.18)
                if j and i:
                    ax.set_xlabel("Movie time (s)")
                    ax.ticklabel_format(axis="x",useOffset=False)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Same source frames  |  triangles: jumps  ·  x: detection failure  ·  lower ticks: unusable")
    fig.savefig(path,dpi=150)
    plt.close(fig)


def make_review(records):
    lines = ["# Stage06.2 · OpenFace 与 MediaPipe Iris 同帧对照", "",
             "MediaPipe 以 IMAGE 模式处理原片 PTS 核对的帧。两套模型各用自己的眼角和眼睑，"
             "在同一画面左右方向的局部眼角坐标系内归一化；共同屏蔽原 Stage05.5 眼闭合候选。"
             "可用门槛是几何检查，并非虹膜可见度或定位准确率的真值。", "",
             "| 区间 | 画面侧别 | 模型 | 可用／总帧 | 跳变数 | 首尾各三帧中位数之差 x／y |",
             "| --- | --- | --- | ---: | ---: | --- |"]
    for name in WINDOWS:
        subset = [r for r in records if r["window"] == name]
        for eye in ("image_left", "image_right"):
            for model in COLORS:
                entries = [r[eye][model] for r in subset]
                xy = [v["xy"] for v in entries]
                net = "—"
                if len(xy) >= 3 and all(v is not None for v in [*xy[:3], *xy[-3:]]):
                    change = np.median(xy[-3:],axis=0)-np.median(xy[:3],axis=0)
                    net = "／".join(f"{v:+.3f}" for v in change)
                lines.append(f"| {name} | {eye} | {model} | {sum(v is not None for v in xy)}/{len(xy)} "
                             f"| {sum(v['jump'] for v in entries)} | {net} |")
    lines.extend(["", "单位为该眼眼角宽度；相邻帧二维变化达到 0.08 记为跳变。"
                  "见 [同帧叠图](iris_comparison.png)、[眼位曲线](eye_position_comparison.png) 和 [逐帧结果](comparison.json)。",""])
    return "\n".join(lines)


def main():
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("stage04","sequences","tracking-run","identity-run","measurement-run","event-run","model","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    args.movie_id,args.sequence_id,args.shot_id,args.track_id=(
        "tt12300742","seq_006","shot_000217","shot_000217_track_001")
    if args.output_dir.exists():
        parser.error("Output directory already exists")
    rows,_,events,_,hashes,video=load(args,LANDMARKS)
    rows=prepare_frames(rows,events)
    selected=[(name,r) for name,(lo,hi) in WINDOWS.items() for r in rows
              if lo-.0005 <= r["movie_timestamp_sec"] <= hi+.0005]
    pictures=decode_review_frames(Path(video["path"]),video,[r for _,r in selected])
    records=[]
    options=FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)))
    with FaceLandmarker.create_from_options(options) as landmarker:
        for name,r in selected:
            frame=pictures[r["global_frame_index"]]
            result=landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.asarray(frame)))
            mesh=result.face_landmarks
            pts=None
            if len(mesh)==1 and len(mesh[0])==478:
                pts=np.array([[p.x*frame.width,p.y*frame.height] for p in mesh[0]])
                # Fail closed if the mesh points do not align with the measured face/eye ordering.
                centres=[eye_points(r,i)[[8,14]].mean(axis=0) for i in (0,1)]
                if (pts[468,0]>=pts[473,0] or any(not np.isfinite(c).all() or
                        np.linalg.norm(pts[j]-c)>.4*np.linalg.norm(centres[0]-centres[1])
                        for c,j in zip(centres,(468,473),strict=True))):
                    pts=None
            records.append({"window":name,"frame":r["global_frame_index"],
                            "time_s":r["movie_timestamp_sec"],"next_s":r["source_next_timestamp_sec"],
                            "head_yaw_rad":r["pose_Ry"],"image_left":geometry(r,pts,0),
                            "image_right":geometry(r,pts,1)})
    mark_jumps(records)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    overlay(records,pictures,args.output_dir/"iris_comparison.png")
    plot(records,args.output_dir/"eye_position_comparison.png")
    with (args.output_dir/"comparison.json").open("x",encoding="utf-8") as stream:
        json.dump({"schema_version":"stage06.2.iris_comparison.v1","movie_id":args.movie_id,
                   "sequence_id":args.sequence_id,"shot_id":args.shot_id,"track_id":args.track_id,
                   "mediapipe":version("mediapipe"),"model_sha256":sha256_file(args.model),
                   "source_sha256":hashes,"windows_s":WINDOWS,"jump_eye_width":JUMP,
                   "mode":"IMAGE, independent frames","frames":records},stream,indent=2,allow_nan=False)
    with (args.output_dir/"review.md").open("x",encoding="utf-8") as stream:
        stream.write(make_review(records))
    for name in WINDOWS:
        subset=[r for r in records if r["window"]==name]
        print(name,len(subset),{(eye,model):dict(Counter(r[eye][model]["status"] for r in subset))
                               for eye in ("image_left","image_right") for model in COLORS})


if __name__=="__main__":
    main()
