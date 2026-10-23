"""Standalone quantitative plots; no events, emotion or gaze targets inferred."""
import csv

import cv2
import numpy as np
from PIL import Image, ImageDraw


def comparisons(root, rows, groups, main_mode="static"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tids = sorted({r["track_id"] for r in rows})
    by = {(tid,mode):[r for r in rows if r["track_id"]==tid and r["extraction_mode"]==mode]
          for tid in tids for mode in ["static","dynamic"]}
    results = []
    for tid in tids:
        for name in groups["au_intensity"]:
            result = {"track_id":tid,"au":name}
            for mode in ["static","dynamic"]:
                values = np.array([r["features"][name] for r in by[tid,mode] if r["reliability"]["au_valid"]],dtype=float)
                result[mode+"_valid_count"] = len(values)
                for key,value in {"min":values.min() if len(values) else None,"max":values.max() if len(values) else None,
                                  "median":np.median(values) if len(values) else None,"range":np.ptp(values) if len(values) else None,
                                  "first12_valid_median":np.median(values[:12]) if len(values) else None}.items():
                    result[mode+"_"+key] = float(value) if value is not None else None
            valid = [{r["global_frame_index"]:r["features"][name] for r in by[tid,m] if r["reliability"]["au_valid"]}
                     for m in ["static","dynamic"]]
            indices = sorted(valid[0].keys() & valid[1].keys())
            a,b = [np.array([v[i] for i in indices]) for v in valid]
            result["paired_valid_count"] = len(indices)
            result["paired_pearson"] = float(np.corrcoef(a,b)[0,1]) if len(a)>2 and a.std()>1e-8 and b.std()>1e-8 else None
            result["dynamic_minus_static_median"] = float(np.median(b-a)) if len(a) else None
            result["paired_rmse"] = float(np.sqrt(np.mean((b-a)**2))) if len(a) else None
            results.append(result)
    with (root/"comparisons/au_static_vs_dynamic.csv").open("x",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    ranked = sorted(groups["au_intensity"],key=lambda n:(-max((r[main_mode+"_range"] or 0) for r in results if r["au"]==n),n))
    def series(axis, data, name, validity, label=None):
        axis.plot([r["movie_timestamp_sec"] for r in data],
                  [r["features"].get(name) if r["reliability"][validity] else np.nan for r in data],lw=.9,label=label or name)
    def grid(names, validity, destination, overlay=False, ylim=None):
        fig,axes = plt.subplots(len(tids),len(names),figsize=(max(9,len(names)*3),len(tids)*2.2),squeeze=False,layout="constrained")
        for i,tid in enumerate(tids):
            for j,name in enumerate(names):
                axis = axes[i,j]
                for mode in (["static","dynamic"] if overlay else [main_mode]):
                    series(axis,by[tid,mode],name,validity,mode)
                axis.set_title(f"{tid}\n{name}",fontsize=8)
                axis.set_xlabel("Movie timestamp (s)",fontsize=7)
                axis.tick_params(labelsize=7)
                axis.grid(alpha=.2)
                if ylim: axis.set_ylim(*ylim)
                if overlay and i==0 and j==0: axis.legend(fontsize=7)
        fig.savefig(root/destination,dpi=150)
        plt.close(fig)
    grid(ranked[:6],"au_valid","previews/au_timeseries.png",ylim=(-.1,5.1))
    if by[tids[0],"dynamic"] and by[tids[0],"static"]:
        grid(ranked[:4],"au_valid","comparisons/au_static_vs_dynamic.png",overlay=True,ylim=(-.1,5.1))
    grid(["pose_Rx","pose_Ry","pose_Rz"],"head_pose_valid","previews/head_pose_timeseries.png")
    grid(["gaze_angle_x","gaze_angle_y"],"gaze_valid","previews/gaze_timeseries.png")
    return results,ranked


def association_sheet(root, paths, rows, modes):
    panels = []
    for sid,path in sorted(paths.items()):
        data = [r for r in rows if r["shot_id"]==sid and r["extraction_mode"]==modes[0]]
        selected = {1,len(data)//2+1,len(data)}
        # Sequential scratch decode only; global frame identity was independently verified.
        cap = cv2.VideoCapture(str(path))
        try:
            for local in range(1,len(data)+1):
                ok,image = cap.read()
                if not ok: raise ValueError("Cannot read verified scratch video for visual QC")
                if local not in selected: continue
                row = data[local-1]
                agreement = row["reliability"]["association"]
                box = agreement.get("yunet_bbox_xywh")
                if box:
                    x,y,w,h = [int(v) for v in box]
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
                for index in range(68):
                    x,y = row["features"].get(f"x_{index}"),row["features"].get(f"y_{index}")
                    if x is not None and y is not None:
                        cv2.circle(image,(int(x),int(y)),2,(0,255,0),-1)
                picture = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                picture.thumbnail((540,360))
                panel = Image.new("RGB",(540,400),"#181b20")
                panel.paste(picture,(0,40))
                draw = ImageDraw.Draw(panel)
                draw.text((5,3),f"{sid} global={row['global_frame_index']} local={local}",fill="white")
                draw.text((5,20),f"YuNet RED / OpenFace GREEN | IoU={agreement.get('landmark_bbox_iou',0):.3f}",fill="white")
                panels.append(panel)
        finally:
            cap.release()
    sheet = Image.new("RGB",(1620,400*((len(panels)+2)//3)),"#181b20")
    for i,panel in enumerate(panels): sheet.paste(panel,((i%3)*540,(i//3)*400))
    sheet.save(root/"previews/landmark_association_contact_sheet.jpg",quality=92)
