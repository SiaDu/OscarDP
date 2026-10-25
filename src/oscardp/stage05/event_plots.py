"""Auditable original-movie-time plots, raw/filtered curves, candidates and masks."""
import os
import tempfile
from pathlib import Path

import numpy as np


def timelines(root,by,traces,events,eye_info,config,compare_dynamic):
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"oscardp-event-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tids = sorted({t for t,m in by})
    lookup = {}
    for t in traces: lookup.setdefault((t["track_id"],t["extraction_mode"],t["feature"]),[]).append(t)
    for name,typ in [("au","au_activation"),("eye_closure","eye_closure_candidate"),
                     ("head","head_orientation_change"),("gaze","camera_relative_gaze_change")]:
        fig,axes = plt.subplots(len(tids),1,figsize=(16,5*len(tids) if name=="au" else 3*len(tids)),squeeze=False,layout="constrained")
        for i,tid in enumerate(tids):
            axis = axes[i,0]
            rows = by[tid,"static"]
            times = [r["movie_timestamp_sec"] for r in rows]
            mask_name = {"au":"au_valid","eye_closure":"landmarks_2d_valid","head":"head_pose_valid","gaze":"gaze_valid"}[name]
            for row in rows:
                if not row[mask_name]: axis.axvspan(row["movie_timestamp_sec"],row["source_next_timestamp_sec"],color="red",alpha=.25)
            selected = [e for e in events if e["track_id"]==tid and e["event_type"]==typ]
            if name=="au":
                fields = sorted(n for n in rows[0] if n.startswith("AU") and n.endswith("_r"))
                for j,field in enumerate(fields):
                    data = lookup[tid,"static",field]
                    raw = [d["raw_values"][0]/6+j if d["raw_values"][0] is not None else np.nan for d in data]
                    processed = [d["processed_values"][0]/6+j if d["processed_values"] is not None else np.nan for d in data]
                    axis.plot(times,raw,color="gray",alpha=.5,lw=.6,label="raw intensity / 6 + lane" if j==0 else None)
                    axis.plot(times,processed,color="#1763a5",lw=.9,label="time-median / 6 + lane" if j==0 else None)
                    for e in selected:
                        if e["feature"]==field:
                            axis.fill_between([e["start_sec"],e["end_sec"]],j-.08,j+.85,color="orange",alpha=.25)
                axis.set_yticks(range(len(fields)),fields,fontsize=7)
            elif name=="eye_closure":
                data = lookup[tid,"static","eye_geometry"]
                reference = eye_info[tid]["reference_ear"]
                for j,label in enumerate(["image-left","image-right"]):
                    axis.plot(times,[d["raw_values"][j]/reference[j] if reference and d["raw_values"][j] is not None else np.nan for d in data],alpha=.35,label=label+" raw relative EAR")
                    axis.plot(times,[d["processed_values"][j] if d["processed_values"] is not None else np.nan for d in data],label=label+" processed relative EAR")
                for mode in (["static","dynamic"] if compare_dynamic else ["static"]):
                    axis.plot(times,[r["AU45_r"]/5 if r.get("AU45_r") is not None else np.nan for r in by[tid,mode]],"--",lw=.8,label=mode+" AU45_r / 5")
                    axis.plot(times,[r["AU45_c"] for r in by[tid,mode]],":",lw=.7,label=mode+" AU45_c")
                axis.axhline(config["eye"]["partial_ratio"],color="gray",ls=":",lw=.5)
                axis.set_ylabel("Relative EAR / AU45 scales (see legend)")
            else:
                field = "head_orientation" if name=="head" else "gaze_direction"
                data = lookup[tid,"static",field]
                labels = ["Rx","Ry","Rz"] if name=="head" else ["angle_x","angle_y"]
                for j,label in enumerate(labels):
                    line, = axis.plot(times,[d["raw_values"][j] for d in data],alpha=.25,lw=.7)
                    axis.plot(times,[d["processed_values"][j] if d["processed_values"] is not None else np.nan for d in data],color=line.get_color(),label=label+" filtered")
                axis.set_ylabel("Model orientation (rad)")
                second = axis.twinx()
                second.plot(times,[d["derived_value"] if d["derived_value"] is not None else np.nan for d in data],color="gray",lw=.6,alpha=.4)
                second.set_ylabel("Angular speed (rad/s)",fontsize=8)
            if name!="au":
                for e in selected:
                    axis.axvspan(e["start_sec"],e["end_sec"],color="orange",alpha=.2)
                    axis.text(e["start_sec"],.97,e["event_id"],transform=axis.get_xaxis_transform(),rotation=90,fontsize=6,va="top")
            for e in selected:
                if e["qc_flags"]: axis.axvline(e["start_sec"],color="red",alpha=.5,lw=.6)
            axis.set_title(f"{tid} | {typ} | static primary; orange=candidate, red=invalid/review",fontsize=9)
            axis.set_xlabel("Original movie timestamp (s)")
            axis.ticklabel_format(axis="x",style="plain",useOffset=False)
            axis.grid(alpha=.15)
            axis.legend(fontsize=6,loc="upper left",ncol=4)
        fig.savefig(root/f"previews/{name}_event_timelines.png",dpi=130)
        plt.close(fig)
