"""PTS-checked original frames and per-Track review cards."""

from collections import defaultdict
import json
from pathlib import Path
import textwrap

from PIL import Image,ImageDraw,ImageFont

from oscardp.shots.transnet import sha256_file
from oscardp.stage06_2.evidence_replay import write_json
from oscardp.stage06_2.review import decode_review_frames

SOURCE = "original_movie_ffmpeg_pts_verified"
BG = "#111b26"


def track_boxes(source):
    batches = {}
    result = {}
    for shot in source["shots"]:
        sid = shot["shot_id"]
        path = Path(shot["stage05"]["tracking"])/"face_detections.jsonl"
        if path not in batches:
            found=defaultdict(list)
            with path.open() as stream:
                for line in stream:
                    row=json.loads(line)
                    if row["track_id"] and row["detection_status"]=="observed":
                        found[row["shot_id"]].append(row)
            batches[path]=found
        for obs in batches[path][sid]:
            key=sid,obs["frame_index"]
            result.setdefault(key,[]).append(obs)
    return result


def choose_frame(rows, start, end, boxes, near=None, eye_crop=False):
    valid=[r for r in rows if start<=r["global_frame_index"]<end]
    if not valid:
        return None
    def key(r):
        frame=r["global_frame_index"]
        tracked=any(b["track_id"]==r["track_id"] for b in boxes.get((r["shot_id"],frame),[]))
        midpoint=(start+end-1)/2 if near is None else near
        ocular=any(e["usable"] for e in r["eye_position_status"].values())
        return (not tracked,eye_crop and not ocular,abs(frame-midpoint))
    return min(valid,key=key)


def banner(draw,text,xy=(12,8),fill="white",size=16):
    try:
        font=ImageFont.truetype("DejaVuSans.ttf",size)
    except OSError:
        font=ImageFont.load_default()
    draw.text(xy,text,fill=fill,font=font)


def annotated(image, frame, actor):
    im=image.copy()
    draw=ImageDraw.Draw(im)
    scale=max(1,im.width//360)
    for box in frame["face_boxes"]:
        x,y,w,h=box["bbox_xywh"]
        selected=box["track_id"]==frame["track_id"]
        color="#55efb1" if selected else "#f2bc76"
        draw.rectangle((x,y,x+w,y+h),outline=color,width=scale)
        label=("GAZER " if selected else "")+box["character"]+"? "+box["track_id"].rsplit("_",1)[-1]
        position=(max(0,x),max(2,y-9*scale))
        draw.rectangle((position[0],position[1],position[0]+len(label)*scale*5,position[1]+8*scale),fill=BG)
        banner(draw,label,position,color,max(12,5*scale))
    if not any(b["track_id"]==frame["track_id"] for b in frame["face_boxes"]):
        banner(draw,f"GAZER {actor}?  Track box unavailable",(10,12),size=24)
    return im


def sample_visual(output, source, mapping, states, events, qc):
    boxes = track_boxes(source)
    people={m["track_id"]:m["character"] for m in mapping["tracks"]}
    by_event={e["event_id"]:e for e in events}
    rows_per_state={}
    for s in states:
        tid=s["track_id"]
        anchor=s["gaze_evidence"]["anchor"] or s
        state_row=choose_frame(qc[tid],max(s["start_frame"],anchor["start_frame"]),
                               min(s["end_frame"],anchor["end_frame"]),boxes,eye_crop=s["direction_source"]=="eye_head")
        selected=[("stable" if s["look_direction"] else "qc_sample",state_row)]
        event=by_event.get(s["trigger_event"])
        if event:
            for before,role in ((True,"pre_motion"),(False,"post_motion")):
                a=event["pre_look" if before else "post_look"]
                start=a["start_frame"] if before else max(a["start_frame"],event["end_frame"])
                end=min(a["end_frame"],event["start_frame"]) if before else a["end_frame"]
                near=end-1 if before else start
                selected.append((role,choose_frame(qc[tid],start,end,boxes,near)))
        if s["shot_id"]=="shot_000217":
            for t in (1475.391,1475.724,1476.016):
                f=min(qc[tid],key=lambda r:abs(r["movie_timestamp_sec"]-t))
                if s["start_s"]-.001<=f["movie_timestamp_sec"]<s["end_s"]:
                    selected.append(("focus",f))
        if s["shot_id"]=="shot_000211":
            for t in (1432.765,1434.809):
                f=min(qc[tid],key=lambda r:abs(r["movie_timestamp_sec"]-t))
                if s["start_s"]<=f["movie_timestamp_sec"]<s["end_s"] and f is not state_row:
                    selected.append(("direction_qc_comparison",f))
        if s["shot_id"]=="shot_000212":
            for t in (1436.435,1437.31,1437.69,1439.23,1441.73):
                f=min(qc[tid],key=lambda r:abs(r["movie_timestamp_sec"]-t))
                if s["start_s"]<=f["movie_timestamp_sec"]<s["end_s"]:
                    selected.append(("v1_0_comparison",f))
        if s["shot_id"] in {"shot_000219","shot_000221"} and s["state_origin"]=="initial":
            candidates=[e for e in events if e["track_id"]==tid and e["pre_look"] and e["post_look"]
                        and e["source_event"]["type"] in {"weak_eye_motion_candidate","eye_shift_candidate"}]
            if candidates:
                e=max(candidates,key=lambda e:abs(e["source_event"].get("supporting_eye_delta_x",0.)))
                for side,label in (("pre_look","weak_motion_before"),("post_look","weak_motion_after")):
                    a=e[side]
                    selected.append((label,choose_frame(qc[tid],a["start_frame"],a["end_frame"],boxes)))
        entries=[]
        for role,f in selected:
            if f is None:
                continue
            n=f["global_frame_index"]
            sid=f["shot_id"]
            observed=[v for v in boxes.get((sid,n),[]) if abs(v["timestamp_sec"]-f["movie_timestamp_sec"])<=1e-6
                      and people.get(v["track_id"],"UNKNOWN")!="UNKNOWN"]
            record={k:f[k] for k in ("shot_id","track_id","global_frame_index","movie_timestamp_sec")}
            entries.append({**record,"role":role,"state_id":s["state_id"],"source":SOURCE,
                            "within_state":s["start_frame"]<=n<s["end_frame"] and s["start_s"]<=f["movie_timestamp_sec"]<s["end_s"],
                            "file":f"frames/frame_{n}.jpg",
                            "eye_crop_available":any(e["usable"] for e in f["eye_position_status"].values()),
                            "face_boxes":[{"track_id":v["track_id"],"character":people[v["track_id"]],
                                           "bbox_xywh":v["bbox_xywh"]} for v in observed]})
        if not entries or not entries[0]["within_state"]:
            raise ValueError(f"Missing in-state review sample: {s['state_id']}")
        rows_per_state[s["state_id"]]=entries
    all_rows=[f for ff in rows_per_state.values() for f in ff]
    video=mapping["video"]
    frames=decode_review_frames(Path(video["path"]),video,all_rows)
    (output/"frames").mkdir()
    for n,image in frames.items():
        image.save(output/f"frames/frame_{n}.jpg",quality=94)
    for f in all_rows:
        f["sha256"]=sha256_file(output/f["file"])
    make_inspection_sheets(output,states,rows_per_state)
    write_json(output/"visual_packet.json",
               {"source":SOURCE,"video":video,"states":rows_per_state,"unique_original_frames":len(frames)})


def make_inspection_sheets(output, states, by_state):
    # Pack a bounded number of panels per page while keeping state identifiers explicit.
    cards=[(s,f) for s in states for f in by_state[s["state_id"]]]
    for start in range(0,len(cards),9):
        page=cards[start:start+9]
        im=Image.new("RGB",(1500,355*((len(page)+2)//3)),BG)
        draw=ImageDraw.Draw(im)
        for i,(s,f) in enumerate(page):
            x,y=(i%3)*500,(i//3)*355
            banner(draw,f'{s["state_id"][5:]}  {s["character"]}?',(x+5,y+3),size=15)
            banner(draw,f'{f["role"]}  f{f["global_frame_index"]}  {f["movie_timestamp_sec"]:.4f}s',(x+5,y+24),size=13)
            with Image.open(output/f["file"]) as source:
                image=annotated(source,f,s["character"])
                image.thumbnail((493,304))
                im.paste(image,(x+4,y+45))
        im.save(output/f"inspect_{start//9+1:02d}.jpg",quality=94)


def track_review(output, states, frames, events, visual_dir):
    by_track=defaultdict(list)
    for s in states:
        by_track[s["track_id"]].append(s)
    by_event={e["event_id"]:e for e in events}
    for tid,groups in by_track.items():
        path=output/"review"/groups[0]["shot_id"]/f'track_{tid.rsplit("_",1)[-1]}'
        path.mkdir(parents=True)
        lines=[f"# {groups[0]['shot_id']} · {tid}","",
               "青色框为注视者候选，橙色框为同帧其他角色候选。XYZ 为该镜头相机坐标方向。",""]
        for s in groups:
            f=frames[s["state_id"]][0]
            direction=s["look_direction"]
            dstr=(f'{direction["meaning"]} XYZ '+", ".join(f"{x:+.2f}" for x in direction["vector"])
                  if direction else "unobservable")
            event=by_event.get(s["trigger_event"])
            trigger=(f'{event["source_event"]["type"]} at {event["start_s"]:.3f}s' if event else s["state_origin"])
            visual=s["visual_evidence"]
            visible=sorted(({b["character"] for b in f["face_boxes"] if b["track_id"]!=tid}
                            |set(visual["visible_characters"]))-{s["character"]})
            names=[v["target"] for v in s["possible_targets"]]
            offscreen=[v for v in names if v not in visible and v not in {"UNKNOWN","OTHER","AWAY"}
                       ]
            im=Image.new("RGB",(1040,700),BG)
            draw=ImageDraw.Draw(im)
            banner(draw,f'{s["shot_id"]} / {tid}  |  f{f["global_frame_index"]} at {f["movie_timestamp_sec"]:.3f}s',(15,10),size=19)
            banner(draw,f'GAZER {s["character"]}?   >   {dstr}',(15,40),size=18)
            banner(draw,f'POSSIBLE {", ".join(names)}    |    SELECTED {s["selected_target"]}',(15,67),"#66edb0",20)
            banner(draw,f'Trigger: {trigger}   |    {s["evidence_level"]}   |   {s["qc_status"]}',(15,97),size=15)
            banner(draw,f'In frame: {", ".join(visible) or "none identified"}  |  Offscreen person candidates: {", ".join(offscreen) or "none"}',
                   (15,121),"#f5c989",15)
            with Image.open(visual_dir/f["file"]) as source:
                image=annotated(source,f,s["character"])
                image.thumbnail((780,500))
                im.paste(image,(12,156))
                box=next((v["bbox_xywh"] for v in f["face_boxes"] if v["track_id"]==tid),None)
                if f["eye_crop_available"] and box and visual["eye_visibility"] not in {"unobservable","closed"}:
                    x,y,w,h=box
                    crop=source.crop((round(x+.08*w),round(y+.17*h),round(x+.92*w),round(y+.56*h)))
                    crop.thumbnail((232,140))
                    im.paste(crop,(797,191))
                    banner(draw,"EYE AREA",(797,165),"#66edb0",16)
                else:
                    banner(draw,"EYE AREA",(797,165),"#f5c989",16)
                    banner(draw,"unobservable / no usable fit",(797,192),size=13)
                banner(draw,f'{s["start_s"]:.3f}–{s["end_s"]:.3f}s',(797,355),size=14)
                banner(draw,"MODEL CANDIDATE",(797,380),size=13)
            name=f'{s["state_id"].rsplit("_",1)[-1]}.jpg'
            card=path/name
            im.save(card,quality=93)
            s["review_card"]=str(card.relative_to(output))
            o=s["visual_evidence"]["observation_zh"]
            lines.extend([f'## {s["start_s"]:.3f}–{s["end_s"]:.3f} s','',
                          f'[{s["character"]} · {s["selected_target"]}]({name})','',o,'',
                          '候选：'+', '.join(names)+'。',''])
            for v in visual["frames"][1:]+visual["cross_shot_frames"]:
                scope="镜头参照" if v["shot_id"]!=s["shot_id"] else v["role"]
                lines.append(f'- [{scope} · {v["shot_id"]} · {v["movie_timestamp_sec"]:.3f} s](../../../{v["file"]})')
            lines.append("")
        (path/"review.md").write_text("\n".join(lines),encoding="utf-8")
