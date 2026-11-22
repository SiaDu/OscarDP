"""Select and decode interval-internal review frames using the saved movie clock."""

from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from oscardp.shots.transnet import sha256_file
from oscardp.stage06_2.review import decode_review_frames

WINDOW_217 = (1475.391,1476.058)
SOURCE = "original_movie_ffmpeg_pts_verified"


def choose_intervals(before, after, transitions):
    picked = {}

    def add(record, reason):
        picked.setdefault(record["interval_id"],[]).append(reason)

    def spread(items, n, label):
        if n==1:
            indices = [len(items)//2] if items else []
        elif items:
            indices = list(dict.fromkeys(round((i+.5)*len(items)/min(n,len(items))-.5) for i in range(min(n,len(items)))))
        else:
            indices = []
        for i in indices:
            add(items[i],label)

    for old,new in zip(before,after,strict=True):
        if old["shot_id"]=="shot_000212" and old["selected_target"]!="UNKNOWN":
            add(new,"baseline_unique_person")
        if new["shot_id"]=="shot_000217" and new["start_s"]<WINDOW_217[1] and new["end_s"]>WINDOW_217[0]:
            add(new,"217_relative_eye_motion")
        if any(c["target"] in {"AWAY","OTHER"} for c in new["target_candidates"]):
            add(new,"nonperson_candidate")
        if new["comparison"]["orientation_filter_changes_prediction"]:
            add(new,"orientation_filter_effect")
    by_id = {r["interval_id"]:r for r in after}
    for transition in transitions:
        if transition["type"]=="target_switch_candidate":
            for key in ("from_interval_id","to_interval_id"):
                add(by_id[transition[key]],"neighbor_of_target_switch")
    for number,n in ((217,1),(219,3),(221,3)):
        spread([r for r in after if r["shot_id"]==f"shot_{number:06d}" and r["direction_source"]=="gaze"
                and r["interval_id"] not in picked],n,"matching_closeup_view")
    spread([r for r in after if r["track_id"]=="shot_000222_track_002" and r["direction_source"]=="head_only"],
           4,"head_only_ambiguity")
    spread([r for r in after if r["shot_id"]=="shot_000222" and r["direction_source"]=="gaze"],
           1,"coframe_gaze")
    spread([r for r in after if r["shot_id"]=="shot_000223" and r["direction_source"]=="gaze"
            and {"TEDDY","DON"}<={c["target"] for c in r["target_candidates"]}],3,"offscreen_person_ambiguity")
    for number in (211,218,223):
        spread([r for r in after if r["shot_id"]==f"shot_{number:06d}" and r["evidence_level"]=="insufficient"
                and r["evidence"]["measured_frames"]>=3],1,"insufficient_evidence")
    return [(r,picked[r["interval_id"]]) for r in after if r["interval_id"] in picked]


def interval_samples(record, frames):
    inside = [f for f in frames if f["shot_id"]==record["shot_id"] and f["track_id"]==record["track_id"]
              and record["start_frame"]<=f["global_frame_index"]<record["end_frame"]
              and record["start_s"]<=f["movie_timestamp_sec"]<record["end_s"]]
    if not inside:
        return []
    direction = record["direction_source"]
    def useful(f):
        if direction=="gaze":
            return f["gaze_direction_status"]["usable"]
        if direction=="head_only":
            return f["head_pose_status"]["usable"]
        return (f["gaze_direction_status"]["usable"] or f["head_pose_status"]["usable"]
                or any(x["usable"] for x in f["eye_position_status"].values()))
    valid = [f for f in inside if useful(f)]
    pool = sorted(valid or inside,key=lambda f:f["movie_timestamp_sec"])
    midpoint = (record["start_s"]+record["end_s"])/2
    roles = (("near_start",pool[0]),
             ("midpoint",min(pool,key=lambda f:abs(f["movie_timestamp_sec"]-midpoint))),
             ("near_end",pool[-1]))
    result = {}
    for role,f in roles:
        n = f["global_frame_index"]
        value = result.setdefault(n,{"interval_id":record["interval_id"],"shot_id":record["shot_id"],
                                      "track_id":record["track_id"],"movie_timestamp_sec":f["movie_timestamp_sec"],
                                      "global_frame_index":n,"within_interval":True,"source":SOURCE,
                                      "has_usable_review_signal":useful(f),"roles":[]})
        value["roles"].append(role)
    return list(result.values())


def publish_aligned(output, review, qc, video):
    destination = output/"review_contact_sheets"
    destination.mkdir()
    image_dir = destination/"frames"
    image_dir.mkdir()
    by_track = defaultdict(list)
    for f in qc:
        by_track[f["track_id"]].append(f)
    items = []
    for record, reasons in review:
        rows = interval_samples(record,by_track[record["track_id"]])
        items.append({**{k:record[k] for k in ("interval_id","shot_id","track_id","character","start_s","end_s",
                                               "selected_target","target_candidates","evidence_level","uncertainty_reason")},
                      "review_reasons":reasons,"representative_frames":rows})
    samples = [r for i in items for r in i["representative_frames"]]
    images = decode_review_frames(Path(video["path"]),video,samples) if samples else {}
    for r in samples:
        n = r["global_frame_index"]
        path = image_dir/f"frame_{n}.jpg"
        if not path.exists():
            images[n].save(path,quality=93)
        r.update(file=str(path.relative_to(output)),sha256=sha256_file(path))
    for start in range(0,len(items),4):
        section = items[start:start+4]
        panel = Image.new("RGB",(1440,len(section)*354),"#16212c")
        draw = ImageDraw.Draw(panel)
        path = destination/f"intervals_{start//4+1:02d}.jpg"
        for row,item in enumerate(section):
            item["contact_sheet"] = str(path.relative_to(output))
            y = row*354
            targets = " / ".join(c["target"] for c in item["target_candidates"])
            draw.text((8,y+4),f'{item["interval_id"]}   {item["character"]}   {targets}   [{item["evidence_level"]}]',fill="white")
            for col,f in enumerate(item["representative_frames"]):
                x = 480*col
                image = images[f["global_frame_index"]].copy()
                image.thumbnail((474,316))
                draw.text((x+6,y+23),f'{f["movie_timestamp_sec"]:.6f}s  f{f["global_frame_index"]}  {"/".join(f["roles"])}',fill="#d8e5fa")
                panel.paste(image,(x+3,y+38))
        panel.save(path,quality=95)
    return items
