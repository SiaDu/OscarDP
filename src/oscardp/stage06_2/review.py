"""Plot measurements and extract a few PTS-checked original frames for review."""

from collections import Counter
from pathlib import Path
import json
import re
import subprocess
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .geometry import angular_change, direction_angles, unit
from .inputs import EYES
from .temporal import BINOCULAR_LIMIT_DEG, valid_blocks


def plot(rows, events, output):
    times = np.array([r["movie_timestamp_sec"] for r in rows])
    end = rows[-1]["source_next_timestamp_sec"]
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True, layout="constrained")
    specs = [
        ("head_direction", "Head forward (camera axes)", "#1c6599"),
        ("gaze_direction", "Mean gaze (camera axes)", "#176e4d"),
        ("eye_relative_direction", "Mean gaze (head-local axes)", "#844b90"),
    ]
    for ax, (field, title, color) in zip(axes[:3], specs, strict=True):
        for key, suffix, alpha in [(field, "", 1.), (field+"_qc_rejected", " · QC rejected", .48)]:
            mask = [r.get(key) is not None for r in rows]
            if not any(mask):
                continue
            for i, indices in enumerate(valid_blocks(rows, mask)):
                angles = np.array([direction_angles(rows[j][key]) for j in indices])
                for axis, name, style in [(0, "Yaw", "-"), (1, "Pitch", "--")]:
                    ax.plot(times[indices], angles[:, axis], style, color=color, alpha=alpha,
                            label=name+suffix if i == 0 else None)
        ax.set_title(title, loc="left")
        ax.set_ylabel("deg")
        ax.grid(alpha=.18)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", ncol=2, fontsize=8)

    qc = axes[-1]
    values = np.array([r["binocular_disagreement_deg"] for r in rows], dtype=float)
    for i, block in enumerate(valid_blocks(rows, np.isfinite(values))):
        qc.plot(times[block], values[block], color="#91452c", label="Left/right ray angle" if i == 0 else None)
    qc.axhline(BINOCULAR_LIMIT_DEG, color="#91452c", linestyle=":", label="Agreement gate")
    qc.set_title("Binocular QC", loc="left")
    qc.set_ylabel("deg")
    qc.set_xlabel("Movie global time (s)  |  gray: unusable eyes  ·  red: closure candidate  ·  blue: movement")
    qc.legend(loc="upper right", fontsize=8)
    qc.ticklabel_format(useOffset=False, axis="x")
    axes[-1].set_xlim(times[0], end)
    for ax in axes:
        for indices in valid_blocks(rows, [not r["eye_relative_usable"] for r in rows]):
            ax.axvspan(times[indices[0]], rows[indices[-1]]["source_next_timestamp_sec"],
                       color="#858585", alpha=.11, zorder=0)
        for indices in valid_blocks(rows, [bool(r["closure_event_ids"]) for r in rows]):
            ax.axvspan(times[indices[0]], rows[indices[-1]]["source_next_timestamp_sec"],
                       color="#d76753", alpha=.25, zorder=0)
        for event in events:
            ax.axvspan(event["start_s"], event["end_s"], color="#4f93dc", alpha=.20, zorder=0)
    fig.suptitle(f"{rows[0]['shot_id']}  ·  OpenFace direction estimates")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def frame_pairs(rows, events):
    by_frame = {r["global_frame_index"]: r for r in rows}
    pairs = [(e["event_id"], by_frame[e["start_frame"]], by_frame[e["end_frame"]-1]) for e in events[:4]]
    valid = [(i, r["binocular_disagreement_deg"]) for i, r in enumerate(rows)
             if r["binocular_disagreement_deg"] is not None]
    if valid:
        apex, _ = max(valid, key=lambda item: item[1])
        a, b = max(0, apex-1), min(len(rows)-1, apex+1)
        pairs.append(("Peak binocular disagreement ±1 frame", rows[a], rows[b]))
    return pairs


def extract_review_frames(video, video_record, rows, events, output):
    stat = video.stat()
    if (stat.st_size, stat.st_mtime_ns) != (video_record["size"], video_record["mtime_ns"]):
        raise ValueError("Review video does not match Stage05 provenance")
    pairs = frame_pairs(rows, events)
    if not pairs:
        return []
    origin = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=start_time",
         "-of", "json", str(video)], text=True)
    origin = float(json.loads(origin)["streams"][0]["start_time"])
    targets = {r["global_frame_index"]: r for _, a, b in pairs for r in (a, b)}
    ordered = sorted(targets.values(), key=lambda r: r["movie_timestamp_sec"])
    times = [r["movie_timestamp_sec"] + origin for r in ordered]
    expression = "+".join(f"lt(abs(t-{t:.9f}),0.002)" for t in times)
    with TemporaryDirectory(prefix="oscardp-gaze-") as temp:
        pattern = str(Path(temp)/"%03d.png")
        command = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", "-threads", "4",
                   "-ss", f"{max(0., ordered[0]['movie_timestamp_sec'] - 1):.6f}", "-copyts",
                   "-noautorotate", "-i", str(video), "-map", "0:v:0", "-an", "-sn",
                   "-vf", f"select='{expression}',settb=AVTB,showinfo", "-fps_mode", "passthrough",
                   "-frames:v", str(len(ordered)), "-threads", "2", pattern]
        decoded = subprocess.run(command, capture_output=True, text=True, check=True)
        pts = [int(n)/1_000_000 for n in re.findall(r"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:", decoded.stderr)]
        if len(pts) != len(times) or any(abs(a-b) > .002 for a, b in zip(pts, times, strict=True)):
            raise ValueError("Review frame PTS mismatch")
        pictures = {}
        for i, row in enumerate(ordered, 1):
            # Decode first, then make a single fixed crop per pair below.
            with Image.open(Path(temp)/f"{i:03d}.png") as frame:
                pictures[row["global_frame_index"]] = frame.convert("RGB")
    sheet = Image.new("RGB", (1000, 285*len(pairs)), "#131b24")
    draw = ImageDraw.Draw(sheet)
    for i, (title, first, last) in enumerate(pairs):
        rows_pair = (first, last)
        y = 285*i
        draw.text((12, y+4), title, fill="white")
        boxes = [r["face_bbox_xywh"] for r in rows_pair if r["face_bbox_xywh"]]
        for j, row in enumerate(rows_pair):
            index = row["global_frame_index"]
            frame = pictures[index]
            if boxes:
                frame = frame.crop((max(0, int(min(a for a, _, _, _ in boxes))-30),
                                    max(0, int(min(b for _, b, _, _ in boxes))-60),
                                    min(frame.width, int(max(a+w for a, _, w, _ in boxes))+30),
                                    min(frame.height, int(max(b+h*.62 for _, b, _, h in boxes))+30)))
            frame.thumbnail((490, 235))
            x = j*500
            sheet.paste(frame, (x+(500-frame.width)//2, y+39))
            draw.text((x+12, y+20), f"Frame {index}  |  {row['movie_timestamp_sec']:.6f} s", fill="#c9dae9")
    sheet.save(output)
    return [{"label": title, "frames": [a["global_frame_index"], b["global_frame_index"]],
             "times_s": [a["movie_timestamp_sec"], b["movie_timestamp_sec"]]} for title, a, b in pairs]


def qc_summary(rows):
    angles = [r["binocular_disagreement_deg"] for r in rows if r["binocular_disagreement_deg"] is not None]
    per_eye_steps = [[], []]
    mask = [r["binocular_disagreement_deg"] is not None for r in rows]
    for block in valid_blocks(rows, mask):
        for a, b in zip(block, block[1:]):
            for eye, names in enumerate(EYES):
                per_eye_steps[eye].append(angular_change(unit([rows[a][k] for k in names]),
                                                        unit([rows[b][k] for k in names])))
    return {
        "frames": len(rows), "source_head_valid": sum(r["head_pose_valid"] for r in rows),
        "source_gaze_valid": sum(r["gaze_valid"] for r in rows),
        "tracking_usable": sum(r["tracking_usable"] for r in rows),
        "head_usable": sum(r["head_usable"] for r in rows),
        "camera_gaze_usable": sum(r["camera_gaze_usable"] for r in rows),
        "eye_relative_usable": sum(r["eye_relative_usable"] for r in rows),
        "qc_counts": dict(Counter(flag for r in rows for flag in r["qc_flags"])),
        "binocular_disagreement_deg_min_median_p95_max": (
            np.percentile(angles, [0, 50, 95, 100]).tolist() if angles else None),
        "per_eye_adjacent_steps_deg_median_p95_max": [
            np.percentile(v, [50, 95, 100]).tolist() if v else None for v in per_eye_steps],
    }


def markdown(rows, measurement, coverage, events, qc, files, pairs, video_error=None):
    r = rows[0]
    n = qc["frames"]
    lines = [
        f"# Stage06.2 v1.0 · {r['shot_id']}", "",
        f"- Track：`{r['track_id']}`，`{r['identity_cluster_id']}`。人物候选 {r['person'] or '未知'}；"
        "仅由同镜头单一 Track 与 Stage04 视觉候选对应，未人工核验。",
        f"- Stage05.4 主模式：{r['extraction_mode']}，{n} 帧；"
        f"原始 Pose / Gaze 可用 {qc['source_head_valid']} / {qc['source_gaze_valid']} 帧，"
        f"跟踪可用 {qc['tracking_usable']} 帧。",
        f"- Stage06.2 可用：头部 {qc['head_usable']}/{n}（{qc['head_usable']/n:.1%}），"
        f"综合相机 Gaze {qc['camera_gaze_usable']}/{n}（{qc['camera_gaze_usable']/n:.1%}），"
        f"综合 Eye-relative {qc['eye_relative_usable']}/{n}（{qc['eye_relative_usable']/n:.1%}）。",
        f"- 拒用原因：{qc['qc_counts']}。Stage05.5 partial closure 是几何候选，"
        "本轮保守地屏蔽其整个区间，不能解读为逐帧已证实闭眼。",
        f"- 分析区间：{r['movie_timestamp_sec']:.6f}–{rows[-1]['source_next_timestamp_sec']:.6f} s，"
        "电影全局时间、末端不含。沿用纠偏版 Stage05 边界，仍待人工复核。",
        "- OpenFace：相机 +X 向右、+Y 向下、+Z 远离摄影机；Pose 为弧度。"
        "使用原 C++ 的 Rx·Ry·Rz（不反号），head-local = Rᵀg；目视图的 yaw/pitch 为 "
        "atan2(x, −z)/atan2(y, −z) 的度数，分别在相机或头部坐标中定义。",
        "- Head change 为完整旋转的 SO(3) 角，包含 roll；Gaze 和 Eye-relative 为两个单位向量的夹角。"
        "图中 QC rejected 曲线用于检查测量，未用于分类。",
        "- OpenFace 使用启发式内参，头部姿态和眼球估计误差会同时进入分解。静帧和候选检测均不能验证细微独立眼动。",
        "", "## 连续性与双眼 QC", "",
    ]
    if qc["binocular_disagreement_deg_min_median_p95_max"]:
        stats = ", ".join(f"{v:.2f}°" for v in qc["binocular_disagreement_deg_min_median_p95_max"])
        lines += [f"- 双眼夹角最小／中位／P95／最大：{stats}。一致性阈值 {BINOCULAR_LIMIT_DEG:g}°。"]
    for side, v in zip(("左眼 OpenFace 0", "右眼 OpenFace 1"),
                       qc["per_eye_adjacent_steps_deg_median_p95_max"], strict=True):
        if v:
            lines.append(f"- {side} 相邻帧角变化中位／P95／最大：{', '.join(f'{x:.2f}°' for x in v)}。"
                         "包含真实运动与测量波动，快速跳变需对照画面检查。")
    lines += ["", "## 候选事件", "",
              "| ID | 电影时间 (s) | 类型 | Head / Gaze / Eye-relative (°) | 有效比例 | QC / Stage05 重叠 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for e in events:
        changes = [e[k+"_change_deg"] for k in ("head", "gaze", "eye_relative")]
        change = " / ".join(f"{v:.2f}" if v is not None else "—" for v in changes)
        notes = ", ".join(e["qc_flags"]+e["stage05_event_ids"])
        lines.append(f"| {e['event_id']} | {e['start_s']:.6f}–{e['end_s']:.6f} | {e['movement_type']} | "
                     f"{change} | {e['valid_frame_ratio']:.1%} | {notes or '—'} |")
    if not events:
        lines += ["", "没有通过最小持续时间和净角位移门槛的候选。"]
    if not any(e["movement_type"] == "eye_dominant" for e in events):
        lines += ["", "本轮没有可信的眼球独立运动候选。"]
    lines += ["", "## 原片对照", ""]
    if pairs:
        lines.append("已从同一原片按 PTS 抽取 [对照帧](review_frames.jpg)。表中的起止是对应画面 PTS；后图为区间最后一个包含的帧。")
        for p in pairs:
            lines.append(f"- {p['label']}：帧 {p['frames']}，{p['times_s'][0]:.6f}–{p['times_s'][1]:.6f} s。")
    else:
        lines.append(f"原片对照未生成。{video_error or '运行时未提供原片参数。'}")
    lines += ["", "## Stage05 实际覆盖", "",
              "| Shot | Stage05.2 corrected Track | Stage05.3 / 人物候选 | Stage05.4（Pose / Gaze / 总帧） | Stage05.5（AU / Eye / Head / Gaze） |",
              "| --- | --- | --- | --- | --- |"]
    for c in coverage:
        tracks = ", ".join(c["tracking"]) or "—"
        identity = ", ".join(c["identities"].values()) or "—"
        if c["character_candidate"]:
            identity += f" / {c['character_candidate']}"
        measured = ", ".join(f"{m['head']}/{m['gaze']}/{m['frames']}" for m in c["measurements"].values()) or "—"
        counts = c["event_counts"]
        count = "/".join(str(counts.get(key, 0)) for key in ("au_activation", "eye_closure_candidate",
                         "head_orientation_change", "camera_relative_gaze_change")) if counts else "—"
        lines.append(f"| {c['shot_id']} | {tracks} | {identity} | {measured} | {count} |")
    lines += ["", "## 输入溯源", "", "完整 SHA-256、覆盖、源事件与参数见 `gaze_events.json`。"]
    lines.extend(f"- `{name}`" for name in files)
    return "\n".join(lines) + "\n"
