"""Review clips and the reviewer guide for the Stage06.3 v1.4 gaze timeline."""

import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from .timeline import CERTAINTY, EXPRESSIVE, LOOK_AT

PAD_S = 0.5
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def clip_name(shot_id, character):
    return f"clips/{shot_id}_{character.lower()}.mp4"


def median_box(observations):
    boxes = [o["bbox_xywh"] for o in observations]
    return [statistics.median(b[i] for b in boxes) for i in range(4)] if boxes else None


def make_clip(video, path, start, end, labels, boxes):
    """Re-encode one review clip with movie-time and unit overlays (times relative to clip start)."""
    font = f"fontfile={FONT}:" if FONT.exists() else ""
    text = f"drawtext={font}fontsize=34:box=1:boxcolor=black@0.6:"
    filters = [f"{text}fontcolor=white:x=w-tw-20:y=20:text='%{{pts\\:flt\\:{start:.3f}}}'"]
    for (a, b), box in boxes:
        x, y, w, h = (round(v) for v in box)
        filters.append(f"drawbox=x={x}:y={y}:w={w}:h={h}:color=0x55efb1@1:t=4:enable='between(t,{a:.3f},{b:.3f})'")
    for (a, b), label, line in labels:
        color = "yellow" if line else "white"
        filters.append(f"{text}fontcolor={color}:x=20:y=h-{60 + 50 * line}:text='{label}':"
                       f"enable='between(t,{a:.3f},{b:.3f})'")
    filters.append("scale=960:-2")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.6f}", "-i", str(video),
                    "-t", f"{end - start:.6f}", "-map", "0:v:0", "-map", "0:a:0?", "-vf", ",".join(filters),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(path)], check=True)


def make_clips(review, video, shots, units, boxes):
    (review / "clips").mkdir()
    groups = defaultdict(list)
    for u in units:
        groups[u["shot_id"], u["character"]].append(u)
    for shot in shots:
        bounds = shot["stage01_temporal"]
        start, end = max(0., bounds["start"] - PAD_S), bounds["end"] + PAD_S
        for (sid, person), group in groups.items():
            if sid != shot["shot_id"]:
                continue
            windows = {u["unit_id"]: (u["start_s"] - start, u["end_s"] - start) for u in group}
            labels = [(windows[u["unit_id"]], f"{u['unit_id'].removeprefix(sid + '_')} {u['kind']}",
                       int(u["kind"] == "MOVE")) for u in group]
            frame_boxes = []
            for u in group:
                if u["kind"] in ("HOLD", "BLINK", "UNMEASURED"):
                    seen = [o for n in range(u["start_frame"], u["end_frame"])
                            for o in boxes.get((sid, n), []) if o["track_id"] == u["track_id"]]
                    box = median_box(seen)
                    if box:
                        frame_boxes.append((windows[u["unit_id"]], box))
            make_clip(video, review / clip_name(sid, person), start, end, labels, frame_boxes)


GUIDE = """# Bugonia seq_006 · 注视时间线审核（v1.4）

每个镜头、每个角色一条时间线。[timeline.csv](timeline.csv) 的一行表示"从 `start_s` 起看向哪里"，
一直持续到同一镜头、同一角色的下一行开始。只标眼睛看哪里、什么时候换。

## 怎么填

1. 打开片段。右上角是正片时间（秒），绿框是被审核的人，左下角是机器切分的行（白字状态，黄字运动）。
2. 每行填 `gaze`，可选 `certainty` 和 `notes`。
3. `start_s` 改成眼睛落到新目标的第一帧。不是真正换目标的行：删掉，或填和上一行相同（自动合并）。
4. 漏掉的切换：新增一行，填 `shot_id`、`character`、`start_s`、`gaze`。行的顺序不重要。
5. `machine_*`、`source_unit`、`v1_3_label` 只是提示，不用改。

## 词表

| 列 | 可填 |
| --- | --- |
| gaze | {look_at}（看向的人或物；OTHER 在 notes 写是什么）· {expressive}（无明确目标的表演性注视；CLOSED 指明显的持续闭眼）· BLINK（机器标的眨眼，视线视为延续）· UNCLEAR |
| certainty | {certainty} |

不标：头部运动、视线转移的时长、普通眨眼。它们由测量数据和你标的切换时刻自动计算。
落点在人的身体部位和地面之间无法区分时，按表演性注视填（如 DOWN）并在 notes 说明。

填完后运行：

```sh
uv --no-cache run --no-sync python -m oscardp.stage06_3.behavior_pilot evaluate --pilot {pilot}
```
"""


def write_guide(path, rows, pilot, clips):
    lines = [GUIDE.format(look_at=" / ".join(LOOK_AT), expressive=" / ".join(EXPRESSIVE),
                          certainty=" / ".join(CERTAINTY), pilot=pilot)]
    groups = defaultdict(list)
    for r in rows:
        groups[r["shot_id"], r["character"]].append(r)
    for (sid, person), group in groups.items():
        link = f"[片段]({clip_name(sid, person)})" if clips else "（未生成片段）"
        lines += [f"## {sid} · {person}?", "", link, "",
                  "| 候选 start_s | 机器 | 机器提示 |", "| --- | --- | --- |"]
        for r in group:
            lines.append(f"| {float(r['start_s']):.3f} | {r['machine_kind']} | {r['machine_hint']} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
