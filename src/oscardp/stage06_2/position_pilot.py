"""Write the v1.2 single-shot extension alongside the preserved legacy runs."""

from collections import Counter
import json
from pathlib import Path
import subprocess

import pyarrow.parquet as pq

from .eye_position import WINDOWS, attach_positions, overlays
from .review import decode_review_frames, plot
from .stable import analyze, POSITION_FIELDS


def publish(args, rows, source, adjunct, video, hashes, frame_table):
    attach_positions(rows)
    states, motion, picks = analyze(rows, source, adjunct)
    out = args.output_dir

    def write(name, **fields):
        with (out/name).open("x",encoding="utf-8") as stream:
            json.dump({"schema_version":"stage06.2.eye_position.v1.2",
                       "movie_id":args.movie_id,"sequence_id":args.sequence_id,
                       "shot_id":args.shot_id,"track_id":rows[0]["track_id"],
                       "status":"model_candidate",**fields},stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write("\n")

    sample_times = (1468,1471.637,1472.137,1472.263,1475.39,1475.72,1476.016)
    samples = [min(rows,key=lambda r:abs(r["movie_timestamp_sec"]-t)) for t in sample_times]
    overlay_status = "未请求原片叠图。"
    if args.review_frames:
        try:
            pics = decode_review_frames(Path(args.video or video["path"]),video,samples)
            overlays(samples,pics,out/"eye_landmark_overlay.png")
            overlay_status = "已按原片 PTS 核对并生成 [眼部叠图](eye_landmark_overlay.png)。"
        except (OSError,ValueError,subprocess.CalledProcessError) as exc:
            overlay_status = f"原片叠图未生成：{type(exc).__name__}。"
    pq.write_table(frame_table(rows,"stage06.2.eye_position.v1.2"),out/"eye_position_frames.parquet")
    write("gaze_states.json",states=states,input_sha256=hashes)
    write("motion_events.json",events=motion,direction_motion_features=adjunct)
    write("selected_gaze_frames.json",frames=picks,total_source_frames=len(rows))
    plot(rows, [e for e in motion if e["type"] not in {"unknown","blink_candidate"}],
         out/"eye_motion_review.png",eye_states=states)
    detail = []
    for start,end in WINDOWS:
        hits = lambda records: Counter(r["type"] for r in records if r["start_s"]<end and r["end_s"]>start)
        detail.append(f"- {start:.3f}–{end:.3f} s：重叠稳定区间 {dict(hits(states))}；主要事件 {dict(hits(motion))}。")
    with (out/"review.md").open("x",encoding="utf-8") as stream:
        stream.write("\n".join([
            "# Stage06.2 v1.2 · 眼部拟合与稳定方向复核","",
            "OpenFace 的每眼局部索引 0–7 是虹膜轮廓拟合点。GetPupilPosition 取这八点均值；"
            "这里使用其 2D 对应点。现有数据没有独立实测瞳孔中心。位置是拟合代理量，可被遮挡和透视误导。"
            "坐标以两眼角的连线为水平轴，x 向画面右、y 向下，均除以该眼角宽度。","",
            overlay_status,
            f"- 双眼拟合代理通过几何和可见度门槛：{sum(all(r[k] is not None for k in POSITION_FIELDS) for r in rows)}/{len(rows)} 帧。",
            f"- 稳定区间：{dict(Counter(s['type'] for s in states))}。",
            f"- 事件：{dict(Counter(e['type'] for e in motion))}。",
            f"- 代表帧：{len(picks)}/{len(rows)}；尚未调用其他模型，也未做耗时测试。",
            "- 保留源约 24 FPS。未做 8 FPS 下采样实验。所有状态与方向变化均为同镜头候选。",
            "- 方向稳定要求眼位稳定和双眼及综合相机向量稳定。方向段之间需有至少 6° 综合方向持续差、"
            "两眼变化均至少 4° 且变化相容。不跨闭合或无效帧连接。附加的头眼运动类别不参与主事件定性。",
            "", "## 指定片段",*detail,
            "- 人工观察记录：1475.391–1476.058 s 可见眼球相对头部运动，但观看位置似乎保持。"
            "这不是已核实的方向量或所看对象。建议按原速和慢放核对眼睑遮挡、虹膜拟合以及运动前后。",
            "- 1472 s 附近优先核对双眼叠图。拟合中心偏离虹膜时，即使输出通过几何门槛也应谨慎。",
            "",
        ]))
