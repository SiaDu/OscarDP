"""Prediction comparisons and human review materials for Bugonia seq_006."""

import csv
from itertools import groupby
from shutil import copy2

from PIL import Image

from .hypotheses import STATE_KEYS, ranked_targets
from .hypothesis_evaluation import METHODS
from .look_review import annotated

METHOD_LABELS = ("Visual-only", "Geometry-only", "Fusion")
TEDDY_DON = {"TEDDY", "DON"}
REVIEW_STATES = tuple(
    f"shot_{shot:06d}_track_{track:03d}_look_{look:03d}"
    for shot, track, look in (
        (212, 1, 1), (212, 1, 3), (212, 1, 6),
        (214, 1, 2), (214, 1, 3), (215, 3, 2), (217, 1, 1), (218, 2, 2),
        (219, 1, 1), (220, 2, 1), (220, 2, 2), (221, 2, 1),
        (222, 2, 1), (222, 2, 3), (222, 2, 5), (222, 3, 2), (223, 3, 1), (223, 3, 2),
    )
)


def ranking(row):
    if not row["preferred_target"] and len(row["alternative_targets"]) == 1:
        return f"候选 {row['alternative_targets'][0]}"
    return " > ".join(" / ".join(tier) for tier in row["target_ranking"]) or "—"


def ranked_tiers(row):
    return tuple(frozenset(tier) for tier in row["target_ranking"])


def grouped_predictions(documents):
    return {
        rows[0]["state_id"]: dict(zip(METHODS, rows, strict=True))
        for rows in zip(*(documents[method]["predictions"] for method in METHODS), strict=True)
    }


def state_list(states):
    return "、".join(f"`{state}`" for state in states) or "无"


def write_comparison(output, states, predictions, frame_count, old_rows, old_source):
    lines = [
        "# Stage06.3 v1.3 · Gaze Target Hypothesis Evaluation", "",
        f"Bugonia `tt12300742` · `seq_006`。复用 {len(states)} 个冻结状态，"
        f"其中 {sum(s['look_direction'] is None for s in states)} 个缺少持续方向。"
        f"独立视觉检查覆盖 {frame_count} 张不同的原片帧，包含基于 Track 框的脸与眼区裁切。", "",
        "视觉检查使用原片与相邻镜头，输入无数值布局、几何结果或旧预测排序。请求已点明 212 的预期假设，"
        "因此该重点案例对预期目标未完全盲评。几何复用示意机位约束和冻结方向，未对机位做标定。", "",
        "| 方法 | Preferred | 无 Preferred | insufficient_evidence | ambiguous | Context-only Preferred | "
        "无测量方向但有 Preferred | Teddy / Don 同列 | Teddy / Don 无优先 | Selected | Preferred 有、Selected 未确认 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, label in zip(METHODS, METHOD_LABELS, strict=True):
        items = [predictions[s["state_id"]][method] for s in states]
        has_pair = [TEDDY_DON <= ranked_targets(row) for row in items]
        counts = (
            sum(bool(row["preferred_target"]) for row in items),
            sum(not row["preferred_target"] for row in items),
            sum(row["decision_status"] == "insufficient_evidence" for row in items),
            sum(row["decision_status"] == "ambiguous" for row in items),
            sum(bool(row["preferred_target"]) and row["evidence_level"] == "context_only" for row in items),
            sum(s["look_direction"] is None and bool(row["preferred_target"])
                for s, row in zip(states, items, strict=True)),
            sum(has_pair),
            sum(pair and row["preferred_target"] not in TEDDY_DON
                for pair, row in zip(has_pair, items, strict=True)),
            sum(row["selected_target"] != "UNKNOWN" for row in items),
            sum(bool(row["preferred_target"]) and row["selected_target"] == "UNKNOWN" for row in items),
        )
        lines.append(f"| {label} | " + " | ".join(map(str, counts)) + " |")

    fused = {state: rows["fused"] for state, rows in predictions.items()}
    agreements = [state for state, row in fused.items() if row["visual_geometry_preference"] == "agree"]
    diverged = [state for state, row in fused.items() if row["visual_geometry_preference"] == "conflict"]
    other_conflicts = [
        (state, [item["reason"] for item in row["conflicts"] if item["kind"] != "preferred_targets_disagree"])
        for state, row in fused.items() if any(item["kind"] != "preferred_targets_disagree" for item in row["conflicts"])
    ]
    changed = {
        method: sum(ranked_tiers(row[method]) != ranked_tiers(row["fused"]) for row in predictions.values())
        for method in METHODS[:2]
    }
    reordered = {
        method: [state for state, row in predictions.items()
                 if row[method]["preferred_target"] != row["fused"]["preferred_target"]]
        for method in METHODS[:2]
    }
    lines.extend([
        "", "## 一致、冲突与排序", "",
        f"- 两种单独方法均有 Preferred 的状态共 {len(agreements) + len(diverged)} 个。"
        f"一致 {len(agreements)} 个，见 {state_list(agreements)}。",
        f"- 两种单独方法的 Preferred 冲突 {len(diverged)} 个。"
        f"其余 {len(states) - len(agreements) - len(diverged)} 个至少有一种方法未提出 Preferred。",
        f"- {sum(predictions[s]['fused']['preferred_target'] == predictions[s]['visual']['preferred_target'] for s in agreements)} "
        "个一致状态的共同首选也由融合保留。",
        f"- 融合相对 Visual-only 改变 {changed['visual']} 个排序，相对 Geometry-only 改变 {changed['geometry']} 个。"
        "排序按明确首位及其余并列候选比较，包含新增候选。仅首位改变分别为 "
        f"{len(reordered['visual'])} 和 {len(reordered['geometry'])} 个。",
        f"- 相对 Visual-only 的首位变化见 {state_list(reordered['visual'])}。",
        f"- 相对 Geometry-only 的首位变化见 {state_list(reordered['geometry'])}。",
        f"- {len(other_conflicts)} 个状态出现视觉候选与几何相容性之间的分歧，含未达到几何相容阈值的视觉首选。",
    ])
    lines.extend(f"  - `{state}`。{' '.join(reasons)}" for state, reasons in other_conflicts)
    if diverged:
        lines.append(f"- Preferred 冲突的状态为 {state_list(diverged)}。")
    if {row["state_id"] for row in old_rows} != predictions.keys():
        raise ValueError("Historical candidates must cover the same frozen states")
    old_pairs = [row for row in old_rows if TEDDY_DON <= {item["target"] for item in row["possible_targets"]}]
    preferred_now = [row["state_id"] for row in old_pairs if fused[row["state_id"]]["preferred_target"] in TEDDY_DON]
    lines.extend([
        "", "## Teddy / Don 与 v1.2 的比较", "",
        f"- v1.2 的 {len(old_pairs)} 个同列状态均未确认目标。其中 {len(preferred_now)} 个状态在 v1.3 "
        f"获得 Teddy 或 Don 的首选，见 {state_list(preferred_now)}。",
        "- v1.3 的两人候选与排序统计见上表。相对 Visual-only，融合在 223 的两个状态把 Teddy 排在前面，"
        "仍保留 Don。两人歧义尚未通过人工审核。",
        f"- v1.2 候选 SHA-256 `{old_source['sha256']}`。[来源]({old_source['file']})",
    ])
    lines.extend(["", "## 重点案例", ""])
    focal = "shot_000212_track_001_look_003"
    if focal in predictions:
        for method, name in zip(METHODS, METHOD_LABELS, strict=True):
            row = predictions[focal][method]
            lines.append(f"- `{focal}` · {name} 排序为 **{ranking(row)}**。{row['reason']}")
    lines.extend([
        "- 217、219、221 的 Michelle 特写保留两位画外人物候选。222 Track 002 的 Head-only 状态可支持大致扇区，"
        "宽景中的人脸分辨率仍不足以确定两人中的一位。",
        "- 223 的视觉画面显示 Michelle 朝画外右上看，两位人物均有宽景参照。几何可以给出相对排序，"
        "尚不能确认落点。",
        "- 214、220 的睁眼样本有与相邻画面呼应的眼线，闭眼样本的提名仅靠互动延续。",
        "- 212 有可见手势及文件夹附近区域作为 OTHER 候选。画面无法把眼线与前景人物分开，"
        "也未记录得到独立端点证据的 AWAY。",
        "", "## 全部状态", "",
        "“/”表示并列，“>”表示有优先排序，“—”表示未提出候选。Teddy / Don 同列包括已有优先顺序但两人都保留的状态。"
        "“无优先”统计保留两人且两人均未获 Preferred 的状态。", "",
        "| Look State | 冻结方向 | Visual-only | Geometry-only | Fusion | Fused Selected | Fused evidence |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for state in states:
        state_id = state["state_id"]
        rows = predictions[state_id]
        order = " | ".join(ranking(rows[method]) for method in METHODS)
        lines.append(f"| `{state_id}` | {state['direction_source']} | {order} | "
                     f"{rows['fused']['selected_target']} | {rows['fused']['evidence_level']} |")
    lines.extend(["", "人工标签均为空。准确率、召回率和准确率–覆盖率仅在填写有效人工标签后计算。", ""])
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def write_card(path, state, mapping, frames, predictions, assets, adjacent_frames):
    path.mkdir(parents=True)
    frame = frames[0]
    track_id = state["track_id"]
    original = assets / f"frame_{frame['global_frame_index']}.jpg"
    with Image.open(original) as source:
        annotated(source, frame, mapping["character"]).save(path / "track_box.jpg", quality=92)
        box = next((b["bbox_xywh"] for b in frame["face_boxes"] if b["track_id"] == track_id), None)
        if box:
            x, y, width, height = box
            source.crop(tuple(round(v) for v in (x, y, x + width, y + height))).save(path / "face.jpg")
            source.crop(tuple(round(v) for v in (
                x + .05 * width, y + .19 * height, x + .95 * width, y + .57 * height
            ))).save(path / "eye_region.jpg")
    state_id = state["state_id"]
    lines = [f"# {state_id}", "", f"正片 {state['start_s']:.3f}–{state['end_s']:.3f} s。", "",
             f"![原片](../../../frames/{original.name})", ""]
    if box:
        lines.extend(["[Face crop](face.jpg) · [Eye region](eye_region.jpg)", ""])
    for row in frames[1:]:
        number = row["global_frame_index"]
        scope = "本状态" if row["within_state"] else "状态之前"
        lines.append(f"- [{scope} · 帧 {number} · {row['movie_timestamp_sec']:.3f} s]"
                     f"(../../../frames/frame_{number}.jpg)")
    lines.append("")
    for number in adjacent_frames:
        lines.append(f"- [相邻镜头 · 帧 {number}](../../../frames/frame_{number}.jpg)")
    lines.append("")
    (path / "blind.md").write_text("\n".join(lines), encoding="utf-8")

    direction = state["look_direction"]
    direction_text = "无持续可用方向"
    if direction:
        vector = ", ".join(f"{value:+.3f}" for value in direction["vector"])
        direction_text = f"{direction['meaning']} · XYZ ({vector}) · {direction['coordinates']}"
    gazers = " / ".join(mapping["character_candidates"])
    visual = predictions["visual"]["observation"]
    fused = predictions["fused"]
    lines = [f"# {state_id}", "", "[原片、裁切和参照帧](blind.md)", "",
             "![原片与 Face Track Bounding Box](track_box.jpg)", "",
             f"- Gazer 身份候选 {gazers}，映射状态 `{mapping['mapping_strength']}`。"
             f"独立画面识别为 {visual['gazer']}。",
             f"- Look Direction {direction_text}。证据来源 `{state['direction_source']}`。"
             f"方向 QC `{state['qc_status']}`。",
             f"- Fused Selected Target **{fused['selected_target']}**。",
             f"- 融合决策 `{fused['decision_status']}`，证据等级 `{fused['evidence_level']}`。",
             "", "| 方法 | Preferred | 备选目标 |", "| --- | --- | --- |"]
    for method, name in zip(METHODS, METHOD_LABELS, strict=True):
        row = predictions[method]
        preferred = row["preferred_target"] or "—"
        alternatives = " / ".join(row["alternative_targets"]) or "—"
        lines.append(f"| {name} | {preferred} | {alternatives} |")
    lines.extend(["", fused["reason"], ""])
    if visual.get("nonperson_cue"):
        lines.extend([visual["nonperson_cue"], ""])
    for number in visual["reference_frames"]:
        lines.append(f"- [独立视觉记录所用参照帧 {number}](../../../frames/frame_{number}.jpg)")
    lines.append("")
    (path / "review.md").write_text("\n".join(lines), encoding="utf-8")


def write_review(output, states, mappings, packet, visual_source, predictions):
    review = output / "review"
    assets = review / "frames"
    assets.mkdir(parents=True)
    frames_by_shot = {}
    copied = set()
    for frames in packet["states"].values():
        for frame in frames:
            name = frame["file"]
            frames_by_shot.setdefault(frame["shot_id"], set()).add(frame["global_frame_index"])
            if name not in copied:
                copy2(visual_source / name, assets / name.rsplit("/", 1)[-1])
                copied.add(name)
    shots = sorted(frames_by_shot)
    adjacent = {shot: ([max(frames_by_shot[shots[i - 1]])] if i else [])
                      + ([min(frames_by_shot[shots[i + 1]])] if i + 1 < len(shots) else [])
                for i, shot in enumerate(shots)}
    selected = {row["state_id"]: row for row in states if row["state_id"] in REVIEW_STATES}
    if selected.keys() != set(REVIEW_STATES):
        raise ValueError("The pilot's fixed review sample is missing frozen Look States")
    lines = ["# Bugonia seq_006 · 人工审核", "", f"共 {len(selected)} 个 Look State。按 Shot、Track、Look State 排列。",
             "样本涵盖视觉和几何一致、候选相容性分歧、Head-only、Teddy / Don 歧义、OTHER 与缺测。",
             "本批无 AWAY 候选。", "", "先打开 [盲审入口](review/blind.md)，用其中的空表记录判断。",
             "审核完成后可打开下表查看预测。", ""]
    blind = ["# Bugonia seq_006 · 原片盲审", "", "[空白标签表](blind_labels.csv)", "",
             "人工标签可填 MICHELLE / TEDDY / DON / AWAY / OTHER / UNCLEAR。确定程度和备注可自由填写。", ""]
    with (output / "human_review.csv").open("x", newline="", encoding="utf-8-sig") as stream, \
            (review / "blind_labels.csv").open("x", newline="", encoding="utf-8-sig") as blind_stream:
        writer = csv.writer(stream)
        blind_writer = csv.writer(blind_stream)
        human_fields = ("human_target", "human_certainty", "notes")
        writer.writerow((*STATE_KEYS, "visual_target", "geometry_target", "fused_target", *human_fields))
        blind_writer.writerow((*STATE_KEYS, *human_fields))
        for shot_id, shot_rows in groupby(selected.values(), key=lambda row: row["shot_id"]):
            heading = [f"## {shot_id}", ""]
            lines.extend(heading)
            blind.extend(heading)
            for track_id, track_rows in groupby(shot_rows, key=lambda row: row["track_id"]):
                track = "track_" + track_id.rsplit("_", 1)[-1]
                heading = [f"### {track}", ""]
                lines.extend(heading)
                blind.extend(heading)
                for state in track_rows:
                    state_id = state["state_id"]
                    look = "look_" + state_id.rsplit("_", 1)[-1]
                    path = f"{shot_id}/{track}/{look}"
                    rows = predictions[state_id]
                    keys = [state[key] for key in STATE_KEYS]
                    writer.writerow([*keys, *(rows[method]["preferred_target"] for method in METHODS), "", "", ""])
                    blind_writer.writerow([*keys, "", "", ""])
                    write_card(review / path, state, mappings[track_id], packet["states"][state_id],
                               rows, assets, adjacent[shot_id])
                    lines.append(f"- [{look}](review/{path}/review.md)")
                    blind.append(f"- [{look}]({path}/blind.md)")
                lines.append("")
                blind.append("")
    (output / "review.md").write_text("\n".join(lines), encoding="utf-8")
    (review / "blind.md").write_text("\n".join(blind), encoding="utf-8")
