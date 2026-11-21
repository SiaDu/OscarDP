"""Render only supported positions, and expose uncertainty for human review."""

from html import escape
from math import hypot
from pathlib import Path
import xml.etree.ElementTree as ET


def make_svg(candidate):
    layouts = candidate["layouts"]
    shots = candidate["shots"]
    width, panel_height = 720, 440
    sidebar = []
    if "camera_groups" in candidate:
        unplaced = {s["shot_id"]: s for s in shots if s["camera_position"] is None}

        def list_shots(ids):
            sidebar.extend("  ".join(ids[start:start + 2]) for start in range(0, len(ids), 2))
            sidebar.append("")

        for group in candidate["camera_groups"]:
            ids = [sid for sid in group["shot_ids"] if sid in unplaced]
            if ids:
                subjects = ", ".join(sorted({unplaced[sid]["camera_subject"] for sid in ids}))
                sidebar.append(f"{group['camera_group']} → {subjects}")
                list_shots(ids)
        ungrouped = [sid for sid, s in unplaced.items() if s["camera_group"] is None]
        if ungrouped:
            sidebar.append("Ungrouped")
            list_shots(ungrouped)
    if sidebar:
        width += 330
    height = max(75 + panel_height * max(1, len(layouts)), 188 + 23 * len(sidebar))
    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width=str(width),
                     height=str(height), viewBox=f"0 0 {width} {height}")

    def add(tag, content=None, **attrs):
        node = ET.SubElement(svg, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})
        node.text = content
        return node

    add("title", "Approximate layout — not calibrated")
    add("style", """
        text { font: 14px sans-serif; fill: #142539 }
        .heading { font-size: 23px; font-weight: bold }
        .axis { stroke: #7c597e; stroke-width: 2; stroke-dasharray: 6 5 }
        .facing { stroke: #09624d; stroke-width: 2; marker-end: url(#body-arrow) }
        .camera-direction { stroke: #315994; stroke-width: 2; marker-end: url(#camera-arrow) }
        .person { fill: #158267; stroke: white; stroke-width: 2 }
        .camera { fill: #315994; stroke: white; stroke-width: 2 }
    """)
    defs = ET.SubElement(svg, "defs")
    for marker_id, color in (("body-arrow", "#09624d"), ("camera-arrow", "#315994")):
        marker = ET.SubElement(defs, "marker", id=marker_id, markerWidth="8", markerHeight="8",
                               refX="6", refY="3", orient="auto")
        ET.SubElement(marker, "path", d="M0 0 L6 3 L0 6 Z", fill=color)
    add("rect", width="100%", height="100%", fill="#f1f4f8")
    add("text", "Approximate layout — not calibrated", x=28, y=35, **{"class": "heading"})
    add("text", "Circles: people   •   Squares: cameras   •   Dashed: axis   •   Green: body   •   Blue: camera",
        x=28, y=58)
    if not layouts:
        add("text", "No supported layout to plot.", x=28, y=115)

    for i, layout in enumerate(layouts):
        top = 75 + i * panel_height
        title = f"{layout['layout_id']}  |  {layout['start_shot_id']}–{layout['end_shot_id']}  |  {layout['layout_status']}"
        add("text", title, x=28, y=top + 24)
        add("rect", x=28, y=top + 37, width=664, height=378, rx=10, fill="white", stroke="#bac7d6")

        def point(p):
            return (68 + p[0] * 570, top + 65 + p[1] * 308)

        plotted = [p for p in layout["people"] if p["position"] is not None]
        named = {p["person"]: point(p["position"]) for p in plotted if p["person"] != "UNKNOWN"}
        pair = layout["axis_pair"]
        if pair and all(p in named for p in pair):
            (x1, y1), (x2, y2) = (named[p] for p in pair)
            add("line", x1=x1, y1=y1, x2=x2, y2=y2, **{"class": "axis"})

        for person in plotted:
            x, y = point(person["position"])
            target = named.get(person["facing_to"])
            if target:
                dx, dy = target[0] - x, target[1] - y
                distance = hypot(dx, dy)
                if distance > 25:
                    length = min(32, distance - 10)
                    add("line", x1=x + dx * 9 / distance, y1=y + dy * 9 / distance,
                        x2=x + dx * length / distance, y2=y + dy * length / distance,
                        **{"class": "facing"})
        for person in plotted:
            x, y = point(person["position"])
            add("circle", cx=x, cy=y, r=8, **{"class": "person"})
            add("text", person["person"], x=x, y=y - 13, text_anchor="middle")

        camera_points, directions = {}, set()
        for shot in shots:
            if shot["layout_id"] != layout["layout_id"] or shot["camera_position"] is None:
                continue
            x, y = point(shot["camera_position"])
            camera_points.setdefault((x, y), []).append(shot["shot_id"].removeprefix("shot_"))
            target = named.get(shot["camera_subject"])
            if target is not None:
                directions.add(((x, y), target))
        for (x, y), (tx, ty) in sorted(directions):
            dx, dy = tx - x, ty - y
            distance = hypot(dx, dy)
            if distance > 20:
                add("line", x1=x + dx * 9 / distance, y1=y + dy * 9 / distance,
                    x2=tx - dx * 10 / distance, y2=ty - dy * 10 / distance,
                    **{"class": "camera-direction"})
        for (x, y), ids in camera_points.items():
            add("rect", x=x - 6, y=y - 6, width=12, height=12, rx=2, **{"class": "camera"})
            add("text", "/".join(ids), x=x, y=y + 23, text_anchor="middle")
    if sidebar:
        add("rect", x=715, y=112, width=307, height=52 + 23 * len(sidebar), rx=10,
            fill="white", stroke="#bac7d6")
        add("text", "Cameras without positions", x=733, y=141, font_weight="bold")
        for i, line in enumerate(sidebar):
            if line:
                add("text", line, x=733, y=170 + 23 * i)
    ET.indent(svg)
    return ET.tostring(svg, encoding="unicode") + "\n"


def make_review(candidate, packet, images, args, *, rechecked=None):
    def cell(text):
        return escape(str(text), quote=False).replace("|", "&#124;").replace("\n", " ")

    sequence = f"{candidate['movie_id']} / {candidate['sequence_id']} / {candidate['scene_id']}"
    version = " v1.1" if "camera_groups" in candidate else ""
    image_summary = f"{sum(map(len, images.values()))} 张不同的 Stage01 JPG、{len(images)} 个镜头"
    provenance = [f"- 实际查看：{image_summary}。"]
    if rechecked:
        provenance = [
            f"- 累计视觉输入：{image_summary}，包含沿用的 v1.0 证据。",
            f"- 本轮复核：{sum(map(len, rechecked.values()))} 张不同 JPG、{len(rechecked)} 个镜头，"
            f"联系表 {', '.join(dict.fromkeys(args.rechecked_sheet))}。",
            f"- v1.0 来源：`{args.candidate}`。v1.1 沿用其中的布局、机位坐标和轴侧判断。",
        ]
    lines = [
        f"# Stage06.1{version} 人工复核 · {sequence}", "",
        f"- VLM 与方式：{args.vlm}",
        *provenance,
        f"- 电影全局时间：{candidate['start_s']:.6f}–{candidate['end_s']:.6f} 秒。",
        "- 状态：`model_candidate` / `pending`。人物名是外观连续性候选。",
        "- 坐标仅为俯视示意。三张采样帧之间的走位或换轴仍需连续画面复核。",
        "- 参考资料：输入清单 `inputs.json` 中记录 Stage01–04.3 来源、JPG 路径和 SHA-256。",
        "", "## 布局与人物", "",
    ]
    if not candidate["layouts"]:
        lines += ["没有足够证据建立布局。", ""]
    for layout in candidate["layouts"]:
        refs = ", ".join(layout["reference_shots"])
        lines += [
            f"### {layout['layout_id']} · {layout['start_shot_id']}–{layout['end_shot_id']}",
            "",
            f"- 状态：`{layout['layout_status']}`。参考：{refs}。",
            f"- {layout['note']}", "",
            "| 人物候选 | 位置 | 姿态 | 身体朝向 | 位置依据 | 人工核对 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for p in layout["people"]:
            cols = (p["person"], p["position"] if p["position"] is not None else "未知", p["posture"],
                    p["facing_to"], p["position_quality"], p["note"])
            lines.append("| " + " | ".join(map(cell, cols)) + " |")
        lines.append("")

    if version:
        lines += ["## 重复机位候选", "", "分组仅表示拍摄位置或方向相似的候选，组内坐标各自保留。", ""]
        if not candidate["camera_groups"]:
            lines += ["暂无足够图像证据建立分组。", ""]
        for group in candidate["camera_groups"]:
            members = "、".join(group["shot_ids"])
            evidence = "、".join(group["evidence_shots"])
            lines += [f"- `{group['camera_group']}`（{group['group_quality']}）：{members}。"
                      f"参考 {group['reference_shot']}，证据 {evidence}。{group['note']}"]
        lines.append("")
    lines += [
        "## 逐镜头检查", "",
        "| Shot | 布局 / 状态 | 机位 / 依据 | Camera Group | 主拍人物 | 轴侧 / 轴线 / 参照 | 人工核对 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in sorted(candidate["shots"], key=lambda row: row["shot_id"]):
        pair = "–".join(s["axis_pair"]) if s["axis_pair"] else "—"
        comparison = f"{s['axis_side']} / {pair} / {s['axis_reference_shot'] or '—'}"
        pos = s["camera_position"] if s["camera_position"] is not None else "未知"
        cols = (s["shot_id"], f"{s['layout_id'] or '未知'} / {s['layout_status']}",
                f"{pos} / {s['camera_quality']}", s.get("camera_group") or "—",
                s["camera_subject"], comparison, s["note"])
        lines.append("| " + " | ".join(map(cell, cols)) + " |")

    lines += [
        "", "## 建议优先复核", "",
        "- 用参考镜头的同框、前后景遮挡与房间固定物检查三人的相对位置和身体朝向。",
        "- 对照参考镜头用的同一人物轴线，复核同侧判断与近景无法定位的角度。",
        "- 检查三张采样帧之间和剪切处是否有走位或换轴。", "",
        "## 已查看的图片", "",
        f"- 累计联系表：{', '.join(dict.fromkeys([*args.inspected_sheet, *(getattr(args, 'rechecked_sheet', None) or [])]))}。",
    ]
    for shot in packet["shots"]:
        sid = shot["shot_id"]
        if sid in images:
            names = ", ".join(f"`{Path(path).name}`" for path in images[sid])
            prefix = "（v1.1 复核）" if rechecked and sid in rechecked else ""
            lines.append(f"- {sid}{prefix}：{names}")
    return "\n".join(lines) + "\n"
