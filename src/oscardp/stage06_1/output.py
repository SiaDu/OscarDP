"""Render only supported positions, and expose uncertainty for human review."""

from html import escape
from math import hypot
from pathlib import Path
import xml.etree.ElementTree as ET


def make_svg(candidate):
    layouts = candidate["layouts"]
    width, panel_height = 720, 440
    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width=str(width),
                     height=str(75 + panel_height * max(1, len(layouts))),
                     viewBox=f"0 0 {width} {75 + panel_height * max(1, len(layouts))}")

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
        .person { fill: #158267; stroke: white; stroke-width: 2 }
        .camera { fill: #315994; stroke: white; stroke-width: 2 }
    """)
    marker = ET.SubElement(ET.SubElement(svg, "defs"), "marker", id="body-arrow", markerWidth="8",
                           markerHeight="8", refX="6", refY="3", orient="auto")
    ET.SubElement(marker, "path", d="M0 0 L6 3 L0 6 Z", fill="#09624d")
    add("rect", width="100%", height="100%", fill="#f1f4f8")
    add("text", "Approximate layout — not calibrated", x=28, y=35, **{"class": "heading"})
    add("text", "Circles: people   •   Squares: cameras   •   Dashed: axis   •   Arrows: body facing",
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

        camera_points = {}
        for shot in candidate["shots"]:
            if shot["layout_id"] == layout["layout_id"] and shot["camera_position"] is not None:
                camera_points.setdefault(point(shot["camera_position"]), []).append(shot["shot_id"].removeprefix("shot_"))
        for (x, y), ids in camera_points.items():
            add("rect", x=x - 6, y=y - 6, width=12, height=12, rx=2, **{"class": "camera"})
            add("text", "/".join(ids), x=x, y=y + 23, text_anchor="middle")
    ET.indent(svg)
    return ET.tostring(svg, encoding="unicode") + "\n"


def make_review(candidate, packet, images, args):
    def cell(text):
        return escape(str(text), quote=False).replace("|", "&#124;").replace("\n", " ")

    sequence = f"{candidate['movie_id']} / {candidate['sequence_id']} / {candidate['scene_id']}"
    lines = [
        f"# Stage06.1 人工复核 · {sequence}", "",
        f"- VLM 与方式：{args.vlm}",
        f"- 实际查看：{sum(map(len, images.values()))} 张不同的 Stage01 JPG、{len(images)} 个镜头。",
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

    lines += [
        "## 逐镜头检查", "",
        "| Shot | 布局 / 状态 | 机位 / 依据 | 主拍人物 | 轴侧 / 轴线 / 参照 | 人工核对 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in sorted(candidate["shots"], key=lambda row: row["shot_id"]):
        pair = "–".join(s["axis_pair"]) if s["axis_pair"] else "—"
        comparison = f"{s['axis_side']} / {pair} / {s['axis_reference_shot'] or '—'}"
        pos = s["camera_position"] if s["camera_position"] is not None else "未知"
        cols = (s["shot_id"], f"{s['layout_id'] or '未知'} / {s['layout_status']}",
                f"{pos} / {s['camera_quality']}", s["camera_subject"], comparison, s["note"])
        lines.append("| " + " | ".join(map(cell, cols)) + " |")

    lines += [
        "", "## 建议优先复核", "",
        "- 用参考镜头的同框、前后景遮挡与房间固定物检查三人的相对位置和身体朝向。",
        "- 对照参考镜头用的同一人物轴线，复核同侧判断与近景无法定位的角度。",
        "- 检查三张采样帧之间和剪切处是否有走位或换轴。", "",
        "## 已查看的图片", "",
        f"- 联系表：{', '.join(dict.fromkeys(args.inspected_sheet))}。",
    ]
    for shot in packet["shots"]:
        sid = shot["shot_id"]
        if sid in images:
            names = ", ".join(f"`{Path(path).name}`" for path in images[sid])
            lines.append(f"- {sid}：{names}")
    return "\n".join(lines) + "\n"
