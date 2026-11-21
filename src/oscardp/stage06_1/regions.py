"""Filter schematic camera cells using only independently annotated visual clues."""

from math import atan2, degrees, hypot

GRID_STEP = 0.1
# x increases right and y increases down in the plan. Clockwise from human
# body forward is positive, so the person's own right is +90 degrees.
TORSO_ANGLES = {"front": 0, "front_left": -45, "front_right": 45,
                "left_side": -90, "right_side": 90, "back_left": -135,
                "back_right": 135, "back": 180}
REGION_CLUES = {"view_of_torso", "depth_order", "screen_order", "axis_side"}
EPS = 1e-9


def delta(a, b):
    return a[0] - b[0], a[1] - b[1]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def infer_region(shot, layout, shots):
    """Return the reproducible region and messages for the review, including exclusions."""
    clues = shot["camera_clues"]
    messages = []
    if clues["camera_height"] != "unknown":
        messages.append("camera_height 仅记录高度，不参与俯视筛选。")
    if not layout:
        return None, messages + ["无可沿用的布局。"]
    people = {p["person"]: p for p in layout["people"] if p["person"] != "UNKNOWN"}
    positions = {name: p["position"] for name, p in people.items()
                 if p["position"] is not None and p["position_quality"] != "unknown"}
    checks = {}

    view = clues["view_of_torso"]
    if view != "unknown":
        person = clues["view_person"]
        p = positions.get(person)
        target = positions.get(people.get(person, {}).get("facing_to"))
        if p is not None and target is not None and p != target:
            body_forward = delta(target, p)

            def torso_check(camera):
                ray = delta(camera, p)
                if hypot(*ray) < EPS:
                    return False
                angle = degrees(atan2(cross(body_forward, ray), dot(body_forward, ray)))
                return abs((angle - TORSO_ANGLES[view] + 180) % 360 - 180) <= 30

            checks["view_of_torso"] = torso_check
        else:
            messages.append("未用 view_of_torso：人物或身体朝向的参照位置不足。")

    aim = positions.get(shot["camera_subject"])
    if shot["camera_subject"] == "GROUP":
        visible = {name for kind in ("depth_order", "screen_order")
                   for pair in clues[kind] for name in pair.values()}
        if len(visible) >= 2 and visible <= positions.keys():
            # Approximate framing centre for projection only. It is not a camera point.
            aim = tuple(sum(positions[name][i] for name in sorted(visible)) / len(visible) for i in (0, 1))

    def projection_check(camera, pairs, screen):
        forward = delta(aim, camera)
        length = hypot(*forward)
        if length < EPS:
            return False
        forward = forward[0] / length, forward[1] / length
        right = -forward[1], forward[0]
        for a, b in pairs:
            ra, rb = delta(positions[a], camera), delta(positions[b], camera)
            za, zb = dot(ra, forward), dot(rb, forward)
            if min(za, zb) <= EPS:
                return False
            # Positive horizontal image coordinate is screen-right. Depth is
            # along the ray to the aim point, never estimated from image size.
            if screen:
                if dot(rb, right) / zb - dot(ra, right) / za <= EPS:
                    return False
            elif zb - za <= EPS:
                return False
        return True

    for kind, a, b in (("depth_order", "nearer", "farther"), ("screen_order", "left", "right")):
        pairs = [(pair[a], pair[b]) for pair in clues[kind]]
        usable = [pair for pair in pairs if all(name in positions for name in pair)]
        if pairs and aim is None:
            messages.append(f"未用 {kind}：无可用的投影拍摄朝向。")
        elif pairs:
            if len(usable) < len(pairs):
                missing = "、".join(f"{a}/{b}" for a, b in pairs if (a, b) not in usable)
                messages.append(f"未用 {kind} 中的 {missing}：缺人物示意位置。")
            if usable:
                checks[kind] = lambda camera, pairs=usable, screen=(kind == "screen_order"): (
                    projection_check(camera, pairs, screen))

    side = shot["axis_side"]
    if side != "unknown":
        pair = shot["axis_pair"]
        ref = shots.get(shot["axis_reference_shot"], {})
        comparable = (pair and all(name in positions for name in pair)
                      and ref.get("layout_id") == shot["layout_id"]
                      and ref.get("camera_position") is not None
                      and set(ref.get("axis_pair") or []) == set(pair))
        if comparable:
            start, end = (positions[name] for name in pair)
            axis_forward = delta(end, start)
            width = hypot(*axis_forward)
            if width > EPS:
                def axis_distance(camera):
                    return cross(axis_forward, delta(camera, start)) / width

                ref_side = axis_distance(ref["camera_position"])
                if side == "on_axis":
                    checks["axis_side"] = lambda camera: abs(axis_distance(camera)) <= GRID_STEP / 2 + EPS
                elif abs(ref_side) > GRID_STEP / 2:
                    sign = 1 if (ref_side > 0) == (side == "same") else -1
                    checks["axis_side"] = lambda camera: sign * axis_distance(camera) > EPS
        if "axis_side" not in checks:
            messages.append("未用 axis_side：参照机位或同一布局下的轴侧不足。")

    if not checks:
        return None, messages + ["没有可转换为几何筛选的线索。"]
    cells = [[i / 10, j / 10] for j in range(11) for i in range(11)
             if all(check((i / 10, j / 10)) for check in checks.values())]
    if not cells:
        return None, messages + [f"图内无共同可行网格（{', '.join(checks)}）；线索可能冲突，"
                                 "也可能受投影近似或示意范围限制。"]
    result_note = "示意坐标系的可行网格，依赖当前布局及粗略投影，不代表真实摄影机位置。"
    if any(0 in cell or 1 in cell for cell in cells):
        result_note += "区域触及示意边界，图外可能仍有可行位置。"
    if shot["camera_position"] is not None and shot["camera_position"] not in cells:
        result_note += "v1.1 机位点未落入本次候选网格，保留原点供复核。"
    return {"grid_step": GRID_STEP, "cells": cells, "region_quality": "rough",
            "used_clues": list(checks), "note": result_note}, messages
