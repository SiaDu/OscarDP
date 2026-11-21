"""Read only the selected Stage01–04.3 evidence and prepare visual review sheets."""

import json
import math
import re

from PIL import Image, ImageDraw

from oscardp.shots.frame_samples import load_json, physical_samples
from oscardp.shots.transnet import sha256_file

ROLES = ("early", "middle", "late")
PRIORITY = ("shot_000212", "shot_000218", "shot_000222")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def indexed(rows, field):
    result = {}
    for row in rows:
        key = row[field]
        if key in result:
            raise ValueError(f"Duplicate {field}: {key}")
        result[key] = row
    return result


def prepare_inputs(movie_dir, movie_id, sequence_id):
    if not re.fullmatch(r"tt[0-9]+", movie_id):
        raise ValueError("Invalid IMDb movie ID")
    movie_dir = movie_dir.resolve()
    sources, data = {}, {}
    for stage in ("01", "02", "03", "04", "04_3"):
        path = movie_dir / f"Stage{stage}_{movie_id}.json"
        data[stage] = load_json(path)
        if data[stage]["movie_id"] != movie_id:
            raise ValueError(f"Movie ID mismatch: {path}")
        sources[f"stage{stage}"] = {"path": str(path), "sha256": sha256_file(path)}

    sequence = indexed(data["04_3"]["sequence_candidates"], "sequence_id")[sequence_id]
    scene_id = sequence["scene_id"]
    scene = indexed(data["04"]["scenes"], "scene_id")[scene_id]
    stages = {stage: indexed(data[stage]["shots"], "shot_id") for stage in ("01", "02", "03", "04")}
    shot_ids = sequence["shot_ids"]
    first, last = sequence["first_shot_id"], sequence["last_shot_id"]
    if not shot_ids or (shot_ids[0], shot_ids[-1]) != (first, last):
        raise ValueError("Stage04.3 bounds do not match its shot list")
    order = list(stages["01"])
    if shot_ids != order[order.index(first):order.index(last) + 1]:
        raise ValueError("Stage04.3 shots must form a contiguous ordered Stage01 range")
    start, end = sequence["start_seconds"], sequence["end_seconds"]
    first_time, last_time = (stages["01"][sid]["temporal"] for sid in (first, last))
    if not (all(type(t) in (int, float) and math.isfinite(t) for t in (start, end))
            and first_time["start"] <= start < first_time["end"]
            and last_time["start"] < end <= last_time["end"] and start < end):
        raise ValueError("Stage04.3 time range must begin and end inside its boundary shots")

    shots = []
    for sid in shot_ids:
        if not re.fullmatch(r"shot_[0-9]{6}", sid):
            raise ValueError(f"Invalid shot ID: {sid}")
        stage01, stage02, stage03, stage04 = (stages[s][sid] for s in ("01", "02", "03", "04"))
        if stage04["scene_id"] != scene_id:
            raise ValueError(f"{sid} does not belong to {scene_id} in Stage04")
        temporal = stage01["temporal"]
        if temporal["start"] >= end or temporal["end"] <= start:
            raise ValueError(f"{sid} does not overlap the sequence")
        samples = physical_samples(sid, temporal["start_frame"], temporal["end_frame"])
        stage03_samples = indexed(stage03["samples"], "role")
        images = {}
        for role in ROLES:
            sample = samples[role]
            canonical = movie_dir / sample["image_relpath"]
            names = [canonical]
            if canonical.name == f"{sid}_middle.jpg":
                names.append(movie_dir / "shot_img" / f"{sid}.jpg")
            hint = stage03_samples.get(role, {}).get("image_file")
            if hint in {path.name for path in names}:
                names.sort(key=lambda path: path.name != hint)
            path = next((p for p in names if p.is_file()), None)
            images[role] = {
                "path": str(path) if path else None,
                "sha256": sha256_file(path) if path else None,
                "frame_index": sample["frame_index"] if path else None,
                "expected_name": canonical.name,
            }
        visual = stage03["visual"]
        shots.append({
            "shot_id": sid, "temporal": temporal, "images": images,
            "stage02_speaker_hints": sorted({line["speaker"] for line in stage02["dialogue"] if line["speaker"]}),
            "stage03_framing_hint": {key: visual[key] for key in ("shot_scale", "over_the_shoulder")},
            "stage04_character_candidates": stage04["character_candidates"],
        })

    priority = [sid for sid in PRIORITY if sid in shot_ids] if (movie_id, sequence_id) == ("tt12300742", "seq_006") else []
    return {
        "movie_id": movie_id, "sequence_id": sequence_id, "scene_id": scene_id,
        "start_s": start, "end_s": end, "sources": sources, "shots": shots,
        "stage04_participant_hints": scene["narrative_participants"],
        "priority_review_shots": priority,
    }


def create_sheets(packet, output):
    """Keep the three sample roles in parallel columns with one shot per row."""
    by_id = indexed(packet["shots"], "shot_id")
    priority = packet["priority_review_shots"]
    order = priority + [sid for sid in by_id if sid not in priority]
    width, height, label_height = 480, 270, 25
    packet["contact_sheets"] = {}
    for start in range(0, len(order), 3):
        shot_ids = order[start:start + 3]
        sheet = Image.new("RGB", (width * len(ROLES), (height + label_height) * len(shot_ids)), "#181a1f")
        draw = ImageDraw.Draw(sheet)
        for row, sid in enumerate(shot_ids):
            for col, role in enumerate(ROLES):
                sample = by_id[sid]["images"][role]
                x, y = col * width, row * (height + label_height)
                suffix = "missing" if not sample["path"] else f"frame {sample['frame_index']}"
                draw.text((x + 8, y + 6), f"{sid} / {role} / {suffix}", fill="white")
                if sample["path"]:
                    with Image.open(sample["path"]) as image:
                        if image.format != "JPEG":
                            raise ValueError(f"Expected JPEG: {sample['path']}")
                        image.thumbnail((width, height))
                        sheet.paste(image.convert("RGB"), (x + (width - image.width) // 2, y + label_height))
        name = f"shots_{start // 3 + 1:02d}.jpg"
        with (output / name).open("xb") as stream:
            sheet.save(stream, format="JPEG", quality=93)
        packet["contact_sheets"][name] = shot_ids


def inspected_images(packet, sheet_names):
    """Resolve explicitly inspected contact sheets to distinct source JPGs."""
    shot_ids = {sid for name in sheet_names for sid in packet["contact_sheets"][name]}
    images = {}
    for shot in packet["shots"]:
        if shot["shot_id"] not in shot_ids:
            continue
        paths = {sample["path"] for sample in shot["images"].values() if sample["path"]}
        if paths:
            images[shot["shot_id"]] = sorted(paths)
    return images
