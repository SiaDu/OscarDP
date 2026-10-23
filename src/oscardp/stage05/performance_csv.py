"""Strict numeric OpenFace CSV parsing with separate AU output families."""
import csv
import math
import re


def columns_by_modality(columns):
    return {
        "au_intensity":[c for c in columns if re.fullmatch(r"AU\d+_r",c)],
        "au_occurrence":[c for c in columns if re.fullmatch(r"AU\d+_c",c)],
        "pose":[c for c in columns if c.startswith("pose_")],
        "gaze":[c for c in columns if c.startswith("gaze_")],
        "landmarks_2d":[c for c in columns if re.fullmatch(r"[xy]_\d+",c)],
        "landmarks_3d":[c for c in columns if re.fullmatch(r"[XYZ]_\d+",c)],
        "eye_landmarks":[c for c in columns if c.startswith("eye_lmk_")],
        "model_parameters":[c for c in columns if c.startswith("p_")],
    }


def parse_csv(path, expected_count, temporary_fps):
    with path.open(newline="",encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = [c.strip() for c in next(reader)]
        if len(set(header)) != len(header) or not {"frame","face_id","timestamp","confidence","success"} <= set(header):
            raise ValueError("Invalid/missing OpenFace columns")
        rows = {}
        for cells in reader:
            if not cells:
                continue
            if len(cells) != len(header):
                raise ValueError("OpenFace CSV row length mismatch")
            row,nonfinite = {},[]
            for name,text in zip(header,cells):
                try:
                    value = float(text.strip()) if text.strip() else float("nan")
                except ValueError as exc:
                    raise ValueError(f"Nonnumeric OpenFace {name}: {text}") from exc
                if not math.isfinite(value):
                    value = None
                    nonfinite.append(name)
                if name in {"frame","face_id","success"}:
                    if value is None or not value.is_integer():
                        raise ValueError(f"Invalid OpenFace integer column: {name}")
                    value = int(value)
                row[name] = value
            local = row["frame"]
            if not 1 <= local <= expected_count or local in rows:
                raise ValueError("Duplicate/out-of-range OpenFace local frame; possible multiple-person output")
            if row["face_id"] != 0 or row["success"] not in {0,1}:
                raise ValueError("Unexpected OpenFace face ID/success; single-person pilot only")
            if row["timestamp"] is not None and abs(row["timestamp"]-(local-1)/temporary_fps) > .002:
                raise ValueError("OpenFace local timestamp/frame offset mismatch")
            row["nonfinite_columns"] = nonfinite
            rows[local] = row
    return header,rows
