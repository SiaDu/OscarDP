"""Read-only source-video audit. Never applies shot boundary corrections.

Reference path: an independent FFmpeg invocation emits every sequential frame's
thumbnail hash (no select), and selected native BGR frames on a second branch.
The selected frames are re-scaled with the identical FFmpeg filter and matched
against hashes indexed by sequential record ordinal. A separate Stage05 decode
then compares native pixel hashes and timestamps. No face/model inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from oscardp.shots.discovery import discover_movies
from oscardp.shots.media import probe_video
from .output import atomic_json, identity, load_json
from .video_reader import iter_frames

THUMB_FILTER = "scale=96:64:flags=neighbor"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def selected_indices(boundaries):
    return sorted({n for b in boundaries for n in range(b["frame"]-2, b["frame"]+4) if n >= 0})


def parse_sequential_hashes(path, wanted):
    """Count actual output records from zero; never infer indices from PTS/FPS."""
    result = {}
    ordinal = 0
    time_base = None
    with path.open() as stream:
        for line in stream:
            if line.startswith("#tb 0:"):
                numerator, denominator = line.split(":", 1)[1].strip().split("/")
                time_base = int(numerator)/int(denominator)
            if line.startswith("#") or not line.strip():
                continue
            fields = [s.strip() for s in line.split(",")]
            if len(fields) != 6 or time_base is None:
                raise ValueError(f"Malformed sequential framehash: {line}")
            if ordinal in wanted or ordinal == 0:
                result[ordinal] = {"thumbnail_sha256": fields[-1], "pts_sec": int(fields[2])*time_base}
            ordinal += 1
    if not set(wanted) <= result.keys():
        raise ValueError("Reference stream ended before requested frames")
    return result, ordinal


def reference_decode(video, metadata, indices, root):
    expression = "+".join(f"eq(n,{n})" for n in indices)
    filters = (f"[0:v:0]format=bgr24,split=2[a][b];"
               f"[a]{THUMB_FILTER}[seq];[b]select='{expression}'[picked]")
    hash_path = root/"sequential_framehash.txt"
    command = ["ffmpeg", "-hide_banner", "-nostats", "-v", "error", "-threads", "4", "-filter_complex_threads", "2", "-noautorotate",
               "-i", str(video), "-filter_complex", filters,
               "-map", "[seq]", "-frames:v", str(indices[-1]+1), "-c:v", "rawvideo", "-threads", "1",
               "-pix_fmt", "bgr24", "-enc_time_base", "1:1000000", "-fps_mode", "passthrough",
               "-f", "framehash", "-hash", "sha256", str(hash_path),
               "-map", "[picked]", "-frames:v", str(len(indices)), "-c:v", "rawvideo", "-threads", "1",
               "-pix_fmt", "bgr24", "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]
    atomic_json(root/"reference_command.json", command)
    result = {}
    native_path = root/"selected_native.bgr.tmp"
    with (root/"reference_stderr.txt").open("wb") as log, native_path.open("wb") as native:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
        try:
            for index in indices:
                raw = process.stdout.read(metadata.width*metadata.height*3)
                if len(raw) != metadata.width*metadata.height*3:
                    raise RuntimeError(f"Reference decode incomplete at {index}; see reference_stderr.txt")
                native.write(raw)
                image = np.frombuffer(raw, np.uint8).reshape(metadata.height, metadata.width, 3)
                resized = cv2.resize(image, (960, round(metadata.height*960/metadata.width)), interpolation=cv2.INTER_AREA)
                target = root/"frames"/f"frame_{index:08d}.png"
                if not cv2.imwrite(str(target), resized):
                    raise RuntimeError(f"Cannot write evidence frame: {target}")
                result[index] = {"native_bgr_sha256": digest(raw), "evidence_image": str(target.relative_to(root)),
                                 "image_sha256": digest(target.read_bytes())}
                print(f"reference global frame {index} ({len(result)}/{len(indices)})", flush=True)
            if process.wait():
                raise RuntimeError("Reference FFmpeg failed; see reference_stderr.txt")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
    # Re-scale selected native frames independently. Compare with the no-select
    # sequential branch's hashes: this anchors select n to actual ordinal n.
    command = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size",
               f"{metadata.width}x{metadata.height}", "-i", str(native_path), "-vf", THUMB_FILTER,
               "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    thumbs = subprocess.run(command, capture_output=True, check=True).stdout
    if len(thumbs) != len(indices)*96*64*3:
        raise RuntimeError("Thumbnail verification returned incomplete frames")
    sequence, frame_count = parse_sequential_hashes(hash_path, set(indices))
    origin = sequence[0]["pts_sec"]
    for i, index in enumerate(indices):
        checksum = digest(thumbs[i*96*64*3:(i+1)*96*64*3])
        if checksum != sequence[index]["thumbnail_sha256"]:
            raise RuntimeError(f"Selected frame differs from sequential ordinal {index}")
        result[index].update(reference_ordinal=index, reference_pts_sec=sequence[index]["pts_sec"]-origin,
                             sequential_thumbnail_sha256=checksum, ordinal_verified=True)
    native_path.unlink()
    return result, frame_count


def stage05_decode(video, metadata, indices, reference, root, cfr):
    # Merge overlapping windows into non-overlapping half-open ranges.
    runs = []
    for n in indices:
        if runs and runs[-1][1] == n:
            runs[-1][1] = n+1
        else:
            runs.append([n, n+1])
    shots = [{"temporal": {"start_frame": a, "end_frame": b, "end": b/metadata.fps}}
             for a, b in runs]
    decoded = 0
    for frame in iter_frames(video, shots, metadata, cfr):
        item = reference[frame.index]
        if digest(frame.image.tobytes()) != item["native_bgr_sha256"]:
            raise RuntimeError(f"Stage05 native pixel mismatch at global frame {frame.index}")
        error = abs(frame.timestamp-item["reference_pts_sec"])
        if error > max(.002, .1/metadata.fps):
            raise RuntimeError(f"Stage05 timestamp mismatch at {frame.index}: {error}s")
        item.update(stage05_native_pixels_match=True, stage05_timestamp_sec=frame.timestamp,
                    timestamp_error_sec=error)
        decoded += 1
        if decoded % 12 == 0:
            print(f"Stage05 compare {decoded}/{len(indices)} global frame {frame.index}", flush=True)
    if decoded != len(indices):
        raise RuntimeError("Stage05 did not deliver every selected frame")


def contact_sheets(boundaries, frames, root):
    """Six context frames; required N-1/N/N+1 are the central three columns."""
    for page in range((len(boundaries)+5)//6):
        rows = boundaries[page*6:page*6+6]
        sheet = Image.new("RGB", (1920, len(rows)*270), "#16181b")
        draw = ImageDraw.Draw(sheet)
        for row, b in enumerate(rows):
            draw.text((6, row*270+5), f"{b['shot_id']} {b['edge']} N={b['frame']} | {b['scene_context'][:145]}", fill="white")
            for col, offset in enumerate(range(-2, 4)):
                n = b["frame"]+offset
                image = Image.open(root/frames[n]["evidence_image"]).convert("RGB")
                image.thumbnail((320, 214))
                sheet.paste(image, (col*320, row*270+45))
                label = f"N{offset:+d} | frame {n} | {frames[n]['reference_pts_sec']:.6f}s"
                draw.text((col*320+3, row*270+28), label, fill="#f9d761" if offset in (-1, 0, 1) else "white")
        sheet.save(root/"contact_sheets"/f"boundaries_{page+1:02d}.jpg", quality=94)


def boundary_measurements(boundaries, frames, root):
    """Adjacent-image differences are evidence, never automatic cut approvals."""
    measurements = []
    for b in boundaries:
        n = b["frame"]
        images = {i: cv2.resize(cv2.imread(str(root/frames[i]["evidence_image"])), (64, 36),
                               interpolation=cv2.INTER_AREA).astype(np.float32)
                  for i in range(n-2, n+4)}
        differences = [{"left_frame": i, "right_frame": i+1,
                        "normalized_mean_absolute_bgr_difference": float(np.abs(images[i+1]-images[i]).mean()/255)}
                       for i in range(n-2, n+3)]
        measurements.append({"boundary_id": f"frame_{n:08d}", "shot_id": b["shot_id"],
                             "edge": b["edge"], "original_frame": n, "adjacent_differences": differences,
                             "interpretation": "Image differences only; human/model visual review must distinguish cuts, motion, fades and flashes."})
    atomic_json(root/"boundary_measurements.json", measurements)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--movie-id", required=True)
    p.add_argument("--analysis-root", type=Path, required=True)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--video", type=Path)
    p.add_argument("--shot-ids", nargs="+", required=True, help="Audit selected start boundaries")
    p.add_argument("--end-shot-ids", nargs="+", default=[], help="Also audit selected exclusive ends")
    p.add_argument("--output-dir", type=Path, required=True, help="New audit directory; existing outputs refused")
    p.add_argument("--phase", choices=["reference", "compare"], required=True,
                   help="First independent reference extraction, then separate Stage05 pixel comparison")
    args = p.parse_args(argv)
    if not re.fullmatch(r"tt\d{7,10}", args.movie_id):
        p.error("Invalid movie ID")
    source_dir = args.analysis_root/args.movie_id
    stage01_path = source_dir/f"Stage01_{args.movie_id}.json"
    stage01 = load_json(stage01_path)
    if stage01["movie_id"] != args.movie_id:
        raise ValueError("Stage01 movie ID mismatch")
    lookup = {s["shot_id"]: s for s in stage01["shots"]}
    contexts = {}
    stage04_path = source_dir/f"Stage04_{args.movie_id}.json"
    if stage04_path.exists():
        for scene in load_json(stage04_path).get("scenes", []):
            for n in range(int(scene["start_shot_id"].split("_")[1]), int(scene["end_shot_id"].split("_")[1])+1):
                contexts[f"shot_{n:06d}"] = f"{scene['scene_id']}: {scene.get('title', '')} / {scene.get('location', '')}"
    boundaries = []
    for edge, ids in [("start", args.shot_ids), ("end", args.end_shot_ids)]:
        for sid in ids:
            if sid not in lookup:
                raise ValueError(f"Unknown shot: {sid}")
            n = lookup[sid]["temporal"][f"{edge}_frame"]
            if n < 2:
                raise ValueError("Boundary needs two preceding frames")
            boundaries.append({"shot_id": sid, "edge": edge, "frame": n,
                               "original_temporal": lookup[sid]["temporal"],
                               "scene_context": contexts.get(sid, "Stage04 context unavailable")})
    if not boundaries:
        raise ValueError("Explicit boundaries required")
    indices = selected_indices(boundaries)
    if args.video:
        video = args.video.resolve(strict=True)
    else:
        refs = discover_movies(args.input_root, movie_key=args.movie_id)
        if len(refs) != 1:
            raise ValueError("Video discovery did not resolve exactly one movie; use --video")
        video = refs[0].source_path
    metadata, _ = probe_video(video, video.name)
    if metadata.frame_count is not None and indices[-1]+1 >= metadata.frame_count:
        raise ValueError("Audit context reaches beyond available video frames")
    provenance = {p.name: {**identity(p), "sha256": digest(p.read_bytes())}
                  for p in [source_dir/f"Stage{s:02d}_{args.movie_id}.json" for s in range(1, 5)] if p.exists()}
    config = {"audit_version": "1.0", "movie_id": args.movie_id, "video": identity(video),
              "sources": provenance, "fps": metadata.fps, "native_size": [metadata.width, metadata.height],
              "boundaries": boundaries, "frame_indices": indices,
              "context_source": "Stage04 secondary annotations, sampling labels only; not verified scene ground truth",
              "frame_semantics": "zero-based; Stage01 start inclusive/end exclusive",
              "source_writes": False}
    root = args.output_dir
    start = time.perf_counter()
    if args.phase == "reference":
        root.mkdir(parents=True, exist_ok=False)
        (root/"frames").mkdir()
        (root/"contact_sheets").mkdir()
        atomic_json(root/"audit_config.json", config)
        frames, count = reference_decode(video, metadata, indices, root)
        atomic_json(root/"reference_frames.json", {"frames": frames, "sequential_decoded_frame_count": count,
                                                   "elapsed_sec": time.perf_counter()-start})
        contact_sheets(boundaries, frames, root)
        boundary_measurements(boundaries, frames, root)
    else:
        if load_json(root/"audit_config.json") != config:
            raise ValueError("Audit inputs or selection changed; refusing comparison")
        if (root/"decoder_comparison.json").exists():
            raise FileExistsError("Comparison already exists; preserve it and use a new audit directory")
        reference = load_json(root/"reference_frames.json")
        frames = {int(k): v for k, v in reference["frames"].items()}
        cfr = all(abs(s["temporal"][key]-s["temporal"][frame_key]/metadata.fps) < .0001
                  for s in stage01["shots"] for key, frame_key in [("start", "start_frame"), ("end", "end_frame")])
        stage05_decode(video, metadata, indices, frames, root, cfr)
        atomic_json(root/"decoder_comparison.json", {"status": "passed", "frame_count": len(frames),
                    "elapsed_sec": time.perf_counter()-start, "frames": frames,
                    "maximum_timestamp_error_sec": max(f["timestamp_error_sec"] for f in frames.values()),
                    "independence_limit": "Both paths use FFmpeg codec backend; reference indices are independently anchored by every-frame sequential hash ordinal, not Stage05 indexing or select n alone."})
    print(f"{args.phase} audit complete: {root}", flush=True)


if __name__ == "__main__":
    main()
