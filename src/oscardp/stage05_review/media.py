"""One exact source decode shared by five review videos and frame stepping images."""
import json
import subprocess
from pathlib import Path

from PIL import Image

from oscardp.shots.media import probe_video
from oscardp.stage05.video_reader import iter_frames
from .integration import write


def make_media(data, viewer):
    config = data["tracking_config"]
    metadata, _ = probe_video(data["video"], data["video"].name)
    if (metadata.width, metadata.height) != (config["width"], config["height"]):
        raise ValueError("Movie dimensions changed")
    cfr = config["timestamp_rule"] == "frame_index_cfr"
    metadata.fps = config["fps"]
    shots = [s for s in config["shots"] if s["shot_id"] in data["run"]["selected_shot_ids"]]
    expected_times = {}
    for row in data["traces"]:
        key = row["shot_id"], row["global_frame_index"]
        previous = expected_times.setdefault(key, row["movie_timestamp_sec"])
        if previous != row["movie_timestamp_sec"]:
            raise ValueError("Source traces disagree on frame time")
    # CFR is only the preview encoding grid; actual source timestamps remain authoritative.
    preview_fps = config["fps"] if cfr else 24.0
    maps = {s["shot_id"]: [] for s in shots}
    processes = {}
    width = 810
    height = round(metadata.height*width/metadata.width/2)*2
    try:
        for s in shots:
            sid = s["shot_id"]
            (viewer/"media"/sid).mkdir(parents=True)
            processes[sid] = subprocess.Popen([
                "ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(preview_fps), "-i", "pipe:0", "-an",
                "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(viewer/"media"/f"{sid}.mp4")], stdin=subprocess.PIPE,
                stderr=subprocess.PIPE)
        for frame in iter_frames(data["video"], shots, metadata, cfr):
            shot = next(s for s in shots if s["temporal"]["start_frame"] <= frame.index < s["temporal"]["end_frame"])
            sid = shot["shot_id"]
            if abs(frame.timestamp-expected_times[sid, frame.index]) > 1e-6:
                raise ValueError("Video/measurement source frame-time mismatch")
            image = Image.fromarray(frame.image[:, :, ::-1]).resize((width, height), Image.Resampling.LANCZOS)
            image.save(viewer/"media"/sid/f"{frame.index}.jpg", quality=87)
            processes[sid].stdin.write(image.tobytes())
            maps[sid].append({"preview_frame_index": len(maps[sid]),
                "preview_time_sec": len(maps[sid])/preview_fps,
                "global_frame_index": frame.index, "movie_timestamp_sec": frame.timestamp,
                "next_movie_timestamp_sec": frame.next_timestamp,
                "still": f"media/{sid}/{frame.index}.jpg"})
        for sid, process in processes.items():
            process.stdin.close()
            error = process.stderr.read().decode()
            if process.wait():
                raise RuntimeError(error)
            process.stderr.close()
            shot = next(s for s in shots if s["shot_id"] == sid)
            expected = list(range(shot["temporal"]["start_frame"], shot["temporal"]["end_frame"]))
            if [f["global_frame_index"] for f in maps[sid]] != expected:
                raise ValueError("Missing/duplicated review video frame")
            payload = {"shot_id": sid, "source_timestamp_rule": config["timestamp_rule"],
                "preview_fps": preview_fps, "source_measurement_fps": config["fps"] if cfr else None,
                "frame_boundaries": "zero-based inclusive start / exclusive end",
                "decode": "sequential original movie frame select; no seek or temporal resampling",
                "audio": "not included; Stage02 text provided separately",
                "frames": maps[sid]}
            write(viewer/"data"/f"{sid}_frames.json", payload)
            print(f"Review media: {sid}: {len(maps[sid])} frames", flush=True)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
                process.wait()
    return {sid: len(frames) for sid, frames in maps.items()}


def verify_media(root, shots):
    results = {}
    for shot in shots:
        sid = shot["shot_id"]
        mapping = json.loads((root/"viewer/data"/f"{sid}_frames.json").read_text())
        frames = mapping["frames"]
        temporal = shot["effective_processing_temporal"]
        if [f["global_frame_index"] for f in frames] != list(range(temporal["start_frame"], temporal["end_frame"])):
            raise ValueError("Incomplete original frame mapping")
        for i, f in enumerate(frames):
            if abs(f["preview_time_sec"]-i/mapping["preview_fps"]) > 1e-9:
                raise ValueError("Incorrect preview frame mapping")
            if not f["movie_timestamp_sec"] < f["next_movie_timestamp_sec"]:
                raise ValueError("Invalid source timestamp")
            if i and abs(frames[i-1]["next_movie_timestamp_sec"]-f["movie_timestamp_sec"]) > 1e-7:
                raise ValueError("Source timestamp gap inside review clip")
            if not (root/"viewer"/f["still"]).is_file():
                raise ValueError("Missing frame-step still")
        for measured, target in [(frames[0]["movie_timestamp_sec"], temporal["start"]),
                                 (frames[-1]["next_movie_timestamp_sec"], temporal["end"])]:
            if abs(measured-target) > 1e-6:
                raise ValueError("Clip effective boundary mismatch")
        process = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "json", str(root/"viewer/media"/f"{sid}.mp4")],
            capture_output=True, text=True, check=True)
        if int(json.loads(process.stdout)["streams"][0]["nb_read_frames"]) != len(frames):
            raise ValueError("Encoded video frame count differs from mapping")
        results[sid] = len(frames)
    return results
