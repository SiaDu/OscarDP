"""Full-frame FFV1 scratch videos with exact, verified local/global frame maps."""
import hashlib
import subprocess
import time
from pathlib import Path

from .output import atomic_json
from .video_reader import iter_frames


def frame_mapping(shot, track_id, frame, local_frame):
    temporal = shot["effective_processing_temporal"]
    expected = temporal["start_frame"]+local_frame-1
    if frame.index != expected or frame.timestamp >= temporal["end"] or frame.timestamp < temporal["start"]-1e-6:
        raise ValueError("Exact local/global frame or source timestamp mismatch")
    return {"shot_id":shot["shot_id"],"track_id":track_id,"openface_local_frame":local_frame,
            "global_frame_index":frame.index,"movie_timestamp_sec":frame.timestamp,
            "source_next_timestamp_sec":frame.next_timestamp,
            "source_bgr_sha256":hashlib.sha256(frame.image.tobytes()).hexdigest()}


def verify_lossless(path, mapping, width, height):
    args = ["ffmpeg","-v","error","-threads","4","-i",str(path),"-an","-sn","-vsync","0",
            "-pix_fmt","bgr24","-f","rawvideo","pipe:1"]
    process = subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for row in mapping:
            raw = process.stdout.read(width*height*3)
            if len(raw) != width*height*3 or hashlib.sha256(raw).hexdigest() != row["source_bgr_sha256"]:
                raise ValueError("Scratch FFV1 changed source pixels/frame order")
        if process.stdout.read(1):
            raise ValueError("Unexpected extra scratch frame")
        error = process.stderr.read().decode(errors="replace")
        if process.wait():
            raise RuntimeError(error)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


def prepare(video, shots, tracks, metadata, cfr, scratch, output):
    mappings, paths = {},{}
    decode_sec = encode_sec = 0.
    stream = iter_frames(video,shots,metadata,cfr)
    try:
        for shot in sorted(shots,key=lambda s:s["temporal"]["start_frame"]):
            sid = shot["shot_id"]
            tid = next(tid for tid,t in tracks.items() if t["shot_id"]==sid)
            path = scratch/f"{sid}.mkv"
            # Temporary CFR cadence is only a transport clock; movie PTS stays in the map.
            args = ["ffmpeg","-v","error","-n","-f","rawvideo","-pixel_format","bgr24",
                    "-video_size",f"{metadata.width}x{metadata.height}","-framerate",str(metadata.fps),
                    "-i","pipe:0","-an","-c:v","ffv1","-level","3","-threads","4","-pix_fmt","bgr0",str(path)]
            process = subprocess.Popen(args,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            rows = []
            try:
                for local in range(1,shot["temporal"]["end_frame"]-shot["temporal"]["start_frame"]+1):
                    tick = time.perf_counter()
                    frame = next(stream)
                    decode_sec += time.perf_counter()-tick
                    row = frame_mapping(shot,tid,frame,local)
                    rows.append(row)
                    tick = time.perf_counter()
                    process.stdin.write(frame.image.tobytes())
                    encode_sec += time.perf_counter()-tick
                tick = time.perf_counter()
                process.stdin.close()
                error = process.stderr.read().decode(errors="replace")
                code = process.wait()
                encode_sec += time.perf_counter()-tick
                if code:
                    raise RuntimeError(error)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
                if not process.stdin.closed:
                    process.stdin.close()
                process.stderr.close()
            tick = time.perf_counter()
            verify_lossless(path,rows,metadata.width,metadata.height)
            encode_sec += time.perf_counter()-tick
            atomic_json(output/f"{sid}.json",{"shot_id":sid,"track_id":tid,"original_temporal":shot["original_temporal"],
                        "effective_processing_temporal":shot["effective_processing_temporal"],
                        "temporary_codec":"FFV1 level3 bgr0; source BGR pixels verified after round trip",
                        "temporary_fps":metadata.fps,"temporary_timestamps_are_source_truth":False,
                        "source_timestamp_rule":"frame_index_cfr" if cfr else "relative_source_pts", "frames":rows})
            mappings[sid],paths[sid] = rows,path
            print(f"prepared {sid}: {len(rows)} exact full-frame FFV1 frames",flush=True)
        if next(stream,None) is not None:
            raise ValueError("Extra decoded source frames")
    finally:
        stream.close()
    return mappings,paths,{"source_decode_wait_sec":decode_sec,"lossless_encode_write_verify_sec":encode_sec}
