"""Exact global-frame decoding. No approximate seeking or frame resampling.

FFmpeg decodes the prefix through the last requested frame, but only selected
shots (plus boundary timestamps) cross the rawvideo pipe. This avoids a new
decoder dependency and preserves Stage01's sequential FFmpeg frame ordering.
"""
import queue
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oscardp.shots.media import ExternalToolError


@dataclass
class Frame:
    index: int
    timestamp: float
    next_timestamp: float
    image: np.ndarray


def iter_frames(video: Path, shots, metadata, cfr):
    indices = sorted({0} | {n for s in shots for n in range(
        s["temporal"]["start_frame"], s["temporal"]["end_frame"]+1)
        if metadata.frame_count is None or n < metadata.frame_count})
    wanted = {n for s in shots for n in range(s["temporal"]["start_frame"], s["temporal"]["end_frame"])}
    expression = "+".join(["eq(n,0)"] + [f"between(n,{s['temporal']['start_frame']},{s['temporal']['end_frame']})" for s in shots])
    command = ["ffmpeg", "-hide_banner", "-nostats", "-v", "info", "-threads", "4", "-noautorotate",
               "-i", str(video), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-vf", f"select='{expression}',settb=AVTB,showinfo",
               "-frames:v", str(len(indices)), "-vsync", "0", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    timing = queue.Queue()
    pattern = re.compile(r"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:")
    with tempfile.TemporaryFile(mode="w+t") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def drain():
            for line in iter(process.stderr.readline, b""):
                line = line.decode(errors="replace")
                log.write(line)
                match = pattern.search(line)
                if match:
                    timing.put(int(match[1])/1_000_000)
            timing.put(None)
        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        pending = None
        origin = None
        try:
            for index in indices:
                size = metadata.width*metadata.height*3
                raw = process.stdout.read(size)
                if len(raw) != size:
                    if (not raw and metadata.frame_count is None and index == indices[-1]
                            and pending is not None and pending[0]+1 == index
                            and any(s["temporal"]["end_frame"] == index for s in shots)):
                        # MKV often omits nb_frames: EOF may be the requested
                        # exclusive movie boundary, never a missing observed frame.
                        break
                    raise ExternalToolError(f"Incomplete exact decode at global frame {index}")
                pts = timing.get(timeout=120)
                if pts is None:
                    raise ExternalToolError("Missing frame PTS")
                if origin is None:
                    origin = pts
                relative = pts-origin
                timestamp = index/metadata.fps if cfr else relative
                if cfr and abs(relative-timestamp) > max(.002, .1/metadata.fps):
                    raise ExternalToolError(f"Stage01 CFR timeline mismatch at frame {index}: {relative} vs {timestamp}")
                if pending:
                    previous_index, previous_time, image = pending
                    if index != previous_index+1 or timestamp <= previous_time:
                        raise ExternalToolError("Missing/nonmonotonic boundary timestamp")
                    yield Frame(previous_index, previous_time, timestamp, image)
                pending = (index, timestamp, np.frombuffer(raw, np.uint8).reshape(metadata.height, metadata.width, 3)) if index in wanted else None
            if pending:
                # The final movie frame has no following frame; Stage01 owns its exclusive boundary.
                end = next(s["temporal"]["end"] for s in shots if s["temporal"]["end_frame"] == pending[0]+1)
                yield Frame(pending[0], pending[1], end, pending[2])
            code = process.wait()
            thread.join()
            if code:
                log.seek(0)
                raise ExternalToolError(log.read()[-4000:])
        finally:
            if process.poll() is None:
                process.kill()
            process.stdout.close()
            process.wait()
            thread.join()
            process.stderr.close()
