#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oscardp.stage0.media import (
    GIB, gpu_hdr_video_filter, probe, select_audio_stream, target_fps,
)
from oscardp.stage0.pipeline import (
    discover_videos, output_path, run_ffmpeg_with_progress, validate_output,
)
from oscardp.shots.progress import ProgressReporter


def ffmpeg_command(source, destination, info, cq, bitrate=None):
    command = [
        "ffmpeg", "-hide_banner", "-y", "-v", "error",
        "-progress", "pipe:1", "-nostats",
        "-init_hw_device", "vulkan=vk:0",
        "-filter_hw_device", "vk",
        "-i", str(source),
        "-map", "0:v:0",
    ]

    audio = select_audio_stream(info)
    if audio is not None:
        command += [
            "-map", f"0:{audio}",
            "-c:a", "aac", "-profile:a", "aac_low",
            "-b:a", "192k", "-ar", "48000", "-ac", "2",
        ]

    command += [
        "-vf", gpu_hdr_video_filter(info),
        "-r", f"{target_fps(info):.6f}",
        "-c:v", "hevc_nvenc", "-preset", "p6",
        "-pix_fmt", "yuv420p", "-rc", "vbr",
    ]

    if bitrate is None:
        command += ["-cq", str(cq), "-b:v", "0"]
    else:
        command += [
            "-b:v", str(bitrate),
            "-maxrate", str(bitrate),
            "-bufsize", str(bitrate * 2),
        ]

    command += [
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-movflags", "+faststart",
        str(destination),
    ]
    return command


def main():
    parser = argparse.ArgumentParser(
        description="GPU HDR-to-SDR Stage 0 transcoding for one movie"
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--movie-id", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cq", type=int, default=25)
    parser.add_argument("--max-size-gib", type=float, default=4.5)
    args = parser.parse_args()

    if not 0 <= args.cq <= 51:
        parser.error("--cq must be between 0 and 51")
    if args.max_size_gib <= 0:
        parser.error("--max-size-gib must be positive")

    root = args.input_root.resolve()
    sources = [
        path for path in discover_videos(root, args.movie_id, None, None)
        if "_standardized" not in path.stem
    ]
    if len(sources) != 1:
        parser.error(
            f"Expected exactly one source for {args.movie_id}; found {len(sources)}"
        )

    source = sources[0]
    info = probe(source)
    if info.dynamic_range != "HDR":
        parser.error(f"This script expects an HDR source: {source}")

    final = output_path(source, root, root)

    if final.exists() and not args.force:
        valid, message, _ = validate_output(final, info, args.max_size_gib)
        if valid:
            print(f"Existing valid output; skipped: {final}")
            return 0
        parser.error(f"Existing output is invalid; inspect it before --force: {message}")

    print(f"Source: {source}")
    print(f"Output: {final}")
    if not args.execute:
        print("Dry run. Add --execute to transcode.")
        return 0

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{final.stem}.", suffix=".partial.mp4", dir=final.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    progress = ProgressReporter()

    try:
        print("Encoding with GPU HDR tone mapping...")
        run_ffmpeg_with_progress(
            ffmpeg_command(source, temporary, info, args.cq),
            info, progress, "GPU HDR transcoding",
        )

        if temporary.stat().st_size > args.max_size_gib * GIB:
            temporary.unlink()
            target_gib = args.max_size_gib * 0.95
            bitrate = int(target_gib * GIB * 8 / info.duration_sec - 192_000)
            if bitrate <= 0:
                raise RuntimeError("Size limit gives a non-positive video bitrate")
            print("Output exceeds size limit; retrying with a bitrate cap...")
            run_ffmpeg_with_progress(
                ffmpeg_command(source, temporary, info, args.cq, bitrate),
                info, progress, "Retry GPU HDR transcoding for size limit",
            )

        valid, message, _ = validate_output(
            temporary, info, args.max_size_gib
        )
        if not valid:
            raise RuntimeError(f"Output validation failed: {message}")

        os.replace(temporary, final)
        print(f"Complete: {final}")
        return 0
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
