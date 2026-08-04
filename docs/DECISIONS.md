# OscarDP implementation decisions

This document records decisions that fill gaps in `AGENTS.md`. `AGENTS.md`
remains authoritative when the two differ.

## MVP scope

The first implementation supports discovery, `process-one`, and validation. It
does not expose full-dataset processing until one movie has passed manual QC.

## TransNetV2 provenance

The model definition in `src/oscardp/vendor/transnetv2_pytorch.py` is a pinned
snapshot retrieved on 2026-08-04 from:

<https://github.com/soCzech/TransNetV2/blob/master/inference-pytorch/transnetv2_pytorch.py>

The vendored snapshot SHA-256 is
`0b200b65a15dea9df6e054cbd7fcf63e6d4f3f7774050487dfa0857587eb2deb`.

The upstream MIT license is retained in
`src/oscardp/vendor/TRANSNETV2_LICENSE`. Model weights are not committed. Every
run requires `--weights PATH`; the SHA-256 digest is used in resume state and
written to the processing log. The user is responsible for supplying weights
converted from the official TransNetV2 TensorFlow checkpoint.

Inference uses 100-frame windows with a stride of 50, padding 25 frames of
context at each video end and retaining each window's center 50 predictions.
A maximal run above the threshold becomes one boundary at the run's
highest-confidence frame. This differs intentionally from the upstream
inclusive scene helper so OscarDP can preserve gap-free, end-exclusive ranges.

Weights are intentionally loaded into host memory with
`torch.load(..., map_location="cpu")`, then strictly applied before the model is
moved to its selected device. This limits transient GPU memory use and does not
imply CPU inference. `auto` selects `cuda:0` when CUDA is available; explicit
`cuda` fails rather than falling back when CUDA is unavailable. Model execution
uses `torch.inference_mode()`.

The process log records the Python and PyTorch runtime, requested and selected
devices, actual model and first-input devices, GPU identity/capability, model
load time, model-only inference time, inference-stage time including frame
decode, CUDA peak allocated memory, and total pipeline time. CUDA timings are
bounded by synchronization calls. These operational metrics do not change any
dataset JSON/JSONL schema.

## CUDA dependency

The project pins PyTorch 2.13.0 in `pyproject.toml`. WSL CUDA installations use
`requirements-cu126.txt`, which points only the PyTorch installation at the
official CUDA 12.6 wheel index. The project does not use uv yet, so it does not
carry `tool.uv.sources` or a synthetic `uv.lock`; those will be added together
only if uv becomes the project's verified dependency workflow.

## Time and frame semantics

Frame PTS values come from an ffprobe decoded-frame scan and are normalized so
the first decoded frame is time zero. A stream is VFR when fewer than 99.9% of
positive adjacent PTS deltas are within 0.1% of `1 / avg_frame_rate`. VFR end
times use the next decoded PTS; the final end uses packet duration and falls
back to the median positive PTS delta. CFR timestamps use `frame_index / fps`.

The ffprobe PTS count must equal the frame count decoded for TransNetV2. A
mismatch is a hard failure rather than a timestamp approximation.

## Boundary confidence

For each shot, `boundary_before_confidence` is the confidence of its start
boundary and `boundary_after_confidence` is the confidence of its end boundary.
The missing outer boundary of the first or final shot is `null`.

## QC defaults

A very short shot lasts less than 0.5 seconds. A low-confidence boundary has a
confidence below `min(1.0, threshold + 0.1)`. Default QC prioritizes those
boundaries, boundaries adjacent to very short shots, and up to 24 evenly spaced
samples, with a total cap of 100. The contact sheet shows at most 24 pairs.

## Resume, publish, and index

Incomplete work lives in `OUTPUT_ROOT/.work/<movie_key>` and is reusable only
when source size/mtime, threshold, and model SHA-256 match. Valid work is
published atomically. `--overwrite` moves prior output into
`OUTPUT_ROOT/.backup/` before publishing replacement output.

`OUTPUT_ROOT/index.jsonl` contains one current row per movie, sorted by
`movie_key`:

```json
{
  "movie_key": "tt32536315",
  "source_video_relpath": "tt32536315_BlueMoon/movie.mkv",
  "status": "completed",
  "output_relpath": "tt32536315",
  "shot_count": 100,
  "validation_passed": true,
  "error": null
}
```

Failed rows use `status="failed"`, `output_relpath=null`, and a non-null error.

## CSV export

`shots.jsonl` remains the only primary shot dataset. When interoperability or
an acceptance check requires CSV, `python -m oscardp.shots export-csv
--movie-dir PATH` creates `shots.csv` on demand from the existing JSONL. The
export has the same record count and top-level fields; the nested `model`
object is encoded as compact JSON in its CSV cell. Normal processing does not
create or maintain CSV.
