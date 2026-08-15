# OscarDP project status

Last updated: 2026-08-15

## Stage 2 production release

The Stage 2 production reviewer and retrieval stack is frozen as
`v3.2.1-production.3-retrieval-v3-validator-v3`. Seven of the eight production
targets have passed final QC and have versioned reviewed alignment and shot
context. `tt30144839` (One Battle After Another) remains
`BLOCKED_WITH_EXPLICIT_REASON` because its screenplay is unavailable.

`/mnt/i/datasets/oscar_script/tt30343021_SongSungBlue.pdf` is not a substitute:
its title page identifies it as *Song Sung Blue*, and it is already the source
for completed movie `tt30343021`.

The dataset-level frozen release is generated under:

```text
/mnt/i/datasets/oscar_movie_processed/stage2_releases/v3_2_1_production_3_final_seven
```

Expected terminal summary:

- COMPLETE: 7/8
- TERMINAL: 8/8
- BLOCKED: 1/8
- Pending isolated human ambiguities: 4

The release includes a manifest, independent validation report, self-contained
pending-ambiguity package, and Stage 3 handoff. Stage 3 must use only the
reviewed shot-context paths named by that release and must exclude the blocked
movie from resolved-quality claims.

## Current milestone

The real-video smoke test passed on both the historical CPU environment and the
current CUDA environment using the same 30.030-second excerpt from `tt32536315`
(Blue Moon). CPU and GPU produced identical transition frames and shot ranges.
This does not authorize full-dataset processing; no batch was started during
the CUDA qualification work.

## Smoke-test input

- Source: `/mnt/i/datasets/oscar_movie/tt32536315_BlueMoon/Blue.Moon.2025.2160p.WEB-DL.SDR.H.265..DDP5.1.Atmos-HamiltonShare.mkv`
- Source interval: `00:10:00.000` to approximately `00:10:30.000`
- Local sample: `data/smoke_test/input/sample_30s.mp4` (Git-ignored)
- Sample properties: H.264, 1280×534, 24000/1001 fps, 720 frames, 30.030 s
- Sample provenance and extraction command: `data/smoke_test/README.md`

## Historical CPU runtime and model

- Interpreter: `/home/sia/OscarDP/.venv/bin/python` (Python 3.12.3)
- Torch: `2.13.0+cpu`; inference device: CPU
- Model: vendored official PyTorch TransNetV2 implementation
- Weights: `models/transnetv2/transnetv2-pytorch-weights.pth` (Git-ignored)
- Weight SHA-256: `53f3e734bc191ae1c58ef61121711518c40767013ea32644fa5f1db9dcbb5ae8`
- Load result: strict state-dict load succeeded with 90 keys and 7,616,258 parameters

## Historical CPU pipeline command

```bash
.venv/bin/python -m oscardp.shots process-one \
  --video data/smoke_test/input/sample_30s.mp4 \
  --input-root data/smoke_test/input \
  --output-root data/smoke_test/output \
  --weights models/transnetv2/transnetv2-pytorch-weights.pth \
  --device cpu \
  --save-raw-predictions
```

CSV was exported only for the acceptance check:

```bash
.venv/bin/python -m oscardp.shots export-csv \
  --movie-dir data/smoke_test/output/sample_30s
```

## Historical CPU results

- Published movie key: `sample_30s`
- Shot count: 7 (`shot_000001` through `shot_000007`)
- Frame coverage: 0 through 720, end-exclusive and gap-free
- Time coverage: 0.000 through 30.030 seconds
- Raw predictions: 720 values; min `6.3347107e-07`, max `0.98609877`
- Frames above threshold 0.5: 6
- Keyframes: 7/7 present and readable
- Boundary pairs: 6/6 present and readable
- `shots.jsonl` / on-demand `shots.csv`: 7 / 7 records, core fields consistent
- Pipeline validation: passed with no errors and no missing keyframes
- Automated tests: 22 passed
- Ruff: passed
- Manual QC: contact sheet and a midpoint keyframe were visually inspected;
  the excerpt and detected cuts are plausible and readable.

All generated smoke-test video and output files remain under ignored
`data/smoke_test/` subdirectories.

## CUDA smoke test

The GPU-only rerun used the same input, official weights, threshold, decoding,
and boundary conversion as the historical CPU smoke test. It wrote to the
separate ignored output root `data/smoke_test/output/gpu_smoke_cu126` and did
not overwrite the CPU output.

```bash
.venv/bin/python -m oscardp.shots process-one \
  --video data/smoke_test/input/sample_30s.mp4 \
  --input-root data/smoke_test/input \
  --output-root data/smoke_test/output/gpu_smoke_cu126 \
  --weights models/transnetv2/transnetv2-pytorch-weights.pth \
  --threshold 0.5 \
  --device cuda \
  --save-raw-predictions
```

Runtime evidence and metrics:

- Interpreter: `/home/sia/OscarDP/.venv/bin/python`
- Torch: `2.13.0+cu126`; CUDA runtime: 12.6
- GPU: NVIDIA GeForce GTX 1060; capability `(6, 1)`
- Selected/model/input device: `cuda:0` / `cuda:0` / `cuda:0`
- Model load: 0.424596 s
- Model-only inference: 2.034668 s
- Inference stage including FFmpeg decode: 2.651468 s
- Total pipeline: 7.463130 s
- Peak CUDA allocated memory: 167,716,864 bytes (159.95 MiB)
- Transitions/shots: 6 / 7
- Validation: passed; 0 missing keyframes

Compared with the historical CPU output:

- Transition frames are identical: `103, 422, 502, 552, 628, 676`.
- All seven shot start/end ranges are identical.
- All 720 raw predictions are present in both outputs.
- Maximum/mean absolute prediction difference: `1.1920929e-7` /
  `4.0459764e-10`.
- Serialized boundary confidences are identical at six-decimal precision.

The historical CPU run did not record model-only inference time, so no CPU/GPU
model-inference speedup is claimed. Its approximately 14.1-second pipeline time
is retained only as historical context, not as a controlled benchmark.

CUDA qualification checks completed with Ruff passing, 26 pytest tests
passing, all seven GPU keyframes readable, all 12 QC boundary images readable,
and the contact sheet readable. One full movie had previously been processed on
CPU, but no complete movie or dataset batch was started as part of this CUDA
smoke test.
