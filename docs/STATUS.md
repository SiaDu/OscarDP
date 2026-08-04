# OscarDP project status

Last updated: 2026-08-04

## Current milestone

The first real-video smoke test passed on a 30.030-second excerpt from
`tt32536315` (Blue Moon). This proves the single-video pipeline and official
PyTorch TransNetV2 weights work together in the project environment. It does
not authorize full-dataset processing; a complete test movie has not yet been
manually accepted.

## Smoke-test input

- Source: `/mnt/i/datasets/oscar_movie/tt32536315_BlueMoon/Blue.Moon.2025.2160p.WEB-DL.SDR.H.265..DDP5.1.Atmos-HamiltonShare.mkv`
- Source interval: `00:10:00.000` to approximately `00:10:30.000`
- Local sample: `data/smoke_test/input/sample_30s.mp4` (Git-ignored)
- Sample properties: H.264, 1280×534, 24000/1001 fps, 720 frames, 30.030 s
- Sample provenance and extraction command: `data/smoke_test/README.md`

## Runtime and model

- Interpreter: `/home/sia/OscarDP/.venv/bin/python` (Python 3.12.3)
- Torch: `2.13.0+cpu`; inference device: CPU
- Model: vendored official PyTorch TransNetV2 implementation
- Weights: `models/transnetv2/transnetv2-pytorch-weights.pth` (Git-ignored)
- Weight SHA-256: `53f3e734bc191ae1c58ef61121711518c40767013ea32644fa5f1db9dcbb5ae8`
- Load result: strict state-dict load succeeded with 90 keys and 7,616,258 parameters

## Pipeline command

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

## Results

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
`data/smoke_test/` subdirectories. No full movie and no dataset batch was run.
