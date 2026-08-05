# OscarDP

OscarDP detects movie shot boundaries with the official PyTorch TransNetV2
model, extracts midpoint keyframes, and writes boundary QC images.

The project has two independent stages. Stage 1 creates `shots.jsonl` with
TransNetV2. Stage 2 parses a screenplay PDF, cleans an SRT, aligns subtitle
dialogue to existing screenplay blocks, and maps that context onto every
existing shot. Stage 2 never modifies `shots.jsonl` or Stage 1 media/QC files.

## WSL CUDA environment

The production environment uses Python 3.12, PyTorch 2.13.0, and the official
CUDA 12.6 wheel index. Install the CUDA build first, then install OscarDP:

```bash
/home/sia/OscarDP/.venv/bin/python -m pip install -r requirements-cu126.txt
/home/sia/OscarDP/.venv/bin/python -m pip install -e ".[test]"
```

Install deterministic Stage 2 dependencies with:

```bash
.venv/bin/python -m pip install -e ".[context,test]"
```

Semantic similarity is optional: use `.[context,semantic,test]` and explicitly
pass `--semantic-model`. The default exact/fuzzy/sequence pipeline does not
download or load a semantic model.

Verify that PyTorch can allocate a tensor on the GPU:

```bash
/home/sia/OscarDP/.venv/bin/python -c \
  'import torch; x=torch.zeros(1, device="cuda:0"); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), x.device)'
```

The CUDA runtime is supplied by the PyTorch wheel. NVIDIA's WSL driver and the
model weights are external prerequisites and are not project dependencies.

## Process one movie

```bash
.venv/bin/python -m oscardp.shots process-one \
  --video /path/to/movie.mkv \
  --input-root /path/to/input \
  --output-root /path/to/output \
  --weights models/transnetv2/transnetv2-pytorch-weights.pth \
  --device auto
```

`--device auto` selects `cuda:0` when CUDA is available and otherwise selects
CPU. `--device cuda` fails if CUDA is unavailable; it never silently falls
back to CPU.

## Stage 2 screenplay context

```bash
.venv/bin/python -m oscardp.script_context process-one \
  --movie-key tt32536315 \
  --screenplay /path/to/screenplay.pdf \
  --subtitle /path/to/subtitle.srt \
  --shots /path/to/movie-output/shots.jsonl \
  --output-dir /path/to/movie-output \
  --subtitle-language en \
  --disable-semantic \
  --llm-mode export \
  --resume
```

`movie_script_context.json` preserves parsed screenplay scenes and source
blocks. `subtitle_script_alignment.jsonl` records deterministic dialogue
matches and review status. `shot_script_context.jsonl` preserves every input
shot and adds timestamp-overlapping subtitle/script references.

LLMs are never used to rewrite the screenplay or dialogue. `--llm-mode export`
only emits constrained requests for broken pages and low-confidence local
alignment choices; `apply` accepts only existing subtitle, scene, and block IDs.
