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

### Stage 2.1 deterministic repair

Stage 2.1 uses monotonic anchors and token spans so consecutive subtitle
fragments can reference the same long screenplay block. It also exports grouped,
locally constrained review requests and `review/alignment_diagnostics.json`.

### Stage 2.4 sparse-anchor candidate retrieval

When reliable anchors are missing or span a wide screenplay interval, review
export now uses deterministic lexical retrieval over a bounded fallback range.
It keeps the strongest dialogue candidates and their adjacent dialogue blocks in
global screenplay order; action blocks are never offered to the reviewer.
Candidate generation records its interval, method, lexical score, and fallback
status, but never approves an alignment. Optional local semantic retrieval may
augment this candidate set only when `--semantic-model` is explicitly supplied.

The fallback range and final request size are configurable with
`--review-local-window`, `--review-fallback-window`, and
`--review-candidate-limit`. `prepare-openai-pilot` records both the full request
pool and selected-pilot distributions for normal, fallback, and insufficient
candidate windows, together with any material representativeness warning.

Stage 2.4.1 adds a pre-OpenAI screenplay-structure gate for residual narrative
text under active character cues, editorial object headings, and fragmented
parentheticals. The gate records affected blocks and prevents review export when
confirmed parser errors remain. Pilot manifests also record candidate-limit
saturation for the source pool and every selected request. Pilot selection is
diagnostic-balanced—not statistically representative—so evaluation reports raw
diagnostic and per-stratum accuracy separately from a source-weighted estimate.
Pilot acceptance uses complete-resolution exactness: the predicted decision
must equal the gold decision and the block-ID sets must be equal. The legacy
`block_set_exact_match` field retains its name for compatibility but follows
this combined definition; block-ID-only agreement is diagnostic and never
drives acceptance or source-weighted accuracy.

### Stage 2.2 optional OpenAI review

OpenAI review is an optional, separate workflow. Normal `process-one` never
imports the SDK or makes a paid request. Install support with:

```bash
.venv/bin/python -m pip install -e ".[context,openai,test]"
export OPENAI_API_KEY='...'
export OPENAI_MODEL='your-explicit-model-id'
```

Prepare a deterministic pilot and locally validated Responses Batch file:

```bash
python -m oscardp.script_context prepare-openai-pilot \
  --requests /path/to/review/alignment_requests.jsonl \
  --alignment /path/to/subtitle_script_alignment.jsonl \
  --output-dir /path/to/review/openai --count 30

python -m oscardp.script_context prepare-openai-batch \
  --requests /path/to/review/openai/pilot_requests.jsonl \
  --output /path/to/review/openai/pilot_batch_input.jsonl \
  --model "$OPENAI_MODEL"
```

The remaining lifecycle commands are `submit-openai-batch`,
`check-openai-batch`, `fetch-openai-batch`, `validate-openai-responses`,
`apply-openai-responses`, and `evaluate-openai-pilot`. Submission is the only
paid operation and refuses to run without `--confirm-submit` and
`OPENAI_API_KEY`.

Reviewed outputs use `.llm_reviewed.jsonl` names and never replace deterministic
alignment or shot-context files. Model output is constrained to request-local
IDs, structurally validated, and still requires human pilot evaluation before
any promotion.

### Stage 2.5 constrained pilot responses

Batch preparation now embeds a request-specific strict schema in every line.
`request_id` is fixed to that request and `block_ids` is an enum containing
exactly its supplied dialogue candidates. Subtitle order is still enforced by
local validation. The reviewer normally preserves screenplay order, but a
small same-scene backward move (three dialogue blocks by default) is accepted
only with the explicit `repeated_or_reordered_dialogue` basis. Distant and
cross-scene jumps remain invalid; `no_match` and `uncertain` do not move the
local cursor.

Validate a frozen pilot gold file without changing it:

```bash
python -m oscardp.script_context validate-openai-pilot-gold \
  --gold /path/to/pilot_gold_filled.jsonl \
  --requests /path/to/pilot_requests.jsonl \
  --output /path/to/pilot_gold_validation.json
```

This report separates malformed or foreign labels, ordinary monotonic
mappings, bounded final-cut repetitions/reorders, and unrepresentable sequence
movement. `prepare-openai-batch` remains local-only: it writes a Batch JSONL
and manifest but neither uploads nor submits them. The manifest records the
instruction hash and request-level schema hashes for controlled comparisons.

### Stage 2.5.1 validation layers

OpenAI pilot review uses three deliberately separate checks:

- **Hard validation** asks whether the response is a structurally legal,
  request-constrained response. It rejects malformed output, missing or
  reordered subtitle resolutions, invalid enums/confidence, foreign candidate
  IDs, invalid block arrays, and multi-block selections that are internally
  unordered, non-adjacent, or cross scenes.
- **Sequence-quality diagnostics** ask whether the alignment behavior looks
  suspicious. Backward mappings, missing `repeated_or_reordered_dialogue`
  basis, large jumps, and repeated block use are retained as warnings or
  high-risk events without changing the prediction.
- **Gold evaluation** asks whether a structurally legal prediction is correct.
  Selecting the wrong supplied candidate is an evaluation error, not a
  malformed API response.

Evaluation preserves the three-way `match`/`no_match`/`uncertain` history and
also reports a derived candidate task. In that view, `no_match` and `uncertain`
both mean that no supplied candidate should be selected; a `match` is correct
only when its block-ID set matches gold. This derived metric does not rewrite
gold or the active Batch response schema.

### Stage 2.3.1 reviewed-output QC

Reviewed shot context uses global screenplay block order. Every row includes
`scene_transition` and ordered `scene_candidates`. For a transition shot the
legacy `scene` field remains the primary scene, the alignment is explicitly
marked `scene_transition`, and `local_script_context` contains actions from the
primary scene only; actions from multiple scenes are never mixed in that field.
`scene_candidates` represents direct subtitle/script overlap evidence only.
Shots assigned by same-scene interpolation therefore use an empty
`scene_candidates` array; their inherited confidence remains in
`scene.confidence` and is not presented as synthetic overlap evidence.

Non-anchor span regressions are exported for human review rather than rejected
globally. `build-openai-human-audit-v2` creates a self-contained provisional
audit package with null human-label fields. After those fields are completed,
`apply-human-corrections` validates every edit against the original request
candidates and writes only a new explicitly tagged alignment and shot context.
Non-anchor audit schema 2.0 separately names the human `review_target` and the
sequence `regression_trigger`, so a previous LLM mapping selected for review is
never confused with the current subtitle that exposes the regression.
