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

For the binary v3-family candidate task, a v3.1 request-context experiment can
add bounded neighboring subtitles without changing target subtitle IDs,
candidate block IDs, the response schema, or the frozen v3 policy:

```bash
python -m oscardp.script_context prepare-review-context-v3-1 \
  --requests /path/to/pilot_requests.jsonl \
  --alignment /path/to/subtitle_script_alignment.jsonl \
  --output /path/to/pilot_requests.v3_1.jsonl \
  --radius 2

python -m oscardp.script_context prepare-openai-batch-v3 \
  --requests /path/to/pilot_requests.v3_1.jsonl \
  --annotation-policy /path/to/frozen_annotation_policy.md \
  --output /path/to/pilot_batch_input.v3_1.jsonl \
  --model "$OPENAI_MODEL"
```

The added `review_context` contains only non-target subtitle ID, text, and time
on each side of the target group. Its manifest records source/output hashes,
radius, unchanged target/candidate projections, and that no gold labels were
included. Nearby rows provide discourse context only and are never additional
resolution targets.

Reviewer policy experiments are separately versioned. The v3.2 policy path
keeps the original v3 request payload and binary response schema while changing
only generic reviewer instructions:

```bash
python -m oscardp.script_context prepare-openai-batch-v3-2-policy \
  --requests /path/to/frozen_pilot_requests.jsonl \
  --annotation-policy /path/to/frozen_annotation_policy.md \
  --output /path/to/pilot_batch_input.v3_2_policy.jsonl \
  --model "$OPENAI_MODEL"
```

Use `submit-openai-batch-v3-2-policy` for that exact artifact. Historical v3
and v3.1 Batch inputs retain their original validators and instructions.

A v3.3 request-context experiment can keep the v3.2 instructions and binary
candidate schema frozen while adding bounded screenplay action around supplied
dialogue candidates:

```bash
python -m oscardp.script_context prepare-review-action-context-v3-3 \
  --requests /path/to/frozen_pilot_requests.jsonl \
  --screenplay-context /path/to/movie_script_context.json \
  --output /path/to/pilot_requests.v3_3_action_context.jsonl \
  --radius 8 --max-actions 24

python -m oscardp.script_context prepare-openai-batch-v3-3-action-context \
  --requests /path/to/pilot_requests.v3_3_action_context.jsonl \
  --annotation-policy /path/to/frozen_annotation_policy.md \
  --output /path/to/pilot_batch_input.v3_3_action_context.jsonl \
  --model "$OPENAI_MODEL"
```

Action rows are explicitly non-selectable and never enter the response block-ID
enum. The augmentation manifest proves that subtitle targets and dialogue
candidate IDs are unchanged and that no reference or gold labels were added.
Use `submit-openai-batch-v3-3-action-context` only for this versioned input;
the version-specific validator rejects historical v3/v3.1/v3.2 payloads.

Independent calibration references are frozen before reviewer output and are
explicitly identified as provisional evidence rather than human gold. Validate
the self-contained reference, its exact source request projection, candidate
IDs, decision contracts, and frozen hashes before preparing a paid Batch:

```bash
python -m oscardp.script_context validate-independent-calibration-reference \
  --reference /path/to/calibration_reference_frozen.jsonl \
  --requests /path/to/pilot_requests.jsonl \
  --reference-manifest /path/to/calibration_reference_manifest.json \
  --output /path/to/calibration_reference_validation.json
```

After version-specific response validation, use
`evaluate-independent-calibration-v3` to report candidate-task, candidate
presence, exact-block, negative-discrimination, and sequence metrics. Its
numeric gate is necessary but not sufficient: promotion still requires an
error-class audit showing no unresolved systematic failure.

After a reviewer is globally promoted and frozen, production processing uses a
separate v3 lifecycle. It partitions pilot and remaining requests, prepares and
submits only the frozen production schema, revalidates each partition with the
binary v3 validator, reconstructs full request order, and writes tagged reviewed
outputs without touching deterministic inputs:

```bash
python -m oscardp.script_context prepare-openai-production-remaining-v3 \
  --full-requests /path/to/alignment_requests.jsonl \
  --pilot-requests /path/to/pilot_requests.jsonl \
  --output /path/to/remaining_requests.v3_2_production_1.jsonl \
  --manifest /path/to/remaining_manifest.v3_2_production_1.json \
  --reviewer-manifest /path/to/stage2_production_reviewer_v3_2.json

python -m oscardp.script_context prepare-openai-production-batch-v3 \
  --requests /path/to/remaining_requests.v3_2_production_1.jsonl \
  --reviewer-manifest /path/to/stage2_production_reviewer_v3_2.json \
  --output /path/to/remaining_batch_input.v3_2_production_1.jsonl

python -m oscardp.script_context submit-openai-production-batch-v3 \
  --batch-input /path/to/remaining_batch_input.v3_2_production_1.jsonl \
  --reviewer-manifest /path/to/stage2_production_reviewer_v3_2.json \
  --job-file /path/to/remaining_batch_job.v3_2_production_1.json \
  --confirm-submit
```

If a remaining set is too large for the account's enqueued-token limit, split
the immutable request JSONL before preparing Batch inputs. Packing is
deterministic and order-preserving; it records exact source, reviewer, and
chunk hashes. Submit and complete the chunks serially, validate each raw Batch
output against its exact chunk requests, and merge the validated responses in
manifest order:

```bash
python -m oscardp.script_context split-openai-production-requests-v3 \
  --requests /path/to/remaining_requests.v3_2_production_1.jsonl \
  --reviewer-manifest /path/to/stage2_production_reviewer_v3_2.json \
  --output-dir /path/to/chunked_retry_1 \
  --max-estimated-tokens 300000 \
  --max-requests 100

python -m oscardp.script_context merge-openai-production-response-chunks-v3 \
  --chunk-manifest /path/to/chunked_retry_1/chunk_manifest.v3_2_production_1.json \
  --chunk-response /path/to/chunk_001_result/validated_responses.jsonl \
  --chunk-response /path/to/chunk_002_result/validated_responses.jsonl \
  --reviewer-manifest /path/to/stage2_production_reviewer_v3_2.json \
  --output /path/to/remaining_validated_responses.v3_2_production_1.jsonl \
  --report /path/to/remaining_chunk_merge_report.v3_2_production_1.json
```

The byte-based token value is a conservative packing estimate, not a tokenizer
measurement. Each chunk still passes the normal version-aware local Batch QA;
do not resubmit a failed oversized input or run historical v1/v2 validation on
these binary candidate-task responses.

The production lifecycle also supports the frozen `v3.2-production.2`
manifest. That version keeps the production.1 model, prompt, request context,
and response JSON schema, but binds responses to the independently calibrated
`candidate_task_v3_structure_v2` hard-validation contract. Its outputs use the
separate `v3_2_production_2` tag. Production.1 chunks can be carried forward
only when the production.2 manifest records and verifies the exact inherited
production.1 manifest hash; arbitrary cross-version merging is rejected.

After the full v3 response set is applied, build the self-contained risk audit
and freeze final per-movie QC. The audit includes request-local candidates,
selected blocks, screenplay/sequence/subtitle context, shot/keyframe links,
downstream `no_candidate_match` classification, diagnostics, and empty human
adjudication fields. The finalizer re-hashes protected source, Stage 1, and
deterministic Stage 2 files and refuses to emit a `COMPLETE` manifest unless
response coverage and both deterministic/reviewed validation pass:

```bash
python -m oscardp.script_context build-openai-production-risk-audit-v3 \
  --requests /path/to/alignment_requests.jsonl \
  --validated-responses /path/to/full_validated_responses.v3_2_production_2.jsonl \
  --reviewed-alignment /path/to/subtitle_script_alignment.llm_reviewed_v3_2_production_2.jsonl \
  --reviewed-shot-context /path/to/shot_script_context.llm_reviewed_v3_2_production_2.jsonl \
  --screenplay-context /path/to/movie_script_context.json \
  --output /path/to/reviewed/v3_2_production_2/high_risk_audit.jsonl \
  --summary /path/to/reviewed/v3_2_production_2/high_risk_audit_summary.json

python -m oscardp.script_context finalize-openai-production-movie-v3 \
  --movie-key tt00000000 \
  --inventory /path/to/stage2_goal_inventory.json \
  --status /path/to/stage2_goal_status.json \
  --screenplay-context /path/to/movie_script_context.json \
  --deterministic-alignment /path/to/subtitle_script_alignment.jsonl \
  --deterministic-shot-context /path/to/shot_script_context.jsonl \
  --reviewed-alignment /path/to/subtitle_script_alignment.llm_reviewed_v3_2_production_2.jsonl \
  --reviewed-shot-context /path/to/shot_script_context.llm_reviewed_v3_2_production_2.jsonl \
  --shots /path/to/shots.jsonl \
  --requests /path/to/alignment_requests.jsonl \
  --validated-responses /path/to/full_validated_responses.v3_2_production_2.jsonl \
  --reviewer-manifest /path/to/stage2_production_reviewer_v3_2_validation_contract_v2.json \
  --lifecycle-report /path/to/full_merge_report.v3_2_production_2.json \
  --risk-audit /path/to/reviewed/v3_2_production_2/high_risk_audit.jsonl \
  --risk-summary /path/to/reviewed/v3_2_production_2/high_risk_audit_summary.json \
  --qc-report /path/to/reviewed/v3_2_production_2/qc_report.json \
  --manifest /path/to/reviewed/v3_2_production_2/manifest.json
```

Use `merge-openai-production-responses-v3` only after both partitions pass
v3 hard validation, then `apply-openai-production-responses-v3`. The latter
preserves each original binary resolution as provenance while emitting only
`subtitle_script_alignment.llm_reviewed_v3_2_production_1.jsonl` and
`shot_script_context.llm_reviewed_v3_2_production_1.jsonl`. Existing versioned
artifacts are never overwritten; a matching completed apply may only resume
after source hashes and output existence are rechecked.

Candidate-task v3 inputs have an explicit submission command and validator;
the historical command remains reserved for v1/v2 inputs:

```bash
python -m oscardp.script_context submit-openai-batch-v3 \
  --batch-input /path/to/pilot_batch_input_v3.jsonl \
  --job-file /path/to/pilot_batch_job_v3.json \
  --confirm-submit
```

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
  IDs, invalid block arrays, and internally unordered multi-block selections.
  The historical v1/v2 three-way contract additionally requires adjacent,
  same-scene blocks. Candidate-task v3 keeps ordered non-adjacent or cross-scene
  selections structurally valid because a single final-film subtitle can omit
  an intervening screenplay turn or span a scene cut; both patterns are
  explicit high-risk diagnostics and must enter the audit package.
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

### Stage 2.5.2 gold adjudication package

`build-openai-gold-adjudication` creates a local, self-contained human-review
package containing only candidate-task disagreements. Each pending row includes
nearby subtitles and deterministic mappings, the complete unmodified request
candidate list, screenplay context outside that list, gold and prediction
provenance, sequence diagnostics, and null human-decision fields. It never
changes frozen gold or model responses. `validate-openai-gold-adjudication`
checks the selected population, candidate/source consistency, pending human
fields, and source hashes recorded in the artifact manifest.

```bash
python -m oscardp.script_context build-openai-gold-adjudication \
  --gold /path/to/pilot_gold_filled.jsonl \
  --validated-responses /path/to/validated_responses.jsonl \
  --requests /path/to/pilot_requests.jsonl \
  --manifest /path/to/pilot_manifest.json \
  --screenplay-context /path/to/movie_script_context.json \
  --alignment /path/to/subtitle_script_alignment.jsonl \
  --evaluation /path/to/pilot_evaluation_stage251.json \
  --disagreements /path/to/pilot_disagreements_stage251.jsonl \
  --output-dir /path/to/gold_adjudication_stage252
```

The generated annotation policy is intentionally a TODO template. Diagnostic
tags are non-final review aids and never populate adjudication labels.

### Stage 2.5.3 policy-aware candidate pilot

The versioned v3 workflow leaves historical v1/v2 schemas untouched. Its
response decision is binary: `match` selects one or more request-local dialogue
IDs, while `no_candidate_match` selects none. It applies annotation policy v1
at screenplay-turn level: prefer the most specific proposition, allow
fragmentation/repetition/reordering, reject merely related new propositions,
and treat graphic or insert text as non-dialogue.

```bash
python -m oscardp.script_context prepare-openai-batch-v3 \
  --requests /path/to/pilot_requests.jsonl \
  --annotation-policy /path/to/annotation_policy_v1.md \
  --output /path/to/pilot_batch_input_v3.jsonl \
  --model gpt-5.6-terra
```

Preparation is local-only and creates no job. Every line retains a
request-specific strict candidate enum. Future v3 responses use
`validate-openai-responses-v3` and `evaluate-openai-pilot-v3`. The evaluator
reports all provisional records and a resolved-gold view excluding only
explicitly ambiguous adjudications; acceptance uses the resolved view without
deleting the provisional record.

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
