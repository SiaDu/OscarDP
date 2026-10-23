# Stage05.3 cross-shot identity association pilot

This recognition-only pilot consumes actual observed YuNet detections and tracks
from an explicitly corrected Stage05.2 run. It writes independent, nameless,
tentative identity clusters. No character, actor, expression, emotion, gaze or
performance identity is verified. It does not alter Stage01–04, Stage05.2,
boundary reviews, or source media, and never runs a detector.

## Reproduce

Recognition requires the optional `stage05-identity` dependencies. Install with
`uv sync --extra stage05-identity` when preparing a new environment. It does not
require replacing PyTorch, installing the full InsightFace Python stack or
downloading weights. The actual pilot used numpy 2.5.3, OpenCV 4.14.0, PyTorch
2.14.0+cu130 (unused), ONNX 1.23.2 and ONNX Runtime 1.31.0. The first three
versions were preserved; only ONNX Runtime/ONNX and their missing dependencies
were added.

Use an existing, explicitly chosen local InsightFace recognition model:

```sh
uv run --no-sync python -m oscardp.stage05 associate-identities \
  --movie-id tt12300742 \
  --source-run reports/stage05_2_corrected/tt12300742 \
  --shot-ids shot_000209 shot_000210 shot_000211 shot_000214 shot_000217 \
  --model models/insightface/models/buffalo_l/w600k_r50.onnx \
  --provider CPUExecutionProvider \
  --output-dir reports/stage05_3/tt12300742/pilot_002
```

`pilot_001` contains the completed first run. Choose a new directory to repeat;
existing output directories are always refused, even for dry runs. `--dry-run`
validates input provenance, model and candidate selection without source-frame
decode or report creation. CPU is the default provider; requesting unavailable
CUDA fails. All parameters are recorded; see CLI `--help` for quality and
association controls. There is no whole-movie default.

```sh
uv run --no-sync python scripts/verify_stage05_identity_pilot.py \
  reports/stage05_3/tt12300742/pilot_001
uv run --no-sync python -m pytest -q
```

## Inputs and provenance

Four source files are hashed: `face_tracks.json`, `face_detections.jsonl`,
`run_config.json`, `qc_report.json`. The loader checks movie ID, the boundary
sidecar SHA256, Stage01 SHA256, video identity, matching corrected temporal
metadata across files, actual observed detections, counts and endpoints.
Original-range runs and inconsistent provenance fail before decoding.

The present pilot intentionally accepts only the existing
`experimental_model_reviewed_pending_human` correction with explicitly applied
shot IDs, pending model reviews and false human verification/approval. This is
not a general approved-release reader. Boundary metadata, including known
conflicts with uncorrected neighbors, is copied unchanged. Every sample and
track records both `original_temporal` and `effective_processing_temporal`.
Global indices and the exact Stage05.2 timestamp rule are retained.

## Sampling, alignment and recognition

Up to 16 candidates per track are selected from evenly distributed observation
bins, choosing the highest detection score within each bin, then native face
size and earliest frame as tie breakers. Bins partition observations; heavily
missing observations can make their elapsed-time coverage uneven. Rejected
candidates are recorded and are not silently replaced with adjacent frames.

The unchanged Stage05 exact FFmpeg reader decodes source frames sequentially;
no approximate seek, QC-video inputs or Stage03 representative stills are used.
Sample timestamps must agree with Stage05.2 observations. YuNet's five points
are fitted to the official ArcFace 112-pixel template with a least-squares
similarity transform. A small ONNX Runtime adapter follows the official
[InsightFace ArcFace preprocessing](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/arcface_onnx.py)
and [alignment template](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py).
The InsightFace Python package is not installed or invoked. All tracks use the
same local `buffalo_l/w600k_r50.onnx` weights, RGB, 112×112 input, mean/std 127.5
for these weights and normalized 512-dimensional output. Model path, SHA256,
runtime version and actual provider are saved.

Quality gates include detection score ≥0.85, native box minimum edge ≥80 px,
aligned Laplacian variance ≥30, valid landmarks and eye/nose/mouth ordering,
native eye separation ≥20 px, fit RMSE ≤10 px and source-supported aligned crop
fraction ≥0.95. Landmark geometry offers limited pose/occlusion clues; it does
not establish unoccluded facial regions or head pose. High confidence does not
override crop rejection. No enhancement or face restoration is performed.

## Association policy

All accepted sample vectors are preserved. Track representations are normalized
means. Within-track pairwise mean/median/q10/minimum and minimum sample-to-center
cosine remain available. Fewer than three accepted samples yield insufficient
evidence; median <0.65 or q10 <0.50 yields an inconsistent track, ineligible for
association. Zero, NaN, infinite, dimension-mismatched or mixed-model vectors
fail rather than becoming identity evidence.

Defaults are uncalibrated research candidate gates, fixed before the pilot:

| Condition | Default |
|---|---|
| Same candidate | Center cosine ≥0.55 + margin 0.05; cross-sample q10 ≥0.45 |
| Different candidate | Center cosine ≤0.25 − margin 0.05; cross-sample q90 ≤0.40 |
| Ambiguous | Eligible evidence between decision gates |
| Insufficient evidence | Either track fails sample count/consistency gates |

Pairwise outputs include counts, internal consistency, cross-sample statistics,
threshold margins and review flags. These thresholds were not transferred from
SFace and do not claim a calibrated verification error rate. Similarity margin
means distance from a decision threshold, not a calibrated confidence score.
Complete-link merging requires **every** cross-member pair to pass the same
candidate gate. A≈B and B≈C cannot conceal conflicting A–C evidence. Rejected
transitive merges are recorded. Singletons remain unknown. Same-shot pairs are
not merged by this cross-shot pilot.

## Outputs and review

`identity_clusters.json` holds source/model references, matrix, track consistency,
cluster members and supporting pairs. Every cluster is tentative or unknown,
with `verified_identity=false`, `human_verified=false`, `review_needed=true` and
empty `character_candidates`.

`track_embeddings.npz` contains `sample_embeddings` (N×512), `sample_ids`,
`sample_track_ids`, `track_embeddings` (M×512), `track_ids` and one `model_sha256`.
Load with `allow_pickle=False`. `embedding_samples.jsonl` maps accepted samples
to `embedding_npz_row`, global frame/time, quality, source ranges and aligned
crop hash/path. Rejected candidates have null NPZ rows and explicit reasons.
`pairwise_similarity.csv`, `qc_report.json`, `run_config.json`, 112×112 PNG crops,
cluster Contact Sheet and closest cross-cluster comparisons complete the record.

The verification script independently checks saved vectors, aggregation,
consistency, matrix, complete-link support, source/crop hashes and corrected
observation correspondence. It does not rerun recognition or establish human
identity truth. A failed run retains `failure.json`; select a new directory.

## Limits and later character names

Five tracks from two visible people cannot calibrate a general identity system.
Temporal bins do not guarantee pose diversity. Crop-boundary filtering can bias
the retained interval; subtitles and background sometimes remain in the aligned
image. Their influence has not been ablated. The boundary draft still requires
human review; uncorrected neighboring shots retain known one-frame conflicts.

Later name association should be an independent versioned mapping keyed by
identity-run fingerprint and cluster ID. Store candidate character name, source
Stage04 reference, supporting evidence, confidence/uncertainty and review status
as `unverified`. Stage04 semantic context and subtitles must not alter embedding
vectors or cosine scores. Keep multiple candidates or unknown when evidence is
insufficient; preserve the original detection and clustering files.
