# Stage05 Boundary Correction Sidecar v1 — proposal contract

This sidecar proposes processing ranges for Stage05 while preserving immutable
Stage01–04 records. It is not a corrected Stage01 release, new shot inventory,
identity annotation, or an automatic offset policy. No Stage05 reader consumes
it yet. The boundary audit must precede any decision to change production logic.

Store `Stage05_<movie_id>_boundary_corrections.proposed.json` beside an independent
audit report under `reports/stage05_2_boundary_audit/`, never in official movie
output directories. Schema: `schemas/stage05_boundary_correction_v1.schema.json`.

Each correction retains the stable `shot_id` and exact original temporal object:
`start`, `end`, `start_frame`, `end_frame`. A separate `proposed_processing_temporal`
records proposed bounds using the same zero-based, inclusive-start/exclusive-end
convention. Both edges require their own actual source-frame evidence. A measured
start offset does not by itself justify shifting the exclusive end.

For a hard cut whose last old-image frame is N, the first new-image frame is N+1.
If the current next-shot start is N, a candidate corrected boundary is N+1. When
both original edges have this issue, a range `[start,end)` becomes
`[start+1,end+1)`, preserving frame count. Simply trimming the start would omit
the final old-shot frame at the original end boundary. This is a per-edge
evidence rule, not permission to shift every shot. The opening frame and final
movie EOF are special endpoints and must never be moved beyond the movie.
Gradual transitions, multi-frame prediction runs and uncertain boundaries need
individual review; prediction argmax +1 is not a universal first-clean-frame rule.

`evidence` references the audit manifest and distinct start/end boundary IDs,
the actual global source-frame indices, and source-frame images. The audit
manifest records frame PTS, decoder agreement and native pixel hashes. It also
identifies the source movie and original Stage01 JSON so a stale sidecar can be
detected. This is lightweight traceability, not a frozen release system.

Review states are `proposed`, `model_reviewed_pending_human`, `approved`, or
`rejected`. An LLM examining actual decoded frames is still model review;
it must not self-assign human verification. Approval metadata identifies who,
when and what evidence was reviewed. A draft does not authorize processing.
Future application must be an explicit new run, record the sidecar identity and
the effective ranges in its configuration, preserve original-shot references,
and refuse conflicting corrections, unapproved entries, source mismatches,
out-of-range endpoints or unexplained overlaps/gaps. None of that application
logic is implemented by this audit.

Adjacent audited shots share one physical boundary. Reuse the same boundary ID
for the left exclusive end and right inclusive start. A proposed correction of
one selected shot must not silently edit unselected neighbors; disclose any
resulting overlap with an uncorrected neighbor in the plan. If downstream
processing uses only a subset, the stored Stage01 interval remains untouched.

## Evidence namespaces

Stage03 `shot_boundary_qc_v1` has only `sampled_images`, `user_video_report`, and
`human_video_review` bases and Early/Middle/Late roles. It lacks a distinct
Stage05 decoder-verified source-frame basis. A JSON file can satisfy that schema
while still causing provenance confusion if it is named Stage03 but generated
from Stage05 previews.

New Stage05 video-frame audit evidence belongs in
`Stage05_<movie_id>_boundary_audit.json`, with explicit `source_video_frames`,
actual frame indices/PTS, extraction method and model/human review status.
Do not extend the established Stage03 schema for this purpose. Preserve older
Stage03-named artifacts as history and issue a separate provenance erratum;
they must not be treated as evidence that the Stage01 Early/Middle/Late JPGs
were inspected, nor as proof of true interior cuts or human verification.

## Reproduction

The read-only extraction and decoder comparison are available via
`python -m oscardp.stage05.boundary_audit --help`. Run `--phase reference` into
a new directory, then `--phase compare` with the identical explicit boundary
selection. The first path writes a hash record for **every** decoded thumbnail
from frame zero through the final context frame. Selected native images are
anchored to those independently counted ordinals. The second path uses the
existing Stage05 reader and compares native BGR SHA256 and timestamps.
Both paths use FFmpeg codecs, so this excludes an application-index/select
shift but is not an independent codec implementation validation.

No detector, recognition model, TransNet inference, correction consumer or
Stage05.3 code is invoked. Generated Contact Sheets show N−2 through N+3;
N−1 / N / N+1 are explicitly labelled. Audit targets are selected boundary
windows, not whole-movie face tracking; precise no-seek decoding still incurs
the source prefix decode through the last selected boundary.
