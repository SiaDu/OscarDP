# Stage05.6 Event Review, Scene Integration & Stage06 Interface v1

This module prepares local evidence and an immutable candidate packet. It does
not detect events, call an LLM, infer emotion/intent or verify acting evidence.
No Stage01–05.5 output is edited. Generated media stay under the git-ignored
`reports/` tree. No video, face crops, source frames or reviews are uploaded.

## Build and verify

Use the existing environment and Stage05 performance dependencies; no new
framework, model or download is needed. FFmpeg and FFprobe must be installed.

```sh
PYTHONPATH=src .venv/bin/python -m oscardp.stage05_review build \
  --event-run reports/stage05_5/tt12300742/pilot_001 \
  --output-dir reports/stage05_6/tt12300742/pilot_001 \
  --review-dir reports/stage05_6_reviews/tt12300742

PYTHONPATH=src .venv/bin/python -m oscardp.stage05_review verify \
  reports/stage05_6/tt12300742/pilot_001
```

Existing output directories are refused, including incomplete runs. Choose a
new run directory after a failure; do not overwrite previous evidence. Only the
explicit source run's selected Shots are processed. Source video is decoded
sequentially through the last requested index to preserve original frame order;
only the selected ranges are encoded. This prefix decode is not full-movie
batch processing or detector inference.

`run_config.json` records hashes of 184 inputs in the initial Bugonia pilot,
including the source movie, Stage01–04, corrected Stage05.2, Stage05.3,
Stage05.4 and all Stage05.5 files. The source-run fingerprint is SHA256 of the
canonical sorted input path/hash mapping. Input hashes are checked before and
after preparation. `artifact_sha256.json` protects the resulting review files.
Source-code hashes describe this implementation separately.

## Contracts and authority

The real Stage04 `4.0-prototype` stores `scene_id` and `character_candidates`
inside `shots`, with summary/location/participants/status inside `scenes`.
This adapter uses those fields and retains their contents and review statuses.
Stage04 old temporal context and Stage01 original boundaries remain separate
from the corrected `effective_processing_temporal` owned by Stage05.2.

Stage02 Dialogue Speaker is narrative attribution. It is neither visible
speaker truth nor a verified identity. Character mappings are separate
Stage04-derived candidates with `character_mapping_status=unverified` and
`track_to_stage03_subject_match=unverified`. Clusters remain tentative. The
pilot's 209/211/217 cluster has a MICHELLE candidate; 210/214 has TEDDY.
These are not recognized actor identities.

The packet hierarchy is Movie → Scene → Shot → Face Track → Identity Cluster
→ Event IDs. `event_candidates` provides each event's original numerical
summary, phases, censoring, QC, content hash and source evidence lookup.
Source measurement masks and complete row references remain in Stage05.5;
the viewer keeps an exact copy of the 25,012 processed trace rows, not in the
Stage06 LLM input. No overlapping AUs are semantically merged. Pose and gaze
are camera-relative within a Shot. No cross-shot interpolation is allowed.
Unprocessed ranges and per-feature invalid source frames are explicit.

`stage06_input_pilot.json` contains all 126 candidate references and a separate
list of 48 first-round review IDs. “Pilot” does not discard the other events.
It is a research input snapshot labeled `algorithm_candidate`, with no
Explainable Acting Annotation. Human reviews form a separate overlay; this
snapshot remains unreviewed even after someone later saves review records.

## Sampling

`stratified_temporal_diversity_v1` uses no random process, seed or amplitude
ranking. It includes all 4 eye closure, 8 head orientation and 7 gaze change
candidates, then 6 rejected, 5 algorithm-negative and 4 unsupported-AU45
intervals. Required QC strata include mode disagreement, censoring, boundary
proximity, invalid neighbors and source spikes. Remaining places cover AU
features, penalizing temporal overlap and crowded Shots. Ties use stable Shot,
frame and Review IDs. Every selected row records its sampling reason/version.
The full 205-row CSV is copied byte-for-byte from Stage05.5.

## Use the local viewer

```sh
PYTHONPATH=src .venv/bin/python -m oscardp.stage05_review serve \
  reports/stage05_6/tt12300742/pilot_001 --port 8766
```

Open <http://127.0.0.1:8766>. Use the Python review server, not `file://` or a
generic static server. It listens only on loopback. Stop with Ctrl-C.

Select Pilot or Full Queue, Shot and Review Item. The selected item shows its
event/feature/frame range, QC, dialogue, scene and candidate identity context.
The timeline is clickable. Use Event ±0.5 seconds to play source-time context;
use normal play for the entire Shot. Frame buttons show individually extracted
source stills, independent of approximate browser video seeks. The frame label
always distinguishes movie timestamps and preview time.

Each Shot has one silent H.264 review clip, one still per original frame and
a complete frame map. Spatial compression is for review only. No temporal
downsampling is used. Mapping preserves original zero-based indices, inclusive
start/exclusive end, movie time and next-frame time. CFR preview encoding has
its own `preview_fps`; a VFR source would still use explicit per-frame movie
timestamps rather than preview fps for measurement. The existing exact reader
checks decoded CFR PTS against the recorded timeline. FFprobe fully decodes
the output clips to verify their frame counts. Stage02 text is provided;
audio is not included in these clips.

Curve panels have separate scales. Eye raw values are EAR and processed values
are relative EAR; AU values are model intensities; head/gaze angles are radians.
Static traces are solid, dynamic AU traces dashed. Invalid frames break curves
and are marked red. The numeric table retains validity, raw/processed values,
source time and derived angular speed. It permits inspection outside the event
window, including censored and rejected/negative intervals.

## Independent human annotation

`reports/stage05_6_reviews/tt12300742/<source_run_sha256>.json` holds an
independent source-specific overlay. Initially its 205 records are blank:
reviewer/time/decisions null, pending, `human_verified=false`. The server never
writes algorithm outputs. Tests use temporary review directories only.

Each record contains `review_id`, nullable `source_event_id`,
`source_run_sha256`, `shot_id`, `track_id`, `reviewed_frame_range`,
`observed_event_label`, `event_presence_decision`, `temporal_boundary_decision`,
`evidence_quality`, `reviewer`, `reviewed_at`, `notes`, nullable
`suggested_boundaries`, `video_review_attested`, `review_status`, and
`human_verified`. Reviewed ranges are original inclusive/exclusive frame
indices, limited to effective Shot bounds. Null Event IDs support negative,
rejected and unsupported-AU45 intervals.

Presence: `present / absent / uncertain / insufficient_video_evidence`.
Boundary: `acceptable / needs_adjustment / not_assessable`, independent of
presence. Quality: `good / limited / insufficient`. Suggested boundaries are
nullable onset/apex/offset source frame indices and never edit the algorithm.
Use offset as exclusive end. Suggestions require `needs_adjustment`.

After actually watching the video, fill the form, enter the viewed range and
attest personal review. The explicit save sets reviewer/time and
`review_status=human_reviewed`; this tool continues to retain
`human_verified=false` because review completion is not a verified acting
evidence release. Presence/quality/boundary conclusions remain separate.

Save one completed item at a time; unreviewed items remain blank. Refreshing
restores disk progress. Export/import permits carrying partial progress.
Imports and saves validate every source fingerprint, Event/Review/Track ID,
decision and range. A stale revision returns HTTP 409 to prevent overwriting
another window's progress. Writes use atomic replacement and fsync. Do not
import synthetic test reviews into the real overlay.

The initial pilot still needs real human review for event presence and phases,
the merged early Shot217 eye cycle (35132–35160 requested audit window), all
four eye candidates versus AU45, static/dynamic disagreements, censored and
near-boundary events, rejected/negative intervals, and Shot214 invalid frame
34648. Experimental boundaries and character mappings need their own review;
event annotation cannot upgrade either status.

## Tests

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage05_review.py -q
PYTHONPATH=src .venv/bin/python -m pytest -q
node --check src/oscardp/stage05_review/viewer.js
```

HTTP tests require local loopback sockets. Data-specific acceptance checks skip
when the optional local research artifacts are absent. The standalone `verify`
command rehashes inputs, checks source/event/scene/identity/time/mask/queue
integrity, deterministic sampling and all encoded video frame counts. These
checks establish software and provenance integrity, not human event accuracy.
