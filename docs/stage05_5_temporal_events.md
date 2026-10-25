# Stage05.5 temporal performance event candidates v1

This stage consumes actual Stage05.4 numeric measurements. It produces
observable facial-action **candidates**, not human labels, emotion, acting
intent, decisions, social gaze targets or micro-expression truth. No detector,
recognition or OpenFace model runs, and no movie frames are decoded. Earlier
outputs are read-only. The taxonomy consists of `au_activation`,
`eye_closure_candidate`, `head_orientation_change` and
`camera_relative_gaze_change`.

## Reproduce

Uses existing `stage05-performance` Python dependencies (numpy, pyarrow,
matplotlib); no new model or environment changes are necessary.

```sh
uv run --no-sync python -m oscardp.stage05 detect-events \
  --movie-id tt12300742 \
  --measurement-run reports/stage05_4/tt12300742/pilot_001 \
  --shot-ids shot_000209 shot_000210 shot_000211 shot_000214 shot_000217 \
  --au-mode static --compare-dynamic \
  --output-dir reports/stage05_5/tt12300742/pilot_002
```

`pilot_001` is the real completed first run. A repeated output directory always
fails before source access, including dry runs. `--dry-run` validates inputs and
prints the frozen default/override configuration without creating outputs.
`--config PATH` accepts a full versioned threshold JSON or known numeric
parameter overrides. Unknown keys, versions, nonfinite values and inconsistent
hysteresis settings fail. `thresholds.json` records the resolved configuration
and its canonical SHA256. `run_config.json` fingerprints detector source files.

```sh
uv run --no-sync python scripts/verify_stage05_event_pilot.py \
  reports/stage05_5/tt12300742/pilot_001
uv run --no-sync python -m pytest -q
```

## Source preflight

The six requested Stage05.4 input files are hashed, and the entire measurement
run plus its earlier protected provenance are snapshotted. Stage05.4 has no
signed release manifest for its Parquet; this implementation does not invent
one. Instead it verifies recorded raw CSV SHA256, every Parquet feature against
CSV, source frame maps, structured status references, original source QC rules,
Track QC counts and mode comparison statistics. Current input hashes are then
frozen in the Stage05.5 provenance and checked again at completion.

Movie/schema versions, source Stage05.2/05.3 references and hashes, unique Track
IDs, original/effective ranges, tentative identity and experimental boundary
reviews must match. Every mode must retain the complete ordered source-frame
range and strictly monotonic movie timestamps. Source next-frame times must
match their neighbors. Parquet metadata must reference the same provenance and
static primary baseline. Static/dynamic Pose/Gaze duplicates must agree; they
are processed only from static rows. AU sensitivity uses dynamic rows separately.

The inherited AU, pose, gaze, landmarks and tracking masks are checked against
Stage05.4 and retained. A failed measurement is not repaired or upgraded. Shot
214 frame 34648 cannot contribute an event, phase or interpolated value.

## Time processing

All processing is confined to one Shot and Track. Any invalid frame breaks the
modality's continuous segment, whether the missing interval is brief or long.
Nonconsecutive frame indices and timestamp gaps >0.25 s also break segments.
There is no gap interpolation or merge across these boundaries. This is an
explicit rule for VFR pauses as well as missing data, not a 24-fps assumption.

Finite-window medians use original movie seconds, truncated at segment edges.
Centered filters use future samples within their declared radius and are offline
candidate extraction, not causal real-time detection. Defaults were fixed before
the first real detection and were not tuned to the desired event count.

## AU activation

For each actual `AUxx_r`, a median filter of radius 0.08 s is applied within
valid segments. Descriptive baseline = segment q20; it is not a neutral-face
estimate. High = max(0.9, baseline+0.55); low = max(0.45, baseline+0.25).
Above-low intervals containing a high seed form hysteresis proposals. Internal
below-low gaps up to 0.08 s may merge within the same valid segment. Keep only
≥3 frames, ≥0.12 s and peak-minus-baseline prominence ≥0.45.

Stable high segments (median ≥1.2, variation <0.45, duration ≥0.25 s) can instead
be sustained observations with no forced apex or onset/offset. AU occurrence is
additional evidence only and is not required to have a matching intensity
field. AU25/AU26 are marked speech-sensitive; speech itself is unverified.

The strongest processed maximum is retained with a ±0.05 plateau tolerance.
Nearby maxima remain one event until the specified below-low return; this
avoids subdividing each jitter peak. A declining phase is reported only when
the observed fall meets the prominence gate. Min/max and raw/processed traces
remain available. A high value already present at a segment start is
left-censored, and activity extending through its end is right-censored.

## Eye closure geometry

Compute eye aspect ratios separately from face points 36–41 and 42–47:
`(||p1-p5|| + ||p2-p4||) / (2||p0-p3||)`. These are image-left/right eyes.
Normalize each by its own Track q80 over usable landmark/tracking frames. This
is a descriptive geometric reference, not verified open-eye ground truth. A
reference below 0.05 yields insufficient reference evidence. No shared absolute
eye width threshold is introduced. The original landmark/tracking mask is used,
not the gaze-open-eye gate that would discard closures.

The eye median radius is only 0.025 s. A relative opening ≤0.70 can seed a
partial closure; retain until both eyes recover to ≥0.85. At least two frames
must support the drop and duration must reach 0.045 s. Bilateral opening ≤0.35
at a common frame supports `complete_closure_candidate`; otherwise classify
`partial_closure`. Missing preceding opening/reopening yields an incomplete,
uncertain cycle with the unobserved phase null.

Static AU45 intensity/occurrence supports rather than defines the event.
Geometry without AU45 support, static/dynamic peak differences and AU-only
activation intervals enter QC/review. No event is called a verified blink.
Slow narrowing, pose-related foreshortening and short closure can remain
ambiguous. Fixed recovery hysteresis can merge repeated local dips if the
intervening recovery never reaches both relative references; the first pilot
exhibits this in the specifically requested Shot 217 window. Do not reinterpret
that merged candidate as a blink count.

## Head orientation and camera-relative gaze

Head uses the same OpenFace `Rx*Ry*Rz`, left-handed-positive relative rotation
convention, with median radius 0.06 s. Geodesic orientation differences divided
by actual timestamp intervals give angular speed. High/low = 0.25/0.10 rad/s,
minimum duration 0.12 s, three frames and net displacement 0.10 rad. Within-valid
speed dips up to 0.08 s can merge. Record before/after angles, geodesic net
displacement, peak/median speed, all-axis displacement/velocity, dominant axis
and descriptive residual/velocity stability. No nod/agreement label is inferred.

Gaze uses its two measured angles to reconstruct normalized camera-reference
directions. Radius 0.06 s; high/low speed = 0.45/0.18 rad/s; minimum duration
0.08 s, two frames and net direction displacement 0.10 rad; merge gap 0.04 s.
Keep measured per-eye vectors as additional before/after evidence. Head
displacement over the interval and overlapping head events are recorded. All
gaze events retain `head_motion_confound_possible`; ≥0.07 rad concurrent head
displacement or detected overlap adds simultaneous-motion QC. Head motion is
not subtracted, eye-only motion is never asserted and no gaze target is named.

Heuristic intrinsics and model outputs remain relative estimates. A net
displacement gate can miss out-and-back motion, and temporal medians can remove
brief genuine changes; these limitations are explicit rather than repaired
with semantic inference.

## Event contract and censoring

The JSON schema is `docs/schemas/stage05_temporal_events_v1.schema.json`.
Events sort by Shot, Track, start, type, feature and end, receiving reproducible
`event_000001` IDs and content hashes. They reference config/code/source hashes,
original source Parquet rows, processing trace keys, numerical evidence and
separate source-measurement versus event-detection QC. `rule_support` is not a
probability. Every record remains `algorithm_candidate`, `human_verified=false`,
with tentative/unknown identity and experimental boundary state.

Start frames are inclusive and end frames exclusive. Event time uses the first
source PTS and the final included frame's next PTS. Onset is an observed
threshold-entry candidate; offset is the first valid below-low/recovered frame
after the active interval and can equal its exclusive end frame. If either lies
outside the valid segment, it is null with censoring. An apex plateau is an
interval; a plateau does not need a fabricated single apex frame. Motion apex
means maximum model angular speed and eye apex means minimum opening, not an
independently annotated acting-action apex. Even a segment-edge observed maximum
may be only a censored portion of a larger action.

## Output and independent review

- `performance_events.json` / `event_candidates.jsonl`: identical static-primary
  candidate stream. Pose/gaze are counted once.
- `processed_timeseries.parquet`: 25,012 long-form rows in the first run,
  including raw/filtered values, validity, source rows and motion speeds. Eye
  raw values are EAR; processed values are relative EAR, with references in
  sensitivity/eye metadata. Invalid raw values remain auditable; processed
  values at invalid frames are null.
- `sensitivity_analysis.json`: dynamic AU candidates and interval overlaps,
  explicitly outside the formal stream; Track eye references and AU-only QC.
- `rejected_candidates.json`: seeded proposals failing duration/prominence/net
  displacement. This is not a count of every rejected nonzero measurement.
- `event_review_queue.csv`: all candidates, rejection intervals, algorithm-negative
  intervals per Track/family and unsupported AU45 intervals. Low-flag samples
  are not assumed true, and negatives are not negative ground truth. Reviewer
  is null, status pending independent review, human verification false.
- `known_interval_review.json`: the requested 35132–35160 inclusive numerical
  audit, generated **after** detection; no expected event count influences rules.
- `stage06_event_preview.json`: structured Track/Identity/Event references only;
  annotation generation false. It does not implement Stage06.
- `event_qc_report.json`, `run_config.json`, `thresholds.json`, SHA256 manifest
  and four raw/filtered/event/mask/review timeline PNGs complete provenance.

The verifier rechecks original inputs and artifact fingerprints, replays every
detector with the same config, and compares event IDs/content, all processing
traces, rejections, sensitivity, review queue and reference-only preview. Its
result demonstrates reproducibility/integrity, not human event accuracy.
