# Stage 01 output contract

Stage01 writes only `Stage01_<movie_id>.json` and `shot_img/*.jpg` in its official movie directory. In the Canonical Workspace, the same directory may also contain the current `Stage02_<movie_id>.json`, future validated `Stage03_<movie_id>.json` and `Stage04_<movie_id>.json`, `<movie_id>.srt` and optional `stage02_curated/` history; Stage01 validates its own outputs without treating these named coexisting entries as errors. The JSON has exactly `movie_id`, `movie_name` (verified title or `null`), and ordered `shots`. Each shot has exactly `shot_id` and `temporal`; `temporal` has exactly `start`, `end`, `start_frame`, and `end_frame`. Frame indices are zero-based; `start_frame` is inclusive and `end_frame` exclusive. Consecutive shots share a boundary. Seconds are measured from the first decoded video frame. CFR uses frame index divided by the actual frame rate; VFR uses decoded PTS. The final end time is the exclusive endpoint.

The historical representative frame is `floor((start_frame + end_frame - 1) / 2)`. Legacy output used `shot_000001_HH-MM-SS-mmm.jpg`. New output uses the three-frame profile described below, preserving this exact Middle frame. Images are extracted in one sequential FFmpeg pass. TransNetV2 uses its official PyTorch inference model. Model weights are supplied with `--weights`.

Technical metadata and optional QC live under `<output-root>/.internal/<movie_id>/`. Interrupted work lives under `.work/<movie_id>/`. Published outputs are validated before resume skips them. `--overwrite` replaces only a Stage01-only movie directory and retains a backup under `.backup/`; directories containing downstream or unknown entries are protected. A JSON title mapping supplied with `--titles` has the form `{"tt1234567": "Verified Title"}`. Unresolved titles remain `null`.

Stage 01 never writes shot clips or future-stage placeholders. Source videos and historical `oscar_movie_processed` and `oscar_movie_standardized` directories are untouched.

Stage 01 discovery accepts `.mp4`, `.mkv`, `.mov`, `.m4v`, `.avi`, `.webm` and `.rmvb`. Batch processing requires `--titles data/metadata/movie_titles.json`; the keys select exactly which IMDb IDs to process and the values provide `movie_name`. The mapping is validated before batch discovery, and unlisted videos are excluded before duplicate checks. RMVB streams may have variable decoded PTS even when their nominal frame rate is about 24 fps; Stage 01 uses decoded PTS when the frame spacing is variable.

## Three representative frames (sampling policy v1)

New processing writes `shot_000001_early.jpg`, `shot_000001_middle.jpg` and
`shot_000001_late.jpg` in the existing `shot_img/` directory. There are no timestamp
suffixes. The movie JSON and its schema remain unchanged.

Let `s=start_frame`, `e=end_frame`, `N=e-s`, and `m=(s+e-1)//2`:

- For `N>=3`: Early is `min(s+(N-1)//10, m-1)`, Middle is `m`, and Late is
  `max(s+9*(N-1)//10, m+1)`. This floors 10%/90% positions on the inclusive
  sample span, then enforces distinct ordered samples without crossing `e`.
- For `N=2`: Early aliases Middle at `s`; Late is `s+1`. Only `_middle.jpg`
  and `_late.jpg` exist.
- For `N=1`: Early and Late alias Middle at `s`; only `_middle.jpg` exists.

The shared function is used for inference outputs, upgrades and validation.
`<output-root>/.internal/<movie_id>/frame_samples_v1.json` records policy/version,
Stage01 hash, source identity, shot IDs, all requested roles, actual frame indices,
role aliases, relative paths, JPEG hashes, decoded timestamps relative to the first
video frame, timing source, rename mappings and completion status. This is technical
provenance, outside the official movie directory. No sampling fields are added to
Stage01 JSON or the frozen schema. Original inference fingerprints are preserved.

Extraction uses one software FFmpeg decode through the last missing frame. Fixed
size RGB frames are read in presentation order; only selected indices are JPEG
encoded. QC indices share this pass. This avoids a large select expression and its
per-frame evaluation cost, and uses memory bounded by a decoded frame. It trades
full-resolution pipe bandwidth for predictable selection cost; throughput on high
resolution movies depends on CPU and pipe bandwidth. No full-movie performance
claim is made from small integration tests. Extraction caches fsynced JPEG hashes
per exact frame index, never clears a supplied cache directory, and does not invoke
FFmpeg when every requested frame is already verified. After interruption, remaining
samples use a single decode from the start through the last missing frame (no
unsafe time seeking). VFR samples use decoded PTS, never nominal FPS interpolation.
A missing decoded PTS fails with unavailable timing rather than dropping a frame
and shifting subsequent indices.

`validate --frame-profile legacy` requires exactly one valid historical timestamp
JPEG per shot. `--frame-profile three_sample` requires the exact physical set,
policy indices, aliases, JPEG decoding, hashes, matching sampling metadata and
preserved Middle hashes from rename provenance. `auto` (default) recognizes both
profiles; malformed or incomplete new outputs do not fall back to legacy. Stage03
readiness must explicitly require `three_sample`. Boundary, timestamp and movie JSON
validation still apply in both profiles, including final decoded-frame checks.

See [frame upgrade instructions](stage01_frames_upgrade.md) for frozen-selected,
image-only upgrades. `process-one --overwrite` refuses a movie directory containing
Stage02, Stage03, Stage04, subtitles, curation history or any other content beyond Stage01 and
`shot_img/`; it is not an existing-movie upgrade mechanism.
