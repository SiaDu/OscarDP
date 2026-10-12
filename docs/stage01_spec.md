# Stage 01 output contract

The official movie directory contains only `Stage01_<movie_id>.json` and `shot_img/*.jpg`. The JSON has exactly `movie_id`, `movie_name` (verified title or `null`), and ordered `shots`. Each shot has exactly `shot_id` and `temporal`; `temporal` has exactly `start`, `end`, `start_frame`, and `end_frame`. Frame indices are zero-based; `start_frame` is inclusive and `end_frame` exclusive. Consecutive shots share a boundary. Seconds are measured from the first decoded video frame. CFR uses frame index divided by the actual frame rate; VFR uses decoded PTS. The final end time is the exclusive endpoint.

The representative frame is `floor((start_frame + end_frame - 1) / 2)`. Its JPEG is named `shot_000001_HH-MM-SS-mmm.jpg`, using the frame's timestamp. Images are extracted in one sequential FFmpeg pass. TransNetV2 uses its official PyTorch inference model. Model weights are supplied with `--weights`.

Technical metadata and optional QC live under `<output-root>/.internal/<movie_id>/`. Interrupted work lives under `.work/<movie_id>/`. Published outputs are validated before resume skips them. `--overwrite` explicitly replaces an existing output and retains a backup under `.backup/`. A JSON title mapping supplied with `--titles` has the form `{"tt1234567": "Verified Title"}`. Unresolved titles remain `null`.

Stage 01 never writes shot clips or future-stage placeholders. Source videos and historical `oscar_movie_processed` and `oscar_movie_standardized` directories are untouched.
