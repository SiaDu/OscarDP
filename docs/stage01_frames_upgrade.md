# Stage01 frozen-selected image upgrade

This command never invokes TransNetV2, loads model weights, uses a GPU, or replaces
movie directories. It verifies the manifest SHA-256, release ID against its parent
directory, inactive FROZEN status, eligible selected IMDb IDs, selected frozen input
hashes and matching canonical Stage01/Stage02/SRT hashes. `--expect-movies` is
mandatory. Selection comes exclusively from the manifest; `--movie-id` and
`--exclude-movie-id` are mutually exclusive. The pinned release exclusion below must
select exactly 17 of 18 films. New-film inference batches still require `--titles`.

The original Stage01 video metadata and fingerprint must exist. Source resolution
uses their exact path and unambiguous IMDb discovery, never similar titles. Original
size and nanosecond mtime must match, together with media dimensions, codec, FPS,
decoded frame count, CFR/VFR classification and shot-boundary timestamps. A current
source SHA-256 is recorded and checked on recovery. Historical fingerprints did not
record video SHA-256, so the original identity evidence is the recorded path/IMDb,
size/mtime and matching decoded timeline; the new hash is not claimed as a historical
content hash. Missing evidence, mismatches and ambiguity fail before image writes.
Missing PTS fails safely with unavailable timing. Source movies are read only.

All selected films are preflighted before any image changes. Dry-run writes reports
only, including every image's path, dimensions, size, hash, planned rename mapping,
role indices, decoded timestamps and missing extraction indices. Missing/corrupt
Middle images, unknown files and filename collisions stop the operation. Reports
must be outside source, workspace and frozen roots. All writable paths reject
symlinks. Concurrent upgrades cooperate through an exclusive workspace lock.

## Progress display

Existing commands display progress by default in both dry-run and apply. Movie
labels distinguish manifest verification, preflight and application, for example
`[预检 3/17] tt0025316` and `[应用 3/17] tt0025316`. Each long operation has its
own progress: source SHA-256 counts bytes with MiB/s, image checks count files,
and the full decoded timestamp scan counts frames. Apply also reports Middle
migration, sample publication, final validation and cache cleanup. Repeated source
hash checks are displayed too.

During extraction, the frame counter tracks the sequential decoder even between
widely separated requested samples. A separate `samples acquired/required` count
includes verified cached samples. A complete cache does not start a decoder, and
completed movie upgrades explicitly display `已验证，跳过` after verification.

Interactive terminals use an in-place progress bar with percentage, elapsed time,
speed and per-stage ETA when a total is known. Unknown totals show the stage,
completed count and elapsed time without estimating an ETA. Movie counts show batch
position; there is no weighted overall percentage or whole-batch ETA. Updates are
throttled to one every 0.5 seconds, with immediate stage starts, finishes and failures.
When stderr is redirected, progress becomes plain newline-separated logs without
terminal escape codes. The final JSON remains exclusively on stdout; progress is
written and flushed to stderr. Failures identify the active movie and stage.

Add `--no-progress` to any `frames-upgrade` command to disable progress in either
mode. This changes no selection, validation, migration, recovery or report behavior.
Existing running commands keep their original behavior; progress appears in newly
started commands after this update. No extra video reads or decode passes are
introduced for progress reporting.

The following commands are examples for manual execution. They have **not** been
run against the real dataset during implementation.

Single-film dry-run:

```bash
uv run python -m oscardp.shots frames-upgrade \
  --input-root /media/sdu/SiyaoDu/datasets/oscar_movie \
  --output-root /media/sdu/SiyaoDu/datasets/oscar_movie_analysis \
  --frozen-manifest /media/sdu/SiyaoDu/datasets/oscardp_releases/oscardp_stage02_v1_0_18movies_20261018/frozen_manifest.json \
  --expected-manifest-sha256 375a58f5f94fbaae1036efa790475b794d7e60130b9686f96e7dfb2fbf44b443 \
  --movie-id tt12300742 --expect-movies 1 \
  --report-root reports/stage01_frames_tt12300742 --dry-run
```

Single-film apply after reviewing the report:

```bash
uv run python -m oscardp.shots frames-upgrade \
  --input-root /media/sdu/SiyaoDu/datasets/oscar_movie \
  --output-root /media/sdu/SiyaoDu/datasets/oscar_movie_analysis \
  --frozen-manifest /media/sdu/SiyaoDu/datasets/oscardp_releases/oscardp_stage02_v1_0_18movies_20261018/frozen_manifest.json \
  --expected-manifest-sha256 375a58f5f94fbaae1036efa790475b794d7e60130b9686f96e7dfb2fbf44b443 \
  --movie-id tt12300742 --expect-movies 1 \
  --report-root reports/stage01_frames_tt12300742 --apply
```

The other 17 films, dry-run:

```bash
uv run python -m oscardp.shots frames-upgrade \
  --input-root /media/sdu/SiyaoDu/datasets/oscar_movie \
  --output-root /media/sdu/SiyaoDu/datasets/oscar_movie_analysis \
  --frozen-manifest /media/sdu/SiyaoDu/datasets/oscardp_releases/oscardp_stage02_v1_0_18movies_20261018/frozen_manifest.json \
  --expected-manifest-sha256 375a58f5f94fbaae1036efa790475b794d7e60130b9686f96e7dfb2fbf44b443 \
  --exclude-movie-id tt12300742 --expect-movies 17 \
  --report-root reports/stage01_frames_17 --dry-run
```

The other 17 films, apply after reviewing the report:

```bash
uv run python -m oscardp.shots frames-upgrade \
  --input-root /media/sdu/SiyaoDu/datasets/oscar_movie \
  --output-root /media/sdu/SiyaoDu/datasets/oscar_movie_analysis \
  --frozen-manifest /media/sdu/SiyaoDu/datasets/oscardp_releases/oscardp_stage02_v1_0_18movies_20261018/frozen_manifest.json \
  --expected-manifest-sha256 375a58f5f94fbaae1036efa790475b794d7e60130b9686f96e7dfb2fbf44b443 \
  --exclude-movie-id tt12300742 --expect-movies 17 \
  --report-root reports/stage01_frames_17 --apply
```

Readiness check:

```bash
uv run python -m oscardp.shots validate \
  --output-root /media/sdu/SiyaoDu/datasets/oscar_movie_analysis \
  --frame-profile three_sample
```

## Publication, interruption and recovery

`initial_plan.json` retains the first inventory; `latest_plan.json` records current
preflight state, and `upgrade_report.json` records completed results or an apply
failure, including final sample records and their hashes. Each film also has a durable `.internal/<id>/frame_upgrade_journal.json`
with original inventory, frozen/source bindings and original-to-new paths/hashes.
`frame_upgrade_receipts.jsonl` appends fsynced per-image extraction hashes before
publication; it avoids rewriting the full sample record for every selected frame.
A torn final append is uncommitted and can be repaired on resume.
The final lightweight record is `.internal/<id>/frame_samples_v1.json`. Preserve
these files with the dataset and retain the report directory.

Middle publication makes a byte-identical temporary copy under `.internal/`,
fsyncs and verifies its hash, then atomically renames that copy into `shot_img/`
with Linux `renameat2(RENAME_NOREPLACE)`. This works on exFAT without hard links.
The original legacy file remains until every sample is verified. Extraction uses `.internal/<id>/frame_upgrade_work/`; each cached JPEG is
fsynced and its exact-index hash committed before publication. Journal entries bind
expected new hashes before the destination becomes visible. Existing files must
match the journal; even correctly named images are rejected without evidence.
After all samples pass integrity/hash checks and protected inputs still match, old
legacy files are removed. Original bytes remain at the new Middle path, recoverable
using the durable reverse mapping. No duplicate legacy set remains after success.
The temporary extraction and publication caches are removed after completion.

After a crash or error, re-run the **same apply command**. Verified destinations are
skipped, interrupted extraction reuses its persistent cache, and a complete upgrade
skips FFmpeg entirely. Do not clear the journal or manually replace files to bypass
a mismatch. A changed manifest, protected input, source fingerprint, source hash,
metadata or image fails safely and requires investigation. Changed source paths can
also block recovery even when bytes match; retain original paths during an upgrade.

For manual rollback, first stop writers and review each journal mapping. Verify the
new Middle hash equals `original_sha256`, then restore its `original_path` with
no-clobber publication. The same bytes suffice; re-encoding is unnecessary. Any
removal of extracted frames must be limited to exact journal-owned paths with
matching hashes. The command deliberately has no bulk-delete rollback option.

Atomic no-clobber rename and directory fsync must be supported by the output filesystem. Failure
stops safely; there is no silent overwrite fallback. The lock serializes cooperating
commands; external tools must not edit inputs or images during the upgrade. Large
videos need time for source hashing, timeline scanning and software decoding. Raw
RGB pipe bandwidth can limit throughput; only synthetic integration videos have
been benchmarked functionally, not the real 18-film dataset.

The external volume was observed mounted read-only during development. Apply needs
a writable workspace on that volume; this command does not change mount settings.
