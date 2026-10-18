# Stage02 v1.0 validation

The implemented CLI performs **Structural and Source Consistency Validation**.
Annotation generation remains manual interactive GPT-6 High work using the unchanged
[specification](stage02_spec_v1.md) and [prompt](prompts/stage02_prompt_v1.txt).
The validator does not read screenplay PDFs, call an LLM, decode video, or assess
character identity, narrative correctness, emotion, intent or observed performance.
Stages 03–06 remain planned.

## Run

Install Python 3.12+ dependencies with `uv sync --extra test`. This validator needs
neither a GPU nor FFmpeg. The existing project's Torch dependency remains for Stage01;
the validator never imports or runs it.

```bash
# Single movie first; separate report directory preserves this run.
uv run python -m oscardp.stage02 validate --movie-id tt0025316 \
  --report-root reports/stage02_validation_single

# Discover all canonical IMDb directories (no Stage01 --titles requirement here).
uv run python -m oscardp.stage02 validate --expect-movies 18

# Portable explicit roots and optional reviewed subtitle selection.
uv run python -m oscardp.stage02 validate \
  --analysis-root /path/to/oscar_movie_analysis \
  --srt-root /path/to/oscar_movie \
  --report-root /path/to/new/reports \
  --srt-map /path/to/confirmed_srts.json --tolerance 1e-6

uv run python -m oscardp.stage02 validate --help
uv run python -m pytest tests/test_stage02_validator.py
```

Defaults: analysis `/media/sdu/SiyaoDu/datasets/oscar_movie_analysis`, subtitles
`/media/sdu/SiyaoDu/datasets/oscar_movie`, reports `reports/stage02_validation`.
`--movie-id` restricts processing to one ID; a missing selected directory fails.
`--expect-movies` checks the number selected/discovered; a mismatch is a batch error
and still produces reports. Hidden/internal and symlinked movie directories are
excluded from automatic discovery. All canonical IMDb directories are included,
even if a required stage file is missing. Part files never substitute for complete files.

Input files are read only. Reports must be outside both dataset roots. Existing
reports are never overwritten: use a fresh `--report-root` for another run. No source
renames, subtitle time shifts or automatic repairs occur. Reports and actual source
data must not be committed. Exit codes: `0` for PASS or WARNING only, `1` for validation
failure after reports are saved, `2` for configuration/output errors. One movie's
failure does not stop later movies.

## Checks

| Rule group | Deterministic checks |
| --- | --- |
| Files and identity | Expected complete filenames, directory/root IMDb IDs, parseable UTF-8 JSON, unique JSON keys, finite numbers, valid titles and Stage01/02 title consistency |
| Stage01 contract | Root/shot shape, sequential stable IDs, zero-based contiguous exclusive frame ranges, contiguous second boundaries starting at zero; no JPEG/video verification |
| Stage02 schema | Required fields, strict nested types, prohibited extra fields, six narrative fields, structured action objects, snake_case tags/categories, source indices and optional cue shapes |
| Shot coverage | Equal counts, no duplicate/missing/extra IDs, identical ordering and coverage of the first and last upstream shots |
| Shot times | Finite nonnegative start, end greater than start, equality to authoritative Stage01 within absolute tolerance (default `1e-6` seconds; no relative tolerance) |
| SRT discovery | Unique source confirmed by IMDb path ownership or explicit user mapping |
| SRT parsing | Original positive numbers, valid comma-millisecond timestamps, positive cue duration, multiline text, UTF-8 BOM, LF/CRLF; malformed blocks and duplicate numbers are errors |
| Source indices | Each dialogue/sound/lyric reference exists uniquely; merged dialogue indices are unique and consecutive in original source cue order (original numbering gaps are preserved) |
| Source text | Exact source reconstruction or restricted normalization with evidence; mismatches are errors |
| Source overlap | Every referenced cue has positive overlap with the authoritative Stage01 interval, using exclusive endpoints; zero-length contact is not overlap |

The machine schema is [stage02_v1.schema.json](schemas/stage02_v1.schema.json).
A matching packaged resource supports installed wheels; a test enforces equality.
It expresses the existing complete-film contract, with no `pilot` field in a complete
result. `dialogue=[]`, `action=[]`, missing optional arrays and `speaker=null` are
valid. Source indices are positive JSON integers, not booleans or floating values.
`performance_cues.basis` is `screenplay_explicit` or `llm_inferred`, and verification
must be `screenplay_text_only_not_video_observation`. This checks the declared
provenance, not the truth of free-text descriptions. `time_of_day` and
`interior_exterior` remain strings because the spec gives examples rather than a closed enum.

## SRT selection

One recursive file index is built under `--srt-root`, skipping hidden/symlinked
entries. It reuses Stage01's IMDb pattern and video extensions and follows the
preflight's filename-then-nearest-parent IMDb precedence without importing its
workbook/rename workflow. An unnamed SRT in the same directory as an IMDb-identified
video can be associated with that video; shared-directory ownership across several
movie IDs is ambiguous. Multiple candidate SRTs yield `SRT_AMBIGUOUS`, no candidate
yields `SRT_NOT_FOUND`. No fuzzy or title-based selection occurs. Path identity cannot
prove which release/version a subtitle belongs to; users must confirm source versions.

An explicit mapping overrides discovery, including conflicts. Relative SRT paths
resolve against the mapping file's directory. Its existence records the user's source
choice; an unreadable mapped file still fails rather than falling back silently.

```json
{
  "tt0025316": "/absolute/path/to/confirmed.srt",
  "tt0318374": "subtitles/confirmed.srt"
}
```

Malformed or undecodable SRT files fail. WebVTT disguised as `.srt` is explicitly
rejected: numbering it would invent original source indices. UTF-8/UTF-8-BOM is supported; other legacy
encodings are reported as unreadable rather than guessed. Parsing retains original
cue numbers and multiline text, with only BOM/line-ending handling. Duplicate
numbers remain in the parse result; a reference to a duplicate cannot be uniquely
validated. Well-formed cues in an otherwise invalid file are still checked.

## Text and time evidence

For multiple consecutive cues, source text is reconstructed in supplied index order,
with a newline between cue bodies. Exact equality is an **Exact Match**. Otherwise,
the following transformations, and only these, may produce a **Normalized Match**:

- Remove recognised HTML presentation tags (`i`, `b`, `u`, `s`, `font`, `span`) and treat `br` as a line break.
- Remove explicit ASS alignment overrides ` {\an1}` through ` {\an9}`; other brace text is preserved.
- Decode HTML entities, normalize Unicode to NFC, and collapse/trim whitespace.

A normalized match is a WARNING. Case, punctuation, words, speaker labels, hyphens,
lyrics symbols and sound descriptions are preserved. There is no fuzzy similarity,
semantic paraphrase acceptance, offset estimation or timing correction. Differences
beyond these transformations are **Mismatch** errors. A multi-cue merge with spaces
instead of newlines is therefore normalized, not exact.

A cue can validly appear in several overlapping shots. There is no global uniqueness
rule for Stage02 cue use. Every merged constituent cue must overlap the containing
shot, and comparisons use Stage01 boundaries even when Stage02 times are wrong.
A definite lack of overlap is an ERROR; a separated cue additionally receives a
possible-offset/version WARNING for manual investigation. The timing serialization
tolerance does not widen subtitle overlap intervals.

The validator verifies sources of existing entries; it cannot infer which unused
subtitle cues should be ordinary dialogue, lyrics or sound, and does not require
all original subtitles to be annotated. It does not assess screenplay-only claims.

## Status and reports

- **PASS:** all required checks completed, no ERROR or WARNING. INFO does not affect status.
- **WARNING:** no ERROR, but formatting normalization or another review issue exists.
- **FAIL:** any ERROR or incomplete required check. Missing/ambiguous/unreadable SRT can never pass complete validation.

Unknown speakers and absent optional performance cues are INFO statistics. Rule
results distinguish PASS/WARNING/FAIL/INCOMPLETE; `incomplete_checks` lists checks
that could not be fully executed. An error and incompleteness can coexist.

Outputs:

```text
reports/stage02_validation/
  summary.csv
  summary.json
  issues.csv
  movies/tt1234567.json
```

`summary.csv` includes per-movie counts, coverage, SRT status, source match counts,
errors/warnings and status. Match counts include dialogue, sound and lyrics entries.
`summary.json` includes batch errors (e.g. expected movie count), overall status and
the scope/limitations. `issues.csv` includes severity, rule code, shot/index, expected
and actual evidence. Each movie report includes input paths and SHA-256 hashes,
schema hash, tolerance, rule results, incomplete checks, statistics and every source
text comparison with raw cue text/times and normalization reasons. CSV structured
cells use JSON serialization. Every JSON report and both CSV exports explicitly
state the validation scope and limitations.

A passing report is evidence of structural/source consistency, never proof that the
screenplay semantics or actual movie visuals have been evaluated correctly.
