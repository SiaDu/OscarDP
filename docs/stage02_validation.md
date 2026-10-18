# Stage02 Validator v1.1.1 (fixed annotation schema v1.0)

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
| Source text | Whole-cue reconstruction or complete explicitly typed source components, restricted normalization with evidence; language mismatches are errors |
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
numbers remain in the parse result; v1.1 can validate a reference only if exactly one
duplicate occurrence satisfies text and authoritative shot overlap. The duplicate
source remains invalid. Well-formed cues in an otherwise invalid file are still checked.

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
beyond these transformations are **Mismatch** errors for whole-cue comparison; the
bounded mixed-component rules below additionally support valid field extraction.
A multi-cue merge with spaces
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

## Validator v1.1: source integrity and annotation consistency

The fixed Stage02 data schema and historical annotation prompt are unchanged.
v1.1 adds a separate **Source Integrity** result, **Annotation Consistency** result,
`validation_complete`, and independent root-problem counts. The original overall
PASS/WARNING/FAIL contract remains: any error or incomplete required check fails.
An annotation can consistently quote one unambiguous occurrence in an invalid source;
that does not make the source valid or the movie PASS.

- **Source Integrity** checks discovery, decoding, original numbering, nonempty
  captions and valid source timing. Unreferenced empty/zero-duration cues remain
  source errors. Nonempty invalid-duration captions are retained as evidence; their
  timing is never guessed or shifted.
- **Annotation Consistency** checks the unchanged structural/Stage01 contract and
  whether existing entries reproduce source content in the correct explicitly
  marked field. This is deterministic marker-based checking, not a semantic review
  of unmarked lyrics, speaker identity, screenplay claims or actual audio/video.
- **Incomplete validation** identifies blocked checks. Missing/unsupported source
  content blocks index/text/time verification; it does not manufacture missing-index
  errors for every entry. Detailed parser diagnostics are retained even when a
  complete parsing failure is grouped into a single source error.

When a duplicated source number has exactly one occurrence matching both the entry's
text and authoritative Stage01 overlap, that entry can be validated with its original
source block number recorded. The duplicate source still causes an ERROR. If the
occurrence cannot be uniquely determined, downstream text/time checks remain blocked.

### Complete cues and mixed components

Whole-cue exact/normalized comparisons remain supported. For mixed captions,
extraction accepts only these additional, documented units:

1. A complete span enclosed by paired `♪`/`♫` markers (lyrics), a complete bracketed
   or parenthesized SDH annotation (sound/context), or a complete residual dialogue
   span bounded by those explicit annotations.
2. The complete projection of all components belonging to one field, in source
   order. A multi-cue dialogue merge must still follow consecutive original source
   cue positions; it cannot select arbitrary words or skip language within a component.
3. Source turn delimiters at component boundaries, explicit speaker-label metadata
   (e.g. `[John]`, `MARK:`, `MIKE and CLAIRE:`), and an annotation's following colon
   are distinguished from spoken words. Negative-number signs and punctuation
   inside language remain significant. Standalone instrumental `♪ ♪` is sound,
   not invented lyric words.

No arbitrary substring, fuzzy text, case change, punctuation substitution or semantic
paraphrase is accepted. Omitting part of an ordinary unsegmented line fails. The report
records each source span, full raw cue, rendered source, extraction rule and supplied
index. Span offsets refer to rendered text after presentation-tag handling, not raw
HTML byte offsets. Complete parenthetical/context annotations prove textual provenance;
they do not prove that a sound was actually audible. Source markers alone cannot
classify unmarked text semantically.

A full mixed cue copied into `song_lyrics` can match the source exactly and still
produce `FIELD_CLASSIFICATION` when it contains ordinary spoken dialogue outside the
music markers. Explicit lyrics assigned to dialogue/sound are also errors. Speaker
labels and source contextual annotations accompanying lyrics are not treated as
additional spoken dialogue. These checks can expose genuine problems in movies
that passed v1.0, and those new errors are retained.

### Root problems and occurrence evidence

`issues.csv` preserves every occurrence with `issue_id`, `root_problem_id`, `domain`,
field and entry location. `root_problems.csv` and each movie's `root_problems` group
identical source/index or text/field problems across shots while listing every affected
shot. `errors`/`warnings` remain occurrence counts for compatibility;
`root_errors`/`root_warnings` count independent problems. Distinct field errors or
materially different temporal evidence are not collapsed merely to reduce counts.

### Reproduce the baseline review without overwriting it

```bash
uv run python -m oscardp.stage02 validate --expect-movies 18 \
  --report-root reports/stage02_validation_v1_1

uv run python -m oscardp.stage02 review \
  --baseline-root reports/stage02_validation \
  --validation-root reports/stage02_validation_v1_1 \
  --report-root reports/stage02_validation_v1_1/review \
  --historical-baseline reports/stage02_validation_initial
```

The last argument is optional and adds an earlier first-run comparison. For reruns,
choose fresh validation and review directories. The review checks baseline CSV/JSON
agreement and verifies every original Stage01/Stage02/SRT SHA-256 against both runs,
then rechecks input/report hashes after investigation. A stale input hash prevents
claiming that the review applies to the baseline data.

The review writes:

- `REVIEW.md` and `review_summary.json`: every movie, source/annotation statuses,
  occurrence/root counts and the actual on-disk baseline counts.
- `issue_resolution.json/.csv`: one disposition per baseline ERROR/WARNING,
  original source evidence, decision, approval needs and duplicate/root links.
- `historical_first_run_resolution.json`: optional earlier-run occurrence comparison,
  including dependent parser/index cascades under a single unresolved source root.
- `current_findings.json` and `unresolved.json`: current independent problems,
  newly detected classification issues and all pending approval/source actions.
- `source_investigations.json`: original paths/hashes, rejected raw blocks,
  references, neighboring cues, source proposals and complete original-index maps.
- `proposed_sources/` and `source_scenarios/`: separately hashed, **unapproved**
  derivative SRTs and their independent validation. Only evidenced advertising
  duplicates and unreferenced empty captions are proposed for removal; retained
  cue numbers are unchanged. Nonempty zero-duration cues require authoritative
  timing and are never silently deleted.
- `proposed_stage02_patches.json`: unapplied JSON Patch proposals with exact before/
  after text, source evidence, source-file SHA-256 and `test` guards. Missing
  complementary dialogue/lyrics are preserved through companion-entry proposals
  (unknown speakers remain `null`). Consolidated `file_proposals` resolve overlapping
  shot changes with exact before/after arrays; individual text proposals remain
  supporting evidence. All bundles need human confirmation before application.
- `verification.json`: original input and baseline/current report hash audit.

The six dispositions are `RESOLVED_VALIDATOR_RULE`, `RESOLVED_SOURCE_CURATION`,
`STAGE02_CORRECTION_PROPOSED`, `ACCEPTED_FORMAT_DIFFERENCE`, `NEEDS_HUMAN_REVIEW`
and `UNRESOLVED`. `RESOLVED_SOURCE_CURATION` is explicitly **conditional**: it applies
only to the validated candidate derivative, requires confirmation, and does not
change the original-source movie result. Accepted formatting differences remain
visible as WARNINGs; no actual language mismatch is downgraded to obtain a PASS.

WebVTT content/time signatures may be cross-linked diagnostically to Stage02
references, but those links cannot prove original subtitle numbering. Without an
authoritative numbered source or authenticated original index map, recovery remains
UNRESOLVED; the tool never produces a guessed, sequentially numbered replacement.


## Validator v1.1.1: bounded presentation fixes and manual adjudication

The six v1.2 support gaps now accept a whitespace-separated next-turn hyphen
immediately before a paired lyric span, and leading `MAN 2:` / `WOMAN:` SDH
labels in complete residual segments. Original label evidence remains in segment
rules; it never supplies a character identity. Retained complete SDH presentations
remain traceable. Internal hyphens, interruption punctuation, negative signs and
source language are preserved. Arbitrary substring matching is still prohibited.

An optional `--manual-semantic-decisions PATH` reads an object with a `decisions`
array. Each record requires `decision_id`, `movie_id`, positive `srt_index`,
`original_srt_sha256`, exact complete `source_text`, exact complete segment `text`,
`classification` (`dialogue`, `song_lyrics` or `sound_cues`), `affected_shots`,
`source` equal to `user_confirmed_audiovisual_review`, and nonempty `evidence`.
Dialogue records without confirmed identity require `speaker: null`.
Only a unique matching original cue and complete source segment may bind.
A binding error fails validation; an unavailable source leaves adjudication incomplete.
The override applies only in listed shots. The confirmed segment must occur exactly
once in its confirmed field in each listed shot; legitimate repetition across shots
is allowed. Other cues and shots gain no exemption. Schema, coverage, ordering,
source integrity and temporal checks always run independently.

Reports identify bound decisions, their canonical digest and comparisons supported
by them. This is user audiovisual evidence, never a claim that Codex watched the
film. Automatic source consistency can pass an ordinary utterance without proving
its semantic function. Source Integrity evaluates the original subtitle file;
Annotation Consistency evaluates annotations against that source and any explicitly
supplied decisions. Neither implies that every annotation has been semantically
reviewed. Phase A previews apply guarded patches only in memory; file-level
proposals require separate user authorization before any Phase B application.
