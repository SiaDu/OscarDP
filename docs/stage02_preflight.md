# Stage02 Source Preflight CLI v1.0

This deterministic, read-only quality gate runs **before** manual GPT-6 High annotation.
It neither generates Stage02 nor replaces Validator v1.1.1. The fixed Stage02 Schema v1.0,
specification and annotation prompts are unchanged. Inventory/Excel auditing remains in
`tools/dataset_preflight.py`.

## Run

Install Python 3.12+ and project dependencies with `uv sync --extra test`. PDF text-layer
inspection uses the lightweight `pypdf>=6,<7` dependency; no OCR, GPU, video decoding or LLM.

```bash
uv run python -m oscardp.stage02 preflight \
  --stage01 /path/to/Stage01_tt1234567.json \
  --srt /path/to/tt1234567.srt \
  --screenplay /path/to/tt1234567.pdf \
  --report-root reports/stage02_preflight/tt1234567

# Optional: inspect compatibility after a source change. Use a fresh report directory.
uv run python -m oscardp.stage02 preflight \
  --stage01 /path/to/Stage01_tt1234567.json \
  --srt /path/to/tt1234567.srt --screenplay /path/to/tt1234567.pdf \
  --stage02 /path/to/Stage02_tt1234567.json \
  --baseline-manifest /path/to/previous_validator_movie_report.json \
  --report-root reports/stage02_preflight/tt1234567_compatibility

uv run python -m oscardp.stage02 preflight --help
uv run python -m pytest
```

Required options are `--stage01`, `--srt`, `--screenplay`, `--report-root`.
Optional `--stage02` accepts an existing complete annotation, `--approved-exceptions`
accepts the approval format below, and repeatable `--baseline-manifest` compares explicit
historical fingerprints. Paths are portable; there is no automatic dataset scan or hardcoded
movie/Cue exception. Reports must be outside every input's directory, and outside any
baseline/approval input. Existing nonempty reports and symlink report targets are rejected.

## Gate and exit codes

| Quality status | Meaning | LLM annotation allowed |
| --- | --- | --- |
| PASS | All required checks completed without findings | Yes |
| WARNING | No hard blocker, but source risk needs review | Only after all necessary issue-specific approvals bind successfully |
| BLOCKED | Integrity/structure problem, invalid approval, or incomplete required check | No |

The JSON independently records `preflight_status`, `llm_annotation_allowed` and `exit_code`.
Exit `0` means admitted, `1` means blocked or awaiting required review, `2` means
configuration/output/execution failure. Source read/parse failures are reported BLOCKED,
not presented as a missing optional check. A warning remains WARNING even after approval.

## Executed checks

| Input/check | Deterministic rules and disposition |
| --- | --- |
| Stage01 | Shared strict JSON reader rejects duplicate keys and non-finite numbers. Canonical IMDb identity must match `Stage01_<movie_id>.json` and any IMDb parent. Fixed root/shot shape, ordered sequential unique IDs, finite nonnegative contiguous times from zero, and contiguous inclusive-start/exclusive-end integer frame ranges reuse `MovieValidation.shots/times`. Failures block. Last valid Stage01 boundary is the authoritative movie extent; the CLI does not independently inspect video. |
| Subtitle format | Content detection identifies SRT, WebVTT including dot-millisecond/headerless WebVTT, unsupported content and unreadable/damaged files. UTF-8 with/without BOM is supported by the shared SRT parser; other BOMs and undecodable bytes block without guessed conversion. Encoding, BOM, CRLF/LF/CR counts and presentation marker counts are recorded. Mixed newline styles require review. |
| Subtitle integrity | Shared `parse_srt` retains original Cue IDs, text and order. Missing/nonpositive IDs, duplicates with all affected blocks, malformed separators/timestamps, reversed time and nonempty zero-duration cues block. Even a disambiguated reference does not waive a duplicated source ID. An empty, valid-timed Cue without a reference in a supplied Stage02 is WARNING; referenced or invalid-timed empty cues block. With no Stage02, the report explicitly states references were not checked. Nothing is deleted. |
| Advertising | URL/domain and explicit promotional patterns are heuristics. `SRT_SUSPECT_ADVERTISING` includes the exact matched strings and affected block. It requires review, cannot establish that text is non-plot, and never authorizes automatic deletion. |
| Subtitle timeline | Rules below inspect source block order, neighboring times and movie extent. No timing shift, interpolation or replacement timing is generated. |
| Screenplay | Actual PDF header, parseability, encryption, page count and text extraction on every page. Encrypted/damaged PDFs block. Extraction failures leave the check incomplete. A minimally usable text layer needs at least 200 alphanumeric characters total and a page with at least 100; otherwise BLOCKED, possibly scanned/textless. If most pages have fewer than 20 alphanumeric characters despite meeting the total threshold, WARNING. Counts and per-page text hashes are recorded, not the screenplay text. |
| Optional Stage02 | Fixed Schema, identity, Stage01 shot coverage/times, original source indices, complete bounded source text and overlap reuse existing Validator methods. An out-of-shot cue yields `STAGE02_STALE_CUE_REFERENCE` with the current Cue, affected shot and Stage01 overlap candidates. Missing IDs/text mismatches block. Candidates are diagnostic only; no dialogue movement or narrative edit occurs. This optional check does not run or become required without `--stage02`. |
| Provenance | Resolved actual paths, bytes, mtime UTC/ns, SHA-256, Python/PDF/parser versions and implementation hashes. Selected inputs are hashed again before admission. A source or review-input change during execution blocks admission. |

The exact movie/subtitle edition and screenplay/movie semantic match remain `UNVERIFIED`.
There is no `SCRIPT_MATCH_PASS`. These limitations alone do not manufacture warnings for
every otherwise valid movie, but observed range/format conflicts require review or block.

### Timeline policy v1.0

Thresholds are fixed and included in `subtitle_timeline.thresholds` for reproducibility:

- A backward start jump of **at least 30 seconds**, with the current Cue ending at/before
  the previous Cue starts, is `SRT_TIMELINE_BACKWARD_JUMP` (BLOCKED). Nonoverlapping
  regressions of **at least 2 seconds** but less than 30 seconds need review.
- A Cue at least 30 seconds before/after **both** neighbors, whose neighbor bridge is at
  most 30 seconds, is `SRT_TIMELINE_ISOLATED_OUTLIER` (BLOCKED). This catches an isolated
  mistaken hour/digit such as the historical 01:30 → 00:30 → 01:30 pattern without
  hardcoding IMDb IDs, Cue numbers or an inferred correction.
- Normal temporal overlaps and nested captions are allowed. The rule is **not**
  `previous.end <= current.start`; overlapping unordered captions do not trigger the
  backward-jump check simply because their starts regress.
- Any Cue ending beyond Stage01 duration by more than `1e-6` seconds is reported.
  A Cue beginning at/after movie end, or overrunning by at least 30 seconds, blocks;
  a smaller end overrun needs review. Cue durations above **120 seconds** need review.
- For movies over 300 seconds with at least three valid timed Cues, a first Cue after
  80% of movie duration or a last Cue before 50% flags a possible offset/incomplete track.
  Sparse legitimate subtitles may trigger this heuristic; no global offset is calculated.

Output includes the anomalous Cue, preceding/following source blocks and measured deltas.
This detects evidence of a conflict, not an authoritative correction. Audio alignment,
consistent shifts staying inside the movie range, small digit errors, translation quality,
and narrative truth may require external version evidence or audiovisual review.

### Source version comparison

Nothing searches the user's report history implicitly. Supply `--baseline-manifest` to
compare a prior `source_preflight.json`, a per-movie Validator JSON (`summary.movie_id`
and `inputs`), or a release manifest's `movies` entries using
`original_srt_path` / `original_srt_sha256` (and corresponding Stage01/Stage02 or curated
fields). Paths and movie identity must match the actual selected input. Relative historical
paths resolve from the manifest's directory. Batch summary rows without input hashes do
not establish provenance.

Every mismatch produces `SOURCE_VERSION_CHANGED` with old/new SHA-256. An unmatched
baseline needs review. No baseline means `NOT_COMPARED`, not an assertion that historical
bytes match. User-confirmed on-disk edits remain the current source version: the CLI
does not restore historical SRTs or update earlier manifests. Supplying an old Stage02
against a changed timestamp can independently cause stale-reference BLOCKED findings.

## Approval records

First run preflight, inspect the exact findings and hashes, and create a separate file:

```json
{
  "schema_version": "1.0",
  "exceptions": [
    {
      "issue_id": "COPY_EXACT_ISSUE_ID_FROM_REPORT",
      "rule_code": "SRT_EMPTY_CUE",
      "movie_id": "tt1234567",
      "input_sha256": {
        "Stage01": "COPY_64_LOWERCASE_HEX_CHARACTERS",
        "SRT": "COPY_64_LOWERCASE_HEX_CHARACTERS",
        "Screenplay": "COPY_64_LOWERCASE_HEX_CHARACTERS"
      },
      "approved_by": "Reviewer's name",
      "evidence": "Specific evidence that this exact valid-timed empty cue is harmless; no meaningful text removed.",
      "decision": "APPROVED_NONBLOCKING"
    }
  ]
}
```

The sample hashes/ID are placeholders. Use real values and rerun to a fresh report root
with `--approved-exceptions /path/to/reviewed_exceptions.json`. Add the **Stage02** SHA-256
if `--stage02` was selected; the key set must exactly match all selected input files.
Each warning requiring review needs its own record. Issue IDs derive from the version,
rule, input hash and precise evidence, never execution time or issue order.

Wrong movie/hash/issue/rule, duplicate records, blanket decisions, and attempted blocker
waivers yield `APPROVAL_BINDING_INVALID` and keep the gate closed. Malformed approval
configuration returns exit 2. These are explicit reviewer declarations with checked
scope and binding, not cryptographic proof of who wrote the approval file. Even an
authoritative timing review cannot waive an invalid on-disk Cue; a separately authorized
valid source version must be selected before the gate can pass.

Preflight's approved harmless-empty-cue disposition does **not** modify Validator v1.1.1's
standards or make the same original SRT pass post-annotation validation. Any curated source
change still requires separate authorization and a traceable release.

## Reports and privacy

```text
reports/stage02_preflight/<movie_id>/
  PREFLIGHT.md
  source_preflight.json
  issues.csv
```

JSON records input provenance, Stage01 quality, actual SRT format/integrity, subtitle
timeline, PDF readability, optional compatibility, source versions, all issues and evidence,
approval bindings, executed rule groups, completeness and the independent admission gate.
CSV preserves one row per issue. Markdown summarizes the decision and limitations.

Only anomalous blocks and relevant reference comparisons include caption evidence; long
malformed blocks are truncated at 4000 characters with full-block hashes and lengths.
The CLI never copies an entire normal subtitle or PDF text into a report. Keep audit reports
under the Git-ignored `reports/` directory and do not publish source language or large reports.
It never writes original SRT/PDF/Stage01/Stage02, Phase B copies, prompts, schemas or specs.
No automatic repair, renumbering, OCR, Narrative Beat, Curated Release or Pipeline Activation.
