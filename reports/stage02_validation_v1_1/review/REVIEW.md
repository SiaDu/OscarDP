# Stage02 Validator v1.1 review

Structural and Source Consistency Validation. Does not validate screenplay semantics, character identity, emotion, intent or actual movie visuals.

The required on-disk baseline counts are {'PASS': 8, 'WARNING': 2, 'FAIL': 8}. No baseline files were rewritten. The separately audited historical first-run counts are {'PASS': 8, 'WARNING': 1, 'FAIL': 9}; the primary review uses the on-disk baseline.

v1.1 original-source results: {'PASS': 7, 'WARNING': 2, 'FAIL': 9}.

Reviewed 746 baseline ERROR/WARNING occurrences, 517 independent roots and 229 duplicate occurrences.

## Movie results

| Movie | Baseline → v1.1 | Source integrity | Annotation consistency | Root errors / warnings | Patch proposals |
| --- | --- | --- | --- | --- | --- |
| tt0025316 It Happened One Night | FAIL → FAIL | FAIL | PASS | 1 / 0 | 0 |
| tt0025878 The Thin Man | PASS → PASS | PASS | PASS | 0 / 0 | 0 |
| tt0318374 The Cooler | PASS → PASS | PASS | PASS | 0 / 0 | 0 |
| tt11563598 A Complete Unknown | FAIL → FAIL | PASS | FAIL | 7 / 0 | 8 |
| tt12300742 Bugonia | WARNING → WARNING | PASS | WARNING | 0 / 249 | 0 |
| tt1262426 Wicked | PASS → FAIL | PASS | FAIL | 12 / 0 | 24 |
| tt1312221 Frankenstein | PASS → PASS | PASS | PASS | 0 / 0 | 0 |
| tt14905854 Hamnet | PASS → PASS | PASS | PASS | 0 / 0 | 0 |
| tt14961016 I'm Still Here | FAIL → FAIL | FAIL | PASS | 1 / 0 | 0 |
| tt17526714 The Substance | PASS → PASS | PASS | PASS | 0 / 0 | 0 |
| tt18382850 If I Had Legs I'd Kick You | FAIL → FAIL | FAIL | INCOMPLETE | 1 / 0 | 0 |
| tt26581740 Weapons | FAIL → FAIL | FAIL | PASS | 2 / 0 | 0 |
| tt27714581 Sentimental Value | FAIL → FAIL | FAIL | PASS | 2 / 0 | 0 |
| tt27847051 The Secret Agent | PASS → PASS | PASS | PASS | 0 / 0 | 0 |
| tt30343021 Song Sung Blue | FAIL → FAIL | PASS | FAIL | 19 / 1 | 25 |
| tt31193180 Sinners | PASS → FAIL | PASS | FAIL | 8 / 0 | 13 |
| tt32536315 Blue Moon | WARNING → WARNING | PASS | WARNING | 0 / 141 | 0 |
| tt32916440 Marty Supreme | FAIL → PASS | PASS | PASS | 0 / 0 | 0 |

## Baseline issue dispositions

- RESOLVED_VALIDATOR_RULE: 157
- RESOLVED_SOURCE_CURATION: 4
- STAGE02_CORRECTION_PROPOSED: 0
- ACCEPTED_FORMAT_DIFFERENCE: 582
- NEEDS_HUMAN_REVIEW: 2
- UNRESOLVED: 1

Every occurrence, old rule, source text, affected shot, decision and duplicate link is in issue_resolution.json/.csv.

## Source investigations and pending approval

Derivatives and candidate validation only; no promotion or original/Stage02 edits. RESOLVED_SOURCE_CURATION is conditional on approval and applies only to the candidate scenario.

### tt0025316

Source: /media/sdu/SiyaoDu/datasets/oscar_movie/tt0025316_ItHappenedOneNight/tt0025316.srt; SHA-256: 979e9f8d98e2e2b454302e8a5ce4430beb821c0ebfad150eaa11dda9e31d8e10.

- Unapproved derivative: `/home/sdu/Desktop/OscarDP/reports/stage02_validation_v1_1/review/proposed_sources/tt0025316.srt`. Candidate result: PASS; source: PASS. Removed original block numbers: [1447]. Original numbering retained for all other cues.

### tt14961016

Source: /media/sdu/SiyaoDu/datasets/oscar_movie/tt14961016_ImStillHere/tt14961016.srt; SHA-256: 4dd14cbe63cb2aeb2a2aec305eb90733899fdc0fb231714d2e446fb512315302.

- Unapproved derivative: `/home/sdu/Desktop/OscarDP/reports/stage02_validation_v1_1/review/proposed_sources/tt14961016.srt`. Candidate result: PASS; source: PASS. Removed original block numbers: [1714]. Original numbering retained for all other cues.

### tt18382850

Source: /media/sdu/SiyaoDu/datasets/oscar_movie/tt18382850_IfIHadLegsIdKickYou/tt18382850.srt; SHA-256: 3001257873b7238fc52aa1f49d41695dd598b5eadeb4dd85ee4903c3abcac9cc.

- RECOVER_AUTHORITATIVE_NUMBERED_SOURCE: No cue identifiers survive in the source. Stage02 numbering and content/time links cannot prove original SRT index provenance; sequential numbering is not authorized.
- 2491 timed blocks; 2068 distinct time/text signatures. Diagnostic cross-links do not establish original indices. No renumbered SRT was generated.
- 0 numeric cue identifiers; 2048 Stage02 reference indices have a unique text/time candidate, 19 have multiple candidates, 0 have no candidate. Even unique content/time links do not establish original numbering.

### tt26581740

Source: /media/sdu/SiyaoDu/datasets/oscar_movie/tt26581740_Weapons/tt26581740.srt; SHA-256: f0eb05a407f2195dd5a65ea0033675baf10b47c8bb653f50d0f1a62db27358f6.

- Unapproved derivative: `/home/sdu/Desktop/OscarDP/reports/stage02_validation_v1_1/review/proposed_sources/tt26581740.srt`. Candidate result: PASS; source: PASS. Removed original block numbers: [1, 2]. Original numbering retained for all other cues.

### tt27714581

Source: /media/sdu/SiyaoDu/datasets/oscar_movie/tt27714581_SentimentalValue/tt27714581.srt; SHA-256: 61ae31ad5ace99d1abaf192fd16d47089f452e58930531a71a1af70882cb6be2.

- RESTORE_TIMESTAMPS_FROM_AUTHORITATIVE_SOURCE: Nonempty zero/invalid-duration caption; no timing authority. Do not delete meaningful text or infer offsets.
- RESTORE_TIMESTAMPS_FROM_AUTHORITATIVE_SOURCE: Nonempty zero/invalid-duration caption; no timing authority. Do not delete meaningful text or infer offsets.

## Newly detected classification problems

46 independent field-classification roots were absent from the baseline rules. These remain errors; counts are not minimized to force passing movies.

proposed_stage02_patches.json contains 70 text proposals consolidated into 4 file bundles, with 62 companion entries preserving displaced dialogue/lyrics. Exact before/after arrays, source evidence, file hashes and JSON Patch test guards are included. No patch has been applied. Null companion speakers require contextual review.

## Verification and artifacts

- verification.json records all 54 input hashes before/after, baseline/current report fingerprints and CSV/JSON evidence agreement.
- source_investigations.json preserves rejected cue blocks, neighboring timing evidence, candidate-source lineage and complete original index maps.
- current_findings.json and unresolved.json retain current error roots, affected shots and pending source/annotation actions.
- Source candidate scenarios are separate from the 18-movie original-source validation. The baseline, original SRTs and Stage02 JSON remain unchanged.
