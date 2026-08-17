<!--
Frozen Stage 2 production-goal specification snapshot.
Source: /home/sia/.codex/attachments/0ebd8f3b-074a-49b9-b359-2a8827210100/pasted-text.txt
Source SHA-256: e3c524d90784a99dc81508a603d4bb77dedab44990bad0191fb7c782774d8bb5
Do not edit in place; record any later specification change as a new revision.
-->

GOAL: Finish the production-quality Stage 2 screenplay/subtitle alignment
pipeline for the remaining OscarDP movies, while autonomously improving and
versioning the reviewer only when evidence requires it.

Repository:
/home/sia/OscarDP
https://github.com/SiaDu/OscarDP

Current expected baseline commit:
33b6519ecb9f222bdb9bda12288b13651ae87a94

Dataset roots:

MOVIE_ROOT=/mnt/g/datasets/oscar_movie
SCRIPT_ROOT=/mnt/g/datasets/oscar_script
PROCESSED_ROOT=/mnt/g/datasets/oscar_movie_processed

Movies still to process:

tt12300742
tt1312221
tt14905854
tt18382850
tt27714581
tt30144839
tt30343021
tt31193180

Existing calibration / development movies:

tt32536315  Blue Moon
tt27847051  The Secret Agent

======================================================================
MISSION
======================================================================

Take the eight remaining movies from their CURRENT filesystem/repository state
to validated, versioned, reproducible Stage 2 outputs:

screenplay
    ↓
screenplay parsing
    ↓
subtitle parsing
    ↓
deterministic screenplay-subtitle alignment
    ↓
candidate retrieval / review requests
    ↓
policy-aware LLM candidate alignment where needed
    ↓
quality validation
    ↓
risk/adjudication review
    ↓
reviewed subtitle-script alignment
    ↓
shot-script context
    ↓
per-movie reproducibility/QC manifest

You are responsible for deciding:

- whether the current reviewer version is adequate;
- whether a new reviewer version is needed;
- what version number to use;
- whether an error comes from parser, retrieval, request context, prompt,
  response schema, validator, gold, or model behavior;
- what should be changed;
- whether another pilot is justified;
- whether a movie needs a full 30-item calibration or a smaller stratified
  pilot;
- when a reviewer version is stable enough to become the production version;
- when a movie is ready for its remaining/full Batch;
- when to stop modifying the global pipeline.

Do not wait for the user to tell you "make v3.1" or "make v3.2".

Make those engineering decisions from measured evidence.

However, obey all experimental-integrity and safety constraints below.

======================================================================
BOUNDARY OF THIS GOAL
======================================================================

This goal ends when all eight movies are either:

A. Stage-2 production complete;

or

B. explicitly marked BLOCKED with a concrete missing source/input or a small
   list of genuine unresolved human ambiguities.

Do NOT proceed to:

- OpenFace;
- facial AU extraction;
- actor clip extraction;
- face tracking;
- Explainable Acting Annotation;
- Stage 3+ dataset construction.

Those are outside this goal.

======================================================================
FIRST ACTION: INVENTORY, DO NOT ASSUME
======================================================================

Before modifying anything:

1. Verify git status and HEAD.

2. Inspect AGENTS.md and README.md.

3. Discover the actual filesystem state for every movie using its IMDb ID.

Search:

$MOVIE_ROOT
$SCRIPT_ROOT
$PROCESSED_ROOT

Do not assume filenames.

For each movie identify:

- video file;
- English subtitle file;
- screenplay PDF/text;
- shots.jsonl;
- movie_script_context.json;
- subtitle_script_alignment.jsonl;
- shot_script_context.jsonl;
- review/openai artifacts;
- any previous pilot;
- any previous manifests;
- any partial/failed outputs.

Build:

/mnt/g/datasets/oscar_movie_processed/stage2_goal_inventory.json

Do not regenerate artifacts merely because they already exist.

Hash important existing artifacts before touching the movie.

======================================================================
IMMUTABILITY
======================================================================

Treat existing source artifacts as immutable unless a new versioned artifact is
explicitly required.

Never overwrite:

- video;
- screenplay source;
- subtitle source;
- shots.jsonl;
- historical deterministic outputs;
- historical pilots;
- historical raw OpenAI outputs;
- frozen gold;
- historical v1/v2/v3 artifacts.

Use versioned outputs/tags.

Stage 1 shots.jsonl is immutable once validated.

Never silently "fix" historical files.

======================================================================
CURRENT REVIEWER STATE
======================================================================

The latest candidate-task architecture is v3:

decision:
- match
- no_candidate_match

It uses:

- request-specific candidate enums;
- foreign-ID prevention;
- separate hard validation and sequence diagnostics;
- policy-aware candidate matching;
- annotation_policy_v1;
- versioned v3 submission/validation/evaluation.

The Secret Agent v3 result is DEVELOPMENT EVIDENCE:

resolved gold:
92 / 103 candidate-task correct
= 0.8932038835

candidate-presence:
93 / 103
= 0.9029126214

gold no_candidate_match:
44 / 44 correct

gold match:
48 / 59 exact-block correct

structural:
30 / 30 requests valid
0 missing
0 foreign candidate IDs

Therefore:

The current v3 behavior is strongly conservative.

It is excellent at rejecting true no-candidate cases but still misses or
under-selects legitimate matches.

Do NOT weaken negative discrimination blindly.

Do NOT change the gold merely to make the score cross 0.90.

The Secret Agent 30-request pilot has now been repeatedly inspected and used to
guide reviewer design.

From this point onward treat it as a DEVELOPMENT SET, not an unbiased final
validation set.

You may use it to diagnose and improve the reviewer.

Do not claim production generalization merely because a future version passes
this same set.

======================================================================
KNOWN DEVELOPMENT FAILURE CLASSES
======================================================================

Current v3 false negatives / exact-block failures show recurring cases such as:

- short contextual replies;
- subtitle fragmentation;
- expanded/contracted realization of a screenplay turn;
- repeated fragments of one long screenplay speech;
- locally reordered dialogue;
- vocative/name fragments;
- minor name/spelling variants;
- subtitles containing multiple speaker turns;
- multi-block under-selection.

Current v3 successfully fixed many earlier false positives such as:

- added new propositions;
- improvised lines;
- courtesy/filler not represented in screenplay;
- graphic/telegram/sign text;
- merely semantically related nearby dialogue.

When developing the next version, preserve this successful negative behavior.

======================================================================
AUTONOMOUS VERSIONING POLICY
======================================================================

You decide version numbers.

Do NOT create a new version for every bug.

Use roughly:

PATCH:
implementation/validation/submission bug that should not change model behavior.

Examples:
v3.0.1
v3.0.2

MINOR:
generic prompt/context/retrieval behavior change.

Examples:
v3.1
v3.2

MAJOR:
meaning of the prediction task or response schema changes.

Example:
v4

Current binary candidate task should remain v3-family unless strong evidence
shows the task itself is wrong.

For every behavioral version create a concise version report recording:

- parent version;
- hypothesis;
- evidence that motivated the change;
- exact changed layer;
- expected improvement;
- possible regression;
- experiment used to test it;
- result;
- keep/revert decision.

Never bump a version only to rerun identical prompts and hope stochasticity
makes the score pass.

======================================================================
ERROR ATTRIBUTION ORDER
======================================================================

When something fails, classify it BEFORE changing code.

Use this order:

1. SOURCE ERROR
   missing/corrupt subtitle, screenplay, video, shots.

2. SCREENPLAY PARSER ERROR
   action parsed as dialogue;
   false speaker;
   broken parenthetical;
   bad scene heading;
   duplicated/missing blocks;
   OCR/layout failure.

3. DETERMINISTIC ALIGNMENT / RETRIEVAL ERROR
   true screenplay block not available to reviewer;
   zero candidate request;
   candidate saturation;
   wrong local window;
   systematic missing target.

4. REQUEST-CONTEXT ERROR
   reviewer lacks necessary nearby subtitle context;
   vocative/fragment cannot be interpreted because adjacent film subtitle text
   is absent.

5. REVIEWER POLICY / PROMPT ERROR
   candidates contain correct answer but model systematically abstains or
   over-matches.

6. RESPONSE SCHEMA / VALIDATOR ERROR
   legal semantic errors accidentally treated as malformed output;
   foreign IDs possible;
   schema version mismatch.

7. GOLD / ANNOTATION POLICY ERROR
   frozen label genuinely contradicted by source evidence.

8. NORMAL MODEL ERROR
   isolated mistake without systematic pattern.

Do not fix a Layer 5 problem by changing Layer 2.

Do not fix a Layer 3 recall problem by making the LLM invent IDs.

Do not fix a Layer 7 ambiguity by tuning the model toward the current label.

======================================================================
CHANGE ONE CAUSAL LAYER AT A TIME
======================================================================

Prefer one meaningful layer change per experiment.

Example:

If correct candidate exists but vocative subtitles fail because adjacent film
subtitle text is missing:

first test adding nearby subtitle context.

Do not simultaneously:
- rewrite parser;
- change retrieval;
- change prompt;
- change schema.

Only combine changes when the evidence shows they are inseparable.

======================================================================
LIKELY NEXT REVIEWER EXPERIMENT
======================================================================

Do not blindly implement this if inspection finds a better explanation, but the
current evidence suggests that the next v3-family reviewer should investigate:

A. nearby final-film subtitle context, approximately ±2 subtitles, supplied as
context only;

B. explicit "before abstaining" reasoning rules for:

- short contextual replies;
- fragments of a longer turn;
- expanded/contracted realizations;
- repeated fragments of a long screenplay turn;
- vocatives;
- minor spelling/name variation;

C. stronger multi-speaker / multi-block instruction:

If a subtitle contains multiple speaker turns and adjacent supplied screenplay
blocks correspond to those turns, select every required adjacent block.

D. preserve:

- graphic/insert → no_candidate_match;
- genuinely new proposition → no_candidate_match;
- no invented IDs;
- request-specific enums.

Do NOT require screenplay monotonicity.

Sequence movement remains diagnostic evidence.

======================================================================
NEARBY SUBTITLE CONTEXT DESIGN
======================================================================

If you implement nearby subtitle context:

include previous/next 1–2 final-film subtitles when available.

The context subtitles:

- help interpret the target;
- are NOT additional resolution targets;
- must not change request IDs;
- must not alter target subtitle IDs;
- must not alter candidate IDs;
- must not leak gold labels;
- must not include future LLM answers.

Record the exact context-window design in the reviewer manifest.

======================================================================
EXPERIMENTAL INTEGRITY: DO NOT OVERFIT THE SECRET AGENT
======================================================================

This is critical.

The Secret Agent existing 30-request pilot is now a DEV set.

You may run targeted development experiments on it.

But once a reviewer is tuned using those errors:

DO NOT claim that passing the same 30 proves generalization.

Instead select a THIRD CALIBRATION MOVIE from the eight remaining movies.

Choose it objectively AFTER deterministic processing.

Select a film that is structurally different from:

- Blue Moon;
- The Secret Agent.

Use evidence such as:

- screenplay formatting/layout;
- scene count;
- dialogue/action distribution;
- unresolved deterministic alignment rate;
- fallback retrieval rate;
- candidate saturation;
- narration/voice-over;
- songs;
- graphics/inserts;
- subtitle fragmentation;
- repeated/reordered dialogue;
- OCR/layout anomalies.

Record why the selected movie is the cross-film calibration movie.

======================================================================
THIRD CALIBRATION MOVIE
======================================================================

For the selected third calibration movie:

prepare a FULL 30-request stratified reference set.

The reference labels MUST be frozen BEFORE viewing reviewer outputs.

Important:

Codex may inspect screenplay, subtitles and video to produce provisional
reference adjudication.

But:

- never use reviewer output as evidence to construct/change its own gold;
- ambiguous cases must remain ambiguous;
- exclude explicitly ambiguous cases only in a separately reported
  resolved-gold metric;
- do not call machine-generated reference labels "human gold";
- store provenance;
- make a self-contained adjudication package.

If a case requires video/speaker evidence, inspect local video where practical.

If it remains genuinely ambiguous, mark it ambiguous.

Never force a label to improve metrics.

======================================================================
REFERENCE LABEL FREEZE
======================================================================

Before submitting any calibration pilot:

write:

pilot_reference_frozen.jsonl
reference_manifest.json

Record:

- generation time;
- source hashes;
- reviewer version NOT YET RUN;
- ambiguous IDs;
- reference evidence/provenance.

After the reviewer output has been seen, do not modify this file in-place.

Any later adjudication must create a new version and explicitly report how it
changes metrics.

======================================================================
REVIEWER PROMOTION RULE
======================================================================

A reviewer version may become the GLOBAL PRODUCTION VERSION only after:

1. structural validation passes;

2. it performs acceptably on a calibration movie that was NOT used to design
   that version;

3. there is no new systematic failure class.

Default full-calibration gates on RESOLVED gold:

hard:
- invalid request count = 0;
- missing prediction count = 0;
- foreign candidate ID count = 0.

quality:
- candidate_task_accuracy >= 0.90;
- candidate_presence_decision_accuracy >= 0.90;

when sample size permits:
- easy accuracy >= 0.95;

also inspect:
- match exact-block accuracy;
- no_candidate_match accuracy;
- multi-block accuracy;
- high-risk sequence events;
- candidate-recall failures.

Do not promote a version solely because a rounded metric barely passes if there
is an obvious systematic failure.

Do not reject a version solely for benign sequence movement if predictions are
correct.

======================================================================
PILOT SIZE POLICY FOR THE OTHER MOVIES
======================================================================

After a global production reviewer has passed an independent full calibration:

For ordinary movies with familiar structure:

use a frozen stratified 10–15 request pilot.

Include coverage of:

- easy;
- fuzzy;
- multi;
- difficult;
- early/middle/late;
- fallback retrieval;
- candidate saturation;
- no-candidate examples;
- fragments/repeats when present.

For a movie with a new regime, use a full 30.

New regime examples:

- substantially different screenplay PDF format;
- OCR corruption;
- many songs/lyrics;
- heavy narration;
- extensive on-screen text;
- bilingual/multilingual dialogue;
- unusually high candidate saturation;
- unusually high fallback retrieval;
- parser structural failures;
- systematic candidate recall failures.

Every several ordinary films, perform another full 30 calibration if useful.

You decide based on evidence, not a fixed movie count.

======================================================================
SMALL-PILOT INTERPRETATION
======================================================================

Do not over-interpret percentages from 10–15 items.

For small pilots require:

- zero structural errors;
- zero foreign IDs;
- no obvious systematic semantic error;
- reviewed risk cases look consistent with policy.

If one isolated error appears in a 10-item pilot, diagnose it rather than
blindly treating 90% as a statistically precise boundary.

======================================================================
PAID API BUDGET / LOOP CONTROL
======================================================================

Use OpenAI Batch, not unnecessary synchronous calls.

Do not create uncontrolled retry loops.

Per movie:

- maximum 2 paid pilot submissions without discovering a genuinely new causal
  issue;
- never rerun identical input/prompt purely hoping for a different random
  answer;
- never submit full remaining requests when the pilot gate is unresolved;
- after two failed reviewer revisions without a clear new hypothesis, mark the
  movie REVIEW_NEEDED and continue to another movie.

A global reviewer change does not require eight immediate paid reruns.

Validate it first on:
- dev regression cases;
- an independent calibration movie.

Only then propagate.

======================================================================
PER-MOVIE WORKFLOW
======================================================================

For each of the eight movies autonomously execute:

STAGE A — inventory

Find exact source paths.
Hash inputs.
Detect existing outputs.
Do not overwrite.

STAGE B — Stage 1 check

If shots.jsonl exists:
validate it and treat it as immutable.

If Stage 1 is genuinely absent:
use the current repository Stage 1 workflow to generate it once,
then freeze/hash it.

STAGE C — screenplay structural parse

Run parser.

Audit:

- scene count;
- dialogue count;
- action count;
- broken blocks;
- duplicate IDs;
- missing speakers;
- action-like dialogue;
- editorial labels as speakers;
- fragmented parentheticals;
- suspicious scene headings;
- known parser diagnostics.

If structural errors exist:
fix GENERIC parser logic,
add regression tests,
rerun only affected downstream deterministic artifacts.

Regression-test already processed calibration movies.

STAGE D — deterministic subtitle alignment

Run deterministic alignment.

Report:

- subtitle count;
- auto;
- needs_review;
- no_match;
- unresolved;
- reliable anchors;
- review request count;
- zero-candidate count;
- insufficient-candidate count;
- fallback retrieval count;
- candidate saturation;
- monotonicity/window diagnostics.

Zero/insufficient candidate groups should generally be eliminated before LLM
review unless explicitly unavoidable.

STAGE E — request QA

Verify each review request:

- request-local candidate IDs unique;
- correct scenes/orders;
- no parser contamination;
- no malformed dialogue;
- candidate list bounded;
- target likely represented when retrieval evidence supports it.

STAGE F — choose pilot strategy

Decide:
- no LLM needed;
- 10–15 frozen pilot;
- full 30 calibration.

Record the reason.

STAGE G — freeze reference labels

Freeze them before reviewer output.

STAGE H — reviewer run

Use current global production candidate-task reviewer.

Hard validate before evaluation.

STAGE I — diagnose result

If it fails:
classify each failure using ERROR ATTRIBUTION ORDER.

Only change pipeline when failures are systematic or causally understood.

STAGE J — promote / retry / block

Choose:
PASS
REVISE_GLOBAL_REVIEWER
FIX_MOVIE_PARSER
FIX_RETRIEVAL
REFERENCE_ADJUDICATION_NEEDED
BLOCKED

STAGE K — remaining/full review

Only after movie gate passes.

STAGE L — merge/apply

Use versioned candidate-task production paths.

Never use historical v1/v2 three-way validators to merge/apply v3-family
responses.

STAGE M — risk audit

Build a self-contained review package for high-risk outputs.

STAGE N — final Stage 2 QC

Validate reviewed alignment and shot context.

Freeze manifests/hashes.

======================================================================
IMPORTANT: COMPLETE THE V3-FAMILY PRODUCTION LIFECYCLE
======================================================================

The repository currently has versioned v3:

- prepare pilot Batch;
- submit;
- validate;
- evaluate.

Before the FIRST full production remaining Batch under a v3-family reviewer,
audit the remaining/merge/apply lifecycle.

Do NOT accidentally use historical three-way response validation.

Implement versioned equivalents where needed, conceptually:

prepare-openai-remaining-v3
merge-openai-validated-responses-v3
apply-openai-responses-v3

or a cleaner generic version-aware architecture.

Requirements:

- candidate_task_v3-family schema;
- request-specific candidate enums;
- match/no_candidate_match;
- v3 hard validator;
- sequence diagnostics separated;
- immutable deterministic inputs;
- versioned reviewed outputs;
- no overwrite.

Add regression tests before using it on a full movie.

======================================================================
DOWNSTREAM CLASSIFICATION OF no_candidate_match
======================================================================

The LLM should NOT decide:

"true screenplay absence"
versus
"candidate retrieval miss".

After prediction classify no_candidate_match using deterministic evidence into
statuses such as:

probable_true_no_match
candidate_recall_risk
ambiguous_needs_review

Use evidence including:

- candidate saturation;
- fallback retrieval;
- lexical/semantic retrieval scores;
- strong screenplay text outside supplied candidate window;
- neighboring anchors;
- deterministic alignment evidence.

This is downstream diagnostic metadata.

Do not change the LLM binary decision.

======================================================================
HIGH-RISK AUDIT
======================================================================

After full reviewer output, automatically surface at least:

- low confidence;
- multi-block selection;
- multi-speaker subtitle;
- candidate-limit saturated request;
- fallback retrieval;
- strong lexical overlap + no_candidate_match;
- high semantic score + no_candidate_match;
- backward/large sequence jump;
- cross-scene sequence event;
- repeated/reordered mappings;
- graphic/insert-like text;
- vocative fragment;
- very short subtitle;
- candidate recall risk;
- parser structural warning.

The audit must be self-contained:

include subtitle context,
candidate texts,
selected blocks,
screenplay local context,
sequence context,
diagnostics,
pending adjudication fields.

Codex may resolve obvious deterministic cases when evidence is strong.

Leave genuinely ambiguous cases pending.

======================================================================
SHOT CONTEXT
======================================================================

Once reviewed subtitle-script alignment is complete:

regenerate reviewed shot-script context as a NEW versioned output.

Never overwrite deterministic shot_script_context.jsonl.

Preserve:

- global screenplay order;
- transition-shot semantics;
- same-scene interpolation contract;
- empty candidate evidence for interpolation;
- explicit unaligned status where appropriate.

Run full shot-context QC.

======================================================================
GLOBAL REGRESSION PROTECTION
======================================================================

Any generic parser/retrieval/reviewer change must be checked against existing
calibration movies.

At minimum protect:

tt32536315
tt27847051

Do not overwrite their historical data.

Use hash/snapshot checks.

A generic fix that solves one movie but creates regression in another must not
be promoted without explicit justification.

======================================================================
DO NOT MAKE MOVIE-SPECIFIC PROMPT HACKS
======================================================================

Do not hard-code:

movie IDs,
character names,
specific subtitle text,
specific screenplay lines

into production prompt/parser/retrieval rules.

Movie-specific artifacts belong in data/adjudication, not generic code.

Generic parser support for a screenplay formatting convention is allowed.

======================================================================
COMMIT POLICY
======================================================================

Commit source changes in small meaningful commits.

Examples:

fix: handle screenplay editorial labels
feat: add nearby subtitle review context
feat: add candidate-task v3.1 reviewer
feat: add versioned candidate-task remaining workflow

Do not commit:

- videos;
- screenplay PDFs;
- subtitles;
- generated Batch payloads;
- raw API outputs;
- generated dataset artifacts unless repository policy explicitly says so.

Keep working tree clean at stable checkpoints.

Push when credentials are available.

Do not stop simply because push credentials are unavailable.

======================================================================
MOVIE STATUS FILE
======================================================================

Maintain:

/mnt/g/datasets/oscar_movie_processed/stage2_goal_status.json

For every movie store:

movie_id
source_status
stage1_status
parser_version
parser_qc
deterministic_alignment_status
review_request_count
reviewer_version
pilot_type
pilot_request_count
reference_status
pilot_metrics
production_batch_status
high_risk_audit_count
pending_ambiguity_count
reviewed_alignment_path
reviewed_shot_context_path
final_qc_status
blocked_reason
artifact_hashes

Update atomically.

This file is operational state, not scientific gold.

======================================================================
GLOBAL EXPERIMENT LOG
======================================================================

Maintain:

/mnt/g/datasets/oscar_movie_processed/stage2_reviewer_experiments.jsonl

One row for every reviewer experiment:

experiment_id
timestamp
parent_version
candidate_version
movie_id
dataset_role = dev | calibration | spot_check
hypothesis
changed_layer
changed_files
prompt_hash
schema_hash
request_context_version
pilot_ids_hash
reference_hash
API_batch_id
hard_validation
metrics
error_categories
decision = promote | reject | revise
reason

Never overwrite experiment history.

======================================================================
FINAL PER-MOVIE DIRECTORY
======================================================================

Use a clear versioned structure.

Example concept:

$PROCESSED_ROOT/<movie_id>/
    movie_script_context.json
    subtitle_script_alignment.jsonl
    shot_script_context.jsonl
    review/
        openai/
            ...
    reviewed/
        <production_version>/
            subtitle_script_alignment.reviewed.jsonl
            shot_script_context.reviewed.jsonl
            qc_report.json
            high_risk_audit.jsonl
            manifest.json

Adapt naming to repository conventions, but preserve historical artifacts.

======================================================================
SUCCESS CRITERIA FOR A MOVIE
======================================================================

A movie is Stage-2 production complete when:

- source inputs are identified and hashed;
- screenplay structural QC passes;
- deterministic alignment/retrieval has no known structural blocker;
- reviewer calibration/spot-check policy has passed;
- all required review requests are covered;
- 0 malformed production responses;
- 0 missing production responses;
- 0 foreign candidate IDs;
- high-risk audit has been generated;
- unresolved ambiguity is explicitly recorded;
- reviewed alignment exists;
- reviewed shot context exists;
- QC passes;
- manifest/hashes exist;
- deterministic originals remain unchanged.

A small number of explicitly documented ambiguous cases does not necessarily
block the movie if they remain marked needs_review and are excluded from claims
of resolved accuracy.

======================================================================
SUCCESS CRITERIA FOR THE WHOLE GOAL
======================================================================

The goal is complete when all eight movie IDs have one of:

COMPLETE
or
BLOCKED_WITH_EXPLICIT_REASON

and when:

- one global production reviewer version is documented;
- its independent calibration evidence is documented;
- reviewer experiment history is preserved;
- every COMPLETE movie has reviewed Stage 2 outputs and QC manifest;
- no historical artifacts were silently overwritten;
- no unresolved systematic parser/retrieval/reviewer failure remains.

======================================================================
AUTONOMOUS STOP CONDITIONS
======================================================================

Stop a specific branch of work and reassess rather than looping when:

- two paid pilot revisions fail without a new causal hypothesis;
- source input is missing/corrupt;
- gold/reference cannot be resolved without genuinely unavailable evidence;
- a proposed change would require movie-specific production hacks;
- a global change regresses prior calibration movies materially;
- API failures are unrelated to model quality;
- remaining/full submission would occur before calibration gate passes.

In those cases record the blocker and continue with other movies where safe.

======================================================================
FINAL REPORT
======================================================================

At goal completion produce:

1. final git commit;
2. global production reviewer version;
3. reviewer version history and why versions were accepted/rejected;
4. independent calibration movie and metrics;
5. one table for all eight movies containing:
   - deterministic status;
   - parser version;
   - review version;
   - pilot size;
   - pilot metrics;
   - full reviewed request count;
   - high-risk audit count;
   - pending ambiguities;
   - Stage 2 final status;
6. all generated reviewed alignment paths;
7. all reviewed shot-context paths;
8. protected-artifact verification;
9. Batch IDs and model versions;
10. total paid pilot runs per movie;
11. unresolved blockers;
12. recommendations for Stage 3.

Do not ask the user which reviewer version to try next.

Choose it from evidence and continue toward the goal.
