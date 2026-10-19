# OscarDP — Stage03 GPT Visual Annotation v1.0

**Status:** Interactive GPT annotation contract (not a local CV pipeline). Fixed structure: [Stage03 JSON Schema](schemas/stage03_gpt_visual_v1.schema.json). Default new-window entry: [Stage03 Prompt](prompts/stage03_gpt_visual_prompt_v1.md).

## Inputs — only three resources

The user provides **only** (typically in one Google Drive folder):
1. `Stage01_<movie_id>.json`: authoritative movie ID, ordered shot IDs and boundaries.
2. `Stage02_<movie_id>.json`: text-derived context for referencing/inventory checks, **not proof of visible objects, characters or behaviour**.
3. Actual Early/Middle/Late JPGs for each selected Shot, named `shot_000001_early.jpg`, `shot_000001_middle.jpg`, `shot_000001_late.jpg`, etc.

**Do not ask the user for Frozen Manifest, manually supplied SHA-256, `frame_samples_v1.json`, a batch manifest, screenplay, SRT, videos or YOLO output.** GPT uses actual accessible images, not Drive file names/metadata as a substitute for visual inspection. If any file cannot be opened or viewed, report that instead of inventing evidence. Stage02 may be reviewed for bookkeeping, but must **not** dictate Stage03 visual labels.

Provenance/frozen-source hashing is deliberately **not part of the GPT output schema**. It can be computed independently by local post-annotation validation and release packaging; no visual inference is a formal release by itself.

## Output

For fully completed films: `Stage03_<movie_id>.json`, including all Stage01 shots in order. For *partial* work: `Stage03_<movie_id>_batch_001.json`, increment on continuation, with optional root `batch` object specifying `batch_id`, inclusive `start_shot_id`, inclusive `end_shot_id`. Never present a partial batch as full-movie coverage.

Root: `movie_id`, `stage_version="3.0"`, `shots[]`; optional `batch` for partial files. No `inputs` hash object. Every Shot has `shot_id`, `samples`, `visual`, `subjects`, `objects`, `visible_relations`, `visual_summary`, `review_status`. Do not copy Stage01 time boundaries or Stage02 dialogue/narrative into the visual JSON.

`samples[]` lists exactly three **logical roles** `early`, `middle`, `late` with each one's `image_file` (actual viewed JPG name, not a path or hash). Stage01 1–2-frame Shots may alias sample roles to the same physical image; if confirmed, reuse the actual image filename. Never fabricate an unavailable image or silently substitute a neighboring shot. Unresolvable missing images block that shot.

## Shot-level visual tags

| Field | Allowed Values |
| --- | --- |
| `shot_scale` | `ECU, CU, MCU, MS, MLS, LS, ELS, mixed, not_applicable, unknown` |
| `camera_angle` | `eye_level, high, low, overhead, unknown` |
| `camera_motion_evidence` | `no_evident_motion, possible_motion, unknown` |
| `primary_subject_position` | `left, center, right, multiple, not_applicable, unknown` |
| `over_the_shoulder` | `true, false, null` |
| `visual_focus` | `person, object, environment, mixed, unknown` |
| `evidence_roles` | Nonempty subset of `early,middle,late` (distinct roles) |

Shot Scale is human-relative: use `not_applicable` when human scale cannot be judged, even for an object closeup. `mixed` requires a visible change across samples, not uncertainty. Three stills cannot reliably prove detailed camera movements; use the cautious evidence tags.

## Visible subjects and props

`subjects[]`: each visible person has a **Shot-local** `subject_id` (e.g. `subject_001`), `evidence_roles`, `subject_facing` (`frontal,three_quarter,profile,back,mixed,unknown`), `face_visibility` (`fully_visible,partially_visible,not_visible,mixed,unknown`), `face_quality` (`good,limited,unusable,mixed,not_applicable,unknown`), `occlusion` (`none,partial,heavy,mixed,not_applicable,unknown`), and `appearance_mode` (`direct,reflection,screen_or_photo,unknown`). Link a person across sparse samples only when visually supportable. Do not claim verified actor/character ID, face track or offscreen visibility.

`objects[]`: salient items only, each with Shot-local `object_id`, English noun `label`, `object_framing` (`detail_closeup,prominent,contextual,unknown`), and `evidence_roles`. No bounding boxes.

`visible_relations[]`: references to valid Shot-local `subject_id` and `object_id`, a `relation_type` (`holding,touching,wearing,sitting_on,leaning_on,near,uncertain`) and `evidence_roles`. Include **only** visually evident relations. Empty arrays are valid; do not infer gaze target, emotions, intent, offscreen acts or motion continuity from stills.

`visual_summary` is brief, visual-only prose. `review_status`: `unreviewed,model_reviewed,human_verified,human_corrected`; GPT checking itself can reach at most `model_reviewed`.

## Verification & downstream

Validate strict JSON schema, Stage01 inventory/order, unique Shot IDs/roles, image correspondence, valid person/object refs and evidence roles. If a programmatic validator is not run, say so. The GPT output may be a pilot and should be labelled as such.

Stage04 combines Stage02 narrative and Stage03 visuals to construct actual movie scenes and *unverified* character candidates. Stage05 owns continuous-video identity verification, face tracking, AU and gaze; Stage06 owns explainable acting decisions.
