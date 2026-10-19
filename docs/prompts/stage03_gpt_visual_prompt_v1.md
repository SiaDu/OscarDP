# OscarDP — Stage03 GPT Visual Annotation v1.0

**Reusable GPT instruction for every new chat.**

Repository: https://github.com/SiaDu/OscarDP  
Minimum user request: **“做 OscarDP Stage03。Google Drive：<文件夹链接>”**

## Automatically follow the repository contracts

Open the latest `main` versions of:
- `docs/stage03_gpt_visual_spec_v1.md`
- `docs/schemas/stage03_gpt_visual_v1.schema.json`
- This file, `docs/prompts/stage03_gpt_visual_prompt_v1.md`.

Use the connected Google Drive tool to open the supplied folder and locate `Stage01_<movie_id>.json`, `Stage02_<movie_id>.json`, and the Shot Early/Middle/Late JPGs (including images in subfolders). The user does **not** need to submit a Frozen Manifest, `frame_samples_v1.json`, SRT, screenplay, pre-generated hashes, batch manifest or local CV detections. Do **not** ask for these as preparation. Identify `movie_id` from the actual JSON documents.

**Crucial:** Do not claim that you viewed an image merely because a Drive file name or thumbnail appears. Actually open or load the image and inspect it. If inaccessible, explain exactly which files could not be viewed and ask only for a working link, access or direct photo/ZIP upload. Do not fill any labels for unseen frames.

## Do the visual annotation, not a workflow proposal

Read Stage01 and Stage02 for Shot IDs and inventory. Treat Stage02 as text-only background; it is **not** visual proof. Inspect actual Early/Middle/Late images, and produce Stage03 annotations **using exactly the schema and allowed tags**:

- `visual`: shot scale, angle, camera-motion evidence, main subject positioning, OTS and visual focus.
- `subjects[]`: visually present person instances, face visibility/quality, facing, occlusion, appearance in direct view/reflection/screen.
- `objects[]`: salient props, especially object closeups (no object BBoxes).
- `visible_relations[]`: holding, touching, wearing, sitting on, leaning on or near a visible object, with supporting image roles.
- `visual_summary`: short visual-only description, plus correct `review_status`.

No YOLO, SCRFD, BBoxes, face embeddings, actor/character identification, AU, emotion, speculative intent or precise gaze target. Three stills are insufficient to prove detailed camera movement or action between samples. Use `unknown`, `null`, or empty arrays when evidence is insufficient.

## Default handling without additional questions

When the user does not choose a Shot range, work on the **first up to 20 consecutive Shots** that have their necessary JPGs accessible; report exactly what was accessible. Respect Stage01 1–2-frame short-shot aliases instead of inventing separate physical images. If fewer are accessible, complete only those with sufficient visual evidence and report missing Shots.

For a partial batch, produce `Stage03_<movie_id>_batch_001.json` with the `batch` object specifying exact start/end Shot IDs. Do not call a batch the completed `Stage03_<movie_id>.json`. For a subsequent message “继续”, work on the next unprocessed contiguous batch with incremented batch ID rather than repeating completed work. If selected images are noncontiguous, split batches or disclose gaps instead of implying complete coverage.

Make a downloadable JSON file when file-generation tools are available, not merely a code-fenced example. Verify the structure against the JSON Schema where possible and report whether validation was executable or manual. Never invent SHA-256 strings; hashes for formal releases can be added later by separate trusted local tooling.

Do not modify GitHub main, source data or frozen releases as part of Stage03 annotation. Prioritize **actually performing the requested image annotation** if images are accessible.
