# Stage06.1 Scene Layout Prototype

This pilot uses the [Stage06.1 Prompt v1.0](prompts/stage06_1_scene_layout_prompt_v1.md)
for a rough, reviewable layout. Stage04.3 supplies the sequence, global time and
ordered shot IDs. Stage01 supplies temporal and frame boundaries and physical JPG
samples. Stage02 speaker labels, Stage03 framing and Stage04 candidates are
context only. The input packet records source paths and hashes. Missing sample
roles are marked in the contact sheets; short-shot aliases count as one source
image. Preparing a packet does not run inference.

From the repository root, set the movie folder to the location of the existing
Stage01, Stage02, Stage03, Stage04 and Stage04.3 JSONs and the `shot_img/` folder.

```sh
run=reports/stage06_1/tt12300742/seq_006
movie_dir=/media/sdu/SiyaoDu/datasets/oscar_movie_analysis/tt12300742

uv --no-cache run --no-sync python -m oscardp.stage06_1 prepare \
  --movie-dir "$movie_dir" --output-dir "${run}_inputs"
```

Use the prompt with a VLM that can view the generated JPG sheets. Supply exact
sequence metadata from `inputs.json`, and examine the original JPGs as needed.
Save the actual response as `${run}_inputs/interactive_candidate.json`. Only list
sheets the VLM actually inspected and describe the VLM and method used. For the
Bugonia pilot all five sheets were viewed with interactive Codex vision, and
the original middle JPGs for shots 216, 222 and 223 were reviewed again.

```sh
uv --no-cache run --no-sync python -m oscardp.stage06_1 render \
  --prepared "${run}_inputs" --candidate "${run}_inputs/interactive_candidate.json" \
  --inspected-sheet shots_01.jpg --inspected-sheet shots_02.jpg \
  --inspected-sheet shots_03.jpg --inspected-sheet shots_04.jpg \
  --inspected-sheet shots_05.jpg \
  --vlm "Codex interactive vision, 5 sheets plus middle originals for 216, 222 and 223" \
  --output-dir "$run"
```

The renderer checks the contract, exact shot coverage, and links to the inspected
evidence. It writes `scene_layout.json`, `scene_layout.svg`, and `review.md`
to a new directory. Validation checks the representation, while the review
checks the actual spatial claims. If inference is unavailable, stop after
preparation. To repeat a preparation or render, choose a fresh output directory.
This prototype uses only in-sequence shots as evidence.

## Add v1.1 camera groups to an existing result

Use the [v1.1 prompt](prompts/stage06_1_scene_layout_prompt_v1_1.md) to compare
actual images for possible repeated viewpoints. Save the resulting group
objects as a JSON array. Group membership can be added directly to the original
candidate, keeping the existing layout, positions and axis judgments.

```sh
root=reports/stage06_1/tt12300742
uv --no-cache run --no-sync python -m oscardp.stage06_1 render \
  --prepared "$root/seq_006_inputs" --candidate "$root/seq_006/scene_layout.json" \
  --camera-groups "$root/seq_006_v1_1_inputs/camera_groups.json" \
  --inspected-sheet shots_01.jpg --inspected-sheet shots_02.jpg --inspected-sheet shots_03.jpg \
  --inspected-sheet shots_04.jpg --inspected-sheet shots_05.jpg \
  --rechecked-sheet shots_02.jpg --rechecked-sheet shots_03.jpg --rechecked-sheet shots_04.jpg \
  --vlm "Codex interactive vision, compared the three rechecked sheets" \
  --output-dir "$root/seq_006_v1_1"
```

The inspected sheets cover the inherited evidence. The rechecked sheets cover
the fresh grouping evidence, listed separately in the review. A full v1.1 JSON
can also be rendered directly with `--candidate` and its inspected sheets.
The SVG lists cameras without positions beside the map and draws a shooting
arrow only when both the camera and its named subject have map positions.

## Infer v1.2 camera regions

Use the [v1.2 prompt](prompts/stage06_1_scene_layout_prompt_v1_2.md) to classify
visible clues. Save an object keyed by shot ID, with `layout_basis` and
`camera_clues` for each shot. The file holds visual annotations only. Python
computes the grid, leaving the previous camera points and groups intact.
`--static-layout` explicitly allows shots marked `assumed_static` to use the
named layout in this run. It changes neither layout stability nor review status.

```sh
root=reports/stage06_1/tt12300742
uv --no-cache run --no-sync python -m oscardp.stage06_1 render \
  --prepared "$root/seq_006_inputs" --candidate "$root/seq_006_v1_1/scene_layout.json" \
  --camera-clues "$root/seq_006_v1_2_inputs/camera_clues.json" --static-layout layout_001 \
  --inspected-sheet shots_01.jpg --inspected-sheet shots_02.jpg --inspected-sheet shots_03.jpg \
  --inspected-sheet shots_04.jpg --inspected-sheet shots_05.jpg \
  --rechecked-sheet shots_01.jpg --rechecked-sheet shots_02.jpg --rechecked-sheet shots_03.jpg \
  --rechecked-sheet shots_04.jpg --rechecked-sheet shots_05.jpg \
  --vlm "Codex interactive vision, all 39 Stage01 samples" \
  --output-dir "$root/seq_006_v1_2"
```

Grid centres have spacing 0.1 within the existing unit square. Torso views
use sectors of ±30° around the named view, relative to the person's body
forward. Depth and image order use the camera's approximate direction toward
the named subject, or the centre of visible people for a group shot. Positive
image horizontal coordinates point screen-right. Axis constraints need an axis
and a positioned reference in the same layout. Camera height and camera groups
are descriptive. The review records unused constraints, empty intersections
and cases where an inherited camera point falls outside its candidate grid.
The SVG renders separate shaded grids so overlapping regions can be compared.
