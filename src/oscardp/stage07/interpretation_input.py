"""Build the input packet for the Stage07 acting-interpretation prompt and start a new analysis directory.

The packet joins the narrative context (Stage04/04.1/04.2/04.3), dialogue, human gaze labels and machine face/head
evidence for one prepared Stage07 run. Optional: gaze statistics (mutual gaze, joint attention, following) and the
reviewer's verdicts on an earlier version. Reused legacy branch files are copied unchanged with their hashes.
The LLM then writes acting_annotations.json into the same directory (docs/prompts/stage07_acting_interpretation_prompt_v2.md).
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

from oscardp.shots.transnet import sha256_file
from oscardp.stage06_2.evidence_replay import write_json

VERSION = "stage07.interpretation_input.v2"
REUSED = ("visual_affect.json", "narrative_affect.json", "visual_identity_map.json", "ablation_A.json",
          "ablation_B.json", "affect_states.json", "relation_assessments.json", "comparison_abc.json")
AU_NAMES = {"AU01": "inner brow raiser", "AU02": "outer brow raiser", "AU04": "brow lowerer",
            "AU05": "upper lid raiser", "AU06": "cheek raiser", "AU07": "lid tightener", "AU09": "nose wrinkler",
            "AU10": "upper lip raiser", "AU12": "lip corner puller", "AU14": "dimpler", "AU15": "lip corner depressor",
            "AU17": "chin raiser", "AU20": "lip stretcher", "AU23": "lip tightener", "AU25": "lips part",
            "AU26": "jaw drop"}
OBJECTIVE_FIELDS = ("scene_objective", "selected_sequence_objective")
SEQUENCE_FIELDS = ("title_zh", "acting_opportunity_tags", "selection_reason_zh", "acting_observation_to_review_zh",
                   "review_questions_zh", "limitations_zh")
POLICY = [
    "Gaze segments are human labels (human_verified=true); their notes are the reviewer's own words.",
    "AU episodes and head movements are unreviewed OpenFace measurements from reliable close-up tracks.",
    ("AU episodes are speech-gated: jaw opening is ignored during the character's own speech; mouth AUs during own "
     "speech passed a stricter rule but can still contain articulation."),
    ("Head movements exclude gaze shifts and 0.75 s after landing. semantic_candidate marks nod/shake/tilt; beats are "
     "speech rhythm and belong to speaking style, not to the plan."),
    "Stage04/04.1/04.2/04.3 context and legacy visual notes are earlier LLM/assistant hypotheses, not facts.",
    "Unknown or unavailable evidence is missing coverage, not a negative observation.",
    ("AU episodes are hints about where the face changes. Do not name emotions from them: OpenFace misses subtle and "
     "sustained expressions, and the same actor's levels shift between shots."),
    ("Eye-region AUs (05/06/07/09) are masked during labelled gaze shifts, blinks, closed eyes and vertical gaze, "
     "widened by 0.3 s; outside the mask they can still be eyelid movement."),
    ("gaze_relations are computed from human labels. Only shots with two labelled characters can show mutual gaze, "
     "joint attention or following."),
    "review_feedback holds a reviewer's verdicts and notes on an earlier version; the notes are human observations.",
]
FEEDBACK_FIELDS = ("annotation_id", "character", "start_s", "end_s", "beat", "tactic", "interpretation",
                   "human_agree", "human_notes")
HEAD_FEEDBACK_FIELDS = ("head_id", "character", "start_s", "end_s", "movement", "meaning", "human_agree", "human_notes")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def review_feedback(review_dir):
    """Rows the reviewer filled in an earlier review sheet."""
    out = {}
    for name, fields in (("review.csv", FEEDBACK_FIELDS), ("head_gestures.csv", HEAD_FEEDBACK_FIELDS)):
        path = review_dir / name
        if path.exists():
            with path.open(encoding="utf-8-sig") as stream:
                out[name.removesuffix(".csv")] = [{k: r[k] for k in fields} for r in csv.DictReader(stream)
                                                  if r["human_agree"].strip() or r["human_notes"].strip()]
    return out


def gaze_relations(stats, labels):
    segment = {x["shift_id"]: x["segment_id"] for t in labels["tracks"] for x in t["shifts"]}
    return {"by_role": stats["characters"], "two_person_shots": stats["two_person_shots"],
            "mutual_gaze": stats["mutual_gaze"], "joint_attention": stats["joint_attention"],
            "gaze_following": [{**f, "follower_gaze_id": segment.get(f["follower_shift_id"])}
                               for f in stats["gaze_following"]]}


def build_packet(inp, labels, face_head, stage03, stage04_1, stage04_2, stage04_3, states, visual, identity,
                 sequence_id, gaze_stats=None, feedback=None):
    scene_id = inp["scene_id"]
    names = set(inp["character_candidate"])
    scene = next(s for s in stage04_2["scene_objectives"] if s["scene_id"] == scene_id)
    objectives = []
    for c in scene["characters"]:
        for field in OBJECTIVE_FIELDS:
            value = c.get(field)
            if value:
                text = value if isinstance(value, dict) else {"statement_en": value}
                objectives.append({"objective_ref": f"{scene_id}:{c['character_name']}:{field}",
                                   "character": c["character_name"], "kind": field,
                                   "text": text.get("statement_en"), "text_zh": text.get("statement_zh")})
    visual_shots = {s["shot_id"]: s["visual"] for s in stage03["shots"]}
    sequence = next(s for s in stage04_3["sequence_candidates"] if s["sequence_id"] == sequence_id)
    notes = {s["segment_id"]: s for t in labels["tracks"] for s in t["segments"]}
    mapping = {m["visual_subject_id"]: m["character_candidate"] for m in identity["mappings"]}
    head_style = {}
    for t in face_head["tracks"]:
        if t["reliability"]["reliable"]:
            style = head_style.setdefault(t["character"], {"tracks": [], "eye_head_patterns": {}})
            style["tracks"].append({"track_id": t["track_id"], "head_speed_deg_s": t["head_speed_deg_s"]})
            for x in t["eye_head"]:
                style["eye_head_patterns"][x["pattern"]] = style["eye_head_patterns"].get(x["pattern"], 0) + 1
    refs = inp.get("face_head_references", [])
    return {
        "schema_version": VERSION, "movie_id": inp["movie_id"], "sequence_id": sequence_id, "scene_id": scene_id,
        "time_range": {"start": inp["start"], "end": inp["end"]}, "evidence_policy": POLICY,
        "characters": [p for p in stage04_1["profiles"] if p["character_name"] in names],
        "scene": {k: inp["scene_context"][k] for k in ("title", "location", "summary", "narrative_participants")},
        "scene_objectives": [{k: c.get(k) for k in ("character_name", "scene_objective", "obstacle", "tactics",
                                                     "selected_sequence_objective", "selected_sequence_role",
                                                     "uncertainty")} for c in scene["characters"]],
        "objective_refs": objectives,
        "sequence": {k: sequence.get(k) for k in SEQUENCE_FIELDS},
        "shots": [{"shot_id": s["shot_id"], "start": s["temporal"]["start"], "end": s["temporal"]["end"],
                   "shot_scale": visual_shots.get(s["shot_id"], {}).get("shot_scale"),
                   "camera_angle": visual_shots.get(s["shot_id"], {}).get("camera_angle")} for s in inp["stage02_context"]],
        "dialogue": [{"srt_id": f"srt_{s['srt_index']:06d}", "start": s["start"], "end": s["end"],
                      "speaker_candidate": s["speaker_candidate"], "text": s["text"]} for s in inp["subtitles"]],
        "affect_states": [{k: s[k] for k in ("state_id", "character_candidate", "start", "end", "affect",
                                              "expression_style")} for s in states["states"]],
        "gaze": [{**{k: g[k] for k in ("gaze_id", "shot_id", "character_candidate", "kind", "start", "end", "target",
                                        "form", "form_text", "from_gaze", "blink", "certainty")},
                  "reviewer_notes": notes[g["gaze_id"]]["notes"]} for g in inp["stage06_3_gaze_references"]],
        "gaze_unavailable": inp["stage06_3_gaze_unavailable"],
        "au_episodes": [{**{k: r[k] for k in ("evidence_id", "shot_id", "character_candidate", "region",
                                               "speech_state", "start", "end", "peak_rise")},
                         "au": r["label"], "au_name": AU_NAMES.get(r["label"])} for r in refs if r["modality"] == "au"],
        "head_movements": [{**{k: r.get(k) for k in ("evidence_id", "shot_id", "character_candidate", "axes",
                                                      "semantic_candidate", "speech_state", "start", "end",
                                                      "peak_deg", "jaw_check")}, "type": r["label"]}
                           for r in refs if r["modality"] == "head"],
        "head_style": head_style,
        "utterance_head_style": [{"track_id": t["track_id"], "shot_id": t["shot_id"], "character": t["character"], **u}
                                 for t in face_head["tracks"] if t["reliability"]["reliable"]
                                 for u in t.get("utterance_style", [])],
        "face_unreliable_tracks": [{"track_id": t["track_id"], "character": t["character"],
                                    "flags": t["reliability"]["flags"]} for t in face_head["tracks"]
                                   if not t["reliability"]["reliable"]],
        "legacy_visual_notes": [{"proposal_id": p["proposal_id"], "start": p["start"], "end": p["end"],
                                 "character_candidate": mapping.get(p["character_candidate"], "Unknown"),
                                 "observable_behaviour": p["observable_behaviour"]} for p in visual["proposals"]],
        **({"gaze_relations": gaze_relations(gaze_stats, labels)} if gaze_stats else {}),
        **({"review_feedback": feedback} if feedback is not None else {}),
    }


def run(args):
    output = args.output_dir
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inp = read(args.run / "stage07_input.json")
    if not inp.get("stage06_3_gaze_references") or not inp.get("face_head_references"):
        raise ValueError("Prepare the Stage07 run with --gaze-labels and --face-head first")
    movie = args.movie_dir
    paths = {"stage07_input": args.run / "stage07_input.json",
             "gaze_labels": Path(inp["sources"]["stage06_3_gaze"]), "face_head": Path(inp["sources"]["stage07_face_head"]),
             **{f"stage0{n}": movie / f"Stage0{n}_{inp['movie_id']}.json" for n in ("3", "4_1", "4_2", "4_3")}}
    for name in ("gaze_labels", "face_head"):
        if sha256_file(paths[name]) != inp[f"{'stage06_3_gaze' if name == 'gaze_labels' else 'face_head'}_sha256"]:
            raise ValueError(f"{name} changed since the Stage07 run was prepared")
    docs = {name: read(path) for name, path in paths.items()}
    if args.gaze_stats:
        paths["gaze_stats"] = args.gaze_stats
    feedback = review_feedback(args.review) if args.review else None
    if args.review:
        paths.update({f"review_{p.stem}": p for p in sorted(args.review.glob("*.csv"))})
    packet = build_packet(inp, docs["gaze_labels"], docs["face_head"], docs["stage03"], docs["stage04_1"],
                          docs["stage04_2"], docs["stage04_3"], read(args.reuse / "affect_states.json"),
                          read(args.reuse / "visual_affect.json"), read(args.reuse / "visual_identity_map.json"),
                          args.sequence_id, read(args.gaze_stats) if args.gaze_stats else None, feedback)
    packet["sources"] = {name: {"file": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}
    output.mkdir(parents=True)
    for name in REUSED:
        shutil.copyfile(args.reuse / name, output / name)
    write_json(output / "interpretation_input.json", packet)
    write_json(output / "provenance.json", {
        "reused_unchanged_from": str(args.reuse), "reused_sha256": {n: sha256_file(output / n) for n in REUSED},
        "interpretation_input_sha256": sha256_file(output / "interpretation_input.json"),
        "prompt": "docs/prompts/stage07_acting_interpretation_prompt_v2.md",
        "to_author": "acting_annotations.json"})
    print(f"Packet: {len(packet['dialogue'])} cues, {len(packet['gaze'])} gaze segments, "
          f"{len(packet['au_episodes'])} AU episodes, {len(packet['head_movements'])} head movements → {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="prepared Stage07 run (with gaze and face/head)")
    parser.add_argument("--movie-dir", required=True, type=Path, help="dataset folder with Stage03/04 JSON")
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--reuse", required=True, type=Path, help="analysis directory whose branch files are reused")
    parser.add_argument("--gaze-stats", type=Path, help="Stage07 gaze_stats.json (optional)")
    parser.add_argument("--review", type=Path, help="earlier review directory with filled CSVs (optional)")
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
