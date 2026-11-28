"""Write human review sheets for one assembled Stage07 run: acting annotations and semantic head gestures."""

import argparse
import csv
import json
from pathlib import Path

COLUMNS = ("annotation_id", "shot_ids", "character", "start_s", "end_s", "beat", "objective", "tactic", "gaze",
           "face_head", "dialogue", "interpretation", "alternatives", "human_agree", "human_notes")
HEAD_COLUMNS = ("head_id", "shot_ids", "character", "start_s", "end_s", "movement", "speech_state", "words", "meaning",
                "alternatives", "human_agree", "human_notes")
GUIDE = """# Stage07 表演解读审核

由助手按 `docs/prompts/stage07_acting_interpretation_prompt_v1.md` 写成，是假设，不是标签。
对照 [带标注的片段]({video})：右侧面板显示正片时间、当前镜头（SHOT）、生效的解读编号（ANNOTATION）和人工 gaze。
正片时间 = 播放时间 + {start:.3f}。

1. [review.csv](review.csv)：{count} 条解读。`objective` 是 Stage04.2 的场景目标，`tactic` 是助手提出的战术。
2. [head_gestures.csv](head_gestures.csv)：{heads} 个有语义的头部动作（点头、摇头、歪头）及其含义解读。

每行填 `human_agree`（YES / PARTLY / NO）和 `human_notes`（理由或改法；画面里看不出的 AU、头动也请注明）。其他列不用改。
"""


def load(run):
    data = json.loads((run / "stage07_prototype.json").read_text(encoding="utf-8"))
    inp = json.loads((run / "stage07_input.json").read_text(encoding="utf-8"))
    return data, inp


def rows(data, inp):
    cues = {f"srt_{s['srt_index']:06d}": s["text"].replace("\n", " ") for s in inp["subtitles"]}
    objectives = {o["objective_ref"]: o.get("text_zh") or o["text"] for o in data.get("objective_refs", [])}

    def pick(a, modalities):
        return " | ".join(o["description"] for o in a["observation"] if o["source_modality"] in modalities)
    for a in data["acting_annotations"]:
        lines = [cues[i] for o in a["observation"] if o["source_modality"] == "narrative"
                 for i in o["evidence_ids"] if i in cues]
        yield {"annotation_id": a["annotation_id"], "shot_ids": " ".join(a.get("shot_ids", [])),
               "character": a["character_candidate"], "start_s": f"{a['start']:.3f}", "end_s": f"{a['end']:.3f}",
               "beat": a.get("beat", ""), "objective": " | ".join(objectives.get(r, r) for r in a.get("objective_refs", [])),
               "tactic": a.get("tactic", ""), "gaze": pick(a, ("gaze",)), "face_head": pick(a, ("au", "head")),
               "dialogue": " / ".join(lines), "interpretation": a["interpretation"],
               "alternatives": " | ".join(a["alternative_interpretation"]), "human_agree": "", "human_notes": ""}


def head_rows(data, inp):
    for r in data.get("head_gesture_readings", []):
        words = [s["text"].replace("\n", " ") for s in inp["subtitles"] if s["start"] < r["end"] and s["end"] > r["start"]]
        yield {"head_id": r["evidence_id"], "shot_ids": " ".join(r.get("shot_ids", [])),
               "character": r["character_candidate"], "start_s": f"{r['start']:.3f}", "end_s": f"{r['end']:.3f}",
               "movement": r["movement_type"], "speech_state": r["speech_state"], "words": " / ".join(words),
               "meaning": r["meaning"], "alternatives": " | ".join(r["alternatives"]), "human_agree": "",
               "human_notes": ""}


def write_csv(path, columns, records):
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def write(run, output):
    data, inp = load(run)
    sheet, heads = list(rows(data, inp)), list(head_rows(data, inp))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "review.csv", COLUMNS, sheet)
    write_csv(output / "head_gestures.csv", HEAD_COLUMNS, heads)
    video = Path("..") / run.name / "review_video_annotated.mp4"
    (output / "README.md").write_text(GUIDE.format(count=len(sheet), heads=len(heads), video=video.as_posix(),
                                                   start=data["time_range"]["start"]), encoding="utf-8")
    return len(sheet), len(heads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="assembled Stage07 run directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory next to the run")
    args = parser.parse_args()
    annotations, heads = write(args.run, args.output_dir)
    print(f"{annotations} annotations, {heads} head gestures → {args.output_dir}")


if __name__ == "__main__":
    main()
