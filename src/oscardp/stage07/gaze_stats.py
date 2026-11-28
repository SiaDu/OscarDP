"""Gaze statistics from human Stage06.3 labels, split by who is speaking.

Per character and role (speaking / listening / silence / uncertain): share of labelled time on each person, away,
with eyes closed and shifting; shifts per minute; HOLD durations. Two-person events need two labelled characters in
the same shot: mutual gaze, joint attention on a third person, and gaze following (a character turns to the person
another character landed on shortly before).
"""

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from oscardp.shots.transnet import sha256_file
from oscardp.stage06_2.evidence_replay import write_json
from oscardp.stage06_2.inputs import read
from oscardp.stage07.face_head import speech_state, union

VERSION = "stage07.gaze_stats.v1"
STEP_S = 0.01
FOLLOW_WINDOW_S = 2.0  # the follower starts turning within this time after the leader landed
ROLES = ("speaking", "listening", "silence", "uncertain")


def gaze_class(segment, people):
    if segment["kind"] == "MOVE":
        return "shifting"
    if segment["kind"] == "BLINK" or segment["form"] == "CLOSED":
        return "closed"
    return segment["target"] if segment["target"] in people else "away"


def role(t, who, people, cues, voiced):
    own = speech_state(t, who, cues, voiced)
    if own == "speaking":
        return "speaking"
    if any(speech_state(t, p, cues, voiced) == "speaking" for p in people if p != who):
        return "listening"
    return own if own == "uncertain" else "silence"


def speaker_at(t, who, people, cues, voiced):
    return next((p for p in people if p != who and speech_state(t, p, cues, voiced) == "speaking"), None)


def samples(segment):
    return np.arange(segment["start_s"], segment["end_s"], STEP_S)


def character_stats(tracks, people, cues, voiced):
    out = {}
    for who in sorted({t["character"] for t in tracks}):
        time = defaultdict(lambda: defaultdict(float))
        on_speaker = [0., 0.]
        shifts, holds = defaultdict(int), defaultdict(list)
        for track in (t for t in tracks if t["character"] == who):
            for s in track["segments"]:
                cls = gaze_class(s, people)
                for t in samples(s):
                    r = role(t, who, people, cues, voiced)
                    time[r][cls] += STEP_S
                    if r == "listening":
                        on_speaker[1] += STEP_S
                        on_speaker[0] += STEP_S * (cls == speaker_at(t, who, people, cues, voiced))
                r = role(s["start_s"], who, people, cues, voiced)
                if s["kind"] == "MOVE":
                    shifts[r] += 1
                elif s["kind"] == "HOLD":
                    holds[role((s["start_s"] + s["end_s"]) / 2, who, people, cues, voiced)].append(s["end_s"] - s["start_s"])
        roles = {}
        for r in ROLES:
            total = sum(time[r].values())
            if total < STEP_S:
                continue
            roles[r] = {"seconds": round(total, 2),
                        "share": {c: round(v / total, 3) for c, v in sorted(time[r].items(), key=lambda kv: -kv[1])},
                        "on_any_person": round(sum(v for c, v in time[r].items() if c in people) / total, 3),
                        "shifts": shifts[r], "shifts_per_min": round(60 * shifts[r] / total, 1),
                        "holds": len(holds[r]),
                        "median_hold_s": round(statistics.median(holds[r]), 2) if holds[r] else None}
        if on_speaker[1]:
            roles["listening"]["on_current_speaker"] = round(on_speaker[0] / on_speaker[1], 3)
        out[who] = roles
    return out


def target_at(segments, t, people):
    s = next((s for s in segments if s["start_s"] <= t < s["end_s"]), None)
    return (gaze_class(s, people), s) if s else (None, None)


def pair_events(tracks, people):
    """Mutual gaze, joint attention and gaze following for characters labelled in the same shot."""
    by_shot = defaultdict(list)
    for t in tracks:
        by_shot[t["shot_id"]].append(t)
    mutual, joint, following = [], [], []
    for shot, group in sorted(by_shot.items()):
        for a in group:
            for b in group:
                if a is b:
                    continue
                if a["character"] < b["character"]:
                    start = max(min(s["start_s"] for s in x["segments"]) for x in (a, b))
                    end = min(max(s["end_s"] for s in x["segments"]) for x in (a, b))
                    both = defaultdict(float)
                    for t in np.arange(start, end, STEP_S):
                        ga, gb = target_at(a["segments"], t, people)[0], target_at(b["segments"], t, people)[0]
                        if ga == b["character"] and gb == a["character"]:
                            both["mutual"] += STEP_S
                        elif ga == gb and ga in people:
                            both[ga] += STEP_S
                    if both.get("mutual"):
                        mutual.append({"shot_id": shot, "characters": [a["character"], b["character"]],
                                       "overlap_s": round(end - start, 2), "seconds": round(both["mutual"], 2)})
                    for p, v in both.items():
                        if p != "mutual":
                            joint.append({"shot_id": shot, "characters": [a["character"], b["character"]],
                                          "target": p, "seconds": round(v, 2)})
                for shift in b["shifts"]:
                    target = shift["to"]
                    if target not in people or target in (a["character"], b["character"]):
                        continue
                    cls, seg = target_at(a["segments"], shift["start_s"], people)
                    if cls != target:
                        continue
                    landed = next((x["landing_s"] for x in a["shifts"] if x["to"] == target
                                   and x["landing_s"] <= shift["start_s"]
                                   and seg["start_s"] <= x["landing_s"] + STEP_S), seg["start_s"])
                    lag = shift["start_s"] - landed
                    if lag <= FOLLOW_WINDOW_S:
                        following.append({"shot_id": shot, "leader": a["character"], "follower": b["character"],
                                          "target": target, "leader_landing_s": round(landed, 3),
                                          "follower_shift_start_s": shift["start_s"],
                                          "follower_landing_s": shift["landing_s"], "lag_s": round(lag, 2),
                                          "follower_shift_id": shift["shift_id"]})
    return mutual, joint, following


def build(labels, cues, voiced):
    tracks = [t for t in labels["tracks"] if t["status"] == "annotated"]
    people = {t["character"] for t in labels["tracks"]}
    mutual, joint, following = pair_events(tracks, people)
    two = sorted({t["shot_id"] for t in tracks if sum(x["shot_id"] == t["shot_id"] for x in tracks) > 1})
    return {"characters": character_stats(tracks, people, cues, voiced), "two_person_shots": two,
            "mutual_gaze": mutual, "joint_attention": joint, "gaze_following": following}


def summary(doc):
    lines = [f"# {doc['sequence_id']} · gaze statistics", "",
             ("From human gaze labels. Roles: own voiced cue = speaking; another character voiced = listening. "
              "Rates from under 10 s of labels are unstable."), "",
             "| Character | Role | Seconds | On a person | Away | Closed | Shifting | Shifts/min | Median hold s |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for who, roles in doc["characters"].items():
        for r, s in roles.items():
            share = s["share"]
            people = ", ".join(f"{c} {v:.0%}" for c, v in share.items() if c not in ("away", "closed", "shifting"))
            lines.append(f"| {who} | {r} | {s['seconds']} | {people or '0'} | {share.get('away', 0):.0%} | "
                         f"{share.get('closed', 0):.0%} | {share.get('shifting', 0):.0%} | {s['shifts_per_min']} | "
                         f"{s['median_hold_s']} |")
    lines += ["", f"Two-person shots: {', '.join(doc['two_person_shots']) or 'none'}.", ""]
    for m in doc["mutual_gaze"]:
        lines.append(f"- Mutual gaze {' and '.join(m['characters'])} in {m['shot_id']}: {m['seconds']} s of {m['overlap_s']} s.")
    for j in doc["joint_attention"]:
        lines.append(f"- Joint attention on {j['target']} ({' and '.join(j['characters'])}, {j['shot_id']}): {j['seconds']} s.")
    for f in doc["gaze_following"]:
        lines.append(f"- Following: {f['follower']} turns to {f['target']} at {f['follower_shift_start_s']:.2f} s, "
                     f"{f['lag_s']} s after {f['leader']} landed on {f['target']} ({f['shot_id']}).")
    return "\n".join(lines) + "\n"


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    labels = read(args.gaze_labels)
    inp = read(args.stage07_run / "stage07_input.json")
    audio = read(args.stage07_run / "audio_affect.json")
    voiced = union([tuple(x) for p in audio["proposals"] for x in p["speech_intervals"]])
    cues = [(s["start"], s["end"], s["speaker_candidate"]) for s in inp["subtitles"]]
    doc = {"schema_version": VERSION, "movie_id": labels["movie_id"], "sequence_id": labels["sequence_id"],
           "human_verified": False, "parameters": {"STEP_S": STEP_S, "FOLLOW_WINDOW_S": FOLLOW_WINDOW_S},
           "sources": {name: {"file": str(p), "sha256": sha256_file(p)} for name, p in (
               ("gaze_labels", args.gaze_labels), ("stage07_input", args.stage07_run / "stage07_input.json"),
               ("audio_affect", args.stage07_run / "audio_affect.json"))},
           **build(labels, cues, voiced)}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "gaze_stats.json", doc)
    (args.output_dir / "README.md").write_text(summary(doc), encoding="utf-8")
    print(f"{len(doc['characters'])} characters, {len(doc['gaze_following'])} following events → {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaze-labels", required=True, type=Path, help="Stage06.3 gaze_labels.json")
    parser.add_argument("--stage07-run", required=True, type=Path, help="Stage07 run with stage07_input and audio_affect")
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
