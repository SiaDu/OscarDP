"""Timeline figures for one Stage07 run: every layer of a sequence on one time axis, plus close-up cases.

Overview lanes: shots, dialogue by speaker, and per character the human gaze labels, machine head movements and AU
hints, and the acting-interpretation beats (with the earlier review verdict when given). Cases (a JSON file kept
with the reports, since their labels quote dialogue) zoom in on given windows with a raw OpenFace trace. Colours follow characters (as speakers and as gaze targets).
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from matplotlib.patches import Patch, Rectangle

PEOPLE = ("MICHELLE", "TEDDY", "DON")
COLOR = {"MICHELLE": "#257de0", "TEDDY": "#f47230", "DON": "#00b894"}  # categorical slots 1–3, validated all-pairs
AWAY, CLOSED, UNKNOWN, NA = "#c4c4c4", "#4d4d4d", "#9c9c9c", "#efefef"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
VERDICT = {"YES": "Y", "PARTLY": "P", "NO": "N"}
CHAR_IN = 0.052  # average width of a 6.5 pt character, inches
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK, "text.color": INK, "font.family": "DejaVu Sans"})


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gaze_color(segment):
    if segment["kind"] == "BLINK" or segment["form"] == "CLOSED":
        return CLOSED
    return COLOR.get(segment["target"], AWAY)


def bar(ax, y, start, end, color, h=0.62, **kw):
    return ax.add_patch(Rectangle((start, y - h / 2), end - start, h, facecolor=color, **{"edgecolor": "none", **kw}))


def cue_bar(ax, y, cue, alpha=1., text=False, h=0.5):
    """One subtitle cue; a white edge keeps adjacent cues apart and the text stays inside its own bar."""
    patch = bar(ax, y, cue["start"], cue["end"], COLOR.get(cue["speaker_candidate"], UNKNOWN), h=h, alpha=alpha,
                edgecolor="white", linewidth=1.2)
    if text:
        x0, x1 = ax.get_xlim()
        inches = ax.figure.get_size_inches()[0] * ax.get_position().width
        fits = int((min(cue["end"], x1) - cue["start"]) / (x1 - x0) * inches / CHAR_IN)
        words = cue["text"].replace("\n", " ")
        if fits >= 4:
            ax.text(cue["start"] + 0.05, y, words if len(words) <= fits else words[:fits - 1] + "…", fontsize=6.5,
                    va="center")
    return patch


def joint_spans(labels, people):
    """Times when every given character holds a gaze on the same third person (human labels)."""
    holds = {p: [s for t in labels["tracks"] if t["character"] == p and t["status"] == "annotated"
                 for s in t["segments"] if s["kind"] == "HOLD"] for p in people}
    out = []
    for target in PEOPLE:
        if target in people:
            continue
        spans = None
        for p in people:
            mine = [(s["start_s"], s["end_s"]) for s in holds[p] if s["target"] == target]
            spans = mine if spans is None else [(max(a, c), min(b, d)) for a, b in spans for c, d in mine
                                                if max(a, c) < min(b, d)]
        out += [(a, b, target) for a, b in spans or []]
    return out


def gaze_lane(ax, y, track):
    if track["status"] != "annotated":
        return
    for s in track["segments"]:
        if s["kind"] == "MOVE":
            mid = (s["start_s"] + s["end_s"]) / 2
            ax.plot([mid, mid], [y - 0.36, y + 0.36], color=INK, lw=0.7, solid_capstyle="butt")
        else:
            bar(ax, y, s["start_s"], s["end_s"], gaze_color(s))


def verdicts(review_dir):
    if not review_dir:
        return {}
    with (Path(review_dir) / "review.csv").open(encoding="utf-8-sig") as stream:
        return {r["annotation_id"]: VERDICT.get(r["human_agree"].strip().upper(), "") for r in csv.DictReader(stream)}


def overview(data, inp, labels, face_head, stats, earlier, path):
    start, end = data["time_range"]["start"], data["time_range"]["end"]
    rows = ["Shot", "Dialogue"] + [f"{p.title()} {k}" for k in ("gaze", "head · AU", "beats") for p in PEOPLE]
    y = {name: len(rows) - 1 - i for i, name in enumerate(rows)}
    fig, ax = plt.subplots(figsize=(15, 6.4))
    for n, s in enumerate(data["shots"]):
        bar(ax, y["Shot"], s["start"], s["end"], "#f2f2f2" if n % 2 else "#e2e2e2", h=0.8)
        ax.text((s["start"] + s["end"]) / 2, y["Shot"], s["shot_id"][-3:], ha="center", va="center", color=INK2)
        ax.axvline(s["start"], color=GRID, lw=0.6, zorder=0)
    for c in inp["subtitles"]:
        cue_bar(ax, y["Dialogue"], c)
    for t in labels["tracks"]:
        lane = y[f"{t['character'].title()} gaze"]
        if t["status"] == "annotated":
            gaze_lane(ax, lane, t)
        else:
            shot = next(s for s in data["shots"] if s["shot_id"] == t["shot_id"])
            bar(ax, lane, shot["start"], shot["end"], NA, hatch="////", linewidth=0)
    for t in face_head["tracks"]:
        if not t["reliability"]["reliable"]:
            continue
        lane = y[f"{t['character'].title()} head · AU"]
        for e in t["au"]:
            ax.plot([e["start"], e["end"]], [lane - 0.22] * 2, color=UNKNOWN, lw=2.2, solid_capstyle="butt")
        for m in t["head"]:
            mid = (m["start"] + m["end"]) / 2
            if m["semantic_candidate"]:
                ax.scatter([mid], [lane + 0.14], marker="v", s=34, color=INK, edgecolor="white", linewidth=0.8, zorder=3)
            else:
                ax.plot([mid, mid], [lane, lane + 0.3], color=INK2, lw=0.9)
    for a in data["acting_annotations"]:
        lane = y[f"{a['character_candidate'].title()} beats"]
        bar(ax, lane, a["start"] + 0.08, a["end"] - 0.08, "#f5f5f5", h=0.66, linewidth=0.8, edgecolor=INK2)
        tag = a["annotation_id"][-3:] + (f" {earlier[a['annotation_id']]}" if earlier.get(a["annotation_id"]) else "")
        ax.text((a["start"] + a["end"]) / 2, lane, tag, ha="center", va="center", fontsize=7)
    for f in stats.get("gaze_following", []):
        ax.annotate("", xy=(f["follower_shift_start_s"], y[f"{f['follower'].title()} gaze"] + 0.36),
                    xytext=(f["leader_landing_s"], y[f"{f['leader'].title()} gaze"] - 0.36),
                    arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.9})
        ax.text(f["follower_shift_start_s"] + 0.6, (y[f"{f['follower'].title()} gaze"] + y[f"{f['leader'].title()} gaze"]) / 2,
                f"{f['follower'].title()} follows {f['leader'].title()} to {f['target'].title()} ({f['lag_s']} s)",
                fontsize=6.5, color=INK, va="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    ax.set_xlim(start, end)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(list(y.values()), list(y))
    ticks = list(range(0, int(end - start) + 1, 10))
    ax.set_xticks([start + t for t in ticks], [str(t) for t in ticks])
    ax.set_xlabel(f"seconds from sequence start (movie time {start:.1f} s)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(color=COLOR[p], label=p.title()) for p in PEOPLE] + [
        Patch(color=UNKNOWN, label="unattributed line"), Patch(color=AWAY, label="gaze away"),
        Patch(color=CLOSED, label="eyes closed"), Patch(facecolor=NA, hatch="////", label="gaze not annotatable"),
        plt.Line2D([], [], color=INK, lw=0.8, label="gaze shift"),
        plt.Line2D([], [], marker="v", color=INK, lw=0, label="nod / shake / tilt"),
        plt.Line2D([], [], color=INK2, lw=0.9, label="other head movement"),
        plt.Line2D([], [], color=UNKNOWN, lw=2.2, label="AU hint")]
    ax.legend(handles=handles, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    note = "Colour = speaker in Dialogue, gaze target in gaze lanes. Gaze: human labels. Head and AU: OpenFace on close-up tracks."
    note += " Beats: annotation number and earlier review (Y agree, P partly, N disagree)." if earlier else ""
    fig.text(0.01, 0.005, note, fontsize=7, color=INK2)
    ax.set_title(f"{data.get('sequence_id', '')} · every layer on one time axis", loc="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    save(fig, path)


def save(fig, path):
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def trace(source, column, start, end, degrees=False):
    rows = pq.read_table(source, columns=["movie_timestamp_sec", "openface_success", column]).to_pylist()
    pts = [(r["movie_timestamp_sec"], math.degrees(r[column]) if degrees else r[column]) for r in rows
           if r["openface_success"] and start <= r["movie_timestamp_sec"] <= end]
    return [t for t, _ in pts], [v for _, v in pts]


def case(spec, inp, labels, face_head, stats, path):
    """spec: title, start, end, characters (gaze lanes), optional trace {track_id, column, label, degrees, lines}."""
    a, b = spec["start"], spec["end"]
    has_trace = "trace" in spec
    fig, axes = plt.subplots(2 if has_trace else 1, 1, figsize=(7.2, 3.4 if has_trace else 2.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]} if has_trace else None, squeeze=False)
    ax = axes[0][0]
    ax.set_xlim(a, b)
    fig.tight_layout()
    lanes = ["Dialogue"] + [f"{p.title()} gaze" for p in spec["characters"]]
    y = {name: len(lanes) - 1 - i for i, name in enumerate(lanes)}
    for c in inp["subtitles"]:
        if c["end"] > a and c["start"] < b:
            cue_bar(ax, y["Dialogue"], dict(c, start=max(c["start"], a)), alpha=0.35, text=True)
    if spec.get("joint"):
        lanes_y = [y[f"{p.title()} gaze"] for p in spec["characters"]]
        for s, e, target in joint_spans(labels, spec["characters"]):
            if e > a and s < b:
                ax.add_patch(Rectangle((s, min(lanes_y) - 0.45), e - s, max(lanes_y) - min(lanes_y) + 0.9,
                                       facecolor="none", edgecolor=INK, lw=0.8, ls="--"))
                ax.text((s + e) / 2, min(lanes_y) - 0.5, f"both on {target.title()} {e - s:.2f} s", ha="center",
                        va="top", fontsize=6.5)
    for t in labels["tracks"]:
        if t["character"] in spec["characters"] and t["status"] == "annotated":
            gaze_lane(ax, y[f"{t['character'].title()} gaze"], t)
    for f in stats.get("gaze_following", []):
        if a <= f["follower_shift_start_s"] <= b and f["follower"] in spec["characters"] and f["leader"] in spec["characters"]:
            ax.annotate("", xy=(f["follower_shift_start_s"], y[f"{f['follower'].title()} gaze"] + 0.36),
                        xytext=(f["leader_landing_s"], y[f"{f['leader'].title()} gaze"] - 0.36),
                        arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.9})
    for x, text in spec.get("marks", []):
        ax.axvline(x, color=INK2, lw=0.6, ls=":")
        ax.text(x, len(lanes) - 0.45, text, fontsize=6.5, ha="center", va="bottom", color=INK)
    ax.set_ylim(-0.9 if spec.get("joint") else -0.6, len(lanes) - 0.4)
    ax.set_yticks(list(y.values()), list(y))
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title(spec["title"], loc="left", fontsize=8.5, pad=16)
    if has_trace:
        tr = spec["trace"]
        low = axes[1][0]
        for track_id, style in tr["tracks"]:
            t = next(x for x in face_head["tracks"] if x["track_id"] == track_id)
            xs, vs = trace(t["source"], tr["column"], a, b, tr.get("degrees", False))
            low.plot(xs, vs, color=COLOR[t["character"]], lw=1.6 if style == "-" else 1.0, ls=style)
        for value, text in tr.get("lines", []):
            low.axhline(value, color=INK2, lw=0.7, ls="--")
            low.text(b, value, f" {text}", fontsize=6.5, va="center", color=INK2)
        for x, _ in spec.get("marks", []):
            low.axvline(x, color=INK2, lw=0.6, ls=":")
        for x, value, text in tr.get("notes", []):
            low.text(x, value, text, fontsize=6.5, color=INK2, va="bottom")
        low.set_ylabel(tr["label"])
        low.grid(axis="y", color=GRID, lw=0.6)
        for spine in ("top", "right"):
            low.spines[spine].set_visible(False)
    last = axes[-1][0]
    last.set_xlim(a, b)
    last.set_xlabel("movie time (s)")
    fig.tight_layout()
    save(fig, path)


def au_lines(face_head, trace_spec):
    """Baseline median and episode threshold for a trace of one AU, from the face/head evidence."""
    track = next(t for t in face_head["tracks"] if t["track_id"] == trace_spec["tracks"][0][0])
    median, sd = track["au_baselines"][trace_spec["au_baseline"]]
    rise = max(face_head["parameters"]["AU_MIN_RISE"], face_head["parameters"]["AU_Z"] * sd)
    state = trace_spec["au_baseline"].split("/")[1]
    return [(median, f"{state} median"), (median + rise, "episode threshold")]


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    data, inp = read(args.run / "stage07_prototype.json"), read(args.run / "stage07_input.json")
    labels, face_head, stats = read(args.gaze_labels), read(args.face_head), read(args.gaze_stats)
    data.setdefault("sequence_id", labels["sequence_id"])
    args.output_dir.mkdir(parents=True)
    overview(data, inp, labels, face_head, stats, verdicts(args.review), args.output_dir / "timeline")
    cases = read(args.cases) if args.cases else {}
    for name, spec in cases.items():
        if "au_baseline" in spec.get("trace", {}):
            spec["trace"]["lines"] = au_lines(face_head, spec["trace"])
        case(spec, inp, labels, face_head, stats, args.output_dir / name)
    if args.cases:
        (args.output_dir / "cases.json").write_text(args.cases.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Figures → {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="assembled Stage07 run")
    parser.add_argument("--gaze-labels", required=True, type=Path)
    parser.add_argument("--face-head", required=True, type=Path)
    parser.add_argument("--gaze-stats", required=True, type=Path)
    parser.add_argument("--review", type=Path, help="filled review directory whose verdicts label the beats")
    parser.add_argument("--cases", type=Path, help="JSON of close-up cases: title, start, end, characters, marks, "
                                                  "optional joint and trace (tracks, column, label, degrees, notes, "
                                                  "au_baseline)")
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
