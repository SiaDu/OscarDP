"""Stage07 face and head evidence for one sequence: speech-gated AU episodes and non-gaze head movement.

Each frame gets a speech state for its character (own subtitle cue + voice activity, widened by a margin).
Head movement inside human gaze HOLDs, after the gaze-shift follow window, is treated as performative or
postural. Thresholds scale with each track's own silent jitter; none are tuned on human labels.
All outputs are machine evidence (human_verified=false).
"""

import argparse
import math
import statistics
import wave
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq

from oscardp.shots.transnet import sha256_file
from oscardp.stage06_2.evidence_replay import write_json
from oscardp.stage06_2.inputs import read

VERSION = "stage07.face_head.v1"
SPEECH_MARGIN_S = 0.2  # mouths open shortly before voicing and close after it
HEAD_FOLLOW_S = 0.75  # the head keeps turning after the eyes land (Stage06.3 evaluation)
MIN_FACE_PX = 150
MAX_PROFILE_DEG = 35.
MIN_SUCCESS = 0.9
MIN_HEAD_DEG = 3.
NOISE_FACTOR = 3.
BASELINE_WINDOW_S = 1.0  # rolling median half-width for head excursions
NOD_S = (0.2, 1.0)
SHAKE_WINDOW_S = 1.5
TILT_MIN_S = 0.3
BEAT_MAX_S = 0.4
BEAT_ALIGN_S = 0.1
POSTURE_MIN_S = 1.0
UPPER = ("AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09")
MOUTH = ("AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23")
OPEN = ("AU25", "AU26")
AUS = (*UPPER, *MOUTH, *OPEN)
AU_Z = 3.
AU_MIN_RISE = 0.5  # OpenFace intensity units (0-5)
AU_MIN_S = 0.2
AU_MIN_SPEAKING_S = 0.3
MERGE_S = 0.1
FRAME_GAP_S = 0.065  # a missing frame at 23.976 fps breaks a run
AXES = ("pitch", "yaw", "roll")


# ---------- speech ----------

def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def speech_state(t, character, cues, voiced):
    """'speaking', 'uncertain' (voice under an unattributed cue) or 'silent', with SPEECH_MARGIN_S slack."""
    if not any(a - SPEECH_MARGIN_S <= t < b + SPEECH_MARGIN_S for a, b in voiced):
        return "silent"
    near = {who for a, b, who in cues if a - SPEECH_MARGIN_S <= t < b + SPEECH_MARGIN_S}
    if character in near:
        return "speaking"
    return "uncertain" if "Unknown" in near else "silent"


def rms_envelope(path, offset, hop=0.01, window=0.04):
    with wave.open(str(path)) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("Expected mono 16-bit PCM")
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768
    size, step = int(window * rate), int(hop * rate)
    starts = np.arange(0, len(data) - size, step)
    rms = np.array([math.sqrt(float(np.mean(data[s:s + size] ** 2))) for s in starts])
    return offset + (starts + size / 2) / rate, rms


def stress_peaks(times, rms, spans, quantile=0.75, separation=0.25):
    """Prominent energy peaks inside a speaker's own speech: a proxy for stressed syllables."""
    inside = np.zeros(len(times), bool)
    for a, b in spans:
        inside |= (times >= a) & (times < b)
    if not inside.any():
        return []
    level = float(np.quantile(rms[inside], quantile))
    peaks = []
    for i in range(1, len(rms) - 1):
        if inside[i] and rms[i] >= level and rms[i] >= rms[i - 1] and rms[i] > rms[i + 1]:
            if peaks and times[i] - peaks[-1][0] < separation:
                if rms[i] > peaks[-1][1]:
                    peaks[-1] = (float(times[i]), float(rms[i]))
                continue
            peaks.append((float(times[i]), float(rms[i])))
    return [t for t, _ in peaks]


def chance_alignment(peaks, spans, step=0.01):
    """Share of own-speech time lying within BEAT_ALIGN_S of a stress peak."""
    grid = [a + k * step for a, b in spans for k in range(int((b - a) / step))]
    if not grid:
        return None
    peaks = np.array(peaks)
    return round(sum(bool(len(peaks)) and float(np.min(np.abs(peaks - t))) <= BEAT_ALIGN_S for t in grid) / len(grid), 3)


# ---------- frames ----------

def robust_sd(values):
    values = list(values)
    if len(values) < 2:
        return 0.
    med = statistics.median(values)
    return 1.4826 * statistics.median(abs(v - med) for v in values)


def runs(frames):
    """Split frames into runs without missing frames."""
    out = []
    for f in frames:
        if out and f["t"] - out[-1][-1]["t"] <= FRAME_GAP_S:
            out[-1].append(f)
        else:
            out.append([f])
    return out


def smooth(values):
    return [statistics.median(values[max(0, i - 1):i + 2]) for i in range(len(values))]


def eye_roll(row):
    """Image-plane roll from the two eye-corner midpoints (rigid points, unaffected by the jaw)."""
    try:
        left = [(row[f"eye_lmk_x_{i}"], row[f"eye_lmk_y_{i}"]) for i in (8, 14)]
        right = [(row[f"eye_lmk_x_{i}"], row[f"eye_lmk_y_{i}"]) for i in (36, 42)]
    except KeyError:
        return None
    if any(v is None or not math.isfinite(v) for p in left + right for v in p):
        return None
    (ax, ay), (bx, by) = (np.mean(left, axis=0), np.mean(right, axis=0))
    return math.degrees(math.atan2(by - ay, bx - ax))


def load_frames(path, character, cues, voiced):
    names = set(pq.read_schema(path).names)
    cols = ["movie_timestamp_sec", "global_frame_index", "openface_success", "head_pose_valid", "face_bbox_xywh",
            "pose_Rx", "pose_Ry", "pose_Rz", *(f"{a}_r" for a in AUS),
            *(f"eye_lmk_{x}_{i}" for x in "xy" for i in (8, 14, 36, 42) if f"eye_lmk_{x}_{i}" in names)]
    frames = []
    for r in pq.read_table(path, columns=cols).to_pylist():
        ok = bool(r["openface_success"] and r["head_pose_valid"])
        box = r["face_bbox_xywh"]
        frames.append({
            "t": r["movie_timestamp_sec"], "frame": r["global_frame_index"], "ok": ok,
            "width": box[2] if box else None, "centre": (box[0] + box[2] / 2, box[1] + box[3] / 2) if box else None,
            "pitch": math.degrees(r["pose_Rx"]) if ok else None, "yaw": math.degrees(r["pose_Ry"]) if ok else None,
            "roll": math.degrees(r["pose_Rz"]) if ok else None, "roll_2d": eye_roll(r) if ok else None,
            "au": {a: r[f"{a}_r"] for a in AUS} if ok else None,
            "speech": speech_state(r["movie_timestamp_sec"], character, cues, voiced),
        })
    return frames


def reliability(frames):
    ok = [f for f in frames if f["ok"]]
    widths = [f["width"] for f in frames if f["width"]]
    width = statistics.median(widths) if widths else 0.
    yaw = statistics.median(abs(f["yaw"]) for f in ok) if ok else None
    flags = [name for name, bad in (("small_face", width < MIN_FACE_PX),
                                    ("profile", yaw is None or yaw > MAX_PROFILE_DEG),
                                    ("low_success", len(ok) < MIN_SUCCESS * len(frames))) if bad]
    return {"face_width_px": round(width), "median_abs_yaw_deg": round(yaw, 1) if yaw is not None else None,
            "success_ratio": round(len(ok) / len(frames), 3) if frames else 0., "flags": flags,
            "reliable": not flags}


# ---------- head ----------

def gaze_windows(segments):
    """Times owned by gaze shifts: each MOVE plus the head-follow window after landing."""
    return [(s["start_s"], s["end_s"] + HEAD_FOLLOW_S) for s in segments if s["kind"] == "MOVE"]


def holds(segments):
    return [(s["start_s"], s["end_s"], s["segment_id"]) for s in segments if s["kind"] == "HOLD"]


def in_any(t, spans):
    return any(a <= t < b for a, b, *_ in spans)


def head_noise(frames):
    """Per-axis measurement jitter: robust SD of frame differences in silent frames, divided by sqrt(2)."""
    silent = [f for f in frames if f["speech"] == "silent"]
    use = silent if len(silent) >= 20 else frames
    noise = {}
    for axis in AXES:
        diffs = [b[axis] - a[axis] for run in runs(use) for a, b in pairwise(run)]
        noise[axis] = robust_sd(diffs) / math.sqrt(2)
    return noise


def excursions(run, axis, threshold):
    """Departures of one axis from its rolling median: peak >= threshold, bounded where |d| < threshold / 2."""
    times = [f["t"] for f in run]
    values = smooth([f[axis] for f in run])
    base = [statistics.median(v for t2, v in zip(times, values) if abs(t2 - t) <= BASELINE_WINDOW_S) for t in times]
    dev = [v - b for v, b in zip(values, base)]
    found, i = [], 0
    while i < len(dev):
        if abs(dev[i]) < threshold / 2:
            i += 1
            continue
        sign, j = math.copysign(1, dev[i]), i
        while j + 1 < len(dev) and dev[j + 1] * sign >= threshold / 2:
            j += 1
        peak = max(range(i, j + 1), key=lambda k: abs(dev[k]))
        if abs(dev[peak]) >= threshold:
            end = times[j + 1] if j + 1 < len(times) else times[j] + (times[j] - times[j - 1] if j else 0.04)
            found.append({"axis": axis, "start": times[i], "end": end, "peak_t": times[peak],
                          "peak_deg": round(dev[peak], 2), "returns": 0 < i and j + 1 < len(dev),
                          "i": i, "j": j})
        i = j + 1
    return found


def classify(axis_excursions):
    """Name excursions; two or more alternating yaw excursions within SHAKE_WINDOW_S form one shake."""
    out = []
    yaw = sorted((e for e in axis_excursions if e["axis"] == "yaw"), key=lambda e: e["start"])
    used = set()
    for k, e in enumerate(yaw):
        if k in used:
            continue
        group = [k]
        for m in range(k + 1, len(yaw)):
            last = yaw[group[-1]]
            if yaw[m]["start"] - e["start"] <= SHAKE_WINDOW_S and yaw[m]["peak_deg"] * last["peak_deg"] < 0:
                group.append(m)
        if len(group) >= 2:
            used |= set(group)
            parts = [yaw[g] for g in group]
            out.append({**parts[0], "end": parts[-1]["end"], "j": parts[-1]["j"], "type": "shake", "reversals": len(group) - 1,
                        "peak_deg": max((p["peak_deg"] for p in parts), key=abs)})
    shaken = {id(yaw[k]) for k in used}
    for e in axis_excursions:
        if id(e) in shaken:
            continue
        duration = e["end"] - e["start"]
        if duration < BEAT_MAX_S and not (e["axis"] == "pitch" and duration >= NOD_S[0] and e["returns"]) and not (
                e["axis"] == "roll" and duration >= TILT_MIN_S):
            kind = "beat"
        elif e["axis"] == "pitch":
            kind = "nod" if NOD_S[0] <= duration <= NOD_S[1] and e["returns"] else "pitch_shift"
        elif e["axis"] == "roll":
            kind = "tilt"
        else:
            kind = "turn"
        out.append({**e, "type": kind})
    return sorted(out, key=lambda e: (e["start"], e["axis"]))


TYPE_ORDER = ("shake", "nod", "tilt", "turn", "pitch_shift", "beat")
SEMANTIC = {"nod", "shake", "tilt"}  # candidates for meaning (agreement, negation, emphasis, attitude)


def merge_axes(components):
    """One physical movement often crosses several axis thresholds at once: overlapping components are one movement."""
    groups = []
    for c in sorted(components, key=lambda c: (c["start"], c["axis"])):
        if groups and c["start"] < max(g["end"] for g in groups[-1]):
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def movement_type(group):
    kinds = [k for k in TYPE_ORDER if any(c["type"] == k for c in group)]
    if len(kinds) > 1 and "beat" in kinds:  # a fast component of a larger gesture is not a separate beat
        kinds.remove("beat")
    return "+".join(kinds)


def head_movements(frames, segments, peaks):
    """Non-gaze head movement inside gaze HOLDs, after removing gaze shifts and their follow windows."""
    noise = head_noise([f for f in frames if f["ok"]])
    thresholds = {a: max(MIN_HEAD_DEG, NOISE_FACTOR * noise[a]) for a in AXES}
    owned, held = gaze_windows(segments), holds(segments)
    usable = [f for f in frames if f["ok"] and in_any(f["t"], held) and not in_any(f["t"], owned)]
    result = []
    for run in runs(usable):
        if len(run) < 5:
            continue
        components = classify([e for axis in AXES for e in excursions(run, axis, thresholds[axis])])
        for group in merge_axes(components):
            main = max(group, key=lambda c: abs(c["peak_deg"]) / thresholds[c["axis"]])
            start, end = min(c["start"] for c in group), max(c["end"] for c in group)
            part = run[min(c["i"] for c in group):max(c["j"] for c in group) + 1]
            states = [f["speech"] for f in part]
            state = max(("speaking", "uncertain", "silent"), key=states.count)
            centres = [f["centre"] for f in part if f["centre"]]
            shift = (math.dist(centres[0], centres[-1]) / statistics.median(f["width"] for f in part if f["width"])
                     if len(centres) > 1 else 0.)
            rolls = [f["roll_2d"] for f in part if f["roll_2d"] is not None]
            result.append({
                "type": movement_type(group), "semantic_candidate": any(c["type"] in SEMANTIC for c in group),
                "axis": main["axis"], "axes": [a for a in AXES if any(c["axis"] == a for c in group)],
                "start": start, "end": end, "peak_t": main["peak_t"], "peak_deg": main["peak_deg"],
                "duration_s": round(end - start, 3), "speech_state": state,
                "stress_aligned": state == "speaking" and any(abs(p - main["peak_t"]) <= BEAT_ALIGN_S for p in peaks),
                "components": [{k: c[k] for k in ("axis", "type", "start", "end", "peak_deg")} for c in group],
                "face_translation_widths": round(shift, 3),
                "eye_line_roll_range_deg": round(max(rolls) - min(rolls), 2) if len(rolls) > 1 else None,
            })
    return result, {a: round(v, 3) for a, v in noise.items()}, thresholds


def postures(frames, segments, thresholds):
    """Sustained offsets of a whole HOLD from the track's median orientation."""
    ok = [f for f in frames if f["ok"]]
    if not ok:
        return []
    centre = {a: statistics.median(f[a] for f in ok) for a in AXES}
    out = []
    for a, b, sid in holds(segments):
        inside = [f for f in ok if a <= f["t"] < b]
        if b - a < POSTURE_MIN_S or len(inside) < 10:
            continue
        offset = {ax: statistics.median(f[ax] for f in inside) - centre[ax] for ax in AXES}
        big = {ax: round(v, 2) for ax, v in offset.items() if abs(v) >= thresholds[ax]}
        if big:
            out.append({"segment_id": sid, "start": a, "end": b, "offset_deg": big})
    return out


def eye_head_lags(frames, segments, threshold):
    """How much of its turn the head has made when the gaze lands, for each human MOVE.

    The head turn is the change from before the MOVE to the settled window HEAD_FOLLOW_S after landing.
    """
    ok = [f for f in frames if f["ok"]]
    starts = [s["start_s"] for s in segments if s["kind"] == "MOVE"]

    def median(a, b):
        xs = [f for f in ok if a <= f["t"] < b]
        return (statistics.median(f["yaw"] for f in xs), statistics.median(f["pitch"] for f in xs)) if len(xs) >= 3 else None
    out = []
    for s in segments:
        if s["kind"] != "MOVE":
            continue
        landing = s["end_s"]
        stop = min([landing + HEAD_FOLLOW_S + 0.5, *(t for t in starts if t > s["start_s"])])
        before, after = median(s["start_s"] - 0.3, s["start_s"]), median(landing + HEAD_FOLLOW_S, stop)
        at = median(landing - 0.05, landing + 0.05) or median(landing - 0.1, landing + 0.1)
        if before is None or after is None or at is None:
            continue
        shift = math.dist(before, after)
        done = math.dist(before, at) / shift if shift else None
        kind = ("eyes_only" if shift < threshold else "head_done_at_landing" if done >= 0.9 else "head_still_moving")
        arrival = next((f["t"] for f in ok if s["start_s"] <= f["t"] < stop
                        and math.dist(before, (f["yaw"], f["pitch"])) >= 0.9 * shift), None) if kind != "eyes_only" else None
        out.append({"segment_id": s["segment_id"], "landing_s": landing, "head_shift_deg": round(shift, 2),
                    "head_done_at_landing": round(done, 2) if done is not None else None, "pattern": kind,
                    "head_90_lag_s": round(arrival - landing, 3) if arrival is not None else None})
    return out


def head_speed(frames, segments):
    """Median head angular speed (deg/s) in gaze HOLDs outside follow windows, by speech state."""
    owned, held = gaze_windows(segments), holds(segments)
    use = [f for f in frames if f["ok"] and in_any(f["t"], held) and not in_any(f["t"], owned)]
    speeds = defaultdict(list)
    for run in runs(use):
        angles = {x: smooth([f[x] for f in run]) for x in AXES}  # 3-frame median removes single-frame jitter
        for k in range(len(run) - 1):
            a, b = run[k], run[k + 1]
            if a["speech"] == b["speech"]:
                step = math.sqrt(sum((angles[x][k + 1] - angles[x][k]) ** 2 for x in AXES))
                speeds[a["speech"]].append(step / (b["t"] - a["t"]))
    return speeds


# ---------- AU ----------

def au_rules(au, state):
    """Baseline condition to compare against, and minimum duration; None excludes the frame."""
    if state == "uncertain":
        return ("speaking", AU_MIN_S) if au in UPPER else None
    if au in OPEN and state == "speaking":
        return None
    if au in MOUTH and state == "speaking":
        return ("speaking", AU_MIN_SPEAKING_S)
    return (state, AU_MIN_S)


def au_episodes(frames):
    ok = [f for f in frames if f["ok"]]
    base = {}
    for au in AUS:
        for state in ("speaking", "silent"):
            values = [f["au"][au] for f in ok if f["speech"] == state]
            if len(values) >= 12:
                base[au, state] = (statistics.median(values), robust_sd(values))
    out = []
    for au in AUS:
        hot = []
        for f in ok:
            rule = au_rules(au, f["speech"])
            if not rule or (au, rule[0]) not in base:
                continue
            med, sd = base[au, rule[0]]
            if f["au"][au] - med >= max(AU_MIN_RISE, AU_Z * sd):
                hot.append((f, rule, f["au"][au] - med))
        groups = []
        for f, rule, rise in hot:
            g = groups[-1] if groups else None
            if g and f["t"] - g[-1][0]["t"] <= MERGE_S + FRAME_GAP_S and g[-1][1] == rule and \
                    g[-1][0]["speech"] == f["speech"]:
                g.append((f, rule, rise))
            else:
                groups.append([(f, rule, rise)])
        for g in groups:
            start, end = g[0][0]["t"], g[-1][0]["t"] + 1 / 23.976
            if end - start < g[0][1][1]:
                continue
            out.append({"au": au, "region": "upper" if au in UPPER else "mouth" if au in MOUTH else "open",
                        "start": start, "end": end, "duration_s": round(end - start, 3),
                        "speech_state": g[0][0]["speech"], "baseline": g[0][1][0],
                        "peak": round(max(f["au"][au] for f, _, _ in g), 2),
                        "peak_rise": round(max(r for _, _, r in g), 2)})
    return sorted(out, key=lambda e: (e["start"], e["au"])), {f"{a}/{s}": [round(m, 2), round(d, 2)]
                                                                for (a, s), (m, d) in base.items()}


def correlation(xs, ys):
    if len(xs) < 12 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return None
    return round(statistics.correlation(xs, ys), 3)


def speech_coupling(tracks, times, rms):
    """Per character and AU: r(AU, speech energy) over all frames vs silent frames only (track medians removed)."""
    by_char = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for t in tracks:
        if not t["reliability"]["reliable"]:
            continue
        ok = [f for f in t["frames"] if f["ok"]]
        energy = np.interp([f["t"] for f in ok], times, rms)
        for au in AUS:
            med = statistics.median(f["au"][au] for f in ok)
            for f, e in zip(ok, energy):
                for group in ("all", f["speech"]):
                    by_char[t["character"]][au][group].append((f["au"][au] - med, float(e)))
    out = {}
    for who, aus in by_char.items():
        out[who] = {au: {g: correlation(*zip(*pairs)) if len(pairs) >= 12 else None
                         for g, pairs in ((g, aus[au].get(g, [])) for g in ("all", "speaking", "silent"))}
                    for au in AUS}
    return out


# ---------- jaw check (68-point landmarks) ----------

def nose_pitch_proxy(row):
    """Nose-tip height below the eye-corner line, in inter-ocular units, after removing roll."""
    p = {i: (row[f"x_{i}"], row[f"y_{i}"]) for i in (30, 36, 45)}
    if any(v is None or not math.isfinite(v) for xy in p.values() for v in xy):
        return None
    (lx, ly), (rx, ry), (nx, ny) = p[36], p[45], p[30]
    dx, dy = rx - lx, ry - ly
    iod = math.hypot(dx, dy)
    if iod < 10:
        return None
    mx, my = (lx + rx) / 2, (ly + ry) / 2
    return ((ny - my) * dx - (nx - mx) * dy) / iod ** 2


def jaw_check(landmarks, tracks):
    """Does OpenFace pitch follow jaw opening (AU26) beyond what rigid points show? Close-up tracks only."""
    rows = pq.read_table(landmarks, columns=["track_id", "extraction_mode", "global_frame_index",
                                              *(f"{x}_{i}" for x in "xy" for i in (30, 36, 45))]).to_pylist()
    proxy = {(r["track_id"], r["global_frame_index"]): nose_pitch_proxy(r) for r in rows
             if r["extraction_mode"] == "static"}
    out = {}
    for t in tracks:
        pairs = [(f["pitch"], proxy.get((t["track_id"], f["frame"])), f["au"]["AU26"], f["speech"])
                 for f in t["frames"] if f["ok"]]
        pairs = [p for p in pairs if p[1] is not None]
        if len(pairs) < 30:
            continue
        pitch, nose, _, _ = zip(*pairs)
        slope, intercept = np.polyfit(nose, pitch, 1)
        speaking = [(p - (slope * n + intercept), p, n, j) for p, n, j, s in pairs if s == "speaking"]
        resid, sp_pitch, sp_nose, sp_jaw = zip(*speaking) if len(speaking) >= 12 else ([], [], [], [])
        out[t["track_id"]] = {
            "frames": len(pairs), "r_pitch_nose_proxy": correlation(pitch, nose),
            "speaking_frames": len(speaking), "r_pitch_au26_speaking": correlation(sp_pitch, sp_jaw),
            "r_nose_proxy_au26_speaking": correlation(sp_nose, sp_jaw),
            "r_pitch_residual_au26_speaking": correlation(resid, sp_jaw),
        }
    return out


# ---------- review ----------

def summary_stats(tracks, coupling, jaw, chance):
    reliable = [t for t in tracks if t["reliability"]["reliable"]]
    lines = ["# Bugonia seq_006 · face and head evidence", "",
             "Machine evidence only. Reliable tracks: close-up (face ≥150 px), not in profile, OpenFace success ≥90 %.",
             "", "## Head movement inside gaze HOLDs", "",
             "| Track | Character | Speaking s | Silent s | Movements per 10 s speaking / silent | Types |",
             "| --- | --- | --- | --- | --- | --- |"]
    for t in reliable:
        ok = [f for f in t["frames"] if f["ok"]]
        held = [f for f in ok if in_any(f["t"], holds(t["segments"])) and not in_any(f["t"], gaze_windows(t["segments"]))]
        secs = {s: sum(f["speech"] == s for f in held) / 23.976 for s in ("speaking", "silent")}
        count = {s: sum(m["speech_state"] == s for m in t["head"]) for s in ("speaking", "silent")}
        rate = " / ".join(f"{10 * count[s] / secs[s]:.1f}" if secs[s] >= 2 else "–" for s in ("speaking", "silent"))
        types = defaultdict(int)
        for m in t["head"]:
            types[m["type"]] += 1
        lines.append(f"| {t['track_id']} | {t['character']} | {secs['speaking']:.1f} | {secs['silent']:.1f} | {rate} | "
                     + ", ".join(f"{k} {v}" for k, v in sorted(types.items())) + " |")
    lines += ["", "Head speed inside gaze HOLDs (median deg/s over frame pairs, all axes):", "",
              "| Character | Speaking | Silent | Frames speaking / silent | Median per-frame jitter (deg) |",
              "| --- | --- | --- | --- | --- |"]
    for who in sorted({t["character"] for t in reliable}):
        pooled = defaultdict(list)
        for t in reliable:
            if t["character"] == who:
                for k, v in t["_speeds"].items():
                    pooled[k] += v
        cell = {k: f"{statistics.median(pooled[k]):.1f}" if len(pooled[k]) >= 20 else "–" for k in ("speaking", "silent")}
        jitter = statistics.median(t["head_noise_deg"][a] for t in reliable if t["character"] == who for a in AXES)
        lines.append(f"| {who} | {cell['speaking']} | {cell['silent']} | {len(pooled['speaking'])} / {len(pooled['silent'])} "
                     f"| {jitter:.2f} |")
    beats = [m for t in reliable for m in t["head"] if m["speech_state"] == "speaking" and m["duration_s"] < BEAT_MAX_S]
    aligned = sum(m["stress_aligned"] for m in beats)
    lines += ["", f"Short movements (<{BEAT_MAX_S} s) during own speech: {len(beats)}; within ±{BEAT_ALIGN_S} s of a "
              f"stress peak: {aligned} ({aligned / len(beats):.0%} vs chance {chance:.0%})." if beats and chance is not None
              else "", "", "## Eye–head timing", ""]
    lags = [x for t in reliable for x in t["eye_head"]]
    if lags:
        pattern = defaultdict(int)
        for x in lags:
            pattern[x["pattern"]] += 1
        shifts = [x["head_shift_deg"] for x in lags]
        lines.append(f"Human gaze shifts with head data: {len(lags)}. Head turn per shift: median {statistics.median(shifts):.1f}° "
                     f"(range {min(shifts):.1f}–{max(shifts):.1f}°). Patterns: "
                     + ", ".join(f"{k} {v}" for k, v in sorted(pattern.items())) + ".")
    lines += ["", "## AU episodes", "", "| Character | Region | Speaking | Silent | Uncertain |", "| --- | --- | --- | --- | --- |"]
    counts = defaultdict(int)
    for t in reliable:
        for e in t["au"]:
            counts[t["character"], e["region"], e["speech_state"]] += 1
    for who in sorted({t["character"] for t in reliable}):
        for region in ("upper", "mouth", "open"):
            lines.append(f"| {who} | {region} | " + " | ".join(str(counts[who, region, s]) for s in
                                                             ("speaking", "silent", "uncertain")) + " |")
    lines += ["", "## AU coupling with speech energy (r, track medians removed)", "",
              "| Character | AU | all frames | own speech | silent |", "| --- | --- | --- | --- | --- |"]
    for who, aus in coupling.items():
        for au in ("AU25", "AU26", "AU12", "AU15", "AU01", "AU04"):
            r = aus[au]
            lines.append(f"| {who} | {au} | " + " | ".join("–" if r[g] is None else f"{r[g]:+.2f}"
                                                         for g in ("all", "speaking", "silent")) + " |")
    lines += ["", "## Jaw check (68-point landmarks)", "",
              ("| Track | r(pitch, nose proxy) | r(pitch, AU26) speaking | r(nose proxy, AU26) speaking | "
               "r(pitch residual, AU26) speaking |"), "| --- | --- | --- | --- | --- |"]
    for tid, j in jaw.items():
        lines.append(f"| {tid} | " + " | ".join("–" if j[k] is None else f"{j[k]:+.2f}" for k in (
            "r_pitch_nose_proxy", "r_pitch_au26_speaking", "r_nose_proxy_au26_speaking",
            "r_pitch_residual_au26_speaking")) + " |")
    lines += ["", "## Semantic head-movement candidates (nod / shake / tilt)", "",
              "| head_id | Character | Start–end s | Type | Speech |", "| --- | --- | --- | --- | --- |"]
    for t in reliable:
        for m in t["head"]:
            if m["semantic_candidate"]:
                lines.append(f"| {m['head_id']} | {t['character']} | {m['start']:.2f}–{m['end']:.2f} | {m['type']} | "
                             f"{m['speech_state']} |")
    unreliable = [f"{t['track_id']} ({', '.join(t['reliability']['flags'])})" for t in tracks if not t["reliability"]["reliable"]]
    lines += ["", "Not used for statistics: " + ", ".join(unreliable), ""]
    return "\n".join(lines)


# ---------- run ----------

def build(args):
    timeline = read(args.full_sequence / "sequence_gaze_timeline.json")
    mapping = read(args.track_mapping)
    labels = read(args.gaze_labels)
    inp = read(args.stage07_run / "stage07_input.json")
    audio = read(args.stage07_run / "audio_affect.json")
    voiced = union([tuple(x) for p in audio["proposals"] for x in p["speech_intervals"]])
    cues = [(s["start"], s["end"], s["speaker_candidate"]) for s in inp["subtitles"]]
    times, rms = rms_envelope(args.stage07_run / "scene_audio.wav", inp["start"])
    people = {t["track_id"]: t["character"] for t in mapping["tracks"] if t["measured"]}
    segments = {(t["shot_id"], t["character"]): t["segments"] for t in labels["tracks"] if t["status"] == "annotated"}
    own = defaultdict(list)
    for a, b, who in cues:
        own[who] += [[max(a, x), min(b, y)] for x, y in voiced if max(a, x) < min(b, y)]
    peaks = {who: stress_peaks(times, rms, spans) for who, spans in own.items()}
    tracks = []
    for shot in timeline["shots"]:
        for track in shot["tracks"]:
            who = people.get(track["track_id"])
            if not track.get("source") or who is None:
                continue
            path = Path(track["source"]) / "eye_position_frames.parquet"
            frames = load_frames(path, who, cues, voiced)
            segs = segments.get((shot["shot_id"], who), [])
            record = {"track_id": track["track_id"], "shot_id": shot["shot_id"], "character": who,
                      "source": str(path), "sha256": sha256_file(path), "frames": frames, "segments": segs,
                      "reliability": reliability(frames)}
            head, noise, thresholds = head_movements(frames, segs, peaks.get(who, []))
            for n, m in enumerate(head, 1):
                m["head_id"] = f"{track['track_id']}_head_{n:03d}"
            speeds = head_speed(frames, segs)
            record["head_speed_deg_s"] = {k: {"frames": len(v), "median": round(statistics.median(v), 2)}
                                          for k, v in speeds.items() if v}
            record["_speeds"] = speeds
            record.update(head=head, head_noise_deg=noise, head_threshold_deg={a: round(v, 2) for a, v in thresholds.items()},
                          postures=postures(frames, segs, thresholds),
                          eye_head=eye_head_lags(frames, segs, max(MIN_HEAD_DEG, thresholds["yaw"])))
            episodes, baselines = au_episodes(frames)
            for n, e in enumerate(episodes, 1):
                e["au_id"] = f"{track['track_id']}_au_{n:03d}"
            record.update(au=episodes, au_baselines=baselines,
                          speech_frames={s: sum(f["speech"] == s for f in frames) for s in ("speaking", "uncertain", "silent")})
            tracks.append(record)
    spans = [s for v in own.values() for s in v]
    chance = chance_alignment([p for v in peaks.values() for p in v], spans)
    coupling = speech_coupling(tracks, times, rms)
    jaw = jaw_check(args.landmarks, [t for t in tracks if t["reliability"]["reliable"]])
    for t in tracks:
        check = jaw.get(t["track_id"])
        verdict = "unchecked" if not check or check["r_pitch_residual_au26_speaking"] is None else (
            "possibly_jaw_driven" if check["r_pitch_residual_au26_speaking"] >= 0.3 else "rigid_points_agree")
        for m in t["head"]:
            if "pitch" in m["axes"] and m["speech_state"] == "speaking":
                m["jaw_check"] = verdict
    return timeline, mapping, tracks, coupling, jaw, chance, peaks


def run(args):
    output = args.output_dir
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    timeline, _, tracks, coupling, jaw, chance, peaks = build(args)
    sources = {name: {"file": str(path), "sha256": sha256_file(path)} for name, path in (
        ("timeline", args.full_sequence / "sequence_gaze_timeline.json"), ("track_mapping", args.track_mapping),
        ("gaze_labels", args.gaze_labels), ("stage07_input", args.stage07_run / "stage07_input.json"),
        ("audio_affect", args.stage07_run / "audio_affect.json"), ("scene_audio", args.stage07_run / "scene_audio.wav"),
        ("landmarks", args.landmarks))}
    parameters = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, tuple))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        staged = Path(temp) / output.name
        staged.mkdir()
        write_json(staged / "face_head_evidence.json", {
            "schema_version": VERSION, "movie_id": timeline["movie_id"], "sequence_id": timeline["sequence_id"],
            "human_verified": False, "parameters": parameters, "sources": sources,
            "stress_peak_chance_alignment": chance, "stress_peaks": {k: len(v) for k, v in peaks.items()},
            "speech_coupling": coupling, "jaw_check": jaw,
            "tracks": [{k: v for k, v in t.items() if k not in ("frames", "segments", "_speeds")} for t in tracks],
        })
        (staged / "summary.md").write_text(summary_stats(tracks, coupling, jaw, chance), encoding="utf-8")
        staged.rename(output)
    print(f"{len(tracks)} tracks, {sum(len(t['head']) for t in tracks)} head movements "
          f"({sum(m['semantic_candidate'] for t in tracks for m in t['head'])} semantic candidates), "
          f"{sum(len(t['au']) for t in tracks)} AU episodes → {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-sequence", required=True, type=Path)
    parser.add_argument("--track-mapping", required=True, type=Path, help="Stage06.3 character_track_mapping.json")
    parser.add_argument("--gaze-labels", required=True, type=Path)
    parser.add_argument("--stage07-run", required=True, type=Path, help="prepared Stage07 run (input, VAD, audio)")
    parser.add_argument("--landmarks", required=True, type=Path, help="Stage05.4 frame_features.parquet")
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
