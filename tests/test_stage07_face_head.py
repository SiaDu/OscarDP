"""Synthetic checks for the Stage07 face/head evidence functions."""
import math

import pytest

from oscardp.stage07 import face_head as fh

FPS = 23.976


def frame(t, pitch=0., yaw=0., roll=0., speech="silent", au=None):
    values = dict.fromkeys(fh.AUS, 0.2)
    values.update(au or {})
    return {"t": t, "frame": round(t * FPS), "ok": True, "width": 600, "centre": (500., 400.), "pitch": pitch,
            "yaw": yaw, "roll": roll, "roll_2d": roll, "au": values, "speech": speech}


def times(start, end):
    return [start + k / FPS for k in range(round((end - start) * FPS))]


def bump(t, centre, half_width, height):
    return height * max(0., 1 - abs(t - centre) / half_width)


def test_speech_state_uses_attribution_voice_and_margin():
    cues = [(10., 12., "MICHELLE"), (15., 16., "Unknown")]
    voiced = [[10.5, 11.5], [15.2, 15.8]]
    assert fh.speech_state(10.4, "MICHELLE", cues, voiced) == "speaking"  # 0.2 s before voicing
    assert fh.speech_state(10.2, "MICHELLE", cues, voiced) == "silent"
    assert fh.speech_state(11., "TEDDY", cues, voiced) == "silent"  # someone else's line
    assert fh.speech_state(15.5, "TEDDY", cues, voiced) == "uncertain"
    assert fh.speech_state(13., "MICHELLE", cues, voiced) == "silent"  # no voice


def test_au_rules_separate_articulation_from_expression():
    assert fh.au_rules("AU26", "speaking") is None
    assert fh.au_rules("AU26", "silent") == ("silent", fh.AU_MIN_S)
    assert fh.au_rules("AU12", "speaking") == ("speaking", fh.AU_MIN_SPEAKING_S)
    assert fh.au_rules("AU12", "uncertain") is None
    assert fh.au_rules("AU01", "uncertain") == ("speaking", fh.AU_MIN_S)


def test_au_episodes_respect_speech_gating():
    frames = [frame(t, speech="speaking" if t < 5 else "silent",
                    au={"AU26": 2.5 if 1 <= t < 2 or 6 <= t < 6.5 else 0.2,
                        "AU12": 2.0 if 2.5 <= t < 2.7 or 3.5 <= t < 4.0 else 0.2})
              for t in times(0, 10)]
    found = {(e["au"], e["speech_state"], round(e["start"], 1)) for e in fh.au_episodes(frames)[0]}
    assert ("AU26", "silent", 6.0) in found
    assert not any(au == "AU26" and state == "speaking" for au, state, _ in found)  # jaw during speech = lip-sync
    assert ("AU12", "speaking", 3.5) in found and ("AU12", "speaking", 2.5) not in found  # 0.2 s < 0.3 s


def test_head_movements_inside_holds_only():
    def angles(t):
        return {"pitch": bump(t, 3.0, 0.3, 6.) + bump(t, 7.6, 0.3, 6.),  # nod in the hold; nod in the follow window
                "yaw": bump(t, 5.0, 0.15, 5.) - bump(t, 5.4, 0.15, 5.),  # shake
                "roll": bump(t, 8.5, 0.12, 5.)}  # short pulse
    frames = [frame(t, **angles(t), speech="speaking") for t in times(0, 10)]
    segments = [{"segment_id": "g1", "kind": "HOLD", "start_s": 0., "end_s": 7.},
                {"segment_id": "g2", "kind": "MOVE", "start_s": 7., "end_s": 7.2},
                {"segment_id": "g3", "kind": "HOLD", "start_s": 7.2, "end_s": 10.}]
    moves, _, thresholds = fh.head_movements(frames, segments, peaks=[8.5])
    kinds = [(m["type"], round(m["peak_t"], 1)) for m in moves]
    assert ("nod", 3.0) in kinds and ("shake", 5.0) in kinds and ("beat", 8.5) in kinds
    assert not any(abs(t - 7.6) < 0.2 for _, t in kinds)  # gaze-shift follow window is excluded
    assert next(m for m in moves if m["type"] == "beat")["stress_aligned"]
    assert thresholds == {a: fh.MIN_HEAD_DEG for a in fh.AXES}


def test_eye_head_patterns():
    def yaw(t):
        return 0. if t < 1. else 10. * min(1., (t - 1.) / 0.6)  # the head needs 0.6 s, the gaze lands at 1.2 s
    frames = [frame(t, yaw=yaw(t)) for t in times(0, 4)]
    segments = [{"segment_id": "g1", "kind": "MOVE", "start_s": 1., "end_s": 1.2}]
    (lag,) = fh.eye_head_lags(frames, segments, 3.)
    assert lag["pattern"] == "head_still_moving" and lag["head_shift_deg"] == pytest.approx(10., abs=0.1)
    assert lag["head_done_at_landing"] == pytest.approx(0.33, abs=0.08) and lag["head_90_lag_s"] > 0.2
    still = [frame(t) for t in times(0, 4)]
    assert fh.eye_head_lags(still, segments, 3.)[0]["pattern"] == "eyes_only"


def test_nose_pitch_proxy_ignores_roll():
    def row(angle, drop):
        c, s = math.cos(angle), math.sin(angle)
        pts = {36: (-50., 0.), 45: (50., 0.), 30: (0., drop)}
        return {f"{a}_{i}": v for i, (x, y) in pts.items()
                for a, v in (("x", 500 + c * x - s * y), ("y", 400 + s * x + c * y))}
    assert fh.nose_pitch_proxy(row(0., 40.)) == pytest.approx(0.4)
    assert fh.nose_pitch_proxy(row(0.3, 40.)) == pytest.approx(0.4)
    assert fh.nose_pitch_proxy(row(0., 30.)) < fh.nose_pitch_proxy(row(0., 40.))


def test_stress_peaks_and_chance():
    import numpy as np
    t = np.arange(0, 2, 0.01)
    rms = np.where((t > 0.5) & (t < 0.52), 1., 0.1) + np.where((t > 1.5) & (t < 1.52), 0.8, 0.)
    peaks = fh.stress_peaks(t, rms, [[0., 2.]])
    assert len(peaks) == 2 and abs(peaks[0] - 0.51) < 0.02
    assert fh.chance_alignment(peaks, [[0., 2.]]) == pytest.approx(0.2, abs=0.02)


def test_overlapping_axes_are_one_movement():
    frames = [frame(t, pitch=bump(t, 3.0, 0.3, 6.), roll=bump(t, 3.05, 0.3, 5.) + bump(t, 6.0, 0.1, 5.),
                    yaw=bump(t, 3.0, 0.1, 8.)) for t in times(0, 8)]
    segments = [{"segment_id": "g1", "kind": "HOLD", "start_s": 0., "end_s": 8.}]
    moves, _, _ = fh.head_movements(frames, segments, peaks=[])
    assert [(m["type"], m["axes"]) for m in moves] == [("nod+tilt", ["pitch", "yaw", "roll"]), ("beat", ["roll"])]
    assert moves[0]["semantic_candidate"] and not moves[1]["semantic_candidate"]
    assert {c["type"] for c in moves[0]["components"]} == {"nod", "tilt", "beat"}  # the fast yaw part is folded in
