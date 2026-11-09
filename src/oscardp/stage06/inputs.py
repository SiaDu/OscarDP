"""Adapters for the actual Stage02/04 and Stage05.6 prototype JSONs."""
import json
import math
from pathlib import Path

from oscardp.stage02.srt import parse_srt

MOVIE = "tt12300742"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def interval(start, end, bounds=None):
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            or not math.isfinite(start) or not math.isfinite(end) or start >= end):
        raise ValueError(f"Invalid half-open time interval: {start}, {end}")
    if bounds and not bounds[0] - 1e-6 <= start < end <= bounds[1] + 1e-6:
        raise ValueError(f"Interval outside selected movie range: {start}, {end}")


def overlap(a, b):
    return max(0., min(a[1], b[1]) - max(a[0], b[0]))


def load_sources(pilot):
    pilot = Path(pilot).resolve()
    snapshot = read(pilot / "stage06_input_pilot.json")
    config = read(pilot / "run_config.json")
    packet = snapshot["packet"]
    if packet["movie_id"] != MOVIE:
        raise ValueError("This pilot CLI only processes Bugonia")
    paths = list(config["input_sha256"])
    # Reuse recorded source paths, without introducing a new manifest/hash system.
    sources = {}
    for n in (1, 2, 4):
        found = [p for p in paths if Path(p).name == f"Stage0{n}_{MOVIE}.json"]
        if len(found) != 1:
            raise ValueError(f"Missing/ambiguous Stage0{n} input")
        sources[f"stage0{n}"] = Path(found[0])
    event_run = Path(config["event_run"])
    events = read(event_run / "performance_events.json")
    movie_path = read(event_run / "run_config.json")["source_provenance"]["earlier_provenance"]["video"]["path"]
    sources.update(video=Path(movie_path), srt=Path(movie_path).with_suffix(".srt"),
                   stage05=event_run / "performance_events.json", stage05_6=pilot / "stage06_input_pilot.json")
    data = {k: read(sources[k]) for k in ("stage01", "stage02", "stage04")}
    if any(d["movie_id"] != MOVIE for d in data.values()) or events["movie_id"] != MOVIE:
        raise ValueError("Movie ID mismatch")
    parsed = parse_srt(sources["srt"].read_text(encoding="utf-8-sig"))
    if parsed.issues:
        raise ValueError(f"Original SRT has parse issues: {parsed.issues[:2]}")
    data.update(paths={k: str(v) for k, v in sources.items()}, packet=packet, subtitles=parsed.cues)
    event_map = {e["event_id"]: e for e in events["events"]}
    if len(event_map) != len(events["events"]):
        raise ValueError("Duplicate Stage05 Event ID")
    for e in packet["event_candidates"]:
        original = event_map[e["event_id"]]
        for key in ("shot_id", "track_id", "start_sec", "end_sec", "review_status", "feature"):
            if e[key] != original[key]:
                raise ValueError(f"Stage05.5/05.6 disagree: {e['event_id']} {key}")
        interval(e["start_sec"], e["end_sec"])
        if e["review_status"] != "algorithm_candidate" or e["human_verified"]:
            raise ValueError("Unexpected event review authority")
    data["events"] = packet["event_candidates"]
    return data


def event_refs(data, start, end):
    mapping = {}
    for scene in data["packet"]["scenes"]:
        for shot in scene["shots"]:
            for track in shot["face_tracks"]:
                candidates = track["identity_cluster"]["candidate_mapping"]["candidates"]
                names = {c["character_name"] for c in candidates if c.get("character_name")}
                mapping[track["track_id"]] = (next(iter(names)) if len(names) == 1 else "Unknown")
    keys = ("event_id", "shot_id", "track_id", "event_type", "feature", "qc_flags", "boundary_status",
            "review_status", "human_verified", "evidence_reference", "left_censored", "right_censored")
    return [{**{k: e[k] for k in keys}, "start": e["start_sec"], "end": e["end_sec"],
             "character_candidate": mapping.get(e["track_id"], "Unknown"),
             "character_mapping_status": "unverified",
             "speech_sensitive_feature": e["measurements"].get("speech_sensitive_feature", False)}
            for e in data["events"] if overlap((start, end), (e["start_sec"], e["end_sec"]))]


def selection(data, start=None, end=None):
    s2 = {s["shot_id"]: s for s in data["stage02"]["shots"]}
    # Pilot starts at the available corrected cut, and ends with the complete negotiation speech.
    corrected = {s["shot_id"]: s["effective_processing_temporal"]
                 for sc in data["packet"]["scenes"] for s in sc["shots"]}
    default_start = corrected["shot_000211"]["start"]
    default_end = s2["shot_000223"]["temporal"]["end"]
    start = default_start if start is None else start
    end = default_end if end is None else end
    interval(start, end)
    scenes = [s for s in data["stage04"]["scenes"] if s["start_seconds"] <= start < end <= s["end_seconds"]]
    if len(scenes) != 1:
        raise ValueError("Select a continuous excerpt inside one Stage04 scene")
    context = scenes[0]
    shots = [s for s in data["stage02"]["shots"]
             if overlap((start, end), (s["temporal"]["start"], s["temporal"]["end"]))]
    shot_ids = {s["shot_id"] for s in shots}
    char_context = [s for s in data["stage04"]["shots"] if s["shot_id"] in shot_ids]
    speakers = {}
    for shot in shots:
        for d in shot["dialogue"]:
            for idx in d["srt_indices"]:
                speakers.setdefault(idx, set()).add(None if d.get("voice_overlap") else d["speaker"])
    subtitles = []
    for c in data["subtitles"]:
        if overlap((start, end), (c.start, c.end)):
            candidates = speakers.get(c.index, {None})
            candidate = next(iter(candidates)) if len(candidates) == 1 else None
            subtitles.append(dict(srt_index=c.index, start=max(start, c.start), end=min(end, c.end),
                                  original_start=c.start, original_end=c.end, text=c.text,
                                  speaker_candidate=candidate or "Unknown",
                                  speaker_status="Stage02 narrative attribution; not voice identity verification"))
    candidates = [
        ("Proclamation and initial awakening", s2["shot_000202"]["temporal"]["start"],
         s2["shot_000207"]["temporal"]["start"], "Complete early exchange; no measured Stage05 events."),
        ("Hair/ship dispute and negotiation opening", s2["shot_000207"]["temporal"]["start"],
         s2["shot_000219"]["temporal"]["start"], "Clear change and all five event Shots, but cuts a continuing negotiation speech."),
        ("Interruption through complete negotiation offer", default_start, default_end,
         "Selected: interruption, attempt to regain control and completed offer; 60–120 second continuous excerpt.")]
    names = sorted({c["character_name"] for s in char_context for c in s["character_candidates"] if c.get("character_name")})
    return dict(schema_version="stage06.research_input.v1", movie_id=MOVIE, scene_id=context["scene_id"],
                status="research_prototype", time_basis="movie_global_seconds_start_inclusive_end_exclusive",
                start=start, end=end, duration=end-start, character_candidate=names,
                character_mapping_status="unverified", sources=data["paths"],
                scene_context=context, stage02_context_references=[s["shot_id"] for s in shots],
                stage04_context_references=[s["shot_id"] for s in char_context],
                stage05_event_references=event_refs(data, start, end),
                scene_candidates=[dict(title=t, start=a, end=b, duration=b-a,
                                       event_count=len(event_refs(data, a, b)), reason=r) for t,a,b,r in candidates],
                audio_video_coverage={"requested": [start, end], "source": data["paths"]["video"],
                                      "audio": "pending_extraction", "video": "pending_extraction"},
                boundary_note="Start uses experimental Stage05 corrected Shot211 cut; end uses original Stage01 Shot223 end. Neither is human-verified. This is an excerpt, not the full 444 s Stage04 scene.",
                subtitles=subtitles, stage02_context=shots, stage04_character_context=char_context)
