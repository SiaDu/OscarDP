"""Adapters for the actual Stage02/04 and Stage05.6 prototype JSONs."""
import json
import math
from pathlib import Path

from oscardp.shots.transnet import sha256_file
from oscardp.stage02.srt import parse_srt

MOVIE = "tt12300742"
GAZE_SCHEMA = "stage06.3.gaze_labels.v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def read_prepared_input(directory):
    """Read Stage07 input or Legacy Compatibility: the old acting pilot input.

    Only the in-memory stage identifier changes; source files, IDs and times stay intact.
    """
    directory = Path(directory)
    current = directory / "stage07_input.json"
    path = current if current.exists() else directory / "stage06_input.json"
    document = read(path)
    if path.name == "stage06_input.json":
        document["schema_version"] = "stage07.research_input.v1"
        document["legacy_compatibility"] = {
            "input_file": str(path),
            "note": "Legacy Stage06 acting prototype, now owned by Stage07; read-only compatibility.",
        }
    return document


def interval(start, end, bounds=None):
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            or not math.isfinite(start) or not math.isfinite(end) or start >= end):
        raise ValueError(f"Invalid half-open time interval: {start}, {end}")
    if bounds and not bounds[0] - 1e-6 <= start < end <= bounds[1] + 1e-6:
        raise ValueError(f"Interval outside selected movie range: {start}, {end}")


def overlap(a, b):
    return max(0., min(a[1], b[1]) - max(a[0], b[0]))


def load_gaze(path):
    """Human Stage06.3 gaze labels; the only human-verified input of this prototype."""
    labels = read(path)
    if labels["schema_version"] != GAZE_SCHEMA or labels["movie_id"] != MOVIE or labels["human_verified"] is not True:
        raise ValueError("Expected human-verified Stage06.3 gaze labels for this movie")
    return labels


def gaze_refs(labels, start, end):
    """Gaze segments inside the excerpt, clipped to it; MOVE segments also carry where the gaze came from."""
    refs, unavailable = [], []
    for track in labels["tracks"]:
        if track["status"] != "annotated":
            unavailable.append({k: track[k] for k in ("shot_id", "character", "reason")})
            continue
        origin = {s["segment_id"]: s["from"] for s in track["shifts"]}
        for s in track["segments"]:
            if overlap((start, end), (s["start_s"], s["end_s"])):
                refs.append(dict(gaze_id=s["segment_id"], shot_id=track["shot_id"], character_candidate=track["character"],
                                 kind=s["kind"], target=s["target"], form=s["form"], form_text=s["form_text"],
                                 from_gaze=origin.get(s["segment_id"]), blink=s["blink"], certainty=s["certainty"],
                                 start=max(start, s["start_s"]), end=min(end, s["end_s"]),
                                 original_start=s["start_s"], original_end=s["end_s"], human_verified=True))
    return refs, unavailable


FACE_HEAD_SCHEMA = "stage07.face_head.v1"


def face_head_refs(doc, start, end):
    """Machine AU episodes and head movements from reliable tracks inside the excerpt."""
    if doc["schema_version"] != FACE_HEAD_SCHEMA or doc["movie_id"] != MOVIE:
        raise ValueError("Expected Stage07 face/head evidence for this movie")
    refs = []
    for t in doc["tracks"]:
        if not t["reliability"]["reliable"]:
            continue
        common = dict(shot_id=t["shot_id"], track_id=t["track_id"], character_candidate=t["character"],
                      human_verified=False)
        for e in t["au"]:
            if overlap((start, end), (e["start"], e["end"])):
                refs.append(dict(evidence_id=e["au_id"], modality="au", label=e["au"], region=e["region"],
                                 speech_state=e["speech_state"], peak_rise=e["peak_rise"],
                                 start=max(start, e["start"]), end=min(end, e["end"]), **common))
        for m in t["head"]:
            if overlap((start, end), (m["start"], m["end"])):
                refs.append(dict(evidence_id=m["head_id"], modality="head", label=m["type"], axis=m["axis"],
                                 axes=m.get("axes", [m["axis"]]), semantic_candidate=m.get("semantic_candidate"),
                                 peak_deg=m["peak_deg"], speech_state=m["speech_state"],
                                 jaw_check=m.get("jaw_check"), start=max(start, m["start"]), end=min(end, m["end"]),
                                 **common))
    return refs


def load_sources(pilot, gaze=None, face_head=None):
    pilot = Path(pilot).resolve()
    # Legacy Compatibility: immutable Stage05.6 packet filename/schema, not the new gaze stage.
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
    if gaze:
        sources["stage06_3_gaze"] = Path(gaze).resolve()
        data["gaze"] = load_gaze(sources["stage06_3_gaze"])
        data["gaze_sha256"] = sha256_file(sources["stage06_3_gaze"])
    if face_head:
        sources["stage07_face_head"] = Path(face_head).resolve()
        data["face_head"] = read(sources["stage07_face_head"])
        data["face_head_sha256"] = sha256_file(sources["stage07_face_head"])
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
    gaze = {}
    if data.get("gaze"):
        refs, unavailable = gaze_refs(data["gaze"], start, end)
        gaze = dict(stage06_3_gaze_sha256=data["gaze_sha256"], stage06_3_gaze_references=refs,
                    stage06_3_gaze_unavailable=unavailable)
    if data.get("face_head"):
        gaze.update(face_head_sha256=data["face_head_sha256"],
                    face_head_references=face_head_refs(data["face_head"], start, end))
    return dict(schema_version="stage07.research_input.v1", movie_id=MOVIE, scene_id=context["scene_id"],
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
                subtitles=subtitles, stage02_context=shots, stage04_character_context=char_context, **gaze)
