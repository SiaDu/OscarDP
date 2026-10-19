"""Frozen-selected, inference-free frame upgrades. No movie-directory publication."""
from __future__ import annotations

import copy
import fcntl
import os
import re
from pathlib import Path

from .discovery import discover_movies
from .frame_samples import (POLICY, atomic_json, image_info, load_json, physical_samples,
                            publish_image, safe_path, sync_dir, append_receipt, read_receipts)
from .media import extract_selected_frames, probe_video, scan_timeline
from .progress import ProgressReporter
from .transnet import sha256_file
from .validation import IMAGE_NAME, MOVIE_ID, validate_movie

RELEASE_ID = "oscardp_stage02_v1_0_18movies_20261018"
MANIFEST_SHA256 = "375a58f5f94fbaae1036efa790475b794d7e60130b9686f96e7dfb2fbf44b443"


def select_movies(manifest: Path, expected_sha256: str, *, movie_id: str | None,
                  exclude_movie_id: str | None, expect_movies: int) -> tuple[dict, list[dict]]:
    safe_path(manifest.parent, manifest.name)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or sha256_file(manifest) != expected_sha256:
        raise ValueError("Frozen Manifest SHA-256 mismatch")
    data = load_json(manifest)
    if (data.get("dataset_status") != "FROZEN" or data.get("activated_for_pipeline") is not False
            or data.get("release_id") != manifest.parent.name):
        raise ValueError("Invalid Frozen release ID, status or activation")
    if expected_sha256 == MANIFEST_SHA256 and data["release_id"] != RELEASE_ID:
        raise ValueError("Unexpected pinned release ID")
    movies = data["movies"]
    ids = [m["movie_id"] for m in movies]
    if (len(ids) != len(set(ids)) or len(ids) != data["movie_count"]
            or not all(MOVIE_ID.fullmatch(mid) for mid in ids)):
        raise ValueError("Invalid frozen movie selection")
    if movie_id and exclude_movie_id:
        raise ValueError("--movie-id and --exclude-movie-id are mutually exclusive")
    if movie_id and movie_id not in ids or exclude_movie_id and exclude_movie_id not in ids:
        raise ValueError("Requested movie identity is absent from Frozen Manifest")
    selected = [m for m in movies if (not movie_id or m["movie_id"] == movie_id)
                and m["movie_id"] != exclude_movie_id]
    if expect_movies < 1 or len(selected) != expect_movies:
        raise ValueError(f"Expected {expect_movies} movies, selected {len(selected)}")
    if data["release_id"] == RELEASE_ID and exclude_movie_id == "tt12300742":
        if len(ids) != 18 or len(selected) != 17:
            raise ValueError("Pinned release exclusion must select exactly 17 of 18 movies")
    for movie in selected:
        if movie.get("release_eligible") is not True or movie.get("activated_for_pipeline") is not False:
            raise ValueError("Frozen movie is not eligible/inactive")
        if set(movie["inputs"]) != {"Stage01", "Stage02", "SRT"}:
            raise ValueError("Expected frozen Stage01, Stage02 and SRT inputs")
        for kind, item in movie["inputs"].items():
            path = safe_path(manifest.parent, item["path"])
            if sha256_file(path) != item["sha256"]:
                raise ValueError(f"Frozen selected input hash mismatch: {path}")
            if kind != "SRT" and load_json(path).get("movie_id") != movie["movie_id"]:
                raise ValueError(f"Frozen movie identity mismatch: {path}")
    return data, selected


def verify_inputs(root: Path, movie: dict) -> dict:
    mid = movie["movie_id"]
    hashes = {}
    for kind, item in movie["inputs"].items():
        name = f"{mid}.srt" if kind == "SRT" else f"{kind}_{mid}.json"
        path = safe_path(root, f"{mid}/{name}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Canonical {kind} hash differs from frozen selection: {mid}")
        if kind != "SRT" and load_json(path).get("movie_id") != mid:
            raise ValueError(f"Canonical movie identity mismatch: {path}")
        hashes[kind] = item["sha256"]
    return hashes


def resolve_video(input_root: Path, mid: str, metadata: dict, fingerprint: dict) -> tuple[Path, dict]:
    # Metadata supplies the authoritative relative path; discovery uses IMDb IDs,
    # never similar titles, and rejects ambiguous candidates before any writes.
    original = safe_path(input_root, metadata["source_video_relpath"])
    refs = discover_movies(input_root, movie_key=mid)
    candidates = {ref.source_path for ref in refs}
    if original.is_file():
        candidates.add(original.resolve())
    if len(candidates) != 1:
        raise ValueError(f"Exactly one authoritative source video required: {mid}")
    video = candidates.pop()
    if original.is_file() and video != original.resolve():
        raise ValueError("Source discovery contradicts processing metadata")
    recorded_path = Path(fingerprint["source_path"])
    if recorded_path.exists() and recorded_path.resolve() != video:
        raise ValueError("Source path differs from original processing fingerprint")
    stat = video.stat()
    if (stat.st_size != fingerprint.get("source_size")
            or stat.st_mtime_ns != fingerprint.get("source_mtime_ns")):
        raise ValueError(f"Source video fingerprint mismatch: {mid}")
    digest = sha256_file(video)
    if fingerprint.get("source_sha256", digest) != digest:
        raise ValueError(f"Source video content hash mismatch: {mid}")
    return video, {"path": str(video), "sha256": digest, "bytes": stat.st_size,
                   "mtime_ns": stat.st_mtime_ns,
                   "original_identity_basis": "Stage01 source path/IMDb discovery + original size/mtime + verified decoded timeline"}


def _verify_timeline(video: Path, metadata: dict, shots: list[dict]) -> dict[int, float]:
    actual, _ = probe_video(video, metadata["source_video_relpath"])
    for key in ("width", "height", "codec_name"):
        if getattr(actual, key) != metadata[key]:
            raise ValueError(f"Source video media identity mismatch: {key}")
    if abs(actual.fps - metadata["fps"]) > max(1e-6, actual.fps * 1e-6):
        raise ValueError("Source video FPS mismatch")
    timeline = scan_timeline(video, actual.fps)
    if timeline.frame_count != metadata["frame_count"] or timeline.is_vfr != metadata["is_vfr"]:
        raise ValueError("Source video decoded timeline mismatch")
    if abs(timeline.exclusive_end_sec - metadata["duration_sec"]) > 1e-4:
        raise ValueError("Source video duration mismatch")
    pts = {}
    for shot in shots:
        temporal = shot["temporal"]
        start, end = temporal["start_frame"], temporal["end_frame"]
        if (abs(timeline.timestamp(start) - temporal["start"]) > 1e-5
                or abs(timeline.exclusive_timestamp(end) - temporal["end"]) > 1e-5):
            raise ValueError("Shot timestamps contradict decoded source timeline")
        for frame in {s["frame_index"] for s in physical_samples(shot["shot_id"], start, end).values()}:
            pts[frame] = timeline.pts_sec[frame]
    return pts


def plan_movie(input_root: Path, output_root: Path, movie: dict, release_id: str,
               manifest_hash: str) -> dict:
    mid = movie["movie_id"]
    protected = verify_inputs(output_root, movie)
    movie_dir = safe_path(output_root, mid)
    internal = safe_path(output_root, f".internal/{mid}")
    metadata = load_json(safe_path(internal, "video_metadata.json"))
    fingerprint = load_json(safe_path(internal, "fingerprint.json"))
    video, identity = resolve_video(input_root, mid, metadata, fingerprint)
    shots = load_json(movie_dir / f"Stage01_{mid}.json")["shots"]
    journal_path = safe_path(internal, "frame_upgrade_journal.json")
    journal = load_json(journal_path) if journal_path.exists() else None
    binding = {"movie_id": mid, "release_id": release_id, "manifest_sha256": manifest_hash,
               "protected_inputs": protected, "source_video": identity,
               "metadata_sha256": sha256_file(internal / "video_metadata.json"),
               "fingerprint_sha256": sha256_file(internal / "fingerprint.json")}
    if journal and journal.get("binding") != binding:
        raise ValueError(f"Upgrade journal belongs to different inputs: {mid}")
    image_dir = safe_path(movie_dir, "shot_img")
    inventory = {path.name: {"path": str(path), **image_info(safe_path(image_dir, path.name))}
                 for path in sorted(image_dir.iterdir())}
    if not journal:
        if safe_path(internal, "frame_samples_v1.json").exists() or safe_path(internal, "frame_upgrade_receipts.jsonl").exists():
            raise ValueError("Existing sample metadata has no upgrade journal; refusing to overwrite it")
        historical = validate_movie(movie_dir, frame_count=metadata["frame_count"], frame_profile="legacy")
        if not historical.passed:
            raise ValueError("Legacy validation failed: " + "; ".join(historical.errors))
        pts = _verify_timeline(video, metadata, shots)
        sample_record = {"policy": POLICY, "movie_id": mid, "status": "upgrading",
                         "stage01_sha256": protected["Stage01"], "source_video": identity,
                         "renames": [], "shots": {}}
        for shot in shots:
            sid, temporal = shot["shot_id"], shot["temporal"]
            roles = physical_samples(sid, temporal["start_frame"], temporal["end_frame"])
            matches = [name for name in inventory if IMAGE_NAME.fullmatch(name) and name.startswith(sid + "_")]
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous legacy Middle: {sid}")
            legacy = matches[0]
            # Historical filename is a millisecond representation of the true midpoint.
            from .schema import format_timestamp, rounded_seconds
            middle_frame = roles["middle"]["frame_index"]
            middle_time = (middle_frame / metadata["fps"] if not metadata["is_vfr"] else pts[middle_frame])
            expected_name = f"{sid}_{format_timestamp(rounded_seconds(middle_time)).replace(':', '-').replace('.', '-')}.jpg"
            if legacy != expected_name:
                raise ValueError(f"Legacy Middle filename contradicts source timing: {legacy}")
            sample_record["renames"].append({"shot_id": sid, "original_path": f"shot_img/{legacy}",
                "original_sha256": inventory[legacy]["sha256"], "new_path": roles["middle"]["image_relpath"]})
            for sample in roles.values():
                sample.update(timestamp_sec=pts[sample["frame_index"]],
                              timestamp_source="decoded_pts_relative_to_first_frame", sha256=None)
                if sample["frame_index"] == roles["middle"]["frame_index"]:
                    sample["sha256"] = inventory[legacy]["sha256"]
            sample_record["shots"][sid] = roles
        journal = {"version": 1, "binding": binding, "original_inventory": inventory,
                   "record": sample_record, "status": "planned"}
    record = journal["record"]
    for receipt in read_receipts(internal / "frame_upgrade_receipts.jsonl"):
        matches = [sample for roles in record["shots"].values() for sample in roles.values()
                   if sample["image_relpath"] == receipt["image_relpath"]
                   and sample["frame_index"] == receipt["frame_index"]]
        if not matches or any(sample["sha256"] not in (None, receipt["sha256"]) for sample in matches):
            raise ValueError("Conflicting extraction receipt")
        for sample in matches:
            sample["sha256"] = receipt["sha256"]
    expected = {}
    for shot in shots:
        sid, t = shot["shot_id"], shot["temporal"]
        spec = physical_samples(sid, t["start_frame"], t["end_frame"])
        if set(record["shots"][sid]) != set(spec):
            raise ValueError("Journal sample roles mismatch")
        for role, sample in record["shots"][sid].items():
            if any(sample.get(k) != v for k, v in spec[role].items()):
                raise ValueError("Journal sampling index/path mismatch")
            expected[Path(sample["image_relpath"]).name] = sample["sha256"]
    legacy = {Path(r["original_path"]).name: r["original_sha256"] for r in record["renames"]}
    for name, item in inventory.items():
        allowed_hash = expected.get(name) if name in expected else legacy.get(name)
        if allowed_hash is None or item["sha256"] != allowed_hash:
            raise ValueError(f"Unexpected, unverified or conflicting image: {name}")
    for rename in record["renames"]:
        old, new = Path(rename["original_path"]).name, Path(rename["new_path"]).name
        if old not in inventory and new not in inventory:
            raise ValueError(f"Original Middle image missing: {mid}/{old}")
    return {"movie_id": mid, "input_root": str(input_root), "video": str(video), "journal": journal,
            "inventory": inventory,
            "missing_frames": sorted({s["frame_index"] for roles in record["shots"].values()
                                       for s in roles.values() if Path(s["image_relpath"]).name not in inventory
                                       and s["image_relpath"] not in {r["new_path"] for r in record["renames"]}})}


def _cleanup_caches(internal: Path, record: dict) -> None:
    """Delete only journal-owned images; unexpected cache files are preserved."""
    physical = {sample["image_relpath"]: sample for roles in record["shots"].values()
                for sample in roles.values()}
    cache_hashes = {f"frame_{sample['frame_index']:08d}.jpg": sample["sha256"]
                    for sample in physical.values()}
    publish_hashes = {Path(path).name: sample["sha256"] for path, sample in physical.items()}
    for name, hashes in (("frame_upgrade_work", cache_hashes), ("frame_publish_work", publish_hashes)):
        directory = safe_path(internal, name)
        if not directory.exists():
            continue
        files = list(directory.iterdir())
        # Check the entire directory before removing even a single cache entry.
        for path in files:
            safe_path(directory, path.name)
            if not path.is_file():
                raise ValueError(f"Unknown cache entry retained: {path}")
            if path.name in hashes:
                if image_info(path)["sha256"] != hashes[path.name]:
                    raise ValueError(f"Changed cache image retained: {path}")
            elif name == "frame_upgrade_work" and path.name == "extraction.json":
                source = load_json(path)["source"]
                identity = record["source_video"]
                if source != {"path": identity["path"], "size": identity["bytes"], "mtime_ns": identity["mtime_ns"]}:
                    raise ValueError("Cache identity mismatch during cleanup")
            elif name == "frame_upgrade_work" and path.name == "frames.jsonl":
                for receipt in read_receipts(path):
                    frame_name = f"frame_{receipt['frame_index']:08d}.jpg"
                    if frame_name not in hashes or receipt.get("status") != "encoding" and receipt["sha256"] != hashes[frame_name]:
                        raise ValueError("Unverified cache receipt retained")
            elif name == "frame_publish_work" and path.name.endswith(".intent.json"):
                image_name = path.name.removesuffix(".intent.json")
                intent = load_json(path)
                if image_name not in hashes or intent["sha256"] != hashes[image_name]:
                    raise ValueError("Unverified publication intent retained")
            else:
                raise ValueError(f"Unknown cache file retained: {path}")
        for path in files:
            path.unlink()
        directory.rmdir()
        sync_dir(internal)


def apply_movie(output_root: Path, movie: dict, plan: dict) -> dict:
    mid = plan["movie_id"]
    movie_dir = safe_path(output_root, mid)
    internal = safe_path(output_root, f".internal/{mid}")
    journal_path = safe_path(internal, "frame_upgrade_journal.json")
    journal = copy.deepcopy(plan["journal"])
    record = journal["record"]
    verify_inputs(output_root, movie)
    video, identity = resolve_video(Path(plan["input_root"]), mid,
        load_json(internal / "video_metadata.json"), load_json(internal / "fingerprint.json"))
    if identity != journal["binding"]["source_video"]:
        raise ValueError("Source video changed after planning")
    # The preflight inventory is rechecked under the shared workspace lock.
    current = {p.name: image_info(p) for p in (movie_dir / "shot_img").iterdir()}
    if {n: i["sha256"] for n, i in current.items()} != {n: i["sha256"] for n, i in plan["inventory"].items()}:
        raise ValueError("Image inventory changed after planning")
    if journal.get("status") == "complete":
        check = validate_movie(movie_dir, frame_profile="three_sample")
        if check.passed:
            _cleanup_caches(internal, record)
            return {"movie_id": mid, "status": "skipped_complete", "protected_inputs_unchanged": True,
                    "sampling_record": load_json(internal / "frame_samples_v1.json"),
                    "sampling_record_sha256": sha256_file(internal / "frame_samples_v1.json")}
    atomic_json(journal_path, journal)  # durable intent before first image publication
    for rename in record["renames"]:
        old = safe_path(movie_dir, rename["original_path"])
        new = safe_path(movie_dir, rename["new_path"])
        if old.exists():
            publish_image(old, new, rename["original_sha256"])
        elif image_info(new)["sha256"] != rename["original_sha256"]:
            raise ValueError("Preserved Middle changed")
    missing = sorted({sample["frame_index"] for roles in record["shots"].values()
                      for sample in roles.values() if not (movie_dir / sample["image_relpath"]).exists()})
    work = safe_path(internal, "frame_upgrade_work")
    if missing:
        progress = ProgressReporter(enabled=True)
        progress.stage(f"{mid}: extract missing frames", len(missing))
        outputs = extract_selected_frames(Path(plan["video"]), missing, work, progress=progress.update)
        mapping = dict(zip(missing, outputs, strict=True))
        for roles in record["shots"].values():
            for sample in roles.values():
                target = safe_path(movie_dir, sample["image_relpath"])
                if target.exists():
                    continue
                source = mapping[sample["frame_index"]]
                digest = image_info(source)["sha256"]
                for alias in roles.values():
                    if alias["frame_index"] == sample["frame_index"]:
                        alias["sha256"] = digest
                append_receipt(internal / "frame_upgrade_receipts.jsonl",
                    {"image_relpath": sample["image_relpath"], "frame_index": sample["frame_index"], "sha256": digest})
                # Fsynced expected hash precedes no-clobber publication without
                # rewriting the entire per-movie sample record for every JPEG.
                publish_image(source, target, digest)
        progress.finish("all missing samples published")
    for roles in record["shots"].values():
        for sample in roles.values():
            if image_info(movie_dir / sample["image_relpath"])["sha256"] != sample["sha256"]:
                raise ValueError("Final sample hash mismatch")
    verify_inputs(output_root, movie)
    source_stat = video.stat()
    if (source_stat.st_size != identity["bytes"] or source_stat.st_mtime_ns != identity["mtime_ns"]
            or sha256_file(video) != identity["sha256"]):
        raise ValueError("Source video changed during extraction")
    # Every original remains recoverable at its new path before legacy files are removed.
    for rename in record["renames"]:
        if image_info(movie_dir / rename["new_path"])["sha256"] != rename["original_sha256"]:
            raise ValueError("Middle preservation verification failed")
        old = movie_dir / rename["original_path"]
        if old.exists():
            if image_info(old)["sha256"] != rename["original_sha256"]:
                raise ValueError("Legacy source changed before cleanup")
            old.unlink()
    sync_dir(movie_dir / "shot_img")
    record["status"] = "complete"
    atomic_json(internal / "frame_samples_v1.json", record)
    check = validate_movie(movie_dir, frame_profile="three_sample")
    if not check.passed:
        raise ValueError("Final validation failed: " + "; ".join(check.errors))
    journal["status"] = "complete"
    atomic_json(journal_path, journal)
    _cleanup_caches(internal, record)
    return {"movie_id": mid, "status": "complete", "extracted_frames": len(missing),
            "renames": record["renames"], "protected_inputs_unchanged": True,
            "sampling_record": record,
            "sampling_record_sha256": sha256_file(internal / "frame_samples_v1.json")}


def frames_upgrade(*, input_root: Path, output_root: Path, frozen_manifest: Path,
                   expected_manifest_sha256: str, expect_movies: int, report_root: Path,
                   movie_id: str | None = None, exclude_movie_id: str | None = None,
                   apply: bool = False, dry_run: bool = False) -> dict:
    if apply and dry_run:
        raise ValueError("--apply and --dry-run are mutually exclusive")
    input_root, output_root = input_root.absolute(), output_root.absolute()
    frozen_manifest, report_root = frozen_manifest.absolute(), report_root.absolute()
    roots = (input_root, output_root, frozen_manifest.parent)
    for root in (*roots, report_root):
        safe_path(root)
    for i, root in enumerate(roots):
        for other in roots[i + 1:]:
            if root.is_relative_to(other) or other.is_relative_to(root):
                raise ValueError("Source, workspace and frozen release roots must be disjoint")
    if any(report_root.is_relative_to(root) or root.is_relative_to(report_root) for root in roots):
        raise ValueError("Report root must be disjoint from all data roots")
    frozen, selected = select_movies(frozen_manifest, expected_manifest_sha256, movie_id=movie_id,
                                     exclude_movie_id=exclude_movie_id, expect_movies=expect_movies)
    fd = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Full selected-batch preflight finishes before reports or dataset mutations.
        plans = [plan_movie(input_root, output_root, movie, frozen["release_id"], expected_manifest_sha256)
                 for movie in selected]
        report = {"release_id": frozen["release_id"], "manifest_sha256": expected_manifest_sha256,
                  "policy": POLICY, "movie_ids": [m["movie_id"] for m in selected],
                  "status": "apply" if apply else "dry_run", "plans": plans}
        report_root.mkdir(parents=True, exist_ok=True)
        # Preserve the initial inventory across repeated dry-runs/apply/recovery.
        initial = report_root / "initial_plan.json"
        binding = {"manifest_sha256": expected_manifest_sha256, "output_root": str(output_root),
                   "input_root": str(input_root), "movie_ids": report["movie_ids"]}
        if initial.exists():
            if load_json(initial)["binding"] != binding:
                raise ValueError("Report directory belongs to a different upgrade selection")
        else:
            atomic_json(initial, {"binding": binding, "report": report})
        atomic_json(report_root / "latest_plan.json", report)
        if apply:
            report["results"] = []
            try:
                for movie, plan in zip(selected, plans, strict=True):
                    report["results"].append(apply_movie(output_root, movie, plan))
                    atomic_json(report_root / "upgrade_report.json", report)
                report["status"] = "complete"
            except Exception as exc:
                report["status"], report["error"] = "interrupted", str(exc)
                atomic_json(report_root / "upgrade_report.json", report)
                raise
        atomic_json(report_root / "upgrade_report.json", report)
        return {"status": report["status"], "movie_ids": report["movie_ids"],
                "movie_count": len(selected), "report_root": str(report_root),
                "results": report.get("results", [])}
    finally:
        os.close(fd)
