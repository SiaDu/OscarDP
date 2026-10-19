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


def _hash_file(path: Path, progress: ProgressReporter, label: str) -> str:
    progress.stage(label, path.stat().st_size, unit="bytes")
    digest = sha256_file(path, progress=progress.update)
    progress.finish()
    return digest


def _inventory(image_dir: Path, progress: ProgressReporter, label: str) -> dict:
    paths = sorted(image_dir.iterdir())
    progress.stage(label, len(paths), unit="files")
    inventory = {}
    for index, path in enumerate(paths, 1):
        inventory[path.name] = {"path": str(path), **image_info(safe_path(image_dir, path.name))}
        progress.update(index)
    progress.finish()
    return inventory


def select_movies(manifest: Path, expected_sha256: str, *, movie_id: str | None,
                  exclude_movie_id: str | None, expect_movies: int,
                  progress: ProgressReporter | None = None) -> tuple[dict, list[dict]]:
    progress = progress or ProgressReporter(enabled=False)
    safe_path(manifest.parent, manifest.name)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or _hash_file(manifest, progress, "验证 Frozen Manifest SHA-256") != expected_sha256:
        raise ValueError("Frozen Manifest SHA-256 mismatch")
    progress.stage("验证清单身份和电影选择", unit="movies")
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
    progress.finish(f"selected {len(selected)} movies")
    for index, movie in enumerate(selected, 1):
        progress.context = f"[清单 {index}/{len(selected)}] {movie['movie_id']}"
        progress.stage("冻结输入校验", len(movie["inputs"]), unit="files")
        if movie.get("release_eligible") is not True or movie.get("activated_for_pipeline") is not False:
            raise ValueError("Frozen movie is not eligible/inactive")
        if set(movie["inputs"]) != {"Stage01", "Stage02", "SRT"}:
            raise ValueError("Expected frozen Stage01, Stage02 and SRT inputs")
        for number, (kind, item) in enumerate(movie["inputs"].items(), 1):
            path = safe_path(manifest.parent, item["path"])
            if sha256_file(path) != item["sha256"]:
                raise ValueError(f"Frozen selected input hash mismatch: {path}")
            if kind != "SRT" and load_json(path).get("movie_id") != movie["movie_id"]:
                raise ValueError(f"Frozen movie identity mismatch: {path}")
            progress.update(number)
        progress.finish()
    progress.context = ""
    return data, selected


def verify_inputs(root: Path, movie: dict, progress: ProgressReporter | None = None) -> dict:
    progress = progress or ProgressReporter(enabled=False)
    progress.stage("输入校验 Stage01 / Stage02 / SRT", len(movie["inputs"]), unit="files")
    mid = movie["movie_id"]
    hashes = {}
    for number, (kind, item) in enumerate(movie["inputs"].items(), 1):
        name = f"{mid}.srt" if kind == "SRT" else f"{kind}_{mid}.json"
        path = safe_path(root, f"{mid}/{name}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Canonical {kind} hash differs from frozen selection: {mid}")
        if kind != "SRT" and load_json(path).get("movie_id") != mid:
            raise ValueError(f"Canonical movie identity mismatch: {path}")
        hashes[kind] = item["sha256"]
        progress.update(number)
    progress.finish()
    return hashes


def resolve_video(input_root: Path, mid: str, metadata: dict, fingerprint: dict,
                  progress: ProgressReporter | None = None) -> tuple[Path, dict]:
    progress = progress or ProgressReporter(enabled=False)
    progress.stage("定位并核对源视频", unit="files")
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
    progress.finish()
    digest = _hash_file(video, progress, "源视频 SHA-256")
    if fingerprint.get("source_sha256", digest) != digest:
        raise ValueError(f"Source video content hash mismatch: {mid}")
    return video, {"path": str(video), "sha256": digest, "bytes": stat.st_size,
                   "mtime_ns": stat.st_mtime_ns,
                   "original_identity_basis": "Stage01 source path/IMDb discovery + original size/mtime + verified decoded timeline"}


def _verify_timeline(video: Path, metadata: dict, shots: list[dict],
                     progress: ProgressReporter | None = None) -> dict[int, float]:
    progress = progress or ProgressReporter(enabled=False)
    progress.stage("探测视频元数据")
    actual, _ = probe_video(video, metadata["source_video_relpath"])
    for key in ("width", "height", "codec_name"):
        if getattr(actual, key) != metadata[key]:
            raise ValueError(f"Source video media identity mismatch: {key}")
    if abs(actual.fps - metadata["fps"]) > max(1e-6, actual.fps * 1e-6):
        raise ValueError("Source video FPS mismatch")
    progress.finish()
    progress.stage("扫描时间线", metadata["frame_count"])
    timeline = scan_timeline(video, actual.fps, progress.update)
    progress.finish()
    if timeline.frame_count != metadata["frame_count"] or timeline.is_vfr != metadata["is_vfr"]:
        raise ValueError("Source video decoded timeline mismatch")
    if abs(timeline.exclusive_end_sec - metadata["duration_sec"]) > 1e-4:
        raise ValueError("Source video duration mismatch")
    pts = {}
    progress.stage("核对镜头时间线", len(shots), unit="shots")
    for index, shot in enumerate(shots, 1):
        temporal = shot["temporal"]
        start, end = temporal["start_frame"], temporal["end_frame"]
        if (abs(timeline.timestamp(start) - temporal["start"]) > 1e-5
                or abs(timeline.exclusive_timestamp(end) - temporal["end"]) > 1e-5):
            raise ValueError("Shot timestamps contradict decoded source timeline")
        for frame in {s["frame_index"] for s in physical_samples(shot["shot_id"], start, end).values()}:
            pts[frame] = timeline.pts_sec[frame]
        progress.update(index)
    progress.finish()
    return pts


def plan_movie(input_root: Path, output_root: Path, movie: dict, release_id: str,
               manifest_hash: str, progress: ProgressReporter | None = None) -> dict:
    progress = progress or ProgressReporter(enabled=False)
    mid = movie["movie_id"]
    protected = verify_inputs(output_root, movie, progress)
    movie_dir = safe_path(output_root, mid)
    internal = safe_path(output_root, f".internal/{mid}")
    progress.stage("读取处理元数据", unit="files")
    metadata = load_json(safe_path(internal, "video_metadata.json"))
    fingerprint = load_json(safe_path(internal, "fingerprint.json"))
    progress.finish()
    video, identity = resolve_video(input_root, mid, metadata, fingerprint, progress)
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
    inventory = _inventory(image_dir, progress, "图片清单与 JPEG 完整性检查")
    if not journal:
        if safe_path(internal, "frame_samples_v1.json").exists() or safe_path(internal, "frame_upgrade_receipts.jsonl").exists():
            raise ValueError("Existing sample metadata has no upgrade journal; refusing to overwrite it")
        progress.stage("Legacy 输出验证", len(inventory), unit="files")
        historical = validate_movie(movie_dir, frame_count=metadata["frame_count"], frame_profile="legacy",
                                    image_progress=progress.update)
        progress.finish()
        if not historical.passed:
            raise ValueError("Legacy validation failed: " + "; ".join(historical.errors))
        pts = _verify_timeline(video, metadata, shots, progress)
        sample_record = {"policy": POLICY, "movie_id": mid, "status": "upgrading",
                         "stage01_sha256": protected["Stage01"], "source_video": identity,
                         "renames": [], "shots": {}}
        progress.stage("生成采样计划", len(shots), unit="shots")
        for index, shot in enumerate(shots, 1):
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
            progress.update(index)
        progress.finish()
        journal = {"version": 1, "binding": binding, "original_inventory": inventory,
                   "record": sample_record, "status": "planned"}
    record = journal["record"]
    receipts = read_receipts(internal / "frame_upgrade_receipts.jsonl")
    progress.stage("核对采样日志与计划", len(receipts) + len(shots), unit="records")
    for index, receipt in enumerate(receipts, 1):
        matches = [sample for roles in record["shots"].values() for sample in roles.values()
                   if sample["image_relpath"] == receipt["image_relpath"]
                   and sample["frame_index"] == receipt["frame_index"]]
        if not matches or any(sample["sha256"] not in (None, receipt["sha256"]) for sample in matches):
            raise ValueError("Conflicting extraction receipt")
        for sample in matches:
            sample["sha256"] = receipt["sha256"]
        progress.update(index)
    expected = {}
    for index, shot in enumerate(shots, 1):
        sid, t = shot["shot_id"], shot["temporal"]
        spec = physical_samples(sid, t["start_frame"], t["end_frame"])
        if set(record["shots"][sid]) != set(spec):
            raise ValueError("Journal sample roles mismatch")
        for role, sample in record["shots"][sid].items():
            if any(sample.get(k) != v for k, v in spec[role].items()):
                raise ValueError("Journal sampling index/path mismatch")
            expected[Path(sample["image_relpath"]).name] = sample["sha256"]
        progress.update(len(receipts) + index)
    progress.finish()
    progress.stage("核对图片冲突及 Middle 映射", len(inventory) + len(record["renames"]), unit="files")
    legacy = {Path(r["original_path"]).name: r["original_sha256"] for r in record["renames"]}
    for index, (name, item) in enumerate(inventory.items(), 1):
        allowed_hash = expected.get(name) if name in expected else legacy.get(name)
        if allowed_hash is None or item["sha256"] != allowed_hash:
            raise ValueError(f"Unexpected, unverified or conflicting image: {name}")
        progress.update(index)
    for index, rename in enumerate(record["renames"], 1):
        old, new = Path(rename["original_path"]).name, Path(rename["new_path"]).name
        if old not in inventory and new not in inventory:
            raise ValueError(f"Original Middle image missing: {mid}/{old}")
        progress.update(len(inventory) + index)
    progress.finish()
    return {"movie_id": mid, "input_root": str(input_root), "video": str(video), "journal": journal,
            "inventory": inventory,
            "missing_frames": sorted({s["frame_index"] for roles in record["shots"].values()
                                       for s in roles.values() if Path(s["image_relpath"]).name not in inventory
                                       and s["image_relpath"] not in {r["new_path"] for r in record["renames"]}})}


def _cleanup_caches(internal: Path, record: dict, progress: ProgressReporter | None = None) -> None:
    """Delete only journal-owned images; unexpected cache files are preserved."""
    progress = progress or ProgressReporter(enabled=False)
    progress.stage("检查缓存目录", unit="files")
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
        progress.stage(f"校验缓存 {name}", len(files), unit="files")
        for index, path in enumerate(files, 1):
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
            progress.update(index)
        progress.finish()
        progress.stage(f"清理缓存 {name}", len(files), unit="files")
        for index, path in enumerate(files, 1):
            path.unlink()
            progress.update(index)
        directory.rmdir()
        sync_dir(internal)
        progress.finish()


def apply_movie(output_root: Path, movie: dict, plan: dict,
                progress: ProgressReporter | None = None) -> dict:
    progress = progress or ProgressReporter(enabled=False)
    mid = plan["movie_id"]
    movie_dir = safe_path(output_root, mid)
    internal = safe_path(output_root, f".internal/{mid}")
    journal_path = safe_path(internal, "frame_upgrade_journal.json")
    journal = copy.deepcopy(plan["journal"])
    record = journal["record"]
    verify_inputs(output_root, movie, progress)
    video, identity = resolve_video(Path(plan["input_root"]), mid,
        load_json(internal / "video_metadata.json"), load_json(internal / "fingerprint.json"), progress)
    if identity != journal["binding"]["source_video"]:
        raise ValueError("Source video changed after planning")
    # The preflight inventory is rechecked under the shared workspace lock.
    current = _inventory(movie_dir / "shot_img", progress, "复核应用前图片清单")
    if {n: i["sha256"] for n, i in current.items()} != {n: i["sha256"] for n, i in plan["inventory"].items()}:
        raise ValueError("Image inventory changed after planning")
    if journal.get("status") == "complete":
        progress.stage("验证已完成输出", len(current), unit="files")
        check = validate_movie(movie_dir, frame_profile="three_sample", image_progress=progress.update)
        progress.finish()
        if check.passed:
            _cleanup_caches(internal, record, progress)
            progress.stage("已验证，跳过", unit="movies")
            progress.finish()
            return {"movie_id": mid, "status": "skipped_complete", "protected_inputs_unchanged": True,
                    "sampling_record": load_json(internal / "frame_samples_v1.json"),
                    "sampling_record_sha256": sha256_file(internal / "frame_samples_v1.json")}
    progress.stage("保存迁移日志", unit="files")
    atomic_json(journal_path, journal)  # durable intent before first image publication
    progress.finish()
    progress.stage("Middle 迁移", len(record["renames"]), unit="files")
    for index, rename in enumerate(record["renames"], 1):
        old = safe_path(movie_dir, rename["original_path"])
        new = safe_path(movie_dir, rename["new_path"])
        if old.exists():
            publish_image(old, new, rename["original_sha256"])
        elif image_info(new)["sha256"] != rename["original_sha256"]:
            raise ValueError("Preserved Middle changed")
        progress.update(index)
    progress.finish()
    missing = sorted({sample["frame_index"] for roles in record["shots"].values()
                      for sample in roles.values() if not (movie_dir / sample["image_relpath"]).exists()})
    work = safe_path(internal, "frame_upgrade_work")
    if missing:
        progress.stage("抽帧：核对恢复缓存", len(missing), unit="samples")
        samples_done = 0
        decoded = False

        def sample_progress(count: int) -> None:
            nonlocal samples_done
            samples_done = count
            if not decoded:
                progress.update(count)

        def decode_progress(current: int, total: int) -> None:
            nonlocal decoded
            if not decoded:
                progress.finish(f"cached {samples_done}/{len(missing)} samples")
                progress.stage("抽帧：顺序解码", total)
                decoded = True
            progress.update(current, total, detail=f"samples {samples_done:,}/{len(missing):,}")

        outputs = extract_selected_frames(Path(plan["video"]), missing, work,
                                          progress=sample_progress, decode_progress=decode_progress)
        progress.finish(f"samples {samples_done:,}/{len(missing):,}")
        progress.stage("发布 Early / Late 图片", len(missing), unit="files")
        published = 0
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
                published += 1
                progress.update(published)
        progress.finish("all missing samples published")
    sample_count = sum(len(roles) for roles in record["shots"].values())
    progress.stage("最终样本哈希及完整性校验", sample_count, unit="samples")
    verified = 0
    for roles in record["shots"].values():
        for sample in roles.values():
            if image_info(movie_dir / sample["image_relpath"])["sha256"] != sample["sha256"]:
                raise ValueError("Final sample hash mismatch")
            verified += 1
            progress.update(verified)
    progress.finish()
    verify_inputs(output_root, movie, progress)
    progress.stage("最终源视频身份校验", unit="files")
    source_stat = video.stat()
    if source_stat.st_size != identity["bytes"] or source_stat.st_mtime_ns != identity["mtime_ns"]:
        raise ValueError("Source video changed during extraction")
    progress.finish()
    if _hash_file(video, progress, "最终源视频 SHA-256") != identity["sha256"]:
        raise ValueError("Source video changed during extraction")
    # Every original remains recoverable at its new path before legacy files are removed.
    progress.stage("验证 Middle 并清理旧文件名", len(record["renames"]), unit="files")
    for index, rename in enumerate(record["renames"], 1):
        if image_info(movie_dir / rename["new_path"])["sha256"] != rename["original_sha256"]:
            raise ValueError("Middle preservation verification failed")
        old = movie_dir / rename["original_path"]
        if old.exists():
            if image_info(old)["sha256"] != rename["original_sha256"]:
                raise ValueError("Legacy source changed before cleanup")
            old.unlink()
        progress.update(index)
    sync_dir(movie_dir / "shot_img")
    progress.finish()
    record["status"] = "complete"
    atomic_json(internal / "frame_samples_v1.json", record)
    progress.stage("最终 Three-sample 输出验证", len({s["image_relpath"] for roles in record["shots"].values() for s in roles.values()}), unit="files")
    check = validate_movie(movie_dir, frame_profile="three_sample", image_progress=progress.update)
    progress.finish()
    if not check.passed:
        raise ValueError("Final validation failed: " + "; ".join(check.errors))
    journal["status"] = "complete"
    atomic_json(journal_path, journal)
    _cleanup_caches(internal, record, progress)
    progress.stage("电影升级完成", unit="movies")
    progress.finish()
    return {"movie_id": mid, "status": "complete", "extracted_frames": len(missing),
            "renames": record["renames"], "protected_inputs_unchanged": True,
            "sampling_record": record,
            "sampling_record_sha256": sha256_file(internal / "frame_samples_v1.json")}


def _frames_upgrade(*, input_root: Path, output_root: Path, frozen_manifest: Path,
                   expected_manifest_sha256: str, expect_movies: int, report_root: Path,
                   movie_id: str | None = None, exclude_movie_id: str | None = None,
                   apply: bool = False, dry_run: bool = False,
                   progress: ProgressReporter) -> dict:
    progress.stage("检查命令参数和路径", unit="files")
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
                                     exclude_movie_id=exclude_movie_id, expect_movies=expect_movies, progress=progress)
    progress.stage("获取工作区锁", unit="files")
    fd = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Full selected-batch preflight finishes before reports or dataset mutations.
        progress.finish()
        plans = []
        for index, movie in enumerate(selected, 1):
            progress.context = f"[预检 {index}/{len(selected)}] {movie['movie_id']}"
            plans.append(plan_movie(input_root, output_root, movie, frozen["release_id"],
                                    expected_manifest_sha256, progress))
        progress.context = ""
        progress.stage("写入预检报告", unit="files")
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
        progress.finish()
        if apply:
            report["results"] = []
            try:
                for index, (movie, plan) in enumerate(zip(selected, plans, strict=True), 1):
                    progress.context = f"[应用 {index}/{len(selected)}] {movie['movie_id']}"
                    report["results"].append(apply_movie(output_root, movie, plan, progress))
                    progress.stage("保存电影升级报告", unit="files")
                    atomic_json(report_root / "upgrade_report.json", report)
                    progress.finish()
                report["status"] = "complete"
            except Exception as exc:
                report["status"], report["error"] = "interrupted", str(exc)
                atomic_json(report_root / "upgrade_report.json", report)
                raise
        progress.context = ""
        progress.stage("保存最终报告", unit="files")
        atomic_json(report_root / "upgrade_report.json", report)
        progress.finish()
        return {"status": report["status"], "movie_ids": report["movie_ids"],
                "movie_count": len(selected), "report_root": str(report_root),
                "results": report.get("results", [])}
    finally:
        os.close(fd)


def frames_upgrade(*, input_root: Path, output_root: Path, frozen_manifest: Path,
                   expected_manifest_sha256: str, expect_movies: int, report_root: Path,
                   movie_id: str | None = None, exclude_movie_id: str | None = None,
                   apply: bool = False, dry_run: bool = False, no_progress: bool = False) -> dict:
    progress = ProgressReporter(enabled=not no_progress)
    try:
        result = _frames_upgrade(input_root=input_root, output_root=output_root,
            frozen_manifest=frozen_manifest, expected_manifest_sha256=expected_manifest_sha256,
            expect_movies=expect_movies, report_root=report_root, movie_id=movie_id,
            exclude_movie_id=exclude_movie_id, apply=apply, dry_run=dry_run, progress=progress)
    except BaseException as exc:
        progress.fail(str(exc) or "操作已取消")
        raise
    progress.stage("升级完成" if apply else "Dry-run 完成", unit="movies")
    progress.finish(f"{result['movie_count']} movies; reports: {result['report_root']}")
    return result
