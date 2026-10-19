"""Progress must observe the existing upgrade, without altering its reports or IO."""
import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from oscardp.shots import frames_upgrade as upgrade, media
from oscardp.shots.__main__ import main
from oscardp.shots.frame_samples import atomic_json
from oscardp.shots.transnet import sha256_file
from test_frame_samples import fixture, real_video, tree_hashes


def cli_args(options, *flags):
    args = ["frames-upgrade"]
    for key, value in options.items():
        args.extend(["--" + key.replace("_", "-"), str(value)])
    return args + list(flags)


def test_single_movie_stages_and_json_only_stdout(fixture, capsys, monkeypatch):
    def scan(path, fps, progress=None):
        if progress:
            for n in (1, 10, 20):
                progress(n)
        return fixture["timeline"]
    monkeypatch.setattr(upgrade, "scan_timeline", scan)
    assert main(cli_args(fixture["options"], "--dry-run")) == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["status"] == "dry_run"
    labels = ["验证 Frozen Manifest SHA-256", "冻结输入校验", "[预检 1/1] tt1234567 输入校验",
              "源视频 SHA-256", "图片清单与 JPEG 完整性检查", "Legacy 输出验证", "扫描时间线",
              "生成采样计划", "核对采样日志与计划", "写入预检报告", "Dry-run 完成"]
    positions = [err.index(label) for label in labels]
    assert positions == sorted(positions)
    assert "20/20 frames" in err
    assert "files" in err and "bytes" in err
    assert "\r" not in err and "\x1b" not in err


def test_no_progress_preserves_report_and_dataset(fixture, capsys):
    options = fixture["options"]
    assert main(cli_args(options, "--dry-run")) == 0
    before = tree_hashes(options["output_root"])
    reports = tree_hashes(options["report_root"])
    capsys.readouterr()
    assert main(cli_args(options, "--dry-run", "--no-progress")) == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["status"] == "dry_run" and not err
    assert tree_hashes(options["output_root"]) == before
    assert tree_hashes(options["report_root"]) == reports
    assert main(cli_args(options, "--apply", "--no-progress")) == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["status"] == "complete" and not err


def test_failure_names_movie_and_stage(fixture, capsys):
    fixture["video"].write_bytes(b"changed video")
    assert main(cli_args(fixture["options"], "--dry-run")) == 2
    out, err = capsys.readouterr()
    assert not out
    assert "[预检 1/1] tt1234567 定位并核对源视频 FAILED" in err
    assert "Source video fingerprint mismatch" in err
    assert not fixture["options"]["report_root"].exists()


def test_completed_movie_is_verified_and_skipped(fixture, capsys):
    upgrade.frames_upgrade(**fixture["options"], apply=True)
    capsys.readouterr()
    before = tree_hashes(fixture["options"]["output_root"])
    assert main(cli_args(fixture["options"], "--apply")) == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["results"][0]["status"] == "skipped_complete"
    assert "[应用 1/1] tt1234567 验证已完成输出" in err
    assert "[应用 1/1] tt1234567 已验证，跳过" in err
    assert "抽帧：顺序解码" not in err
    assert tree_hashes(fixture["options"]["output_root"]) == before


def test_17_movie_preflight_order_and_selection(fixture, capsys):
    options = dict(fixture["options"])
    manifest = options["frozen_manifest"]
    release = manifest.parent
    workspace = options["output_root"]
    source = options["input_root"]
    ids = ["tt12300742"] + [f"tt{i:07d}" for i in range(17)]
    movies = []
    for mid in ids:
        movie = workspace / mid
        shutil.copytree(fixture["movie"], movie)
        inputs = {}
        for kind in ("Stage01", "Stage02", "SRT"):
            old = movie / (f"tt1234567.srt" if kind == "SRT" else f"{kind}_tt1234567.json")
            name = f"{mid}.srt" if kind == "SRT" else f"{kind}_{mid}.json"
            if kind != "SRT":
                payload = json.loads(old.read_text())
                payload["movie_id"] = mid
                atomic_json(movie / name, payload)
            else:
                shutil.copy2(old, movie / name)
            old.unlink()
            frozen = release / mid / name
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(movie / name, frozen)
            inputs[kind] = {"path": f"{mid}/{name}", "sha256": sha256_file(frozen)}
        video = source / f"{mid}.mkv"
        shutil.copy2(fixture["video"], video)
        internal = workspace / ".internal" / mid
        internal.mkdir()
        metadata = dataclasses.asdict(fixture["metadata"])
        metadata["source_video_relpath"] = video.name
        atomic_json(internal / "video_metadata.json", metadata)
        atomic_json(internal / "fingerprint.json", {"source_path": str(video), "source_size": video.stat().st_size,
                    "source_mtime_ns": video.stat().st_mtime_ns, "threshold": .5, "model_sha256": "unchanged"})
        movies.append({"movie_id": mid, "release_eligible": True, "activated_for_pipeline": False, "inputs": inputs})
    atomic_json(manifest, {"release_id": release.name, "dataset_status": "FROZEN", "activated_for_pipeline": False,
                          "movie_count": 18, "movies": movies})
    options.pop("movie_id")
    options.update(expected_manifest_sha256=sha256_file(manifest), expect_movies=17, exclude_movie_id="tt12300742")
    before = tree_hashes(workspace)
    assert main(cli_args(options, "--dry-run")) == 0
    out, err = capsys.readouterr()
    result = json.loads(out)
    assert result["movie_count"] == 17 and result["movie_ids"] == ids[1:]
    positions = [err.index(f"[预检 {i}/17] {mid} 输入校验") for i, mid in enumerate(ids[1:], 1)]
    assert positions == sorted(positions)
    assert "tt12300742" not in err
    assert tree_hashes(workspace) == before


@pytest.mark.parametrize("vfr", [False, True])
def test_real_decode_progress_between_sparse_samples_and_cache(tmp_path, vfr):
    video = real_video(tmp_path, vfr)
    samples, decode = [], []
    target = tmp_path / "cache"
    outputs = media.extract_selected_frames(video, [0, 19], target, progress=samples.append,
                                            decode_progress=lambda n, total: decode.append((n, total, samples[-1])))
    assert decode[0] == (0, 20, 0)
    assert decode[-1] == (20, 20, 2)
    assert [n for n, _, _ in decode] == list(range(21))
    assert all(total == 20 for _, total, _ in decode)
    assert (10, 20, 1) in decode  # Decoder advances even when no new sample is selected.
    cached = []
    assert media.extract_selected_frames(video, [0, 19], target, progress=cached.append,
        decode_progress=lambda *a: pytest.fail("complete cache must not decode")) == outputs
    assert cached[-1] == 2


def test_real_upgrade_progress_from_cli(fixture, tmp_path, monkeypatch, capsys):
    synthetic = real_video(tmp_path)
    shutil.copy2(synthetic, fixture["video"])
    video = fixture["video"]
    metadata, _ = media.probe_video(video, video.name)
    metadata.frame_count = 20
    atomic_json(fixture["internal"] / "video_metadata.json", dataclasses.asdict(metadata))
    stat = video.stat()
    atomic_json(fixture["internal"] / "fingerprint.json", {"source_path": str(video), "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns, "threshold": .5, "model_sha256": "unchanged"})
    monkeypatch.setattr(upgrade, "probe_video", media.probe_video)
    monkeypatch.setattr(upgrade, "scan_timeline", media.scan_timeline)
    monkeypatch.setattr(upgrade, "extract_selected_frames", media.extract_selected_frames)
    assert main(cli_args(fixture["options"], "--apply")) == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["status"] == "complete"
    assert "[应用 1/1] tt1234567 抽帧：顺序解码" in err
    assert "samples 5/5" in err
    assert "18/18 frames" in err  # The last requested missing frame is index 17.
    for stage in ("Middle 迁移", "发布 Early / Late 图片", "最终样本哈希", "最终源视频 SHA-256",
                  "验证 Middle 并清理旧文件名", "最终 Three-sample 输出验证", "清理缓存", "升级完成"):
        assert stage in err
    assert "FAILED" not in err
