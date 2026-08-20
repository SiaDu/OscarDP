from __future__ import annotations

from pathlib import Path

from oscardp.stage0.media import MediaInfo
from oscardp.stage0.preflight import PreflightOptions, run


def info(duration: float) -> MediaInfo:
    return MediaInfo("matroska", "hevc", 1920, 1080, 24.0, 8, "SDR", 1.0, duration, "yuv420p", "bt709", "bt709", "bt709", ())


def test_preflight_plans_without_mutating_sources(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "raw"; movie = root / "tt12300742_Bugonia"; movie.mkdir(parents=True)
    primary = movie / "release.mkv"; primary.touch()
    sample = movie / "sample.mp4"; sample.touch()
    (movie / "English.srt").write_text("", encoding="utf-8")
    (movie / "poster.jpg").touch(); (movie / "readme.nfo").touch(); (movie / "keep.zip").touch()
    monkeypatch.setattr("oscardp.stage0.preflight._video", lambda path: (info(7200.0 if path == primary else 30.0), None))
    rows, plan, summary = run(PreflightOptions(root, tmp_path / "reports"))
    row = rows[0]
    assert row["primary_video_original_name"] == "release.mkv"
    assert row["rename_status"] == "PLANNED"
    assert primary.exists() and sample.exists()
    assert {item["category"] for item in plan if item["action"] == "QUARANTINE"} == {"AUXILIARY_VIDEO", "IMAGE_ASSET", "RELEASE_METADATA"}
    assert not any(item["category"] == "ARCHIVE" for item in plan)
    assert summary["planned_rename_count"] == 2


def test_similar_full_movies_are_manual_review(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "raw"; movie = root / "tt12300742_Bugonia"; movie.mkdir(parents=True)
    one = movie / "one.mkv"; two = movie / "two.mkv"; one.touch(); two.touch()
    monkeypatch.setattr("oscardp.stage0.preflight._video", lambda _path: (info(7200.0), None))
    rows, plan, _summary = run(PreflightOptions(root, tmp_path / "reports"))
    assert rows[0]["primary_video_path"] == ""
    assert rows[0]["manual_review_required"] is True
    assert "MULTIPLE_FULL_MOVIE_CANDIDATES" in str(rows[0]["manual_review_reasons"])
    assert not plan
