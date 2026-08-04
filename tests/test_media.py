import pytest

from oscardp.shots import media
from oscardp.shots.media import (
    ExternalToolError,
    build_timeline,
    parse_fraction,
    probe_video,
)


def test_parse_fraction() -> None:
    assert parse_fraction("24000/1001") == pytest.approx(23.976023976)
    assert parse_fraction("0/0") == 0


def test_cfr_timeline() -> None:
    values = [(index / 25, 0.04) for index in range(100)]
    timeline = build_timeline(values, 25)
    assert timeline.is_vfr is False
    assert timeline.frame_count == 100
    assert timeline.exclusive_end_sec == pytest.approx(4.0)


def test_vfr_timeline_normalizes_pts_and_uses_last_duration() -> None:
    values = [(10.0, 0.04), (10.04, 0.08), (10.12, 0.04), (10.16, 0.06)]
    timeline = build_timeline(values, 25)
    assert timeline.is_vfr is True
    assert timeline.pts_sec[0] == 0
    assert timeline.exclusive_end_sec == pytest.approx(0.22)


def test_probe_video_parses_stream(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(media, "require_media_tools", lambda: None)
    monkeypatch.setattr(
        media,
        "_run_text",
        lambda command: (
            '{"streams":[{"codec_name":"hevc","width":3840,"height":2160,'
            '"avg_frame_rate":"24000/1001","r_frame_rate":"24/1","nb_frames":"120"}],' 
            '"format":{"duration":"5.005"}}'
        ),
    )
    metadata, _ = probe_video(tmp_path / "movie.mkv", "folder/movie.mkv")
    assert metadata.codec_name == "hevc"
    assert metadata.frame_count == 120
    assert metadata.fps == pytest.approx(24000 / 1001)


def test_probe_unreadable_video_fails(tmp_path) -> None:
    video = tmp_path / "broken.mp4"
    video.write_bytes(b"not a video")
    with pytest.raises(ExternalToolError):
        probe_video(video, "broken.mp4")
