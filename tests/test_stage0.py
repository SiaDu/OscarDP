from __future__ import annotations

from oscardp.stage0.media import MediaInfo, classification, target_dimensions, transcode_reasons
from oscardp.stage0.pipeline import validate_output


def info(**changes: object) -> MediaInfo:
    values: dict[str, object] = {
        "container": "matroska", "codec": "hevc", "width": 1920, "height": 1080,
        "fps": 23.976, "bit_depth": 8, "dynamic_range": "SDR", "size_gib": 4.5,
        "duration_sec": 100.0, "pixel_format": "yuv420p", "color_primaries": "bt709",
        "color_transfer": "bt709", "color_space": "bt709", "audio_streams": (),
    }
    values.update(changes)
    return MediaInfo(**values)  # type: ignore[arg-type]


def test_keep_ignores_container_codec_fps_and_audio() -> None:
    source = info(container="matroska,webm", codec="h264", fps=60.0)
    assert transcode_reasons(source, 4.5) == []
    assert classification([]) == "KEEP"


def test_classification_records_all_independent_reasons() -> None:
    source = info(width=3840, height=2160, bit_depth=10, dynamic_range="HDR", size_gib=8.0)
    reasons = transcode_reasons(source, 4.5)
    assert reasons == ["TRANSCODE_SIZE", "TRANSCODE_RESOLUTION", "TRANSCODE_HDR", "TRANSCODE_BIT_DEPTH"]
    assert classification(reasons) == "TRANSCODE_MULTIPLE"
    assert target_dimensions(source) == (1920, 1080)


def test_target_dimensions_never_upscale_and_are_even() -> None:
    assert target_dimensions(info(width=1281, height=719)) == (1280, 718)
    assert target_dimensions(info(width=3840, height=1600)) == (1920, 800)


def test_validation_requires_bt709(monkeypatch, tmp_path) -> None:
    output = tmp_path / "output.mp4"; output.touch()
    monkeypatch.setattr("oscardp.stage0.pipeline.probe", lambda _: info(color_space="unknown"))
    passed, message, _ = validate_output(output, info(), 4.5)
    assert not passed
    assert "BT.709" in message
