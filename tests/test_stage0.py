from __future__ import annotations

import pytest

from oscardp.stage0.media import MediaInfo, _is_hdr, classification, inventory_classification, profile_reasons, target_dimensions, transcode_reasons, video_filter
from oscardp.stage0.pipeline import discover_videos, output_path, validate_output


def info(**changes: object) -> MediaInfo:
    values: dict[str, object] = {
        "container": "matroska", "codec": "hevc", "width": 1920, "height": 1080,
        "fps": 23.976, "bit_depth": 8, "dynamic_range": "SDR", "size_gib": 4.5,
        "duration_sec": 100.0, "pixel_format": "yuv420p", "color_primaries": "bt709",
        "color_transfer": "bt709", "color_space": "bt709", "audio_streams": (),
    }
    values.update(changes)
    return MediaInfo(**values)  # type: ignore[arg-type]


def test_inventory_ignores_fps_but_normalization_schedules_high_fps() -> None:
    source = info(container="matroska,webm", codec="h264", fps=60.0)
    assert inventory_classification(source) == "PASS"
    assert transcode_reasons(source, 4.5) == ["fps exceeds 30; normalize to 24 fps"]
    assert classification(transcode_reasons(source, 4.5)) == "TRANSCODE"


def test_classification_records_all_independent_reasons() -> None:
    source = info(width=3840, height=2160, bit_depth=10, dynamic_range="HDR", size_gib=8.0)
    reasons = transcode_reasons(source, 4.5)
    assert reasons == ["HDR", "filesize exceeds 4.5 GiB", "resolution exceeds 1920x1080", "bit depth is 10-bit"]
    assert classification(reasons) == "HDR_TONEMAP"
    assert inventory_classification(source) == "HDR_TONEMAP"
    assert target_dimensions(source) == (1920, 1080)


def test_inventory_profile_prioritizes_oversize_and_does_not_use_fps_or_bitrate() -> None:
    source = info(size_gib=5.0, fps=120.0)
    assert inventory_classification(source) == "OVERSIZE"
    assert profile_reasons(source) == ["filesize exceeds 4.5 GiB"]
    assert inventory_classification(info(codec="vp9")) == "TRANSCODE"


def test_target_dimensions_never_upscale_and_are_even() -> None:
    assert target_dimensions(info(width=1281, height=719)) == (1280, 718)
    assert target_dimensions(info(width=3840, height=1600)) == (1920, 800)


def test_hdr_resize_first_is_default_and_scales_before_tonemap() -> None:
    source = info(width=3840, height=2160, bit_depth=10, dynamic_range="HDR")
    candidate = video_filter(source)
    reference = video_filter(source, hdr_filter_order="tonemap-first")
    assert candidate.startswith("zscale=w=1920:h=1080:t=linear:npl=100")
    assert candidate.index("zscale=w=1920:h=1080") < candidate.index("tonemap=mobius")
    assert ",scale=" not in candidate
    assert reference.index("tonemap=mobius") < reference.index("scale=1920:1080")
    with pytest.raises(ValueError, match="Unknown HDR filter order"):
        video_filter(source, hdr_filter_order="not-a-mode")


def test_explicit_bt709_is_sdr_even_if_static_hdr_side_data_is_retained() -> None:
    static_hdr = [{"side_data_type": "Mastering display metadata"}]
    assert not _is_hdr({"color_transfer": "bt709", "side_data_list": static_hdr})
    assert _is_hdr({"color_transfer": "", "side_data_list": static_hdr})
    assert _is_hdr({"color_transfer": "smpte2084", "side_data_list": []})
    assert _is_hdr({"color_transfer": "bt709", "side_data_list": [{"side_data_type": "DOVI configuration record"}]})


def test_validation_requires_bt709(monkeypatch, tmp_path) -> None:
    output = tmp_path / "output.mp4"; output.touch()
    monkeypatch.setattr("oscardp.stage0.pipeline.probe", lambda _: info(color_space="unknown"))
    passed, message, _ = validate_output(output, info(), 4.5)
    assert not passed
    assert "BT.709" in message


def test_standardized_output_is_written_beside_source(tmp_path) -> None:
    input_root = tmp_path / "movies"
    source = input_root / "tt1234567" / "Feature.mkv"
    report_root = tmp_path / "stage0_reports"
    source.parent.mkdir(parents=True)
    source.touch()

    result = output_path(source, input_root, report_root)

    assert result == source.parent / "Feature_standardized.mp4"
    assert result != source


def test_discovery_does_not_reprocess_standardized_derivatives(tmp_path) -> None:
    source = tmp_path / "tt1234567" / "Feature.mkv"
    derivative = source.with_name("Feature_standardized.mp4")
    source.parent.mkdir(parents=True)
    source.touch()
    derivative.touch()

    assert discover_videos(tmp_path, None, None, None) == [source]
