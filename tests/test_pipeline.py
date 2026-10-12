from pathlib import Path

import json

import pytest
from PIL import Image

from oscardp.shots import pipeline
from oscardp.shots.pipeline import ProcessOptions, build_shot_records, process_one
from oscardp.shots.schema import FrameTimeline, MovieRef, VideoMetadata
from oscardp.shots.transnet import Boundary
from oscardp.shots.validation import validate_output_root


def test_build_shots_is_contiguous_and_uses_midpoint(tmp_path: Path) -> None:
    movie = MovieRef("tt1234567", tmp_path / "movie.mkv", "movie.mkv")
    timeline = FrameTimeline(
        pts_sec=[index / 24 for index in range(120)],
        frame_duration_sec=[1 / 24] * 120,
        is_vfr=False,
        nominal_fps=24,
        exclusive_end_sec=5.0,
    )
    metadata = VideoMetadata("movie.mkv", 5.0, 1920, 1080, "h264", 24, 120, False, "frame_index_cfr")
    shots = build_shot_records(movie, metadata, timeline, [Boundary(40, 0.8), Boundary(90, 0.9)], 0.5)
    assert [(shot.start_frame, shot.end_frame) for shot in shots] == [(0, 40), (40, 90), (90, 120)]
    assert [shot.keyframe_frame for shot in shots] == [19, 64, 104]
    assert shots[0].boundary_before_confidence is None
    assert shots[0].boundary_after_confidence == 0.8
    assert shots[-1].boundary_after_confidence is None


def test_process_one_publishes_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    video = input_root / "tt1234567_test.mp4"
    video.write_bytes(b"video")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    timeline = FrameTimeline(
        pts_sec=[index / 10 for index in range(10)],
        frame_duration_sec=[0.1] * 10,
        is_vfr=False,
        nominal_fps=10,
        exclusive_end_sec=1.0,
    )
    metadata = VideoMetadata(
        "tt1234567_test.mp4", 1.0, 64, 32, "h264", 10, 10, False, "frame_index_cfr"
    )

    def fake_inference(*args, **kwargs):
        return metadata, timeline, [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]

    def fake_extract(
        video_path: Path,
        indices: list[int],
        target_dir: Path,
        progress=None,
    ) -> list[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for ordinal, _ in enumerate(sorted(set(indices)), 1):
            path = target_dir / f"frame_{ordinal:08d}.jpg"
            Image.new("RGB", (64, 32), color=(ordinal, 0, 0)).save(path)
            outputs.append(path)
        return outputs

    monkeypatch.setattr(pipeline, "_load_or_run_inference", fake_inference)
    monkeypatch.setattr(pipeline, "extract_selected_frames", fake_extract)
    options = ProcessOptions(input_root=input_root, output_root=output_root, weights=weights)
    result = process_one(video, options)
    assert result["status"] == "completed"
    movie_dir = output_root / "tt1234567"
    stage = json.loads((movie_dir / "Stage01_tt1234567.json").read_text())
    assert set(stage) == {"movie_id", "movie_name", "shots"}
    assert stage["movie_name"] is None
    assert len(stage["shots"]) == 2
    assert set(stage["shots"][0]) == {"shot_id", "temporal"}
    assert len(list((movie_dir / "shot_img").glob("*.jpg"))) == 2
    assert sorted(path.name for path in (movie_dir / "shot_img").glob("*.jpg")) == [
        "shot_000001_00-00-00-200.jpg", "shot_000002_00-00-00-700.jpg"
    ]
    assert sorted(path.name for path in movie_dir.iterdir()) == ["Stage01_tt1234567.json", "shot_img"]
    assert not (movie_dir / ".resume").exists()
    resumed = process_one(video, options)
    assert resumed["resumed"] is True
    assert validate_output_root(output_root, "tt1234567")["tt1234567"].passed
    image = next((movie_dir / "shot_img").glob("*.jpg"))
    image.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        process_one(video, options)
    assert image.read_bytes() == b"corrupt"
