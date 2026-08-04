import json
from pathlib import Path

from oscardp.shots.validation import validate_movie


def write_fixture(root: Path) -> None:
    (root / "keyframes").mkdir(parents=True)
    metadata = {
        "source_video_relpath": "movie.mp4", "duration_sec": 1.0, "width": 10,
        "height": 10, "codec_name": "h264", "fps": 10.0, "frame_count": 10,
        "is_vfr": False, "timestamp_source": "frame_index_cfr",
    }
    (root / "video_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    shot = {
        "movie_key": "movie", "shot_id": "shot_000001", "source_video_relpath": "movie.mp4",
        "start_frame": 0, "end_frame": 10, "frame_count": 10,
        "start_sec": 0.0, "end_sec": 1.0, "keyframe_frame": 4,
        "keyframe_relpath": "keyframes/shot_000001.jpg",
    }
    (root / "shots.jsonl").write_text(json.dumps(shot) + "\n", encoding="utf-8")
    (root / "keyframes" / "shot_000001.jpg").touch()


def test_valid_movie(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    assert validate_movie(tmp_path).passed


def test_missing_keyframe_and_final_frame_fail(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    (tmp_path / "keyframes" / "shot_000001.jpg").unlink()
    shot = json.loads((tmp_path / "shots.jsonl").read_text())
    shot["end_frame"] = 9
    (tmp_path / "shots.jsonl").write_text(json.dumps(shot) + "\n")
    result = validate_movie(tmp_path)
    assert not result.passed
    assert result.missing_keyframes == 1
    assert any("final shot" in error for error in result.errors)
