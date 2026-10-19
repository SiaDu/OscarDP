import json
from pathlib import Path

from PIL import Image

from oscardp.shots.validation import validate_movie


def write_fixture(root: Path) -> None:
    root = root / "tt1234567"
    (root / "shot_img").mkdir(parents=True)
    data = {"movie_id": "tt1234567", "movie_name": None, "shots": [
        {"shot_id": "shot_000001", "temporal": {"start": 0.0, "end": 1.0, "start_frame": 0, "end_frame": 10}}
    ]}
    (root / "Stage01_tt1234567.json").write_text(json.dumps(data), encoding="utf-8")
    Image.new("RGB", (10, 10)).save(root / "shot_img" / "shot_000001_00-00-00-400.jpg")


def test_valid_movie(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    assert validate_movie(tmp_path / "tt1234567", frame_count=10).passed


def test_missing_image_and_final_frame_fail(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    root = tmp_path / "tt1234567"
    (root / "shot_img" / "shot_000001_00-00-00-400.jpg").unlink()
    result = validate_movie(root, frame_count=11)
    assert not result.passed
    assert result.missing_keyframes == 1
    assert any("final shot" in error for error in result.errors)


def test_corrupt_image_and_extra_schema_fail(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    root = tmp_path / "tt1234567"
    (root / "shot_img" / "shot_000001_00-00-00-400.jpg").write_bytes(b"broken")
    assert not validate_movie(root).passed
    data = json.loads((root / "Stage01_tt1234567.json").read_text())
    data["shots"][0]["extra"] = 1
    (root / "Stage01_tt1234567.json").write_text(json.dumps(data))
    assert not validate_movie(root).passed


def test_stage01_accepts_only_named_later_stage_files(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    root = tmp_path / "tt1234567"
    for filename in ("Stage03_tt1234567.json", "Stage04_tt1234567.json"):
        (root / filename).write_text("{}", encoding="utf-8")
    assert validate_movie(root, frame_count=10).passed
    (root / "Stage03_tt7654321.json").write_text("{}", encoding="utf-8")
    assert not validate_movie(root, frame_count=10).passed
