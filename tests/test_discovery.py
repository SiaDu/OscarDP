from pathlib import Path

import pytest

from oscardp.shots.discovery import derive_movie_key, discover_movies


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_discovery_extensions_hidden_partial_and_imdb(tmp_path: Path) -> None:
    for suffix in ("mp4", "mkv", "mov", "m4v", "avi", "webm"):
        touch(tmp_path / f"movie_{suffix}.{suffix}")
    touch(tmp_path / ".hidden" / "secret.mp4")
    touch(tmp_path / "download.mp4.part")
    touch(tmp_path / "subtitle.srt")
    touch(tmp_path / "tt32536315_Title" / "feature.MKV")
    movies = discover_movies(tmp_path)
    assert len(movies) == 7
    assert any(movie.movie_key == "tt32536315" for movie in movies)
    assert all(".hidden" not in movie.source_video_relpath for movie in movies)


def test_filename_imdb_precedes_parent_and_fallback_is_relative(tmp_path: Path) -> None:
    path = tmp_path / "tt1111111_parent" / "Title.tt2222222.mkv"
    touch(path)
    assert derive_movie_key(path, tmp_path) == "tt2222222"
    fallback = tmp_path / "Some Folder" / "Mövie Name.mov"
    touch(fallback)
    assert derive_movie_key(fallback, tmp_path) == "some_folder__m_vie_name"


def test_duplicate_keys_fail(tmp_path: Path) -> None:
    touch(tmp_path / "a" / "tt1234567.mp4")
    touch(tmp_path / "b" / "tt1234567.mkv")
    with pytest.raises(ValueError, match="Duplicate movie_key"):
        discover_movies(tmp_path)


def test_outside_root_fails(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.mp4"
    outside.touch(exist_ok=True)
    with pytest.raises(ValueError, match="outside input root"):
        derive_movie_key(outside, tmp_path)


def test_non_ascii_only_name_has_stable_fallback(tmp_path: Path) -> None:
    path = tmp_path / "电影.mkv"
    touch(path)
    first = derive_movie_key(path, tmp_path)
    assert first.startswith("movie_")
    assert first == derive_movie_key(path, tmp_path)
