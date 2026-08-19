from pathlib import Path

import pytest

from oscardp.artifact_paths import parse_path_maps, resolve_artifact_path


def test_path_maps_use_complete_longest_prefix_and_preserve_unmatched_path(tmp_path: Path) -> None:
    root = tmp_path / "data"
    maps = parse_path_maps([f"/mnt={tmp_path / 'broad'}", f"/mnt/i={root}"])

    assert resolve_artifact_path("/mnt/i/movie/file.json", maps) == root / "movie/file.json"
    assert resolve_artifact_path("/mnt/inside/file.json", maps) == tmp_path / "broad/inside/file.json"
    assert resolve_artifact_path("/opt/file.json", maps) == Path("/opt/file.json")


@pytest.mark.parametrize(
    "specification",
    ["", "old=new=again", "= /tmp", "/old=", "relative=/tmp", "/old=relative"],
)
def test_path_maps_reject_invalid_specs(specification: str) -> None:
    with pytest.raises(ValueError):
        parse_path_maps([specification])


def test_path_maps_reject_duplicate_sources() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_path_maps(["/mnt/i=/tmp/one", "/mnt/i=/tmp/two"])
