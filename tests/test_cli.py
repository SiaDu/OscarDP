import json
from pathlib import Path

import pytest

from oscardp.shots.__main__ import DEFAULT_INPUT_ROOT, DEFAULT_OUTPUT_ROOT, build_parser, main


def test_cli_defaults_and_options() -> None:
    parser = build_parser()
    assert parser.parse_args(["discover"]).input_root == DEFAULT_INPUT_ROOT
    args = parser.parse_args(["process-one", "--video", "/tmp/tt1234567.mp4", "--weights", "/tmp/model.pth", "--device", "cuda"])
    assert args.output_root == DEFAULT_OUTPUT_ROOT
    assert args.device == "cuda"
    assert not args.overwrite
    assert parser.parse_args(["validate", "--movie-id", "tt1234567"]).movie_id == "tt1234567"


def test_batch_requires_titles() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(['process', '--weights', '/tmp/model.pth'])


def test_batch_dry_run_uses_titles_and_skips_unlisted(tmp_path: Path, capsys) -> None:
    root = tmp_path / 'movies'
    (root / 'tt1234567_Movie').mkdir(parents=True)
    (root / 'tt1234567_Movie' / 'film.rmvb').touch()
    (root / 'tt7654321_Extra').mkdir()
    (root / 'tt7654321_Extra' / 'extra.mp4').touch()
    titles = tmp_path / 'titles.json'
    titles.write_text(json.dumps({'tt1234567': 'Movie'}))
    result = main(['process', '--input-root', str(root), '--output-root', str(tmp_path / 'out'),
                   '--weights', str(tmp_path / 'weights.pth'), '--titles', str(titles), '--dry-run'])
    assert result == 0
    output = capsys.readouterr().out
    assert 'tt1234567' in output and 'tt7654321' not in output
    assert not (tmp_path / 'out').exists()
