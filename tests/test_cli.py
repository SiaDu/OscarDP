from oscardp.shots.__main__ import DEFAULT_INPUT_ROOT, DEFAULT_OUTPUT_ROOT, build_parser


def test_cli_defaults_and_overwrite_exclusion() -> None:
    parser = build_parser()
    discover = parser.parse_args(["discover"])
    assert discover.input_root == DEFAULT_INPUT_ROOT
    process = parser.parse_args([
        "process-one", "--video", "/tmp/movie.mp4", "--weights", "/tmp/model.pth", "--overwrite"
    ])
    assert process.output_root == DEFAULT_OUTPUT_ROOT
    assert process.overwrite is True
    assert process.resume is True
