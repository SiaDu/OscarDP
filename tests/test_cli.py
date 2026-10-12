from oscardp.shots.__main__ import DEFAULT_INPUT_ROOT, DEFAULT_OUTPUT_ROOT, build_parser


def test_cli_defaults_and_options() -> None:
    parser = build_parser()
    assert parser.parse_args(["discover"]).input_root == DEFAULT_INPUT_ROOT
    args = parser.parse_args(["process-one", "--video", "/tmp/tt1234567.mp4", "--weights", "/tmp/model.pth", "--device", "cuda"])
    assert args.output_root == DEFAULT_OUTPUT_ROOT
    assert args.device == "cuda"
    assert not args.overwrite
    assert parser.parse_args(["validate", "--movie-id", "tt1234567"]).movie_id == "tt1234567"
