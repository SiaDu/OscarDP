from pathlib import Path

import pytest
import torch

from oscardp import vendor
from oscardp.shots.transnet import (
    TransNetRunner,
    infer_stream,
    predictions_to_boundaries,
    select_device,
)
from oscardp.vendor.transnetv2_pytorch import TransNetV2


class FakePredictor:
    def __init__(self) -> None:
        self.windows: list[list[bytes]] = []

    def predict_window(self, frames: list[bytes]) -> list[float]:
        self.windows.append(list(frames))
        return [float(value[0]) for value in frames[25:75]]


def test_streaming_windows_match_length_and_padding() -> None:
    predictor = FakePredictor()
    frames = [bytes([index]) for index in range(74)]
    progress: list[int] = []
    predictions = infer_stream(frames, predictor, progress.append)
    assert len(predictions) == 74
    assert predictions == [float(index) for index in range(74)]
    assert len(predictor.windows) == 2
    assert predictor.windows[0][:25] == [bytes([0])] * 25
    assert predictor.windows[-1][-1] == bytes([73])
    assert progress == list(range(1, 75))


def test_transition_runs_become_apex_boundaries() -> None:
    predictions = [0.1, 0.7, 0.9, 0.8, 0.1, 0.6, 0.2]
    boundaries = predictions_to_boundaries(predictions, 0.5)
    assert [(item.frame, item.confidence) for item in boundaries] == [(2, 0.9), (5, 0.6)]


def test_edge_boundary_is_discarded() -> None:
    boundaries = predictions_to_boundaries([0.9, 0.1, 0.1], 0.5)
    assert boundaries == []


def test_vendored_model_accepts_official_input_shape() -> None:
    model = TransNetV2().eval()
    frames = torch.zeros((1, 100, 27, 48, 3), dtype=torch.uint8)
    with torch.no_grad():
        logits, extra = model(frames)
    assert tuple(logits.shape) == (1, 100, 1)
    assert tuple(extra["many_hot"].shape) == (1, 100, 1)


def test_missing_weights_fail_before_model_load(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="weights not found"):
        TransNetRunner.load(tmp_path / "missing.pth", "cpu")


def test_device_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(select_device("auto")) == "cpu"
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        select_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert str(select_device("auto")) == "cuda:0"
    assert str(select_device("cuda")) == "cuda:0"


class TinyTransNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.used_inference_mode = False

    def forward(self, frames: torch.Tensor):
        self.used_inference_mode = torch.is_inference_mode_enabled()
        logits = torch.zeros((1, 100, 1), device=frames.device) * self.scale
        return logits, {"many_hot": logits}


def test_load_keeps_cpu_map_location_and_inference_devices_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    expected_state = TinyTransNet().state_dict()
    observed: dict[str, object] = {}

    def fake_load(path, *, map_location, weights_only):
        observed["map_location"] = map_location
        observed["weights_only"] = weights_only
        return expected_state

    monkeypatch.setattr(vendor.transnetv2_pytorch, "TransNetV2", TinyTransNet)
    monkeypatch.setattr(torch, "load", fake_load)
    runner = TransNetRunner.load(weights, "cpu", model_sha256="known-sha")
    runner.reset_inference_metrics()
    frame = bytes(27 * 48 * 3)
    predictions = runner.predict_window([frame] * 100)

    assert observed == {"map_location": "cpu", "weights_only": True}
    assert runner.model_device == "cpu"
    assert runner.input_device == "cpu"
    assert runner.model.used_inference_mode is True
    assert predictions == [0.5] * 50
