import torch

from oscardp.shots.transnet import infer_stream, predictions_to_boundaries
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
    predictions = infer_stream(frames, predictor)
    assert len(predictions) == 74
    assert predictions == [float(index) for index in range(74)]
    assert len(predictor.windows) == 2
    assert predictor.windows[0][:25] == [bytes([0])] * 25
    assert predictor.windows[-1][-1] == bytes([73])


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
