from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Boundary:
    frame: int
    confidence: float


class WindowPredictor(Protocol):
    def predict_window(self, frames: list[bytes]) -> list[float]: ...


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class TransNetRunner:
    def __init__(self, model: Any, device: Any, model_sha256: str):
        self.model = model
        self.device = device
        self.model_sha256 = model_sha256

    @classmethod
    def load(cls, weights_path: Path, device_name: str = "auto") -> TransNetRunner:
        if not weights_path.is_file():
            raise FileNotFoundError(f"TransNetV2 weights not found: {weights_path}")
        import torch

        from oscardp.vendor.transnetv2_pytorch import TransNetV2

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        device = torch.device(device_name)
        model = TransNetV2()
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)
        return cls(model, device, sha256_file(weights_path))

    def predict_window(self, frames: list[bytes]) -> list[float]:
        if len(frames) != 100:
            raise ValueError("TransNetV2 windows must contain exactly 100 frames")
        import numpy as np
        import torch

        array = np.frombuffer(b"".join(frames), dtype=np.uint8).reshape(1, 100, 27, 48, 3).copy()
        tensor = torch.from_numpy(array).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(tensor)
            predictions = torch.sigmoid(logits)[0, 25:75, 0].detach().cpu().tolist()
        return [float(value) for value in predictions]


def infer_stream(
    frames: Iterable[bytes],
    predictor: WindowPredictor,
    progress: Callable[[int], None] | None = None,
) -> list[float]:
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("Cannot run TransNetV2 on an empty video") from exc

    frame_count = 1
    if progress is not None:
        progress(frame_count)
    buffer = [first] * 25 + [first]
    predictions: list[float] = []
    last = first
    for frame in iterator:
        buffer.append(frame)
        last = frame
        frame_count += 1
        if progress is not None:
            progress(frame_count)
        if len(buffer) == 100:
            predictions.extend(predictor.predict_window(buffer))
            del buffer[:50]
    while len(predictions) < frame_count:
        while len(buffer) < 100:
            buffer.append(last)
        predictions.extend(predictor.predict_window(buffer))
        del buffer[:50]
    return predictions[:frame_count]


def predictions_to_boundaries(predictions: Iterable[float], threshold: float = 0.5) -> list[Boundary]:
    values = [float(value) for value in predictions]
    boundaries: list[Boundary] = []
    run_start: int | None = None
    for index, confidence in enumerate(values + [float("-inf")]):
        if index < len(values) and confidence > threshold:
            if run_start is None:
                run_start = index
            continue
        if run_start is None:
            continue
        run_end = index
        apex = max(range(run_start, run_end), key=lambda item: values[item])
        if 0 < apex < len(values):
            boundaries.append(Boundary(apex, values[apex]))
        run_start = None
    return boundaries
