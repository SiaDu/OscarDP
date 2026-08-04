from __future__ import annotations

import hashlib
import time
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


def select_device(device_name: str) -> Any:
    import torch

    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device_name}")
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda:0")
    return torch.device("cpu")


class TransNetRunner:
    def __init__(
        self,
        model: Any,
        device: Any,
        model_sha256: str,
        *,
        requested_device: str,
        model_load_sec: float,
    ):
        self.model = model
        self.device = device
        self.model_sha256 = model_sha256
        self.requested_device = requested_device
        self.model_load_sec = model_load_sec
        self.model_device = str(next(model.parameters()).device)
        self.input_device: str | None = None
        self.model_inference_sec = 0.0

    @classmethod
    def load(
        cls,
        weights_path: Path,
        device_name: str = "auto",
        model_sha256: str | None = None,
    ) -> TransNetRunner:
        if not weights_path.is_file():
            raise FileNotFoundError(f"TransNetV2 weights not found: {weights_path}")
        import torch

        from oscardp.vendor.transnetv2_pytorch import TransNetV2

        device = select_device(device_name)
        load_started = time.perf_counter()
        model = TransNetV2()
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_load_sec = time.perf_counter() - load_started
        return cls(
            model,
            device,
            model_sha256 or sha256_file(weights_path),
            requested_device=device_name,
            model_load_sec=model_load_sec,
        )

    def reset_inference_metrics(self) -> None:
        import torch

        self.input_device = None
        self.model_inference_sec = 0.0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

    def peak_memory_allocated(self) -> int:
        if self.device.type != "cuda":
            return 0
        import torch

        torch.cuda.synchronize(self.device)
        return int(torch.cuda.max_memory_allocated(self.device))

    def predict_window(self, frames: list[bytes]) -> list[float]:
        if len(frames) != 100:
            raise ValueError("TransNetV2 windows must contain exactly 100 frames")
        import numpy as np
        import torch

        array = np.frombuffer(b"".join(frames), dtype=np.uint8).reshape(1, 100, 27, 48, 3).copy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_started = time.perf_counter()
        tensor = torch.from_numpy(array).to(self.device)
        actual_input_device = str(tensor.device)
        if actual_input_device != self.model_device:
            raise RuntimeError(
                f"TransNetV2 model is on {self.model_device}, input tensor is on "
                f"{actual_input_device}"
            )
        if self.input_device is None:
            self.input_device = actual_input_device
        with torch.inference_mode():
            logits, _ = self.model(tensor)
            predictions = torch.sigmoid(logits)[0, 25:75, 0].detach().cpu().tolist()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.model_inference_sec += time.perf_counter() - inference_started
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
