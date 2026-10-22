from pathlib import Path
from typing import Protocol

import numpy as np


class FaceDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[dict]: ...


class YuNetDetector:
    def __init__(self, model: Path, score_threshold=0.8, max_size=0, device="cpu"):
        import cv2
        if not model.is_file():
            raise FileNotFoundError(f"YuNet ONNX model missing: {model}")
        if device != "cpu":
            raise ValueError("This YuNet baseline supports --device cpu only")
        if not 0 < score_threshold <= 1 or max_size < 0:
            raise ValueError("Invalid detector threshold or max size")
        self.max_size = max_size
        self.net = cv2.FaceDetectorYN.create(str(model), "", (320, 320),
                                           score_threshold, 0.3, 5000,
                                           cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)

    @staticmethod
    def map_faces(faces, sx, sy):
        result = []
        for row in ([] if faces is None else faces):
            if not np.isfinite(row).all() or row[2] <= 0 or row[3] <= 0:
                continue
            result.append({"bbox_xywh": [float(row[0]*sx), float(row[1]*sy),
                                         float(row[2]*sx), float(row[3]*sy)],
                           "landmarks5": [[float(row[i]*sx), float(row[i+1]*sy)]
                                          for i in range(4, 14, 2)],
                           "detection_score": float(row[14])})
        return result

    def detect(self, image):
        import cv2
        h, w = image.shape[:2]
        scale = min(1., self.max_size/max(h, w)) if self.max_size else 1.
        iw, ih = max(1, round(w*scale)), max(1, round(h*scale))
        sample = cv2.resize(image, (iw, ih)) if (iw, ih) != (w, h) else image
        self.net.setInputSize((iw, ih))
        _, faces = self.net.detect(sample)
        return self.map_faces(faces, w/iw, h/ih)


class SCRFDDetector:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SCRFD is an extension point; no recognition model is loaded")
