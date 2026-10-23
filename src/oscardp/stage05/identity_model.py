"""InsightFace ArcFace ONNX adapter; recognition only, no model downloads.

Input preprocessing mirrors InsightFace ArcFaceONNX. Similarity alignment uses
the ArcFace 112 template and a deterministic least-squares similarity transform,
equivalent to skimage's similarity fit without that additional dependency.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np

TEMPLATE = np.array([[38.2946,51.6963], [73.5318,51.5014], [56.0252,71.7366],
                     [41.5493,92.3655], [70.7299,92.2041]], dtype=np.float64)


def normalize(vector, dimension=None):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or (dimension is not None and vector.size != dimension):
        raise ValueError("Embedding dimension mismatch")
    if not np.isfinite(vector).all():
        raise ValueError("Nonfinite embedding")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Zero/degenerate embedding")
    return vector/norm


def align(image, landmarks):
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5,2) or not np.isfinite(points).all():
        raise ValueError("Invalid five-point landmarks")
    # YuNet order: anatomical right eye, left eye, nose, right mouth, left mouth;
    # these correspond to the image-left/right order of the ArcFace template.
    if points[0,0] >= points[1,0] or points[3,0] >= points[4,0]:
        raise ValueError("Reversed/invalid landmark ordering")
    source = points-points.mean(axis=0)
    target = TEMPLATE-TEMPLATE.mean(axis=0)
    variance = float((source**2).sum()/5)
    if variance < 1:
        raise ValueError("Degenerate landmarks")
    u, singular, vt = np.linalg.svd(target.T@source/5)
    sign = np.ones(2)
    if np.linalg.det(u)*np.linalg.det(vt) < 0:
        sign[-1] = -1
    rotation = u@np.diag(sign)@vt
    scale = float(singular@sign/variance)
    matrix = np.column_stack([scale*rotation, TEMPLATE.mean(axis=0)-scale*rotation@points.mean(axis=0)])
    fitted = points@matrix[:,:2].T+matrix[:,2]
    rmse = float(np.sqrt(((fitted-TEMPLATE)**2).sum(axis=1).mean()))
    crop = cv2.warpAffine(image, matrix, (112,112), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(np.ones(image.shape[:2], np.uint8), matrix, (112,112), flags=cv2.INTER_NEAREST)
    return crop, matrix, rmse, float(mask.mean())


class ArcFaceRecognizer:
    def __init__(self, model: Path, provider="CPUExecutionProvider"):
        import onnx
        import onnxruntime as ort
        if not model.is_file():
            raise FileNotFoundError(f"Recognition ONNX model missing: {model}")
        if provider not in ort.get_available_providers():
            raise ValueError(f"Requested provider unavailable: {provider}; available={ort.get_available_providers()}")
        graph = onnx.load(str(model))
        first_nodes = [n.name for n in graph.graph.node[:8]]
        embedded = any(n.startswith(("Sub","_minus")) for n in first_nodes) and any(n.startswith(("Mul","_mul")) for n in first_nodes)
        self.mean, self.std = (0.,1.) if embedded else (127.5,127.5)
        del graph
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=[provider])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0].shape[1:] != [3,112,112] or outputs[0].shape[-1] != 512:
            raise ValueError("Expected an InsightFace ArcFace 112x112/512D recognition model")
        if self.session.get_providers()[0] != provider:
            raise RuntimeError("Recognition provider silently fell back")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.dimension = 512
        self.info = {"name": "InsightFace ArcFace", "weight_filename": model.name,
                     "model_path": str(model.resolve()), "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                     "onnxruntime_version": ort.__version__, "onnx_version": onnx.__version__,
                     "requested_provider": provider, "actual_providers": self.session.get_providers(),
                     "input_shape": inputs[0].shape, "embedding_dimension": self.dimension,
                     "preprocessing": {"color": "RGB (swapRB from source BGR)", "mean": self.mean, "std": self.std,
                                       "alignment": "ArcFace 112 five-point least-squares similarity; YuNet point ordering preserved"},
                     "implementation": "OscarDP ONNX Runtime recognition-only adapter matching InsightFace ArcFaceONNX; InsightFace Python package not required"}

    def embed(self, crop):
        if crop.shape != (112,112,3):
            raise ValueError("Recognition input must be an aligned 112x112 BGR crop")
        blob = cv2.dnn.blobFromImage(crop, 1/self.std, (112,112), (self.mean,)*3, swapRB=True)
        output = self.session.run([self.output_name], {self.input_name: blob})[0]
        return normalize(output.reshape(-1), self.dimension)
