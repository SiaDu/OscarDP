"""Temporal sampling and observable crop-quality proxies, never pose recovery."""
import cv2
import numpy as np

from .identity_model import align


def candidates(records, limit=16):
    """One high-confidence candidate per equally spaced temporal bin."""
    if not records or not 2 <= limit <= 16:
        raise ValueError("Require observations and 2–16 temporal candidate bins")
    ordered = sorted(records, key=lambda r: r["frame_index"])
    groups = np.array_split(np.arange(len(ordered)), min(limit,len(ordered)))
    return [max((ordered[int(i)] for i in group), key=lambda r: (r["detection_score"], min(r["bbox_xywh"][2:]), -r["frame_index"]))
            for group in groups]


def crop_quality(image, record, min_score=.85, min_size=80, min_sharpness=30., max_fit_error=10., min_valid_fraction=.95):
    reasons = []
    x,y,w,h = record["bbox_xywh"]
    height,width = image.shape[:2]
    metrics = {"detection_score": record["detection_score"], "native_bbox_width": w, "native_bbox_height": h,
               "bbox_visible_fraction": max(0,min(width,x+w)-max(0,x))*max(0,min(height,y+h)-max(0,y))/max(1,w*h),
               "occlusion_status": "not_verified; five landmarks do not establish visibility of all face regions",
               "pose_status": "2D landmark geometry proxies only; no head-pose measurement"}
    if record["detection_score"] < min_score: reasons.append("low_detection_score")
    if min(w,h) < min_size: reasons.append("insufficient_native_face_resolution")
    try:
        points = np.asarray(record["landmarks5"], np.float64)
        if points.shape != (5,2) or not np.isfinite(points).all():
            raise ValueError("invalid five landmarks")
        if (points[:,0] < 0).any() or (points[:,0] >= width).any() or (points[:,1] < 0).any() or (points[:,1] >= height).any():
            reasons.append("landmark_outside_source_image")
        if not (points[:2,1].mean() < points[2,1] < points[3:,1].mean()):
            reasons.append("implausible_eye_nose_mouth_geometry")
        if (points[:,0] < x-.2*w).any() or (points[:,0] > x+1.2*w).any() or (points[:,1] < y-.2*h).any() or (points[:,1] > y+1.2*h).any():
            reasons.append("landmark_outside_expanded_face_box")
        crop, matrix, fit, fraction = align(image, points)
        sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        eye_distance = float(np.linalg.norm(points[0]-points[1]))
        nose_fraction = float(np.linalg.norm(points[2]-points[0])/max(1,eye_distance))
        metrics.update(landmark_valid=True, native_eye_distance=eye_distance,
                       landmark_fit_rmse_pixels_112=fit, crop_valid_source_fraction=fraction,
                       sharpness_laplacian_variance_112=sharpness, nose_eye_distance_ratio_proxy=nose_fraction,
                       affine_source_to_crop=matrix.tolist())
        if sharpness < min_sharpness: reasons.append("blurred_aligned_crop")
        if fit > max_fit_error: reasons.append("inconsistent_alignment_or_pose_geometry")
        if fraction < min_valid_fraction: reasons.append("crop_crosses_source_boundary")
        if eye_distance < 20: reasons.append("insufficient_eye_resolution")
        if not .15 < nose_fraction < 1.8: reasons.append("extreme_or_invalid_landmark_geometry")
    except (ValueError, np.linalg.LinAlgError) as exc:
        metrics.update(landmark_valid=False, landmark_error=str(exc))
        crop = None
        reasons.append("invalid_landmarks")
    return crop, metrics, reasons
