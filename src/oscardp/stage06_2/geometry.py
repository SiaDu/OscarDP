"""OpenFace model-to-camera rotations and direction decomposition."""

import numpy as np

FORWARD = np.array([0., 0., -1.])


def unit(vector):
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        return None
    length = np.linalg.norm(v)
    return v / length if np.isfinite(length) and length > 1e-12 else None


def head_rotation(angles):
    """Match OpenFace Euler2RotationMatrix on the recorded radians, without negation."""
    a = np.asarray(angles, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        return None
    sx, sy, sz = np.sin(a)
    cx, cy, cz = np.cos(a)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rx @ ry @ rz


def decompose(pose, left, right):
    rotation = head_rotation(pose)
    left, right = unit(left), unit(right)
    gaze = unit(left + right) if left is not None and right is not None else None

    def local(vector):
        return rotation.T @ vector if rotation is not None and vector is not None else None

    return {
        "head_direction": rotation @ FORWARD if rotation is not None else None,
        "gaze_direction": gaze,
        "left_eye_relative_direction": local(left),
        "right_eye_relative_direction": local(right),
        "eye_relative_direction": local(gaze),
    }


def angular_change(a, b):
    """Angular distance of two unit rays, in degrees."""
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def head_change(a, b):
    relative = head_rotation(a).T @ head_rotation(b)
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def direction_angles(vector):
    x, y, z = vector
    return np.degrees(np.arctan2([x, y], -z))
