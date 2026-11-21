"""OpenFace 28-point eye fits, checked separately from gaze usability."""
import numpy as np
from PIL import Image, ImageDraw

LANDMARKS = [f"eye_lmk_{axis}_{i}" for axis in "xy" for i in range(56)]
WINDOWS = [(1472.096, 1472.763), (1475.391, 1476.058)]


def eye_points(row, eye):
    return np.array([[row[f"eye_lmk_{a}_{i}"] for a in "xy"]
                     for i in range(28*eye, 28*(eye+1))], dtype=float)


def overlays(rows, pictures, path):
    sheet = Image.new("RGB", (800, 155*len(rows)), "#171f29")
    draw = ImageDraw.Draw(sheet)
    for j,row in enumerate(rows):
        index = row["global_frame_index"]
        draw.text((10,j*155+3), f"{index} | {row['movie_timestamp_sec']:.6f} s   cyan: iris fit   yellow: model centre", fill="white")
        for eye in (0,1):
            pts = eye_points(row, eye)
            if not np.isfinite(pts).all():
                continue
            frame = pictures[index]
            x,y = np.maximum(0, np.floor(pts.min(axis=0)-18)).astype(int)
            a,b = np.minimum(frame.size, np.ceil(pts.max(axis=0)+18)).astype(int)
            if a <= x or b <= y:
                continue
            crop = frame.crop((x,y,a,b))
            crop.thumbnail((390,125))
            marks = ImageDraw.Draw(crop)
            coords = (pts-[x,y]) * (np.array(crop.size)/[a-x,b-y])
            for k,(u,v) in enumerate(coords):
                marks.ellipse((u-2,v-2,u+2,v+2),fill="#56eaff" if k < 8 else "#e58d61")
            u,v = coords[:8].mean(axis=0)
            marks.ellipse((u-3,v-3,u+3,v+3),outline="#ffff64",width=2)
            sheet.paste(crop,(eye*400+5,j*155+24))
    sheet.save(path)


def attach_positions(rows):
    for row in rows:
        for i, side in enumerate(("left", "right")):
            pts = eye_points(row, i)
            opening = row["eye_aspect_ratio_proxies"]
            if row["closure_event_ids"]:
                qc = "closure_candidate"
            elif not row["tracking_usable"] or not row["landmarks_2d_valid"] or not row["head_usable"]:
                qc = "tracking_or_pose_unavailable"
            elif opening is None or opening[i] is None or opening[i] < .18 or abs(row["pose_Ry"]) > .5:
                qc = "low_visibility_proxy"
            else:
                qc = "model_fit_candidate" if position(pts) is not None else "poor_iris_fit_geometry"
            xy = position(pts) if qc == "model_fit_candidate" else None
            row.update({f"{side}_eye_x": xy[0] if xy else None, f"{side}_eye_y": xy[1] if xy else None,
                        f"{side}_eye_visibility": qc})


def position(points):
    if not np.isfinite(points).all():
        return None
    left, right = sorted(points[[8, 14]], key=lambda p: p[0])
    delta = right-left
    width = np.linalg.norm(delta)
    if width < 10:
        return None
    x = delta/width
    y = np.array([-x[1], x[0]])
    iris = points[:8]
    centre = iris.mean(axis=0)
    coords = np.array([x, y]) @ (centre-left)/width
    # Only the centre is expected inside the fitted aperture: eyelids occlude the contour.
    lid = (points[8:20]-left) @ np.array([x, y]).T / width
    t = coords[0]
    if not 0 < t < 1:
        return None
    boundaries = []
    for contour in (lid[:7], np.vstack((lid[6:], lid[0]))):
        order = np.argsort(contour[:, 0])
        boundaries.append(np.interp(t, contour[order, 0], contour[order, 1]))
    if not min(boundaries) <= coords[1] <= max(boundaries):
        return None
    return coords.tolist()
