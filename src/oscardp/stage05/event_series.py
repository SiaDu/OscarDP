"""Time-based finite-window operations; never interpolate or bridge invalid frames."""
import numpy as np


def segments(rows, mask, max_gap_sec):
    result, current = [], []
    for i, good in enumerate(mask):
        continuous = not current or (rows[i]["global_frame_index"] == rows[current[-1]]["global_frame_index"]+1
                                    and rows[i]["movie_timestamp_sec"]-rows[current[-1]]["movie_timestamp_sec"] <= max_gap_sec)
        if current and (not good or not continuous):
            result.append(current)
            current = []
        if good: current.append(i)
    if current: result.append(current)
    return result


def median_filter(times, values, radius_sec):
    values = np.asarray(values, dtype=float)
    return np.asarray([np.median(values[np.abs(times-t)<=radius_sec+1e-10],axis=0) for t in times])


def derivative(times, values):
    values = np.asarray(values)
    return np.gradient(values,times,axis=0) if len(times)>1 else np.zeros_like(values)


def hysteresis(times, next_times, values, high, low, merge_gap_sec=0.):
    """Maximal above-low runs containing a high seed; merge only short internal dips."""
    runs, start = [], None
    for i in range(len(values)+1):
        active = i<len(values) and values[i]>low
        if active and start is None: start = i
        if start is not None and not active:
            runs.append([start,i])
            start = None
    merged = []
    for a,b in runs:
        if merged and times[a]-next_times[merged[-1][1]-1] <= merge_gap_sec+1e-10:
            merged[-1][1] = b
        else: merged.append([a,b])
    return [(a,b) for a,b in merged if max(values[a:b])>=high]


def eye_geometry(row):
    values = []
    for start in [36,42]:
        points = np.array([[row.get(f"x_{i}"),row.get(f"y_{i}")] for i in range(start,start+6)],dtype=float)
        width = np.linalg.norm(points[0]-points[3])
        if not np.isfinite(points).all() or width<=0: return None
        values.append(float((np.linalg.norm(points[1]-points[5])+np.linalg.norm(points[2]-points[4]))/(2*width)))
    return values


def rotation(angles):
    # OpenFace left-handed-positive Rx * Ry * Rz convention.
    x,y,z = -np.asarray(angles)
    cx,sx,cy,sy,cz,sz = np.cos(x),np.sin(x),np.cos(y),np.sin(y),np.cos(z),np.sin(z)
    return np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])


def rotation_distance(a,b):
    return float(np.arccos(np.clip((np.trace(rotation(a).T@rotation(b))-1)/2,-1,1)))


def gaze_direction(angles):
    vector = np.array([np.tan(angles[0]),np.tan(angles[1]),-1.])
    return vector/np.linalg.norm(vector)


def gaze_distance(a,b):
    return float(np.arccos(np.clip(gaze_direction(a)@gaze_direction(b),-1,1)))
