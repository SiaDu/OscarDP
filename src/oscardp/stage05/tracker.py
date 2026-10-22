"""Conservative mutual-best one-to-one assignment; ambiguous links are broken."""
import math


def similarity(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax+aw, bx+bw)-max(ax, bx))*max(0, min(ay+ah, by+bh)-max(ay, by))
    iou = intersection / max(1e-9, aw*ah+bw*bh-intersection)
    distance = math.hypot(ax+aw/2-bx-bw/2, ay+ah/2-by-bh/2)/max(1, math.hypot(aw, ah))
    size_ratio = max(aw/bw, bw/aw, ah/bh, bh/ah)
    return iou - .2*distance if (iou >= .15 and distance <= .6 and size_ratio < 2) else -1.


class ShotTracker:
    def __init__(self, shot_id, max_missed=3, min_hits=3, ambiguity_margin=.12):
        self.shot_id = shot_id
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.margin = ambiguity_margin
        self.tracks = []
        self.ambiguities = []

    def break_links(self, frame_index, difference):
        """A visual discontinuity is association uncertainty, not a new shot ID."""
        affected = [t["track_id"] for t in self.tracks if not t["closed"]]
        for t in self.tracks:
            t["closed"] = True
        self.ambiguities.append({"frame_index": frame_index,
                                 "reason": "visual_discontinuity_association_reset",
                                 "affected_track_ids": affected,
                                 "normalized_image_difference": difference,
                                 "interpretation": "Possible cut, flash or rapid motion; not a verified shot boundary."})

    def update(self, detections, frame_index, timestamp, next_timestamp):
        active = [t for t in self.tracks if not t["closed"]]
        scores = []
        for t in active:
            dt = frame_index-t["last_frame"]
            predicted = [t["bbox"][k]+t["velocity"][k]*dt for k in range(4)]
            predicted[2:] = t["bbox"][2:]
            scores.append([similarity(predicted, d["bbox_xywh"]) for d in detections])
        assigned = {}
        for i, row in enumerate(scores):
            if not row or max(row) < 0:
                continue
            j = max(range(len(row)), key=row.__getitem__)
            column = [r[j] for r in scores]
            rivals = [v for k, v in enumerate(row) if k != j] + [v for k, v in enumerate(column) if k != i]
            if rivals and row[j]-max(rivals) < self.margin:
                self.ambiguities.append({"frame_index": frame_index,
                                         "track_id": active[i]["track_id"],
                                         "reason": "ambiguous_association_link_broken"})
                active[i]["closed"] = True
                continue
            if column.index(max(column)) == i:
                assigned[j] = active[i]
        used = {t["track_id"] for t in assigned.values()}
        for t in active:
            if t["track_id"] not in used and frame_index-t["last_frame"] > self.max_missed:
                t["closed"] = True
        records = []
        for j, d in enumerate(detections):
            t = assigned.get(j)
            if t is None:
                t = {"track_id": f"{self.shot_id}_track_{len(self.tracks)+1:03d}",
                     "shot_id": self.shot_id, "start_frame": frame_index,
                     "start_sec": timestamp, "last_frame": frame_index,
                     "bbox": d["bbox_xywh"], "velocity": [0.]*4,
                     "count": 0, "missed": 0, "score": 0., "width": 0., "height": 0., "closed": False}
                self.tracks.append(t)
            gap = frame_index-t["last_frame"]
            if t["count"]:
                t["missed"] += gap-1
                t["velocity"] = [(d["bbox_xywh"][k]-t["bbox"][k])/gap for k in range(4)]
            t.update(last_frame=frame_index, end_sec=next_timestamp, bbox=d["bbox_xywh"])
            t["count"] += 1
            t["score"] += d["detection_score"]
            t["width"] += d["bbox_xywh"][2]
            t["height"] += d["bbox_xywh"][3]
            records.append(dict(d, shot_id=self.shot_id, frame_index=frame_index,
                                timestamp_sec=timestamp, track_id=t["track_id"], detection_status="observed"))
        return records

    def summaries(self):
        return [{"track_id": t["track_id"], "shot_id": self.shot_id,
                 "start_frame": t["start_frame"], "end_frame": t["last_frame"]+1,
                 "start_sec": t["start_sec"], "end_sec": t["end_sec"],
                 "detected_frame_count": t["count"], "missed_frame_count": t["missed"],
                 "detection_score_mean": t["score"]/t["count"],
                 "bbox_width_mean": t["width"]/t["count"], "bbox_height_mean": t["height"]/t["count"],
                 "valid_detection_ratio": t["count"]/(t["last_frame"]+1-t["start_frame"]),
                 "tracking_status": "confirmed_ended" if t["count"] >= self.min_hits else "tentative_ended"}
                for t in self.tracks]
