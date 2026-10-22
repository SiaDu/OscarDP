from pathlib import Path


class DiscontinuityGuard:
    """Cheap conservative association guard, not a shot segmentation correction."""
    def __init__(self, threshold=.18):
        self.previous = None
        self.threshold = threshold

    def update(self, image):
        import cv2
        import numpy as np
        sample = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        difference = 0. if self.previous is None else float(np.abs(sample-self.previous).mean()/255)
        self.previous = sample
        return difference if difference >= self.threshold else None


class Preview:
    """One low-resolution, 6 FPS QC video; each display frame samples real observations."""
    def __init__(self, path: Path, width, height, fps=6):
        import cv2
        scale = min(1., 960/width)
        self.size = (int(width*scale)//2*2, int(height*scale)//2*2)
        self.scale = (self.size[0]/width, self.size[1]/height)
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open preview writer: {path}")
        self.next_time = None
        self.fps = fps

    def add(self, frame, records, shot_id):
        import cv2
        if self.next_time is not None and frame.timestamp < self.next_time:
            return
        if self.next_time is None:
            self.next_time = frame.timestamp
        image = cv2.resize(frame.image, self.size)
        sx, sy = self.scale
        for r in records:
            x,y,w,h = r["bbox_xywh"]
            p = (round(x*sx), round(y*sy))
            cv2.rectangle(image, p, (round((x+w)*sx), round((y+h)*sy)), (0,255,0), 2)
            cv2.putText(image, f"{r['track_id']} {r['detection_score']:.2f}",
                        p, cv2.FONT_HERSHEY_SIMPLEX, .45, (0,255,0), 1)
        cv2.putText(image, f"{shot_id} frame {frame.index} t={frame.timestamp:.6f}s", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,255), 1)
        while self.next_time <= frame.timestamp:
            self.writer.write(image)
            self.next_time += 1/self.fps

    def close(self):
        self.writer.release()


def report(results, detector):
    tracks = [t for r in results for t in r.get("tracks", [])]
    warnings = [w for r in results for w in r.get("tracking_ambiguities", [])]
    return {"schema_version": "stage05.2.v1", "processed_shot_count": sum(r["status"] == "complete" for r in results),
            "decoded_frame_count": sum(r.get("decoded_frame_count", 0) for r in results),
            "face_detection_count": sum(t["detected_frame_count"] for t in tracks),
            "face_track_count": len(tracks), "shots": results,
            "short_track_count": sum(t["detected_frame_count"] < 3 for t in tracks),
            "low_quality_detection_count": sum(r.get("low_quality_detection_count", 0) for r in results),
            "tracking_warning_count": len(warnings),
            "visual_discontinuity_count": sum(w.get("reason") == "visual_discontinuity_association_reset" for w in warnings),
            "track_valid_detection_ratios": {t["track_id"]: t["valid_detection_ratio"] for t in tracks},
            "incomplete_shots": [r["shot_id"] for r in results if r["status"] != "complete"],
            "detector": detector, "interpretation": "Program observations only; not annotated detection recall or tracking accuracy. Track IDs are not identities.",
            "decoded_frame_count_semantics": "Requested shot frames delivered to the pipeline; excludes sequential prefix/gap decode and boundary-only timestamp frames.",
            "missed_frame_semantics": "Undetected frames between first and last observations; trailing lost frames excluded."}
