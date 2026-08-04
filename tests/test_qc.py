from oscardp.shots.qc import build_qc_summary, select_qc_boundaries
from oscardp.shots.schema import ShotRecord
from oscardp.shots.transnet import Boundary


def make_shot(index: int, start: int, end: int, duration: float) -> ShotRecord:
    return ShotRecord(
        movie_key="movie", shot_id=f"shot_{index:06d}", source_video_relpath="movie.mp4",
        start_frame=start, end_frame=end, frame_count=end-start,
        start_time="00:00:00.000", end_time="00:00:01.000", start_sec=0,
        end_sec=duration, duration_sec=duration, keyframe_frame=start,
        keyframe_time_sec=0, keyframe_relpath=f"keyframes/shot_{index:06d}.jpg",
        boundary_before_confidence=None, boundary_after_confidence=None,
        shot_scale=None, camera_movement=None, model={"name": "TransNetV2", "threshold": 0.5},
    )


def test_qc_prioritizes_low_confidence_and_short_shots() -> None:
    boundaries = [Boundary(10, 0.55), Boundary(20, 0.9), Boundary(30, 0.8)]
    shots = [make_shot(1, 0, 10, 1.0), make_shot(2, 10, 20, 0.2), make_shot(3, 20, 40, 1.0)]
    selected = select_qc_boundaries(boundaries, shots, threshold=0.5, sample_count=0)
    assert [item.frame for item in selected] == [10, 20]
    summary = build_qc_summary(shots, missing_keyframes=0, validation_passed=True)
    assert summary["very_short_shot_count"] == 1
