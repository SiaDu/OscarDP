"""Checks for the optional v2 parts of the Stage07 interpretation packet."""
import csv

from oscardp.stage07.interpretation_input import (
    FEEDBACK_FIELDS,
    gaze_relations,
    review_feedback,
)


def test_review_feedback_keeps_filled_rows_only(tmp_path):
    with (tmp_path / "review.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*FEEDBACK_FIELDS, "gaze"))
        writer.writeheader()
        writer.writerow(dict.fromkeys((*FEEDBACK_FIELDS, "gaze"), "") | {"annotation_id": "a1", "human_agree": "NO",
                                                                        "human_notes": "blink, not disgust"})
        writer.writerow(dict.fromkeys((*FEEDBACK_FIELDS, "gaze"), "") | {"annotation_id": "a2"})
    feedback = review_feedback(tmp_path)
    assert [r["annotation_id"] for r in feedback["review"]] == ["a1"] and "gaze" not in feedback["review"][0]
    assert "head_gestures" not in feedback


def test_gaze_relations_name_the_follower_segment():
    stats = {"characters": {}, "two_person_shots": ["s1"], "mutual_gaze": [], "joint_attention": [],
             "gaze_following": [{"follower_shift_id": "s1_b_shift_001", "follower": "B"}]}
    labels = {"tracks": [{"shifts": [{"shift_id": "s1_b_shift_001", "segment_id": "s1_b_g002"}]}]}
    assert gaze_relations(stats, labels)["gaze_following"][0]["follower_gaze_id"] == "s1_b_g002"
