"""Validate independent, partially completed human review without changing candidates."""
import json
from datetime import datetime

PRESENCE = {"present", "absent", "uncertain", "insufficient_video_evidence"}
BOUNDARY = {"acceptable", "needs_adjustment", "not_assessable"}
QUALITY = {"good", "limited", "insufficient"}


def blank_record(item, source_hash):
    return {"review_id": item["review_id"], "source_event_id": item["event_id"] or None,
            "source_run_sha256": source_hash, "shot_id": item["shot_id"], "track_id": item["track_id"],
            "reviewed_frame_range": [int(item["start_frame"]), int(item["end_frame"])],
            "observed_event_label": None, "event_presence_decision": None,
            "temporal_boundary_decision": None, "evidence_quality": None,
            "reviewer": None, "reviewed_at": None, "notes": "", "suggested_boundaries": None,
            "video_review_attested": False, "review_status": "pending_independent_review",
            "human_verified": False}


def validate_records(records, items, source_hash, shot_ranges):
    lookup = {x["review_id"]: x for x in items}
    seen = set()
    for r in records:
        rid = r.get("review_id")
        if rid in seen or rid not in lookup:
            raise ValueError("Unknown/duplicate Review ID")
        seen.add(rid)
        original = blank_record(lookup[rid], source_hash)
        if set(r) != set(original):
            raise ValueError("Review fields differ from independent review contract")
        for field in ["source_event_id", "source_run_sha256", "shot_id", "track_id"]:
            if r[field] != original[field]:
                raise ValueError("Event ID / Source Hash / Track mismatch: "+field)
        a, b = r["reviewed_frame_range"]
        bounds = shot_ranges[r["shot_id"]]
        if not (type(a) is int and type(b) is int and bounds[0] <= a < b <= bounds[1]):
            raise ValueError("Reviewed range outside effective source Shot")
        if r["human_verified"] is not False:
            raise ValueError("This review tool does not issue verified acting evidence")
        for field, allowed in [("event_presence_decision", PRESENCE), ("temporal_boundary_decision", BOUNDARY),
                               ("evidence_quality", QUALITY)]:
            if r[field] is not None and r[field] not in allowed:
                raise ValueError("Invalid review decision: "+field)
        if not isinstance(r["notes"], str):
            raise ValueError("Notes must be text")
        if r["review_status"] == "pending_independent_review":
            if r != original:
                raise ValueError("Pending records must remain blank; complete records may be saved individually")
        elif r["review_status"] == "human_reviewed":
            if (r["video_review_attested"] is not True or not isinstance(r["reviewer"], str)
                    or not r["reviewer"].strip() or not isinstance(r["observed_event_label"], str)
                    or not r["observed_event_label"].strip()
                    or any(r[x] is None for x in ["event_presence_decision", "temporal_boundary_decision", "evidence_quality"])):
                raise ValueError("Human review needs video attestation, reviewer, observed label and decisions")
            stamp = datetime.fromisoformat(r["reviewed_at"])
            if stamp.tzinfo is None:
                raise ValueError("Human review time must include timezone")
        else:
            raise ValueError("Invalid human review status")
        suggestion = r["suggested_boundaries"]
        if suggestion is not None:
            if r["temporal_boundary_decision"] != "needs_adjustment" or set(suggestion) != {"onset_frame", "apex_frame", "offset_frame"}:
                raise ValueError("Boundary suggestion requires needs_adjustment and three phase fields")
            values = [suggestion[k] for k in ["onset_frame", "apex_frame", "offset_frame"] if suggestion[k] is not None]
            if any(type(x) is not int or not bounds[0] <= x <= bounds[1] for x in values) or values != sorted(values):
                raise ValueError("Invalid suggested phase order/range")
    # Reject JSON NaN and infinities even in optional fields.
    json.dumps(records, allow_nan=False)
    return records
