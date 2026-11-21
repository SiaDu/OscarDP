"""Contract checks for Stage06.1; fixtures are synthetic, never pilot evidence."""

import xml.etree.ElementTree as ET

import pytest

from oscardp.stage06_1.output import make_svg
from oscardp.stage06_1.validation import validate_candidate


@pytest.fixture
def inputs():
    first, last = "shot_000001", "shot_000002"
    pair = ["MICHELLE", "TEDDY"]
    scope = {"movie_id": "tt12300742", "sequence_id": "seq_006", "scene_id": "scene_000021",
             "start_s": 1, "end_s": 4}
    packet = {**scope, "shots": [{"shot_id": sid} for sid in (first, last)]}
    reference = dict(shot_id=first, layout_id="layout_001", camera_position=[0.5, 0.8],
                     camera_subject="GROUP", axis_pair=pair, axis_reference_shot=None,
                     axis_side="unknown", camera_quality="rough", layout_status="unknown",
                     evidence_shots=[first], note="测试参照。")
    person = dict(person="MICHELLE", position=None, posture="sitting", facing_to="UNKNOWN",
                  position_quality="unknown", evidence_shots=[first], note="测试位置未知。")
    candidate = {
        **scope, "status": "model_candidate", "review_status": "pending",
        "layouts": [dict(layout_id="layout_001", start_shot_id=first, end_shot_id=last,
                         reference_shots=[first], axis_pair=pair, people=[person],
                         layout_status="unknown", note="测试布局。")],
        "shots": [reference, {**reference, "shot_id": last, "camera_position": None,
                              "axis_pair": list(reversed(pair)), "axis_reference_shot": first,
                              "axis_side": "same", "evidence_shots": [first, last]}],
    }
    return candidate, packet, {first: ["first.jpg"], last: ["last.jpg"]}


@pytest.mark.parametrize(("field", "value", "error"), [
    ("camera_subject", "Teddy", "camera_subject"),
    ("camera_position", [1.1, 0.4], "coordinates"),
    ("camera_position", [0.15, 0.4], "coordinates"),
    ("camera_position", [True, 0.4], "coordinates"),
    ("layout_id", "layout_002", "unknown layout_id"),
    ("evidence_shots", ["shot_000003"], "inspected images"),
    ("axis_reference_shot", "shot_000002", "different inspected shot"),
    ("axis_reference_shot", None, "same pair"),
    ("axis_pair", ["TEDDY", "DON"], "same pair"),
    ("gaze_target", "TEDDY", "expected fields"),
])
def test_invalid_shot(inputs, field, value, error):
    candidate, packet, inspected = inputs
    candidate["shots"][1][field] = value
    with pytest.raises(ValueError, match=error):
        validate_candidate(candidate, packet, inspected)


def test_requires_coverage_and_inspected_images(inputs):
    candidate, packet, inspected = inputs
    candidate["shots"].pop()
    with pytest.raises(ValueError, match="every selected shot"):
        validate_candidate(candidate, packet, inspected)
    inspected.pop("shot_000002")
    with pytest.raises(ValueError, match="inspected image"):
        validate_candidate(candidate, packet, inspected)


def test_svg_draws_real_coordinates_only(inputs):
    candidate, packet, inspected = inputs
    validate_candidate(candidate, packet, inspected)
    svg = ET.fromstring(make_svg(candidate))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    assert svg.find("svg:title", ns).text == "Approximate layout — not calibrated"
    assert len(svg.findall('svg:rect[@class="camera"]', ns)) == 1
    assert svg.findall("svg:circle", ns) == []
    assert svg.findall("svg:line", ns) == []
