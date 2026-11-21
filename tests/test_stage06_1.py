"""Contract checks for Stage06.1; fixtures are synthetic, never pilot evidence."""

import xml.etree.ElementTree as ET

import pytest

from oscardp.stage06_1.output import make_svg
from oscardp.stage06_1.regions import infer_region
from oscardp.stage06_1.validation import add_camera_groups, add_camera_regions, validate_candidate


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


@pytest.fixture
def camera_group(inputs):
    candidate, _, _ = inputs
    shot_ids = [shot["shot_id"] for shot in candidate["shots"]]
    return dict(camera_group="camera_001", shot_ids=shot_ids, reference_shot=shot_ids[0],
                group_quality="rough", evidence_shots=list(shot_ids), note="相似角度候选。")


@pytest.mark.parametrize(("field", "value", "error"), [
    ("camera_group", "group_001", "invalid or duplicate"),
    ("shot_ids", ["shot_000001"], "at least two"),
    ("shot_ids", ["shot_000001", "shot_000001"], "duplicate evidence shot"),
    ("shot_ids", ["shot_000001", "shot_000003"], "shot IDs"),
    ("reference_shot", "shot_000003", "must belong"),
    ("group_quality", "certain", "group_quality"),
    ("evidence_shots", ["shot_000001"], "every member"),
])
def test_invalid_camera_group(inputs, camera_group, field, value, error):
    candidate, _, inspected = inputs
    camera_group[field] = value
    with pytest.raises(ValueError, match=error):
        add_camera_groups(candidate, [camera_group], inspected)


def test_camera_group_reference_integrity(inputs, camera_group):
    candidate, packet, inspected = inputs
    with pytest.raises(ValueError, match="already belongs"):
        add_camera_groups(candidate, [camera_group, {**camera_group, "camera_group": "camera_002"}], inspected)
    with pytest.raises(ValueError, match="inspected images"):
        add_camera_groups(candidate, [camera_group], {"shot_000001": ["first.jpg"]})
    grouped = add_camera_groups(candidate, [camera_group], inspected)
    for link in ("camera_002", None):
        grouped["shots"][1]["camera_group"] = link
        with pytest.raises(ValueError, match="disagrees"):
            validate_candidate(grouped, packet, inspected)
    del grouped["shots"][1]["camera_group"]
    with pytest.raises(ValueError, match="expected fields"):
        validate_candidate(grouped, packet, inspected)


def test_camera_direction_and_unplaced_group(inputs, camera_group):
    candidate, packet, inspected = inputs
    candidate["layouts"][0]["people"][0].update(position=[0.2, 0.5], position_quality="rough")
    for shot in candidate["shots"]:
        shot["camera_subject"] = "MICHELLE"
    grouped = add_camera_groups(candidate, [camera_group], inspected)
    validate_candidate(grouped, packet, inspected)
    assert "camera_groups" not in candidate
    assert all("camera_group" not in shot for shot in candidate["shots"])
    assert [s["camera_position"] for s in grouped["shots"]] == [[0.5, 0.8], None]

    svg = ET.fromstring(make_svg(grouped))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    assert len(svg.findall('svg:line[@class="camera-direction"]', ns)) == 1
    assert len(svg.findall('svg:rect[@class="camera"]', ns)) == 1
    assert {"camera_001 → MICHELLE", "shot_000002"} <= {e.text for e in svg.findall("svg:text", ns)}


@pytest.fixture
def region_inputs(inputs, camera_group):
    base, packet, inspected = inputs
    michelle = base["layouts"][0]["people"][0]
    michelle.update(position=[0.2, 0.5], position_quality="rough", facing_to="TEDDY")
    base["layouts"][0]["people"].append(
        {**michelle, "person": "TEDDY", "position": [0.8, 0.5], "facing_to": "MICHELLE"})
    base["shots"][1]["layout_id"] = None
    base = add_camera_groups(base, [camera_group], inspected)
    clues = dict(view_person="UNKNOWN", view_of_torso="unknown", depth_order=[],
                 screen_order=[], camera_height="low", note="测试条件。")
    annotations = {
        "shot_000001": dict(layout_basis="observed", camera_clues=clues),
        "shot_000002": dict(layout_basis="assumed_static", camera_clues={
            **clues, "view_person": "MICHELLE", "view_of_torso": "right_side",
            "screen_order": [{"left": "MICHELLE", "right": "TEDDY"}],
        }),
    }
    return base, annotations, packet, inspected


def test_region_increment_preserves_legacy_state_and_points(region_inputs):
    base, annotations, packet, inspected = region_inputs
    validate_candidate(base, packet, inspected)
    with pytest.raises(ValueError, match="explicit opt-in"):
        add_camera_regions(base, annotations)
    result = add_camera_regions(base, annotations, static_layout="layout_001")
    validate_candidate(result, packet, inspected)
    assert result["layouts"] == base["layouts"]
    assert result["camera_groups"] == base["camera_groups"]
    assert result["review_status"] == base["review_status"]
    assert result["shots"][0]["camera_region"] is None  # height and group are metadata
    shot = result["shots"][1]
    assert base["shots"][1]["layout_id"] is None
    assert shot["layout_id"] == "layout_001" and shot["layout_status"] == "unknown"
    assert shot["camera_position"] is None
    region = shot["camera_region"]
    assert set(region["used_clues"]) == {"view_of_torso", "screen_order", "axis_side"}
    assert [0.2, 0.9] in region["cells"]
    assert all(y > 0.5 for _, y in region["cells"])
    svg = ET.fromstring(make_svg(result))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    assert len(svg.findall('svg:rect[@class="camera"]', ns)) == 1
    assert len(svg.findall('svg:rect[@class="region-cell"]', ns)) == len(region["cells"])


@pytest.mark.parametrize(("invalid", "message"), [
    ("basis", "layout_basis"),
    ("pair", "distinct pair"),
    ("boundary", "coordinates"),
    ("duplicate", "duplicate grid centre"),
    ("forged", "does not match"),
])
def test_validate_region_payload(region_inputs, invalid, message):
    base, annotations, packet, inspected = region_inputs
    result = add_camera_regions(base, annotations, static_layout="layout_001")
    shot = result["shots"][1]
    cells = shot["camera_region"]["cells"]
    if invalid == "basis":
        shot["layout_basis"] = "verified"
    elif invalid == "pair":
        shot["camera_clues"]["depth_order"] = [{"nearer": "DON", "farther": "DON"}]
    elif invalid == "boundary":
        cells.append([1.1, 0.1])
    elif invalid == "duplicate":
        cells.append(cells[0])
    elif invalid == "forged":
        cells.append([0, 0])
    with pytest.raises(ValueError, match=message):
        validate_candidate(result, packet, inspected)


def test_region_screen_projection_and_depth(region_inputs):
    base, annotations, _, _ = region_inputs
    shot = {**base["shots"][1], **annotations["shot_000002"], "layout_id": "layout_001",
            "axis_side": "unknown"}
    shot["camera_clues"] = {**shot["camera_clues"], "view_of_torso": "unknown",
                            "depth_order": [{"nearer": "MICHELLE", "farther": "TEDDY"}]}
    layout = base["layouts"][0]
    region, _ = infer_region(shot, layout, {})
    assert [0.3, 0.8] in region["cells"]
    assert [0.7, 0.8] not in region["cells"]  # reversed depth relative to the group aim
    assert [0.3, 0.2] not in region["cells"]  # reversed screen sides
    shot["camera_clues"]["screen_order"].append({"left": "TEDDY", "right": "MICHELLE"})
    empty, diagnostics = infer_region(shot, layout, {})
    assert empty is None and any("图内无共同可行网格" in item for item in diagnostics)
    # Without an aim, neither depth nor screen order gives a usable projection.
    empty, diagnostics = infer_region({**shot, "camera_subject": "UNKNOWN"}, layout, {})
    assert empty is None and any("无可用的投影拍摄朝向" in item for item in diagnostics)
