"""Validate the v1.0 field contract and references against inspected evidence."""

import re

PEOPLE = {"MICHELLE", "TEDDY", "DON"}
QUALITY = {"clear", "rough", "unknown"}
STATUS = {"stable", "changed", "unknown"}
IDENTITY_FIELDS = ("movie_id", "sequence_id", "scene_id", "start_s", "end_s")


def require(condition, location, message):
    if not condition:
        raise ValueError(f"{location}: {message}")


def fields(row, expected, location):
    require(isinstance(row, dict) and set(row) == set(expected.split()), location,
            f"expected fields: {expected}")


def label(value, allowed, location):
    require(isinstance(value, str) and value in allowed, location,
            f"expected one of {', '.join(sorted(allowed))}")


def position(value, location):
    valid = (isinstance(value, list) and len(value) == 2
             and all(type(n) in (int, float) and 0 <= n <= 1 and n == round(n, 1) for n in value))
    require(value is None or valid, location, "expected null or two coordinates in [0, 1] with at most one decimal")


def axis(value, location):
    valid = (isinstance(value, list) and len(value) == 2
             and all(isinstance(name, str) and name in PEOPLE for name in value) and value[0] != value[1])
    require(value is None or valid, location, "expected null or two distinct known character names")


def note(value, location):
    require(isinstance(value, str) and bool(value.strip()), location, "expected a nonempty note")


def references(value, inspected, location, *, required=False):
    require(isinstance(value, list) and all(isinstance(sid, str) and sid in inspected for sid in value),
            location, "expected shot IDs with inspected images")
    require(len(value) == len(set(value)), location, "duplicate evidence shot")
    require(not required or bool(value), location, "claim requires inspected shot evidence")


def camera_membership(groups, shot_ids, inspected):
    require(isinstance(groups, list), "camera_groups", "expected an array")
    membership, seen = {}, set()
    for i, group in enumerate(groups):
        loc = f"camera_groups[{i}]"
        fields(group, "camera_group shot_ids reference_shot group_quality evidence_shots note", loc)
        gid = group["camera_group"]
        require(isinstance(gid, str) and bool(re.fullmatch(r"camera_[0-9]{3,}", gid)) and gid not in seen,
                loc, "invalid or duplicate camera_group")
        seen.add(gid)
        members = group["shot_ids"]
        references(members, shot_ids, loc + ".shot_ids")
        require(len(members) >= 2, loc, "camera group needs at least two shots")
        require(group["reference_shot"] in members, loc, "reference_shot must belong to the group")
        label(group["group_quality"], QUALITY, loc + ".group_quality")
        references(group["evidence_shots"], inspected, loc + ".evidence_shots", required=True)
        require(set(members) <= set(group["evidence_shots"]), loc, "every member must be cited as evidence")
        for sid in members:
            require(sid not in membership, loc, f"{sid} already belongs to a camera group")
            membership[sid] = gid
        note(group["note"], loc)
    return membership


def add_camera_groups(candidate, groups, inspected):
    """Add visual group annotations to a v1.0 candidate without changing its spatial claims."""
    require("camera_groups" not in candidate, "candidate", "expected a v1.0 base for incremental grouping")
    membership = camera_membership(groups, {shot["shot_id"] for shot in candidate["shots"]}, inspected)
    return {**candidate, "camera_groups": groups,
            "shots": [{**shot, "camera_group": membership.get(shot["shot_id"])} for shot in candidate["shots"]]}


def validate_candidate(candidate, packet, inspected):
    grouped = isinstance(candidate, dict) and "camera_groups" in candidate
    fields(candidate, "movie_id sequence_id scene_id start_s end_s layouts shots status review_status"
           + (" camera_groups" if grouped else ""), "result")
    for key in IDENTITY_FIELDS:
        require(candidate[key] == packet[key], key, "does not match Stage04.3 preparation")
    label(candidate["status"], {"model_candidate"}, "status")
    label(candidate["review_status"], {"pending"}, "review_status")
    order = {shot["shot_id"]: i for i, shot in enumerate(packet["shots"])}
    require(set(order) <= inspected.keys(), "images", "every selected shot needs at least one inspected image")
    require(isinstance(candidate["layouts"], list), "layouts", "expected an array")
    require(isinstance(candidate["shots"], list), "shots", "expected an array")
    membership = camera_membership(candidate.get("camera_groups", []), order, inspected)
    ranges = {}
    for i, layout in enumerate(candidate["layouts"]):
        loc = f"layouts[{i}]"
        fields(layout, "layout_id start_shot_id end_shot_id reference_shots axis_pair people layout_status note", loc)
        lid = layout["layout_id"]
        require(isinstance(lid, str) and bool(re.fullmatch(r"layout_[0-9]{3,}", lid)) and lid not in ranges,
                loc, "invalid or duplicate layout_id")
        start, end = layout["start_shot_id"], layout["end_shot_id"]
        require(all(isinstance(sid, str) and sid in order for sid in (start, end)) and order[start] <= order[end],
                loc, "invalid layout shot range")
        ranges[lid] = (order[start], order[end])
        references(layout["reference_shots"], inspected, loc + ".reference_shots", required=True)
        axis(layout["axis_pair"], loc + ".axis_pair")
        label(layout["layout_status"], STATUS, loc + ".layout_status")
        note(layout["note"], loc)
        require(isinstance(layout["people"], list), loc, "people must be an array")
        names = set()
        for j, person in enumerate(layout["people"]):
            person_loc = f"{loc}.people[{j}]"
            fields(person, "person position posture facing_to position_quality evidence_shots note", person_loc)
            name = person["person"]
            label(name, PEOPLE | {"UNKNOWN"}, person_loc + ".person")
            require(name == "UNKNOWN" or name not in names, person_loc, "duplicate named person in layout")
            names.add(name)
            position(person["position"], person_loc + ".position")
            label(person["posture"], {"sitting", "standing", "moving", "unknown"}, person_loc + ".posture")
            label(person["facing_to"], PEOPLE | {"OTHER", "UNKNOWN"}, person_loc + ".facing_to")
            label(person["position_quality"], QUALITY, person_loc + ".position_quality")
            require(person["position"] is None or person["position_quality"] != "unknown",
                    person_loc, "a plotted person must have supported position quality")
            claim = (person["position"] is not None or name != "UNKNOWN" or person["posture"] != "unknown"
                     or person["facing_to"] != "UNKNOWN" or person["position_quality"] != "unknown")
            references(person["evidence_shots"], inspected, person_loc + ".evidence_shots", required=claim)
            note(person["note"], person_loc)

    shots = {}
    for i, shot in enumerate(candidate["shots"]):
        loc = f"shots[{i}]"
        fields(shot, "shot_id layout_id camera_position camera_subject axis_pair axis_reference_shot axis_side "
                    "camera_quality layout_status evidence_shots note" + (" camera_group" if grouped else ""), loc)
        sid, lid = shot["shot_id"], shot["layout_id"]
        require(isinstance(sid, str) and sid in order and sid not in shots, loc, "unknown or duplicate shot ID")
        require(shot.get("camera_group") == membership.get(sid), sid, "camera_group disagrees with camera_groups")
        shots[sid] = shot
        if lid is not None:
            require(isinstance(lid, str) and lid in ranges, sid, "unknown layout_id")
            start, end = ranges[lid]
            require(start <= order[sid] <= end, sid, "shot outside its layout range")
        position(shot["camera_position"], sid + ".camera_position")
        label(shot["camera_subject"], PEOPLE | {"GROUP", "OTHER", "UNKNOWN"}, sid + ".camera_subject")
        axis(shot["axis_pair"], sid + ".axis_pair")
        label(shot["axis_side"], {"same", "opposite", "on_axis", "unknown"}, sid + ".axis_side")
        label(shot["camera_quality"], QUALITY, sid + ".camera_quality")
        label(shot["layout_status"], STATUS, sid + ".layout_status")
        require(shot["camera_position"] is None or (lid is not None and shot["camera_quality"] != "unknown"),
                sid, "a plotted camera needs a layout and supported camera quality")
        ref = shot["axis_reference_shot"]
        require(ref is None or (isinstance(ref, str) and ref in inspected and ref != sid),
                sid, "axis reference must be a different inspected shot")
        note(shot["note"], sid)
        claim = (shot["camera_subject"] != "UNKNOWN" or shot["camera_quality"] != "unknown"
                 or shot["axis_pair"] is not None or shot["layout_status"] != "unknown" or lid is not None
                 or ref is not None)
        references(shot["evidence_shots"], inspected, sid + ".evidence_shots", required=claim)
        require(not claim or sid in shot["evidence_shots"], sid, "shot claim requires its own visual evidence")
    require(shots.keys() == order.keys(), "shots", "expected exactly every selected shot")
    for sid, shot in shots.items():
        if shot["axis_side"] != "unknown":
            ref = shot["axis_reference_shot"]
            # Comparisons need a reference record declaring its own axis.
            pair = shot["axis_pair"]
            reference_pair = shots.get(ref, {}).get("axis_pair")
            require(pair is not None and reference_pair is not None and set(pair) == set(reference_pair),
                    sid, "known axis side requires a reference shot using the same pair")
            require(ref in shot["evidence_shots"], sid, "axis reference must be cited as evidence")
