"""Stage06.3 v1.3 target hypotheses on frozen Look States."""

from .inference import CHARACTERS, MIN_GROUP_FRACTION
from .look_targets import geometry_for

PHYSICAL_TARGETS = (*CHARACTERS, "AWAY", "OTHER")
VISUAL_LEVELS = {"context_only", "visual_directional", "visual_endpoint", "insufficient"}
MEDIAN_MARGIN_DEG = 15.0
STATE_KEYS = ("state_id", "shot_id", "track_id")


def ranked_targets(row):
    return {target for tier in row["target_ranking"] for target in tier}


def decision(preferred, alternatives, level, reason, *, supported=False, selected="UNKNOWN"):
    alternatives = list(dict.fromkeys(target for target in alternatives if target != preferred))
    if preferred:
        status = "supported_hypothesis" if supported else "preferred_but_unconfirmed"
    elif len(alternatives) > 1 and level != "insufficient":
        status = "ambiguous"
    else:
        status = "insufficient_evidence"
    ranking = [[preferred]] if preferred else []
    if alternatives:
        ranking.append(alternatives)
    return {
        "preferred_target": preferred,
        "alternative_targets": alternatives,
        "selected_target": selected,
        "evidence_level": level,
        "decision_status": status,
        "reason": reason,
        "target_ranking": ranking,
    }


def visual_prediction(observation):
    preferred = observation["preferred_target"]
    direct = observation["evidence_level"] == "visual_endpoint"
    return {
        **decision(preferred, observation["alternative_targets"], observation["evidence_level"],
                   observation["reason"], supported=direct, selected=preferred if direct else "UNKNOWN"),
        "observation": observation,
    }


def compatible(fit):
    return any(branch["compatible"] > 0 for branch in fit["camera_branches"].values())


def substantial(fit):
    return any(branch["tested"] and branch["compatible"] >= MIN_GROUP_FRACTION * branch["tested"]
               for branch in fit["camera_branches"].values())


def leads_in_branches(fit, rival):
    aligned = fit["camera_branches"]
    competing = rival["camera_branches"]
    return bool(aligned) and aligned.keys() == competing.keys() and all(
        branch["compatible"] >= competing[name]["compatible"]
        and competing[name]["median_error_deg"] - branch["median_error_deg"] >= MEDIAN_MARGIN_DEG
        for name, branch in aligned.items()
    )


def geometry_prediction(state, mapping, camera_inputs):
    gazers = sorted(set(mapping["character_candidates"]) & set(CHARACTERS))
    # Replay the spatial matcher without v1.1 context priors or target observations.
    results = {actor: geometry_for({**state, "character": actor}, *camera_inputs, [])
               for actor in gazers}
    fits_by_gazer = {
        actor: (result["geometry"] or {}).get("tested_targets", [])
        for actor, result in results.items()
    }
    possible = {fit["target"] for fits in fits_by_gazer.values() for fit in fits if compatible(fit)}
    good_by_gazer = [
        {fit["target"] for fit in fits if substantial(fit)} for fits in fits_by_gazer.values()
    ]
    strong = set.intersection(*good_by_gazer) if good_by_gazer else set()
    preferred = None
    selected = "UNKNOWN"
    if len(gazers) == 1:
        actor = gazers[0]
        fits = fits_by_gazer[actor]
        # A lead in each modeled branch is a ranking signal. The previous, stricter
        # all-branch rule remains the prerequisite for a geometry-only selection.
        if len(fits) > 1:
            preferred = next((
                fit["target"] for fit in fits if fit["target"] in strong
                and all(leads_in_branches(fit, rival) for rival in fits if rival is not fit)
            ), None)
        if (preferred and state["qc_status"] == "stable" and mapping["mapping_strength"] == "supported"
                and results[actor]["selected_target"] == preferred):
            selected = preferred
    measured_level = "head_supported" if state["direction_source"] == "head_only" else "gaze_supported"
    level = measured_level if strong else "insufficient"
    if not state["look_direction"]:
        reason = "冻结状态缺少持续可用的相机 Gaze 或头向。"
    elif not any(fits_by_gazer.values()):
        reason = "缺少可用于比较人物方向的平面机位或朝向约束。"
    elif preferred:
        reason = f"{preferred} 在可用机位分支中的角差均领先，且至少一个分支有足够相容样本。机位与俯仰仍未标定。"
    elif not strong:
        reason = "可用机位假设不足以稳定支持目标排序。"
    else:
        reason = "身份候选或机位分支未给出足以区分人物的共同方向。"
    if preferred and state["qc_status"] == "heterogeneous_stable_support":
        reason += " 冻结代表值来自含多个方向的状态。"
    return {
        **decision(preferred, sorted(possible), level, reason,
                   supported=selected != "UNKNOWN", selected=selected),
        "gazer_candidates": gazers,
        "mapping_strength": mapping["mapping_strength"],
        "substantially_compatible_targets": sorted(strong),
        "spatial_by_gazer": results,
    }


def assess_conflicts(visual, spatial):
    preferred = visual["preferred_target"]
    geometric = spatial["preferred_target"]
    if preferred and geometric:
        comparison = "agree" if preferred == geometric else "conflict"
    else:
        comparison = "unpaired"
    conflicts = []
    if comparison == "conflict":
        conflicts.append({"kind": "preferred_targets_disagree", "target": preferred,
                          "reason": f"视觉优先 {preferred}，几何排序优先 {geometric}。"})
    fits_by_target = {}
    for result in spatial["spatial_by_gazer"].values():
        for fit in (result["geometry"] or {}).get("tested_targets", []):
            fits_by_target.setdefault(fit["target"], []).append(fit)
    for target in sorted(ranked_targets(visual) & fits_by_target.keys()):
        fits = fits_by_target[target]
        if not any(compatible(fit) for fit in fits):
            conflicts.append({"kind": "visual_candidate_outside_geometry", "target": target,
                              "reason": f"视觉提出 {target}，但其所有已测平面机位假设均不相容。"})
        elif target == preferred and not any(substantial(fit) for fit in fits):
            conflicts.append({"kind": "visual_preference_weak_geometry", "target": target,
                              "reason": f"视觉偏好 {target}，但没有机位分支达到相容样本比例阈值。"})
    return comparison, conflicts


def fused_prediction(state, visual, spatial):
    visual_target = visual["preferred_target"]
    geometry_target = spatial["preferred_target"]
    visual_candidates = ranked_targets(visual)
    candidates = visual_candidates | ranked_targets(spatial)
    directional_visual = visual["evidence_level"] in {"visual_directional", "visual_endpoint"}
    comparison, conflicts = assess_conflicts(visual, spatial)
    preferred = visual_target
    reason = visual["reason"]
    if comparison == "agree":
        reason += f" 几何排序也优先 {preferred}。"
        if not directional_visual:
            reason += " 视觉依据为场景上下文，眼睛落点在画面中不可读。"
    elif (comparison == "conflict" and not directional_visual and state["qc_status"] == "stable"
          and spatial["evidence_level"] == "gaze_supported" and spatial["mapping_strength"] == "supported"):
        preferred = geometry_target
        reason += f" 融合优先采用稳定 Gaze 支持的 {preferred}，同时保留视觉上下文候选。"
    elif not preferred and geometry_target:
        head_only_tie = state["direction_source"] == "head_only" and len(visual_candidates) > 1
        outside_visual = visual_candidates and geometry_target not in visual_candidates
        if head_only_tie or outside_visual:
            reason += f" 几何偏向 {geometry_target}，融合保留画面候选的歧义。"
        else:
            preferred = geometry_target
            reason += f" 几何排序支持把 {preferred} 放在首位。"
    if conflicts:
        reason += " " + " ".join(item["reason"] for item in conflicts)
    if preferred and state["qc_status"] == "heterogeneous_stable_support":
        reason += " 状态内仍有多个稳定方向，几何结果反映冻结方向代表值。"
    strong = set(spatial["substantially_compatible_targets"])
    measured = ((preferred in strong) if preferred else bool(candidates & strong))
    level = spatial["evidence_level"] if measured else visual["evidence_level"]
    supported = bool(comparison == "agree" and directional_visual and measured
                     and state["qc_status"] == "stable")
    confirmed = bool(supported and visual["selected_target"] == spatial["selected_target"] == preferred
                     and candidates == {preferred})
    return {
        **decision(preferred, sorted(candidates), level, reason, supported=supported,
                   selected=preferred if confirmed else "UNKNOWN"),
        "visual_geometry_preference": comparison,
        "conflicts": conflicts,
    }
