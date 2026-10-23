"""Conservative, auditable complete-link identity candidates; no verified names."""
from dataclasses import asdict, dataclass

import numpy as np

from .identity_model import normalize


@dataclass(frozen=True)
class AssociationConfig:
    same_cosine: float = .55
    different_cosine: float = .25
    margin: float = .05
    same_sample_q10: float = .45
    different_sample_q90: float = .40
    consistency_median: float = .65
    consistency_q10: float = .50
    min_samples: int = 3

    def validate(self):
        values = asdict(self)
        if not all(np.isfinite(x) for x in values.values()):
            raise ValueError("Nonfinite association parameters")
        if not -1 <= self.different_cosine < self.same_cosine <= 1 or not 0 <= self.margin < 1:
            raise ValueError("Invalid similarity thresholds/margin")
        if self.same_cosine+self.margin > 1 or self.different_cosine-self.margin < -1 or self.min_samples < 2:
            raise ValueError("Unusable association thresholds/sample count")
        if any(not -1 <= v <= 1 for k,v in values.items() if k not in {"margin","min_samples"}):
            raise ValueError("Cosine thresholds outside [-1,1]")


def representation(track_id, vectors, model_ids, config):
    config.validate()
    if len(model_ids) != len(vectors) or (len(vectors) and len(set(model_ids)) != 1):
        raise ValueError("Cannot mix different embedding model weights in a track")
    if len(vectors) == 0:
        return {"track_id": track_id, "sample_count": 0, "status": "insufficient_evidence", "eligible": False, "review_needed": True}, None
    dimension = np.asarray(vectors[0]).size
    samples = np.stack([normalize(v, dimension) for v in vectors])
    mean = samples.mean(axis=0)
    center = normalize(mean, dimension) if np.linalg.norm(mean) >= 1e-8 else None
    matrix = np.clip(samples@samples.T, -1, 1)
    pairs = matrix[np.triu_indices(len(samples), 1)]
    consistency = {"mean": float(pairs.mean()) if len(pairs) else None,
                   "median": float(np.median(pairs)) if len(pairs) else None,
                   "q10": float(np.quantile(pairs,.1)) if len(pairs) else None,
                   "minimum": float(pairs.min()) if len(pairs) else None,
                   "minimum_sample_to_centroid": float((samples@center).min()) if center is not None else None}
    enough = len(samples) >= config.min_samples
    coherent = center is not None and len(pairs) and consistency["median"] >= config.consistency_median and consistency["q10"] >= config.consistency_q10
    status = "consistent_candidate" if enough and coherent else "inconsistent_track" if enough else "insufficient_evidence"
    return {"track_id": track_id, "model_sha256": model_ids[0], "sample_count": len(samples),
            "embedding_dimension": dimension, "internal_consistency": consistency, "status": status,
            "eligible": bool(enough and coherent), "review_needed": True}, center


def associate(track_info, sample_vectors, track_vectors, config):
    config.validate()
    ids = sorted(track_info)
    model_ids = {t["model_sha256"] for t in track_info.values() if t["sample_count"]}
    if len(model_ids) > 1:
        raise ValueError("Cannot associate embeddings from different model weights")
    pairs = []
    matrix = np.full((len(ids),len(ids)), np.nan)
    decisions = {}
    for i, aid in enumerate(ids):
        if aid in track_vectors:
            matrix[i,i] = 1.
        for j in range(i+1,len(ids)):
            bid = ids[j]
            a,b = track_info[aid],track_info[bid]
            pair = {"track_a": aid, "track_b": bid, "sample_count_a": a["sample_count"], "sample_count_b": b["sample_count"],
                    "consistency_a": a.get("internal_consistency"), "consistency_b": b.get("internal_consistency"),
                    "review_needed": True, "candidate_condition_met": False, "cosine_similarity": None}
            if aid in track_vectors and bid in track_vectors:
                cosine = float(np.clip(track_vectors[aid]@track_vectors[bid],-1,1))
                cross = np.clip(sample_vectors[aid]@sample_vectors[bid].T,-1,1)
                matrix[i,j] = matrix[j,i] = cosine
                pair.update(cosine_similarity=cosine, cross_sample_mean=float(cross.mean()), cross_sample_median=float(np.median(cross)),
                            cross_sample_q10=float(np.quantile(cross,.1)), cross_sample_q90=float(np.quantile(cross,.9)),
                            cross_sample_min=float(cross.min()), cross_sample_max=float(cross.max()),
                            same_threshold_margin=cosine-config.same_cosine,
                            different_threshold_margin=config.different_cosine-cosine)
            if not a["eligible"] or not b["eligible"]:
                decision = "insufficient_evidence"
            elif a["shot_id"] == b["shot_id"]:
                decision = "ambiguous"  # Pilot associates across shots only.
            elif pair["cosine_similarity"] >= config.same_cosine+config.margin and pair["cross_sample_q10"] >= config.same_sample_q10:
                decision = "same_identity_candidate"
                pair["candidate_condition_met"] = True
            elif pair["cosine_similarity"] <= config.different_cosine-config.margin and pair["cross_sample_q90"] <= config.different_sample_q90:
                decision = "different_identity_candidate"
            else:
                decision = "ambiguous"
            pair["decision"] = decision
            decisions[frozenset((aid,bid))] = decision
            pairs.append(pair)
    clusters = [{tid} for tid in ids]
    rejected = []
    proposed = sorted([p for p in pairs if p["decision"] == "same_identity_candidate"],
                      key=lambda p: (-p["cosine_similarity"],p["track_a"],p["track_b"]))
    for pair in proposed:
        left = next(c for c in clusters if pair["track_a"] in c)
        right = next(c for c in clusters if pair["track_b"] in c)
        if left is right:
            continue
        conflicts = [[a,b] for a in sorted(left) for b in sorted(right)
                     if decisions[frozenset((a,b))] != "same_identity_candidate"]
        if conflicts:
            rejected.append({"proposed_pair": [pair["track_a"],pair["track_b"]], "conflicting_pairs": conflicts,
                             "reason": "complete_link_consistency_failed; transitive evidence is not sufficient"})
        else:
            clusters.remove(left)
            clusters.remove(right)
            clusters.append(left|right)
    result = []
    for index, members in enumerate(sorted(clusters,key=lambda c:sorted(c)),1):
        supporting = [p for p in pairs if p["track_a"] in members and p["track_b"] in members]
        conflicts = [r for r in rejected if set(r["proposed_pair"]) & members]
        result.append({"cluster_id": f"identity_{index:03d}", "track_ids": sorted(members),
                       "status": "tentative_identity_cluster" if len(members)>1 else "unknown",
                       "verified_identity": False, "human_verified": False, "review_needed": True,
                       "supporting_pairs": supporting, "rejected_merge_conflicts": conflicts,
                       "uncertain_factors": ["Uncalibrated research thresholds", "No human identity ground truth", "Boundary correction is experimentally model reviewed"],
                       "character_candidates": []})
    return result, pairs, matrix, rejected
