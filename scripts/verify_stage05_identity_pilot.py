"""Independently recompute saved identity evidence; never decode or modify sources."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from oscardp.stage05.identity_inputs import file_hash, load_corrected


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    config = json.loads((root/"run_config.json").read_text())
    result = json.loads((root/"identity_clusters.json").read_text())
    qc = json.loads((root/"qc_report.json").read_text())
    rows = [json.loads(line) for line in (root/"embedding_samples.jsonl").read_text().splitlines()]
    source, tracks, observations, files = load_corrected(Path(config["source_corrected_run"]),
                                                       config["movie_id"], config["selected_shot_ids"])
    require(files == config["source_files"] == result["source_files"], "Source provenance changed")
    require(source["boundary_sidecar"] == config["boundary_sidecar"] == result["boundary_sidecar"], "Boundary status changed")
    require(not result["human_verified"], "Unverified pilot promoted to human verified")
    for path, sha in qc["protected_source_sha256"].items():
        require(file_hash(Path(path)) == sha, f"Protected source changed: {path}")
    model_sha = file_hash(Path(config["model"]["model_path"]))
    require(model_sha == config["model"]["sha256"] == result["model"]["sha256"], "Model weights changed")
    with np.load(root/"track_embeddings.npz", allow_pickle=False) as data:
        samples = data["sample_embeddings"]
        centers = data["track_embeddings"]
        tids = data["track_ids"].tolist()
        require(str(data["model_sha256"]) == model_sha, "NPZ model mismatch")
        for array in [samples, centers]:
            require(array.ndim == 2 and array.shape[1] == config["model"]["embedding_dimension"], "Embedding dimension mismatch")
            require(np.isfinite(array).all(), "Nonfinite embedding")
            require(np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-6), "Embedding not normalized")
        embedded = [r for r in rows if r["status"] == "embedded"]
        require(len(embedded) == len(samples), "Sample count mismatch")
        require([r["embedding_npz_row"] for r in embedded] == list(range(len(samples))), "NPZ row mapping mismatch")
        require(data["sample_ids"].tolist() == [r["sample_id"] for r in embedded], "Sample ID mismatch")
        require(data["sample_track_ids"].tolist() == [r["track_id"] for r in embedded], "Sample Track mismatch")
        observed = {(tid,r["frame_index"]):r for tid,records in observations.items() for r in records}
        require(len({r["sample_id"] for r in rows}) == len(rows), "Duplicate sample")
        for row in rows:
            original = observed[(row["track_id"], row["frame_index"])]
            require(row["model_sha256"] == model_sha, "Sample model mismatch")
            for key in ["timestamp_sec", "bbox_xywh", "landmarks5", "original_temporal", "effective_processing_temporal"]:
                require(row[key] == original[key], f"Source observation mismatch: {key}")
            if row["status"] == "embedded":
                require(not row["rejection_reasons"], "Rejected sample entered embeddings")
                require(file_hash(root/row["aligned_crop"]) == row["aligned_crop_sha256"], "Crop changed")
            else:
                require(row["embedding_npz_row"] is None, "Bad sample entered embeddings")
        for tid, center in zip(tids, centers):
            vectors = samples[data["sample_track_ids"] == tid]
            mean = vectors.mean(axis=0)
            np.testing.assert_allclose(center, mean/np.linalg.norm(mean), atol=1e-6)
            pairwise = np.clip(vectors@vectors.T, -1, 1)[np.triu_indices(len(vectors), 1)]
            info = result["track_representations"][tid]
            require(len(vectors) == info["sample_count"], "Track count mismatch")
            if len(pairwise):
                np.testing.assert_allclose([info["internal_consistency"][k] for k in ["mean", "median", "q10", "minimum"]],
                                           [pairwise.mean(), np.median(pairwise), np.quantile(pairwise,.1), pairwise.min()], atol=1e-6)
        order = result["similarity_matrix_track_order"]
        saved = np.array(result["cosine_similarity_matrix"], dtype=float)
        for a,b in itertools.product(tids, repeat=2):
            np.testing.assert_allclose(saved[order.index(a),order.index(b)],
                                       np.clip(centers[tids.index(a)]@centers[tids.index(b)],-1,1), atol=1e-6)
        for cluster in result["clusters"]:
            require(not cluster["human_verified"] and not cluster["verified_identity"], "Cluster improperly verified")
            members = set(cluster["track_ids"])
            support = cluster["supporting_pairs"]
            require(len(support) == len(members)*(len(members)-1)//2, "Cluster lacks complete-link evidence")
            for pair in support:
                a,b = pair["track_a"],pair["track_b"]
                require(a in members and b in members and pair["decision"] == "same_identity_candidate", "Cluster conflict")
                cross = np.clip(samples[data["sample_track_ids"]==a]@samples[data["sample_track_ids"]==b].T,-1,1)
                np.testing.assert_allclose([pair["cross_sample_mean"],pair["cross_sample_q10"],pair["cross_sample_q90"]],
                                           [cross.mean(),np.quantile(cross,.1),np.quantile(cross,.9)],atol=1e-6)
    return {"status":"passed", "source_hashes_unchanged":True, "model_sha256":model_sha,
            "sample_count":len(embedded), "track_count":len(tracks), "cluster_count":len(result["clusters"]),
            "checks":["corrected provenance", "original/effective temporal and global observation indices",
                      "model hash", "finite normalized vectors and NPZ mapping", "saved crop hashes",
                      "normalized track means", "internal consistency", "full cosine matrix", "complete-link cluster support"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("--output", type=Path, help="New verification JSON; existing files are refused")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Verification output already exists")
    report = verify(args.pilot_dir)
    payload = json.dumps(report, indent=2)+"\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(payload)
    print(payload)
