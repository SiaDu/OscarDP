"""Stage05.3 pilot from corrected real observations, no new face detection."""
import csv
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from oscardp.shots.media import probe_video
from oscardp.shots.schema import json_dumps
from .identity_association import AssociationConfig, representation, associate
from .identity_inputs import file_hash, load_corrected
from .identity_model import ArcFaceRecognizer
from .identity_sampling import candidates, crop_quality
from .identity_qc import contact_sheets
from .output import atomic_json, identity
from .video_reader import iter_frames


def run(args):
    started = time.perf_counter()
    # Refuse before model load, decoding, or any mutation of a completed pilot.
    if args.output_dir.exists():
        raise FileExistsError(f"Identity pilot output already exists; select a new --output-dir: {args.output_dir}")
    if args.output_dir.resolve().is_relative_to(args.source_run.resolve()):
        raise ValueError("Identity output must be independent of the corrected Stage05.2 run")
    source, tracks, records, inputs = load_corrected(args.source_run,args.movie_id,args.shot_ids)
    config = AssociationConfig(same_cosine=args.same_cosine,different_cosine=args.different_cosine,margin=args.margin,
                               same_sample_q10=args.same_sample_q10, consistency_median=args.consistency_median,
                               consistency_q10=args.consistency_q10,min_samples=args.min_samples)
    config.validate()
    if not 2 <= args.max_samples <= 16:
        raise ValueError("--max-samples must be between 2 and 16")
    if (not all(np.isfinite(v) for v in [args.min_detection_score, args.min_face_size, args.min_sharpness])
            or not 0 < args.min_detection_score <= 1 or args.min_face_size <= 0 or args.min_sharpness < 0):
        raise ValueError("Invalid crop quality parameters")
    video = Path(source["video"]["path"])
    metadata,_ = probe_video(video,video.name)
    if (metadata.width,metadata.height) != (source["width"],source["height"]) or abs(metadata.fps-source["fps"]) > 1e-6:
        raise ValueError("Stage05.2 video metadata mismatch")
    cv2.setNumThreads(4)
    model = ArcFaceRecognizer(args.model,args.provider)
    chosen = [(tid,r) for tid in sorted(tracks) for r in candidates(records[tid],args.max_samples)]
    request = {}
    for tid,r in chosen:
        request.setdefault(r["frame_index"],[]).append((tid,r))
    snapshots = {str(Path(v["path"])):v["sha256"] for v in inputs.values()}
    source_dir = Path(source["stage01"]["path"]).parent
    for path in sorted(source_dir.glob(f"Stage0[1-4]_{args.movie_id}.json")):
        snapshots[str(path)] = file_hash(path)
    snapshots[source["boundary_sidecar"]["path"]] = file_hash(Path(source["boundary_sidecar"]["path"]))
    run_config = {"schema_version":"stage05.3.identity.v1", "movie_id":args.movie_id,
                  "source_corrected_run":str(args.source_run.resolve()), "source_files":inputs,
                  "boundary_sidecar":source["boundary_sidecar"], "video":source["video"],
                  "selected_shot_ids":args.shot_ids, "track_ids":sorted(tracks),
                  "source_shot_ranges":source["shots"], "model":model.info, "association_parameters":asdict(config),
                  "threshold_status":"uncalibrated conservative research candidate gates; not a verified identity threshold, not copied from SFace",
                  "margin_semantics":"minimum distance above same_cosine or below different_cosine; a rejection band is retained",
                  "sampling":{"max_candidates_per_track":args.max_samples,"method":"one high-confidence observation per equally spaced temporal bin"},
                  "quality_parameters":{"min_detection_score":args.min_detection_score,"min_native_face_size":args.min_face_size,
                                        "min_aligned_sharpness":args.min_sharpness,"max_alignment_rmse_112":10.,"min_crop_source_fraction":.95},
                  "alignment_reference":"InsightFace ArcFace 112 five-point template; no face enhancement, new detection, head-pose or occlusion inference",
                  "scope":"nameless tentative identity candidates only; no Stage02/03/04 name or speaker features used"}
    if args.dry_run:
        print(json_dumps(dict(run_config,candidate_count=len(chosen),dry_run=True),pretty=True))
        return 0
    root = args.output_dir
    root.mkdir(parents=True,exist_ok=False)
    (root/"previews/crops").mkdir(parents=True)
    atomic_json(root/"run_config.json",run_config)
    rows = []
    vectors = []
    sample_ids = []
    sample_tracks = []
    by_track = {tid:[] for tid in tracks}
    decode_sec = inference_sec = quality_sec = 0.
    windows = [{"temporal":{"start_frame":n,"end_frame":n+1,"end":(n+1)/metadata.fps}} for n in sorted(request)]
    stream = iter_frames(video,windows,metadata,source["timestamp_rule"]=="frame_index_cfr")
    seen = set()
    try:
        while True:
            tick = time.perf_counter()
            frame = next(stream,None)
            decode_sec += time.perf_counter()-tick
            if frame is None:
                break
            seen.add(frame.index)
            for tid,record in request[frame.index]:
                if abs(record["timestamp_sec"]-frame.timestamp) > 1e-6:
                    raise ValueError("Identity source frame timestamp does not match the actual Stage05.2 observation")
                sample_id = f"{tid}_frame_{frame.index:08d}"
                tick = time.perf_counter()
                crop, quality, reasons = crop_quality(frame.image,record,args.min_detection_score,args.min_face_size,args.min_sharpness)
                quality_sec += time.perf_counter()-tick
                row = {"sample_id":sample_id,"track_id":tid,"shot_id":record["shot_id"],"frame_index":frame.index,
                       "timestamp_sec":frame.timestamp,"bbox_xywh":record["bbox_xywh"],"landmarks5":record["landmarks5"],
                       "original_temporal":record["original_temporal"],"effective_processing_temporal":record["effective_processing_temporal"],
                       "embedding_model":model.info["name"],"model_sha256":model.info["sha256"],"quality_metrics":quality,
                       "status":"quality_rejected" if reasons else "embedded","rejection_reasons":reasons,
                       "embedding_npz_row":None}
                if not reasons:
                    tick = time.perf_counter()
                    try:
                        vector = model.embed(crop)
                    except ValueError as exc:
                        row.update(status="embedding_rejected",rejection_reasons=[str(exc)])
                    else:
                        index = len(vectors)
                        row["embedding_npz_row"] = index
                        row["aligned_crop"] = f"previews/crops/{sample_id}.png"
                        if not cv2.imwrite(str(root/row["aligned_crop"]),crop):
                            raise RuntimeError("Cannot save aligned identity evidence")
                        row["aligned_crop_sha256"] = file_hash(root/row["aligned_crop"])
                        row["embedding_dimension"] = len(vector)
                        row["embedding_l2_norm"] = float(np.linalg.norm(vector))
                        vectors.append(vector)
                        by_track[tid].append(index)
                        sample_ids.append(sample_id)
                        sample_tracks.append(tid)
                    inference_sec += time.perf_counter()-tick
                rows.append(row)
                print(f"identity {len(rows)}/{len(chosen)} {tid} frame={frame.index} {row['status']}",flush=True)
        if seen != set(request):
            raise ValueError("Identity candidate frame decode was incomplete")
        samples = np.stack(vectors) if vectors else np.empty((0,model.dimension),np.float32)
        track_info, track_vectors, sample_vectors = {},{},{}
        for tid in sorted(tracks):
            indices = by_track[tid]
            info,center = representation(tid,[vectors[i] for i in indices],[model.info["sha256"]]*len(indices),config)
            info.update(shot_id=tracks[tid]["shot_id"],candidate_count=sum(r["track_id"]==tid for r in rows),
                        original_temporal=tracks[tid]["original_temporal"],effective_processing_temporal=tracks[tid]["effective_processing_temporal"])
            track_info[tid] = info
            if center is not None:
                track_vectors[tid] = center
                sample_vectors[tid] = samples[indices]
        clusters,pairs,matrix,rejected = associate(track_info,sample_vectors,track_vectors,config)
        track_order = sorted(tracks)
        represented = sorted(track_vectors)
        np.savez_compressed(root/"track_embeddings.npz",sample_embeddings=samples,
                            track_embeddings=np.stack([track_vectors[t] for t in represented]) if represented else np.empty((0,model.dimension),np.float32),
                            sample_ids=np.asarray(sample_ids,dtype="U120"),sample_track_ids=np.asarray(sample_tracks,dtype="U80"),
                            track_ids=np.asarray(represented,dtype="U80"),model_sha256=np.asarray(model.info["sha256"]))
        with (root/"embedding_samples.jsonl").open("w") as out:
            for row in rows:
                out.write(json_dumps(row)+"\n")
        with (root/"pairwise_similarity.csv").open("w",newline="") as out:
            fields = ["track_a","track_b","cosine_similarity","sample_count_a","sample_count_b","cross_sample_mean",
                      "cross_sample_median","cross_sample_q10","cross_sample_q90","cross_sample_min","cross_sample_max",
                      "same_threshold_margin","different_threshold_margin","consistency_a","consistency_b",
                      "candidate_condition_met","decision","review_needed"]
            writer = csv.DictWriter(out,fieldnames=fields)
            writer.writeheader()
            for pair in pairs:
                writer.writerow({key:json_dumps(value) if isinstance(value,dict) else value for key,value in pair.items()})
        atomic_json(root/"identity_clusters.json",{"schema_version":"stage05.3.identity.v1","movie_id":args.movie_id,
                    "source_corrected_run":str(args.source_run.resolve()),"source_files":inputs,"model":model.info,
                    "boundary_sidecar":source["boundary_sidecar"],"human_verified":False,"clusters":clusters,
                    "track_representations":track_info,"similarity_matrix_track_order":track_order,
                    "cosine_similarity_matrix":[[float(v) if np.isfinite(v) else None for v in row] for row in matrix],
                    "rejected_transitive_merges":rejected})
        contact_sheets(root,clusters,rows,pairs)
        unchanged = all(file_hash(Path(path))==sha for path,sha in snapshots.items())
        if not unchanged or identity(video) != source["video"]:
            raise ValueError("Source input changed during identity processing")
        decisions = Counter(p["decision"] for p in pairs)
        atomic_json(root/"qc_report.json",{"status":"completed_tentative_identity_pilot","model":model.info,
                    "source_corrected_run":str(args.source_run.resolve()),"boundary_sidecar":source["boundary_sidecar"],
                    "candidate_sample_count":len(rows),"embedded_sample_count":len(vectors),
                    "quality_rejections":dict(Counter(reason for r in rows for reason in r["rejection_reasons"])),
                    "track_representations":track_info,"cluster_count":len(clusters),"pair_decisions":dict(decisions),
                    "unknown_cluster_count":sum(c["status"]=="unknown" for c in clusters),
                    "inconsistent_track_count":sum(t["status"]=="inconsistent_track" for t in track_info.values()),
                    "rejected_transitive_merges":rejected,"source_hashes_unchanged":unchanged,"protected_source_sha256":snapshots,
                    "timing":{"total_sec":time.perf_counter()-started,"decode_wait_sec":decode_sec,"quality_sec":quality_sec,"embedding_sec":inference_sec},
                    "limitations":["Candidate associations, not verified identities", "Quality proxies cannot verify occlusion or identity usability",
                                   "Pilot thresholds are configurable and uncalibrated", "No character/actor identity assigned"]})
        print(json_dumps({"clusters":len(clusters),"embedded_samples":len(vectors),"pair_decisions":dict(decisions),"output":str(root)}))
    except (Exception,KeyboardInterrupt) as exc:
        atomic_json(root/"failure.json",{"status":"failed","error":f"{type(exc).__name__}: {exc}","completed_sample_count":len(rows)})
        raise
    finally:
        stream.close()
    return 0
