"""Stage05.4 full-frame OpenFace measurement pilot; no performance events."""
import math
import tempfile
import time
from pathlib import Path

from oscardp.shots.media import probe_video
from oscardp.shots.schema import json_dumps
from .identity_inputs import file_hash
from .output import atomic_json, identity
from .performance_inputs import load_inputs
from .performance_openface import preflight, extract
from .performance_prepare import prepare
from .performance_csv import parse_csv, columns_by_modality
from .performance_qc import summarize
from .performance_output import align_rows, save_frames
from .performance_plots import comparisons, association_sheet


def run(args):
    started = time.perf_counter()
    root = args.output_dir
    if root.exists():
        raise FileExistsError(f"Performance pilot output already exists; choose a new directory: {root}")
    if any(root.resolve().is_relative_to(p.resolve()) for p in [args.tracking_run,args.identity_run]):
        raise ValueError("Performance output must be independent of input runs")
    if not math.isfinite(args.min_confidence) or not 0 <= args.min_confidence <= 1:
        raise ValueError("Invalid OpenFace tracking confidence gate")
    if args.au_mode not in {"static","dynamic","both"}:
        raise ValueError("Explicit AU mode required")
    source,tracks,observations,identities,provenance = load_inputs(args.tracking_run,args.identity_run,args.movie_id,args.shot_ids)
    model = preflight(args.openface_binary)
    video = Path(source["video"]["path"])
    metadata,_ = probe_video(video,video.name)
    if (metadata.width,metadata.height) != (source["width"],source["height"]) or abs(metadata.fps-source["fps"]) > 1e-6:
        raise ValueError("Source video metadata changed since Stage05.2")
    modes = ["static","dynamic"] if args.au_mode=="both" else [args.au_mode]
    camera = {"fx":(500.*metadata.width/640+500.*metadata.height/480)/2,
              "fy":(500.*metadata.width/640+500.*metadata.height/480)/2,
              "cx":metadata.width/2,"cy":metadata.height/2,
              "source":"explicitly recorded OpenFace heuristic, unknown real camera calibration","calibrated":False}
    config = {"schema_version":"stage05.4.frame_measurements.v1","movie_id":args.movie_id,
              "source_provenance":provenance,"selected_shot_ids":args.shot_ids,"track_ids":sorted(tracks),
              "boundary_sidecar":source["boundary_sidecar"],"identity_status":"tentative_unverified",
              "source_shot_ranges":source["shots"],"source_timestamp_rule":source["timestamp_rule"],
              "sample_stride":1,"au_modes":modes,"main_au_mode":"static" if "static" in modes else "dynamic",
              "au_mode_semantics":{"static":"-au_static; static predictor configuration",
                                   "dynamic":"default video predictor configuration; may contain both static and dynamic AU models, with offline video calibration"},
              "camera":camera,"openface_binary":model["binary"],"binary_sha256":model["binary_sha256"],
              "frame_map":"zero-based global frame = corrected start + one-based OpenFace frame - 1; movie timestamps from exact source reader",
              "temporary_media":"full-frame FFV1, exact pixel/hash/order round trip; transport CFR clock is never source time truth; removed after run",
              "qc_parameters":{"tracking_confidence":args.min_confidence,"bbox_iou_min":.25,"five_point_error_max_bbox_diagonal":.18,
                               "au_max_abs_Rx_rad":.8,"au_max_abs_Ry_rad":1.,"gaze_max_abs_Ry_rad":.9,
                               "gaze_min_eye_aspect_proxy":.12,"gaze_min_eye_width_px":10,
                               "gaze_direction_norm_range":[.9,1.1],"long_failure_threshold_frames":12,
                               "jump_thresholds_per_adjacent_frame":{"AU_intensity":2.,"pose_rad":.35,"gaze_angle_rad":.5},
                               "status":"uncalibrated research usability gates; not ground-truth accuracy"},
              "scope":"frame-level measurements only; no interpolation, emotion, onset/apex/offset, performance events or verified names"}
    if args.dry_run:
        print(json_dumps({**config,"preflight":{"model_file_count":len(model["model_manifest"]),
                          "source_build":model["source_build"].get("cmake_declared_version"),
                          "expected_source_frames":sum(s["temporal"]["end_frame"]-s["temporal"]["start_frame"] for s in source["shots"]) }},pretty=True))
        return 0
    root.mkdir(parents=True,exist_ok=False)
    for directory in ["frame_maps","raw_openface","previews","comparisons"]:
        (root/directory).mkdir()
    atomic_json(root/"run_config.json",config)
    atomic_json(root/"model_provenance.json",model)
    all_rows,headers,summaries,commands = [],[],[],[]
    timing = {}
    try:
        with tempfile.TemporaryDirectory(prefix="oscardp-stage054-",dir=args.scratch_root) as temporary:
            scratch = Path(temporary)
            mappings,paths,timing = prepare(video,source["shots"],tracks,metadata,source["timestamp_rule"]=="frame_index_cfr",scratch,root/"frame_maps")
            groups = None
            for mode in modes:
                for sid,path in paths.items():
                    tid = next(tid for tid,t in tracks.items() if t["shot_id"]==sid)
                    output = root/"raw_openface"/mode/sid
                    print(f"OpenFace {mode} {sid}: starting {len(mappings[sid])} frames",flush=True)
                    csv_path,execution = extract(args.openface_binary,path,output,mode,camera)
                    execution.update(shot_id=sid,track_id=tid,raw_csv=str(csv_path.relative_to(root)),raw_csv_sha256=file_hash(csv_path))
                    commands.append(execution)
                    # Persist each real command immediately, even if later parsing/QC fails.
                    atomic_json(root/"openface_executions.json",commands)
                    header,parsed = parse_csv(csv_path,len(mappings[sid]),metadata.fps)
                    current_groups = columns_by_modality(header)
                    if any(not current_groups[key] for key in ["au_intensity","au_occurrence","pose","gaze","landmarks_2d"]):
                        raise ValueError("OpenFace did not emit required measurement columns")
                    if groups is not None and current_groups != groups:
                        raise ValueError("OpenFace modality schema changed between tracks/modes")
                    groups = current_groups
                    rows = align_rows(args.movie_id,mappings[sid],header,parsed,observations[tid],identities[tid],mode,groups,args.min_confidence)
                    summary = summarize(rows,groups)
                    summary.update(shot_id=sid,track_id=tid,**identities[tid],extraction_mode=mode,
                                   original_temporal=tracks[tid]["original_temporal"],
                                   effective_processing_temporal=tracks[tid]["effective_processing_temporal"],
                                   raw_csv=str(csv_path.relative_to(root)),execution=execution)
                    all_rows.extend(rows)
                    headers.append(header)
                    summaries.append(summary)
                    print(f"OpenFace {mode} {sid}: {summary['valid_counts']} in {execution['elapsed_sec']:.2f}s",flush=True)
            files = save_frames(root,all_rows,headers,provenance,config["main_au_mode"])
            comparison,ranked = comparisons(root,all_rows,groups,config["main_au_mode"])
            association_sheet(root,paths,all_rows,modes)
        timing["openface_total_sec"] = sum(c["elapsed_sec"] for c in commands)
        timing["total_sec"] = time.perf_counter()-started
        protected = provenance["protected_source_sha256"]
        unchanged = all(file_hash(Path(path))==sha for path,sha in protected.items())
        if not unchanged or identity(video) != source["video"]:
            raise ValueError("Protected Stage01–05.3 input or source movie identity changed")
        if file_hash(args.openface_binary) != model["binary_sha256"]:
            raise ValueError("OpenFace binary changed during extraction")
        for relative,info in model["model_manifest"].items():
            if file_hash(Path(model["model_root"])/relative) != info["sha256"]:
                raise ValueError("OpenFace model changed during extraction")
        atomic_json(root/"track_measurements.json",{"schema_version":config["schema_version"],"movie_id":args.movie_id,
                    "main_au_mode":config["main_au_mode"],"boundary_sidecar":source["boundary_sidecar"],
                    "source_provenance":provenance,"frames":files,"modalities":groups,"tracks":summaries})
        atomic_json(root/"qc_report.json",{"status":"completed_frame_measurements_with_qc",
                    "unique_source_frame_count":len({r["global_frame_index"] for r in all_rows}),"measurement_rows":len(all_rows),
                    "au_modes":modes,"tracks":summaries,"timing":timing,"source_hashes_unchanged":unchanged,
                    "boundary_sidecar":source["boundary_sidecar"],"major_au_plot_order":ranked,
                    "au_comparison_count":len(comparison),"human_verified":False,
                    "limitations":["Unknown camera intrinsics: pose/gaze are model estimates, not calibrated 3D truth",
                                   "Spatial landmark/YuNet agreement does not verify an actor identity",
                                   "Separate modality usability gates cannot establish AU/gaze ground-truth accuracy",
                                   "Short videos may not support dynamic neutral calibration; static is a repeatable baseline, not declared more accurate",
                                   "No interpolation or emotion/performance events inferred"]})
        print(json_dumps({"output":str(root),"unique_source_frames":len({r["global_frame_index"] for r in all_rows}),
                          "measurement_rows":len(all_rows),"timing":timing}))
    except (Exception,KeyboardInterrupt) as exc:
        atomic_json(root/"failure.json",{"status":"failed","error":f"{type(exc).__name__}: {exc}","executions_completed":len(commands)})
        raise
    return 0
