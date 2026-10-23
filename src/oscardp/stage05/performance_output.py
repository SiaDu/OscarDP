"""One row per expected source frame and extraction mode; no interpolation."""
import json

from .performance_qc import assess


META = {"frame","face_id","timestamp","confidence","success"}


def align_rows(movie_id, mapping, header, parsed, observations, identity_ref, mode, groups, min_confidence):
    if mode not in {"static","dynamic"}:
        raise ValueError("Cannot combine AU modes in a single frame row")
    lookup = {r["frame_index"]:r for r in observations}
    rows = []
    for frame in mapping:
        raw = parsed.get(frame["openface_local_frame"])
        quality = assess(raw,groups,lookup.get(frame["global_frame_index"]),min_confidence)
        features = {name:raw.get(name) if raw else None for name in header if name not in META}
        rows.append({"movie_id":movie_id,"scene_id":None,"scene_id_source":None,
                     "shot_id":frame["shot_id"],"track_id":frame["track_id"],**identity_ref,
                     "global_frame_index":frame["global_frame_index"],"movie_timestamp_sec":frame["movie_timestamp_sec"],
                     "source_next_timestamp_sec":frame["source_next_timestamp_sec"],
                     "openface_local_frame":frame["openface_local_frame"],
                     "openface_local_timestamp_sec":raw["timestamp"] if raw else None,
                     "openface_confidence":raw["confidence"] if raw else None,
                     "openface_face_id":raw["face_id"] if raw else None,
                     "openface_success_raw":raw["success"] if raw else None,
                     "extraction_mode":mode,"features":features,"reliability":quality})
    return rows


def save_frames(root, rows, headers, provenance, main_mode="static"):
    import pyarrow as pa
    import pyarrow.parquet as pq
    features = sorted(set(n for header in headers for n in header if n not in META))
    string_fields = ["movie_id","scene_id","scene_id_source","shot_id","track_id","identity_cluster_id","identity_status","extraction_mode"]
    int_fields = ["global_frame_index","openface_local_frame","openface_face_id","openface_success_raw","parquet_row"]
    float_fields = ["movie_timestamp_sec","source_next_timestamp_sec","openface_local_timestamp_sec","openface_confidence",
                    "landmark_bbox_iou","five_point_normalized_error"]
    bool_fields = ["identity_human_verified","openface_row_present","openface_success","tracking_usable","au_valid",
                   "head_pose_valid","gaze_valid","landmarks_2d_valid","landmarks_3d_valid"]
    schema = pa.schema([pa.field(n,pa.string()) for n in string_fields]+[pa.field(n,pa.int64()) for n in int_fields]
                       +[pa.field(n,pa.float64()) for n in float_fields+features]+[pa.field(n,pa.bool_()) for n in bool_fields],
                       metadata={b"schema_version":b"stage05.4.frame_measurements.v1",
                                 b"source_provenance":json.dumps(provenance,sort_keys=True).encode(),
                                 b"main_au_mode":main_mode.encode(),b"local_clock_is_movie_time":b"false"})
    flat = []
    with (root/"frame_status.jsonl").open("x") as stream:
        for index,row in enumerate(rows):
            row["parquet_row"] = index
            record = {k:row.get(k) for k in string_fields+int_fields+float_fields+bool_fields}
            record.update(row["features"])
            record.update({k:row["reliability"][k] for k in bool_fields if k!="identity_human_verified"})
            agreement = row["reliability"]["association"]
            record.update({k:agreement.get(k) for k in ["landmark_bbox_iou","five_point_normalized_error"]})
            flat.append(record)
            status = {k:v for k,v in row.items() if k!="features"}
            status.update(frame_features_file="frame_features.parquet",source_provenance_file="run_config.json")
            stream.write(json.dumps(status,allow_nan=False)+"\n")
    table = pa.Table.from_pylist(flat,schema=schema)
    pq.write_table(table,root/"frame_features.parquet",compression="zstd")
    check = pq.read_table(root/"frame_features.parquet",columns=["parquet_row","global_frame_index","extraction_mode","movie_timestamp_sec"])
    if check.to_pylist() != [{k:r[k] for k in check.column_names} for r in flat]:
        raise ValueError("JSON/Parquet frame references do not round-trip")
    return {"rows":len(rows),"columns":table.num_columns,"schema":str(schema.remove_metadata()),
            "parquet":"frame_features.parquet","status":"frame_status.jsonl"}
