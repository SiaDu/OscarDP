"""Verify source provenance and every OpenFace CSV → JSON/Parquet frame reference."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from oscardp.stage05.identity_inputs import file_hash
from oscardp.stage05.output import identity
from oscardp.stage05.performance_inputs import load_inputs
from oscardp.stage05.performance_csv import parse_csv


def require(condition,message):
    if not condition: raise ValueError(message)


def verify(root):
    config = json.loads((root/"run_config.json").read_text())
    measurements = json.loads((root/"track_measurements.json").read_text())
    qc = json.loads((root/"qc_report.json").read_text())
    provenance = config["source_provenance"]
    source,tracks,observations,identities,current = load_inputs(Path(provenance["tracking_run"]),Path(provenance["identity_run"]),
                                                             config["movie_id"],config["selected_shot_ids"])
    require(current == provenance,"Source provenance changed")
    for path,sha in provenance["protected_source_sha256"].items():
        require(file_hash(Path(path))==sha,f"Protected input changed: {path}")
    require(identity(Path(provenance["video"]["path"])) == provenance["video"],"Video identity changed")
    require(config["boundary_sidecar"] == source["boundary_sidecar"] == qc["boundary_sidecar"] == measurements["boundary_sidecar"],
            "Boundary review state changed")
    table = pq.read_table(root/"frame_features.parquet")
    flat = table.to_pylist()
    status = [json.loads(line) for line in (root/"frame_status.jsonl").read_text().splitlines()]
    require(len(flat) == len(status) == qc["measurement_rows"],"JSON/Parquet row count mismatch")
    require(table.schema.metadata[b"main_au_mode"].decode() == config["main_au_mode"],"Main mode mismatch")
    maps = {sid:json.loads((root/f"frame_maps/{sid}.json").read_text()) for sid in config["selected_shot_ids"]}
    lookup = {}
    for summary in measurements["tracks"]:
        sid,tid,mode = summary["shot_id"],summary["track_id"],summary["extraction_mode"]
        require(tid in tracks and tracks[tid]["shot_id"]==sid,"Track mismatch")
        for key in ["original_temporal","effective_processing_temporal"]:
            require(summary[key] == tracks[tid][key] == maps[sid][key],"Track range mismatch")
        path = root/summary["raw_csv"]
        require(file_hash(path)==summary["execution"]["raw_csv_sha256"],"Raw OpenFace CSV changed")
        header,parsed = parse_csv(path,len(maps[sid]["frames"]),source["fps"])
        lookup[sid,mode] = header,parsed,summary
    seen = set()
    meta_names = {"frame","face_id","timestamp","confidence","success"}
    meta_map = {"face_id":"openface_face_id","timestamp":"openface_local_timestamp_sec",
                "confidence":"openface_confidence","success":"openface_success_raw"}
    for index,(row,state) in enumerate(zip(flat,status)):
        require(row["parquet_row"]==state["parquet_row"]==index,"Parquet row reference mismatch")
        for key in ["movie_id","shot_id","track_id","global_frame_index","movie_timestamp_sec","openface_local_frame","extraction_mode",
                    "identity_cluster_id","identity_status","identity_human_verified"]:
            require(row[key] == state[key],f"JSON/Parquet mismatch: {key}")
        sid,tid,mode = row["shot_id"],row["track_id"],row["extraction_mode"]
        key = (sid,mode,row["global_frame_index"])
        require(key not in seen,"Duplicate measurement frame")
        seen.add(key)
        local = row["openface_local_frame"]
        mapped = maps[sid]["frames"][local-1]
        require(mapped["global_frame_index"] == row["global_frame_index"] == maps[sid]["effective_processing_temporal"]["start_frame"]+local-1,
                "Off-by-one global frame mapping")
        require(mapped["movie_timestamp_sec"]==row["movie_timestamp_sec"] and mapped["source_next_timestamp_sec"]==row["source_next_timestamp_sec"],
                "Source movie PTS replaced with local time")
        require(all(row[k]==v for k,v in identities[tid].items()),"Tentative identity state changed")
        header,parsed,_ = lookup[sid,mode]
        raw = parsed.get(local)
        require(state["reliability"]["openface_row_present"] == (raw is not None),"Missing CSV row silently filled")
        for name in header:
            if name in meta_names: continue
            require(row[name] == (raw[name] if raw else None),f"Raw measurement changed or interpolated: {name}")
        for raw_key,saved_key in meta_map.items():
            require(row[saved_key] == (raw[raw_key] if raw else None),"Raw OpenFace metadata mismatch")
    expected = {(sid,mode,frame["global_frame_index"]) for sid,data in maps.items()
                for mode in config["au_modes"] for frame in data["frames"]}
    require(seen == expected,"Missing/extra expected frame or mode")
    for (sid,mode),(_,_,summary) in lookup.items():
        subset = [s for s in status if s["shot_id"]==sid and s["extraction_mode"]==mode]
        require(len(subset)==summary["expected_frame_count"],"Track count mismatch")
        for key,count in summary["valid_counts"].items():
            require(sum(s["reliability"][key] for s in subset)==count,"QC count mismatch")
    return {"status":"passed","measurement_rows":len(flat),"unique_source_frames":len({r["global_frame_index"] for r in flat}),
            "track_count":len(tracks),"modes":config["au_modes"],"protected_source_files":len(provenance["protected_source_sha256"]),
            "source_hashes_unchanged":True,"human_verified":False,
            "checks":["corrected/tentative source hashes and ranges", "no omitted expected frame",
                      "exact one-based-local/zero-based-global mapping", "movie timestamp kept separate from transport clock",
                      "every raw OpenFace feature equals Parquet value or explicit missing null",
                      "JSON/Parquet identity and frame references", "separate AU modes", "QC count recomputation"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir",type=Path)
    parser.add_argument("--output",type=Path,help="New verification JSON; existing output is refused")
    args = parser.parse_args()
    if args.output and args.output.exists(): parser.error("Verification output already exists")
    result = verify(args.pilot_dir)
    payload = json.dumps(result,indent=2)+"\n"
    if args.output:
        with args.output.open("x") as stream: stream.write(payload)
    print(payload)
