import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from oscardp.shots import frames_upgrade as upgrade
from oscardp.shots import media, pipeline
from oscardp.shots.frame_samples import (POLICY, atomic_json, image_info, physical_samples,
                                        sample_indices)
from oscardp.shots.frames_upgrade import frames_upgrade, select_movies
from oscardp.shots.schema import FrameTimeline, VideoMetadata, format_timestamp
from oscardp.shots.transnet import sha256_file
from oscardp.shots.validation import validate_movie, validate_output_root


@pytest.mark.parametrize("n,expected", [(1, (0, 0, 0)), (2, (0, 0, 1)),
    (3, (0, 1, 2)), (4, (0, 1, 2)), (10, (0, 4, 8)), (11, (1, 5, 9)),
    (100, (9, 49, 89)), (101, (10, 50, 90))])
def test_sampling(n, expected):
    roles = sample_indices(17, 17 + n)
    assert tuple(roles.values()) == tuple(17 + v for v in expected)
    assert roles["middle"] == (17 + 17 + n - 1) // 2
    physical = physical_samples("shot_000001", 17, 17 + n)
    assert len({v["image_relpath"] for v in physical.values()}) == min(n, 3)
    assert physical["middle"]["image_relpath"].endswith("_middle.jpg")


def test_sampling_many_boundaries():
    for n in range(3, 1000):
        early, middle, late = sample_indices(42, 42+n).values()
        assert 42 <= early < middle < late < 42+n


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    mid = "tt1234567"
    source, workspace, release = [tmp_path / name for name in ("source", "workspace", "fixture_release")]
    source.mkdir()
    video = source / f"{mid}.mkv"
    video.write_bytes(b"source video identity")
    movie = workspace / mid
    images = movie / "shot_img"
    images.mkdir(parents=True)
    internal = workspace / ".internal" / mid
    internal.mkdir(parents=True)
    pts = [i / 25 for i in range(20)]
    timeline = FrameTimeline(pts, [.04]*20, False, 25, .8)
    metadata = VideoMetadata(video.name, .8, 16, 16, "ffv1", 25, 20, False, "frame_index_cfr")
    import dataclasses
    atomic_json(internal / "video_metadata.json", dataclasses.asdict(metadata))
    stat = video.stat()
    atomic_json(internal / "fingerprint.json", {"source_path": str(video), "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns, "model_sha256": "unchanged", "threshold": .5})
    shots = []
    for i, (start, end) in enumerate([(0, 1), (1, 3), (3, 6), (6, 20)], 1):
        sid = f"shot_{i:06d}"
        shots.append({"shot_id": sid, "temporal": {"start": start/25, "end": end/25,
                                                       "start_frame": start, "end_frame": end}})
        middle = (start+end-1)//2
        name = f"{sid}_{format_timestamp(middle/25).replace(':','-').replace('.','-')}.jpg"
        Image.new("RGB", (16,16), (middle*10,0,0)).save(images/name)
    atomic_json(movie / f"Stage01_{mid}.json", {"movie_id": mid, "movie_name": "Fixture", "shots": shots})
    atomic_json(movie / f"Stage02_{mid}.json", {"movie_id": mid, "unchanged": True})
    (movie / f"{mid}.srt").write_text("subtitle bytes\n")
    release.mkdir()
    inputs = {}
    for kind, name in [("Stage01", f"Stage01_{mid}.json"), ("Stage02", f"Stage02_{mid}.json"), ("SRT", f"{mid}.srt")]:
        shutil.copy2(movie/name, release/name)
        inputs[kind] = {"path": name, "sha256": sha256_file(movie/name)}
    manifest = release/"frozen_manifest.json"
    atomic_json(manifest, {"release_id": release.name, "dataset_status": "FROZEN", "activated_for_pipeline": False,
        "movie_count": 1, "movies": [{"movie_id": mid, "release_eligible": True, "activated_for_pipeline": False, "inputs": inputs}]})
    monkeypatch.setattr(upgrade, "probe_video", lambda *a: (metadata, {}))
    monkeypatch.setattr(upgrade, "scan_timeline", lambda *a: timeline)
    calls = []
    def extract(path, indices, target_dir, progress=None, *, decode_progress=None):
        calls.append(list(indices))
        target_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for frame in indices:
            output = target_dir / f"frame_{frame:08d}.jpg"
            Image.new("RGB", (16,16), (frame*10,0,0)).save(output)
            paths.append(output)
        return paths
    monkeypatch.setattr(upgrade, "extract_selected_frames", extract)
    options = dict(input_root=source, output_root=workspace, frozen_manifest=manifest,
        expected_manifest_sha256=sha256_file(manifest), expect_movies=1, report_root=tmp_path/"reports", movie_id=mid)
    return dict(options=options, movie=movie, internal=internal, video=video, calls=calls, timeline=timeline, metadata=metadata)


def tree_hashes(path):
    return {str(p.relative_to(path)): sha256_file(p) for p in path.rglob("*") if p.is_file()}


def test_dry_run_apply_idempotent_preserves_bytes(fixture, monkeypatch):
    f=fixture
    before = tree_hashes(f["options"]["output_root"])
    frozen = tree_hashes(f["options"]["frozen_manifest"].parent)
    assert frames_upgrade(**f["options"])["status"] == "dry_run"
    assert before == tree_hashes(f["options"]["output_root"])
    assert not f["calls"]
    assert frames_upgrade(**f["options"], apply=True)["status"] == "complete"
    assert len(f["calls"]) == 1
    assert len(list((f["movie"]/"shot_img").iterdir())) == 9
    record = json.loads((f["internal"]/"frame_samples_v1.json").read_text())
    for rename in record["renames"]:
        assert before[f["movie"].name+"/"+rename["original_path"]] == sha256_file(f["movie"]/rename["new_path"])
        assert not (f["movie"]/rename["original_path"]).exists()
    for name, digest in before.items():
        if "/shot_img/" not in name:
            assert sha256_file(f["options"]["output_root"]/name) == digest
    assert frozen == tree_hashes(f["options"]["frozen_manifest"].parent)
    monkeypatch.setattr(upgrade, "scan_timeline", lambda *a: pytest.fail("Unnecessary timeline scan"))
    assert frames_upgrade(**f["options"], apply=True)["results"][0]["status"] == "skipped_complete"
    assert len(f["calls"]) == 1
    assert validate_output_root(f["options"]["output_root"], frame_profile="three_sample")[f["movie"].name].passed
    assert not validate_movie(f["movie"], frame_profile="legacy").passed


@pytest.mark.parametrize("problem", ["missing", "corrupt", "unexpected", "duplicate", "destination", "source", "stage01", "stage02", "srt", "manifest", "count", "video_missing", "ambiguous"])
def test_rejects_before_any_writes(fixture, problem):
    f=fixture
    image=next((f["movie"]/"shot_img").iterdir())
    if problem == "missing": image.unlink()
    if problem == "corrupt": image.write_bytes(b"bad")
    if problem == "unexpected": (image.parent/"notes.txt").write_text("never delete")
    if problem == "duplicate": shutil.copy2(image, image.with_name(image.name[:12]+"00-00-00-999.jpg"))
    if problem == "destination": shutil.copy2(image, image.with_name(image.name[:12]+"middle.jpg"))
    if problem == "source": f["video"].write_bytes(b"different source")
    if problem == "stage01": (f["movie"]/f"Stage01_{f['movie'].name}.json").write_text("{}")
    if problem == "stage02": (f["movie"]/f"Stage02_{f['movie'].name}.json").write_text("{}")
    if problem == "srt": (f["movie"]/f"{f['movie'].name}.srt").write_text("changed")
    if problem == "manifest": f["options"]["expected_manifest_sha256"]="0"*64
    if problem == "count": f["options"]["expect_movies"]=17
    if problem == "video_missing": f["video"].unlink()
    if problem == "ambiguous": shutil.copy2(f["video"], f["video"].with_name(f["movie"].name+"_extra.mkv"))
    before=tree_hashes(f["options"]["output_root"])
    with pytest.raises((ValueError, OSError)):
        frames_upgrade(**f["options"], apply=True)
    assert before == tree_hashes(f["options"]["output_root"])
    assert not f["options"]["report_root"].exists()
    assert not f["calls"]


def test_interrupted_publication_recovers_without_reextract(fixture, monkeypatch):
    original=upgrade.publish_image
    count=0
    def interrupt(source, dest, digest):
        nonlocal count
        original(source, dest, digest)
        if dest.name.endswith("_early.jpg"):
            count+=1
            if count==1: raise RuntimeError("simulated crash after publication")
    monkeypatch.setattr(upgrade, "publish_image", interrupt)
    with pytest.raises(RuntimeError, match="simulated crash"):
        frames_upgrade(**fixture["options"], apply=True)
    monkeypatch.setattr(upgrade, "publish_image", original)
    # Simulated extractor models the real persistent per-frame cache.
    def cached(path, indices, work, progress=None, *, decode_progress=None):
        outputs=[work/f"frame_{n:08d}.jpg" for n in indices]
        assert all(p.exists() for p in outputs)
        return outputs
    monkeypatch.setattr(upgrade, "extract_selected_frames", cached)
    assert frames_upgrade(**fixture["options"], apply=True)["status"] == "complete"
    assert validate_movie(fixture["movie"], frame_profile="three_sample").passed


def test_resume_rejects_conflicting_new_image(fixture, monkeypatch):
    original=upgrade.publish_image
    def stop(source, dest, digest):
        original(source,dest,digest)
        raise RuntimeError("stop")
    monkeypatch.setattr(upgrade, "publish_image", stop)
    with pytest.raises(RuntimeError): frames_upgrade(**fixture["options"], apply=True)
    middle=next((fixture["movie"]/"shot_img").glob("*_middle.jpg"))
    Image.new("RGB", (16,16), "green").save(middle)
    before=tree_hashes(fixture["options"]["output_root"])
    with pytest.raises(ValueError, match="conflicting"):
        frames_upgrade(**fixture["options"], apply=True)
    assert before==tree_hashes(fixture["options"]["output_root"])


def test_sample_validation_tampering(fixture):
    frames_upgrade(**fixture["options"], apply=True)
    path=fixture["internal"]/"frame_samples_v1.json"
    record=json.loads(path.read_text())
    record["shots"]["shot_000001"]["early"]["alias_of"]=None
    atomic_json(path,record)
    assert not validate_movie(fixture["movie"], frame_profile="three_sample").passed


def test_middle_provenance_hash_required(fixture):
    frames_upgrade(**fixture["options"], apply=True)
    path=fixture["internal"]/"frame_samples_v1.json"
    record=json.loads(path.read_text())
    record["renames"][0]["original_sha256"]="0"*64
    atomic_json(path,record)
    assert not validate_movie(fixture["movie"], frame_profile="three_sample").passed


def test_vfr_pts_not_nominal_fps(fixture, monkeypatch):
    f=fixture
    f["metadata"].is_vfr=True
    f["timeline"].is_vfr=True
    f["timeline"].pts_sec=[i*.04 if i<6 else .24+(i-6)*.08 for i in range(20)]
    f["timeline"].exclusive_end_sec=1.36
    f["metadata"].duration_sec=1.36
    import dataclasses
    atomic_json(f["internal"]/"video_metadata.json",dataclasses.asdict(f["metadata"]))
    stage=f["movie"]/f"Stage01_{f['movie'].name}.json"
    data=json.loads(stage.read_text())
    for shot in data["shots"]:
        t=shot["temporal"]
        t["start"]=f["timeline"].timestamp(t["start_frame"])
        t["end"]=f["timeline"].exclusive_timestamp(t["end_frame"])
        middle=(t["start_frame"]+t["end_frame"]-1)//2
        image=next((f["movie"]/"shot_img").glob(shot["shot_id"]+"*"))
        image.rename(image.with_name(f"{shot['shot_id']}_{format_timestamp(f['timeline'].timestamp(middle)).replace(':','-').replace('.','-')}.jpg"))
    atomic_json(stage,data)
    manifest=f["options"]["frozen_manifest"]
    payload=json.loads(manifest.read_text())
    shutil.copy2(stage,manifest.parent/stage.name)
    payload["movies"][0]["inputs"]["Stage01"]["sha256"]=sha256_file(stage)
    atomic_json(manifest,payload)
    f["options"]["expected_manifest_sha256"]=sha256_file(manifest)
    frames_upgrade(**f["options"],apply=True)
    record=json.loads((f["internal"]/"frame_samples_v1.json").read_text())
    sample=record["shots"]["shot_000004"]["middle"]
    assert sample["timestamp_sec"]==f["timeline"].pts_sec[sample["frame_index"]]
    assert sample["timestamp_sec"] != sample["frame_index"]/25


def test_overwrite_refuses_downstream_without_model_access(fixture):
    f=fixture
    options=pipeline.ProcessOptions(input_root=f["options"]["input_root"],output_root=f["options"]["output_root"],
        weights=Path("/nonexistent/model"),overwrite=True)
    with pytest.raises(RuntimeError,match="downstream"):
        pipeline.process_one(f["video"],options)


def test_18_movie_manifest_selection(tmp_path):
    release=tmp_path/"release"
    release.mkdir()
    movies=[]
    ids=["tt12300742"]+[f"tt{i:07d}" for i in range(17)]
    for mid in ids:
        inputs={}
        for kind in ("Stage01","Stage02","SRT"):
            p=release/f"{mid}_{kind}"
            p.write_text(json.dumps({"movie_id":mid}))
            inputs[kind]={"path":p.name,"sha256":sha256_file(p)}
        movies.append({"movie_id":mid,"inputs":inputs,"release_eligible":True,"activated_for_pipeline":False})
    manifest=release/"frozen_manifest.json"
    atomic_json(manifest,{"release_id":"release","dataset_status":"FROZEN","activated_for_pipeline":False,
                          "movie_count":18,"movies":movies})
    _,selected=select_movies(manifest,sha256_file(manifest),movie_id=None,exclude_movie_id="tt12300742",expect_movies=17)
    assert {m["movie_id"] for m in selected}==set(ids)-{"tt12300742"}
    _,single=select_movies(manifest,sha256_file(manifest),movie_id=ids[1],exclude_movie_id=None,expect_movies=1)
    assert [m["movie_id"] for m in single]==[ids[1]]
    with pytest.raises(ValueError,match="Expected"):
        select_movies(manifest,sha256_file(manifest),movie_id=None,exclude_movie_id="tt12300742",expect_movies=18)


def real_video(tmp_path, vfr=False):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe required")
    path=tmp_path/"tt7654321.mkv"
    command=["ffmpeg","-v","error","-f","lavfi","-i","testsrc2=size=32x24:rate=25:duration=0.8"]
    if vfr:
        command += ["-vf", "setpts='if(lt(N,5),N,5+(N-5)*2)'", "-vsync", "vfr"]
    command += ["-c:v","ffv1",str(path)]
    subprocess.run(command,check=True,capture_output=True)
    return path


@pytest.mark.parametrize("vfr", [False, True])
def test_real_ffmpeg_exact_indices_and_cached_resume(tmp_path, monkeypatch, vfr):
    video=real_video(tmp_path,vfr)
    metadata,_=media.probe_video(video,video.name)
    timeline=media.scan_timeline(video,metadata.fps)
    assert timeline.frame_count==20
    assert timeline.is_vfr is vfr
    selected=[0,4,5,13,19]
    cache=tmp_path/"cache"
    def stop(n):
        if n==2: raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError,match="interrupted"):
        media.extract_selected_frames(video,selected,cache,stop)
    preserved={p.name:sha256_file(p) for p in cache.glob("*.jpg")}
    outputs=media.extract_selected_frames(video,selected,cache)
    for name,digest in preserved.items():
        assert sha256_file(cache/name)==digest
    # Independent FFmpeg selection verifies decoded frame ordinal, including VFR.
    for n,path in zip(selected,outputs,strict=True):
        raw=subprocess.run(["ffmpeg","-v","error","-i",str(video),"-vf",f"select=eq(n\\,{n})",
            "-frames:v","1","-vsync","0","-pix_fmt","rgb24","-f","rawvideo","pipe:1"],
            check=True,capture_output=True).stdout
        reference=Image.frombytes("RGB",(32,24),raw)
        with Image.open(path) as actual:
            # JPEG chroma subsampling tolerances; mean error still identifies wrong frames.
            from PIL import ImageChops, ImageStat
            assert sum(ImageStat.Stat(ImageChops.difference(actual,reference)).mean)/3 < 12
    monkeypatch.setattr(media.subprocess,"Popen",lambda *a,**kw: pytest.fail("Unnecessary FFmpeg"))
    assert media.extract_selected_frames(video,selected,cache)==outputs


def test_real_ffmpeg_incomplete_decode(tmp_path):
    video=real_video(tmp_path)
    with pytest.raises(media.ExternalToolError,match="Incomplete decode"):
        media.extract_selected_frames(video,[0,20],tmp_path/"cache")
    assert (tmp_path/"cache"/"frame_00000000.jpg").is_file()


def test_new_film_real_extraction_and_qc(tmp_path,monkeypatch):
    video=real_video(tmp_path,True)
    metadata,_=media.probe_video(video,video.name)
    timeline=media.scan_timeline(video,metadata.fps)
    metadata.frame_count=timeline.frame_count
    metadata.is_vfr=timeline.is_vfr
    metadata.duration_sec=timeline.exclusive_end_sec
    metadata.timestamp_source="decoded_pts_vfr"
    predictions=[0.0]*20
    predictions[5]=.9
    monkeypatch.setattr(pipeline,"_load_or_run_inference",lambda *a:(metadata,timeline,predictions))
    weights=tmp_path/"weights"
    weights.write_bytes(b"no model loading")
    output=tmp_path/"out"
    options=pipeline.ProcessOptions(input_root=tmp_path,output_root=output,weights=weights,qc=True,progress=False)
    result=pipeline.process_one(video,options)
    assert result["status"]=="completed"
    assert len(list((output/"tt7654321"/"shot_img").glob("*.jpg")))==6
    assert (output/".internal"/"tt7654321"/"qc"/"boundary_contact_sheet.jpg").is_file()
    assert validate_output_root(output,frame_profile="three_sample")["tt7654321"].passed
    assert pipeline.process_one(video,options)["resumed"]


@pytest.mark.parametrize("problem",["frame_count","duration","codec","pts","frozen_input","unknown_sidecar"])
def test_media_and_frozen_evidence_fail_closed(fixture,problem):
    f=fixture
    if problem=="frame_count": f["timeline"].pts_sec.pop()
    if problem=="duration": f["timeline"].exclusive_end_sec+=1
    if problem=="codec":
        # Actual probe returns fixture metadata; recorded identity stays unchanged.
        f["metadata"].codec_name="other"
    if problem=="pts": f["timeline"].pts_sec[3]+=.02; f["timeline"].is_vfr=True
    if problem=="frozen_input":
        path=f["options"]["frozen_manifest"].parent/f"Stage02_{f['movie'].name}.json"
        path.write_text("{}")
    if problem=="unknown_sidecar": atomic_json(f["internal"]/"frame_samples_v1.json",{"unknown":True})
    before=tree_hashes(f["options"]["output_root"])
    with pytest.raises(ValueError): frames_upgrade(**f["options"],apply=True)
    assert before==tree_hashes(f["options"]["output_root"])


def test_cli_upgrade_modes_and_profile():
    from oscardp.shots.__main__ import build_parser
    parser=build_parser()
    common=["frames-upgrade","--input-root","/tmp/in","--output-root","/tmp/out",
        "--frozen-manifest","/tmp/release/frozen_manifest.json","--expected-manifest-sha256","a"*64,
        "--expect-movies","17","--report-root","/tmp/report","--exclude-movie-id","tt12300742"]
    args=parser.parse_args(common)
    assert not args.apply and not args.dry_run and args.expect_movies==17
    assert not args.no_progress
    assert parser.parse_args(common+["--no-progress"]).no_progress
    assert parser.parse_args(common+["--apply"]).apply
    with pytest.raises(SystemExit): parser.parse_args(common+["--movie-id","tt1234567"])
    with pytest.raises(SystemExit): parser.parse_args(common+["--dry-run","--apply"])
    assert parser.parse_args(["validate","--frame-profile","three_sample"]).frame_profile=="three_sample"


def test_publication_does_not_require_hardlinks(fixture,monkeypatch):
    import os
    monkeypatch.setattr(os,"link",lambda *a,**kw:pytest.fail("exFAT cannot hardlink"))
    assert frames_upgrade(**fixture["options"],apply=True)["status"]=="complete"
    assert not (fixture["internal"]/"frame_publish_work").exists()


def test_torn_receipt_recovery(tmp_path):
    from oscardp.shots.frame_samples import append_receipt,read_receipts
    path=tmp_path/"receipts.jsonl"
    append_receipt(path,{"frame_index":0,"sha256":"first"})
    with path.open("ab") as stream: stream.write(b'{"interrupted":')
    assert read_receipts(path)==[{"frame_index":0,"sha256":"first"}]
    append_receipt(path,{"frame_index":1,"sha256":"second"})
    assert len(read_receipts(path))==2


def test_incomplete_owned_jpeg_encode_resumes(tmp_path):
    from oscardp.shots.frame_samples import append_receipt
    video=real_video(tmp_path)
    target=tmp_path/"cache"
    target.mkdir()
    stat=video.stat()
    atomic_json(target/"extraction.json",{"source":{"path":str(video.resolve()),"size":stat.st_size,
                                                "mtime_ns":stat.st_mtime_ns},"frames":{}})
    append_receipt(target/"frames.jsonl",{"frame_index":4,"status":"encoding"})
    (target/"frame_00000004.tmp").write_bytes(b"incomplete owned encode")
    outputs=media.extract_selected_frames(video,[4],target)
    assert len(outputs)==1 and image_info(outputs[0])["bytes"]>0


def test_unknown_temporary_jpeg_is_not_overwritten(tmp_path):
    video=real_video(tmp_path)
    target=tmp_path/"cache"
    target.mkdir()
    unknown=target/"frame_00000004.tmp"
    unknown.write_bytes(b"unknown file")
    with pytest.raises((OSError,ValueError,media.ExternalToolError)):
        media.extract_selected_frames(video,[4],target)
    assert unknown.read_bytes()==b"unknown file"


def test_unavailable_pts_never_shift_indices(monkeypatch):
    class Probe:
        stdout=iter(["media_type=video|best_effort_timestamp_time=N/A|pkt_duration_time=0.04\n"])
        def kill(self): pass
        def wait(self): return 0
    monkeypatch.setattr(media,"require_media_tools",lambda:None)
    monkeypatch.setattr(media.subprocess,"Popen",lambda *a,**kw:Probe())
    with pytest.raises(media.ExternalToolError,match="timing unavailable"):
        list(media.iter_frame_timestamps(Path("unused")))


def test_reports_have_complete_extraction_hashes(fixture):
    frames_upgrade(**fixture["options"],apply=True)
    report=json.loads((fixture["options"]["report_root"]/"upgrade_report.json").read_text())
    record=report["results"][0]["sampling_record"]
    assert record["status"]=="complete"
    assert all(sample["sha256"] for roles in record["shots"].values() for sample in roles.values())


def test_downstream_created_during_inference_blocks_publication(tmp_path,monkeypatch):
    video=real_video(tmp_path)
    metadata,_=media.probe_video(video,video.name)
    timeline=media.scan_timeline(video,metadata.fps)
    metadata.frame_count=timeline.frame_count
    output=tmp_path/"out"
    final=output/"tt7654321"
    final.mkdir(parents=True)
    sentinel=final/"Stage01_tt7654321.json"
    sentinel.write_text("original stage01")
    def inference(*args):
        (final/"Stage02_tt7654321.json").write_text("preserve downstream")
        return metadata,timeline,[0.0]*20
    monkeypatch.setattr(pipeline,"_load_or_run_inference",inference)
    weights=tmp_path/"weights"
    weights.write_bytes(b"no inference")
    options=pipeline.ProcessOptions(input_root=tmp_path,output_root=output,weights=weights,overwrite=True,progress=False)
    with pytest.raises(RuntimeError,match="downstream"):
        pipeline.process_one(video,options)
    assert sentinel.read_text()=="original stage01"
    assert (final/"Stage02_tt7654321.json").read_text()=="preserve downstream"


def test_real_h264_pts_scan_ignores_sei_and_blank_records(tmp_path):
    original=real_video(tmp_path)
    video=tmp_path/"h264.mp4"
    subprocess.run(["ffmpeg","-v","error","-i",str(original),"-c:v","libx264",str(video)],
                   check=True,capture_output=True)
    timeline=media.scan_timeline(video,25)
    assert timeline.frame_count==20
    assert timeline.is_vfr is False
