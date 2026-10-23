"""Explicit local OpenFace executable, models, libraries and command provenance."""
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from .identity_inputs import file_hash


def preflight(binary):
    binary = binary.resolve(strict=True)
    if not os.access(binary, os.X_OK):
        raise ValueError("OpenFace binary is not executable")
    root = binary.parent
    required = ["model/main_ceclm_general.txt", "model/cen_general.txt",
                "model/patch_experts/cen_patches_1.00_of.dat", "model/model_eye/main_clnf_synth_left.txt",
                "model/model_eye/main_clnf_synth_right.txt", "AU_predictors/main_static_svms.txt",
                "AU_predictors/main_dynamic_svms.txt"]
    for name in required:
        if not (root/name).is_file():
            raise FileNotFoundError(f"Missing OpenFace model: {root/name}")
    libraries = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=True).stdout
    if "not found" in libraries:
        raise RuntimeError("Missing OpenFace dynamic libraries:\n"+libraries)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("FFmpeg missing")
    ffmpeg_version = subprocess.run([ffmpeg,"-version"],capture_output=True,text=True,check=True).stdout
    files = [p for directory in [root/"model",root/"AU_predictors"] for p in directory.rglob("*") if p.is_file()]
    manifest = {str(p.relative_to(root)): {"size":p.stat().st_size,"sha256":file_hash(p)} for p in sorted(files)}
    repository = next((p for p in binary.parents if (p/"CMakeLists.txt").is_file() and (p/"exe/FeatureExtraction").is_dir()), None)
    source_info = {}
    if repository:
        cmake = (repository/"CMakeLists.txt").read_text()
        match = re.search(r"project\(OpenFace VERSION ([^)]+)\)", cmake)
        source_info["cmake_declared_version"] = match[1] if match else "unknown"
        source_info["version_caveat"] = "Source CMake version, not a runtime --version claim; binary SHA256 identifies the executable"
        source_info["git_commit"] = subprocess.run(["git","-C",str(repository),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
        for path in [repository/"CMakeLists.txt",repository/"build/CMakeCache.txt",
                     repository/"build/exe/FeatureExtraction/CMakeFiles/FeatureExtraction.dir/flags.make"]:
            if path.is_file():
                source_info[str(path.relative_to(repository))] = {"sha256":file_hash(path),"text":path.read_text()}
    return {"implementation":"Original OpenFace FeatureExtraction C++", "binary":str(binary),
            "binary_sha256":file_hash(binary),"source_build":source_info,"dynamic_libraries_ldd":libraries,
            "model_root":str(root),"model_manifest":manifest,"ffmpeg":ffmpeg,"ffmpeg_version":ffmpeg_version,
            "known_optional_warning":"HAAR classifier may be absent; this build defaults to MTCNN, not HAAR",
            "coordinates":{"landmarks_2d":"source full-frame pixels, x right, y down",
                           "landmarks_3d_translation":"model-estimated millimetres in camera frame; x right, y down, z away from camera",
                           "pose_rotation":"radians, OpenFace camera-frame Euler Rx/Ry/Rz (pitch/yaw/roll); no calibrated world-frame truth",
                           "gaze":"camera-frame unit direction per eye; angles atan2(x,-z), atan2(y,-z), radians; no gaze target",
                           "camera_calibrated":False},
            "execution_environment":{"OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"1"}}


def command(binary, video, output, mode, camera):
    if mode not in {"static","dynamic"}:
        raise ValueError("AU extraction mode must be explicit static or dynamic")
    args = [str(binary.resolve()),"-f",str(video.resolve()),"-out_dir",str(output.resolve()),
            "-2Dfp","-3Dfp","-pdmparams","-pose","-aus","-gaze"]
    for key in ["fx","fy","cx","cy"]:
        args += ["-"+key,str(camera[key])]
    if mode == "static":
        args.append("-au_static")
    return args


def extract(binary, video, output, mode, camera):
    output.mkdir(parents=True,exist_ok=False)
    args = command(binary,video,output,mode,camera)
    env = dict(os.environ,OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1",QT_QPA_PLATFORM="offscreen")
    tick = time.perf_counter()
    with (output/"execution.log").open("x") as log:
        completed = subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,env=env,check=False)
    result = {"command":args,"extraction_mode":mode,"elapsed_sec":time.perf_counter()-tick,"returncode":completed.returncode}
    csv_path = output/(video.stem+".csv")
    if completed.returncode or not csv_path.is_file():
        raise RuntimeError(f"OpenFace extraction failed; see {output/'execution.log'}")
    return csv_path,result
