"""Versioned sampling and byte-preserving publication, independent of Stage01 JSON."""
from __future__ import annotations

import ctypes
import json
import shutil
import os
import tempfile
from pathlib import Path

from PIL import Image

from .transnet import sha256_file

POLICY = {"name": "three_sample", "version": 1,
          "formula": "early=floor((N-1)/10), middle=floor((N-1)/2), late=floor(9*(N-1)/10); clamp around middle for N>=3"}
ROLES = ("early", "middle", "late")


def sample_indices(start: int, end: int) -> dict[str, int]:
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        raise ValueError("Invalid shot frame range")
    n = end - start
    middle = (start + end - 1) // 2
    if n == 1:
        return dict.fromkeys(ROLES, start)
    if n == 2:
        return {"early": start, "middle": middle, "late": end - 1}
    return {"early": min(start + (n - 1) // 10, middle - 1),
            "middle": middle,
            "late": max(start + 9 * (n - 1) // 10, middle + 1)}


def physical_samples(shot_id: str, start: int, end: int) -> dict[str, dict]:
    indices = sample_indices(start, end)
    # Middle always owns its physical image, preserving historical bytes for N=1/2.
    owners = {indices["middle"]: "middle"}
    for role in ROLES:
        owners.setdefault(indices[role], role)
    return {role: {"frame_index": frame, "alias_of": owners[frame] if owners[frame] != role else None,
                   "image_relpath": f"shot_img/{shot_id}_{owners[frame]}.jpg"}
            for role, frame in indices.items()}


def safe_path(root: Path, relative: str = ".") -> Path:
    root = root.absolute()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    path = root / rel
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symlink path rejected: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes root: {path}")
    return path


def load_json(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value) -> None:
    safe_path(path.parent, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".frames-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        sync_dir(path.parent)
    finally:
        Path(tmp).unlink(missing_ok=True)


def image_info(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Not a regular image: {path}")
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise ValueError(f"Not a JPEG: {path}")
        image.verify()
    with Image.open(path) as image:
        image.load()  # verify alone does not detect every truncated JPEG
        size = list(image.size)
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size, "size": size}


def rename_noreplace(source: Path, destination: Path) -> None:
    """Linux atomic no-clobber rename, including filesystems without hard links."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))
    sync_dir(destination.parent)


def publish_image(source: Path, destination: Path, expected: str) -> None:
    """Preserve source bytes until verified, then atomically publish without clobber.

    A temporary byte copy works on exFAT as well as native Linux filesystems.
    Neither unknown destinations nor source JPEGs are ever rewritten.
    """
    safe_path(destination.parent, destination.name)
    if image_info(source)["sha256"] != expected:
        raise ValueError(f"Changed image source: {source}")
    if destination.exists():
        if image_info(destination)["sha256"] != expected:
            raise ValueError(f"Conflicting destination image: {destination}")
        return
    staging = destination.parent.parent.parent / ".internal" / destination.parent.parent.name / "frame_publish_work"
    safe_path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    # Deterministic owned temp paths allow interrupted copies to be retried safely.
    temporary = staging / destination.name
    safe_path(staging, destination.name)
    intent_path = safe_path(staging, destination.name + ".intent.json")
    intent = {"destination": str(destination), "source": str(source), "sha256": expected}
    if intent_path.exists():
        if load_json(intent_path) != intent:
            raise ValueError("Conflicting publication intent")
    else:
        if temporary.exists():
            raise ValueError(f"Unknown publication temporary file: {temporary}")
        atomic_json(intent_path, intent)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as target, source.open("rb") as original:
            shutil.copyfileobj(original, target)
            target.flush()
            os.fsync(target.fileno())
        if image_info(Path(temporary))["sha256"] != expected:
            raise ValueError("Image temporary copy hash mismatch")
        rename_noreplace(Path(temporary), destination)
        if image_info(destination)["sha256"] != expected:
            raise ValueError(f"Published image mismatch: {destination}")
    finally:
        Path(temporary).unlink(missing_ok=True)
        intent_path.unlink(missing_ok=True)
        sync_dir(staging)


def read_receipts(path: Path) -> list[dict]:
    """A torn final append is uncommitted; earlier newline-terminated entries survive."""
    safe_path(path.parent, path.name)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw[:raw.rfind(b"\n") + 1].splitlines()
    return [json.loads(line) for line in lines]


def append_receipt(path: Path, value: dict) -> None:
    safe_path(path.parent, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Repair only an uncommitted trailing fragment under the workspace writer lock.
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        length = stream.tell()
        if length:
            stream.seek(length - 1)
            if stream.read(1) != b"\n":
                offset = length
                boundary = 0
                while offset:
                    start = max(0, offset - 4096)
                    stream.seek(start)
                    chunk = stream.read(offset - start)
                    newline = chunk.rfind(b"\n")
                    if newline >= 0:
                        boundary = start + newline + 1
                        break
                    offset = start
                stream.truncate(boundary)
        stream.seek(0, os.SEEK_END)
        stream.write((json.dumps(value, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(path.parent)
