import json
import os
from pathlib import Path

from oscardp.shots.schema import json_dumps


def atomic_json(path: Path, data):
    temp = path.with_suffix(path.suffix+".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        stream.write(json_dumps(data, pretty=True)+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
