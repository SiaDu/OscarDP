"""Loopback-only static viewer with validated, revision-checked independent saves."""
import json
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .annotations import validate_records
from .integration import read, digest


def make_handler(root, review_dir):
    root = root.resolve()
    config = read(root/"run_config.json")
    source_hash = config["source_run_sha256"]
    path = review_dir.resolve()/f"{source_hash}.json"
    if review_dir.resolve().is_relative_to(root):
        raise ValueError("Human reviews must be outside algorithm output")
    index = read(root/"review_items.json")
    items = [i["source_queue_item"] for i in index["items"]]
    packet = read(root/"scene_performance_packets.json")
    ranges = {s["shot_id"]: [s["effective_processing_temporal"]["start_frame"], s["effective_processing_temporal"]["end_frame"]]
              for scene in packet["scenes"] for s in scene["shots"]}

    def state():
        payload = read(path)
        if payload["source_run_sha256"] != source_hash:
            raise ValueError("Saved review Source Hash mismatch")
        validate_records(payload["records"], items, source_hash, ranges)
        return payload

    state()  # Refuse to serve mismatched saved progress.

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root/"viewer"), **kwargs)

        def reply(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlsplit(self.path).path == "/api/reviews":
                try:
                    payload = state()
                    self.reply(200, {"revision": digest(payload), "review_file": str(path), **payload})
                except (ValueError, KeyError, TypeError) as exc:
                    self.reply(409, {"error": str(exc)})
            else:
                super().do_GET()

        def do_POST(self):
            if self.path != "/api/reviews":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            origin = self.headers.get("Origin")
            if (self.headers.get("X-OscarDP-Review") != "1"
                    or self.headers.get_content_type() != "application/json"
                    or (origin and origin != "http://"+self.headers.get("Host", ""))):
                self.reply(403, {"error": "Same-origin explicit JSON review save required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("Invalid review payload size")
                request = json.loads(self.rfile.read(length))
                current = state()
                if request["revision"] != digest(current):
                    self.reply(409, {"error": "Progress changed in another window; reload before saving"})
                    return
                if request["source_run_sha256"] != source_hash:
                    raise ValueError("Source Hash mismatch")
                records = validate_records(request["records"], items, source_hash, ranges)
                current.update(records=records)
                temporary = path.with_suffix(".json.tmp")
                with temporary.open("x") as stream:
                    json.dump(current, stream, ensure_ascii=False, allow_nan=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                self.reply(200, {"revision": digest(current), "review_file": str(path), **current})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.reply(400, {"error": str(exc)})

    return Handler


def serve(root, review_dir=None, port=8766):
    config = read(root/"run_config.json")
    review_dir = review_dir or Path(config["review_directory"])
    server = HTTPServer(("127.0.0.1", port), make_handler(root, review_dir))
    print(f"Review Viewer: http://127.0.0.1:{server.server_port} (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
