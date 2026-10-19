from __future__ import annotations

import io

from oscardp.shots.progress import ProgressReporter


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_progress_reports_percentage_rate_and_eta_to_stream() -> None:
    stream = io.StringIO()
    clock = Clock()
    reporter = ProgressReporter(stream=stream, min_interval=0, clock=clock)
    reporter.stage("[2/5] Scan", 100)
    clock.now = 2.0
    reporter.update(25)
    reporter.finish("done")
    output = stream.getvalue()
    assert "25/100 frames ( 25.0%)" in output
    assert "12.5 frames/s" in output
    assert "ETA 00:00:06" in output
    assert "done" in output


def test_disabled_progress_writes_nothing() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(enabled=False, stream=stream)
    reporter.stage("ignored", 10)
    reporter.update(5)
    reporter.finish()
    assert stream.getvalue() == ""


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_terminal_bar_context_and_file_units():
    stream = Terminal()
    clock = Clock()
    reporter = ProgressReporter(stream=stream, clock=clock)
    reporter.context = "[预检 3/17] tt0025316"
    reporter.stage("Images", 10, unit="files")
    clock.now = 1
    reporter.update(5)
    reporter.finish()
    output = stream.getvalue()
    assert "[预检 3/17] tt0025316 Images" in output
    assert "0/10 files" in output and "5/10 files" in output
    assert "[##########----------]" in output
    assert "\r\x1b[2K" in output and output.endswith("\n")
    assert "frames" not in output


def test_byte_rate_and_unknown_total():
    stream = io.StringIO()
    clock = Clock()
    reporter = ProgressReporter(stream=stream, clock=clock)
    reporter.stage("SHA-256", 4 * 1024 * 1024, unit="bytes")
    clock.now = 2
    reporter.update(2 * 1024 * 1024)
    reporter.finish()
    assert "1.0 MiB/s" in stream.getvalue()
    assert "ETA 00:00:02" in stream.getvalue()
    stream.seek(0)
    stream.truncate()
    reporter.stage("Unknown", unit="files")
    clock.now = 3
    reporter.update(10)
    reporter.finish()
    output = stream.getvalue()
    assert "10 files" in output and "elapsed" in output
    assert all(word not in output for word in ("ETA", "%", "/s", "\x1b", "\r"))


def test_updates_throttled_including_repeated_completion():
    stream = io.StringIO()
    clock = Clock()
    reporter = ProgressReporter(stream=stream, clock=clock)
    reporter.stage("Samples", 100, unit="samples")
    first = stream.getvalue()
    clock.now = .1
    reporter.update(10)
    clock.now = .49
    reporter.update(20)
    assert stream.getvalue() == first
    clock.now = .5
    reporter.update(50, detail="decoded 5/10")
    middle = stream.getvalue()
    assert "decoded 5/10" in middle
    clock.now = .6
    reporter.update(100, detail="decoded 10/10")
    reporter.update(100)
    assert stream.getvalue() == middle
    reporter.finish()
    assert "100/100 samples" in stream.getvalue()
    assert "decoded 10/10" in stream.getvalue()


def test_hash_progress_counts_actual_bytes_without_changing_digest(tmp_path):
    import hashlib
    from oscardp.shots.transnet import sha256_file
    data = b"0123456789"
    path = tmp_path / "source"
    path.write_bytes(data)
    events = []
    assert sha256_file(path, 3, progress=events.append) == hashlib.sha256(data).hexdigest()
    assert events == [3, 6, 9, 10]
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()
