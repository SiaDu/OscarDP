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
