from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import TextIO


def _duration(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressReporter:
    """Small dependency-free terminal progress reporter.

    Progress is intentionally written to stderr so stdout remains suitable for
    the CLI's machine-readable JSON result.
    """

    def __init__(
        self,
        enabled: bool = True,
        stream: TextIO | None = None,
        min_interval: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.min_interval = min_interval
        self.clock = clock
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.label = ""
        self.total: int | None = None
        self.estimated = False
        self.current = 0
        self.started_at = 0.0
        self.last_rendered_at = float("-inf")
        self.line_active = False

    def stage(self, label: str, total: int | None = None, *, estimated: bool = False) -> None:
        if not self.enabled:
            return
        self._end_active_line()
        self.label = label
        self.total = total if total and total > 0 else None
        self.estimated = estimated
        self.current = 0
        self.started_at = self.clock()
        self.last_rendered_at = float("-inf")
        self._render(force=True)

    def update(self, current: int, total: int | None = None) -> None:
        if not self.enabled:
            return
        self.current = max(0, current)
        if total is not None and total > 0:
            self.total = total
        now = self.clock()
        complete = self.total is not None and self.current >= self.total
        if complete or now - self.last_rendered_at >= self.min_interval:
            self._render(now=now, force=True)

    def finish(self, detail: str | None = None) -> None:
        if not self.enabled:
            return
        self._render(force=True, detail=detail)
        self._end_active_line()

    def fail(self, detail: str) -> None:
        if not self.enabled:
            return
        self._end_active_line()
        self.stream.write(f"{self.label} FAILED: {detail}\n")
        self.stream.flush()

    def _render(
        self,
        *,
        now: float | None = None,
        force: bool = False,
        detail: str | None = None,
    ) -> None:
        if not self.enabled or (not force and not self.label):
            return
        now = self.clock() if now is None else now
        elapsed = max(0.0, now - self.started_at)
        fields = [self.label]
        if self.current:
            if self.total is not None:
                total_prefix = "~" if self.estimated else ""
                percent = min(100.0, self.current / self.total * 100.0)
                fields.append(
                    f"{self.current:,}/{total_prefix}{self.total:,} frames ({percent:5.1f}%)"
                )
            else:
                fields.append(f"{self.current:,} frames")
        fields.append(f"elapsed {_duration(elapsed)}")
        if self.current and elapsed > 0:
            rate = self.current / elapsed
            fields.append(f"{rate:,.1f} frames/s")
            if self.total is not None and self.current < self.total and rate > 0:
                fields.append(f"ETA {_duration((self.total - self.current) / rate)}")
        if detail:
            fields.append(detail)
        text = " | ".join(fields)
        if self.is_tty:
            self.stream.write("\r\x1b[2K" + text)
            self.line_active = True
        else:
            self.stream.write(text + "\n")
            self.line_active = False
        self.stream.flush()
        self.last_rendered_at = now

    def _end_active_line(self) -> None:
        if self.line_active:
            self.stream.write("\n")
            self.stream.flush()
            self.line_active = False
