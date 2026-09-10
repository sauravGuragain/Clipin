"""Startup environment checks.

The lesson that produced this module: `ffmpeg` being on PATH tells you nothing
about whether it can do what you need. A lean Homebrew build has no libass, and
the failure would otherwise surface at caption-render time, many phases later.
So we check capabilities, not existence.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field

from app.core.config import settings


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True
    remedy: str | None = None


@dataclass
class EnvironmentReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.fatal)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [
                {
                    "name": c.name,
                    "ok": c.ok,
                    "detail": c.detail,
                    "fatal": c.fatal,
                    "remedy": c.remedy,
                }
                for c in self.checks
            ],
        }


def _run(args: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, "binary not found"
    except subprocess.TimeoutExpired:
        return 124, "timed out"


def check_environment() -> EnvironmentReport:
    report = EnvironmentReport()

    # --- ffmpeg present -------------------------------------------------
    ffmpeg_path = shutil.which(settings.ffmpeg_bin) or settings.ffmpeg_bin
    code, out = _run([settings.ffmpeg_bin, "-version"])
    if code != 0:
        report.checks.append(
            Check(
                name="ffmpeg",
                ok=False,
                detail=f"could not run '{settings.ffmpeg_bin}'",
                remedy="brew install ffmpeg-full, then: brew link --overwrite ffmpeg-full",
            )
        )
        return report

    version_line = out.splitlines()[0] if out else "unknown"
    report.checks.append(
        Check(name="ffmpeg", ok=True, detail=f"{version_line} at {ffmpeg_path}")
    )

    # --- ffprobe --------------------------------------------------------
    code, out_probe = _run([settings.ffprobe_bin, "-version"])
    report.checks.append(
        Check(
            name="ffprobe",
            ok=code == 0,
            detail=(out_probe.splitlines()[0] if code == 0 and out_probe else "not runnable"),
            remedy=None if code == 0 else "ffprobe ships with ffmpeg; reinstall it",
        )
    )

    # --- libass / subtitles filter --------------------------------------
    # Captions depend on this entirely. Check the filter list, not the
    # configure string, because that is what actually gets used at render time.
    code, filters = _run([settings.ffmpeg_bin, "-hide_banner", "-filters"])
    has_subtitles = any(
        line.split()[1:2] == ["subtitles"] for line in filters.splitlines() if line.split()
    )
    report.checks.append(
        Check(
            name="libass (subtitles filter)",
            ok=has_subtitles,
            detail="available" if has_subtitles else "MISSING - captions cannot render",
            remedy=(
                None
                if has_subtitles
                else "brew install ffmpeg-full && brew unlink ffmpeg "
                     "&& brew link --overwrite ffmpeg-full"
            ),
        )
    )

    # --- encoders -------------------------------------------------------
    code, encoders = _run([settings.ffmpeg_bin, "-hide_banner", "-encoders"])
    for enc, fatal in (("libx264", True), ("aac", True), ("h264_videotoolbox", False)):
        present = any(
            line.split()[1:2] == [enc] for line in encoders.splitlines() if line.split()
        )
        report.checks.append(
            Check(
                name=f"encoder: {enc}",
                ok=present,
                detail="available" if present else "missing",
                fatal=fatal,
                remedy=None if present else "use ffmpeg-full",
            )
        )

    return report


def format_report(report: EnvironmentReport) -> str:
    lines = []
    for c in report.checks:
        mark = "OK  " if c.ok else ("FAIL" if c.fatal else "warn")
        lines.append(f"  [{mark}] {c.name}: {c.detail}")
        if not c.ok and c.remedy:
            lines.append(f"         fix: {c.remedy}")
    return "\n".join(lines)
