"""Audio extraction.

Produces the 16 kHz mono WAV that Whisper wants (Phase 3). Progress is parsed
from FFmpeg's own `-progress` stream rather than estimated, because a fake
progress bar that moves smoothly while nothing happens is worse than none.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from app.core.config import settings
from app.core.jobs import JobCancelled, JobContext
from app.models.db import Project, get_session

# Whisper resamples internally to 16 kHz mono. Doing it here means the model
# never has to, and the intermediate file stays small: roughly 115 MB per hour
# versus ~1.2 GB for 48 kHz stereo.
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1

_PROGRESS_LINE = re.compile(r"^(\w+)=(.*)$")


class AudioExtractionError(Exception):
    pass


def audio_path_for(project_id: str) -> Path:
    return settings.projects_dir / project_id / "audio.wav"


async def extract_audio(
    source: Path,
    dest: Path,
    duration: float | None,
    ctx: JobContext | None = None,
) -> Path:
    """Extract to 16 kHz mono PCM WAV, reporting real progress."""
    if not source.exists():
        raise AudioExtractionError(f"Source media is missing: {source.name}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name so an interrupted run never leaves a truncated
    # WAV that a later stage would happily treat as complete.
    tmp = dest.with_suffix(".partial.wav")

    args = [
        settings.ffmpeg_bin,
        "-nostdin",
        "-y",
        "-i", str(source),
        "-vn",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-progress", "pipe:1",
        "-nostats",
        "-loglevel", "error",
        str(tmp),
    ]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def pump_progress() -> None:
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                return
            match = _PROGRESS_LINE.match(raw.decode("utf-8", "replace").strip())
            if not match:
                continue
            key, value = match.groups()
            if key == "out_time_us" and duration and duration > 0:
                try:
                    seconds = int(value) / 1_000_000
                except ValueError:
                    continue
                if ctx:
                    await ctx.progress(seconds / duration, stage="extracting audio")

    pump = asyncio.create_task(pump_progress())

    try:
        while True:
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.5)
                break
            except asyncio.TimeoutError:
                if ctx and ctx.cancelled:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        proc.kill()
                    raise JobCancelled()
    finally:
        pump.cancel()
        stderr = b""
        if proc.stderr is not None:
            try:
                stderr = await proc.stderr.read()
            except Exception:
                pass
        if proc.returncode is None:
            proc.kill()

    if ctx and ctx.cancelled:
        tmp.unlink(missing_ok=True)
        raise JobCancelled()

    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        detail = stderr.decode("utf-8", "replace").strip().splitlines()
        reason = detail[-1] if detail else "FFmpeg exited with an error."
        reason = reason.replace(str(source), source.name)
        raise AudioExtractionError(f"Audio extraction failed. {reason}")

    if not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise AudioExtractionError("Audio extraction produced an empty file.")

    tmp.replace(dest)
    return dest


async def handle_extract_audio(ctx: JobContext) -> dict:
    """Job handler for type 'extract_audio'."""
    project_id = ctx.project_id
    if not project_id:
        raise AudioExtractionError("No project supplied.")

    with get_session() as db:
        project = db.get(Project, project_id)
        if project is None:
            raise AudioExtractionError("Project no longer exists.")
        source = Path(project.source_path)
        duration = project.duration
        existing = project.audio_path

    dest = audio_path_for(project_id)

    # Caching, per PLAN.md section 23. Re-extraction of an unchanged source is
    # pure waste, and the user will hit this every time they change a downstream
    # setting.
    force = bool(ctx.params.get("force"))
    if not force and existing and Path(existing).exists() and Path(existing).stat().st_size > 0:
        await ctx.progress(1.0, stage="cached")
        return {
            "audio_path": existing,
            "cached": True,
            "sample_rate": TARGET_SAMPLE_RATE,
            "channels": TARGET_CHANNELS,
        }

    ctx.raise_if_cancelled()
    await ctx.progress(0.0, stage="extracting audio")

    await extract_audio(source, dest, duration, ctx)

    with get_session() as db:
        project = db.get(Project, project_id)
        if project is not None:
            project.audio_path = str(dest)
            db.commit()

    return {
        "audio_path": str(dest),
        "cached": False,
        "size_bytes": dest.stat().st_size,
        "sample_rate": TARGET_SAMPLE_RATE,
        "channels": TARGET_CHANNELS,
    }
