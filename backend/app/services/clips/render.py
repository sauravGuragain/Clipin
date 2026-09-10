"""Clip rendering: cut, reframe to 9:16, burn captions, encode.

Phase 4 crops to centre. That is wrong for a two-person podcast — the whole
point of Phase 8 is following the active speaker — but it is honestly wrong
rather than fake: the crop is real, the output is watchable, and Phase 8
replaces `CropStrategy` without touching anything else here.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.core.jobs import JobCancelled, JobContext
from app.services.captions.ass import CaptionStyle, build_ass
from app.services.clips.select import ClipWindow

_PROGRESS_LINE = re.compile(r"^(\w+)=(.*)$")


class RenderError(Exception):
    pass


@dataclass
class RenderSettings:
    width: int = 1080
    height: int = 1920
    fps: int | None = None          # None keeps the source frame rate
    crf: int = 20
    preset: str = "medium"
    encoder: str = "libx264"        # h264_videotoolbox for fast previews
    audio_bitrate: str = "160k"
    captions: bool = True
    crop_strategy: str = "center"


def build_crop_filter(strategy: str, width: int, height: int) -> str:
    """Reframe to the target aspect.

    Expressed in FFmpeg's own expression language rather than computed in
    Python so the same filter works for any source resolution, including
    sources that are already vertical.
    """
    target_aspect = width / height

    if strategy == "center":
        # Crop the largest 9:16 rectangle that fits, centred, then scale.
        return (
            f"crop=w='min(iw,ih*{target_aspect})':h='min(ih,iw/{target_aspect})'"
            f":x='(iw-ow)/2':y='(ih-oh)/2',"
            f"scale={width}:{height}:flags=lanczos,setsar=1"
        )

    if strategy == "fit":
        # Whole frame, letterboxed onto a blurred fill of itself. Useful when
        # cropping would cut someone out of shot.
        return (
            f"split[bg][fg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=40:2[bgb];"
            f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1"
        )

    raise RenderError(f"Unknown crop strategy '{strategy}'. Use 'center' or 'fit'.")


async def render_clip(
    source: Path,
    window: ClipWindow,
    dest: Path,
    settings_: RenderSettings | None = None,
    caption_style: CaptionStyle | None = None,
    ctx: JobContext | None = None,
    progress_base: float = 0.0,
    progress_span: float = 1.0,
) -> Path:
    """Render one clip. Progress is read from FFmpeg's own output."""
    settings_ = settings_ or RenderSettings()
    if not source.exists():
        raise RenderError(f"Source media is missing: {source.name}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".partial.mp4")
    ass_path = dest.with_suffix(".ass")

    filters = build_crop_filter(settings_.crop_strategy, settings_.width, settings_.height)

    if settings_.captions and window.words:
        style = caption_style or CaptionStyle()
        # Caption timings are relative to the clip, not the podcast.
        ass_path.write_text(
            build_ass(
                window.words,
                style,
                settings_.width,
                settings_.height,
                time_offset=window.start,
            )
        )
        # Escaping for the filter graph: colons and commas separate options.
        escaped = str(ass_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filters += f",subtitles='{escaped}'"

    args = [
        settings.ffmpeg_bin, "-nostdin", "-y",
        # Seek before -i for speed, then -t for length. Re-encoding means the
        # cut is frame-accurate despite the fast seek.
        "-ss", f"{window.start:.3f}",
        "-i", str(source),
        "-t", f"{window.duration:.3f}",
        "-filter_complex" if settings_.crop_strategy == "fit" else "-vf", filters,
        "-c:v", settings_.encoder,
    ]

    if settings_.encoder == "libx264":
        args += ["-crf", str(settings_.crf), "-preset", settings_.preset]
    else:
        # VideoToolbox has no CRF; it wants a bitrate.
        args += ["-b:v", "6M"]

    if settings_.fps:
        args += ["-r", str(settings_.fps)]

    args += [
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", settings_.audio_bitrate, "-ar", "48000",
        "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats", "-loglevel", "error",
        str(tmp),
    ]

    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    async def pump() -> None:
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                return
            match = _PROGRESS_LINE.match(raw.decode("utf-8", "replace").strip())
            if not match:
                continue
            key, value = match.groups()
            if key == "out_time_us" and ctx and window.duration > 0:
                try:
                    seconds = int(value) / 1_000_000
                except ValueError:
                    continue
                fraction = min(1.0, seconds / window.duration)
                await ctx.progress(progress_base + fraction * progress_span, "rendering")

    pump_task = asyncio.create_task(pump())

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
        pump_task.cancel()
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
        lines = stderr.decode("utf-8", "replace").strip().splitlines()
        reason = lines[-1] if lines else "FFmpeg exited with an error."
        reason = reason.replace(str(source), source.name)
        raise RenderError(f"Render failed. {reason}")

    if not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RenderError("Render produced an empty file.")

    tmp.replace(dest)
    return dest
