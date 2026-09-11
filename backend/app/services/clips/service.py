"""Clip generation: orchestration, persistence and quality checking."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from app.core.config import settings
from app.core.jobs import JobContext
from app.models.db import Clip, Project, get_session, new_id
from app.services.captions.ass import CaptionStyle
from app.services.clips.render import RenderSettings, render_clip
from app.services.ai.service import load_candidates
from app.services.clips.boundaries import BoundaryWeights
from app.services.clips.select import ClipWindow, select_clips
from app.services.transcription.base import Transcript
from app.services.transcription.service import transcript_path


class ClipGenerationError(Exception):
    pass


def clips_dir(project_id: str) -> Path:
    return settings.projects_dir / project_id / "clips"


def quality_check(path: Path, expected_duration: float) -> dict:
    """Verify a rendered clip is actually usable.

    A render can exit 0 and still produce something broken — no audio stream,
    a duration that does not match the request, wrong dimensions. Catching that
    here beats discovering it after exporting twenty clips.
    """
    issues: list[str] = []

    probe = subprocess.run(
        [settings.ffprobe_bin, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        return {"ok": False, "issues": ["Rendered file could not be read back."]}

    try:
        data = json.loads(probe.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "issues": ["Rendered file produced unreadable metadata."]}

    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video is None:
        issues.append("No video stream.")
    if audio is None:
        issues.append("No audio stream — the clip is silent.")

    try:
        actual = float(data.get("format", {}).get("duration", 0))
    except (TypeError, ValueError):
        actual = 0.0

    if expected_duration and abs(actual - expected_duration) > 1.0:
        issues.append(
            f"Duration is {actual:.1f}s but {expected_duration:.1f}s was requested."
        )

    width = int(video["width"]) if video and video.get("width") else 0
    height = int(video["height"]) if video and video.get("height") else 0
    if video and (width, height) != (settings.output_width, settings.output_height):
        issues.append(
            f"Dimensions are {width}x{height}, expected "
            f"{settings.output_width}x{settings.output_height}."
        )

    return {
        "ok": not issues,
        "issues": issues,
        "duration": round(actual, 2),
        "width": width,
        "height": height,
        "size_bytes": path.stat().st_size,
    }


def load_transcript(project_id: str) -> Transcript:
    path = transcript_path(project_id)
    if not path.exists():
        raise ClipGenerationError(
            "No transcript for this project. Transcribe it first."
        )
    try:
        return Transcript.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, OSError) as exc:
        raise ClipGenerationError("The stored transcript could not be read.") from exc


async def handle_generate_clips(ctx: JobContext) -> dict:
    """Job handler for type 'generate_clips'."""
    project_id = ctx.project_id
    if not project_id:
        raise ClipGenerationError("No project supplied.")

    with get_session() as db:
        project = db.get(Project, project_id)
        if project is None:
            raise ClipGenerationError("Project no longer exists.")
        source = Path(project.source_path)

    params = ctx.params
    count = int(params.get("count", settings.default_clip_count))
    min_duration = float(params.get("min_duration", settings.clip_min_duration))
    max_duration = float(params.get("max_duration", settings.clip_max_duration))
    crop_strategy = params.get("crop_strategy", settings.crop_strategy)
    captions = bool(params.get("captions", True))

    await ctx.progress(0.02, "reading transcript")
    transcript = load_transcript(project_id)

    if not transcript.has_word_timestamps and captions:
        raise ClipGenerationError(
            "The transcript has no word timestamps, so captions cannot be built. "
            "Re-transcribe with a backend that provides them."
        )

    await ctx.progress(0.05, "selecting moments")
    weights = BoundaryWeights(
        opener=settings.weight_opener,
        starts_sentence=settings.weight_starts_sentence,
        ends_sentence=settings.weight_ends_sentence,
        not_dangling=settings.weight_not_dangling,
        low_filler=settings.weight_low_filler,
        duration_fit=settings.weight_duration_fit,
    )
    # Prefer discovered moments when they exist. use_discovery=False forces
    # the even-anchor path, which is useful for comparing the two.
    anchors = None
    if params.get("use_discovery", True):
        stored = load_candidates(project_id)
        if stored:
            anchors = stored[:max(count * 2, count)]

    windows: list[ClipWindow] = select_clips(
        transcript, count, min_duration, max_duration,
        anchors=anchors,
        padding=settings.clip_padding,
        weights=weights,
        min_boundary_score=float(params.get("min_boundary_score",
                                            settings.min_boundary_score)),
    )

    if not windows:
        raise ClipGenerationError(
            "No usable clips could be found. The transcript may be too short, or "
            "the duration limits too narrow for its sentence structure."
        )

    if len(windows) < count:
        # Say so rather than padding the output with rubbish, per spec 22.
        shortfall = f"Found {len(windows)} usable clips, fewer than the {count} requested."
    else:
        shortfall = None

    # Replace any previous run for this project.
    with get_session() as db:
        for old in db.query(Clip).filter(Clip.project_id == project_id).all():
            if old.render_path:
                Path(old.render_path).unlink(missing_ok=True)
            db.delete(old)
        db.commit()

    render_settings = RenderSettings(
        width=settings.output_width,
        height=settings.output_height,
        crf=settings.output_crf,
        encoder=settings.output_encoder,
        captions=captions,
        crop_strategy=crop_strategy,
    )
    style = CaptionStyle()

    output_dir = clips_dir(project_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    span = 0.9 / len(windows)

    for index, window in enumerate(windows):
        ctx.raise_if_cancelled()
        base = 0.08 + span * index
        await ctx.progress(base, f"rendering clip {index + 1}/{len(windows)}")

        clip_id = new_id()
        dest = output_dir / f"clip_{index + 1:02d}_{clip_id}.mp4"

        await render_clip(
            source=source,
            window=window,
            dest=dest,
            settings_=render_settings,
            caption_style=style,
            ctx=ctx,
            progress_base=base,
            progress_span=span * 0.9,
        )

        qc = quality_check(dest, window.duration)

        with get_session() as db:
            db.add(Clip(
                id=clip_id,
                project_id=project_id,
                index=index + 1,
                start=window.start,
                end=window.end,
                duration=window.duration,
                text=window.text[:5000],
                strategy=window.strategy,
                boundary_score=window.boundary_score,
                boundary_notes="; ".join(window.boundary_notes) or None,
                hook=(window.discovery or {}).get("hook"),
                score=(window.discovery or {}).get(
                    "final_score", (window.discovery or {}).get("normalized_score")
                ),
                topic=(window.discovery or {}).get("topic"),
                category=(window.discovery or {}).get("category"),
                crop_strategy=crop_strategy,
                render_path=str(dest),
                qc_ok=qc["ok"],
                qc_issues="; ".join(qc["issues"]) if qc["issues"] else None,
                size_bytes=qc.get("size_bytes", 0),
            ))
            db.commit()

        results.append({
            "id": clip_id,
            "index": index + 1,
            "start": round(window.start, 2),
            "end": round(window.end, 2),
            "duration": round(window.duration, 2),
            "boundary_score": window.boundary_score,
            "boundary_notes": window.boundary_notes,
            "qc": qc,
        })

    await ctx.progress(1.0, "done")
    return {
        "clips": len(results),
        "requested": count,
        "anchor_source": "discovery" if anchors else "even spacing",
        "note": shortfall,
        "failed_qc": [r["index"] for r in results if not r["qc"]["ok"]],
        "details": results,
    }
