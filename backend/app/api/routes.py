"""HTTP API: projects, media inspection, jobs and progress streaming."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select

from app.core.config import settings
from app.core.environment import check_environment
from app.core.jobs import job_to_dict, runner
from app.models.db import TERMINAL_STATUSES, Clip, Job, Project, get_session, new_id
from app.services.media import MediaInspectionError, inspect
from app.services.transcription.base import Transcript
from app.services.clips.service import clips_dir
from app.services.transcription.service import transcript_path

router = APIRouter(prefix="/api")


def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash first + last 8 MB plus size. Full hashing of a multi-GB podcast is
    slow and unnecessary - this is for cache keying, not integrity."""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(8 * chunk_size))
        if size > 16 * chunk_size:
            fh.seek(-8 * chunk_size, 2)
            h.update(fh.read())
    return h.hexdigest()


def _clip_count(project_id: str) -> int:
    with get_session() as db:
        return db.query(Clip).filter(Clip.project_id == project_id).count()


def _serialise(p: Project) -> dict:
    return {
        "id": p.id,
        "name": p.name,
        "source_filename": p.source_filename,
        "duration": p.duration,
        "width": p.width,
        "height": p.height,
        "fps": p.fps,
        "video_codec": p.video_codec,
        "audio_codec": p.audio_codec,
        "has_audio": p.has_audio,
        "size_bytes": p.size_bytes,
        "audio_path": p.audio_path,
        "has_extracted_audio": bool(p.audio_path and Path(p.audio_path).exists()),
        "has_transcript": transcript_path(p.id).exists(),
        "clip_count": _clip_count(p.id),
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }


# --- environment --------------------------------------------------------


@router.get("/environment")
def environment() -> dict:
    return check_environment().as_dict()


# --- projects -----------------------------------------------------------


@router.get("/projects")
def list_projects() -> list[dict]:
    with get_session() as db:
        rows = db.scalars(select(Project).order_by(Project.created_at.desc())).all()
        return [_serialise(p) for p in rows]


@router.post("/projects", status_code=201)
async def create_project(file: UploadFile = File(...)) -> dict:
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()

    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix or 'none'}'. "
                f"Supported: {', '.join(settings.allowed_extensions)}"
            ),
        )

    settings.ensure_dirs()
    project_id = new_id()
    dest = settings.uploads_dir / f"{project_id}{suffix}"

    # Stream to disk rather than reading into memory - sources are multi-GB
    # and this machine has 16 GB shared between CPU and GPU.
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out, length=1 << 20)
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not save upload: {exc}") from exc
    finally:
        await file.close()

    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    if dest.stat().st_size > settings.max_upload_bytes:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="File exceeds the configured size limit.")

    try:
        info = inspect(dest)
    except MediaInspectionError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not info.has_audio:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="This file has no audio track. Transcription is impossible without one.",
        )

    project = Project(
        id=project_id,
        name=Path(filename).stem,
        source_path=str(dest),
        source_filename=filename,
        source_hash=_hash_file(dest),
        duration=info.duration,
        width=info.width,
        height=info.height,
        fps=info.fps,
        video_codec=info.video_codec,
        audio_codec=info.audio_codec,
        has_audio=info.has_audio,
        size_bytes=info.size_bytes,
    )

    with get_session() as db:
        db.add(project)
        db.commit()

    return {"project": _serialise(project), "media": info.as_dict()}


@router.get("/projects/{project_id}")
def get_project(project_id: str) -> dict:
    with get_session() as db:
        project = db.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        return _serialise(project)


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str) -> None:
    with get_session() as db:
        project = db.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        Path(project.source_path).unlink(missing_ok=True)
        if project.audio_path:
            Path(project.audio_path).unlink(missing_ok=True)
        for clip in db.query(Clip).filter(Clip.project_id == project_id).all():
            db.delete(clip)
        derived = settings.projects_dir / project_id
        if derived.exists():
            shutil.rmtree(derived, ignore_errors=True)
        db.delete(project)
        db.commit()


# --- jobs ---------------------------------------------------------------


@router.post("/projects/{project_id}/extract-audio", status_code=202)
async def start_extract_audio(project_id: str, force: bool = False) -> dict:
    with get_session() as db:
        if db.get(Project, project_id) is None:
            raise HTTPException(status_code=404, detail="Project not found.")

    job_id = await runner.submit("extract_audio", project_id, {"force": force})
    with get_session() as db:
        return job_to_dict(db.get(Job, job_id))


@router.post("/projects/{project_id}/transcribe", status_code=202)
async def start_transcribe(project_id: str, force: bool = False) -> dict:
    with get_session() as db:
        project = db.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        if not project.audio_path or not Path(project.audio_path).exists():
            raise HTTPException(
                status_code=409,
                detail="Extract audio for this project first.",
            )

    job_id = await runner.submit("transcribe", project_id, {"force": force})
    with get_session() as db:
        return job_to_dict(db.get(Job, job_id))


@router.get("/projects/{project_id}/transcript")
def get_transcript(project_id: str, full: bool = False) -> dict:
    path = transcript_path(project_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="No transcript for this project yet.")
    transcript = Transcript.from_dict(json.loads(path.read_text()))
    payload = {
        "stats": transcript.stats(),
        "language": transcript.language,
        "backend": transcript.backend,
        "model": transcript.model,
        "issues": [
            {"kind": i.kind, "detail": i.detail, "at": round(i.at, 2)}
            for i in transcript.validate()[:20]
        ],
    }
    if full:
        payload["segments"] = [s.to_dict() for s in transcript.segments]
    else:
        payload["preview"] = transcript.text[:2000]
    return payload


@router.get("/transcription/backends")
def transcription_backends() -> list[dict]:
    from app.services.transcription.backends import BACKENDS

    out = []
    for name, cls in BACKENDS.items():
        instance = cls()
        available, reason = instance.is_available()
        out.append({
            "name": name,
            "available": available,
            "reason": reason,
            "default_model": instance.default_model,
        })
    return out


# --- clips --------------------------------------------------------------


def _serialise_clip(c: Clip) -> dict:
    return {
        "id": c.id,
        "project_id": c.project_id,
        "index": c.index,
        "start": round(c.start, 2),
        "end": round(c.end, 2),
        "duration": round(c.duration, 2),
        "text": c.text,
        "hook": c.hook,
        "score": c.score,
        "strategy": c.strategy,
        "boundary_score": c.boundary_score,
        "boundary_notes": c.boundary_notes,
        "crop_strategy": c.crop_strategy,
        "qc_ok": c.qc_ok,
        "qc_issues": c.qc_issues,
        "size_bytes": c.size_bytes,
        "has_render": bool(c.render_path and Path(c.render_path).exists()),
        "video_url": f"/api/clips/{c.id}/video",
    }


@router.post("/projects/{project_id}/clips", status_code=202)
async def start_generate_clips(
    project_id: str,
    count: int | None = None,
    crop_strategy: str | None = None,
    captions: bool = True,
) -> dict:
    with get_session() as db:
        if db.get(Project, project_id) is None:
            raise HTTPException(status_code=404, detail="Project not found.")

    if not transcript_path(project_id).exists():
        raise HTTPException(
            status_code=409,
            detail="Transcribe this project before generating clips.",
        )

    params: dict = {"captions": captions}
    if count is not None:
        params["count"] = count
    if crop_strategy is not None:
        params["crop_strategy"] = crop_strategy

    job_id = await runner.submit("generate_clips", project_id, params)
    with get_session() as db:
        return job_to_dict(db.get(Job, job_id))


@router.get("/projects/{project_id}/clips")
def list_clips(project_id: str, sort: str = "index") -> list[dict]:
    order = {
        "index": Clip.index,
        "duration": Clip.duration,
        "score": Clip.boundary_score.desc(),
        "start": Clip.start,
    }.get(sort, Clip.index)
    with get_session() as db:
        rows = db.scalars(
            select(Clip).where(Clip.project_id == project_id).order_by(order)
        ).all()
        return [_serialise_clip(c) for c in rows]


@router.get("/clips/{clip_id}/video")
def clip_video(clip_id: str) -> FileResponse:
    with get_session() as db:
        clip = db.get(Clip, clip_id)
        if clip is None or not clip.render_path:
            raise HTTPException(status_code=404, detail="Clip not found.")
        path = Path(clip.render_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Rendered file is missing.")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.delete("/clips/{clip_id}", status_code=204)
def delete_clip(clip_id: str) -> None:
    with get_session() as db:
        clip = db.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(status_code=404, detail="Clip not found.")
        if clip.render_path:
            Path(clip.render_path).unlink(missing_ok=True)
        db.delete(clip)
        db.commit()


@router.get("/jobs")
def list_jobs(project_id: str | None = None, limit: int = 50) -> list[dict]:
    with get_session() as db:
        query = select(Job).order_by(Job.created_at.desc()).limit(limit)
        if project_id:
            query = select(Job).where(Job.project_id == project_id).order_by(
                Job.created_at.desc()
            ).limit(limit)
        return [job_to_dict(j) for j in db.scalars(query).all()]


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with get_session() as db:
        job = db.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job_to_dict(job)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    with get_session() as db:
        job = db.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.status in TERMINAL_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=f"Job already finished with status {job.status}.",
            )
    runner.cancel(job_id)
    return {"cancelling": True, "job_id": job_id}


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    """Server-Sent Events stream of job progress.

    SSE rather than WebSockets: progress is one-way, and SSE reconnects on its
    own without any client-side plumbing.
    """
    with get_session() as db:
        job = db.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        initial = job_to_dict(job)

    queue = runner.subscribe(job_id)

    async def stream():
        try:
            yield f"data: {json.dumps(initial)}\n\n"
            if initial["status"] in TERMINAL_STATUSES:
                return
            while True:
                if await request.is_disconnected():
                    return
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    # Comment frame keeps proxies and browsers from timing out.
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(payload)}\n\n"
                if payload["status"] in TERMINAL_STATUSES:
                    return
        finally:
            runner.unsubscribe(job_id, queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
