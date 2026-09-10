"""Transcription service: caching, chunked execution, job integration."""

from __future__ import annotations

import asyncio
import hashlib
import time
import json
import subprocess
import tempfile
from pathlib import Path

from app.core.config import settings
from app.core.jobs import JobCancelled, JobContext
from app.models.db import Project, get_session
from app.services.transcription.backends import get_backend
from app.services.transcription.base import Transcript
from app.services.transcription.stitch import Chunk, merge_chunks, plan_chunks


class TranscriptionError(Exception):
    pass


def transcript_path(project_id: str) -> Path:
    return settings.projects_dir / project_id / "transcript.json"


def cache_key(source_hash: str, backend: str, model: str, language: str | None) -> str:
    """Identity of a transcript.

    Changing caption style or crop must not invalidate this (PLAN.md section
    23). Changing the model or the language hint must.
    """
    payload = f"{source_hash}|{backend}|{model}|{language or 'auto'}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_cached(project_id: str, key: str) -> Transcript | None:
    path = transcript_path(project_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("cache_key") != key:
        return None
    return Transcript.from_dict(data)


def save_transcript(project_id: str, transcript: Transcript) -> Path:
    path = transcript_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(transcript.to_dict(), indent=2))
    tmp.replace(path)
    return path


def slice_audio(source: Path, start: float, end: float, dest: Path) -> Path:
    """Cut a chunk of WAV for the model to read.

    Re-encoding rather than stream-copying: PCM slicing must land on exact
    sample boundaries, and a copy can drift by a frame, which would shift every
    timestamp in the chunk.
    """
    args = [
        settings.ffmpeg_bin, "-nostdin", "-y", "-v", "error",
        "-i", str(source),
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not dest.exists():
        raise TranscriptionError(
            f"Could not extract audio chunk at {start:.1f}s: "
            f"{(proc.stderr or '').strip().splitlines()[-1] if proc.stderr else 'unknown error'}"
        )
    return dest


def transcribe_sync(
    audio_path: Path,
    duration: float,
    backend_name: str | None,
    model: str | None,
    language: str | None,
    chunk_seconds: float,
    overlap: float,
    on_chunk_done=None,
    should_cancel=None,
) -> Transcript:
    """Blocking transcription. Runs in a worker thread — see the handler."""
    backend = get_backend(backend_name)
    available, reason = backend.is_available()
    if not available:
        raise TranscriptionError(reason)

    resolved_model = model or backend.default_model
    chunks = plan_chunks(duration, chunk_seconds, overlap)
    if not chunks:
        raise TranscriptionError("Audio has zero duration.")

    results: list[tuple[Chunk, Transcript]] = []

    # Single chunk: hand the file straight to the model, no slicing.
    if len(chunks) == 1:
        if should_cancel and should_cancel():
            raise JobCancelled()
        part = backend.transcribe(str(audio_path), resolved_model, language, offset=0.0)
        results.append((chunks[0], part))
        if on_chunk_done:
            on_chunk_done(1, 1)
    else:
        with tempfile.TemporaryDirectory(prefix="clipper-chunks-") as tmpdir:
            for i, chunk in enumerate(chunks):
                if should_cancel and should_cancel():
                    raise JobCancelled()
                piece = Path(tmpdir) / f"chunk_{chunk.index:04d}.wav"
                slice_audio(audio_path, chunk.read_start, chunk.read_end, piece)
                part = backend.transcribe(
                    str(piece), resolved_model, language, offset=chunk.read_start
                )
                results.append((chunk, part))
                if on_chunk_done:
                    on_chunk_done(i + 1, len(chunks))

    return merge_chunks(
        results,
        language=language,
        backend=backend.name,
        model=resolved_model,
        duration=duration,
    )


async def handle_transcribe(ctx: JobContext) -> dict:
    """Job handler for type 'transcribe'."""
    project_id = ctx.project_id
    if not project_id:
        raise TranscriptionError("No project supplied.")

    with get_session() as db:
        project = db.get(Project, project_id)
        if project is None:
            raise TranscriptionError("Project no longer exists.")
        audio_path = project.audio_path
        source_hash = project.source_hash or ""
        duration = project.duration or 0.0

    if not audio_path or not Path(audio_path).exists():
        raise TranscriptionError(
            "No extracted audio for this project. Run audio extraction first."
        )

    params = ctx.params
    backend_name = params.get("backend") or settings.transcribe_backend
    model = params.get("model") or settings.transcribe_model or None
    language = params.get("language") or settings.transcribe_language or None
    chunk_seconds = float(params.get("chunk_seconds", settings.transcribe_chunk_seconds))
    overlap = float(params.get("overlap", settings.transcribe_overlap))
    force = bool(params.get("force"))

    backend = get_backend(backend_name)
    available, reason = backend.is_available()
    if not available:
        raise TranscriptionError(reason)
    resolved_model = model or backend.default_model
    key = cache_key(source_hash, backend.name, resolved_model, language)

    if not force:
        cached = load_cached(project_id, key)
        if cached is not None:
            await ctx.progress(1.0, "cached")
            return {
                "cached": True,
                "transcript_path": str(transcript_path(project_id)),
                "stats": cached.stats(),
                "issues": len(cached.validate()),
            }

    ctx.raise_if_cancelled()
    await ctx.progress(0.0, "loading model")

    loop = asyncio.get_running_loop()

    def on_chunk_done(done: int, total: int) -> None:
        stage = f"transcribing ({done}/{total})" if total > 1 else "transcribing"
        asyncio.run_coroutine_threadsafe(ctx.progress(done / total, stage), loop)

    started_at = time.monotonic()
    transcript = await asyncio.to_thread(
        transcribe_sync,
        Path(audio_path),
        duration,
        backend_name,
        model,
        language,
        chunk_seconds,
        overlap,
        on_chunk_done,
        lambda: ctx.cancelled,
    )

    ctx.raise_if_cancelled()

    if not transcript.has_word_timestamps:
        raise TranscriptionError(
            f"Backend '{transcript.backend}' returned no word-level timestamps. "
            "Animated captions cannot be built without them."
        )

    transcript.cache_key = key
    await ctx.progress(0.98, "saving")
    path = save_transcript(project_id, transcript)

    elapsed = time.monotonic() - started_at
    issues = transcript.validate()
    return {
        "cached": False,
        "transcript_path": str(path),
        # Recorded so the realtime factor is measured on real work rather than
        # estimated. A 90-minute podcast at 1x is a very different product from
        # one at 20x.
        "elapsed_seconds": round(elapsed, 1),
        "realtime_factor": round(duration / elapsed, 2) if elapsed > 0 else None,
        "backend": transcript.backend,
        "model": transcript.model,
        "stats": transcript.stats(),
        "issues": len(issues),
        # Surface a few rather than all: a systematically broken alignment
        # produces thousands, and a truncated list makes the point.
        "issue_samples": [
            {"kind": i.kind, "detail": i.detail, "at": round(i.at, 2)} for i in issues[:5]
        ],
    }
