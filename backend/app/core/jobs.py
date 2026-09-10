"""Background job system.

Deliberately in-process: an asyncio queue with a small number of workers. No
Celery, no Redis, no broker. This is a single-user local tool, and an external
broker would add operational surface with nothing to show for it.

Concurrency defaults to 1. That is not arbitrary — see STACK.md section 3. The
machine has 16 GB shared between CPU and GPU, and overlapping a Whisper model
with a render will swap. Stages run one at a time unless explicitly told
otherwise.
"""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timezone
from typing import Any

from app.models.db import Job, JobStatus, get_session, new_id, utcnow


class JobCancelled(Exception):
    """Raised inside a handler when the job has been cancelled."""


@dataclass
class JobContext:
    """Passed to every handler. The handler reports progress through this and
    must check `raise_if_cancelled()` at points where stopping is safe."""

    job_id: str
    project_id: str | None
    params: dict[str, Any]
    _runner: JobRunner
    _cancel: asyncio.Event

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def raise_if_cancelled(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled()

    async def progress(self, fraction: float, stage: str | None = None) -> None:
        """Report progress in 0.0-1.0. Values outside are clamped rather than
        rejected, because progress arithmetic from external tools is unreliable
        and a crash here would be worse than a slightly wrong bar."""
        value = max(0.0, min(1.0, float(fraction)))
        await self._runner._update(self.job_id, progress=value, current_stage=stage)


Handler = Callable[[JobContext], Awaitable[dict[str, Any]]]


class JobRunner:
    def __init__(self, concurrency: int = 1) -> None:
        self.concurrency = concurrency
        self._handlers: dict[str, Handler] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._cancels: dict[str, asyncio.Event] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._running = False

    # --- registration ---------------------------------------------------

    def register(self, job_type: str, handler: Handler) -> None:
        self._handlers[job_type] = handler

    # --- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._recover_interrupted()
        for _ in range(self.concurrency):
            self._workers.append(asyncio.create_task(self._worker()))

    async def stop(self) -> None:
        self._running = False
        for task in self._workers:
            task.cancel()
        for task in self._workers:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._workers.clear()

    def _recover_interrupted(self) -> None:
        """A job left PROCESSING means the process died mid-run. Silently
        leaving it in that state would make the UI hang forever."""
        with get_session() as db:
            stale = db.query(Job).filter(Job.status == JobStatus.PROCESSING.value).all()
            for job in stale:
                job.status = JobStatus.FAILED.value
                job.error = "Interrupted: the server stopped while this job was running."
                job.completed_at = utcnow()
            if stale:
                db.commit()

    # --- submission -----------------------------------------------------

    async def submit(
        self, job_type: str, project_id: str | None = None, params: dict | None = None
    ) -> str:
        if job_type not in self._handlers:
            raise ValueError(f"No handler registered for job type '{job_type}'")

        job_id = new_id()
        with get_session() as db:
            db.add(
                Job(
                    id=job_id,
                    project_id=project_id,
                    type=job_type,
                    status=JobStatus.QUEUED.value,
                    progress=0.0,
                    params=params or {},
                )
            )
            db.commit()

        self._cancels[job_id] = asyncio.Event()
        await self._queue.put(job_id)
        return job_id

    def cancel(self, job_id: str) -> bool:
        event = self._cancels.get(job_id)
        if event is None:
            return False
        event.set()
        return True

    # --- progress fan-out -----------------------------------------------

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(job_id, []).append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(job_id, [])
        if queue in subs:
            subs.remove(queue)
        if not subs:
            self._subscribers.pop(job_id, None)

    async def _publish(self, job_id: str, payload: dict) -> None:
        for queue in list(self._subscribers.get(job_id, [])):
            await queue.put(payload)

    async def _update(self, job_id: str, **fields) -> None:
        with get_session() as db:
            job = db.get(Job, job_id)
            if job is None:
                return
            for key, value in fields.items():
                if value is not None or key == "error":
                    setattr(job, key, value)
            db.commit()
            snapshot = job_to_dict(job)
        await self._publish(job_id, snapshot)

    # --- worker ---------------------------------------------------------

    async def _worker(self) -> None:
        while self._running:
            try:
                job_id = await self._queue.get()
            except asyncio.CancelledError:
                return

            try:
                await self._run_one(job_id)
            except asyncio.CancelledError:
                return
            except Exception:
                traceback.print_exc()
            finally:
                self._queue.task_done()

    async def _run_one(self, job_id: str) -> None:
        with get_session() as db:
            job = db.get(Job, job_id)
            if job is None:
                return
            job_type = job.type
            project_id = job.project_id
            params = dict(job.params or {})

        cancel_event = self._cancels.setdefault(job_id, asyncio.Event())

        if cancel_event.is_set():
            await self._update(
                job_id,
                status=JobStatus.CANCELLED.value,
                completed_at=utcnow(),
            )
            return

        await self._update(
            job_id,
            status=JobStatus.PROCESSING.value,
            started_at=utcnow(),
            current_stage="starting",
        )

        ctx = JobContext(
            job_id=job_id,
            project_id=project_id,
            params=params,
            _runner=self,
            _cancel=cancel_event,
        )

        try:
            result = await self._handlers[job_type](ctx)
        except JobCancelled:
            await self._update(
                job_id,
                status=JobStatus.CANCELLED.value,
                current_stage="cancelled",
                completed_at=utcnow(),
            )
        except Exception as exc:
            # Preserve the message for the user; full trace to the console.
            traceback.print_exc()
            await self._update(
                job_id,
                status=JobStatus.FAILED.value,
                error=str(exc) or exc.__class__.__name__,
                completed_at=utcnow(),
            )
        else:
            await self._update(
                job_id,
                status=JobStatus.COMPLETED.value,
                progress=1.0,
                current_stage="done",
                result=result or {},
                completed_at=utcnow(),
            )
        finally:
            self._cancels.pop(job_id, None)


def job_to_dict(job: Job) -> dict:
    def iso(value):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    return {
        "id": job.id,
        "project_id": job.project_id,
        "type": job.type,
        "status": job.status,
        "progress": job.progress,
        "current_stage": job.current_stage,
        "error": job.error,
        "result": job.result or {},
        "created_at": iso(job.created_at),
        "started_at": iso(job.started_at),
        "completed_at": iso(job.completed_at),
    }


runner = JobRunner(concurrency=1)
