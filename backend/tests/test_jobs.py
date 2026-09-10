"""Tests for the job system and audio extraction.

These exercise the real runner with real asyncio, not mocks — the states that
matter (cancellation mid-flight, failure capture, crash recovery) only appear
when the machinery actually runs.
"""

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from app.core.jobs import JobCancelled, JobRunner, job_to_dict
from app.models.db import Job, JobStatus, get_session, new_id
from app.services.audio import TARGET_CHANNELS, TARGET_SAMPLE_RATE, extract_audio


@pytest.fixture
async def runner():
    r = JobRunner(concurrency=1)
    await r.start()
    yield r
    await r.stop()


def status_of(job_id: str) -> str:
    with get_session() as db:
        return db.get(Job, job_id).status


async def wait_for_terminal(job_id: str, timeout: float = 10.0) -> str:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        s = status_of(job_id)
        if s in {"COMPLETED", "FAILED", "CANCELLED"}:
            return s
        await asyncio.sleep(0.02)
    raise AssertionError(f"Job {job_id} did not finish within {timeout}s")


class TestJobLifecycle:
    @pytest.mark.asyncio
    async def test_successful_job_completes(self, runner):
        async def handler(ctx):
            await ctx.progress(0.5, "halfway")
            return {"value": 42}

        runner.register("ok", handler)
        job_id = await runner.submit("ok")
        assert await wait_for_terminal(job_id) == JobStatus.COMPLETED.value

        with get_session() as db:
            job = db.get(Job, job_id)
            assert job.progress == 1.0
            assert job.result == {"value": 42}
            assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_failing_job_records_message(self, runner):
        async def handler(ctx):
            raise ValueError("something specific went wrong")

        runner.register("bad", handler)
        job_id = await runner.submit("bad")
        assert await wait_for_terminal(job_id) == JobStatus.FAILED.value

        with get_session() as db:
            assert "something specific" in db.get(Job, job_id).error

    @pytest.mark.asyncio
    async def test_cancellation_stops_job(self, runner):
        started = asyncio.Event()

        async def handler(ctx):
            started.set()
            for _ in range(200):
                ctx.raise_if_cancelled()
                await asyncio.sleep(0.02)
            return {}

        runner.register("slow", handler)
        job_id = await runner.submit("slow")
        await asyncio.wait_for(started.wait(), timeout=5)
        assert runner.cancel(job_id) is True
        assert await wait_for_terminal(job_id) == JobStatus.CANCELLED.value

    @pytest.mark.asyncio
    async def test_unknown_job_type_rejected(self, runner):
        with pytest.raises(ValueError):
            await runner.submit("does_not_exist")

    @pytest.mark.asyncio
    async def test_cancel_unknown_job_returns_false(self, runner):
        assert runner.cancel("nonexistent") is False

    @pytest.mark.asyncio
    async def test_progress_is_clamped(self, runner):
        async def handler(ctx):
            await ctx.progress(-5.0, "under")
            under = db_progress(ctx.job_id)
            await ctx.progress(99.0, "over")
            return {"under": under, "over": db_progress(ctx.job_id)}

        def db_progress(job_id):
            with get_session() as db:
                return db.get(Job, job_id).progress

        runner.register("clamp", handler)
        job_id = await runner.submit("clamp")
        await wait_for_terminal(job_id)
        with get_session() as db:
            result = db.get(Job, job_id).result
        assert result["under"] == 0.0
        assert result["over"] == 1.0

    @pytest.mark.asyncio
    async def test_jobs_run_sequentially_at_concurrency_one(self, runner):
        """Memory budget depends on this. If it ever runs jobs in parallel,
        Whisper and a render could coexist and swap the machine."""
        active = 0
        peak = 0

        async def handler(ctx):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1
            return {}

        runner.register("counted", handler)
        ids = [await runner.submit("counted") for _ in range(4)]
        for job_id in ids:
            await wait_for_terminal(job_id)
        assert peak == 1


class TestCrashRecovery:
    @pytest.mark.asyncio
    async def test_interrupted_jobs_marked_failed_on_start(self):
        stuck_id = new_id()
        with get_session() as db:
            db.add(Job(id=stuck_id, type="x", status=JobStatus.PROCESSING.value))
            db.commit()

        r = JobRunner(concurrency=1)
        await r.start()
        try:
            with get_session() as db:
                job = db.get(Job, stuck_id)
                assert job.status == JobStatus.FAILED.value
                assert "Interrupted" in job.error
        finally:
            await r.stop()

    @pytest.mark.asyncio
    async def test_recovery_leaves_finished_jobs_alone(self):
        """The sweep must only touch PROCESSING. Rewriting a COMPLETED job
        would destroy a real result."""
        done_id, queued_id = new_id(), new_id()
        with get_session() as db:
            db.add(Job(id=done_id, type="x", status=JobStatus.COMPLETED.value))
            db.add(Job(id=queued_id, type="x", status=JobStatus.QUEUED.value))
            db.commit()

        r = JobRunner(concurrency=1)
        await r.start()
        try:
            with get_session() as db:
                assert db.get(Job, done_id).status == JobStatus.COMPLETED.value
                assert db.get(Job, queued_id).status == JobStatus.QUEUED.value
        finally:
            await r.stop()


class TestJobSerialisation:
    def test_job_to_dict_shape(self):
        job_id = new_id()
        with get_session() as db:
            db.add(Job(id=job_id, type="t", status="QUEUED", project_id="p1"))
            db.commit()
            d = job_to_dict(db.get(Job, job_id))
        assert set(d) >= {
            "id", "project_id", "type", "status", "progress",
            "current_stage", "error", "result", "created_at",
        }


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestAudioExtraction:
    @pytest.fixture
    def sample_video(self, tmp_path):
        out = tmp_path / "sample.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:d=3",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(out)],
            check=True,
        )
        return out

    @pytest.mark.asyncio
    async def test_produces_16k_mono_wav(self, sample_video, tmp_path):
        dest = tmp_path / "audio.wav"
        await extract_audio(sample_video, dest, duration=3.0)

        assert dest.exists() and dest.stat().st_size > 0

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels,codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1", str(dest)],
            capture_output=True, text=True, check=True,
        ).stdout.split()

        assert probe[0] == "pcm_s16le"
        assert int(probe[1]) == TARGET_SAMPLE_RATE
        assert int(probe[2]) == TARGET_CHANNELS

    @pytest.mark.asyncio
    async def test_missing_source_raises(self, tmp_path):
        from app.services.audio import AudioExtractionError

        with pytest.raises(AudioExtractionError):
            await extract_audio(tmp_path / "nope.mp4", tmp_path / "out.wav", 1.0)

    @pytest.mark.asyncio
    async def test_no_partial_file_left_behind_on_failure(self, tmp_path):
        from app.services.audio import AudioExtractionError

        junk = tmp_path / "junk.mp4"
        junk.write_bytes(b"\x00" * 4096)
        dest = tmp_path / "audio.wav"

        with pytest.raises(AudioExtractionError):
            await extract_audio(junk, dest, 1.0)

        assert not dest.exists()
        assert not list(tmp_path.glob("*.partial.wav"))
