# Clipper AI

Turns long-form podcasts into short-form vertical clips. Local-first: runs on
your machine, no cloud APIs required.

**Status: Phase 2 complete** — upload, media inspection, background jobs, audio extraction.

See `PLAN.md` for architecture and `STACK.md` for machine-specific decisions.

---

## Setup

From `/Users/sauravguragain/clipper-ai`:

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
./start.sh
```

Then open http://127.0.0.1:8000

The server prints an environment report on startup and **refuses to boot** if a
required capability is missing. That is deliberate — a missing libass would
otherwise surface as a broken render many phases later.

## Test

```bash
source .venv/bin/activate
pytest -q          # 30 tests, repeatable
```

Tests run against a throwaway database in a temp directory, never
`data/clipper.db`. `backend/tests/test_isolation.py` enforces this — if a future
change breaks the isolation, those tests fail loudly rather than letting the
suite quietly corrupt real project data.

## Generate a test video

No real podcast needed for Phase 1:

```bash
./scripts/make_test_video.sh 180
```

Produces `data/uploads/test_podcast.mp4` — 1920×1080, 29.97 fps, stereo AAC,
two "speaker" panels and a burned-in timecode for verifying cut accuracy later.

**It contains no real speech.** Phase 3 (transcription) and Phase 6 (clip
discovery) need a genuine podcast with dialogue. This fixture cannot test them.

---

## What works

**Phase 1**
- Drag-and-drop upload, streamed to disk (never loaded into memory)
- Real `ffprobe` inspection: duration, resolution, fps, codecs, sample rate, channels
- Rejects unsupported formats, empty files, corrupt media, and files with no audio track
- Persists project metadata to SQLite; media stays on disk
- Startup capability check for ffmpeg, ffprobe, libass, and encoders

**Phase 2**
- In-process job system: QUEUED / PROCESSING / COMPLETED / FAILED / CANCELLED
- Audio extraction to 16 kHz mono PCM WAV, the format Whisper wants
- **Real progress**, parsed from FFmpeg's own `-progress` stream — not estimated
- Live progress in the browser over Server-Sent Events
- Cancellation that actually terminates the running FFmpeg process
- Crash recovery: jobs left PROCESSING when the server died are marked FAILED at startup
- Extraction result cached; re-running returns immediately unless forced
- Atomic writes — an interrupted extraction never leaves a truncated WAV behind

### Concurrency is deliberately 1

`JobRunner(concurrency=1)` in `app/core/jobs.py`. This is a memory decision, not
a performance oversight: 16 GB is shared between CPU and GPU, and overlapping a
Whisper model with a render will swap the machine. A test asserts sequential
execution so this cannot regress silently.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/environment` | Capability report |
| GET | `/api/projects` | List projects |
| POST | `/api/projects` | Upload and inspect |
| GET | `/api/projects/{id}` | One project |
| DELETE | `/api/projects/{id}` | Remove project and its media |
| POST | `/api/projects/{id}/extract-audio` | Queue an extraction job (`?force=true` to bypass cache) |
| GET | `/api/jobs` | List jobs (`?project_id=`) |
| GET | `/api/jobs/{id}` | One job |
| GET | `/api/jobs/{id}/events` | SSE progress stream |
| POST | `/api/jobs/{id}/cancel` | Request cancellation |

---

## Notes on two deviations from PLAN.md

**The UI is a static page, not React yet.** Phase 1's deliverable is proving
that upload → ffprobe → database → display works. A static page does that in one
process with no Node dependency, and your Node is due an upgrade anyway. React
arrives in Phase 2, where background jobs and progress state give it something
real to manage.

**`FFMPEG_BIN` is configurable and startup checks capabilities.** This machine
has two Homebrew ffmpeg builds: `ffmpeg` (lean, no libass) and `ffmpeg-full`.
A future `brew upgrade` could relink the lean one. If startup reports the
subtitles filter missing, set an absolute path in `.env`:

```
FFMPEG_BIN=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg
FFPROBE_BIN=/opt/homebrew/opt/ffmpeg-full/bin/ffprobe
```

---

## On React

Deferred, deliberately. Between now and the results grid the UI is a form and a
progress bar; React would add a Vite scaffold, TS config, a dev proxy and a
build step for no new capability. It earns its place at Phase 11, where clip
cards, sorting and review controls arrive.

## Next: Phase 3

Whisper transcription behind a backend abstraction, word-level timestamps, and
a content-hash-keyed transcript cache.

**Phase 3 needs a real podcast.** The synthetic fixture has no speech and cannot
validate transcription.
