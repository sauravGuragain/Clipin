# Clipper AI

Turns long-form podcasts into short-form vertical clips. Local-first: runs on
your machine, no cloud APIs required.

**Status: Phase 4 complete** — end to end. Upload a podcast, get watchable vertical clips with captions.

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
pytest -q          # 129 tests, repeatable
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
| POST | `/api/projects/{id}/transcribe` | Queue transcription (`?force=true`) |
| GET | `/api/projects/{id}/transcript` | Stats, issues, preview (`?full=true` for segments) |
| GET | `/api/transcription/backends` | Which backends are available here |
| POST | `/api/projects/{id}/clips` | Generate clips (`?count=`, `?crop_strategy=`, `?captions=`) |
| GET | `/api/projects/{id}/clips` | List clips (`?sort=index\|duration\|start`) |
| GET | `/api/clips/{id}/video` | Stream a rendered clip |
| DELETE | `/api/clips/{id}` | Remove one clip |

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

## Phase 3: transcription

Backends live behind one interface and all return the same `Transcript`:

| Backend | When | Notes |
|---|---|---|
| `mlx-whisper` | Apple Silicon (default) | Metal via MLX. Model: `mlx-community/whisper-large-v3-turbo`, ~1.6 GB on first run |
| `faster-whisper` | Fallback / cross-check | **CPU-only on macOS** — CTranslate2 has no Metal backend |
| `stub` | Tests only | Never auto-selected |

### First run

```bash
pip install -r requirements.txt      # installs mlx-whisper on Apple Silicon
```

### The Phase 3 gate — do this before Phase 4

Word timestamps that are subtly wrong will not crash anything. They produce
captions that feel slightly off and clips that start a beat late, and no amount
of staring at JSON will reveal it. So look at them:

```bash
python scripts/verify_word_timestamps.py YOUR_PODCAST.mp4 --start 300 --duration 40
```

This transcribes a 40-second excerpt, burns the raw word timings on as
highlighted captions, and writes `data/verification.mp4`. Watch it. The
highlight should land on each word as it is spoken.

- Consistently early or late → offset bug
- Drift growing through the clip → sample-rate mismatch
- Randomly wrong → the backend's alignment is unreliable; compare against
  `--backend faster-whisper` before trusting either

**Everything in Phase 9 depends on this being right.**

### Chunking

A 90-minute file is transcribed in 5-minute chunks so progress is real rather
than a bar frozen at 0% for six minutes. Chunks overlap by 2s for context, and
each word is committed to exactly one chunk by a midpoint rule, so seams neither
duplicate nor drop words. `TRANSCRIBE_CHUNK_SECONDS=0` disables chunking: one
pass, best context, no progress.

### Captions arrived early

`app/services/captions/ass.py` exists already because verifying timestamps
requires rendering them. It is minimal — one style — but it is the real
mechanism Phase 9 builds on: per-word colour overrides and `\t` scale
transforms, rendered by libass inside FFmpeg. Proving it works removes the
biggest unknown from that phase.

## Phase 4: the end-to-end slice

Upload → extract audio → transcribe → **Generate clips** → vertical MP4s with
burned captions, previewable in the browser.

What is real: transcription, boundary snapping, the 9:16 crop, caption timing
and rendering, quality checking.

What is deliberately naive: **clip selection**. Anchors are spaced evenly
through the podcast — the strategy is literally named `even`. There is no AI
here and no pretence of one. Phase 6 replaces the anchor source; everything
downstream stays.

### Crop strategies

| Strategy | Behaviour |
|---|---|
| `center` (default) | Largest 9:16 rectangle, centred. Wrong for two-person podcasts — Phase 8 fixes that |
| `fit` | Whole frame letterboxed onto a blurred copy of itself. Use when cropping would cut someone out |

### Quality check

Every render is probed after the fact: dimensions, duration against what was
requested, presence of an audio stream, readability. A render exiting 0 does not
mean the file is usable, and finding that out after exporting twenty clips is
too late.

### Two bugs worth recording

Both passed every automated test and were caught by looking at a frame.

**Captions flickered.** Each subtitle event was timed to its own word's
duration, so in the 0.08s gaps between words the screen went blank. The ASS was
valid and the render succeeded. Lines are now continuous — each event holds
until the next word begins, and the finished line lingers for `line_hold`.

**Every clip exceeded its maximum.** Padding was applied after the duration
limit, so a 60s cap produced 60.5s clips. Padding now takes only the available
headroom.

Both have regression tests. The lesson generalises: for anything that renders,
watch the output.

## On React

Deferred, deliberately. Between now and the results grid the UI is a form and a
progress bar; React would add a Vite scaffold, TS config, a dev proxy and a
build step for no new capability. It earns its place at Phase 11, where clip
cards, sorting and review controls arrive.

## Next: Phase 5

Real clip boundaries: filler-word ratio, sentence-quality scoring, a duration
constraint solver, and better opener rejection.
