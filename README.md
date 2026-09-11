# Clipper AI

Turns long-form podcasts into short-form vertical clips. Local-first: runs on
your machine, no cloud APIs required.

**Status: Phase 6 complete** — the model now chooses the moments.

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
pytest -q          # 244 tests, repeatable
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
| POST | `/api/projects/{id}/discover` | Find moments with the LLM (`?provider=`, `?model=`) |
| GET | `/api/projects/{id}/candidates` | Discovered moments, scores and rejections |
| GET | `/api/llm/providers` | Which providers and Ollama models are available |
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

## Phase 5: clip boundaries

Phase 4 grew clips greedily — start past the anchor, add segments until long
enough, stop. Legal clips, often badly cut. Phase 5 **searches** candidate
start/end pairs and scores them.

On a synthetic transcript mixing complete sentences, fragments and filler-heavy
runs, mean boundary score went from **0.44 to 0.80** against the same anchors.

### Scoring dimensions

All weights configurable in `.env`, per spec 28.

| Dimension | Default | What it catches |
|---|---|---|
| `opener` | 0.28 | Clips that begin mid-thought |
| `ends_sentence` | 0.24 | Clips that stop before the point lands |
| `starts_sentence` | 0.18 | Cuts into the middle of a sentence |
| `not_dangling` | 0.12 | Endings on "and", "the", "to" |
| `low_filler` | 0.10 | Runs of "um", "you know", "kind of" |
| `duration_fit` | 0.08 | Prefers the middle of the allowed range |

### Discourse markers vs subordinators

Not all weak openers are equally bad, and treating them alike produced visibly
wrong clips.

- **Discourse markers** — "so", "well", "okay", "anyway". Weak mid-flow, but
  perfectly natural after a pause: that is how people start a new thought aloud.
  **Forgiven** when preceded by ≥0.6s of silence.
- **Subordinators** — "which", "because", "and", "though". These refer back to
  something the viewer never saw. A pause does not repair a grammatically
  dependent clause, so they are **never forgiven**.

### Silence-aware padding

Padding takes half of whatever silence actually surrounds the clip rather than a
fixed amount, so it can never bite into the neighbouring word, with a guaranteed
sliver of head room so the first consonant is not clipped.

### Two bugs found by reading the output

**Multi-word fillers never fired.** `FILLER_PHRASES` held entries like
`"you know"` and `"i mean"`, matched against single cleaned words. They could
not match, so every one was dead code.

**Phrase openers slipped through.** Only the first word was checked, so
"you know it was actually kind of like" scored 0.94 — "you" is not a weak
opener. Openers are now matched as phrases before single words.

### Two bugs found in manual verification

Both had green test suites around them.

**Captions were superimposed.** `line_hold` (added in Phase 4 to stop flicker)
was not clamped to the following line, so each line's final event ran into the
next line's first event. libass does not stack colliding events — it draws them
**on top of each other at the same position**, producing garbled doubled glyphs.
Measured: 33 of 189 sampled frames. The existing continuity test only checked
*within* a line and never saw it.

**The UI was served stale.** `FileResponse` sends ETag and Last-Modified but no
`Cache-Control`. Browsers apply heuristic freshness to such a response and will
serve a cached page without revalidating, which hid an entire phase's UI
changes. Now `no-cache, must-revalidate` — revalidate every time, with the ETag
still yielding a cheap 304.

### A test that passed against the bug it was written for

Worth recording because it is the more useful lesson. The first render-level
test counted horizontal *bands* of text, assuming libass would stack colliding
captions. It superimposes them instead, so both lines occupy identical rows and
the band count stays at one — the test passed against the exact bug it existed
to catch.

The replacement counts **highlighted words**: exactly one word carries the
highlight colour at a time, so two separated clusters of it mean two events are
live. It was verified by reverting the fix and confirming it fails
(18 offending frames), then restoring it.

Every regression test in `test_regressions_phase5.py` was checked this way. A
regression test that has never been seen to fail is an assumption, not a test.

### Transcription timing is now recorded

The transcribe job result carries `elapsed_seconds` and `realtime_factor`,
measured on real work. A 90-minute podcast at 1x realtime is a very different
product from one at 20x, and that number should come from measurement rather
than estimation.

## Phase 6: AI clip discovery

Evenly spaced anchors are gone. A local model reads the transcript and says
where the interesting moments are; Phase 5's solver still decides where to cut.
Discovery chooses **where to look**, the solver chooses **where to cut**.

### Setup

```bash
ollama serve
ollama pull qwen3:8b       # or any ~7-9B instruct model
```

No cloud key is required, ever. Anthropic and OpenAI providers exist so the
local-only decision stays reversible, not because anything needs them.

Then in the UI: **Find moments** → **Generate clips**. Clip cards show the
model's hook, topic and category.

`keep_alive=0` is set on every Ollama call. That is deliberate: leaving a 5 GB
model resident after discovery would collide with the render stage on a 16 GB
machine (STACK.md §3).

### Context window

Ollama derives a default context from available VRAM and may pick **4096**.
`num_ctx` covers prompt *and* generation, so an undersized window does not
error — it silently drops the oldest tokens, which here is the transcript. The
model then produces confident JSON about text it never received, the timestamp
validator rejects nearly all of it, and the whole thing looks like a useless
model rather than a misconfiguration.

So `num_ctx` is set explicitly (`LLM_NUM_CTX`, default 8192) rather than
inherited, output is capped at 768 tokens (three structured candidates is ~300;
2048 was reserving context for output that never arrives), and a preflight check
runs against the widest window before any generation starts. Shipped defaults
leave about 70% of the window free.

### Reasoning models

qwen3 and deepseek-r1 think before answering. Ollama returns that reasoning in a
separate field so it never reaches the JSON parser — but it is still generated,
still costs seconds, and still counts against `num_predict`. A long think can
therefore truncate the answer after it, and a truncated response is a failed
window.

Measured on an M5: a trivial JSON request produced a 50-token answer and **281
completion tokens**. Reasoning is off by default for structured extraction
(`LLM_THINK=0`) and left on for free-form generation, where it may genuinely
help. Set `LLM_THINK=1` to compare.

If a think ever does overrun the budget, Ollama returns reasoning with empty
content — that now raises a message naming `LLM_MAX_OUTPUT` and `LLM_THINK`
rather than a bare "empty message".

### Malformed output is the normal case

A local 8B model asked for JSON complies most of the time and produces
something adjacent the rest of the time. All of these are handled and tested:

markdown fences, prose preamble and postamble, trailing commas, unquoted keys,
smart quotes, a bare object where an array was asked for, `{"clips": [...]}`
wrapping, braces inside string values, and an outright refusal.

Repair runs before retry, because a retry costs a full generation — 20-30
seconds locally — and repair costs microseconds.

### Hallucinated timestamps

Models invent numbers. Every returned timestamp is snapped to a real word
boundary in the transcript; anything more than 6s from any real boundary is
treated as fabricated and the candidate is **discarded, not guessed at**.
Rejections are reported rather than swallowed — if a model produces nothing
usable, the reasons are the only way to distinguish a bad prompt from a bad
model.

### A candidate is an anchor, not a cut

Found on the first real run against qwen3:8b: asked for 25-60 second moments it
returned 9.7 second ones, every candidate was rejected as "too short", and
discovery failed on an entire podcast with zero results.

The rejection was wrong. A candidate only says *where* something interesting
happens — Phase 5's solver decides the actual boundaries and expands a 10s
pointer into a valid 42s clip. The code already made this argument for
over-long candidates and kept them; too-short was rejected out of plain
inconsistency.

Duration violations are now recorded as issues and the anchor is kept. Only a
span under two seconds is dropped, because below that it carries no information
about location either. Verified end to end: 10.3s anchors produce 42.2s clips
at boundary score 0.996, with the model's hook attached.

The prompt now also tells the model that the end will be adjusted to a sentence
boundary, so it need not truncate a thought to hit a target length.

### Rejections report the pattern

The failure above said `too short (9.7s) at 145.5` — one instance out of dozens
of identical failures, which reads like an unlucky candidate rather than a
systematic problem. Rejections are now grouped by kind with counts:
`18 candidate(s) rejected: 15 x spans too short..., 3 x invented timestamps`.

### Score comparability

Windows are scored in independent calls, so an 87 from one and an 87 from
another do not mean the same thing.

Where a window returned enough candidates to have a distribution, scores are
z-scored within it to remove that call's calibration drift. Where it did not —
and **a model asked for three moments routinely returns one** — there is nothing
to z-score against. The first implementation assigned a neutral 0.5 in that
case, which left every candidate tied and produced completely unranked output.
Small groups now fall back to a global min-max over raw scores, compressed into
0.15-0.85 and carrying lower `confidence`, recording that the comparison is
across calls rather than within one.

## On React

Deferred, deliberately. Between now and the results grid the UI is a form and a
progress bar; React would add a Vite scaffold, TS config, a dev proxy and a
build step for no new capability. It earns its place at Phase 11, where clip
cards, sorting and review controls arrive.

## Next: Phase 7

Full scoring and ranking: semantic deduplication via embeddings, a listwise
rerank pass over the survivors, and configurable dimension weights.
