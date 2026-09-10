# STACK.md — Machine-specific decisions

Companion to `PLAN.md`. Resolved against measured environment output, September 2026.
Where this file and PLAN.md disagree, **this file wins** — it is based on the actual machine.

---

## 1. Target machine

| Property | Value | Consequence |
|---|---|---|
| Model | MacBook Air, Apple M5 (2026) | Fanless — throttles on sustained load |
| macOS | 26.5.2 (25F84) | Above every wheel's minimum (macosx_13_0) |
| Arch | arm64 | MLX and Metal available |
| CPU | 10-core (4 performance + 6 efficiency) | Cap parallel encodes at 2–3, not 10 |
| GPU | 8 or 10-core, Neural Accelerator per core | MLX inference path |
| Neural Engine | 16-core | Not directly used by this stack |
| Memory bandwidth | 153 GB/s | Sets local LLM token/sec ceiling |
| **RAM** | **16 GB unified, not upgradeable** | **The binding constraint. See §3** |
| Disk free | 818 GB | Ample; no cache pressure expected |
| Media engine | HW H.264/HEVC encode, AV1 **decode** | videotoolbox for previews; AV1 sources decode cheaply |
| Python | 3.14.6 (Homebrew, arm64 native) | Viable, but see §2 |
| Node | v20.20.2 / npm 10.8.2 | LTS window closed — upgrade |
| FFmpeg | **MISSING** | Blocker, install first |
| Ollama | **MISSING** | Only needed for the offline LLM path |
| Toolchain | Homebrew 6.0.13, git 2.50.1, CLT present | Ready |

Not Rosetta: `platform.machine()` reports `arm64`, matching `uname -m`.

---

## 2. Python version

Wheel availability checked against PyPI directly, latest release of each package,
macOS arm64:

| Package | Latest | cp312 | cp314 | arm64 mac |
|---|---|---|---|---|
| mlx | 0.32.2 | ✅ | ✅ | ✅ |
| mlx-whisper | 0.4.3 | ✅ | ✅ | ✅ |
| opencv-python | 5.0.0.93 | abi3 | abi3 | ✅ |
| onnxruntime | 1.29.0 | ✅ | ✅ | ✅ |
| fastembed | 0.8.0 | ✅ | ✅ | ✅ |
| ctranslate2 | 4.8.2 | ✅ | ✅ | ✅ |
| faster-whisper | 1.2.1 | ✅ | ✅ | ✅ |
| scenedetect | 0.7.1 | ✅ | ✅ | ✅ |
| numba / llvmlite | 0.67.0 / 0.49.0 | ✅ | ✅ | ✅ |
| mediapipe | 1.0.1 | py3-none | py3-none | ✅ |

**Python 3.14 would work.** The historic laggards (numba, llvmlite, av) have caught up.

**Recommendation: Python 3.12 in a venv anyway.** Rationale: thirteen phases means
repeatedly adding libraries not on this list, and the failure mode — a source build
against a missing wheel, mid-phase — costs more than one `brew install` now. This is
insurance, not a blocker. Using 3.14.6 is a defensible alternative.

Never install project dependencies into the Homebrew Python. Always the venv.

---

## 3. The 16 GB budget

Approximate resident memory per component:

| Component | Resident |
|---|---|
| macOS baseline | 4–5 GB |
| Whisper large-v3-turbo (MLX, fp16) | ~1.6 GB |
| 8B LLM at Q4 + KV cache | 5–6 GB |
| FFmpeg render process | 0.5–1 GB each |
| Vite dev server + browser | 2–3 GB |

Any two of the large three (Whisper, LLM, parallel renders) plus a browser will swap.

**Rules this imposes on the implementation:**

1. Pipeline stages run **strictly sequentially**. Never overlap transcription with
   clip discovery.
2. Set `OLLAMA_KEEP_ALIVE=0` so the model unloads immediately after the discovery
   stage rather than idling in memory for five minutes.
3. Explicitly free the Whisper model before the LLM stage begins.
4. Render concurrency configurable, **default 2**, hard max 3.
5. Add a memory-pressure check before the render stage; warn rather than swap.

This is also the strongest practical argument for cloud clip discovery: it removes
5–6 GB from the peak.

---

## 4. Resolved stack

### Transcription
**`mlx-whisper`, model `large-v3-turbo`.** Metal-accelerated, ~1.6 GB, near-large
accuracy at several times the speed.

Fallback if word timestamps prove unreliable: `whisper.cpp` with `--dtw large.v3.turbo`,
which uses DTW alignment for token timing.

Keep `faster-whisper` behind the same interface for portability, but note it is
**CPU-only on macOS** (CTranslate2 has no Metal backend) — it is a fallback, never
the default here.

**Phase 3 gate:** transcribe a 2-minute clip with `word_timestamps=True`, burn the
raw word timings as captions, and watch it. If words drift more than ~80 ms the
animated caption work in Phase 9 will not look right. Fix this before proceeding.

### Clip discovery LLM
**Default: cloud (`AnthropicProvider`).** ~18k tokens for a 90-minute transcript,
a few cents per podcast, roughly a minute of wall time, and materially better
editorial judgment.

**Offline path: Ollama, a ~7–9B instruct model at Q4_K_M.** At 153 GB/s expect
20–28 tok/s generation. Select the model on JSON/schema-following reliability rather
than general benchmark scores — the discovery stage returns structured candidates and
a model that emits malformed JSON costs more in retries than it saves. Check the
current Ollama library at install time rather than pinning a name from an older list.

Both behind the `LLMProvider` protocol from PLAN.md §1. Benchmark both on one episode
in Phase 6 and decide with real output in front of you.

### Embeddings — amends PLAN.md §1.5
**`fastembed` with `BAAI/bge-small-en-v1.5`.** ONNX runtime, ~130 MB, CPU, no PyTorch.

Replaces `sentence-transformers`, which pulls torch (~2.5–3 GB). Combined with
mlx-whisper and OpenCV DNN, this makes the whole stack **torch-free** — a meaningful
saving on a 16 GB machine that cannot be upgraded.

### Face detection — amends PLAN.md §1.2
**YuNet via `cv2.FaceDetectorYN`**, ONNX model from opencv_zoo, ~2 ms/frame at 320×320.

Correction to PLAN.md: MediaPipe now ships a clean `py3-none-macosx_11_0_arm64` wheel,
so the install-fragility argument no longer applies. YuNet remains the default for
leanness — no extra dependency, no protobuf version pinning — and MediaPipe stays
available behind `FaceDetector` if the lip-motion heuristic needs finer landmarks.

### Rendering
`libx264` CRF 19, preset `medium` for final exports — better compression efficiency
per byte than the hardware encoder.

`h264_videotoolbox` for preview and regenerate-on-edit renders. It uses the media
engine rather than the CPU, which matters on a fanless chassis: sustained parallel
x264 encoding will throttle the machine and slow everything else in the pipeline.

Make the encoder a config value, not a constant.

### Frontend
Node 20.20.2 works today, but Node 20's LTS window has ended. Move to **Node 22 or 24
LTS** before Phase 1 — Vite and the tooling chain will assume it shortly.

---

## 5. Install sequence

Run one at a time; verify each before the next.

```bash
# 1. FFmpeg — blocker
brew install ffmpeg

# 2. Verify libass and libx264 are present (Phase 9 depends on libass)
ffmpeg -hide_banner -filters  | grep subtitles
ffmpeg -hide_banner -encoders | grep -E "libx264|h264_videotoolbox"

# 3. Python 3.12
brew install python@3.12

# 4. Node LTS
brew install node@24

# 5. Ollama — only if you want the offline LLM path
brew install ollama
```

Then, in the project directory:

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -V          # expect 3.12.x
pip install --upgrade pip
```

---

## 6. requirements.txt

```
# --- core service ---
fastapi
uvicorn[standard]
pydantic
pydantic-settings
sqlalchemy
python-multipart
sse-starlette

# --- transcription (Apple Silicon) ---
mlx
mlx-whisper

# --- transcription fallback ---
faster-whisper

# --- vision ---
opencv-python
scenedetect

# --- embeddings (torch-free) ---
fastembed

# --- LLM providers ---
anthropic
openai
httpx

# --- utilities ---
numpy
python-dotenv

# --- dev ---
pytest
pytest-asyncio
ruff
```

Pin exact versions with `pip freeze > requirements.lock` once Phase 1 runs clean.
Deliberately absent: `torch`, `sentence-transformers`, `mediapipe`.

Additional download, not a pip package — the YuNet ONNX model from opencv_zoo.
Fetch it in a setup script and cache under `data/models/`.

---

## 7. Expected performance — 90-minute podcast → 20 clips

| Stage | Estimate | Notes |
|---|---|---|
| Media inspection | seconds | ffprobe |
| Audio extraction | under 1 min | I/O bound |
| Transcription | 3–8 min | large-v3-turbo, MLX |
| Clip discovery (cloud) | 1–2 min | ~12 chunks, parallel API calls |
| Clip discovery (local 8B) | 8–15 min | ~600 output tokens × 12 chunks at ~24 tok/s |
| Dedup + rerank | under 1 min | embeddings are cheap |
| Boundary snapping | seconds | pure Python |
| Face detect + crop paths | 2–5 min | sampled at 4–6 fps, not every frame |
| Silence detection | under 1 min | ffmpeg silencedetect |
| Caption generation (ASS) | seconds | text generation only |
| Render 20 clips | 5–12 min | x264, 2–3 parallel |
| Quality check | under 1 min | |
| **Total, cloud discovery** | **~15–30 min** | |
| **Total, local discovery** | **~25–45 min** | |

**These are derived from memory bandwidth, core count and codec throughput — not
measured on an M5.** Instrument every stage from Phase 2 and replace this table with
real numbers. Treat a >2× miss as a signal that something is misconfigured (Rosetta,
CPU-only Whisper, thermal throttling) rather than as the machine being slow.

Thermal note: the Air has no fan. A 20-clip render at high concurrency will throttle
and can end up slower than the same job at concurrency 2. Measure before increasing it.

---

## 8. Open items carried into Phase 1

1. Source podcasts — resolution, single camera or multicam, one or two people on screen?
   Determines whether the §1.3 lip-motion heuristic is needed at all.
2. Cloud provider approved for clip discovery, or offline-only?
3. Default clip duration band.
4. Confirm Python 3.12 vs staying on 3.14.6.
