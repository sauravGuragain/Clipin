# Local AI Podcast Clipper — Architecture & Implementation Plan

Status: proposal, pending environment inspection and approval.
Target: macOS (Apple Silicon assumed, verify), local-first, optional cloud LLM.

---

## 0. Environment inspection (do this first)

Before writing code, capture:

```bash
sw_vers                                    # macOS version
uname -m                                   # arm64 vs x86_64
sysctl -n hw.memsize                       # RAM (unified memory on AS)
sysctl -n machdep.cpu.brand_string         # chip
python3 -V
node -v
ffmpeg -version | head -1
ollama --version
```

Decisions that depend on this output:

| Finding | Consequence |
|---|---|
| arm64 (Apple Silicon) | Use `mlx-whisper` or `whisper.cpp`; Ollama gets Metal acceleration |
| x86_64 (Intel Mac) | Use `faster-whisper` int8; expect ~3-5× slower transcription; local LLM likely impractical |
| RAM < 16 GB | Whisper `small`/`base`; local LLM not recommended — use cloud provider for clip discovery |
| RAM 16–24 GB | Whisper `medium`; ~7-8B instruct model at Q4 |
| RAM ≥ 32 GB | Whisper `large-v3`; ~14-32B model at Q4 |
| No FFmpeg | Hard blocker. `brew install ffmpeg`. Detect at startup, not mid-pipeline |

---

## 1. Architectural decisions and deviations from the spec

Your proposed structure is sound. Six changes, each with a reason.

### 1.1 Transcription backend must be abstracted (not just faster-whisper)

`faster-whisper` runs on CTranslate2, which has **no Metal backend**. On a Mac it is CPU-only — you would leave the GPU and Neural Engine idle on the single most expensive stage in the pipeline.

```
TranscriptionBackend (Protocol)
├── MLXWhisperBackend      # Apple Silicon default, Metal
├── WhisperCppBackend      # portable, Metal on AS, good word timestamps
└── FasterWhisperBackend   # fallback / Intel / Linux
```

All backends must return the same `Transcript` model. **Word-level timestamps are a hard requirement** — animated captions depend on them entirely. Verify word-timestamp fidelity on the chosen backend in Phase 3 before building anything on top of it.

### 1.2 YuNet, not MediaPipe, as the primary face detector

MediaPipe's arm64 macOS wheels have been an ongoing source of install failures and version pinning pain. OpenCV ships **YuNet** (`cv2.FaceDetectorYN`), a small ONNX detector: fast, accurate enough for podcast framing, zero extra dependencies.

```
FaceDetector (Protocol)
├── YuNetDetector          # default
└── MediaPipeDetector      # optional, when landmarks are needed
```

### 1.3 Active speaker detection — scope this honestly

Spec section 11 asks the crop to "follow whichever person is currently speaking." Face detection does not provide this. Real active-speaker detection (TalkNet-ASD and similar) is an audio-visual model — heavy, and a project in itself.

V1 approach, in descending order of confidence:

1. **Single face detected** → track it, smooth crop. Reliable.
2. **Two faces, lip-motion heuristic** → measure mouth-region pixel variance per face over a sliding window; attribute speech to the higher-variance face. Apply **hysteresis** (minimum 1.5–2 s dwell time before switching) so the crop doesn't strobe on backchannel noise.
3. **Low confidence / rapid crosstalk** → fall back to a stacked two-person 9:16 layout rather than guessing.

Full ASD, and audio diarization (pyannote, requires a gated HF token), are post-V1. Design `CropStrategy` so a `TalkNetStrategy` can drop in later.

### 1.4 Captions via ASS/libass, not frame compositing

Word highlighting, active-word emphasis and subtle scale animation are all achievable in ASS subtitles:

- one `Dialogue` event per word-boundary state, with `\c` colour override on the active word
- `\t(start,end,\fscx110\fscy110)` for scale animation
- `\an`, `\pos`, `\marginv` for position and safe margins
- `\blur`, `\bord`, `\shad` for the Bold Highlight and Podcast styles

FFmpeg burns these in via the `subtitles` filter at near-copy speed. Compositing PIL/Skia frames per word is 10–50× slower and buys you very little for these four styles. Keep a `CaptionRenderer` interface so a frame compositor can be added if a future style genuinely needs it.

Caption styles live in **data**, not code — a `styles/*.json` per style defining font, size, colours, position, animation params. Adding a style should require no Python changes.

### 1.5 Score calibration across chunks

A 90-minute podcast is ~13k words. That exceeds comfortable context for local models, so clip discovery is map-reduce over chunks. **LLM scores are not comparable across independent calls** — an 87 from chunk 3 and an 87 from chunk 11 mean different things.

Pipeline:

1. **Map** — per chunk, extract candidates with structured metadata (your section 8 schema).
2. **Normalize** — z-score each chunk's scores, or convert to within-chunk ranks.
3. **Deduplicate** — two passes: time-overlap IoU > 0.5, and semantic similarity via local embeddings (`all-MiniLM-L6-v2`, cosine > 0.85). Keep the higher-scoring member.
4. **Rerank** — one listwise LLM pass over the surviving top ~40, comparing them against each other. This is where cross-chunk comparability is actually established.
5. **Weighted final score** — apply your configurable dimension weights.

Deterministic dimensions (`duration_suitability`, `standalone_meaning` heuristics, filler ratio) are computed in Python, not asked of the LLM. Only genuinely subjective dimensions go to the model.

### 1.6 Phase reordering — end-to-end slice first

Your phase order produces no playable output until Phase 8. FFmpeg filter-graph problems, colour-space issues, audio drift and caption timing bugs are the things most likely to eat days, and they should surface while the codebase is small.

Revised ordering below (§4).

### 1.7 Minor: single process in production

Serve the built React bundle from FastAPI as static files. `start.sh` launches one process. Keep Vite dev server for development only. For a local single-user tool, two long-running processes is unnecessary operational surface.

Job system: **in-process**, `asyncio` for orchestration + `ProcessPoolExecutor` for CPU-bound stages. No Celery, no Redis, no broker. Progress via Server-Sent Events (simpler than WebSockets, sufficient for one-way progress). Job state persisted to SQLite so a crash mid-render leaves a resumable record.

---

## 2. Dependencies

### Local, required

| Component | Package | Notes |
|---|---|---|
| Video/audio | FFmpeg (system) | `brew install ffmpeg` — must include libass and libx264 |
| Transcription | `mlx-whisper` or `whisper.cpp` | Metal on Apple Silicon |
| Transcription fallback | `faster-whisper` | CPU |
| Vision | `opencv-python` | YuNet detector included |
| Scene detection | `scenedetect` | Content-aware detector |
| Embeddings | `sentence-transformers` | MiniLM, runs on MPS |
| Backend | `fastapi`, `uvicorn`, `pydantic`, `sqlalchemy` | |
| Testing | `pytest`, `pytest-asyncio` | |

### Local, optional

| Component | Package | Notes |
|---|---|---|
| Local LLM | Ollama | Metal-accelerated; model choice depends on RAM |
| Landmarks | `mediapipe` | Only if lip-motion heuristic needs finer landmarks |

### Cloud, entirely optional

| Component | Notes |
|---|---|
| Anthropic API | Better clip discovery and hook generation |
| OpenAI API | Alternative provider |

**No API key is ever required to produce clips.** `.env` only; never committed, never hard-coded. The GUI must visibly label which mode is active — a `LOCAL` / `CLOUD` badge on the clip discovery stage.

### Frontend

React 18, TypeScript, Vite, Tailwind. No component library needed for this surface area.

---

## 3. Local vs cloud capability split

| Stage | Local | Cloud optional | Quality gap without cloud |
|---|---|---|---|
| Media inspection | ✅ ffprobe | — | none |
| Audio extraction | ✅ FFmpeg | — | none |
| Transcription | ✅ Whisper | — | none (local Whisper is excellent) |
| Candidate discovery | ✅ local LLM | ✅ better | **significant** |
| Scoring / rerank | ✅ local LLM | ✅ better | **significant** |
| Deduplication | ✅ embeddings | — | none |
| Boundary snapping | ✅ deterministic | — | none |
| Face detect / crop | ✅ OpenCV | — | none |
| Silence removal | ✅ FFmpeg silencedetect | — | none |
| Captions | ✅ from transcript | — | none |
| Hook generation | ✅ local LLM | ✅ better | moderate |
| Render | ✅ FFmpeg | — | none |

The honest summary: **everything mechanical runs locally at full quality. Only editorial judgement degrades.** That is precisely why the provider abstraction matters — it is the one axis where paying money buys something real.

---

## 4. Revised phased plan

Each phase ends with: tests pass → app runs → functionality verified by hand → README updated → stop for review.

### Phase 1 — Skeleton and media inspection
FastAPI + React scaffold. Upload endpoint with chunked upload. `ffprobe` inspection → duration, resolution, fps, codecs, audio channels. SQLite project record. FFmpeg presence check at startup with a real installation message. **Deliverable:** upload a video, see its true metadata in the GUI.

### Phase 2 — Audio extraction and job system
Job model (queued/processing/completed/failed/cancelled), SSE progress, cancellation. Audio extraction to 16 kHz mono WAV. **Deliverable:** upload → background job → progress bar reaches 100% → WAV on disk.

### Phase 3 — Transcription
Backend abstraction, chosen implementation, `Transcript`/`Segment`/`Word` Pydantic models. Content-hash-keyed transcript cache. **Verify word-timestamp accuracy against the video by hand before proceeding.** **Deliverable:** transcript JSON with word timings, second run returns instantly from cache.

### Phase 4 — End-to-end vertical slice ⭐
Deliberately naive: pick 3 arbitrary 45-second windows snapped to sentence boundaries, centre-crop to 1080×1920, burn plain ASS captions, render H.264/AAC. No AI, no smart crop, no animation. **Deliverable: three real, playable, captioned vertical MP4s.** This is the de-risking phase — the entire render path is proven end to end while it is still trivial to debug.

### Phase 5 — Clip boundaries, properly
Sentence and word boundary snapping. Configurable padding. Reject clips starting on "so"/"and"/"but" unless the preceding gap exceeds a threshold. Filler-word ratio. Duration constraint solver. **Deliverable:** clips start and end on natural speech, verified against a real podcast.

### Phase 6 — LLM provider abstraction and candidate discovery
`LLMProvider` protocol; Ollama, Anthropic, OpenAI implementations. Transcript chunking with overlap. Structured candidate extraction with schema validation and retry on malformed JSON. **Deliverable:** candidate list with hooks, topics, categories, reasons.

### Phase 7 — Scoring, dedup, ranking
Configurable weights, chunk normalization, IoU + embedding dedup, listwise rerank. "Not enough good clips" path — tell the user, do not pad the output with weak clips. **Deliverable:** ranked, deduplicated, top-N selection.

### Phase 8 — Smart 9:16 crop
YuNet detection at reduced sample rate. Face track association across frames. Lip-motion heuristic. Crop-path smoothing (moving average + hysteresis). Two-person fallback layout. Source video never modified. **Deliverable:** crop follows the speaker without visible jitter.

### Phase 9 — Animated captions
Four styles as JSON definitions. Word highlighting, active-word emphasis, scale animation via `\t`. Line wrapping, max chars per line, safe margins. **Deliverable:** captions that look like current short-form content.

### Phase 10 — Silence removal and zooms
`silencedetect` → segment list → conservative removal with a configurable minimum silence duration; preserve natural pauses. Zoom keyframes triggered by emphasis, capped in frequency and intensity. **Deliverable:** tighter clips that still sound human.

### Phase 11 — Results and review interface
Clip cards, preview, score, topic, duration, hook. Sort by score/duration/topic. Adjust start/end, change caption style, regenerate captions, regenerate video, export. Lightweight review only — explicitly not an editor.

### Phase 12 — Stage-level caching
Content-addressed cache keyed on `(source_hash, stage, params_hash)`. Dependency graph so changing caption style invalidates only the render, changing crop invalidates crop and render, changing selection invalidates nothing upstream of clip cutting. **Deliverable:** caption style change re-renders in seconds, not minutes.

### Phase 13 — Quality check, packaging, polish
Automated QC per rendered clip: duration within range, non-black first frame, audio stream present and non-silent, captions inside safe area, file demuxes cleanly. Then `start.sh`, first-run setup, model download UX, README.

---

## 5. Data model sketch

```
Project(id, name, source_path, source_hash, created_at, settings_json)
  └── Job(id, project_id, type, status, progress, current_stage, error, timestamps)
  └── Transcript(id, project_id, backend, model, language, cached_path)
  └── Candidate(id, project_id, start, end, hook, topic, category,
                raw_score, normalized_score, final_score, reason,
                confidence, dimension_scores_json)
  └── Clip(id, candidate_id, start, end, hook_selected, caption_style,
           crop_strategy, render_path, qc_status)
```

Media on disk under `data/`, paths only in SQLite.

---

## 6. Testing

Unit, no video required: transcript parsing, timestamp arithmetic and rounding, boundary snapping, duration constraints, scoring weights, chunk normalization, IoU dedup, embedding dedup threshold, caption line wrapping, ASS timing generation, job state transitions, cache key derivation.

Integration, synthetic fixtures: generate a 30-second test video with FFmpeg `testsrc` plus a TTS or tone audio track, committed as a script rather than a binary. Full pipeline smoke test with a stub LLM provider returning fixed candidates — this keeps CI fast and deterministic.

Manual, per phase: run on a real podcast, watch the output.

---

## 7. Open questions before Phase 1

1. Environment inspection output.
2. Typical source podcast: resolution, single camera or multicam cuts, one or two people on screen?
3. Willing to install Ollama, or should clip discovery default to a cloud provider with local as the fallback?
4. Preferred default clip duration band — the 30–60 s range in the spec, or shorter?
