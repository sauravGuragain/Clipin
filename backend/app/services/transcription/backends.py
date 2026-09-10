"""Transcription backends.

Selection is by capability, not preference. On this machine MLX wins because
CTranslate2 — which faster-whisper is built on — has no Metal backend and would
run the most expensive stage in the pipeline on CPU only.
"""

from __future__ import annotations

import importlib.util
import platform

from app.services.transcription.base import Segment, Transcript, Word


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class MLXWhisperBackend:
    """Apple Silicon. Runs on Metal through MLX.

    Default backend on this machine. Model weights download from HuggingFace on
    first use — large-v3-turbo is roughly 1.6 GB.
    """

    name = "mlx-whisper"
    default_model = "mlx-community/whisper-large-v3-turbo"

    def is_available(self) -> tuple[bool, str]:
        if not _is_apple_silicon():
            return False, "MLX requires Apple Silicon (arm64 macOS)."
        if not _module_present("mlx_whisper"):
            return False, "mlx-whisper is not installed. Run: pip install mlx-whisper"
        return True, ""

    def transcribe(
        self,
        audio_path: str,
        model: str,
        language: str | None = None,
        offset: float = 0.0,
    ) -> Transcript:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model,
            word_timestamps=True,
            language=language,
            verbose=None,
        )
        return _from_whisper_dict(result, offset, self.name, model)


class FasterWhisperBackend:
    """Portable fallback.

    CPU-only on macOS: CTranslate2 has no Metal backend. Kept for portability
    and as a cross-check when MLX word timings look wrong, not as a default.
    """

    name = "faster-whisper"
    default_model = "large-v3"

    def is_available(self) -> tuple[bool, str]:
        if not _module_present("faster_whisper"):
            return False, "faster-whisper is not installed. Run: pip install faster-whisper"
        return True, ""

    def transcribe(
        self,
        audio_path: str,
        model: str,
        language: str | None = None,
        offset: float = 0.0,
    ) -> Transcript:
        from faster_whisper import WhisperModel

        engine = WhisperModel(model, device="cpu", compute_type="int8")
        segments_iter, info = engine.transcribe(
            audio_path,
            word_timestamps=True,
            language=language,
        )

        segments: list[Segment] = []
        for seg in segments_iter:
            words = [
                Word(
                    text=w.word,
                    start=float(w.start) + offset,
                    end=float(w.end) + offset,
                    probability=getattr(w, "probability", None),
                )
                for w in (seg.words or [])
            ]
            segments.append(
                Segment(
                    start=float(seg.start) + offset,
                    end=float(seg.end) + offset,
                    text=seg.text.strip(),
                    words=words,
                )
            )

        return Transcript(
            segments=segments,
            language=info.language,
            backend=self.name,
            model=model,
            duration=float(getattr(info, "duration", 0.0)),
        )


class StubBackend:
    """Deterministic fake for tests and for exercising the pipeline without a
    model. Never selected automatically — it must be requested explicitly, so
    it can't silently stand in for real transcription."""

    name = "stub"
    default_model = "stub"

    def __init__(self, words_per_second: float = 2.5) -> None:
        self.words_per_second = words_per_second

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def transcribe(
        self,
        audio_path: str,
        model: str,
        language: str | None = None,
        offset: float = 0.0,
    ) -> Transcript:
        import wave

        try:
            with wave.open(audio_path, "rb") as wav:
                duration = wav.getnframes() / float(wav.getframerate())
        except Exception:
            duration = 10.0

        step = 1.0 / self.words_per_second
        segments: list[Segment] = []
        t = 0.0
        index = 0
        while t < duration - EPS:
            seg_end = min(t + 5.0, duration)
            words: list[Word] = []
            wt = t
            while wt < seg_end - EPS:
                w_end = min(wt + step * 0.8, seg_end)
                words.append(
                    Word(f"word{index}", round(wt + offset, 3), round(w_end + offset, 3), 0.9)
                )
                index += 1
                wt += step
            if words:
                segments.append(
                    Segment(
                        start=round(t + offset, 3),
                        end=round(seg_end + offset, 3),
                        text=" ".join(w.text for w in words),
                        words=words,
                    )
                )
            t = seg_end

        return Transcript(segments, language or "en", self.name, model, duration)


EPS = 1e-9


def _from_whisper_dict(result: dict, offset: float, backend: str, model: str) -> Transcript:
    """Parse the openai-whisper result shape, which mlx-whisper mirrors.

    Pure parsing, kept separate so it can be tested against fixture payloads
    without a model present.
    """
    segments: list[Segment] = []
    for seg in result.get("segments", []):
        words = []
        for w in seg.get("words", []) or []:
            start = w.get("start")
            end = w.get("end")
            if start is None or end is None:
                continue
            words.append(
                Word(
                    text=w.get("word", w.get("text", "")),
                    start=float(start) + offset,
                    end=float(end) + offset,
                    probability=w.get("probability"),
                )
            )
        segments.append(
            Segment(
                start=float(seg.get("start", 0.0)) + offset,
                end=float(seg.get("end", 0.0)) + offset,
                text=(seg.get("text") or "").strip(),
                words=words,
            )
        )

    duration = segments[-1].end - offset if segments else 0.0
    return Transcript(
        segments=segments,
        language=result.get("language", "unknown"),
        backend=backend,
        model=model,
        duration=duration,
    )


BACKENDS: dict[str, type] = {
    MLXWhisperBackend.name: MLXWhisperBackend,
    FasterWhisperBackend.name: FasterWhisperBackend,
    StubBackend.name: StubBackend,
}


def get_backend(name: str | None = None):
    """Resolve a backend by name, or auto-select the best available one."""
    if name and name != "auto":
        cls = BACKENDS.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown transcription backend '{name}'. "
                f"Available: {', '.join(BACKENDS)}"
            )
        return cls()

    # Auto: MLX first on Apple Silicon, then faster-whisper. Never the stub.
    for cls in (MLXWhisperBackend, FasterWhisperBackend):
        instance = cls()
        available, _ = instance.is_available()
        if available:
            return instance

    reasons = []
    for cls in (MLXWhisperBackend, FasterWhisperBackend):
        _, reason = cls().is_available()
        reasons.append(f"{cls.name}: {reason}")
    raise RuntimeError(
        "No transcription backend is available.\n  " + "\n  ".join(reasons)
    )
