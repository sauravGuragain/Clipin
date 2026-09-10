"""Transcript data model.

Word-level timestamps are the load-bearing part. Everything in Phase 9
(animated captions) and Phase 5 (clip boundaries) reads from these, so the
model carries validation that surfaces timing defects loudly rather than
letting them propagate into a render that looks subtly wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def shifted(self, offset: float) -> Word:
        return Word(self.text, self.start + offset, self.end + offset, self.probability)

    def to_dict(self) -> dict:
        d = {"text": self.text, "start": round(self.start, 3), "end": round(self.end, 3)}
        if self.probability is not None:
            d["probability"] = round(self.probability, 4)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Word:
        return cls(
            text=d["text"],
            start=float(d["start"]),
            end=float(d["end"]),
            probability=d.get("probability"),
        )


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def shifted(self, offset: float) -> Segment:
        return Segment(
            start=self.start + offset,
            end=self.end + offset,
            text=self.text,
            words=[w.shifted(offset) for w in self.words],
        )

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        return cls(
            start=float(d["start"]),
            end=float(d["end"]),
            text=d["text"],
            words=[Word.from_dict(w) for w in d.get("words", [])],
        )


@dataclass
class TranscriptIssue:
    kind: str
    detail: str
    at: float


@dataclass
class Transcript:
    segments: list[Segment]
    language: str
    backend: str
    model: str
    duration: float = 0.0
    cache_key: str | None = None

    # --- access ---------------------------------------------------------

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    @property
    def word_count(self) -> int:
        return sum(len(s.words) for s in self.segments)

    @property
    def has_word_timestamps(self) -> bool:
        return self.word_count > 0

    def __iter__(self) -> Iterator[Segment]:
        return iter(self.segments)

    def words_between(self, start: float, end: float) -> list[Word]:
        """Words whose midpoint falls inside the window.

        Midpoint rather than full containment: a word straddling the boundary
        belongs to whichever side holds most of it, which is what produces
        natural-looking caption cuts.
        """
        out = []
        for w in self.words:
            mid = (w.start + w.end) / 2
            if start <= mid < end:
                out.append(w)
        return out

    # --- validation -----------------------------------------------------

    def validate(self) -> list[TranscriptIssue]:
        """Report timing defects. Empty list means the transcript is sane.

        This is not paranoia. Whisper backends differ in word-timestamp
        quality, and a backend that returns plausible-looking but misaligned
        timings will silently poison every downstream stage.
        """
        issues: list[TranscriptIssue] = []

        previous_end = -1.0
        for i, seg in enumerate(self.segments):
            if seg.end < seg.start:
                issues.append(TranscriptIssue(
                    "segment_reversed",
                    f"Segment {i} ends before it starts ({seg.start:.2f} > {seg.end:.2f}).",
                    seg.start,
                ))
            if seg.start < previous_end - 0.001:
                issues.append(TranscriptIssue(
                    "segment_overlap",
                    f"Segment {i} starts at {seg.start:.2f}, before the previous ended "
                    f"at {previous_end:.2f}.",
                    seg.start,
                ))
            previous_end = max(previous_end, seg.end)

            last_word_end = -1.0
            for w in seg.words:
                if w.end < w.start:
                    issues.append(TranscriptIssue(
                        "word_reversed",
                        f"Word '{w.text.strip()}' ends before it starts.",
                        w.start,
                    ))
                if w.start < last_word_end - 0.001:
                    issues.append(TranscriptIssue(
                        "word_overlap",
                        f"Word '{w.text.strip()}' at {w.start:.2f} overlaps the previous word.",
                        w.start,
                    ))
                if w.duration > 4.0:
                    issues.append(TranscriptIssue(
                        "word_too_long",
                        f"Word '{w.text.strip()}' spans {w.duration:.1f}s, which usually "
                        "means alignment failed.",
                        w.start,
                    ))
                last_word_end = max(last_word_end, w.end)

            if seg.words:
                if seg.words[0].start < seg.start - 0.05:
                    issues.append(TranscriptIssue(
                        "word_before_segment",
                        f"Segment {i} contains a word starting before the segment does.",
                        seg.start,
                    ))
                if seg.words[-1].end > seg.end + 0.05:
                    issues.append(TranscriptIssue(
                        "word_after_segment",
                        f"Segment {i} contains a word ending after the segment does.",
                        seg.end,
                    ))

        if self.duration:
            for w in self.words:
                if w.end > self.duration + 0.5:
                    issues.append(TranscriptIssue(
                        "beyond_media",
                        f"Word '{w.text.strip()}' ends at {w.end:.2f}, past the media "
                        f"duration of {self.duration:.2f}.",
                        w.end,
                    ))
                    break

        return issues

    def stats(self) -> dict[str, Any]:
        words = self.words
        speaking = sum(w.duration for w in words)
        probs = [w.probability for w in words if w.probability is not None]
        return {
            "segments": len(self.segments),
            "words": len(words),
            "duration": round(self.duration, 2),
            "speaking_seconds": round(speaking, 2),
            "words_per_minute": round(len(words) / (self.duration / 60), 1) if self.duration else 0,
            "mean_word_probability": round(sum(probs) / len(probs), 4) if probs else None,
            "has_word_timestamps": self.has_word_timestamps,
        }

    # --- serialisation --------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "backend": self.backend,
            "model": self.model,
            "duration": round(self.duration, 3),
            "cache_key": self.cache_key,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Transcript:
        return cls(
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
            language=d.get("language", "unknown"),
            backend=d.get("backend", "unknown"),
            model=d.get("model", "unknown"),
            duration=float(d.get("duration", 0.0)),
            cache_key=d.get("cache_key"),
        )


@runtime_checkable
class TranscriptionBackend(Protocol):
    """Every backend returns the same Transcript regardless of engine.

    Note for anyone adding one: word timestamps are mandatory, not optional.
    A backend that cannot produce them is not usable for this application.
    """

    name: str

    def is_available(self) -> tuple[bool, str]:
        """(available, human-readable reason if not)."""
        ...

    def transcribe(
        self,
        audio_path: str,
        model: str,
        language: str | None = None,
        offset: float = 0.0,
    ) -> Transcript:
        ...
