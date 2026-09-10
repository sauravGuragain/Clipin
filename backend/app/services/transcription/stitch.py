"""Chunked transcription: planning and stitching.

Whisper offers no progress callback. Transcribing a 90-minute podcast as one
opaque call means a progress bar frozen at 0% for five or more minutes, which
violates the "no fake implementations" rule in the spec just as much as a
simulated bar would.

So the audio is transcribed in chunks and progress is reported per completed
chunk. The cost is that a sentence can straddle a boundary; the overlap and the
commit-window rule below handle that, and Phase 5 re-snaps clip boundaries to
sentences anyway.

Set chunk_seconds=0 to disable chunking entirely — one pass, best possible
context, no progress reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.transcription.base import Segment, Transcript, Word

EPSILON = 1e-6


@dataclass
class Chunk:
    """A unit of work.

    `read_start`/`read_end` is the audio actually fed to the model, including
    overlap for context. `commit_start`/`commit_end` is the window whose words
    are kept. The difference is what prevents duplicated words at seams.
    """

    index: int
    read_start: float
    read_end: float
    commit_start: float
    commit_end: float

    @property
    def read_duration(self) -> float:
        return self.read_end - self.read_start


def plan_chunks(duration: float, chunk_seconds: float, overlap: float = 2.0) -> list[Chunk]:
    if duration <= 0:
        return []

    # A single chunk: no seams, no progress granularity.
    if chunk_seconds <= 0 or duration <= chunk_seconds:
        return [Chunk(0, 0.0, duration, 0.0, duration)]

    overlap = max(0.0, min(overlap, chunk_seconds / 2))

    chunks: list[Chunk] = []
    index = 0
    commit_start = 0.0
    while commit_start < duration - EPSILON:
        commit_end = min(commit_start + chunk_seconds, duration)
        read_start = max(0.0, commit_start - overlap)
        read_end = min(duration, commit_end + overlap)
        chunks.append(Chunk(index, read_start, read_end, commit_start, commit_end))
        commit_start = commit_end
        index += 1

    return chunks


def _word_in_window(word: Word, start: float, end: float, is_last: bool) -> bool:
    """A word belongs to the chunk whose commit window contains its midpoint.

    Midpoint rather than start: a word beginning a hair before the boundary but
    mostly after it reads better attached to the later chunk, and using a single
    consistent rule guarantees no word is kept twice or dropped entirely.
    """
    mid = (word.start + word.end) / 2
    if is_last:
        return mid >= start - EPSILON
    return start - EPSILON <= mid < end - EPSILON


def merge_chunks(
    results: list[tuple[Chunk, Transcript]],
    language: str | None = None,
    backend: str = "unknown",
    model: str = "unknown",
    duration: float = 0.0,
) -> Transcript:
    """Stitch per-chunk transcripts into one, dropping overlap duplicates.

    Chunk transcripts must already carry absolute timestamps.
    """
    if not results:
        return Transcript([], language or "unknown", backend, model, duration)

    ordered = sorted(results, key=lambda r: r[0].index)
    last_index = ordered[-1][0].index

    merged: list[Segment] = []
    for chunk, transcript in ordered:
        is_last = chunk.index == last_index
        for seg in transcript.segments:
            if seg.words:
                kept = [
                    w for w in seg.words
                    if _word_in_window(w, chunk.commit_start, chunk.commit_end, is_last)
                ]
                if not kept:
                    continue
                merged.append(
                    Segment(
                        start=kept[0].start,
                        end=kept[-1].end,
                        text=" ".join(w.text.strip() for w in kept).strip(),
                        words=kept,
                    )
                )
            else:
                # A backend without word timestamps: fall back to segment
                # midpoints so stitching still works, though downstream
                # caption stages will reject this transcript.
                mid = (seg.start + seg.end) / 2
                inside = (
                    mid >= chunk.commit_start - EPSILON
                    if is_last
                    else chunk.commit_start - EPSILON <= mid < chunk.commit_end - EPSILON
                )
                if inside:
                    merged.append(seg)

    merged.sort(key=lambda s: s.start)

    resolved_language = language
    if resolved_language is None:
        # Trust the longest chunk's detection over an arbitrary first one.
        best = max(ordered, key=lambda r: r[0].read_duration)
        resolved_language = best[1].language

    return Transcript(
        segments=merged,
        language=resolved_language or "unknown",
        backend=backend,
        model=model,
        duration=duration,
    )
