"""Clip selection — deliberately naive for Phase 4.

Anchors are spaced evenly through the podcast. There is no AI here and no
pretence of one: `strategy="even"` says exactly what it does. Phase 6 adds real
candidate discovery and swaps the anchor source out.

What is *not* naive is the boundary snapping. That code is real, it is what
Phase 5 builds on, and it is the difference between a clip that starts
mid-syllable and one that starts on a sentence.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.transcription.base import Segment, Transcript, Word

# A clip beginning on these reads as though it were cut out of something else.
WEAK_OPENERS = {
    "so", "and", "but", "or", "because", "which", "that", "then",
    "um", "uh", "like", "yeah", "okay", "well", "also", "however",
}

SENTENCE_END = (".", "!", "?", "…")


@dataclass
class ClipWindow:
    start: float
    end: float
    words: list[Word]
    text: str
    strategy: str = "even"
    opener_penalty: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


def _clean(text: str) -> str:
    return text.strip().strip("\"'").lower()


def is_weak_opener(word: Word) -> bool:
    return _clean(word.text).rstrip(",") in WEAK_OPENERS


def ends_sentence(segment: Segment) -> bool:
    return segment.text.strip().endswith(SENTENCE_END)


def anchors_for(duration: float, count: int, edge_fraction: float = 0.05) -> list[float]:
    """Evenly spaced start points, skipping the intro and outro.

    The margins exist because podcast openings are sponsor reads and endings are
    sign-offs, and neither makes a clip.
    """
    if count <= 0 or duration <= 0:
        return []

    margin = duration * edge_fraction
    usable_start = margin
    usable_end = max(margin, duration - margin)
    span = usable_end - usable_start
    if span <= 0:
        return [0.0]
    if count == 1:
        return [usable_start + span / 2]

    step = span / count
    return [usable_start + step * i for i in range(count)]


def snap_to_segments(
    transcript: Transcript,
    anchor: float,
    min_duration: float,
    max_duration: float,
) -> ClipWindow | None:
    """Grow a clip outward from an anchor, respecting segment boundaries.

    Starts at the first segment at or after the anchor, then adds whole
    segments until the clip is long enough, preferring to stop on a sentence
    ending. Whole segments only — cutting inside one is what produces clipped
    words.
    """
    segments = [s for s in transcript.segments if s.words]
    if not segments:
        return None

    start_index = next(
        (i for i, s in enumerate(segments) if s.start >= anchor - 0.001),
        None,
    )
    if start_index is None:
        start_index = len(segments) - 1

    # Prefer a stronger opening: look ahead a little for a segment that does not
    # begin on a filler word, but do not wander far from the anchor.
    for offset in range(0, min(3, len(segments) - start_index)):
        candidate = segments[start_index + offset]
        if candidate.words and not is_weak_opener(candidate.words[0]):
            start_index += offset
            break

    chosen: list[Segment] = []
    for segment in segments[start_index:]:
        prospective = segment.end - segments[start_index].start
        if chosen and prospective > max_duration:
            break
        chosen.append(segment)
        current = chosen[-1].end - chosen[0].start
        if current >= min_duration and ends_sentence(chosen[-1]):
            break
        if current >= max_duration:
            break

    if not chosen:
        return None

    words = [w for s in chosen for w in s.words]
    if not words:
        return None

    start = words[0].start
    end = words[-1].end
    if end - start < 1.0:
        return None

    return ClipWindow(
        start=start,
        end=end,
        words=words,
        text=" ".join(s.text.strip() for s in chosen).strip(),
        strategy="even",
        opener_penalty=is_weak_opener(words[0]),
    )


def _overlaps(a: ClipWindow, b: ClipWindow) -> bool:
    return a.start < b.end and b.start < a.end


def select_clips(
    transcript: Transcript,
    count: int,
    min_duration: float = 25.0,
    max_duration: float = 60.0,
    padding: float = 0.25,
) -> list[ClipWindow]:
    """Pick `count` non-overlapping windows.

    Over-samples anchors then drops overlaps, because snapping moves a window
    away from its anchor and two neighbouring anchors can land on the same
    segment run.
    """
    if not transcript.segments:
        return []

    duration = transcript.duration or transcript.segments[-1].end
    windows: list[ClipWindow] = []

    for anchor in anchors_for(duration, count * 2):
        window = snap_to_segments(transcript, anchor, min_duration, max_duration)
        if window is None:
            continue
        if any(_overlaps(window, existing) for existing in windows):
            continue
        windows.append(window)
        if len(windows) >= count:
            break

    windows.sort(key=lambda w: w.start)

    # Padding is applied last, but must not push a clip past max_duration —
    # the limit is a contract with the caller, not a suggestion.
    if padding:
        for window in windows:
            headroom = max_duration - window.duration
            if headroom <= 0:
                continue
            each = min(padding, headroom / 2)
            window.start = max(0.0, window.start - each)
            window.end = min(duration, window.end + each)

    return windows
