"""Clip boundary quality.

Phase 4 grew a clip greedily: start at the first segment past the anchor, add
segments until long enough, stop. That produces clips that are *legal* — right
duration, whole words — but often badly cut: opening mid-thought, ending on a
dangling conjunction, padded with filler.

This module scores candidate boundaries and searches for the best pair rather
than accepting the first that fits. Weights are configurable, per spec 28.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.transcription.base import Segment, Transcript, Word

SENTENCE_END = (".", "!", "?", "…")

# Two kinds of weak opener, and the difference matters.
#
# A discourse marker ("so", "okay", "well") is weak mid-flow but perfectly fine
# after a pause — that is how people start a new thought out loud.
DISCOURSE_MARKERS = {
    "so", "okay", "well", "right", "yeah", "um", "uh", "anyway", "now",
    "look", "listen", "actually", "basically", "also",
}

# A subordinator ("which", "because", "and") refers back to something the
# viewer never saw. A pause does not repair that — the clause is still
# grammatically dependent on missing context, so it is never forgiven.
SUBORDINATORS = {
    "and", "but", "or", "because", "which", "that", "then", "though",
    "however", "since", "while", "whereas", "unless", "therefore", "thus",
}

WEAK_OPENERS = DISCOURSE_MARKERS | SUBORDINATORS

# Ending on these leaves the thought visibly unfinished.
DANGLING_ENDINGS = {
    "and", "but", "or", "so", "because", "which", "that", "the", "a", "an",
    "to", "of", "in", "for", "with", "if", "when", "as", "at", "on", "is",
    "was", "are", "were", "my", "your", "his", "her", "their", "our",
}

FILLER_WORDS = {
    "um", "uh", "erm", "ah", "eh", "hmm", "mhm", "like", "basically",
    "literally", "actually", "kinda", "y'know", "right", "okay", "yeah",
}

# Multi-word fillers need their own set: matching them against single cleaned
# words silently never fires, which is exactly the bug this replaced.
FILLER_PHRASES = {
    ("you", "know"), ("i", "mean"), ("sort", "of"), ("kind", "of"),
    ("you", "see"), ("or", "whatever"), ("and", "stuff"), ("i", "guess"),
}

# Phrases that make a poor opening even though the first word alone looks fine.
WEAK_OPENING_PHRASES = {
    ("you", "know"), ("i", "mean"), ("it", "was"), ("that", "is"),
    ("which", "is"), ("and", "then"), ("so", "yeah"), ("i", "guess"),
    ("kind", "of"), ("sort", "of"), ("the", "thing", "is"),
}


def clean(text: str) -> str:
    return text.strip().strip("\"'“”‘’").rstrip(",.!?;:").lower()


def ends_sentence(text: str) -> bool:
    return text.strip().rstrip("\"'”’").endswith(SENTENCE_END)


def phrase_at(words: list[Word], index: int, length: int) -> tuple[str, ...]:
    return tuple(clean(w.text) for w in words[index : index + length])


def filler_ratio(words: list[Word]) -> float:
    """Fraction of words that are filler, counting multi-word phrases whole."""
    if not words:
        return 1.0

    hits = 0
    i = 0
    while i < len(words):
        if phrase_at(words, i, 2) in FILLER_PHRASES:
            hits += 2
            i += 2
            continue
        if clean(words[i].text) in FILLER_WORDS:
            hits += 1
        i += 1
    return min(1.0, hits / len(words))


def opener_weakness(words: list[Word]) -> tuple[str, bool] | None:
    """Return (offending opener, forgivable_by_pause), or None if strong.

    Checks phrases before single words: "you know" is weak even though "you"
    on its own is not.
    """
    for length in (3, 2):
        phrase = phrase_at(words, 0, length)
        if len(phrase) == length and phrase in WEAK_OPENING_PHRASES:
            # A phrase opener is forgivable only if its head word is a
            # discourse marker rather than a subordinator.
            return " ".join(phrase), phrase[0] in DISCOURSE_MARKERS
    first = clean(words[0].text) if words else ""
    if first in SUBORDINATORS:
        return first, False
    if first in DISCOURSE_MARKERS:
        return first, True
    return None


def gap_before(segments: list[Segment], index: int) -> float:
    """Silence between the previous segment and this one."""
    if index <= 0:
        return 999.0        # start of media counts as a clean break
    previous = segments[index - 1]
    current = segments[index]
    return max(0.0, current.start - previous.end)


def gap_after(segments: list[Segment], index: int) -> float:
    if index >= len(segments) - 1:
        return 999.0
    return max(0.0, segments[index + 1].start - segments[index].end)


@dataclass
class BoundaryWeights:
    """Configurable, per spec 28. Defaults chosen so that starting cleanly and
    finishing the thought dominate — those are what a viewer notices."""

    opener: float = 0.28
    starts_sentence: float = 0.18
    ends_sentence: float = 0.24
    not_dangling: float = 0.12
    low_filler: float = 0.10
    duration_fit: float = 0.08

    def total(self) -> float:
        return (
            self.opener + self.starts_sentence + self.ends_sentence
            + self.not_dangling + self.low_filler + self.duration_fit
        )


@dataclass
class BoundaryScore:
    total: float
    components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def score_boundaries(
    segments: list[Segment],
    start_index: int,
    end_index: int,
    min_duration: float,
    max_duration: float,
    weights: BoundaryWeights | None = None,
    pause_forgives_opener: float = 0.6,
) -> BoundaryScore:
    """Score a candidate clip spanning segments[start_index:end_index + 1]."""
    weights = weights or BoundaryWeights()
    chosen = segments[start_index : end_index + 1]
    words = [w for s in chosen for w in s.words]

    if not words:
        return BoundaryScore(0.0, {}, ["no words"])

    components: dict[str, float] = {}
    notes: list[str] = []

    # --- opening ---------------------------------------------------------
    weakness = opener_weakness(words)
    if weakness is None:
        components["opener"] = 1.0
    else:
        opener, forgivable = weakness
        pause = gap_before(segments, start_index)
        if forgivable and pause >= pause_forgives_opener:
            # "So, here is the thing" after silence is how people begin a new
            # thought aloud. Legitimate.
            components["opener"] = 0.75
            notes.append(f"opens on '{opener}', forgiven by {pause:.1f}s pause")
        elif forgivable:
            components["opener"] = 0.15
            notes.append(f"opens on '{opener}' with only {pause:.1f}s before it")
        else:
            # Subordinate clause: depends on context the viewer cannot see.
            components["opener"] = 0.0
            notes.append(f"opens on subordinate '{opener}' — refers to unseen context")

    # --- starts at a sentence boundary ----------------------------------
    if start_index == 0 or ends_sentence(segments[start_index - 1].text):
        components["starts_sentence"] = 1.0
    else:
        components["starts_sentence"] = 0.0
        notes.append("starts mid-sentence")

    # --- ends at a sentence boundary ------------------------------------
    if ends_sentence(chosen[-1].text):
        components["ends_sentence"] = 1.0
    else:
        components["ends_sentence"] = 0.0
        notes.append("does not end on a sentence")

    # --- dangling final word --------------------------------------------
    last = clean(words[-1].text)
    if last in DANGLING_ENDINGS:
        components["not_dangling"] = 0.0
        notes.append(f"ends on '{last}'")
    else:
        components["not_dangling"] = 1.0

    # --- filler density --------------------------------------------------
    ratio = filler_ratio(words)
    components["low_filler"] = max(0.0, 1.0 - ratio * 4)
    if ratio > 0.12:
        notes.append(f"{ratio:.0%} filler words")

    # --- duration fit ----------------------------------------------------
    duration = words[-1].end - words[0].start
    if duration < min_duration or duration > max_duration:
        components["duration_fit"] = 0.0
    else:
        # Peak in the middle of the allowed range; the extremes are legal but
        # less desirable.
        midpoint = (min_duration + max_duration) / 2
        half_span = max(1e-6, (max_duration - min_duration) / 2)
        components["duration_fit"] = max(0.0, 1.0 - abs(duration - midpoint) / half_span)

    total = sum(
        components[name] * getattr(weights, name) for name in components
    ) / max(1e-9, weights.total())

    return BoundaryScore(round(total, 4), components, notes)


@dataclass
class BoundaryCandidate:
    start_index: int
    end_index: int
    start: float
    end: float
    score: BoundaryScore

    @property
    def duration(self) -> float:
        return self.end - self.start


def solve_boundaries(
    transcript: Transcript,
    anchor: float,
    min_duration: float,
    max_duration: float,
    weights: BoundaryWeights | None = None,
    search_back: float = 15.0,
    search_forward: float = 25.0,
    max_candidates: int = 400,
) -> BoundaryCandidate | None:
    """Search start/end segment pairs near an anchor, return the best-scoring.

    Bounded deliberately: `search_back`/`search_forward` keep the result near
    the anchor, so when Phase 6 supplies anchors from real clip discovery the
    solver refines that choice rather than wandering off to a different moment.
    """
    segments = [s for s in transcript.segments if s.words]
    if not segments:
        return None

    starts = [
        i for i, s in enumerate(segments)
        if anchor - search_back <= s.start <= anchor + search_forward
    ]
    if not starts:
        # Anchor past the end, or a very sparse transcript: fall back to the
        # nearest segment rather than giving up.
        nearest = min(range(len(segments)), key=lambda i: abs(segments[i].start - anchor))
        starts = [nearest]

    best: BoundaryCandidate | None = None
    evaluated = 0

    for start_index in starts:
        origin = segments[start_index].start
        for end_index in range(start_index, len(segments)):
            duration = segments[end_index].end - origin
            if duration < min_duration:
                continue
            if duration > max_duration:
                break
            evaluated += 1
            if evaluated > max_candidates:
                break

            score = score_boundaries(
                segments, start_index, end_index,
                min_duration, max_duration, weights,
            )
            if best is None or score.total > best.score.total:
                words = [w for s in segments[start_index : end_index + 1] for w in s.words]
                best = BoundaryCandidate(
                    start_index=start_index,
                    end_index=end_index,
                    start=words[0].start,
                    end=words[-1].end,
                    score=score,
                )
        if evaluated > max_candidates:
            break

    return best


def pad_into_silence(
    transcript: Transcript,
    start: float,
    end: float,
    max_padding: float,
    media_duration: float,
    min_leading: float = 0.08,
) -> tuple[float, float]:
    """Extend into surrounding silence instead of by a fixed amount.

    A fixed pad can bite into the neighbouring word; taking half of whatever
    silence is actually there cannot. `min_leading` guarantees a sliver of head
    room so the first consonant is never clipped, which is audible even when the
    timestamp is technically correct.
    """
    words = transcript.words
    if not words:
        return start, end

    before = [w for w in words if w.end <= start + 1e-6]
    after = [w for w in words if w.start >= end - 1e-6]

    lead_gap = start - before[-1].end if before else start
    lead = min(max_padding, max(min_leading, lead_gap / 2)) if lead_gap > 0 else min_leading
    new_start = max(0.0, start - lead)

    trail_gap = after[0].start - end if after else (media_duration - end if media_duration else max_padding)
    trail = min(max_padding, max(min_leading, trail_gap / 2)) if trail_gap > 0 else min_leading
    new_end = end + trail
    if media_duration:
        new_end = min(media_duration, new_end)

    return new_start, new_end
