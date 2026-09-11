"""Clip candidate discovery.

Replaces evenly spaced anchors with moments a model actually finds interesting.

Two things here are load-bearing and easy to get wrong:

**Timestamps.** The model must return times that exist in the podcast. Models
hallucinate numbers freely, so every returned timestamp is validated against the
transcript and snapped to real word boundaries. A candidate whose times cannot
be reconciled is discarded, not guessed at.

**Score comparability.** Chunks are scored in independent calls, so an 87 from
chunk 3 and an 87 from chunk 11 mean different things. Raw scores are normalised
within their chunk before anything compares them across the podcast.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.ai.json_recovery import JSONRecoveryError, extract_json_list
from app.services.transcription.base import Segment, Transcript

# Below this a candidate carries no usable information about *where* the
# interesting moment is, so it is noise rather than a pointer.
MINIMUM_ANCHOR_SECONDS = 2.0

CATEGORIES = {
    "surprising", "controversial", "emotional", "funny", "advice", "insight",
    "story", "revelation", "debate", "educational", "curiosity", "quote",
}

SYSTEM_PROMPT = (
    "You find the most compelling short moments in podcast transcripts for "
    "social video. You reply with JSON only — no prose, no markdown fences."
)

PROMPT_TEMPLATE = """Below is part of a podcast transcript. Each line is prefixed with its start time in seconds.

Find the {want} most compelling moments that would work as standalone short-form videos.

A good moment:
- makes sense without any surrounding context
- opens with something that makes a viewer stop scrolling
- reaches a clear point or conclusion
- lasts between {min_duration} and {max_duration} seconds

A bad moment: setup with no payoff, inside references, pure logistics, or someone reading an ad.

Return a JSON array. Every object must have exactly these keys:
  "start": number - start time in seconds, copied from a line prefix
  "end": number - end time in seconds
  "hook": string - under 12 words, what makes someone watch
  "topic": string - under 6 words
  "category": string - one of: {categories}
  "score": number - 0 to 100, how strong this is
  "reason": string - one sentence on why it works

Use only timestamps that appear in the transcript below. Do not invent times.

Getting the start right matters most. The exact end will be adjusted to land on
a sentence boundary, so approximate it rather than truncating a thought to hit
the target length.

If there are fewer than {want} genuinely good moments, return fewer. Returning
weak moments is worse than returning none.

TRANSCRIPT:
{transcript}

JSON array:"""


@dataclass
class Candidate:
    start: float
    end: float
    hook: str
    topic: str
    category: str
    raw_score: float
    reason: str
    chunk_index: int = 0
    normalized_score: float = 0.0
    confidence: float = 1.0
    issues: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "hook": self.hook,
            "topic": self.topic,
            "category": self.category,
            "raw_score": round(self.raw_score, 2),
            "normalized_score": round(self.normalized_score, 4),
            "reason": self.reason,
            "chunk_index": self.chunk_index,
            "confidence": round(self.confidence, 3),
            "issues": self.issues,
        }


@dataclass
class TranscriptWindow:
    index: int
    start: float
    end: float
    segments: list[Segment]

    def render(self) -> str:
        return "\n".join(
            f"[{s.start:.1f}] {s.text.strip()}" for s in self.segments if s.text.strip()
        )


def estimate_tokens(text: str) -> int:
    """Rough but stable: ~4 characters per token for English prose.

    Deliberately an estimate. A real tokenizer would tie this module to one
    model family, and the number only needs to be good enough to keep windows
    inside a context budget.
    """
    return max(1, len(text) // 4)


def window_transcript(
    transcript: Transcript,
    target_tokens: int = 1800,
    overlap_segments: int = 2,
) -> list[TranscriptWindow]:
    """Split a transcript into windows a model can hold at once.

    Overlap is measured in whole segments rather than seconds so a window never
    starts mid-sentence, and so a moment straddling a boundary is visible to
    both windows rather than invisible to each.
    """
    segments = [s for s in transcript.segments if s.text.strip()]
    if not segments:
        return []

    windows: list[TranscriptWindow] = []
    index = 0
    position = 0

    while position < len(segments):
        current: list[Segment] = []
        tokens = 0
        cursor = position

        while cursor < len(segments):
            cost = estimate_tokens(segments[cursor].text) + 8   # prefix overhead
            if current and tokens + cost > target_tokens:
                break
            current.append(segments[cursor])
            tokens += cost
            cursor += 1

        if not current:
            current = [segments[position]]
            cursor = position + 1

        windows.append(
            TranscriptWindow(index, current[0].start, current[-1].end, current)
        )
        index += 1

        if cursor >= len(segments):
            break
        position = max(position + 1, cursor - overlap_segments)

    return windows


def check_context_budget(
    prompt: str, max_output: int, num_ctx: int
) -> str | None:
    """Return a warning if the request cannot fit the model's context.

    Overflow does not raise in Ollama — it silently drops the oldest tokens,
    which is the transcript. The model then produces confident JSON about text
    it never received, and the timestamp validator rejects nearly all of it.
    That reads as "this model is useless" when the real fault is configuration,
    so it is worth detecting explicitly.
    """
    needed = estimate_tokens(prompt) + max_output
    if needed <= num_ctx * 0.9:
        return None
    return (
        f"Prompt (~{estimate_tokens(prompt)} tokens) plus output ({max_output}) "
        f"needs ~{needed} tokens but the context window is {num_ctx}. "
        f"Lower LLM_WINDOW_TOKENS or raise LLM_NUM_CTX."
    )


def build_prompt(
    window: TranscriptWindow,
    want: int,
    min_duration: float,
    max_duration: float,
) -> str:
    return PROMPT_TEMPLATE.format(
        want=want,
        min_duration=int(min_duration),
        max_duration=int(max_duration),
        categories=", ".join(sorted(CATEGORIES)),
        transcript=window.render(),
    )


def _coerce_float(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group())
    return None


def snap_to_transcript(
    transcript: Transcript, start: float, end: float, tolerance: float = 6.0
) -> tuple[float, float] | None:
    """Move model-supplied times onto real word boundaries.

    Models round, drift and occasionally invent. A time near a real boundary is
    snapped; one beyond `tolerance` from anything is treated as hallucinated and
    the candidate is dropped.
    """
    words = transcript.words
    if not words:
        return None

    nearest_start = min(words, key=lambda w: abs(w.start - start))
    nearest_end = min(words, key=lambda w: abs(w.end - end))

    if abs(nearest_start.start - start) > tolerance:
        return None
    if abs(nearest_end.end - end) > tolerance:
        return None
    if nearest_end.end <= nearest_start.start:
        return None

    return nearest_start.start, nearest_end.end


def parse_candidates(
    raw: str,
    transcript: Transcript,
    window: TranscriptWindow,
    min_duration: float,
    max_duration: float,
) -> tuple[list[Candidate], list[str]]:
    """Turn model output into validated candidates.

    Returns (candidates, rejection reasons). Rejections are kept rather than
    swallowed — if a model produces nothing usable, the reasons are the only way
    to tell a bad prompt from a bad model.
    """
    rejections: list[str] = []

    try:
        items = extract_json_list(raw)
    except JSONRecoveryError as exc:
        return [], [str(exc)]

    candidates: list[Candidate] = []
    for item in items:
        if not isinstance(item, dict):
            rejections.append(f"not an object: {str(item)[:60]}")
            continue

        start = _coerce_float(item.get("start", item.get("start_time")))
        end = _coerce_float(item.get("end", item.get("end_time")))
        if start is None or end is None:
            rejections.append(f"missing start/end: {str(item)[:80]}")
            continue

        if end < start:
            start, end = end, start

        snapped = snap_to_transcript(transcript, start, end)
        if snapped is None:
            rejections.append(
                f"timestamps {start:.1f}-{end:.1f} do not match the transcript"
            )
            continue
        start, end = snapped

        duration = end - start
        issues: list[str] = []

        # A candidate is an *anchor*, not a final cut. Phase 5's solver decides
        # the real boundaries and will grow or trim this to a legal duration.
        # So duration violations are recorded, not fatal - discarding them
        # throws away the model's judgement about where something interesting
        # happens over a number the model was never going to respect. Small
        # models routinely propose 10s moments when asked for 25-60s.
        #
        # The one exception is a span so short it carries no information about
        # location either; that is noise rather than a pointer.
        if duration < MINIMUM_ANCHOR_SECONDS:
            rejections.append(f"degenerate span ({duration:.1f}s) at {start:.1f}")
            continue
        if duration < min_duration:
            issues.append(
                f"model proposed {duration:.1f}s, below the minimum - "
                "the solver will expand it"
            )
        if duration > max_duration:
            issues.append(
                f"model proposed {duration:.1f}s, above the maximum - "
                "the solver will trim it"
            )

        category = str(item.get("category", "")).strip().lower()
        if category not in CATEGORIES:
            issues.append(f"unknown category '{category}'")
            category = "insight"

        score = _coerce_float(item.get("score"))
        if score is None:
            score = 50.0
            issues.append("no score returned")
        score = max(0.0, min(100.0, score))

        hook = str(item.get("hook", "")).strip()[:200]
        if not hook:
            issues.append("no hook returned")

        candidates.append(Candidate(
            start=start,
            end=end,
            hook=hook,
            topic=str(item.get("topic", "")).strip()[:120],
            category=category,
            raw_score=score,
            reason=str(item.get("reason", "")).strip()[:400],
            chunk_index=window.index,
            issues=issues,
        ))

    return candidates, rejections


def normalize_scores(candidates: list[Candidate], min_group: int = 3) -> list[Candidate]:
    """Make scores comparable across independent model calls.

    Each window is scored in its own call with no knowledge of the others, so
    raw scores are only meaningful within a window. Where a window produced
    enough candidates to have a distribution, z-scoring removes that call's
    calibration drift.

    Where it did not — and a model asked for three moments routinely returns
    one — there is nothing to z-score against. Assigning a neutral 0.5 in that
    case leaves every candidate tied and throws away real information: a model
    that said 93 for one moment and 55 for another meant something by it. So
    small groups fall back to a global min-max over raw scores, carrying lower
    confidence to record that the comparison is across calls rather than within
    one.
    """
    by_chunk: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        by_chunk.setdefault(candidate.chunk_index, []).append(candidate)

    small_groups: list[Candidate] = []

    for group in by_chunk.values():
        if len(group) < min_group:
            small_groups.extend(group)
            continue

        scores = [c.raw_score for c in group]
        mean = sum(scores) / len(scores)
        stdev = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5

        if stdev < 1e-6:
            # The model scored everything identically, which carries no
            # ranking information at all.
            for candidate in group:
                candidate.normalized_score = 0.5
                candidate.confidence = 0.3
            continue

        for candidate in group:
            z = (candidate.raw_score - mean) / stdev
            candidate.normalized_score = max(0.0, min(1.0, 0.5 + z / 4))
            candidate.confidence = min(1.0, 0.6 + len(group) / 20)

    if small_groups:
        raw = [c.raw_score for c in small_groups]
        low, high = min(raw), max(raw)
        spread = high - low
        for candidate in small_groups:
            if spread < 1e-6:
                candidate.normalized_score = 0.5
            else:
                # Mapped into 0.15-0.85 rather than the full range: these
                # scores are less trustworthy than within-window ones and
                # should not outrank them at the extremes.
                candidate.normalized_score = 0.15 + 0.7 * (candidate.raw_score - low) / spread
            candidate.confidence = 0.4

    return candidates


def deduplicate(candidates: list[Candidate], iou_threshold: float = 0.5) -> list[Candidate]:
    """Drop candidates covering the same moment.

    Window overlap means the same moment is legitimately offered twice. Time
    overlap is the reliable signal at this stage; semantic near-duplicates on
    different moments are Phase 7's problem, where embeddings arrive.
    """
    ordered = sorted(candidates, key=lambda c: c.normalized_score, reverse=True)
    kept: list[Candidate] = []

    for candidate in ordered:
        duplicate = False
        for existing in kept:
            overlap = min(candidate.end, existing.end) - max(candidate.start, existing.start)
            if overlap <= 0:
                continue
            union = max(candidate.end, existing.end) - min(candidate.start, existing.start)
            if union > 0 and overlap / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    kept.sort(key=lambda c: c.start)
    return kept
