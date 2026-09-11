"""Listwise reranking and final scoring.

Phase 6 scored each window in its own call. That makes scores incomparable
across windows, and when a model returns one candidate per window — which is
what qwen3:8b actually did — it produces a flat tie: four candidates all at
0.85, which is no ranking at all.

Independent scoring cannot fix this. The only thing that establishes a real
order is showing the model the candidates *together* and asking it to compare
them. That is one extra call for the whole podcast, not one per window.

Final score then blends what the model thinks with what the transcript
structure says, under configurable weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.ai.discovery import Candidate
from app.services.ai.json_recovery import JSONRecoveryError, extract_json_list
from app.services.ai.similarity import SimilarityBackend
from app.services.transcription.base import Transcript

RERANK_SYSTEM = (
    "You compare candidate podcast clips and rank them for short-form video. "
    "You reply with JSON only — no prose, no markdown fences."
)

RERANK_PROMPT = """Below are {count} candidate moments from one podcast. Rank them best to worst for short-form social video.

Judge on: does it grab attention in the first two seconds, does it stand alone without context, does it reach a point, and would someone send it to a friend.

Return a JSON array of objects, best first:
  "id": number - the candidate's id below
  "rank": number - 1 is best
  "verdict": string - under 10 words on why it placed there

Include every id exactly once. Rank all {count}.

CANDIDATES:
{candidates}

JSON array:"""


@dataclass
class ScoringWeights:
    """Configurable, per spec 28 and 9."""

    llm_score: float = 0.30          # the model's own per-window score
    rerank: float = 0.35             # its comparative judgement
    boundary: float = 0.20           # how cleanly the clip can be cut
    duration_fit: float = 0.08
    diversity: float = 0.07          # penalise five clips on one topic

    def total(self) -> float:
        return (
            self.llm_score + self.rerank + self.boundary
            + self.duration_fit + self.diversity
        )


def excerpt_for(transcript: Transcript, start: float, end: float, limit: int = 320) -> str:
    words = [w.text.strip() for w in transcript.words if start <= w.start < end]
    text = " ".join(words)
    return text[:limit] + ("…" if len(text) > limit else "")


def build_rerank_prompt(candidates: list[Candidate], transcript: Transcript) -> str:
    lines = []
    for index, candidate in enumerate(candidates):
        lines.append(
            f"[{index}] hook: {candidate.hook or '(none)'}\n"
            f"     topic: {candidate.topic or '(none)'} | category: {candidate.category}\n"
            f"     says: {excerpt_for(transcript, candidate.start, candidate.end)}"
        )
    return RERANK_PROMPT.format(count=len(candidates), candidates="\n\n".join(lines))


def parse_rankings(raw: str, count: int) -> tuple[dict[int, int], list[str]]:
    """Return {candidate_index: rank} plus problems.

    Models drop entries, repeat them, and rank beyond the list. Every case is
    recoverable — a partial ranking is still better than none — so entries are
    validated individually and the missing ones fall to the back.
    """
    problems: list[str] = []
    try:
        items = extract_json_list(raw, list_keys=("ranking", "ranked", "clips", "results", "items"))
    except JSONRecoveryError as exc:
        return {}, [str(exc)]

    rankings: dict[int, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("id", item.get("index", -1)))
            rank = int(item.get("rank", len(rankings) + 1))
        except (TypeError, ValueError):
            continue
        if not 0 <= index < count:
            problems.append(f"rank refers to unknown candidate {index}")
            continue
        if index in rankings:
            problems.append(f"candidate {index} ranked twice")
            continue
        rankings[index] = rank

    missing = [i for i in range(count) if i not in rankings]
    if missing:
        problems.append(f"{len(missing)} candidate(s) left unranked")
        # Unranked goes to the back rather than being dropped: the model
        # omitting something is not evidence it is bad.
        worst = max(rankings.values(), default=0)
        for offset, index in enumerate(missing, start=1):
            rankings[index] = worst + offset

    return rankings, problems


def apply_rerank(candidates: list[Candidate], rankings: dict[int, int]) -> None:
    """Convert ranks into 0-1 scores, best = 1.0."""
    if not rankings:
        for candidate in candidates:
            candidate.rerank_score = 0.5
        return

    ordered = sorted(rankings.items(), key=lambda kv: kv[1])
    total = len(ordered)
    for position, (index, _) in enumerate(ordered):
        if 0 <= index < len(candidates):
            candidates[index].rerank_score = (
                1.0 if total == 1 else 1.0 - position / (total - 1)
            )


def semantic_deduplicate(
    candidates: list[Candidate],
    backend: SimilarityBackend,
    threshold: float = 0.27,
) -> tuple[list[Candidate], list[str]]:
    """Drop candidates that say the same thing at different times.

    Time-overlap dedup cannot see this: a host making the same point four
    minutes later produces two temporally disjoint candidates that would cut
    into near-identical clips.

    The lexical backend cannot catch paraphrase with no shared vocabulary —
    "the display is asymmetrical" and "the screen has an uneven layout" score
    0.02, indistinguishable from unrelated text. That is a real limit, not a
    tuning problem; the embedding backend exists for it.
    """
    notes: list[str] = []
    ordered = sorted(
        candidates,
        key=lambda c: (c.rerank_score, c.normalized_score),
        reverse=True,
    )
    kept: list[Candidate] = []

    for candidate in ordered:
        text = f"{candidate.hook} {candidate.topic} {candidate.reason}".strip()
        duplicate = False
        for existing in kept:
            other = f"{existing.hook} {existing.topic} {existing.reason}".strip()
            score = backend.similarity(text, other)
            if score >= threshold:
                notes.append(
                    f"dropped '{candidate.hook}' at {candidate.start:.0f}s — "
                    f"{score:.0%} similar to '{existing.hook}' at {existing.start:.0f}s"
                )
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    kept.sort(key=lambda c: c.start)
    return kept, notes


def diversity_scores(candidates: list[Candidate]) -> dict[int, float]:
    """Penalise repeated categories.

    Five clips all tagged "insight" is a worse set than five spread across
    categories, even if each is individually strong. The first of a category
    keeps full marks; later ones decay.
    """
    seen: dict[str, int] = {}
    scores: dict[int, float] = {}
    ordered = sorted(
        range(len(candidates)),
        key=lambda i: candidates[i].rerank_score,
        reverse=True,
    )
    for index in ordered:
        category = candidates[index].category
        count = seen.get(category, 0)
        scores[index] = 1.0 / (1.0 + count * 0.75)
        seen[category] = count + 1
    return scores


def final_scores(
    candidates: list[Candidate],
    min_duration: float,
    max_duration: float,
    weights: ScoringWeights | None = None,
) -> list[Candidate]:
    weights = weights or ScoringWeights()
    diversity = diversity_scores(candidates)

    midpoint = (min_duration + max_duration) / 2
    half_span = max(1e-6, (max_duration - min_duration) / 2)

    for index, candidate in enumerate(candidates):
        fit = max(0.0, 1.0 - abs(candidate.duration - midpoint) / half_span)
        components = {
            "llm_score": candidate.normalized_score,
            "rerank": candidate.rerank_score,
            "boundary": candidate.boundary_hint,
            "duration_fit": fit,
            "diversity": diversity.get(index, 1.0),
        }
        candidate.score_components = {k: round(v, 4) for k, v in components.items()}
        candidate.final_score = round(
            sum(components[name] * getattr(weights, name) for name in components)
            / max(1e-9, weights.total()),
            4,
        )

    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates
