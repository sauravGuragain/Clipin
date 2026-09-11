"""Candidate discovery orchestration."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path

from app.core.config import settings
from app.core.jobs import JobContext
from app.models.db import Project, get_session
from app.services.ai.discovery import (
    SYSTEM_PROMPT,
    Candidate,
    build_prompt,
    check_context_budget,
    deduplicate,
    normalize_scores,
    parse_candidates,
    window_transcript,
)
from app.services.ai.providers import LLMError, LLMUnavailable, get_provider
from app.services.ai.rerank import (
    RERANK_SYSTEM,
    ScoringWeights,
    apply_rerank,
    build_rerank_prompt,
    final_scores,
    parse_rankings,
    semantic_deduplicate,
)
from app.services.ai.similarity import get_similarity
from app.services.clips.boundaries import solve_boundaries
from app.services.transcription.base import Transcript
from app.services.transcription.service import transcript_path


class DiscoveryError(Exception):
    pass


REJECTION_KINDS = (
    ("degenerate span", "spans too short to locate anything"),
    ("do not match the transcript", "invented timestamps"),
    ("No valid JSON", "unparseable output"),
    ("Expected a JSON array", "wrong JSON shape"),
    ("missing start/end", "missing timestamps"),
    ("not an object", "malformed entries"),
)


def summarise_rejections(rejections: list[str]) -> str:
    """Report the pattern, not the first instance.

    Thirty identical failures and one unlucky candidate need different
    responses, and quoting only the first rejection makes them look the same.
    """
    if not rejections:
        return "It returned nothing parseable."

    counts: Counter = Counter()
    for entry in rejections:
        for needle, label in REJECTION_KINDS:
            if needle in entry:
                counts[label] += 1
                break
        else:
            counts["other"] += 1

    parts = [f"{count} x {label}" for label, count in counts.most_common()]
    return (
        f"{len(rejections)} candidate(s) rejected: {', '.join(parts)}. "
        f"Example: {rejections[0]}"
    )


def candidates_path(project_id: str) -> Path:
    return settings.projects_dir / project_id / "candidates.json"


def load_candidates(project_id: str) -> list[dict]:
    """Candidates, best first.

    Sorted on read rather than trusting file order: older files predate
    final_score, and a silently mis-ordered list would look like bad ranking
    rather than a stale file.
    """
    path = candidates_path(project_id)
    if not path.exists():
        return []
    try:
        items = json.loads(path.read_text()).get("candidates", [])
    except (json.JSONDecodeError, OSError):
        return []
    items.sort(
        key=lambda c: c.get("final_score", c.get("normalized_score", 0.0)),
        reverse=True,
    )
    return items


def save_candidates(project_id: str, payload: dict) -> Path:
    path = candidates_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    return path


def discover_sync(
    transcript: Transcript,
    provider_name: str | None,
    model: str | None,
    per_window: int,
    min_duration: float,
    max_duration: float,
    target_tokens: int,
    retries: int,
    rerank: bool = True,
    on_window_done=None,
    should_cancel=None,
) -> dict:
    """Blocking discovery across all windows. Runs in a worker thread."""
    provider = get_provider(provider_name)
    available, reason = provider.is_available()
    if not available:
        raise DiscoveryError(reason)

    resolved_model = model or provider.default_model
    windows = window_transcript(transcript, target_tokens=target_tokens)
    if not windows:
        raise DiscoveryError("The transcript has no usable text.")

    all_candidates: list[Candidate] = []
    rejections: list[str] = []
    failed_windows: list[int] = []
    total_elapsed = 0.0

    # Check the budget once, on the largest window, before spending minutes
    # generating against a truncated prompt.
    max_output = settings.llm_max_output
    num_ctx = getattr(provider, "num_ctx", settings.llm_num_ctx)
    widest = max(windows, key=lambda w: len(w.render()))
    warning = check_context_budget(
        build_prompt(widest, per_window, min_duration, max_duration),
        max_output, num_ctx,
    )
    if warning:
        raise DiscoveryError(warning)

    for position, window in enumerate(windows):
        if should_cancel and should_cancel():
            from app.core.jobs import JobCancelled
            raise JobCancelled()

        prompt = build_prompt(window, per_window, min_duration, max_duration)
        parsed: list[Candidate] = []

        for attempt in range(retries + 1):
            try:
                result = provider.complete(
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    model=resolved_model,
                    json_mode=True,
                    temperature=0.2 + attempt * 0.2,   # nudge off a bad path
                    max_tokens=max_output,
                )
            except LLMUnavailable:
                raise
            except LLMError as exc:
                if attempt >= retries:
                    rejections.append(f"window {window.index}: {exc}")
                    failed_windows.append(window.index)
                    break
                continue

            total_elapsed += result.elapsed
            parsed, window_rejections = parse_candidates(
                result.text, transcript, window, min_duration, max_duration
            )
            if parsed:
                rejections.extend(f"window {window.index}: {r}" for r in window_rejections)
                break

            # Nothing usable. Retrying costs a full generation, so only do it
            # if there is an attempt left — and record why.
            if attempt >= retries:
                rejections.extend(
                    f"window {window.index}: {r}" for r in (window_rejections or ["no candidates returned"])
                )
                failed_windows.append(window.index)

        all_candidates.extend(parsed)
        if on_window_done:
            on_window_done(position + 1, len(windows))

    normalize_scores(all_candidates)
    deduped = deduplicate(all_candidates, settings.candidate_iou_threshold)

    # --- boundary hint --------------------------------------------------
    # How cleanly can this moment actually be cut? A candidate the solver can
    # only cut badly is worth less than its hook suggests, and that is known
    # before any clip is rendered.
    for candidate in deduped:
        solved = solve_boundaries(transcript, candidate.start, min_duration, max_duration)
        candidate.boundary_hint = solved.score.total if solved else 0.0

    # --- listwise rerank -------------------------------------------------
    # The only thing that breaks cross-window ties: show the model the
    # candidates together and ask it to compare them. One call for the whole
    # podcast, not one per window.
    rerank_notes: list[str] = []
    if rerank and len(deduped) > 1:
        shortlist = sorted(deduped, key=lambda c: c.normalized_score, reverse=True)
        shortlist = shortlist[: settings.rerank_shortlist]
        try:
            result = provider.complete(
                prompt=build_rerank_prompt(shortlist, transcript),
                system=RERANK_SYSTEM,
                model=resolved_model,
                json_mode=True,
                max_tokens=max_output,
            )
            total_elapsed += result.elapsed
            rankings, problems = parse_rankings(result.text, len(shortlist))
            apply_rerank(shortlist, rankings)
            rerank_notes.extend(problems)
            if not rankings:
                rerank_notes.append("rerank returned nothing usable; scores left flat")
        except LLMError as exc:
            # Reranking is an improvement, not a requirement. Losing it costs
            # ranking quality, not the whole run.
            rerank_notes.append(f"rerank failed: {exc}")

    # --- semantic dedup ---------------------------------------------------
    similarity = get_similarity(settings.similarity_backend)
    deduped, similarity_notes = semantic_deduplicate(
        deduped, similarity, settings.semantic_dedup_threshold
    )

    # --- final blended score ----------------------------------------------
    weights = ScoringWeights(
        llm_score=settings.weight_llm_score,
        rerank=settings.weight_rerank,
        boundary=settings.weight_boundary,
        duration_fit=settings.weight_duration_fit_final,
        diversity=settings.weight_diversity,
    )
    final_scores(deduped, min_duration, max_duration, weights)

    return {
        "rerank_notes": rerank_notes,
        "similarity_backend": similarity.name,
        "similarity_notes": similarity_notes,
        "provider": provider.name,
        "model": resolved_model,
        "num_ctx": num_ctx,
        "windows": len(windows),
        "failed_windows": failed_windows,
        "raw_count": len(all_candidates),
        "candidates": [c.to_dict() for c in deduped],
        "rejections": rejections[:40],
        "rejection_summary": summarise_rejections(rejections) if rejections else None,
        "llm_seconds": round(total_elapsed, 1),
    }


async def handle_discover(ctx: JobContext) -> dict:
    """Job handler for type 'discover'."""
    project_id = ctx.project_id
    if not project_id:
        raise DiscoveryError("No project supplied.")

    with get_session() as db:
        if db.get(Project, project_id) is None:
            raise DiscoveryError("Project no longer exists.")

    path = transcript_path(project_id)
    if not path.exists():
        raise DiscoveryError("Transcribe this project before discovering clips.")

    try:
        transcript = Transcript.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, OSError) as exc:
        raise DiscoveryError("The stored transcript could not be read.") from exc

    params = ctx.params
    provider_name = params.get("provider") or settings.llm_provider
    model = params.get("model") or settings.llm_model or None
    per_window = int(params.get("per_window", settings.candidates_per_window))
    min_duration = float(params.get("min_duration", settings.clip_min_duration))
    max_duration = float(params.get("max_duration", settings.clip_max_duration))
    target_tokens = int(params.get("target_tokens", settings.llm_window_tokens))
    retries = int(params.get("retries", settings.llm_retries))

    await ctx.progress(0.02, "preparing windows")
    loop = asyncio.get_running_loop()

    def on_window_done(done: int, total: int) -> None:
        asyncio.run_coroutine_threadsafe(
            ctx.progress(done / total * 0.95, f"analysing window {done}/{total}"), loop
        )

    started = time.monotonic()
    result = await asyncio.to_thread(
        discover_sync,
        transcript, provider_name, model, per_window,
        min_duration, max_duration, target_tokens, retries,
        bool(params.get("rerank", settings.rerank_enabled)),
        on_window_done, lambda: ctx.cancelled,
    )
    elapsed = time.monotonic() - started

    ctx.raise_if_cancelled()

    if not result["candidates"]:
        raise DiscoveryError(
            "The model found no usable moments. " + summarise_rejections(result["rejections"])
        )

    await ctx.progress(0.98, "saving")
    result["elapsed_seconds"] = round(elapsed, 1)
    save_candidates(project_id, result)

    summary = dict(result)
    summary["candidates"] = len(result["candidates"])
    summary["top"] = result["candidates"][:5]
    return summary
