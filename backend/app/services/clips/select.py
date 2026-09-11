"""Clip selection.

Anchors are still spaced evenly through the podcast — `strategy="even"` says so
plainly, and Phase 6 replaces that with real discovery. What changed in Phase 5
is everything after the anchor: boundaries are now solved rather than grown
greedily, scored on opening quality, sentence completeness, filler density and
duration fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.clips.boundaries import (
    BoundaryWeights,
    pad_into_silence,
    solve_boundaries,
)
from app.services.transcription.base import Transcript, Word


@dataclass
class ClipWindow:
    start: float
    end: float
    words: list[Word]
    text: str
    strategy: str = "even"
    discovery: dict | None = None
    boundary_score: float = 0.0
    boundary_notes: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def opener_penalty(self) -> bool:
        return self.components.get("opener", 1.0) < 1.0


def anchors_for(duration: float, count: int, edge_fraction: float = 0.05) -> list[float]:
    """Evenly spaced start points, skipping intro and outro.

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


def _overlaps(a: ClipWindow, b: ClipWindow) -> bool:
    return a.start < b.end and b.start < a.end


def select_clips(
    transcript: Transcript,
    count: int,
    min_duration: float = 25.0,
    max_duration: float = 60.0,
    padding: float = 0.35,
    weights: BoundaryWeights | None = None,
    min_boundary_score: float = 0.0,
    anchors: list[dict] | None = None,
) -> list[ClipWindow]:
    """Pick `count` non-overlapping, well-bounded windows.

    Over-samples anchors then drops overlaps, because solving moves a window
    away from its anchor and neighbouring anchors can converge on the same span.
    """
    if not transcript.segments:
        return []

    duration = transcript.duration or transcript.segments[-1].end
    windows: list[ClipWindow] = []

    # Discovered anchors carry the model's judgement; even spacing carries
    # none. Either way the solver refines the actual cut points — discovery
    # chooses *where* to look, Phase 5 chooses *where to cut*.
    anchor_meta: dict[float, dict] = {}
    if anchors:
        anchor_points = [float(a["start"]) for a in anchors]
        anchor_meta = {float(a["start"]): a for a in anchors}
        # Discovery can return fewer moments than the user asked for — five
        # candidates would otherwise cap the output at five clips regardless of
        # the request. Even-spaced points are appended as a fallback tail so
        # the count can still be met; they are marked "even" so the source of
        # each clip stays visible rather than being quietly conflated.
        if len(anchor_points) < count:
            anchor_points += [
                a for a in anchors_for(duration, count * 3)
                if all(abs(a - existing) > min_duration for existing in anchor_points)
            ]
    else:
        anchor_points = anchors_for(duration, count * 3)

    for anchor in anchor_points:
        strategy = "discovered" if anchor in anchor_meta else "even"
        candidate = solve_boundaries(
            transcript, anchor, min_duration, max_duration, weights
        )
        if candidate is None:
            continue
        if candidate.score.total < min_boundary_score:
            continue

        segments = [s for s in transcript.segments if s.words]
        chosen = segments[candidate.start_index : candidate.end_index + 1]
        words = [w for s in chosen for w in s.words]
        if not words:
            continue

        window = ClipWindow(
            start=candidate.start,
            end=candidate.end,
            words=words,
            text=" ".join(s.text.strip() for s in chosen).strip(),
            strategy=strategy,
            boundary_score=candidate.score.total,
            boundary_notes=candidate.score.notes,
            components=candidate.score.components,
            discovery=anchor_meta.get(anchor),
        )

        if any(_overlaps(window, existing) for existing in windows):
            continue

        windows.append(window)
        if len(windows) >= count:
            break

    windows.sort(key=lambda w: w.start)

    # Pad into whatever silence actually surrounds the clip, never past the
    # configured maximum.
    for window in windows:
        headroom = max_duration - window.duration
        if headroom <= 0:
            continue
        allowance = min(padding, headroom / 2)
        window.start, window.end = pad_into_silence(
            transcript, window.start, window.end, allowance, duration
        )

    return windows
