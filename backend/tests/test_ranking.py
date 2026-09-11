"""Tests for Phase 7 ranking.

Driven by what the first real run produced: qwen3:8b returned one candidate per
window, so four of five candidates came out at exactly 0.85 — a flat tie that is
no ranking at all. Independent per-window scoring cannot fix that; only
comparing candidates against each other can.
"""

import itertools

import pytest

from app.services.ai.discovery import Candidate
from app.services.ai.rerank import (
    ScoringWeights,
    apply_rerank,
    build_rerank_prompt,
    diversity_scores,
    excerpt_for,
    final_scores,
    parse_rankings,
    semantic_deduplicate,
)
from app.services.ai.similarity import (
    LexicalSimilarity,
    get_similarity,
    jaccard,
    overlap,
    stem,
    tokenize,
)
from app.services.transcription.base import Segment, Transcript, Word

# The actual hooks qwen3:8b produced, all from genuinely distinct moments.
REAL_HOOKS = [
    "Dynamic island shrinks, but it's still functional",
    "Imagine an iPhone that folds in half",
    "iPhone feels more fun to use than Samsung",
    "This display is totally asymmetrical.",
    "Power Bank works with iPhone Duo and MagSafe",
]

RESTATEMENTS = [
    ("Imagine an iPhone that folds in half", "What if the iPhone could fold in half"),
    ("iPhone feels more fun than Samsung", "Samsung feels more boring than iPhone"),
    ("He lost everything before he understood this",
     "He lost it all before he learned the lesson"),
]


def candidate(start=100.0, hook="h", category="insight", norm=0.5, duration=40.0):
    c = Candidate(start, start + duration, hook, hook.split()[0], category, 80, "r")
    c.normalized_score = norm
    c.boundary_hint = 0.9
    return c


class TestStemmingAndSets:
    @pytest.mark.parametrize("word,expected", [
        ("folds", "fold"), ("folding", "fold"), ("shrinks", "shrink"),
        ("asymmetrical", "asymmetrical"),
    ])
    def test_stem_collapses_inflections(self, word, expected):
        assert stem(word) == expected

    def test_stem_leaves_short_words_alone(self):
        assert stem("is") == "is"
        assert stem("ads") == "ads"

    def test_stopwords_removed(self):
        assert "the" not in tokenize("the iPhone and the display")

    def test_overlap_beats_jaccard_on_containment(self):
        """A short hook fully inside a longer one is a restatement, but
        Jaccard punishes the length difference hard."""
        a, b = {"fold", "iphone"}, {"fold", "iphone", "half", "imagine", "screen"}
        assert overlap(a, b) > jaccard(a, b)

    def test_empty_sets(self):
        assert jaccard(set(), {"a"}) == 0.0
        assert overlap(set(), {"a"}) == 0.0


class TestSimilarityCalibration:
    """The threshold is set by measurement. These tests pin the measurement so
    a metric change that breaks the separation fails loudly."""

    def test_restatements_score_above_the_threshold(self):
        sim = LexicalSimilarity()
        for a, b in RESTATEMENTS:
            assert sim.similarity(a, b) >= 0.27, f"{a!r} vs {b!r}"

    def test_real_distinct_hooks_stay_below_the_threshold(self):
        """False positives are the expensive error: dropping a good clip costs
        more than keeping a near-duplicate the user can reject."""
        sim = LexicalSimilarity()
        for a, b in itertools.combinations(REAL_HOOKS, 2):
            assert sim.similarity(a, b) < 0.27, f"{a!r} vs {b!r}"

    def test_separation_margin_survives(self):
        sim = LexicalSimilarity()
        worst_same = min(sim.similarity(a, b) for a, b in RESTATEMENTS)
        best_diff = max(
            sim.similarity(a, b) for a, b in itertools.combinations(REAL_HOOKS, 2)
        )
        assert worst_same - best_diff > 0.08

    def test_identical_text_is_maximal(self):
        assert LexicalSimilarity().similarity("same words here", "same words here") > 0.95

    def test_empty_input(self):
        assert LexicalSimilarity().similarity("", "anything") == 0.0

    def test_pure_paraphrase_is_a_known_limit(self):
        """No shared vocabulary means lexical matching cannot see it. This is
        a property of the backend, not a bug — documented so a future change
        to the embedding backend has a baseline to beat."""
        score = LexicalSimilarity().similarity(
            "The display is totally asymmetrical", "The screen has an uneven layout"
        )
        assert score < 0.1

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown similarity backend"):
            get_similarity("magic")

    def test_missing_optional_backend_falls_back(self):
        """Degrading dedup quality beats failing the run."""
        assert get_similarity("embedding").name in ("embedding", "lexical")


class TestRankParsing:
    def test_well_formed_ranking(self):
        raw = '[{"id":1,"rank":1},{"id":0,"rank":2},{"id":2,"rank":3}]'
        rankings, problems = parse_rankings(raw, 3)
        assert rankings == {1: 1, 0: 2, 2: 3}
        assert not problems

    def test_fenced_output(self):
        raw = '```json\n[{"id":0,"rank":1},{"id":1,"rank":2}]\n```'
        rankings, _ = parse_rankings(raw, 2)
        assert rankings[0] == 1

    def test_missing_entries_go_to_the_back(self):
        """A model omitting a candidate is not evidence the candidate is bad."""
        rankings, problems = parse_rankings('[{"id":0,"rank":1}]', 3)
        assert set(rankings) == {0, 1, 2}
        assert rankings[1] > rankings[0] and rankings[2] > rankings[0]
        assert any("unranked" in p for p in problems)

    def test_duplicate_ids_are_reported_not_fatal(self):
        raw = '[{"id":0,"rank":1},{"id":0,"rank":2},{"id":1,"rank":3}]'
        rankings, problems = parse_rankings(raw, 2)
        assert rankings[0] == 1
        assert any("twice" in p for p in problems)

    def test_out_of_range_ids_are_reported(self):
        rankings, problems = parse_rankings('[{"id":99,"rank":1},{"id":0,"rank":2}]', 2)
        assert 99 not in rankings
        assert any("unknown candidate" in p for p in problems)

    def test_unparseable_output_returns_empty(self):
        rankings, problems = parse_rankings("I cannot rank these.", 3)
        assert rankings == {}
        assert problems


class TestApplyRerank:
    def test_best_gets_one_worst_gets_zero(self):
        candidates = [candidate(start=i * 100) for i in range(4)]
        apply_rerank(candidates, {0: 3, 1: 1, 2: 2, 3: 4})
        assert candidates[1].rerank_score == 1.0
        assert candidates[3].rerank_score == 0.0

    def test_single_candidate(self):
        candidates = [candidate()]
        apply_rerank(candidates, {0: 1})
        assert candidates[0].rerank_score == 1.0

    def test_no_rankings_leaves_scores_neutral(self):
        candidates = [candidate(start=i * 100) for i in range(3)]
        apply_rerank(candidates, {})
        assert all(c.rerank_score == 0.5 for c in candidates)


class TestDiversity:
    def test_repeated_category_decays(self):
        candidates = [candidate(start=i * 100, category="insight") for i in range(3)]
        for i, c in enumerate(candidates):
            c.rerank_score = 1.0 - i * 0.1
        scores = diversity_scores(candidates)
        assert scores[0] > scores[1] > scores[2]

    def test_distinct_categories_all_keep_full_marks(self):
        categories = ["insight", "story", "funny"]
        candidates = [candidate(start=i * 100, category=c)
                      for i, c in enumerate(categories)]
        assert all(v == 1.0 for v in diversity_scores(candidates).values())


class TestFinalScoring:
    def test_breaks_the_flat_tie_from_the_real_run(self):
        """Four candidates at 0.85 and one at 0.15 — the exact output of the
        first real discovery run, which produced no usable ranking."""
        candidates = [
            candidate(start=145, hook=REAL_HOOKS[0], category="insight", norm=0.85),
            candidate(start=302, hook=REAL_HOOKS[1], category="insight", norm=0.85),
            candidate(start=595, hook=REAL_HOOKS[2], category="insight", norm=0.85),
            candidate(start=763, hook=REAL_HOOKS[3], category="curiosity", norm=0.85),
            candidate(start=981, hook=REAL_HOOKS[4], category="insight", norm=0.15),
        ]
        apply_rerank(candidates, {1: 1, 3: 2, 0: 3, 2: 4, 4: 5})
        final_scores(candidates, 25.0, 60.0)
        scores = [c.final_score for c in candidates]
        assert len(set(scores)) == len(scores), "candidates still tied"

    def test_sorted_best_first(self):
        candidates = [candidate(start=i * 200, hook=h)
                      for i, h in enumerate(REAL_HOOKS[:3])]
        apply_rerank(candidates, {0: 3, 1: 1, 2: 2})
        final_scores(candidates, 25.0, 60.0)
        assert candidates[0].rerank_score == 1.0

    def test_components_are_recorded(self):
        candidates = [candidate()]
        final_scores(candidates, 25.0, 60.0)
        assert set(candidates[0].score_components) == {
            "llm_score", "rerank", "boundary", "duration_fit", "diversity",
        }

    def test_scores_stay_in_range(self):
        candidates = [candidate(start=i * 100, norm=n)
                      for i, n in enumerate([0.0, 0.5, 1.0])]
        final_scores(candidates, 25.0, 60.0)
        assert all(0.0 <= c.final_score <= 1.0 for c in candidates)

    def test_weights_change_the_order(self):
        a = candidate(start=100, hook="first one", norm=1.0)
        a.rerank_score = 0.0
        b = candidate(start=500, hook="second one", norm=0.0)
        b.rerank_score = 1.0

        final_scores([a, b], 25, 60, ScoringWeights(llm_score=1.0, rerank=0.0,
                                                    boundary=0.0, duration_fit=0.0,
                                                    diversity=0.0))
        assert a.final_score > b.final_score

        final_scores([a, b], 25, 60, ScoringWeights(llm_score=0.0, rerank=1.0,
                                                    boundary=0.0, duration_fit=0.0,
                                                    diversity=0.0))
        assert b.final_score > a.final_score

    def test_poor_boundary_lowers_a_strong_hook(self):
        """A moment the solver can only cut badly is worth less than its hook
        suggests, and that is known before anything is rendered."""
        good = candidate(start=100, hook="clean cut here", norm=0.8)
        bad = candidate(start=500, hook="messy cut here", norm=0.8)
        good.boundary_hint, bad.boundary_hint = 1.0, 0.0
        good.rerank_score = bad.rerank_score = 0.5
        final_scores([good, bad], 25.0, 60.0)
        assert good.final_score > bad.final_score


class TestSemanticDedup:
    def _transcript(self):
        words = [Word(f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(200)]
        return Transcript([Segment(0, 80, " ".join(w.text for w in words), words)],
                          "en", "s", "m", 80.0)

    def test_restatement_at_a_different_time_is_dropped(self):
        """Time-overlap dedup cannot see this: the two are minutes apart."""
        candidates = [
            candidate(start=100, hook="Imagine an iPhone that folds in half"),
            candidate(start=500, hook="What if the iPhone could fold in half"),
        ]
        kept, notes = semantic_deduplicate(candidates, LexicalSimilarity(), 0.27)
        assert len(kept) == 1
        assert notes and "similar to" in notes[0]

    def test_real_distinct_hooks_all_survive(self):
        candidates = [candidate(start=i * 200, hook=h)
                      for i, h in enumerate(REAL_HOOKS)]
        kept, _ = semantic_deduplicate(candidates, LexicalSimilarity(), 0.27)
        assert len(kept) == len(REAL_HOOKS)

    def test_the_better_ranked_one_survives(self):
        weak = candidate(start=100, hook="Imagine an iPhone that folds in half")
        strong = candidate(start=500, hook="What if the iPhone could fold in half")
        weak.rerank_score, strong.rerank_score = 0.1, 0.9
        kept, _ = semantic_deduplicate([weak, strong], LexicalSimilarity(), 0.27)
        assert kept[0].start == 500

    def test_output_is_chronological(self):
        candidates = [candidate(start=s, hook=h)
                      for s, h in zip([900, 100, 500], REAL_HOOKS)]
        kept, _ = semantic_deduplicate(candidates, LexicalSimilarity(), 0.27)
        assert [c.start for c in kept] == sorted(c.start for c in kept)


class TestRerankPrompt:
    def _transcript(self):
        words = [Word(f"word{i}", i * 0.4, i * 0.4 + 0.3) for i in range(300)]
        return Transcript([Segment(0, 120, " ".join(w.text for w in words), words)],
                          "en", "s", "m", 120.0)

    def test_includes_every_candidate_with_an_id(self):
        transcript = self._transcript()
        candidates = [candidate(start=i * 20, hook=h) for i, h in enumerate(REAL_HOOKS[:3])]
        prompt = build_rerank_prompt(candidates, transcript)
        for i in range(3):
            assert f"[{i}]" in prompt

    def test_demands_every_id_once(self):
        transcript = self._transcript()
        prompt = build_rerank_prompt([candidate()], transcript)
        assert "exactly once" in prompt

    def test_excerpt_is_bounded(self):
        transcript = self._transcript()
        assert len(excerpt_for(transcript, 0, 120, limit=100)) <= 101


class TestAnchorTopUp:
    """Discovery returning five moments must not cap the output at five clips."""

    def _transcript(self):
        sentences = ["Revenue tripled in a single quarter after the change.",
                     "Nobody believed it would work at first.",
                     "We lost four hundred thousand dollars learning that.",
                     "The whole team quit within a month."]
        segments, t, i = [], 0.0, 0
        while t < 1000:
            text = sentences[i % 4]
            words, wt = [], t
            for token in text.split():
                words.append(Word(token, round(wt, 3), round(wt + 0.3, 3), 0.95))
                wt += 0.42
            segments.append(Segment(t, round(wt, 3), text, words))
            t = wt + 0.4
            i += 1
        return Transcript(segments, "en", "s", "m", duration=t)

    def test_few_anchors_still_yield_the_requested_count(self):
        from app.services.clips.select import select_clips

        anchors = [{"start": 145.5, "hook": "a"}, {"start": 400.0, "hook": "b"}]
        clips = select_clips(self._transcript(), 6, 25.0, 60.0, anchors=anchors)
        assert len(clips) == 6

    def test_discovered_anchors_are_kept_and_labelled(self):
        from app.services.clips.select import select_clips

        anchors = [{"start": 145.5, "hook": "a"}, {"start": 400.0, "hook": "b"}]
        clips = select_clips(self._transcript(), 6, 25.0, 60.0, anchors=anchors)
        discovered = [c for c in clips if c.strategy == "discovered"]
        assert len(discovered) == 2
        assert all(c.discovery for c in discovered)

    def test_filler_clips_are_labelled_even(self):
        """The source of each clip stays visible rather than being conflated."""
        from app.services.clips.select import select_clips

        anchors = [{"start": 145.5, "hook": "a"}]
        clips = select_clips(self._transcript(), 5, 25.0, 60.0, anchors=anchors)
        assert any(c.strategy == "even" for c in clips)
        assert all(c.discovery is None for c in clips if c.strategy == "even")

    def test_enough_anchors_means_no_filler(self):
        from app.services.clips.select import select_clips

        anchors = [{"start": s, "hook": "h"} for s in (100, 300, 500, 700)]
        clips = select_clips(self._transcript(), 3, 25.0, 60.0, anchors=anchors)
        assert all(c.strategy == "discovered" for c in clips)
