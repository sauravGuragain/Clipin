"""Tests for LLM-driven clip discovery.

The model call itself cannot run here — Ollama is not reachable and no weights
are available — so it is verified on the target machine. Everything around it is
tested, and the weight is deliberately on the failure paths: an 8B local model
returns malformed JSON and invented timestamps often enough that the recovery
code runs more than the happy path does.
"""

import pytest

from app.services.ai.discovery import (
    CATEGORIES,
    Candidate,
    build_prompt,
    deduplicate,
    estimate_tokens,
    normalize_scores,
    parse_candidates,
    snap_to_transcript,
    window_transcript,
)
from app.services.ai.json_recovery import (
    JSONRecoveryError,
    extract_json,
    extract_json_list,
)
from app.services.ai.providers import (
    LLMUnavailable,
    StubProvider,
    get_provider,
)
from app.services.transcription.base import Segment, Transcript, Word


def make_transcript(total=600.0):
    sentences = [
        "Revenue tripled in a single quarter.",
        "Nobody believed it would work at first.",
        "We lost four hundred thousand dollars learning that.",
        "The whole team quit within a month.",
    ]
    segments, t, i = [], 0.0, 0
    while t < total:
        text = sentences[i % len(sentences)]
        words, wt = [], t
        for token in text.split():
            words.append(Word(token, round(wt, 3), round(wt + 0.3, 3), 0.95))
            wt += 0.42
        segments.append(Segment(t, round(wt, 3), text, words))
        t = wt + 0.4
        i += 1
    return Transcript(segments, "en", "stub", "m", duration=t)


VALID = '[{"start": 10.0, "end": 45.0, "hook": "He lost everything", "topic": "failure", "category": "story", "score": 87, "reason": "Complete arc."}]'


class TestJSONRecovery:
    """Every input here is something a local model actually produces."""

    def test_plain_array(self):
        assert extract_json('[{"a": 1}]') == [{"a": 1}]

    def test_markdown_fence(self):
        assert extract_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]

    def test_bare_fence(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_prose_preamble(self):
        raw = 'Sure! Here are the best moments:\n[{"a": 1}]'
        assert extract_json(raw) == [{"a": 1}]

    def test_prose_on_both_sides(self):
        raw = 'Here you go:\n[{"a": 1}]\nLet me know if you want more!'
        assert extract_json(raw) == [{"a": 1}]

    def test_trailing_comma_in_object(self):
        assert extract_json('{"a": 1,}') == {"a": 1}

    def test_trailing_comma_in_array(self):
        assert extract_json('[1, 2, 3,]') == [1, 2, 3]

    def test_smart_quotes(self):
        assert extract_json('{\u201ca\u201d: 1}') == {"a": 1}

    def test_unquoted_keys(self):
        assert extract_json('{start: 10, end: 20}') == {"start": 10, "end": 20}

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        raw = 'text before {"hook": "he said {this} out loud", "n": 1} after'
        assert extract_json(raw)["hook"] == "he said {this} out loud"

    def test_escaped_quote_inside_string(self):
        raw = '{"hook": "she said \\"no\\" twice"}'
        assert extract_json(raw)["hook"] == 'she said "no" twice'

    def test_empty_input_raises(self):
        with pytest.raises(JSONRecoveryError):
            extract_json("")

    def test_pure_prose_raises(self):
        with pytest.raises(JSONRecoveryError):
            extract_json("I could not find any good moments in this transcript.")

    def test_error_message_includes_a_preview(self):
        with pytest.raises(JSONRecoveryError, match="totally broken"):
            extract_json("totally broken output")


class TestExtractJSONList:
    def test_plain_list(self):
        assert extract_json_list('[{"a": 1}]') == [{"a": 1}]

    @pytest.mark.parametrize("key", ["clips", "candidates", "moments", "results", "items"])
    def test_wrapped_in_common_keys(self, key):
        assert extract_json_list(f'{{"{key}": [{{"a": 1}}]}}') == [{"a": 1}]

    def test_single_object_becomes_one_item(self):
        """A model returning one moment as a bare object is a formatting
        preference, not an error — discarding it loses a real candidate."""
        result = extract_json_list('{"start": 1, "end": 2, "hook": "x"}')
        assert len(result) == 1

    def test_unusually_named_single_list_is_accepted(self):
        assert extract_json_list('{"best_bits": [{"a": 1}]}') == [{"a": 1}]

    def test_scalar_raises(self):
        with pytest.raises(JSONRecoveryError):
            extract_json_list("42")


class TestWindowing:
    def test_estimate_tokens_scales_with_length(self):
        assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)

    def test_windows_cover_the_transcript(self):
        tr = make_transcript(600)
        windows = window_transcript(tr, target_tokens=300)
        assert windows
        assert windows[0].start == pytest.approx(tr.segments[0].start)
        assert windows[-1].end == pytest.approx(tr.segments[-1].end)

    def test_windows_overlap(self):
        """A moment straddling a boundary must be visible to both windows,
        otherwise it is invisible to each."""
        windows = window_transcript(make_transcript(600), target_tokens=300,
                                    overlap_segments=2)
        assert len(windows) > 1
        for current, following in zip(windows, windows[1:]):
            assert following.start < current.end

    def test_windows_always_advance(self):
        """Overlap larger than the window would otherwise loop forever."""
        windows = window_transcript(make_transcript(300), target_tokens=100,
                                    overlap_segments=50)
        starts = [w.start for w in windows]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)

    def test_rendered_window_carries_timestamps(self):
        window = window_transcript(make_transcript(120), target_tokens=400)[0]
        assert "[0.0]" in window.render()

    def test_empty_transcript(self):
        assert window_transcript(Transcript([], "en", "s", "m", 0)) == []


class TestTimestampSnapping:
    def test_near_miss_snaps_to_a_real_boundary(self):
        tr = make_transcript(120)
        result = snap_to_transcript(tr, 10.3, 40.2)
        assert result is not None
        starts = {w.start for w in tr.words}
        assert result[0] in starts

    def test_hallucinated_time_is_rejected(self):
        """Models invent numbers. A time far from anything real means the
        candidate is fabricated, and guessing at it would be worse."""
        tr = make_transcript(120)
        assert snap_to_transcript(tr, 9000.0, 9050.0) is None

    def test_inverted_range_is_rejected(self):
        """An end well before the start cannot be repaired by snapping."""
        tr = make_transcript(120)
        assert snap_to_transcript(tr, 80.0, 20.0, tolerance=90) is None

    def test_degenerate_range_expands_to_real_boundaries(self):
        """start == end is not an error: each side snaps to its own nearest
        real boundary, which yields a short but valid forward span."""
        tr = make_transcript(120)
        result = snap_to_transcript(tr, 50.0, 50.0, tolerance=60)
        assert result is not None
        assert result[1] > result[0]

    def test_empty_transcript_returns_none(self):
        assert snap_to_transcript(Transcript([], "en", "s", "m", 0), 1, 2) is None


class TestCandidateParsing:
    def _window(self, tr):
        return window_transcript(tr, target_tokens=5000)[0]

    def test_valid_candidate_parses(self):
        tr = make_transcript(200)
        found, rejected = parse_candidates(VALID, tr, self._window(tr), 10, 90)
        assert len(found) == 1
        assert found[0].hook == "He lost everything"
        assert found[0].category == "story"

    def test_fenced_output_parses(self):
        tr = make_transcript(200)
        found, _ = parse_candidates(f"```json\n{VALID}\n```", tr, self._window(tr), 10, 90)
        assert len(found) == 1

    def test_string_numbers_are_coerced(self):
        tr = make_transcript(200)
        raw = '[{"start": "10 seconds", "end": "45", "hook": "x", "topic": "t", "category": "story", "score": "87", "reason": "r"}]'
        found, _ = parse_candidates(raw, tr, self._window(tr), 10, 90)
        assert len(found) == 1

    def test_alternate_key_names_accepted(self):
        tr = make_transcript(200)
        raw = '[{"start_time": 10, "end_time": 45, "hook": "x", "category": "story", "score": 80}]'
        found, _ = parse_candidates(raw, tr, self._window(tr), 10, 90)
        assert len(found) == 1

    def test_reversed_times_are_swapped(self):
        tr = make_transcript(200)
        raw = '[{"start": 45, "end": 10, "hook": "x", "category": "story", "score": 80}]'
        found, _ = parse_candidates(raw, tr, self._window(tr), 10, 90)
        assert len(found) == 1
        assert found[0].start < found[0].end

    def test_hallucinated_timestamps_are_rejected_with_a_reason(self):
        tr = make_transcript(200)
        raw = '[{"start": 8000, "end": 8040, "hook": "x", "category": "story", "score": 90}]'
        found, rejected = parse_candidates(raw, tr, self._window(tr), 10, 90)
        assert found == []
        assert any("do not match" in r for r in rejected)

    def test_too_short_is_kept_as_an_anchor(self):
        """Regression: qwen3:8b asked for 25-60s moments returned 9.7s ones,
        and rejecting them outright produced zero candidates for a whole
        podcast. A candidate is an anchor — the Phase 5 solver decides the real
        boundaries — so a short proposal still points at a real moment."""
        tr = make_transcript(200)
        raw = '[{"start": 10, "end": 19.7, "hook": "x", "category": "story", "score": 90}]'
        found, _ = parse_candidates(raw, tr, self._window(tr), 25, 90)
        assert len(found) == 1
        assert any("below the minimum" in i for i in found[0].issues)

    def test_degenerate_span_is_still_rejected(self):
        """Below a couple of seconds there is no information about *where* the
        moment is either, so it is noise rather than a pointer."""
        tr = make_transcript(200)
        raw = '[{"start": 10, "end": 10.4, "hook": "x", "category": "story", "score": 90}]'
        found, rejected = parse_candidates(raw, tr, self._window(tr), 25, 90)
        assert found == []
        assert any("degenerate" in r for r in rejected)

    def test_duration_violations_do_not_starve_discovery(self):
        """The end-to-end shape of the bug: every candidate violating duration
        must not mean zero candidates."""
        tr = make_transcript(400)
        raw = ('[{"start": 10, "end": 20, "hook": "a", "category": "story", "score": 90},'
               ' {"start": 100, "end": 108, "hook": "b", "category": "insight", "score": 80},'
               ' {"start": 200, "end": 380, "hook": "c", "category": "story", "score": 70}]')
        found, _ = parse_candidates(raw, tr, self._window(tr), 25, 60)
        assert len(found) == 3

    def test_too_long_is_kept_but_flagged(self):
        """Phase 5's solver re-cuts boundaries anyway, so an over-long proposal
        is still a useful pointer at where to look."""
        tr = make_transcript(400)
        raw = '[{"start": 10, "end": 200, "hook": "x", "category": "story", "score": 90}]'
        found, _ = parse_candidates(raw, tr, self._window(tr), 25, 60)
        assert len(found) == 1
        assert any("above the maximum" in i for i in found[0].issues)

    def test_unknown_category_falls_back_and_is_flagged(self):
        tr = make_transcript(200)
        raw = '[{"start": 10, "end": 45, "hook": "x", "category": "spicy", "score": 90}]'
        found, _ = parse_candidates(raw, tr, self._window(tr), 10, 90)
        assert found[0].category in CATEGORIES
        assert any("unknown category" in i for i in found[0].issues)

    def test_missing_score_defaults_and_is_flagged(self):
        tr = make_transcript(200)
        raw = '[{"start": 10, "end": 45, "hook": "x", "category": "story"}]'
        found, _ = parse_candidates(raw, tr, self._window(tr), 10, 90)
        assert found[0].raw_score == 50.0
        assert any("no score" in i for i in found[0].issues)

    def test_score_is_clamped(self):
        tr = make_transcript(200)
        raw = '[{"start": 10, "end": 45, "hook": "x", "category": "story", "score": 9000}]'
        found, _ = parse_candidates(raw, tr, self._window(tr), 10, 90)
        assert found[0].raw_score == 100.0

    def test_non_object_entries_are_rejected(self):
        tr = make_transcript(200)
        found, rejected = parse_candidates('["just a string"]', tr, self._window(tr), 10, 90)
        assert found == []
        assert rejected

    def test_unparseable_output_returns_a_reason_not_an_exception(self):
        tr = make_transcript(200)
        found, rejected = parse_candidates("I cannot help with that.", tr,
                                           self._window(tr), 10, 90)
        assert found == []
        assert len(rejected) == 1


class TestScoreNormalization:
    """Raw scores come from independent calls. An 87 from one window and an 87
    from another do not mean the same thing."""

    def _candidate(self, score, chunk=0, start=0.0):
        return Candidate(start, start + 30, "h", "t", "story", score, "r", chunk_index=chunk)

    def test_scores_normalise_within_a_chunk(self):
        group = [self._candidate(s, 0, i * 100) for i, s in enumerate([10, 50, 90])]
        normalize_scores(group)
        assert group[0].normalized_score < group[1].normalized_score < group[2].normalized_score

    def test_chunks_are_normalised_independently(self):
        """A window where everything scored 80-90 and one where everything
        scored 20-30 should both produce a full spread."""
        generous = [self._candidate(s, 0, i * 100) for i, s in enumerate([80, 85, 90])]
        harsh = [self._candidate(s, 1, 1000 + i * 100) for i, s in enumerate([20, 25, 30])]
        normalize_scores(generous + harsh)
        assert max(c.normalized_score for c in harsh) > min(c.normalized_score for c in generous)

    def test_lone_candidate_overall_is_neutral(self):
        group = [self._candidate(95, 0)]
        normalize_scores(group)
        assert group[0].normalized_score == 0.5
        assert group[0].confidence <= 0.5

    def test_one_candidate_per_window_still_ranks(self):
        """The common case: a model asked for three moments returns one.
        Per-window z-scoring has no distribution to work with, and leaving
        everything tied at 0.5 discards real information — the model meant
        something by scoring one moment 93 and another 55."""
        group = [
            self._candidate(88, 0, 20), self._candidate(71, 1, 140),
            self._candidate(93, 2, 260), self._candidate(55, 3, 600),
        ]
        normalize_scores(group)
        scores = [c.normalized_score for c in group]
        assert len(set(scores)) == len(scores), "all candidates tied"
        ranked = sorted(group, key=lambda c: -c.normalized_score)
        assert [c.raw_score for c in ranked] == [93, 88, 71, 55]

    def test_cross_window_scores_carry_lower_confidence(self):
        """A comparison across independent calls is weaker evidence than one
        within a single call, and the data should say so."""
        rich = [self._candidate(s, 0, i * 100) for i, s in enumerate([80, 85, 90])]
        lonely = [self._candidate(95, 9, 5000)]
        normalize_scores(rich + lonely)
        assert lonely[0].confidence < rich[0].confidence

    def test_cross_window_scores_do_not_outrank_at_the_extremes(self):
        """Globally normalised scores are compressed into 0.15-0.85 so a
        lone, less trustworthy candidate cannot claim a perfect 1.0."""
        group = [self._candidate(100, 0, 0), self._candidate(0, 1, 500)]
        normalize_scores(group)
        assert max(c.normalized_score for c in group) <= 0.85
        assert min(c.normalized_score for c in group) >= 0.15

    def test_identical_scores_carry_no_ranking_information(self):
        group = [self._candidate(70, 0, i * 100) for i in range(4)]
        normalize_scores(group)
        assert all(c.normalized_score == 0.5 for c in group)
        assert all(c.confidence < 0.5 for c in group)

    def test_normalised_scores_stay_in_range(self):
        group = [self._candidate(s, 0, i * 100) for i, s in enumerate([0, 1, 99, 100])]
        normalize_scores(group)
        assert all(0.0 <= c.normalized_score <= 1.0 for c in group)


class TestDeduplication:
    def _candidate(self, start, end, score=0.5):
        c = Candidate(start, end, "h", "t", "story", 80, "r")
        c.normalized_score = score
        return c

    def test_overlapping_duplicates_collapse(self):
        kept = deduplicate([
            self._candidate(10, 50, 0.9),
            self._candidate(12, 52, 0.4),
        ])
        assert len(kept) == 1
        assert kept[0].normalized_score == 0.9

    def test_distinct_moments_both_survive(self):
        kept = deduplicate([self._candidate(10, 50), self._candidate(200, 240)])
        assert len(kept) == 2

    def test_small_overlap_is_not_a_duplicate(self):
        kept = deduplicate([self._candidate(10, 50), self._candidate(48, 90)])
        assert len(kept) == 2

    def test_highest_scoring_survives(self):
        kept = deduplicate([
            self._candidate(10, 50, 0.3),
            self._candidate(11, 51, 0.95),
            self._candidate(12, 52, 0.6),
        ])
        assert len(kept) == 1
        assert kept[0].normalized_score == 0.95

    def test_output_is_chronological(self):
        kept = deduplicate([
            self._candidate(300, 340), self._candidate(10, 50),
            self._candidate(150, 190),
        ])
        assert [c.start for c in kept] == sorted(c.start for c in kept)


class TestPrompt:
    def test_includes_timestamps_and_constraints(self):
        tr = make_transcript(120)
        window = window_transcript(tr, target_tokens=5000)[0]
        prompt = build_prompt(window, want=3, min_duration=25, max_duration=60)
        assert "[0.0]" in prompt
        assert "25" in prompt and "60" in prompt
        assert "story" in prompt

    def test_forbids_invented_timestamps(self):
        tr = make_transcript(120)
        window = window_transcript(tr, target_tokens=5000)[0]
        assert "Do not invent times" in build_prompt(window, 3, 25, 60)


class TestProviderSelection:
    def test_stub_is_never_auto_selected(self):
        try:
            provider = get_provider("auto")
        except LLMUnavailable:
            return          # nothing available here, which is correct
        assert provider.name != "stub"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider("gpt-9000")

    def test_stub_returns_queued_responses_in_order(self):
        stub = StubProvider(["first", "second"])
        assert stub.complete("p").text == "first"
        assert stub.complete("p").text == "second"
        assert stub.complete("p").text == "second"   # last repeats

    def test_stub_records_calls(self):
        stub = StubProvider(["[]"])
        stub.complete("the prompt", system="sys", json_mode=True)
        assert stub.calls[0]["json_mode"] is True
        assert stub.calls[0]["system"] == "sys"

    def test_ollama_reports_a_useful_reason_when_down(self):
        from app.services.ai.providers import OllamaProvider

        available, reason = OllamaProvider(host="http://127.0.0.1:1").is_available()
        assert not available
        assert "ollama serve" in reason


class TestContextBudget:
    """Ollama does not raise on context overflow — it drops the oldest tokens,
    which here is the transcript. The model then answers confidently about text
    it never received and the timestamp validator rejects nearly everything,
    which looks like a bad model rather than bad configuration."""

    def test_comfortable_fit_passes(self):
        from app.services.ai.discovery import check_context_budget

        assert check_context_budget("x" * 4000, max_output=768, num_ctx=8192) is None

    def test_overflow_is_reported(self):
        from app.services.ai.discovery import check_context_budget

        warning = check_context_budget("x" * 40000, max_output=2048, num_ctx=4096)
        assert warning is not None
        assert "LLM_NUM_CTX" in warning

    def test_margin_is_reserved(self):
        """Exactly filling the window is not safe: the token estimate is a
        rough chars/4, so a request that only just fits may not."""
        from app.services.ai.discovery import check_context_budget

        # ~1000 estimated tokens + 3000 output = 4000, just under 4096.
        assert check_context_budget("x" * 4000, max_output=3000, num_ctx=4096) is not None

    def test_default_settings_fit_the_default_context(self):
        """Guards the shipped configuration against itself."""
        from app.core.config import settings
        from app.services.ai.discovery import build_prompt, check_context_budget, window_transcript

        transcript = make_transcript(1200)
        window = window_transcript(transcript, target_tokens=settings.llm_window_tokens)[0]
        prompt = build_prompt(window, settings.candidates_per_window,
                              settings.clip_min_duration, settings.clip_max_duration)
        assert check_context_budget(prompt, settings.llm_max_output,
                                    settings.llm_num_ctx) is None

    def test_ollama_sends_num_ctx(self):
        from app.services.ai.providers import OllamaProvider

        provider = OllamaProvider(num_ctx=8192)
        assert provider.num_ctx == 8192


class TestReasoningModels:
    """qwen3 and deepseek-r1 emit a reasoning pass before answering. Ollama
    returns it in a separate field, so it never reaches the JSON parser — but
    it is still generated, still costs seconds, and still counts against
    num_predict, so a long think can truncate the answer that follows."""

    def test_thinking_is_off_by_default(self):
        from app.services.ai.providers import OllamaProvider

        assert OllamaProvider().think is False

    def test_thinking_can_be_enabled_for_comparison(self, monkeypatch):
        from app.services.ai.providers import OllamaProvider

        monkeypatch.setenv("LLM_THINK", "1")
        assert OllamaProvider().think is True

    def test_explicit_argument_wins_over_env(self, monkeypatch):
        from app.services.ai.providers import OllamaProvider

        monkeypatch.setenv("LLM_THINK", "1")
        assert OllamaProvider(think=False).think is False

    def test_think_flag_only_set_for_structured_calls(self):
        """Free-form generation — hook writing in a later phase — may genuinely
        benefit from reasoning, so the flag is scoped to json_mode."""
        import json as _json

        from app.services.ai import providers

        captured = {}

        def fake_post(url, payload, timeout, headers=None):
            captured.update(payload)
            return {"message": {"content": "[]"}}

        original = providers._post_json
        providers._post_json = fake_post
        try:
            providers.OllamaProvider().complete("p", json_mode=True)
            assert captured.get("think") is False
            captured.clear()
            providers.OllamaProvider().complete("p", json_mode=False)
            assert "think" not in captured
        finally:
            providers._post_json = original

    def test_num_ctx_and_keep_alive_are_sent(self):
        """keep_alive=0 unloads the model immediately. On 16 GB shared memory,
        a 5 GB model lingering for Ollama's default 5 minutes would collide
        with the render stage."""
        from app.services.ai import providers

        captured = {}

        def fake_post(url, payload, timeout, headers=None):
            captured.update(payload)
            return {"message": {"content": "[]"}}

        original = providers._post_json
        providers._post_json = fake_post
        try:
            providers.OllamaProvider(num_ctx=8192).complete("p", json_mode=True)
            assert captured["options"]["num_ctx"] == 8192
            assert captured["keep_alive"] == "0"
        finally:
            providers._post_json = original

    def test_reasoning_without_an_answer_gives_an_actionable_error(self):
        """The failure mode when a think overruns the output budget: Ollama
        returns thinking and an empty content. A bare 'empty message' would
        send you looking in the wrong place."""
        from app.services.ai import providers

        def fake_post(url, payload, timeout, headers=None):
            return {"message": {"content": "", "thinking": "Let me consider..."}}

        original = providers._post_json
        providers._post_json = fake_post
        try:
            with pytest.raises(providers.LLMError, match="LLM_MAX_OUTPUT"):
                providers.OllamaProvider().complete("p", json_mode=True)
        finally:
            providers._post_json = original


class TestRejectionSummary:
    """Thirty identical failures and one unlucky candidate need different
    responses. Quoting only the first rejection made them look identical."""

    def test_counts_are_grouped_by_kind(self):
        from app.services.ai.service import summarise_rejections

        summary = summarise_rejections([
            "window 0: timestamps 9000.0-9040.0 do not match the transcript",
            "window 1: timestamps 8000.0-8040.0 do not match the transcript",
            "window 2: No valid JSON found in model output: sorry",
        ])
        assert "2 x invented timestamps" in summary
        assert "1 x unparseable output" in summary

    def test_total_is_reported(self):
        from app.services.ai.service import summarise_rejections

        assert "3 candidate(s) rejected" in summarise_rejections(["a", "b", "c"])

    def test_an_example_is_included(self):
        from app.services.ai.service import summarise_rejections

        assert "the first one" in summarise_rejections(["the first one", "another"])

    def test_empty_input(self):
        from app.services.ai.service import summarise_rejections

        assert "nothing parseable" in summarise_rejections([])
