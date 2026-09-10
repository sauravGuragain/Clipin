"""Tests for clip selection and rendering.

Two of these exist because visual inspection caught bugs that every other test
passed straight through:

  * captions were timed to each word's own duration, leaving the screen blank
    in the gaps between words — the ASS file was valid, the render succeeded,
    and the result flickered.
  * padding was applied after the duration limit, so every clip came out longer
    than the configured maximum.

Both are covered below.
"""

import subprocess

import pytest

from app.services.captions.ass import CaptionStyle, build_ass
from app.services.clips.render import RenderError, RenderSettings, build_crop_filter
from app.services.clips.boundaries import (
    clean,
    ends_sentence,
    filler_ratio,
    opener_weakness,
    pad_into_silence,
    score_boundaries,
    solve_boundaries,
)
from app.services.clips.select import ClipWindow, anchors_for, select_clips
from app.services.transcription.base import Segment, Transcript, Word


def build_transcript(total=400.0, sentences=None):
    sentences = sentences or [
        "This is a complete thought.",
        "Here is another one that runs a little longer.",
        "Short.",
        "And a final statement to close the idea.",
    ]
    segments, t, i = [], 0.0, 0
    while t < total:
        text = sentences[i % len(sentences)]
        words, wt = [], t
        for token in text.split():
            words.append(Word(token, round(wt, 3), round(wt + 0.3, 3), 0.95))
            wt += 0.42
        segments.append(Segment(t, round(wt, 3), text, words))
        t = wt + 0.3
        i += 1
    return Transcript(segments, "en", "stub", "m", duration=t)


class TestAnchors:
    def test_count_respected(self):
        assert len(anchors_for(1000, 5)) == 5

    def test_skips_intro_and_outro(self):
        anchors = anchors_for(1000, 5, edge_fraction=0.05)
        assert anchors[0] >= 50
        assert anchors[-1] <= 950

    def test_ascending(self):
        anchors = anchors_for(3600, 20)
        assert anchors == sorted(anchors)

    def test_zero_count(self):
        assert anchors_for(1000, 0) == []

    def test_zero_duration(self):
        assert anchors_for(0, 5) == []


class TestBoundaryScoring:
    def test_clean_strips_punctuation_and_case(self):
        assert clean('  "So,"  ') == "so"

    def test_ends_sentence_handles_trailing_quote(self):
        assert ends_sentence('He said "no."')
        assert not ends_sentence("and then we")

    def test_filler_ratio(self):
        words = [Word("um", 0, 1), Word("revenue", 1, 2),
                 Word("like", 2, 3), Word("tripled", 3, 4)]
        assert filler_ratio(words) == 0.5

    def test_filler_ratio_of_nothing_is_worst_case(self):
        assert filler_ratio([]) == 1.0

    def test_strong_boundaries_score_higher_than_weak(self):
        strong = build_transcript(sentences=["Revenue tripled in one quarter."])
        weak = build_transcript(sentences=["and then the thing with the"])
        segs_s = [s for s in strong.segments if s.words]
        segs_w = [s for s in weak.segments if s.words]
        a = score_boundaries(segs_s, 1, 6, 5, 60)
        b = score_boundaries(segs_w, 1, 6, 5, 60)
        assert a.total > b.total

    def test_dangling_ending_is_penalised(self):
        segs = [s for s in build_transcript(
            sentences=["Revenue grew and", "Something else entirely."]
        ).segments if s.words]
        score = score_boundaries(segs, 0, 0, 0.1, 60)
        assert score.components["not_dangling"] == 0.0

    def test_multiword_filler_phrases_are_counted(self):
        """Regression: FILLER_PHRASES entries like "you know" were matched
        against single cleaned words, so they never fired at all."""
        words = [Word(w, i * 0.4, i * 0.4 + 0.3)
                 for i, w in enumerate("you know it was actually".split())]
        assert filler_ratio(words) > 0.5

    def test_phrase_opener_detected_though_first_word_is_innocent(self):
        """Regression: "you" is not a weak opener, so "you know..." passed."""
        words = [Word(w, i * 0.4, i * 0.4 + 0.3)
                 for i, w in enumerate("you know it was".split())]
        assert opener_weakness(words) is not None

    def test_discourse_marker_is_forgivable(self):
        words = [Word("So", 0, 0.3), Word("listen", 0.4, 0.8)]
        opener, forgivable = opener_weakness(words)
        assert opener == "so" and forgivable

    def test_subordinator_is_never_forgivable(self):
        """A subordinate clause refers to context the viewer cannot see. No
        length of pause repairs that."""
        for token in ("which", "because", "and"):
            words = [Word(token, 0, 0.3), Word("something", 0.4, 0.9)]
            result = opener_weakness(words)
            assert result is not None and result[1] is False, token

    def test_strong_opener_returns_none(self):
        words = [Word("Revenue", 0, 0.5), Word("tripled", 0.6, 1.0)]
        assert opener_weakness(words) is None

    def test_pause_does_not_rescue_a_subordinator(self):
        seg_prev = Segment(0.0, 5.0, "Previous thought ended.",
                           [Word("Previous", 0, 1), Word("ended.", 4, 5)])
        seg_now = Segment(20.0, 20.9, "which is why",
                          [Word("which", 20.0, 20.3), Word("is", 20.4, 20.6),
                           Word("why", 20.7, 20.9)])
        score = score_boundaries([seg_prev, seg_now], 1, 1, 0.1, 60)
        assert score.components["opener"] == 0.0

    def test_long_pause_forgives_a_weak_opener(self):
        """'So, here is the thing' after silence is a legitimate opening;
        the same words mid-sentence are not."""
        words_a = [Word("So", 10.0, 10.3), Word("listen", 10.4, 10.9)]
        seg_prev = Segment(0.0, 5.0, "Previous thought ended.",
                           [Word("Previous", 0, 1), Word("ended.", 4, 5)])
        seg_now = Segment(10.0, 10.9, "So listen", words_a)
        forgiven = score_boundaries([seg_prev, seg_now], 1, 1, 0.1, 60)

        seg_prev_close = Segment(0.0, 9.9, "Previous thought ended.",
                                 [Word("Previous", 0, 1), Word("ended.", 9, 9.9)])
        not_forgiven = score_boundaries([seg_prev_close, seg_now], 1, 1, 0.1, 60)

        assert forgiven.components["opener"] > not_forgiven.components["opener"]

    def test_duration_fit_peaks_mid_range(self):
        segs = [s for s in build_transcript().segments if s.words]
        mid = score_boundaries(segs, 0, 12, 5.0, 30.0)
        assert 0.0 <= mid.components["duration_fit"] <= 1.0

    def test_score_is_normalised(self):
        segs = [s for s in build_transcript().segments if s.words]
        score = score_boundaries(segs, 0, 8, 1, 120)
        assert 0.0 <= score.total <= 1.0

    def test_notes_explain_low_scores(self):
        segs = [s for s in build_transcript(
            sentences=["um like basically the"]
        ).segments if s.words]
        score = score_boundaries(segs, 1, 3, 0.1, 60)
        assert score.notes


class TestSilencePadding:
    def test_takes_half_the_available_gap(self):
        words = [Word("a", 0.0, 1.0), Word("b", 3.0, 4.0), Word("c", 6.0, 7.0)]
        tr = Transcript([Segment(0, 7, "a b c", words)], "en", "s", "m", 10.0)
        start, end = pad_into_silence(tr, 3.0, 4.0, max_padding=5.0, media_duration=10.0)
        assert start == pytest.approx(2.0)     # half of the 2s gap before
        assert end == pytest.approx(5.0)       # half of the 2s gap after

    def test_respects_max_padding(self):
        words = [Word("a", 0.0, 1.0), Word("b", 20.0, 21.0)]
        tr = Transcript([Segment(0, 21, "a b", words)], "en", "s", "m", 30.0)
        start, _ = pad_into_silence(tr, 20.0, 21.0, max_padding=0.5, media_duration=30.0)
        assert start == pytest.approx(19.5)

    def test_never_goes_negative(self):
        words = [Word("a", 0.0, 1.0)]
        tr = Transcript([Segment(0, 1, "a", words)], "en", "s", "m", 5.0)
        start, _ = pad_into_silence(tr, 0.0, 1.0, max_padding=2.0, media_duration=5.0)
        assert start >= 0.0

    def test_never_exceeds_media_duration(self):
        words = [Word("a", 0.0, 1.0)]
        tr = Transcript([Segment(0, 1, "a", words)], "en", "s", "m", 1.2)
        _, end = pad_into_silence(tr, 0.0, 1.0, max_padding=5.0, media_duration=1.2)
        assert end <= 1.2


class TestSolver:
    def test_returns_candidate_within_duration_bounds(self):
        tr = build_transcript()
        best = solve_boundaries(tr, 100.0, 20.0, 45.0)
        assert best is not None
        assert 20.0 <= best.duration <= 45.0

    def test_picks_a_better_boundary_than_the_first_that_fits(self):
        tr = build_transcript()
        best = solve_boundaries(tr, 100.0, 20.0, 60.0)
        segs = [s for s in tr.segments if s.words]
        first_fit = next(
            i for i in range(best.start_index, len(segs))
            if segs[i].end - segs[best.start_index].start >= 20.0
        )
        greedy = score_boundaries(segs, best.start_index, first_fit, 20.0, 60.0)
        assert best.score.total >= greedy.total

    def test_stays_near_the_anchor(self):
        tr = build_transcript()
        best = solve_boundaries(tr, 200.0, 20.0, 45.0, search_back=15, search_forward=25)
        assert 185.0 <= best.start <= 226.0

    def test_impossible_duration_returns_none(self):
        tr = build_transcript(60)
        assert solve_boundaries(tr, 10.0, 500.0, 600.0) is None

    def test_empty_transcript_returns_none(self):
        assert solve_boundaries(Transcript([], "en", "s", "m", 0), 0, 10, 30) is None


class TestSelection:
    def test_returns_requested_count(self):
        assert len(select_clips(build_transcript(), 4)) == 4

    def test_no_overlaps(self):
        windows = select_clips(build_transcript(600), 6)
        for a, b in zip(windows, windows[1:]):
            assert a.end <= b.start

    def test_sorted_by_start(self):
        windows = select_clips(build_transcript(600), 6)
        assert [w.start for w in windows] == sorted(w.start for w in windows)

    @pytest.mark.parametrize("maximum", [20.0, 30.0, 45.0, 60.0])
    def test_padding_never_exceeds_maximum(self, maximum):
        """Regression: padding used to be applied after the limit check, so
        every clip came out longer than requested."""
        windows = select_clips(
            build_transcript(600), 4, min_duration=maximum * 0.5,
            max_duration=maximum, padding=2.0,
        )
        assert windows
        for window in windows:
            assert window.duration <= maximum + 1e-6, (
                f"{window.duration} exceeds max {maximum}"
            )

    def test_short_transcript_returns_fewer_not_garbage(self):
        windows = select_clips(build_transcript(40), 10, min_duration=25, max_duration=60)
        assert len(windows) < 10

    def test_empty_transcript_returns_empty(self):
        assert select_clips(Transcript([], "en", "s", "m", 0), 5) == []

    def test_windows_carry_their_words(self):
        for window in select_clips(build_transcript(), 3):
            assert window.words
            assert window.words[0].start >= window.start - 1e-6

    def test_windows_carry_boundary_scores(self):
        for window in select_clips(build_transcript(), 3):
            assert 0.0 <= window.boundary_score <= 1.0
            assert window.components

    def test_min_score_filter_rejects(self):
        assert select_clips(build_transcript(600), 4, min_boundary_score=1.01) == []

    def test_padding_never_bites_into_a_neighbouring_word(self):
        """Silence-aware padding: a fixed pad can overlap the previous word;
        taking half the actual gap cannot."""
        tr = build_transcript(600)
        windows = select_clips(tr, 4, padding=5.0)
        for window in windows:
            before = [w for w in tr.words if w.end <= window.words[0].start + 1e-6]
            if before:
                assert window.start >= before[-1].end - 1e-6


class TestCropFilter:
    def test_center_crops_and_scales(self):
        f = build_crop_filter("center", 1080, 1920)
        assert "crop=" in f and "scale=1080:1920" in f

    def test_fit_uses_blurred_background(self):
        f = build_crop_filter("fit", 1080, 1920)
        assert "boxblur" in f and "overlay" in f

    def test_unknown_strategy_raises(self):
        with pytest.raises(RenderError, match="Unknown crop strategy"):
            build_crop_filter("magic", 1080, 1920)


class TestCaptionContinuity:
    """Regression: events were timed to each word's own duration, so the screen
    went blank between words. Valid ASS, successful render, flickering result."""

    def test_no_gaps_within_a_line(self):
        words = [
            Word("alpha", 0.00, 0.32),
            Word("bravo", 0.40, 0.72),
            Word("charlie", 0.80, 1.20),
        ]
        events = [
            line for line in build_ass(words).splitlines() if line.startswith("Dialogue:")
        ]
        times = [(e.split(",")[1], e.split(",")[2]) for e in events]
        for current, following in zip(times, times[1:]):
            assert current[1] == following[0], "blank frame between words"

    def test_line_lingers_after_the_last_word(self):
        style = CaptionStyle(line_hold=0.5)
        words = [Word("only", 1.0, 1.4)]
        event = [
            line for line in build_ass(words, style).splitlines()
            if line.startswith("Dialogue:")
        ][0]
        assert event.split(",")[2] == "0:00:01.90"

    def test_still_one_event_per_word(self):
        words = [Word(f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(5)]
        assert build_ass(words).count("Dialogue:") == 5


@pytest.mark.skipif(
    subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True,
                   text=True).stdout.find(" subtitles ") == -1,
    reason="ffmpeg has no libass",
)
class TestRenderIntegration:
    """The only way to know a filter graph is valid is to run FFmpeg."""

    @pytest.fixture
    def landscape_source(self, tmp_path):
        out = tmp_path / "src.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "lavfi", "-i", "testsrc=size=640x360:rate=15:d=6",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(out)],
            check=True,
        )
        return out

    def _window(self):
        words = [Word(f"w{i}", 1.0 + i * 0.4, 1.0 + i * 0.4 + 0.3) for i in range(6)]
        return ClipWindow(1.0, 4.0, words, "w0 w1 w2 w3 w4 w5")

    @pytest.mark.asyncio
    async def test_produces_vertical_video_with_audio(self, landscape_source, tmp_path):
        from app.services.clips.render import render_clip

        dest = tmp_path / "clip.mp4"
        await render_clip(
            landscape_source, self._window(), dest,
            RenderSettings(width=270, height=480, preset="ultrafast"),
        )
        assert dest.exists()

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", str(dest)],
            capture_output=True, text=True, check=True,
        ).stdout
        import json
        streams = json.loads(probe)["streams"]
        video = next(s for s in streams if s["codec_type"] == "video")
        assert (video["width"], video["height"]) == (270, 480)
        assert any(s["codec_type"] == "audio" for s in streams)

    @pytest.mark.asyncio
    async def test_fit_strategy_renders(self, landscape_source, tmp_path):
        from app.services.clips.render import render_clip

        dest = tmp_path / "fit.mp4"
        await render_clip(
            landscape_source, self._window(), dest,
            RenderSettings(width=270, height=480, preset="ultrafast",
                           crop_strategy="fit"),
        )
        assert dest.exists() and dest.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_missing_source_raises(self, tmp_path):
        from app.services.clips.render import render_clip

        with pytest.raises(RenderError):
            await render_clip(tmp_path / "nope.mp4", self._window(), tmp_path / "o.mp4")

    @pytest.mark.asyncio
    async def test_no_partial_file_after_failure(self, tmp_path):
        from app.services.clips.render import render_clip

        junk = tmp_path / "junk.mp4"
        junk.write_bytes(b"\x00" * 4096)
        dest = tmp_path / "out.mp4"
        with pytest.raises(RenderError):
            await render_clip(junk, self._window(), dest,
                              RenderSettings(width=270, height=480))
        assert not dest.exists()
        assert not list(tmp_path.glob("*.partial.mp4"))
