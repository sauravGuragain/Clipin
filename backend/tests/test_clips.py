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
from app.services.clips.select import (
    ClipWindow,
    anchors_for,
    is_weak_opener,
    select_clips,
    snap_to_segments,
)
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


class TestWeakOpeners:
    @pytest.mark.parametrize("token", ["so", "And", "but,", "Um", "yeah"])
    def test_detected(self, token):
        assert is_weak_opener(Word(token, 0, 1))

    @pytest.mark.parametrize("token", ["Revenue", "I", "Nobody", "Three"])
    def test_not_flagged(self, token):
        assert not is_weak_opener(Word(token, 0, 1))


class TestSnapping:
    def test_starts_on_a_word_boundary(self):
        tr = build_transcript()
        window = snap_to_segments(tr, 100.0, 25, 60)
        assert window is not None
        assert any(abs(window.start - w.start) < 1e-6 for w in tr.words)

    def test_ends_on_a_word_boundary(self):
        tr = build_transcript()
        window = snap_to_segments(tr, 100.0, 25, 60)
        assert any(abs(window.end - w.end) < 1e-6 for w in tr.words)

    def test_respects_maximum(self):
        tr = build_transcript()
        window = snap_to_segments(tr, 100.0, 10, 30)
        assert window.duration <= 30

    def test_prefers_strong_opener(self):
        tr = build_transcript(sentences=[
            "So anyway that happened.",
            "Revenue tripled in a single quarter.",
        ])
        window = snap_to_segments(tr, 0.0, 5, 30)
        assert not window.opener_penalty

    def test_empty_transcript_returns_none(self):
        empty = Transcript([], "en", "s", "m", 0)
        assert snap_to_segments(empty, 0, 10, 30) is None


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
