"""Tests for ASS caption generation.

Written in Phase 3 because verifying word timestamps requires rendering them.
The escaping and timestamp-format tests matter most: a malformed ASS file makes
libass drop events silently rather than erroring, so a bug here shows up as
missing captions with no diagnostic.
"""

import subprocess

import pytest

from app.services.captions.ass import (
    CaptionStyle,
    build_ass,
    escape,
    group_words_into_lines,
    timestamp,
)
from app.services.transcription.base import Word


def words_from(text: str, start: float = 0.0, step: float = 0.35) -> list[Word]:
    out, t = [], start
    for token in text.split():
        out.append(Word(token, round(t, 3), round(t + step * 0.8, 3), 0.95))
        t += step
    return out


class TestTimestampFormat:
    def test_zero(self):
        assert timestamp(0) == "0:00:00.00"

    def test_centiseconds(self):
        assert timestamp(3.456) == "0:00:03.46"

    def test_minutes_and_hours(self):
        assert timestamp(3725.5) == "1:02:05.50"

    def test_negative_clamps_to_zero(self):
        assert timestamp(-5) == "0:00:00.00"


class TestEscaping:
    def test_braces_escaped(self):
        assert escape("a {b} c") == "a \\{b\\} c"

    def test_backslash_escaped(self):
        assert "\\\\" in escape("back\\slash")

    def test_newline_becomes_space(self):
        assert "\n" not in escape("two\nlines")


class TestLineGrouping:
    def test_respects_character_limit(self):
        style = CaptionStyle(max_chars_per_line=20, max_words_per_line=99,
                             max_line_seconds=999)
        for line in group_words_into_lines(words_from("a" * 5 + " " + "b" * 5 + " " + "c" * 5 + " " + "d" * 5), style):
            assert sum(len(w.text) for w in line) + len(line) - 1 <= 20

    def test_respects_word_limit(self):
        style = CaptionStyle(max_words_per_line=3, max_chars_per_line=999,
                             max_line_seconds=999)
        for line in group_words_into_lines(words_from("one two three four five six seven"), style):
            assert len(line) <= 3

    def test_long_pause_breaks_the_line(self):
        style = CaptionStyle(max_line_seconds=1.0, max_chars_per_line=999,
                             max_words_per_line=99)
        words = [Word("a", 0.0, 0.2), Word("b", 0.3, 0.5), Word("c", 5.0, 5.2)]
        lines = group_words_into_lines(words, style)
        assert len(lines) > 1

    def test_no_words_dropped(self):
        words = words_from("the quick brown fox jumps over the lazy dog again and again")
        lines = group_words_into_lines(words, CaptionStyle())
        assert sum(len(line) for line in lines) == len(words)

    def test_empty_input(self):
        assert group_words_into_lines([], CaptionStyle()) == []

    def test_blank_words_skipped(self):
        words = [Word("  ", 0.0, 0.1), Word("real", 0.2, 0.5)]
        lines = group_words_into_lines(words, CaptionStyle())
        assert sum(len(line) for line in lines) == 1


class TestAssDocument:
    def test_one_event_per_word(self):
        words = words_from("one two three four five")
        assert build_ass(words).count("Dialogue:") == 5

    def test_active_word_is_highlighted(self):
        doc = build_ass(words_from("alpha beta"))
        first = [l for l in doc.splitlines() if l.startswith("Dialogue:")][0]
        assert "alpha" in first and CaptionStyle().highlight in first

    def test_resolution_is_written(self):
        doc = build_ass(words_from("x"), width=1080, height=1920)
        assert "PlayResX: 1080" in doc and "PlayResY: 1920" in doc

    def test_time_offset_shifts_events(self):
        words = words_from("hello world", start=100.0)
        doc = build_ass(words, time_offset=100.0)
        assert "0:00:00.00" in doc
        assert "1:40:" not in doc

    def test_words_entirely_before_offset_are_dropped(self):
        words = [Word("gone", 1.0, 2.0), Word("kept", 11.0, 12.0)]
        doc = build_ass(words, time_offset=10.0)
        assert "gone" not in doc.split("[Events]")[1]

    def test_empty_words_produce_valid_header_only(self):
        doc = build_ass([])
        assert "[Events]" in doc
        assert doc.count("Dialogue:") == 0


@pytest.mark.skipif(
    subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True,
                   text=True).stdout.find(" subtitles ") == -1,
    reason="ffmpeg has no libass",
)
class TestLibassAcceptsOutput:
    """A malformed ASS file makes libass skip events silently. The only real
    check is asking FFmpeg to render it."""

    def test_renders_without_error(self, tmp_path):
        ass = tmp_path / "c.ass"
        ass.write_text(build_ass(words_from("render this caption now"), width=640, height=360))
        out = tmp_path / "f.png"
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "color=c=black:s=640x360:d=3:r=15",
             "-vf", f"subtitles={ass}", "-frames:v", "1", str(out)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert out.exists() and out.stat().st_size > 0

    def test_text_with_braces_still_renders(self, tmp_path):
        words = [Word("{weird}", 0.2, 1.0), Word("back\\slash", 1.1, 2.0)]
        ass = tmp_path / "c.ass"
        ass.write_text(build_ass(words, width=640, height=360))
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "color=c=black:s=640x360:d=3:r=15",
             "-vf", f"subtitles={ass}", "-frames:v", "1", str(tmp_path / 'g.png')],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
