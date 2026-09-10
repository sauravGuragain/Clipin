"""Regression tests for two bugs found in manual verification of Phase 5.

Both had passing test suites around them, which is the point of this file.

**Caption overlap.** Phase 4 added `line_hold` so a finished line lingers rather
than vanishing mid-breath. It was not clamped to the following line, so the last
event of each line ran into the first event of the next. libass resolves
simultaneous events by stacking them, so the viewer saw two captions at once.
The existing continuity test only checked *within* a single line and never saw
it. Measured before the fix: 33 of 189 sampled frames showed two lines.

**Stale UI.** `FileResponse` sends ETag and Last-Modified but no Cache-Control.
Browsers apply heuristic freshness to such responses and will serve a cached
page without revalidating, which silently hid an entire phase's UI changes.

The render-level tests here need Pillow. They skip without it rather than
failing, but the ASS-level invariants below run always and are what actually
catch a regression.
"""

import subprocess

import pytest

from app.services.captions.ass import CaptionStyle, build_ass, group_words_into_lines
from app.services.transcription.base import Word

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

HAVE_LIBASS = subprocess.run(
    ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True
).stdout.find(" subtitles ") != -1


def evenly_spaced(count: int, step: float = 0.40, hold: float = 0.32) -> list[Word]:
    words, t = [], 0.0
    for i in range(count):
        words.append(Word(f"word{i}", round(t, 3), round(t + hold, 3), 0.95))
        t += step
    return words


def event_times(doc: str) -> list[tuple[float, float]]:
    def secs(stamp: str) -> float:
        hours, minutes, seconds = stamp.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    out = []
    for line in doc.splitlines():
        if line.startswith("Dialogue:"):
            parts = line.split(",")
            out.append((secs(parts[1]), secs(parts[2])))
    return out


class TestEventStreamIsSequential:
    """The invariant that was violated: events must never overlap in time.

    Two simultaneous events at the same alignment are exactly what produces
    stacked captions on screen.
    """

    @pytest.mark.parametrize("count", [8, 20, 60, 150])
    def test_no_overlapping_events(self, count):
        times = event_times(build_ass(evenly_spaced(count)))
        overlaps = [
            (a, b) for a, b in zip(times, times[1:]) if b[0] < a[1] - 1e-9
        ]
        assert not overlaps, f"{len(overlaps)} overlapping event pairs"

    @pytest.mark.parametrize("step,hold", [(0.40, 0.32), (0.18, 0.15), (1.2, 1.0)])
    def test_no_overlaps_at_varied_speaking_rates(self, step, hold):
        times = event_times(build_ass(evenly_spaced(40, step, hold)))
        assert all(b[0] >= a[1] - 1e-9 for a, b in zip(times, times[1:]))

    def test_no_overlap_when_lines_abut_tightly(self):
        """The exact trigger: the gap between lines is smaller than line_hold."""
        style = CaptionStyle(line_hold=1.5, max_words_per_line=3)
        times = event_times(build_ass(evenly_spaced(30, step=0.30), style))
        assert all(b[0] >= a[1] - 1e-9 for a, b in zip(times, times[1:]))

    def test_hold_is_clamped_not_dropped(self):
        """A long pause between lines should still let the line linger."""
        words = [
            Word("alpha", 0.0, 0.4), Word("bravo", 0.5, 0.9),
            Word("charlie", 8.0, 8.4),          # long gap -> new line
        ]
        style = CaptionStyle(max_words_per_line=2, line_hold=0.35)
        times = event_times(build_ass(words, style))
        # First line's final event holds past its last word.
        assert times[1][1] > 0.9

    def test_captions_still_continuous_within_a_line(self):
        """The fix must not reintroduce the flicker it replaced."""
        style = CaptionStyle(max_words_per_line=4)
        times = event_times(build_ass(evenly_spaced(4), style))
        for current, following in zip(times, times[1:]):
            assert following[0] == pytest.approx(current[1], abs=1e-9)

    def test_zero_length_words_do_not_create_overlap(self):
        """Whisper occasionally emits a word whose end equals the next start.
        Padding such an event out would push it into its successor."""
        words = [
            Word("a", 0.0, 0.3), Word("b", 0.3, 0.3), Word("c", 0.3, 0.7),
            Word("d", 0.8, 1.2),
        ]
        times = event_times(build_ass(words))
        assert all(b[0] >= a[1] - 1e-9 for a, b in zip(times, times[1:]))
        assert all(end > start for start, end in times)

    def test_every_line_is_represented(self):
        """Skipping degenerate events must not silently drop a whole line."""
        words = evenly_spaced(30)
        lines = group_words_into_lines(words, CaptionStyle())
        doc = build_ass(words)
        for line in lines:
            assert line[0].text in doc


@pytest.mark.skipif(not HAVE_LIBASS, reason="ffmpeg has no libass")
@pytest.mark.skipif(not HAVE_PIL, reason="Pillow not installed")
class TestRenderedFramesShowOneCaption:
    """Render-level check, against the pixels the viewer sees.

    A first attempt counted horizontal bands of text, on the assumption that
    libass would stack colliding events. It does not — it draws them
    superimposed at the same position, so both lines occupy identical rows and
    the band count stays at one. That test passed against the bug it was
    written for, which is worse than having no test.

    This counts **highlighted words** instead. Exactly one word carries the
    highlight colour at any moment, so two separated clusters of it mean two
    caption events are live at once. Measured: 18 offending frames before the
    fix, 0 after.
    """

    @staticmethod
    def _highlight_clusters(png_path, column_gap: int = 12) -> int:
        image = Image.open(png_path).convert("RGB")
        width, height = image.size
        pixels = image.load()

        columns = set()
        for y in range(0, height, 2):
            for x in range(width):
                r, g, b = pixels[x, y]
                if r > 150 and g > 110 and b < 110 and (r - b) > 70:
                    columns.add(x)

        if not columns:
            return 0
        ordered = sorted(columns)
        clusters = 1
        for left, right in zip(ordered, ordered[1:]):
            if right - left > column_gap:
                clusters += 1
        return clusters

    def _render(self, tmp_path, doc: str):
        ass = tmp_path / "c.ass"
        ass.write_text(doc)
        video = tmp_path / "v.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "color=c=black:s=480x854:d=12:r=10",
             "-vf", f"subtitles={ass}",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             str(video)],
            check=True,
        )
        return video

    def test_only_one_word_is_highlighted_at_any_moment(self, tmp_path):
        video = self._render(tmp_path, build_ass(evenly_spaced(28), CaptionStyle()))

        frames_with_caption, offending = 0, []
        for i in range(2, 110):
            frame = tmp_path / f"f{i}.png"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{i * 0.1:.1f}",
                 "-i", str(video), "-frames:v", "1", str(frame)],
                check=True,
            )
            clusters = self._highlight_clusters(frame)
            if clusters >= 1:
                frames_with_caption += 1
            if clusters > 1:
                offending.append(round(i * 0.1, 1))

        assert frames_with_caption > 50, (
            "captions barely rendered; this test is not measuring anything"
        )
        assert not offending, (
            f"two captions on screen at once at t={offending[:8]}"
        )


class TestUiIsNotHeuristicallyCached:
    """A response with only ETag/Last-Modified is heuristically cacheable, so a
    browser can serve a stale page indefinitely without revalidating."""

    def test_index_sets_cache_control(self):
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as client:
            response = client.get("/")
            cache_control = response.headers.get("cache-control", "")
            assert "no-cache" in cache_control, (
                "index.html has no Cache-Control; browsers will serve it stale"
            )

    def test_index_still_revalidates_cheaply(self):
        """no-cache means revalidate, not don't-store. The ETag must survive so
        an unchanged page still returns 304."""
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as client:
            response = client.get("/")
            assert response.headers.get("etag")


class TestClipApiExposesBoundaryFields:
    """The UI could not show a boundary score it was never sent. Asserting the
    contract here means a serialiser change breaks a test rather than a page."""

    def test_serialiser_includes_boundary_fields(self):
        from app.api.routes import _serialise_clip
        from app.models.db import Clip

        clip = Clip(
            id="abc123456789", project_id="p1", index=1,
            start=1.0, end=41.0, duration=40.0,
            boundary_score=0.83, boundary_notes="does not end on a sentence",
        )
        payload = _serialise_clip(clip)
        assert payload["boundary_score"] == 0.83
        assert payload["boundary_notes"] == "does not end on a sentence"

    def test_ui_template_reads_the_fields(self):
        """Guards the other half: the API can send them and the page still not
        render them, which is precisely what happened."""
        from pathlib import Path

        import app.main as main

        html = (Path(main.__file__).parent / "static" / "index.html").read_text()
        assert "boundary_score" in html
        assert "boundary_notes" in html
