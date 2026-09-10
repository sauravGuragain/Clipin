"""Tests for transcription.

The model call itself cannot be tested without Apple Silicon and a 1.6 GB
download, so it is verified by hand on the target machine (see
scripts/verify_word_timestamps.py). Everything around it — the data model,
chunk planning, seam stitching, caching — is tested here, because that is where
a bug would corrupt timings silently rather than crashing.
"""

import json

import pytest

from app.services.transcription.backends import StubBackend, _from_whisper_dict, get_backend
from app.services.transcription.base import Segment, Transcript, Word
from app.services.transcription.service import cache_key, load_cached, save_transcript
from app.services.transcription.stitch import merge_chunks, plan_chunks


def make_transcript(words_per_segment=3, n_segments=2, start=0.0, step=0.4):
    segments = []
    t = start
    idx = 0
    for _ in range(n_segments):
        words = []
        for _ in range(words_per_segment):
            words.append(Word(f"w{idx}", round(t, 3), round(t + step * 0.8, 3), 0.9))
            idx += 1
            t += step
        segments.append(
            Segment(words[0].start, words[-1].end, " ".join(w.text for w in words), words)
        )
    return Transcript(segments, "en", "stub", "stub", duration=t)


class TestTranscriptModel:
    def test_words_flattens_across_segments(self):
        t = make_transcript(3, 2)
        assert len(t.words) == 6
        assert t.word_count == 6

    def test_roundtrip_preserves_word_timings(self):
        t = make_transcript(4, 3)
        restored = Transcript.from_dict(json.loads(json.dumps(t.to_dict())))
        assert [(w.text, w.start, w.end) for w in restored.words] == [
            (w.text, w.start, w.end) for w in t.words
        ]

    def test_words_between_uses_midpoint(self):
        words = [Word("a", 0.0, 1.0), Word("b", 1.0, 2.0), Word("c", 2.0, 3.0)]
        t = Transcript([Segment(0, 3, "a b c", words)], "en", "s", "m", 3.0)
        # "b" has midpoint 1.5, inside [1.2, 2.5); "a" (0.5) and "c" (2.5) are out.
        assert [w.text for w in t.words_between(1.2, 2.5)] == ["b"]

    def test_clean_transcript_has_no_issues(self):
        assert make_transcript(5, 4).validate() == []

    def test_detects_reversed_word(self):
        t = Transcript(
            [Segment(0, 2, "x", [Word("x", 1.5, 0.5)])], "en", "s", "m", 2.0
        )
        assert any(i.kind == "word_reversed" for i in t.validate())

    def test_detects_overlapping_words(self):
        words = [Word("a", 0.0, 1.0), Word("b", 0.4, 1.4)]
        t = Transcript([Segment(0, 1.4, "a b", words)], "en", "s", "m", 2.0)
        assert any(i.kind == "word_overlap" for i in t.validate())

    def test_detects_implausibly_long_word(self):
        t = Transcript(
            [Segment(0, 9, "x", [Word("x", 0.0, 8.5)])], "en", "s", "m", 10.0
        )
        assert any(i.kind == "word_too_long" for i in t.validate())

    def test_detects_word_past_media_end(self):
        t = Transcript(
            [Segment(0, 2, "x", [Word("x", 0.0, 1.0)])], "en", "s", "m", duration=0.2
        )
        assert any(i.kind == "beyond_media" for i in t.validate())

    def test_has_word_timestamps_false_without_words(self):
        t = Transcript([Segment(0, 2, "hello", [])], "en", "s", "m", 2.0)
        assert not t.has_word_timestamps


class TestChunkPlanning:
    def test_short_audio_is_one_chunk(self):
        chunks = plan_chunks(120, chunk_seconds=300)
        assert len(chunks) == 1
        assert (chunks[0].commit_start, chunks[0].commit_end) == (0.0, 120)

    def test_zero_chunk_seconds_disables_chunking(self):
        assert len(plan_chunks(5400, chunk_seconds=0)) == 1

    def test_long_audio_splits(self):
        chunks = plan_chunks(5400, chunk_seconds=300, overlap=2)
        assert len(chunks) == 18

    def test_commit_windows_tile_without_gap_or_overlap(self):
        chunks = plan_chunks(1000, chunk_seconds=300, overlap=5)
        assert chunks[0].commit_start == 0.0
        for a, b in zip(chunks, chunks[1:]):
            assert a.commit_end == b.commit_start
        assert chunks[-1].commit_end == 1000

    def test_read_windows_include_overlap_but_stay_in_bounds(self):
        chunks = plan_chunks(1000, chunk_seconds=300, overlap=5)
        assert chunks[0].read_start == 0.0
        assert chunks[-1].read_end == 1000
        for c in chunks[1:]:
            assert c.read_start == c.commit_start - 5

    def test_overlap_cannot_exceed_half_the_chunk(self):
        chunks = plan_chunks(1000, chunk_seconds=100, overlap=500)
        for c in chunks[1:]:
            assert c.commit_start - c.read_start <= 50

    def test_zero_duration_yields_nothing(self):
        assert plan_chunks(0, 300) == []


class TestStitching:
    """The seam is where a chunked transcript silently corrupts itself:
    duplicated words at boundaries, or words dropped entirely."""

    def _chunked(self, total=30.0, chunk_seconds=10.0, overlap=2.0, step=0.5):
        """Simulate transcribing overlapping chunks of one continuous stream."""
        chunks = plan_chunks(total, chunk_seconds, overlap)
        results = []
        for chunk in chunks:
            words = []
            t = chunk.read_start
            while t < chunk.read_end - 1e-9:
                words.append(Word(f"w{round(t, 2)}", round(t, 3), round(t + step * 0.8, 3), 0.9))
                t = round(t + step, 3)
            seg = Segment(chunk.read_start, chunk.read_end, " ".join(w.text for w in words), words)
            results.append((chunk, Transcript([seg], "en", "stub", "m", total)))
        return chunks, results

    def test_no_duplicate_words_at_seams(self):
        _, results = self._chunked()
        merged = merge_chunks(results, "en", "stub", "m", 30.0)
        starts = [w.start for w in merged.words]
        assert len(starts) == len(set(starts)), "a word was kept by two chunks"

    def test_no_words_lost_at_seams(self):
        _, results = self._chunked(total=30.0, chunk_seconds=10.0, step=0.5)
        merged = merge_chunks(results, "en", "stub", "m", 30.0)
        # 30s at one word per 0.5s.
        assert len(merged.words) == 60

    def test_merged_words_are_monotonic(self):
        _, results = self._chunked(total=45.0, chunk_seconds=7.0, overlap=1.5)
        merged = merge_chunks(results, "en", "stub", "m", 45.0)
        starts = [w.start for w in merged.words]
        assert starts == sorted(starts)

    def test_merged_transcript_validates_clean(self):
        _, results = self._chunked(total=60.0, chunk_seconds=13.0, overlap=2.0)
        merged = merge_chunks(results, "en", "stub", "m", 60.0)
        assert merged.validate() == []

    def test_segments_are_rebuilt_from_kept_words(self):
        _, results = self._chunked()
        merged = merge_chunks(results, "en", "stub", "m", 30.0)
        for seg in merged.segments:
            assert seg.words
            assert seg.start == seg.words[0].start
            assert seg.end == seg.words[-1].end

    def test_empty_input_returns_empty_transcript(self):
        merged = merge_chunks([], "en", "stub", "m", 0.0)
        assert merged.segments == []

    def test_single_chunk_passes_through_unchanged(self):
        chunks = plan_chunks(10, 300)
        t = make_transcript(4, 2)
        merged = merge_chunks([(chunks[0], t)], "en", "stub", "m", 10.0)
        assert len(merged.words) == len(t.words)

    def test_out_of_order_chunks_are_sorted(self):
        _, results = self._chunked()
        merged = merge_chunks(list(reversed(results)), "en", "stub", "m", 30.0)
        starts = [w.start for w in merged.words]
        assert starts == sorted(starts)

    def test_silent_chunk_contributes_nothing(self):
        chunks = plan_chunks(20, 10, 2)
        results = [
            (chunks[0], Transcript([], "en", "stub", "m", 20.0)),
            (chunks[1], make_transcript(3, 1, start=10.0)),
        ]
        merged = merge_chunks(results, "en", "stub", "m", 20.0)
        assert len(merged.words) == 3


class TestWhisperResultParsing:
    def test_parses_openai_shape_with_offset(self):
        payload = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0, "end": 2.0, "text": " Hello there",
                    "words": [
                        {"word": " Hello", "start": 0.0, "end": 0.6, "probability": 0.98},
                        {"word": " there", "start": 0.7, "end": 1.2, "probability": 0.95},
                    ],
                }
            ],
        }
        t = _from_whisper_dict(payload, offset=100.0, backend="mlx-whisper", model="m")
        assert t.words[0].start == 100.0
        assert t.words[1].end == 101.2
        assert t.segments[0].text == "Hello there"

    def test_skips_words_with_missing_timings(self):
        payload = {
            "language": "en",
            "segments": [{
                "start": 0, "end": 1, "text": "a b",
                "words": [
                    {"word": "a", "start": 0.0, "end": 0.4},
                    {"word": "b", "start": None, "end": None},
                ],
            }],
        }
        assert len(_from_whisper_dict(payload, 0.0, "b", "m").words) == 1


class TestBackendSelection:
    def test_stub_is_never_auto_selected(self):
        try:
            backend = get_backend("auto")
        except RuntimeError:
            return  # nothing available here, which is itself correct
        assert backend.name != "stub"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown transcription backend"):
            get_backend("whisper-9000")

    def test_stub_available_on_request(self):
        available, _ = get_backend("stub").is_available()
        assert available


class TestCache:
    def test_key_changes_with_model(self):
        assert cache_key("h", "mlx", "large-v3", None) != cache_key("h", "mlx", "turbo", None)

    def test_key_changes_with_source(self):
        assert cache_key("a", "mlx", "m", None) != cache_key("b", "mlx", "m", None)

    def test_key_changes_with_language_hint(self):
        assert cache_key("h", "mlx", "m", None) != cache_key("h", "mlx", "m", "ne")

    def test_key_is_stable(self):
        assert cache_key("h", "mlx", "m", "en") == cache_key("h", "mlx", "m", "en")

    def test_save_and_load_roundtrip(self):
        t = make_transcript(3, 2)
        t.cache_key = "abc123"
        save_transcript("proj1", t)
        loaded = load_cached("proj1", "abc123")
        assert loaded is not None
        assert loaded.word_count == t.word_count

    def test_load_returns_none_on_key_mismatch(self):
        t = make_transcript(2, 1)
        t.cache_key = "old"
        save_transcript("proj2", t)
        assert load_cached("proj2", "new") is None

    def test_load_returns_none_when_absent(self):
        assert load_cached("never-existed", "k") is None

    def test_corrupt_cache_returns_none_rather_than_raising(self):
        from app.services.transcription.service import transcript_path

        path = transcript_path("proj3")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json")
        assert load_cached("proj3", "k") is None


class TestStubBackend:
    def test_produces_word_timestamps(self, tmp_path):
        import wave

        wav = tmp_path / "a.wav"
        with wave.open(str(wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(b"\x00\x00" * 16000 * 4)  # 4 seconds

        t = StubBackend().transcribe(str(wav), "stub")
        assert t.has_word_timestamps
        assert t.validate() == []
        assert t.duration == pytest.approx(4.0)
