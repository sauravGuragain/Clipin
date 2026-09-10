"""Tests for media inspection.

The parser is tested against fixture JSON so these run without ffprobe and
without a media file.
"""

import pytest

from app.services.media import (
    MediaInspectionError,
    parse_fps,
    parse_probe_output,
)

LANDSCAPE_PODCAST = {
    "format": {
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "duration": "5400.023000",
        "size": "2147483648",
    },
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "30000/1001",
            "r_frame_rate": "30000/1001",
        },
        {
            "codec_type": "audio",
            "codec_name": "aac",
            "sample_rate": "48000",
            "channels": 2,
        },
    ],
}


class TestParseFps:
    def test_ntsc_fraction(self):
        assert parse_fps("30000/1001") == pytest.approx(29.97, abs=0.01)

    def test_integer_fraction(self):
        assert parse_fps("25/1") == 25.0

    def test_zero_denominator_is_none(self):
        assert parse_fps("0/0") is None

    def test_missing_is_none(self):
        assert parse_fps(None) is None
        assert parse_fps("N/A") is None

    def test_division_by_zero_is_none(self):
        assert parse_fps("30/0") is None


class TestParseProbeOutput:
    def test_landscape_podcast(self):
        info = parse_probe_output(LANDSCAPE_PODCAST)
        assert info.duration == pytest.approx(5400.023)
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == pytest.approx(29.97, abs=0.01)
        assert info.video_codec == "h264"
        assert info.audio_codec == "aac"
        assert info.channels == 2
        assert info.has_video and info.has_audio

    def test_landscape_detection(self):
        info = parse_probe_output(LANDSCAPE_PODCAST)
        assert info.is_landscape
        assert info.aspect_ratio == pytest.approx(16 / 9)

    def test_vertical_source_is_not_landscape(self):
        payload = {
            "format": {"format_name": "mp4", "duration": "60"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264",
                 "width": 1080, "height": 1920, "avg_frame_rate": "30/1"},
                {"codec_type": "audio", "codec_name": "aac",
                 "sample_rate": "48000", "channels": 2},
            ],
        }
        assert not parse_probe_output(payload).is_landscape

    def test_audio_only_file(self):
        payload = {
            "format": {"format_name": "mp3", "duration": "1800"},
            "streams": [{"codec_type": "audio", "codec_name": "mp3",
                         "sample_rate": "44100", "channels": 2}],
        }
        info = parse_probe_output(payload)
        assert info.has_audio and not info.has_video
        assert info.aspect_ratio is None
        assert not info.is_landscape

    def test_video_without_audio_is_flagged(self):
        payload = {
            "format": {"format_name": "mp4", "duration": "60"},
            "streams": [{"codec_type": "video", "codec_name": "h264",
                         "width": 1920, "height": 1080, "avg_frame_rate": "30/1"}],
        }
        assert parse_probe_output(payload).has_audio is False

    def test_no_streams_raises(self):
        with pytest.raises(MediaInspectionError):
            parse_probe_output({"format": {"format_name": "x"}, "streams": []})

    def test_duration_falls_back_to_stream(self):
        payload = {
            "format": {"format_name": "mkv"},
            "streams": [{"codec_type": "audio", "codec_name": "opus",
                         "duration": "123.5", "sample_rate": "48000", "channels": 2}],
        }
        assert parse_probe_output(payload).duration == pytest.approx(123.5)

    def test_unparseable_duration_becomes_zero(self):
        payload = {
            "format": {"format_name": "mkv", "duration": "N/A"},
            "streams": [{"codec_type": "audio", "codec_name": "opus",
                         "sample_rate": "48000", "channels": 2}],
        }
        assert parse_probe_output(payload).duration == 0.0

    def test_explicit_size_overrides_format_size(self):
        info = parse_probe_output(LANDSCAPE_PODCAST, size_bytes=999)
        assert info.size_bytes == 999
