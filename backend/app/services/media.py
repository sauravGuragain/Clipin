"""Media inspection via ffprobe.

Pure parsing is separated from subprocess execution so the parser can be tested
against fixture JSON without needing a real file or a real ffprobe.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from app.core.config import settings


class MediaInspectionError(Exception):
    """Raised when a file cannot be probed or contains no usable streams."""


@dataclass
class MediaInfo:
    duration: float
    container: str
    size_bytes: int

    has_video: bool
    width: int | None
    height: int | None
    fps: float | None
    video_codec: str | None

    has_audio: bool
    audio_codec: str | None
    sample_rate: int | None
    channels: int | None

    @property
    def aspect_ratio(self) -> float | None:
        if self.width and self.height:
            return self.width / self.height
        return None

    @property
    def is_landscape(self) -> bool:
        ar = self.aspect_ratio
        return ar is not None and ar > 1.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["aspect_ratio"] = self.aspect_ratio
        d["is_landscape"] = self.is_landscape
        return d


def parse_fps(rate: str | None) -> float | None:
    """ffprobe reports frame rates as fractions like '30000/1001'."""
    if not rate or rate in ("0/0", "N/A"):
        return None
    try:
        value = float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def parse_probe_output(payload: dict, size_bytes: int = 0) -> MediaInfo:
    """Turn raw ffprobe JSON into MediaInfo. No I/O — unit testable."""
    fmt = payload.get("format", {}) or {}
    streams = payload.get("streams", []) or []

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video is None and audio is None:
        raise MediaInspectionError(
            "No video or audio streams found. The file may be corrupt or an "
            "unsupported format."
        )

    # Duration can live on the format or on a stream; prefer format.
    duration_raw = fmt.get("duration")
    if duration_raw in (None, "N/A"):
        duration_raw = (video or audio or {}).get("duration")
    try:
        duration = float(duration_raw) if duration_raw not in (None, "N/A") else 0.0
    except (TypeError, ValueError):
        duration = 0.0

    try:
        reported_size = int(fmt.get("size", 0) or 0)
    except (TypeError, ValueError):
        reported_size = 0

    return MediaInfo(
        duration=duration,
        container=fmt.get("format_name", "unknown"),
        size_bytes=size_bytes or reported_size,
        has_video=video is not None,
        width=int(video["width"]) if video and video.get("width") else None,
        height=int(video["height"]) if video and video.get("height") else None,
        # avg_frame_rate is more reliable than r_frame_rate for VFR sources.
        fps=parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")) if video else None,
        video_codec=video.get("codec_name") if video else None,
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
        sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        channels=int(audio["channels"]) if audio and audio.get("channels") else None,
    )


def inspect(path: Path) -> MediaInfo:
    """Probe a real file on disk."""
    if not path.exists():
        raise MediaInspectionError(f"File not found: {path}")

    args = [
        settings.ffprobe_bin,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=120, check=False)
    except FileNotFoundError as exc:
        raise MediaInspectionError(
            f"ffprobe not found (configured as '{settings.ffprobe_bin}'). "
            "Install FFmpeg or set FFPROBE_BIN in .env."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaInspectionError("ffprobe timed out after 120s.") from exc

    if proc.returncode != 0:
        lines = (proc.stderr or "").strip().splitlines()
        reason = lines[-1] if lines else "Unsupported or corrupt media."
        # ffprobe prefixes its message with the full path. That path is an
        # internal storage detail the user never chose and cannot act on.
        reason = reason.replace(str(path), "").lstrip(": ").strip()
        raise MediaInspectionError(
            f"This file could not be read. {reason or 'It may be corrupt or an unsupported format.'}"
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MediaInspectionError("ffprobe returned output that could not be parsed.") from exc

    return parse_probe_output(payload, size_bytes=path.stat().st_size)
