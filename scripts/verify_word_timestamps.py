#!/usr/bin/env python3
"""Verify word-level timestamp accuracy — the Phase 3 gate.

Timestamps that are subtly wrong will not crash anything. They will produce
captions that feel slightly off, clips that start a beat late, and there is no
way to detect that from a JSON file. So this transcribes a short excerpt and
burns the raw word timings onto the video. You watch it. If the highlight lands
on each word as it is spoken, the timings are good.

Do not move to Phase 4 until this looks right.

Usage:
    python scripts/verify_word_timestamps.py PODCAST.mp4
    python scripts/verify_word_timestamps.py PODCAST.mp4 --start 300 --duration 45
    python scripts/verify_word_timestamps.py PODCAST.mp4 --backend faster-whisper

Pick an excerpt with continuous speech. Music or silence proves nothing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.services.captions.ass import CaptionStyle, build_ass  # noqa: E402
from app.services.transcription.backends import get_backend  # noqa: E402
from app.services.transcription.service import transcribe_sync  # noqa: E402


def run(args: list[str], what: str) -> None:
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        print(f"\n{what} failed: {tail[-1] if tail else 'unknown error'}", file=sys.stderr)
        sys.exit(1)


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [settings.ffprobe_bin, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [settings.ffprobe_bin, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=False,
    ).stdout.split()
    try:
        return int(out[0]), int(out[1])
    except (IndexError, ValueError):
        return 1920, 1080


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--start", type=float, default=0.0, help="excerpt start (s)")
    parser.add_argument("--duration", type=float, default=40.0, help="excerpt length (s)")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--model", default=None)
    parser.add_argument("--language", default=None, help="e.g. en, ne — omit to auto-detect")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "verification.mp4")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Not found: {args.source}", file=sys.stderr)
        return 1

    work = ROOT / "data" / "verify"
    work.mkdir(parents=True, exist_ok=True)
    excerpt = work / "excerpt.mp4"
    audio = work / "excerpt.wav"

    total = probe_duration(args.source)
    if total and args.start >= total:
        print(f"--start {args.start}s is past the end of a {total:.0f}s file.", file=sys.stderr)
        return 1

    print(f"Source: {args.source.name}  ({total:.0f}s)")
    print(f"Excerpt: {args.start:.0f}s -> {args.start + args.duration:.0f}s\n")

    print("[1/5] Cutting excerpt")
    run([settings.ffmpeg_bin, "-y", "-v", "error",
         "-ss", str(args.start), "-i", str(args.source), "-t", str(args.duration),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-c:a", "aac", "-b:a", "160k", str(excerpt)], "Excerpt extraction")

    print("[2/5] Extracting audio (16 kHz mono)")
    run([settings.ffmpeg_bin, "-y", "-v", "error", "-i", str(excerpt),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio)],
        "Audio extraction")

    backend = get_backend(args.backend)
    available, reason = backend.is_available()
    if not available:
        print(f"\nBackend unavailable: {reason}", file=sys.stderr)
        return 1

    model = args.model or backend.default_model
    print(f"[3/5] Transcribing with {backend.name} / {model}")
    if backend.name != "stub":
        print("      First run downloads the model — expect a wait.")
    print()

    started = time.time()
    transcript = transcribe_sync(
        audio_path=audio,
        duration=probe_duration(audio),
        backend_name=args.backend,
        model=args.model,
        language=args.language,
        chunk_seconds=0,          # single pass: no seams in a short excerpt
        overlap=0,
    )
    elapsed = time.time() - started

    stats = transcript.stats()
    speed = args.duration / elapsed if elapsed else 0
    print(f"      Done in {elapsed:.1f}s  ({speed:.1f}x realtime)")
    print(f"      Language: {transcript.language}")
    print(f"      {stats['words']} words, {stats['segments']} segments, "
          f"{stats['words_per_minute']} wpm")
    if stats["mean_word_probability"] is not None:
        print(f"      Mean confidence: {stats['mean_word_probability']:.3f}")

    if not transcript.has_word_timestamps:
        print("\nFAIL: this backend returned no word timestamps. "
              "Animated captions are impossible without them.", file=sys.stderr)
        return 1

    issues = transcript.validate()
    if issues:
        print(f"\n      {len(issues)} timing issue(s):")
        for issue in issues[:8]:
            print(f"        [{issue.kind}] {issue.at:.2f}s — {issue.detail}")
        if len(issues) > 8:
            print(f"        ... and {len(issues) - 8} more")
    else:
        print("      No structural timing issues.")

    print("\n[4/5] Building captions")
    width, height = probe_size(excerpt)
    style = CaptionStyle(
        size=max(36, height // 22),
        margin_v=max(60, height // 12),
        max_chars_per_line=34,
    )
    ass_path = work / "verify.ass"
    ass_path.write_text(build_ass(transcript.words, style, width, height))

    print("[5/5] Rendering")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run([settings.ffmpeg_bin, "-y", "-v", "error", "-i", str(excerpt),
         "-vf", f"subtitles={ass_path}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-c:a", "copy", str(args.out)], "Caption render")

    print(f"\nWrote {args.out}\n")
    print("Watch it. The highlight should land on each word as it is spoken.")
    print("Consistently early or late by a fixed amount means an offset bug.")
    print("Drift that grows through the clip means a sample-rate mismatch.")
    print("Random misalignment means the backend's alignment is unreliable —")
    print("try --backend faster-whisper to compare before trusting either.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
