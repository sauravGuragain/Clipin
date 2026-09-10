"""ASS subtitle generation with word-level highlighting.

Written now rather than in Phase 9 because verifying word timestamps requires
seeing them burned onto video. It is deliberately minimal — one style, one
animation — but it is the real mechanism Phase 9 will build on, so proving it
works here removes the largest unknown from that phase.

Why ASS and libass rather than compositing frames: libass supports per-word
colour overrides and `\\t` transforms, so highlighting and scale animation
render inside FFmpeg at near-copy speed. Compositing PNG frames per word would
be 10-50x slower for the same result.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.transcription.base import Word

# ASS colours are &HAABBGGRR — alpha first, then BGR, not RGB.
WHITE = "&H00FFFFFF"
YELLOW = "&H0000D7FF"
BLACK = "&H00000000"


@dataclass
class CaptionStyle:
    font: str = "Helvetica"
    size: int = 64
    primary: str = WHITE
    highlight: str = YELLOW
    outline: str = BLACK
    outline_width: float = 3.0
    shadow: float = 0.0
    bold: int = 1
    alignment: int = 2          # 2 = bottom centre
    margin_v: int = 180
    margin_h: int = 80
    max_chars_per_line: int = 28
    max_words_per_line: int = 6
    max_line_seconds: float = 3.0
    emphasise_active: bool = True
    # How long the finished line lingers after its last word. Without this the
    # caption vanishes the instant someone stops speaking.
    line_hold: float = 0.35
    scale_percent: int = 112    # active word grows to this


def escape(text: str) -> str:
    """ASS treats braces as override blocks and backslashes as escapes."""
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", " ")
    )


def timestamp(seconds: float) -> str:
    """ASS wants H:MM:SS.cc — centisecond precision, single-digit hour."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:d}:{minutes:02d}:{secs:05.2f}"


def group_words_into_lines(words: list[Word], style: CaptionStyle) -> list[list[Word]]:
    """Break the word stream into caption lines.

    Three limits, whichever hits first: character count (readability), word
    count, and elapsed time. The time limit matters because a long pause
    otherwise leaves one line hanging on screen for many seconds.
    """
    lines: list[list[Word]] = []
    current: list[Word] = []
    chars = 0

    for word in words:
        text = word.text.strip()
        if not text:
            continue

        would_be = chars + len(text) + (1 if current else 0)
        too_long = would_be > style.max_chars_per_line
        too_many = len(current) >= style.max_words_per_line
        too_slow = bool(current) and (word.end - current[0].start) > style.max_line_seconds

        if current and (too_long or too_many or too_slow):
            lines.append(current)
            current = []
            chars = 0
            would_be = len(text)

        current.append(word)
        chars = would_be

    if current:
        lines.append(current)
    return lines


def header(style: CaptionStyle, width: int, height: int) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,{style.font},{style.size},{style.primary},{style.primary},{style.outline},{BLACK},{style.bold},0,0,0,100,100,0,0,1,{style.outline_width},{style.shadow},{style.alignment},{style.margin_h},{style.margin_h},{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass(
    words: list[Word],
    style: CaptionStyle | None = None,
    width: int = 1080,
    height: int = 1920,
    time_offset: float = 0.0,
) -> str:
    """Render words to a complete ASS document.

    One Dialogue event per word, each spanning that word's own time range and
    showing the whole line with the active word recoloured. This is the
    standard karaoke-by-duplication technique; it costs more events than `\\k`
    tags but gives full control over per-word appearance.
    """
    style = style or CaptionStyle()
    out = [header(style, width, height)]

    for line in group_words_into_lines(words, style):
        # The line stays on screen continuously; only the highlight moves.
        # Timing each event to its own word's duration leaves the screen blank
        # in the gaps between words, which reads as flicker.
        line_end = line[-1].end + style.line_hold
        for index, active in enumerate(line):
            start = active.start - time_offset
            # Hold until the next word begins, or until the line ends.
            following = line[index + 1].start if index + 1 < len(line) else line_end
            end = following - time_offset
            if end <= 0:
                continue
            start = max(0.0, start)
            if end - start < 0.02:      # libass drops zero-length events
                end = start + 0.02

            parts = []
            for i, word in enumerate(line):
                text = escape(word.text.strip())
                if i == index:
                    if style.emphasise_active:
                        parts.append(
                            f"{{\\c{style.highlight}"
                            f"\\t(0,120,\\fscx{style.scale_percent}\\fscy{style.scale_percent})}}"
                            f"{text}"
                            f"{{\\c{style.primary}\\fscx100\\fscy100}}"
                        )
                    else:
                        parts.append(f"{{\\c{style.highlight}}}{text}{{\\c{style.primary}}}")
                else:
                    parts.append(text)

            out.append(
                f"Dialogue: 0,{timestamp(start)},{timestamp(end)},Main,,0,0,0,,"
                + " ".join(parts)
            )

    return "\n".join(out) + "\n"
