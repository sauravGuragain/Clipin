"""Recovering JSON from model output.

An 8B local model asked for JSON will comply most of the time and produce
something adjacent the rest of the time: a markdown fence, a sentence of
preamble, a trailing comma, a smart quote picked up from training data. Ollama's
`format: "json"` helps but does not eliminate this, and cloud models drift too.

Retrying costs a full generation — 20-30 seconds locally. Repairing costs
microseconds. So repair first, retry only when repair fails.

Every transformation here is conservative: it either produces valid JSON or
gives up. Nothing guesses at missing values.
"""

from __future__ import annotations

import json
import re

FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
TRAILING_COMMA = re.compile(r",(\s*[}\]])")
# "key": -> only unquoted keys that look like identifiers
UNQUOTED_KEY = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)")

SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2013": "-", "\u2014": "-", "\u00a0": " ",
}


class JSONRecoveryError(Exception):
    pass


def _balanced_span(text: str, opener: str, closer: str) -> str | None:
    """Find the first balanced {...} or [...], ignoring braces inside strings."""
    start = text.find(opener)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        char = text[i]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _candidates(raw: str) -> list[str]:
    """Progressively more aggressive attempts, cheapest first."""
    attempts: list[str] = [raw.strip()]

    fenced = FENCE.search(raw)
    if fenced:
        attempts.append(fenced.group(1).strip())

    # Models often narrate before the JSON: "Sure! Here are the clips: [...]"
    for opener, closer in (("[", "]"), ("{", "}")):
        span = _balanced_span(raw, opener, closer)
        if span:
            attempts.append(span)
            if fenced:
                inner = _balanced_span(fenced.group(1), opener, closer)
                if inner:
                    attempts.append(inner)

    return attempts


def _repairs(text: str) -> list[str]:
    variants = [text]

    normalised = text
    for bad, good in SMART_QUOTES.items():
        normalised = normalised.replace(bad, good)
    if normalised != text:
        variants.append(normalised)

    for base in list(variants):
        stripped = TRAILING_COMMA.sub(r"\1", base)
        if stripped != base:
            variants.append(stripped)

    for base in list(variants):
        quoted = UNQUOTED_KEY.sub(r'\1"\2"\3', base)
        if quoted != base:
            variants.append(quoted)

    return variants


def extract_json(raw: str) -> object:
    """Parse JSON out of model output, repairing common damage.

    Raises JSONRecoveryError if nothing parses — the caller then retries the
    generation, which is the expensive path.
    """
    if not raw or not raw.strip():
        raise JSONRecoveryError("Model returned an empty response.")

    for candidate in _candidates(raw):
        for variant in _repairs(candidate):
            try:
                return json.loads(variant)
            except json.JSONDecodeError:
                continue

    preview = raw.strip().replace("\n", " ")[:160]
    raise JSONRecoveryError(f"No valid JSON found in model output: {preview}")


def extract_json_list(raw: str, list_keys: tuple[str, ...] = ("clips", "candidates", "moments", "results", "items")) -> list:
    """Extract a list, tolerating a model that wraps it in an object.

    Asked for a JSON array, models frequently return {"clips": [...]} instead.
    That is a formatting preference, not an error, so it is accepted.
    """
    parsed = extract_json(raw)

    if isinstance(parsed, list):
        return parsed

    if isinstance(parsed, dict):
        for key in list_keys:
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        # A single object where a list was asked for: treat it as one item
        # rather than discarding a valid candidate.
        if any(k in parsed for k in ("start", "start_time", "hook")):
            return [parsed]
        # Exactly one list-valued key, whatever it is called.
        lists = [v for v in parsed.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]

    raise JSONRecoveryError(
        f"Expected a JSON array of candidates, got {type(parsed).__name__}."
    )
