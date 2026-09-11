"""Text similarity, for catching candidates about the same thing.

Time-overlap dedup (Phase 6) only catches the same *moment*. It cannot catch a
host making the same point twice, four minutes apart — two candidates that are
temporally disjoint but would produce near-identical clips.

Two backends behind one interface:

- **lexical** (default) — token overlap plus character trigrams. No model, no
  download, no dependency, and it runs in microseconds. Good at catching
  restatements that reuse vocabulary, which is most of them in practice.
- **embedding** (optional) — `fastembed` with a small ONNX model. Catches
  paraphrase that shares no words. Requires a ~130 MB download on first use,
  so it is opt-in rather than assumed.

The lexical default is a deliberate choice, not a stopgap: it makes the feature
work offline with no setup, and the embedding backend slots in when paraphrase
detection turns out to matter.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

WORD = re.compile(r"[a-z0-9']+")

# Words too common to signal shared meaning. A shared "the" says nothing.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "was", "are", "were", "be",
    "been", "to", "of", "in", "on", "at", "for", "with", "as", "by", "that",
    "this", "it", "its", "i", "you", "he", "she", "we", "they", "them", "his",
    "her", "their", "our", "my", "your", "so", "if", "then", "than", "there",
    "here", "what", "when", "how", "why", "do", "does", "did", "not", "no",
    "just", "like", "about", "from", "into", "out", "up", "down", "over",
}


@runtime_checkable
class SimilarityBackend(Protocol):
    name: str

    def is_available(self) -> tuple[bool, str]: ...

    def similarity(self, a: str, b: str) -> float: ...


SUFFIXES = ("ing", "ed", "es", "s", "ly", "er")


def stem(word: str) -> str:
    """Crude suffix stripping.

    Not linguistics — just enough that "folds", "folding" and "fold" collide.
    Without it, hooks describing the same thing in different tenses share no
    tokens at all, which is the common case for restatements.
    """
    for suffix in SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> set[str]:
    return {
        stem(w) for w in WORD.findall(text.lower())
        if w not in STOPWORDS and len(w) > 2
    }


def trigrams(text: str) -> set[str]:
    cleaned = re.sub(r"\s+", " ", text.lower().strip())
    return {cleaned[i : i + 3] for i in range(max(0, len(cleaned) - 2))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def overlap(a: set, b: set) -> float:
    """Intersection over the smaller set.

    Jaccard punishes length differences hard, which is wrong here: a short hook
    fully contained in a longer one is a restatement, not a weak match. Hooks
    are under a dozen words, so this matters constantly.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


class LexicalSimilarity:
    """Token Jaccard blended with character trigrams.

    Tokens catch shared vocabulary; trigrams catch morphological variation
    ("fold" / "folding" / "foldable") that token matching misses entirely.
    """

    name = "lexical"

    def __init__(self, token_weight: float = 0.7) -> None:
        self.token_weight = token_weight

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def similarity(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        ta, tb = tokenize(a), tokenize(b)
        # Blend of both set measures: Jaccard keeps unrelated pairs near zero,
        # overlap rescues restatements of differing length.
        token_score = 0.45 * jaccard(ta, tb) + 0.55 * overlap(ta, tb)
        trigram_score = jaccard(trigrams(a), trigrams(b))
        return self.token_weight * token_score + (1 - self.token_weight) * trigram_score


class EmbeddingSimilarity:
    """fastembed with a small ONNX model.

    Catches paraphrase with no shared vocabulary, which the lexical backend
    cannot. Downloads ~130 MB on first use. Chosen over sentence-transformers
    because it needs no PyTorch of its own.
    """

    name = "embedding"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model
        self._model = None
        self._cache: dict[str, list[float]] = {}

    def is_available(self) -> tuple[bool, str]:
        try:
            import fastembed  # noqa: F401
        except ImportError:
            return False, (
                "fastembed is not installed. Run: pip install fastembed "
                "(downloads a ~130 MB model on first use)."
            )
        return True, ""

    def _embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        vector = list(next(iter(self._model.embed([text]))))
        self._cache[text] = vector
        return vector

    def similarity(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        va, vb = self._embed(a), self._embed(b)
        dot = sum(x * y for x, y in zip(va, vb))
        na = sum(x * x for x in va) ** 0.5
        nb = sum(x * x for x in vb) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        # Cosine runs -1..1; clamp to 0..1 since negative similarity is not
        # meaningful for deduplication.
        return max(0.0, dot / (na * nb))


BACKENDS: dict[str, type] = {
    LexicalSimilarity.name: LexicalSimilarity,
    EmbeddingSimilarity.name: EmbeddingSimilarity,
}


def get_similarity(name: str = "lexical"):
    cls = BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown similarity backend '{name}'. Available: {', '.join(BACKENDS)}"
        )
    backend = cls()
    available, reason = backend.is_available()
    if not available:
        # Falling back rather than failing: dedup quality degrades, the feature
        # still works, and the reason is worth surfacing but not fatal.
        return LexicalSimilarity()
    return backend
