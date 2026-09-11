"""Application configuration.

Every tunable lives here or in .env. Nothing in the pipeline hard-codes a path,
a binary name, or a threshold.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: .../clipper-ai
ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- binaries -------------------------------------------------------
    # Deliberately configurable. This machine has two ffmpeg builds installed
    # (Homebrew `ffmpeg` is lean and lacks libass; `ffmpeg-full` has it).
    # A future `brew upgrade` can silently relink the wrong one, so the path
    # is a setting and startup verifies capabilities rather than existence.
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"

    # --- storage --------------------------------------------------------
    data_dir: Path = ROOT_DIR / "data"

    # --- uploads --------------------------------------------------------
    max_upload_bytes: int = 8 * 1024 * 1024 * 1024  # 8 GB
    allowed_extensions: tuple[str, ...] = (
        ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mp3", ".wav", ".m4a",
    )

    # --- transcription --------------------------------------------------
    # "auto" picks MLX on Apple Silicon, faster-whisper elsewhere.
    transcribe_backend: str = "auto"
    transcribe_model: str = ""          # empty = the backend's default
    transcribe_language: str = ""       # empty = auto-detect
    # Chunking exists to give real progress on long files. 0 disables it:
    # one pass, best context, no progress reporting.
    transcribe_chunk_seconds: float = 300.0
    transcribe_overlap: float = 2.0

    # --- clips ----------------------------------------------------------
    default_clip_count: int = 5
    clip_min_duration: float = 25.0
    clip_max_duration: float = 60.0
    crop_strategy: str = "center"       # center | fit
    clip_padding: float = 0.35          # max seconds taken from surrounding silence
    # Reject clips scoring below this. 0 accepts anything the solver returns;
    # raise it once you have seen real output and know what a bad clip scores.
    min_boundary_score: float = 0.0

    # --- boundary weights (spec 28) --------------------------------------
    weight_opener: float = 0.28
    weight_starts_sentence: float = 0.18
    weight_ends_sentence: float = 0.24
    weight_not_dangling: float = 0.12
    weight_low_filler: float = 0.10
    weight_duration_fit: float = 0.08

    # --- LLM / discovery -------------------------------------------------
    # "auto" prefers Ollama, matching the project's local-first stance.
    llm_provider: str = "auto"
    llm_model: str = ""                 # empty = the provider's default
    llm_window_tokens: int = 1800       # transcript tokens per model call
    llm_retries: int = 1                # a retry costs a full generation
    # Context window requested from Ollama. Must comfortably exceed
    # llm_window_tokens + prompt overhead + llm_max_output.
    llm_num_ctx: int = 8192
    # Three candidates of structured JSON is roughly 300 tokens. 2048 was
    # reserving context for output that never arrives.
    llm_max_output: int = 768
    # Reasoning costs seconds and output budget for little gain on structured
    # extraction. Set LLM_THINK=1 to compare quality with it enabled.
    llm_think: bool = False
    candidates_per_window: int = 3
    candidate_iou_threshold: float = 0.5

    # --- Phase 7: ranking -------------------------------------------------
    rerank_enabled: bool = True
    rerank_shortlist: int = 20          # one extra call over the whole podcast
    similarity_backend: str = "lexical"  # lexical | embedding
    # Calibrated by measurement, not guessed. On real hooks from one podcast:
    # genuinely distinct pairs topped out at 0.17; restatements floored at 0.29.
    # Set toward the restatement end because dropping a good clip costs more
    # than keeping a near-duplicate the user can reject in review.
    semantic_dedup_threshold: float = 0.27

    # Final score weights (spec 9 and 28).
    weight_llm_score: float = 0.30
    weight_rerank: float = 0.35
    weight_boundary: float = 0.20
    weight_duration_fit_final: float = 0.08
    weight_diversity: float = 0.07

    # --- output ---------------------------------------------------------
    output_width: int = 1080
    output_height: int = 1920
    output_crf: int = 20
    # libx264 for final quality; h264_videotoolbox uses the M-series media
    # engine and barely heats a fanless chassis - better for previews.
    output_encoder: str = "libx264"

    # --- server ---------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "clipper.db"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.uploads_dir, self.projects_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
