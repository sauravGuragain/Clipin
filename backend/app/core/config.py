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
