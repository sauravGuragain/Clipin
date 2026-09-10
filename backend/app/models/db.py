"""Database models and session management.

Only metadata lives in SQLite. Media files stay on disk and are referenced by
path, per PLAN.md section 20.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Base(DeclarativeBase):
    pass


class JobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES = {
    JobStatus.COMPLETED.value,
    JobStatus.FAILED.value,
    JobStatus.CANCELLED.value,
}


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(255))

    source_path: Mapped[str] = mapped_column(Text)
    source_filename: Mapped[str] = mapped_column(String(255))
    source_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Denormalised media facts. Kept as columns rather than a JSON blob so they
    # can be sorted and filtered in the projects list without deserialising.
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_codec: Mapped[str | None] = mapped_column(String(32), nullable=True)
    audio_codec: Mapped[str | None] = mapped_column(String(32), nullable=True)
    has_audio: Mapped[bool] = mapped_column(default=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)

    # Phase 2: path to the extracted 16 kHz mono WAV, once produced.
    audio_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    project_id: Mapped[str | None] = mapped_column(String(12), nullable=True, index=True)
    type: Mapped[str] = mapped_column(String(48))

    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED.value, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    params: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(12), index=True)
    index: Mapped[int] = mapped_column(Integer, default=0)

    start: Mapped[float] = mapped_column(Float, default=0.0)
    end: Mapped[float] = mapped_column(Float, default=0.0)
    duration: Mapped[float] = mapped_column(Float, default=0.0)

    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    hook: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Phase 4 has no scoring; Phase 5 fills these in.
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    strategy: Mapped[str] = mapped_column(String(32), default="even")
    boundary_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    boundary_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    crop_strategy: Mapped[str] = mapped_column(String(32), default="center")

    render_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    qc_ok: Mapped[bool] = mapped_column(default=True)
    qc_issues: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


engine = create_engine(
    f"sqlite:///{settings.db_path}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    settings.ensure_dirs()
    Base.metadata.create_all(engine)
    _migrate()


def _migrate() -> None:
    """Minimal additive migration.

    Phase 1 shipped a projects table without audio_path. Rather than pull in
    Alembic for a single-user local tool, add missing columns directly. Only
    additive changes are supported - anything more will need a real migration
    tool, and that is the point at which to add one.
    """
    from sqlalchemy import inspect as sa_inspect, text

    inspector = sa_inspect(engine)
    if "projects" not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns("projects")}
    if "audio_path" not in existing:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE projects ADD COLUMN audio_path TEXT"))

    if "clips" in inspector.get_table_names():
        clip_cols = {c["name"] for c in inspector.get_columns("clips")}
        for column, ddl in (
            ("boundary_score", "ALTER TABLE clips ADD COLUMN boundary_score FLOAT"),
            ("boundary_notes", "ALTER TABLE clips ADD COLUMN boundary_notes TEXT"),
        ):
            if column not in clip_cols:
                with engine.begin() as conn:
                    conn.execute(text(ddl))


def get_session() -> Session:
    return SessionLocal()
