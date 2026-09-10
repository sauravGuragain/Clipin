"""Test configuration.

This file must set DATA_DIR **before** anything under `app` is imported.
`app.models.db` builds its SQLAlchemy engine at module import time from
`settings.db_path`, so by the time a test module imports it the destination is
already fixed. Redirecting later would be too late.

Why this exists: the Phase 2 suite originally wrote to the real project
database. That collided with itself on a second run, and — far worse — a test
run would mark a genuinely in-flight job FAILED via the crash-recovery sweep.
Tests get their own throwaway database, always.
"""

import os
import shutil
import tempfile
from pathlib import Path

# --- must happen before any `app` import ------------------------------------
_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="clipper-tests-"))
os.environ["DATA_DIR"] = str(_TEST_DATA_DIR)
# ----------------------------------------------------------------------------

import pytest  # noqa: E402

from app.models.db import Base, engine, init_db  # noqa: E402


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEST_DATA_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolated_db():
    """Fresh schema for every test.

    Dropping and recreating rather than deleting rows keeps tests independent
    of execution order, which is what let the original bug hide: the suite
    passed on a clean checkout and failed on every run after.
    """
    init_db()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Exposed so the isolation guard can assert against it."""
    return _TEST_DATA_DIR
