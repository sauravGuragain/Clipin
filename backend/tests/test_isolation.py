"""Guard: the suite must never touch the real project database.

This exists because it already happened once. The Phase 2 tests wrote to
data/clipper.db, which collided with itself across runs and — the real damage —
would mark a genuinely running job FAILED through the crash-recovery sweep.

If someone later adds an `app` import above the environment setup in
conftest.py, isolation breaks silently and every future test run corrupts real
data. This makes it fail loudly instead.
"""

from app.core.config import settings
from app.models.db import engine


def test_database_lives_in_the_temp_directory(test_data_dir):
    resolved = str(settings.db_path.resolve())
    assert str(test_data_dir) in resolved, (
        f"Tests are pointed at {resolved}, outside the temporary test directory. "
        "DATA_DIR was set too late - check import order in conftest.py."
    )


def test_engine_is_bound_to_the_temp_database(test_data_dir):
    """settings could be right while the engine was built from a stale value,
    since the engine is constructed at import time."""
    assert str(test_data_dir) in str(engine.url)


def test_data_directories_are_under_the_temp_root(test_data_dir):
    for path in (settings.uploads_dir, settings.projects_dir):
        assert str(test_data_dir) in str(path.resolve())
