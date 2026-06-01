"""
Database connection management.

Provides SQLAlchemy engine, session factory, and initialization helpers.
Designed so that swapping SQLite → PostgreSQL requires only changing the URL
in config.py (or the DB_URL environment variable).
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from config import settings

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
_connect_args = {}
if settings.database.url.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

engine = create_engine(
    settings.database.url,
    echo=settings.database.echo,
    connect_args=_connect_args,
    pool_pre_ping=True,
)

# Enable WAL mode and foreign keys for SQLite
if settings.database.url.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
SessionFactory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional session scope.

    Usage::

        with get_session() as session:
            repo = CameraRepository(session)
            cameras = repo.get_all()
    """
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session_dependency() -> Generator[Session, None, None]:
    """FastAPI dependency for injecting a session."""
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Create all tables. Safe to call multiple times."""
    from database.models import Base  # noqa: F811

    Base.metadata.create_all(bind=engine)
