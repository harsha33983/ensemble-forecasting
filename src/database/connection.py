"""
Database Connection and Session Management Module.

Configures SQLite connection via SQLAlchemy ORM with foreign key enforcement
and thread-safe session management.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from src.config import DATABASE_FILE, DATABASE_URL
from src.utils import setup_logger

logger = setup_logger("database_connection")

# Global engine and session factory caches
_ENGINE: Optional[Engine] = None
_SESSION_FACTORY: Optional[sessionmaker] = None


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Ensure SQLite enforces foreign key constraints and runs with optimal concurrency.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.close()


def get_engine(db_url: Optional[str] = None, echo: bool = False) -> Engine:
    """
    Create or retrieve cached SQLAlchemy SQLite engine.
    Ensures parent directories exist automatically.
    """
    global _ENGINE
    target_url = db_url or DATABASE_URL

    if _ENGINE is None or db_url is not None:
        if target_url.startswith("sqlite:///"):
            raw_path = target_url.replace("sqlite:///", "")
            if raw_path and raw_path != ":memory:":
                Path(raw_path).parent.mkdir(parents=True, exist_ok=True)

        engine = create_engine(
            target_url,
            echo=echo,
            connect_args={"check_same_thread": False},
        )
        if db_url is None:
            _ENGINE = engine
            return _ENGINE
        return engine

    return _ENGINE


def get_session_factory(engine: Union[Engine, str, None] = None) -> sessionmaker:
    """
    Get or create configured sessionmaker factory.
    Accepts an Engine instance, a database URL string, or None.
    """
    global _SESSION_FACTORY
    if isinstance(engine, str):
        eng = get_engine(engine)
    else:
        eng = engine or get_engine()

    if _SESSION_FACTORY is None or engine is not None:
        factory = sessionmaker(bind=eng, autoflush=False, autocommit=False, expire_on_commit=False)
        if engine is None:
            _SESSION_FACTORY = factory
        return factory
    return _SESSION_FACTORY


@contextmanager
def get_db_session(engine: Union[Engine, str, None] = None) -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Automatically commits on success or rolls back on exception.
    """
    factory = get_session_factory(engine)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("Database session transaction rolled back due to error: %s", e)
        raise
    finally:
        session.close()

